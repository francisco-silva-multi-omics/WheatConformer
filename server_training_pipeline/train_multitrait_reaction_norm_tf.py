from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .final_evaluation_contract import load_protocol, require_non_discovery_seed
from .kernel_factorization import (
    effective_factorization_mode,
    factorization_training_support,
    kernel_factors,
)
from .kernel_registry_contract import training_input_identities
from .nested_evaluation import (
    SCENARIO_MODES,
    assign_nested_split,
    manifest_identity,
    verify_manifest_contract,
)
from .observation_weights import (
    apply_precision_weight_transform,
    fit_precision_weight_transform,
)
from .split_utils import canonical_split_mode, make_split, split_group_column, split_leakage_record
from .train_multitrait_multikernel_tf import (
    add_expert_indices,
    assign_active_marker_coverage,
    file_identity,
    file_sha256,
    index_digest,
    metric_rows,
    parse_bool,
    read_table,
    require_certified_file,
    safe_name,
    trait_balanced_loss_weights,
    trait_set,
    weighted_mean_std,
    write_table,
)


def estimate_trait_covariance(
    train: pd.DataFrame,
    trait_names: list[str],
    shrinkage: float,
    minimum_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("Trait covariance shrinkage must be between zero and one")
    required = {"genotype_id", "environment_id", "trait_name_canonical", "y_scaled"}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"Trait covariance input is missing columns: {missing}")
    pivot = train.pivot_table(
        index=["genotype_id", "environment_id"],
        columns="trait_name_canonical",
        values="y_scaled",
        aggfunc="mean",
    ).reindex(columns=trait_names)
    count = np.zeros((len(trait_names), len(trait_names)), dtype=np.int64)
    correlation = np.eye(len(trait_names), dtype=np.float64)
    values = pivot.to_numpy(dtype=np.float64)
    for left in range(len(trait_names)):
        for right in range(left + 1, len(trait_names)):
            available = np.isfinite(values[:, left]) & np.isfinite(values[:, right])
            pairs = int(available.sum())
            count[left, right] = count[right, left] = pairs
            if pairs < minimum_pairs:
                value = 0.0
            else:
                x = values[available, left]
                y = values[available, right]
                value = (
                    float(np.corrcoef(x, y)[0, 1])
                    if np.std(x) > 0 and np.std(y) > 0
                    else 0.0
                )
                if not np.isfinite(value):
                    value = 0.0
            correlation[left, right] = correlation[right, left] = value
    count[np.diag_indices_from(count)] = np.isfinite(values).sum(axis=0)
    covariance = (1.0 - shrinkage) * correlation + shrinkage * np.eye(len(trait_names))
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-4)
    covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(covariance))
    covariance = covariance / np.outer(scale, scale)
    covariance = (covariance + covariance.T) * 0.5
    square_root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    square_root = np.diag(1.0 / scale) @ square_root
    if not np.isfinite(covariance).all() or not np.isfinite(square_root).all():
        raise ValueError("Trait covariance estimation produced non-finite values")
    return covariance, square_root, count


def deterministic_sign_projection(
    input_rank: int, output_rank: int, seed: int, label: str
) -> np.ndarray:
    label_seed = int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:4], "little"
    )
    rng = np.random.default_rng((seed + label_seed) % (2**32))
    return rng.choice([-1.0, 1.0], size=(input_rank, output_rank)).astype(np.float32)


class MultiTraitReactionNorm(tf.keras.Model):
    def __init__(
        self,
        expert_specs: list[dict[str, object]],
        factors: list[np.ndarray],
        trait_names: list[str],
        genotype_kernel: str,
        trait_covariance_sqrt: np.ndarray,
        reaction_rank: int,
        ridge_penalty: float,
        residual_scale_floor: float,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        self.expert_specs = expert_specs
        self.trait_names = trait_names
        self.ridge_penalty = float(ridge_penalty)
        self.residual_scale_floor = float(residual_scale_floor)
        self.initialization_seed = int(initialization_seed)
        self._initializer_index = 0
        self.trait_covariance_sqrt = tf.constant(
            trait_covariance_sqrt, dtype=tf.float32
        )
        self.factors = [tf.constant(value, dtype=tf.float32) for value in factors]
        kernel_names = [str(spec["kernel"]) for spec in expert_specs]
        if genotype_kernel not in kernel_names:
            raise ValueError(f"Genotype kernel is not active: {genotype_kernel}")
        self.genotype_index = kernel_names.index(genotype_kernel)
        if str(expert_specs[self.genotype_index]["axis"]) != "genotype":
            raise ValueError("Configured reaction-norm genotype kernel is not a genotype expert")
        other_genotype = [
            str(spec["kernel"])
            for spec in expert_specs
            if str(spec["axis"]) == "genotype"
            and str(spec["kernel"]) != genotype_kernel
        ]
        if other_genotype:
            raise ValueError(f"Reaction-norm baseline contains extra genotype kernels: {other_genotype}")

        self.intercept = self.add_weight(
            name="trait_intercept", shape=(len(trait_names),), initializer="zeros"
        )
        initial_residual = math.log(math.expm1(max(1.0 - residual_scale_floor, 1e-3)))
        self.raw_residual_scale = self.add_weight(
            name="trait_raw_residual_scale",
            shape=(len(trait_names),),
            initializer=tf.keras.initializers.Constant(initial_residual),
        )
        self.main_coefficients: list[tf.Variable] = []
        self.eligibility: list[tf.Tensor] = []
        for spec, factor in zip(expert_specs, factors):
            name = safe_name(str(spec["kernel"]))
            self.main_coefficients.append(
                self.add_weight(
                    name=f"{name}_main_random_coefficients",
                    shape=(factor.shape[1], len(trait_names)),
                    initializer=self._initializer(),
                )
            )
            self.eligibility.append(tf.constant(self._eligibility(spec)))

        genotype_factor = factors[self.genotype_index]
        self.reaction_environment_indices: list[int] = []
        self.reaction_genotype_features: list[tf.Tensor] = []
        self.reaction_environment_features: list[tf.Tensor] = []
        self.reaction_coefficients: list[tf.Variable] = []
        self.reaction_projection_digests: dict[str, dict[str, str]] = {}
        for expert_index, (spec, environment_factor) in enumerate(
            zip(expert_specs, factors)
        ):
            if str(spec["axis"]) != "environment" or not parse_bool(
                spec["interaction_enabled"]
            ):
                continue
            environment_name = str(spec["kernel"])
            g_projection = deterministic_sign_projection(
                genotype_factor.shape[1],
                reaction_rank,
                initialization_seed,
                f"{genotype_kernel}x{environment_name}:genotype",
            )
            e_projection = deterministic_sign_projection(
                environment_factor.shape[1],
                reaction_rank,
                initialization_seed,
                f"{genotype_kernel}x{environment_name}:environment",
            )
            g_features = (genotype_factor @ g_projection).astype(np.float32)
            e_features = (environment_factor @ e_projection).astype(np.float32)
            self.reaction_environment_indices.append(expert_index)
            self.reaction_genotype_features.append(tf.constant(g_features))
            self.reaction_environment_features.append(tf.constant(e_features))
            self.reaction_coefficients.append(
                self.add_weight(
                    name=f"{safe_name(environment_name)}_reaction_coefficients",
                    shape=(reaction_rank, len(trait_names)),
                    initializer=self._initializer(),
                )
            )
            self.reaction_projection_digests[environment_name] = {
                "genotype": hashlib.sha256(g_projection.tobytes()).hexdigest(),
                "environment": hashlib.sha256(e_projection.tobytes()).hexdigest(),
            }
        if not self.reaction_environment_indices:
            raise ValueError("Reaction-norm model has no environment interaction axes")
        self.reaction_scale = tf.constant(1.0 / math.sqrt(reaction_rank), tf.float32)

    def _initializer(self) -> tf.keras.initializers.RandomNormal:
        self._initializer_index += 1
        return tf.keras.initializers.RandomNormal(
            stddev=0.01, seed=self.initialization_seed + self._initializer_index
        )

    def _eligibility(self, spec: dict[str, object]) -> np.ndarray:
        eligible = trait_set(spec["eligible_traits"])
        if eligible is None:
            return np.ones(len(self.trait_names), dtype=bool)
        return np.asarray(
            [trait.upper() in eligible for trait in self.trait_names], dtype=bool
        )

    def correlated_coefficients(self, raw: tf.Tensor) -> tf.Tensor:
        return tf.matmul(raw, self.trait_covariance_sqrt, transpose_b=True)

    def call(self, inputs, training: bool = False):
        expert_indices, trait_index = inputs
        prediction = tf.gather(self.intercept, trait_index)
        gathered_factors: list[tf.Tensor] = []
        availability: list[tf.Tensor] = []
        for expert_index, factor in enumerate(self.factors):
            index = expert_indices[:, expert_index]
            available = index >= 0
            safe_index = tf.maximum(index, 0)
            gathered = tf.gather(factor, safe_index)
            gathered_factors.append(gathered)
            availability.append(available)
            coefficients = self.correlated_coefficients(
                self.main_coefficients[expert_index]
            )
            trait_coefficients = tf.gather(
                tf.transpose(coefficients), trait_index
            )
            effect = tf.reduce_sum(gathered * trait_coefficients, axis=1)
            trait_eligible = tf.gather(self.eligibility[expert_index], trait_index)
            active = available & trait_eligible
            prediction += effect * tf.cast(active, tf.float32)

        genotype_indices = expert_indices[:, self.genotype_index]
        genotype_available = genotype_indices >= 0
        safe_genotype_indices = tf.maximum(genotype_indices, 0)
        for pair_index, environment_index in enumerate(
            self.reaction_environment_indices
        ):
            environment_indices = expert_indices[:, environment_index]
            environment_available = environment_indices >= 0
            safe_environment_indices = tf.maximum(environment_indices, 0)
            g_feature = tf.gather(
                self.reaction_genotype_features[pair_index], safe_genotype_indices
            )
            e_feature = tf.gather(
                self.reaction_environment_features[pair_index], safe_environment_indices
            )
            reaction_feature = g_feature * e_feature * self.reaction_scale
            coefficients = self.correlated_coefficients(
                self.reaction_coefficients[pair_index]
            )
            trait_coefficients = tf.gather(
                tf.transpose(coefficients), trait_index
            )
            effect = tf.reduce_sum(reaction_feature * trait_coefficients, axis=1)
            trait_eligible = tf.gather(
                self.eligibility[environment_index], trait_index
            )
            active = genotype_available & environment_available & trait_eligible
            prediction += effect * tf.cast(active, tf.float32)
        return prediction

    def residual_scales(self) -> tf.Tensor:
        return tf.nn.softplus(self.raw_residual_scale) + self.residual_scale_floor

    def regularization_loss(self) -> tf.Tensor:
        coefficients = [*self.main_coefficients, *self.reaction_coefficients]
        return self.ridge_penalty * tf.add_n(
            [tf.reduce_sum(tf.square(value)) for value in coefficients]
        )

    def component_variance_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for spec, raw in zip(self.expert_specs, self.main_coefficients):
            correlated = self.correlated_coefficients(raw).numpy()
            for trait_index, trait in enumerate(self.trait_names):
                rows.append(
                    {
                        "component": str(spec["kernel"]),
                        "component_type": "main",
                        "trait_name_canonical": trait,
                        "coefficient_mean_square": float(
                            np.mean(np.square(correlated[:, trait_index]))
                        ),
                    }
                )
        for environment_index, raw in zip(
            self.reaction_environment_indices, self.reaction_coefficients
        ):
            correlated = self.correlated_coefficients(raw).numpy()
            component = (
                f"{self.expert_specs[self.genotype_index]['kernel']}x"
                f"{self.expert_specs[environment_index]['kernel']}"
            )
            for trait_index, trait in enumerate(self.trait_names):
                rows.append(
                    {
                        "component": component,
                        "component_type": "reaction",
                        "trait_name_canonical": trait,
                        "coefficient_mean_square": float(
                            np.mean(np.square(correlated[:, trait_index]))
                        ),
                    }
                )
        return pd.DataFrame(rows)


def make_dataset(
    frame: pd.DataFrame,
    expert_columns: list[str],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            (
                frame[expert_columns].to_numpy(dtype=np.int32),
                frame["trait_index"].to_numpy(dtype=np.int32),
            ),
            frame["y_scaled"].to_numpy(dtype=np.float32),
            frame["loss_weight"].to_numpy(dtype=np.float32),
        )
    )
    if shuffle:
        dataset = dataset.shuffle(
            min(len(frame), 100_000), seed=seed, reshuffle_each_iteration=True
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def predict_scaled(
    model: tf.keras.Model,
    frame: pd.DataFrame,
    expert_columns: list[str],
    batch_size: int,
) -> np.ndarray:
    dataset = make_dataset(frame, expert_columns, batch_size, False, 0)
    predictions = [model(inputs, training=False).numpy() for inputs, _, _ in dataset]
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)


def macro_standardized_rmse(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    scores = []
    local = frame.reset_index(drop=True)
    for _, positions in local.groupby("trait_index", sort=True).groups.items():
        index = np.asarray(list(positions), dtype=int)
        error = prediction[index] - local.loc[index, "y_scaled"].to_numpy(dtype=float)
        scores.append(float(np.sqrt(np.mean(np.square(error)))))
    return float(np.mean(scores)) if scores else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a leakage-safe penalized multi-trait reaction-norm mixed baseline."
        )
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--trait-order", type=Path, required=True)
    parser.add_argument("--kernel-registry", type=Path, required=True)
    parser.add_argument("--certification-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="multitrait_reaction_norm")
    parser.add_argument("--model-label", default="multitrait_reaction_norm")
    parser.add_argument("--hyperparameter-label", required=True)
    parser.add_argument("--genotype-kernel", default="K_A_CANONICAL_V3")
    parser.add_argument("--trait", action="append")
    parser.add_argument("--exclude-kernel", action="append", default=[])
    parser.add_argument("--include-disabled-kernel", action="append", default=[])
    parser.add_argument("--split", default="gho_environment")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split-contract", type=Path)
    parser.add_argument("--evaluation-protocol", type=Path)
    parser.add_argument("--reaction-protocol", type=Path, required=True)
    parser.add_argument("--outer-evaluation-protocol", type=Path)
    parser.add_argument("--reaction-selection-lock", type=Path)
    parser.add_argument("--evaluation-scenario", choices=sorted(SCENARIO_MODES))
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--inner-fold", type=int)
    parser.add_argument(
        "--evaluation-stage",
        choices=["discovery", "inner_selection", "outer_evaluation"],
        default="discovery",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-train-rows-per-trait", type=int, default=100)
    parser.add_argument("--min-eval-rows-per-trait", type=int, default=20)
    parser.add_argument("--max-rank-genotype", type=int, default=128)
    parser.add_argument("--max-rank-environment", type=int, default=64)
    parser.add_argument("--reaction-rank", type=int, default=32)
    parser.add_argument("--trait-covariance-shrinkage", type=float, default=0.25)
    parser.add_argument("--trait-covariance-minimum-pairs", type=int, default=20)
    parser.add_argument("--ridge-penalty", type=float, default=1e-4)
    parser.add_argument("--residual-scale-floor", type=float, default=0.05)
    parser.add_argument(
        "--factorization-mode",
        choices=["full_transductive", "train_nystrom"],
        default="train_nystrom",
    )
    parser.add_argument("--factor-cache", type=Path)
    parser.add_argument("--no-center-kernels", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--intra-op-threads", type=int, default=16)
    parser.add_argument("--inter-op-threads", type=int, default=2)
    parser.add_argument(
        "--stage1-policy",
        choices=["existing_adjusted", "leakage_safe_by_scenario"],
        default="leakage_safe_by_scenario",
    )
    parser.add_argument("--fold-local-weights", action="store_true")
    parser.add_argument("--weight-var-floor-quantile", type=float, default=0.01)
    parser.add_argument("--weight-missing-var-quantile", type=float, default=0.75)
    parser.add_argument("--weight-clip-quantile", type=float, default=0.99)
    parser.add_argument("--weight-power", type=float, default=0.0)
    parser.add_argument("--weight-min-effective-sample-fraction", type=float, default=1.0)
    parser.add_argument("--weight-max-top-1pct-share", type=float, default=0.02)
    args = parser.parse_args()
    if args.reaction_rank < 2:
        raise SystemExit("--reaction-rank must be at least 2")

    reaction_protocol = json.loads(args.reaction_protocol.read_text(encoding="utf-8"))
    if reaction_protocol.get("status") != "frozen_before_inner_validation":
        raise SystemExit("Reaction-norm protocol is not frozen before inner validation")
    outer_protocol = None
    selection_lock = None
    if args.evaluation_stage == "outer_evaluation":
        if args.outer_evaluation_protocol is None or args.reaction_selection_lock is None:
            raise SystemExit(
                "Outer reaction-norm evaluation requires its frozen protocol and "
                "completed inner-selection lock"
            )
        outer_protocol = json.loads(
            args.outer_evaluation_protocol.read_text(encoding="utf-8")
        )
        selection_lock = json.loads(
            args.reaction_selection_lock.read_text(encoding="utf-8")
        )
        outer_checks = {
            "outer_protocol_status": outer_protocol.get("status")
            == "frozen_after_inner_validation_before_outer_test",
            "inner_protocol_version": outer_protocol.get(
                "inner_reaction_protocol_version"
            )
            == reaction_protocol.get("protocol_version"),
            "inner_protocol_sha256": outer_protocol.get(
                "inner_reaction_protocol_sha256"
            )
            == file_sha256(args.reaction_protocol),
            "selection_lock_status": selection_lock.get("status") == "PASS",
            "selection_lock_outer_unread": selection_lock.get(
                "outer_test_metrics_read"
            )
            is False,
            "selection_lock_final_holdout_unread": selection_lock.get(
                "final_holdout_outcomes_read"
            )
            is False,
            "selection_lock_allows_outer": selection_lock.get(
                "outer_evaluation_allowed"
            )
            is True,
            "selection_lock_protocol": selection_lock.get(
                "outer_evaluation_protocol_sha256"
            )
            == file_sha256(args.outer_evaluation_protocol),
            "selection_lock_candidate": selection_lock.get("selected_candidate")
            == outer_protocol.get("selected_candidate"),
            "no_further_selection": outer_protocol.get("model_contract", {}).get(
                "no_further_hyperparameter_selection"
            )
            is True,
            "final_holdout_unavailable": outer_protocol.get(
                "model_contract", {}
            ).get("final_holdout_available")
            is False,
        }
        failed_outer = sorted(
            name for name, passed in outer_checks.items() if not passed
        )
        if failed_outer:
            raise SystemExit(
                "Reaction-norm outer-evaluation authorization failed: "
                + ", ".join(failed_outer)
            )
    elif args.outer_evaluation_protocol is not None or args.reaction_selection_lock is not None:
        raise SystemExit(
            "Outer-evaluation authorization artifacts may only be used for outer evaluation"
        )
    if reaction_protocol.get("genotype_kernel") != args.genotype_kernel:
        raise SystemExit("Configured genotype kernel disagrees with the reaction protocol")
    candidate_by_name = {
        str(value["name"]): value for value in reaction_protocol.get("candidates", [])
    }
    if args.hyperparameter_label not in candidate_by_name:
        raise SystemExit(
            "Hyperparameter label is absent from the frozen reaction protocol: "
            f"{args.hyperparameter_label}"
        )
    candidate_contract = candidate_by_name[args.hyperparameter_label]
    training_contract = reaction_protocol["training"]
    exact_contract = {
        "reaction_rank": (args.reaction_rank, int(candidate_contract["reaction_rank"])),
        "max_rank_genotype": (
            args.max_rank_genotype,
            int(training_contract["max_rank_genotype"]),
        ),
        "max_rank_environment": (
            args.max_rank_environment,
            int(training_contract["max_rank_environment"]),
        ),
        "trait_covariance_minimum_pairs": (
            args.trait_covariance_minimum_pairs,
            int(training_contract["trait_covariance_minimum_pairs"]),
        ),
        "epochs": (args.epochs, int(training_contract["epochs"])),
        "batch_size": (args.batch_size, int(training_contract["batch_size"])),
        "patience": (args.patience, int(training_contract["patience"])),
        "intra_op_threads": (
            args.intra_op_threads,
            int(training_contract["intra_op_threads"]),
        ),
        "inter_op_threads": (
            args.inter_op_threads,
            int(training_contract["inter_op_threads"]),
        ),
        "min_train_rows_per_trait": (
            args.min_train_rows_per_trait,
            int(training_contract["minimum_train_rows_per_trait"]),
        ),
        "min_eval_rows_per_trait": (
            args.min_eval_rows_per_trait,
            int(training_contract["minimum_evaluation_rows_per_trait"]),
        ),
    }
    floating_contract = {
        "trait_covariance_shrinkage": (
            args.trait_covariance_shrinkage,
            float(candidate_contract["trait_covariance_shrinkage"]),
        ),
        "ridge_penalty": (
            args.ridge_penalty,
            float(candidate_contract["ridge_penalty"]),
        ),
        "residual_scale_floor": (
            args.residual_scale_floor,
            float(training_contract["residual_scale_floor"]),
        ),
        "learning_rate": (
            args.learning_rate,
            float(training_contract["learning_rate"]),
        ),
        "weight_power": (args.weight_power, float(training_contract["weight_power"])),
        "weight_var_floor_quantile": (
            args.weight_var_floor_quantile,
            float(training_contract["weight_variance_floor_quantile"]),
        ),
        "weight_missing_var_quantile": (
            args.weight_missing_var_quantile,
            float(training_contract["weight_missing_variance_quantile"]),
        ),
        "weight_clip_quantile": (
            args.weight_clip_quantile,
            float(training_contract["weight_clip_quantile"]),
        ),
        "weight_min_effective_sample_fraction": (
            args.weight_min_effective_sample_fraction,
            float(training_contract["minimum_effective_sample_fraction"]),
        ),
        "weight_max_top_1pct_share": (
            args.weight_max_top_1pct_share,
            float(training_contract["maximum_top_1pct_share"]),
        ),
    }
    mismatches = [
        f"{name}: observed={observed} expected={expected}"
        for name, (observed, expected) in exact_contract.items()
        if observed != expected
    ]
    mismatches.extend(
        f"{name}: observed={observed} expected={expected}"
        for name, (observed, expected) in floating_contract.items()
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
    )
    requested_trait_contract = {
        str(value).strip().upper() for value in (args.trait or []) if str(value).strip()
    }
    expected_trait_contract = {
        str(value).strip().upper() for value in reaction_protocol.get("traits", [])
    }
    if requested_trait_contract != expected_trait_contract:
        mismatches.append(
            "traits: "
            f"observed={sorted(requested_trait_contract)} "
            f"expected={sorted(expected_trait_contract)}"
        )
    if args.factorization_mode != training_contract["factorization_mode"]:
        mismatches.append(
            "factorization_mode: "
            f"observed={args.factorization_mode} "
            f"expected={training_contract['factorization_mode']}"
        )
    if args.no_center_kernels or training_contract["kernel_centering"] is not True:
        mismatches.append("kernel_centering must remain enabled")
    if not args.fold_local_weights or training_contract["fold_local_weights"] is not True:
        mismatches.append("fold_local_weights must remain enabled")
    if args.stage1_policy != training_contract["stage1_policy"]:
        mismatches.append(
            "stage1_policy: "
            f"observed={args.stage1_policy} expected={training_contract['stage1_policy']}"
        )
    if args.evaluation_stage not in {"inner_selection", "outer_evaluation"}:
        mismatches.append("evaluation_stage must be inner_selection or outer_evaluation")
    expected_scenarios = (
        {str(reaction_protocol.get("scenario"))}
        if args.evaluation_stage == "inner_selection"
        else set((outer_protocol or {}).get("scenarios", {}))
    )
    if args.evaluation_scenario not in expected_scenarios:
        mismatches.append(
            "evaluation_scenario: "
            f"observed={args.evaluation_scenario} "
            f"expected={sorted(expected_scenarios)}"
        )
    if args.evaluation_stage == "outer_evaluation":
        selected_candidate = str((outer_protocol or {}).get("selected_candidate", ""))
        if args.hyperparameter_label != selected_candidate:
            mismatches.append(
                "outer_candidate: "
                f"observed={args.hyperparameter_label} expected={selected_candidate}"
            )
        selected_model_label = str(
            (outer_protocol or {}).get("selected_model_label", "")
        )
        if args.model_label != selected_model_label:
            mismatches.append(
                "outer_model_label: "
                f"observed={args.model_label} expected={selected_model_label}"
            )
        selected_configuration = (outer_protocol or {}).get(
            "selected_configuration", {}
        )
        observed_configuration = {
            "max_rank_genotype": args.max_rank_genotype,
            "max_rank_environment": args.max_rank_environment,
            "reaction_rank": args.reaction_rank,
            "trait_covariance_shrinkage": args.trait_covariance_shrinkage,
            "trait_covariance_minimum_pairs": args.trait_covariance_minimum_pairs,
            "ridge_penalty": args.ridge_penalty,
            "residual_scale_floor": args.residual_scale_floor,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "intra_op_threads": args.intra_op_threads,
            "inter_op_threads": args.inter_op_threads,
        }
        if observed_configuration != selected_configuration:
            mismatches.append(
                "outer_selected_configuration: "
                f"observed={observed_configuration} expected={selected_configuration}"
            )
        if set((outer_protocol or {}).get("required_kernels", [])) != set(
            reaction_protocol.get("required_kernels", [])
        ):
            mismatches.append("outer required kernels disagree with inner selection")
        if set((outer_protocol or {}).get("traits", [])) != set(
            reaction_protocol.get("traits", [])
        ):
            mismatches.append("outer traits disagree with inner selection")
    if mismatches:
        raise SystemExit(
            "Reaction-norm frozen training contract failed: " + "; ".join(mismatches)
        )

    manifest_arguments = [
        args.split_manifest,
        args.split_contract,
        args.evaluation_scenario,
        args.outer_fold,
        args.inner_fold,
    ]
    if any(value is not None for value in manifest_arguments) and not all(
        value is not None for value in manifest_arguments
    ):
        raise SystemExit("Nested evaluation arguments must be supplied together")
    if args.evaluation_stage != "discovery" and args.split_manifest is None:
        raise SystemExit("Non-discovery evaluation requires an immutable split manifest")
    protocol = None
    if args.split_manifest is not None:
        protocol = load_protocol(args.evaluation_protocol)
        require_non_discovery_seed(args.seed, protocol)
        if outer_protocol is not None:
            if outer_protocol.get("evaluation_protocol_version") != protocol.get(
                "protocol_version"
            ) or outer_protocol.get("evaluation_protocol_sha256") != protocol.get(
                "protocol_sha256"
            ):
                raise SystemExit(
                    "Outer reaction-norm protocol disagrees with the immutable "
                    "evaluation protocol"
                )

    certification = json.loads(args.certification_summary.read_text(encoding="utf-8"))
    if certification.get("status") != "PASS":
        raise SystemExit("Kernel certification is not PASS")
    require_certified_file(args.ledger, certification.get("ledger_identity", {}), "Ledger")
    require_certified_file(
        args.kernel_registry, certification.get("registry_identity", {}), "Kernel registry"
    )
    registry = pd.read_csv(args.kernel_registry, sep="\t")
    enabled = registry["enabled_default"].map(parse_bool)
    enabled |= registry["kernel"].isin(args.include_disabled_kernel)
    enabled &= ~registry["kernel"].isin(args.exclude_kernel)
    registry = registry[enabled].copy().reset_index(drop=True)
    if registry.empty:
        raise SystemExit("No kernel experts remain after registry filtering")
    required_kernels = set(reaction_protocol.get("required_kernels", []))
    active_kernels = set(registry["kernel"].astype(str))
    forbidden_kernels = set(reaction_protocol.get("forbidden_kernels", []))
    forbidden_prefixes = tuple(reaction_protocol.get("forbidden_kernel_prefixes", []))
    unexpected = sorted(active_kernels - required_kernels)
    missing_required = sorted(required_kernels - active_kernels)
    forbidden_active = sorted(
        kernel
        for kernel in active_kernels
        if kernel in forbidden_kernels or kernel.startswith(forbidden_prefixes)
    )
    if missing_required or unexpected or forbidden_active:
        raise SystemExit(
            "Reaction-norm active-kernel contract failed: "
            f"missing={missing_required}; unexpected={unexpected}; "
            f"forbidden={forbidden_active}"
        )
    generic_environment = set(
        reaction_protocol.get("generic_environment_kernels", [])
    )
    trait_specific_environment = set(
        reaction_protocol.get("trait_specific_environment_kernels", [])
    )
    trait_specific_eligibility = {
        str(kernel): {str(trait).upper() for trait in traits}
        for kernel, traits in reaction_protocol.get(
            "trait_specific_environment_eligibility", {}
        ).items()
    }
    for _, spec in registry.iterrows():
        kernel = str(spec["kernel"])
        axis = str(spec["axis"])
        eligible = trait_set(spec["eligible_traits"])
        if kernel == args.genotype_kernel:
            valid_role = axis == "genotype" and eligible is None
        elif kernel in generic_environment:
            valid_role = axis == "environment" and eligible is None
        elif kernel in trait_specific_environment:
            valid_role = (
                axis == "environment"
                and eligible == trait_specific_eligibility.get(kernel, set())
            )
        else:
            valid_role = False
        if not valid_role or not parse_bool(spec["interaction_enabled"]):
            raise SystemExit(
                "Reaction-norm kernel role contract failed: "
                f"kernel={kernel}; axis={axis}; eligible_traits={spec['eligible_traits']}; "
                f"interaction_enabled={spec['interaction_enabled']}"
            )
    certified_kernels = certification.get("kernel_identities", {})
    certified_orders = certification.get("order_identities", {})
    certified_coverage = certification.get("coverage_identities", {})
    for _, spec in registry.iterrows():
        name = str(spec["kernel"])
        require_certified_file(
            Path(str(spec["kernel_path"])), certified_kernels.get(name, {}), name
        )
        require_certified_file(
            Path(str(spec["order_path"])), certified_orders.get(name, {}), f"{name} order"
        )
        coverage_text = str(spec.get("coverage_path", "")).strip()
        if coverage_text and coverage_text.lower() != "nan":
            require_certified_file(
                Path(coverage_text), certified_coverage.get(name, {}), f"{name} coverage"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(args.intra_op_threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.inter_op_threads)

    ledger = read_table(args.ledger)
    trait_order = pd.read_csv(args.trait_order, sep="\t")
    if args.trait:
        requested = {value.strip().upper() for value in args.trait}
        ledger = ledger[
            ledger["trait_name_canonical"].fillna("").astype(str).str.upper().isin(requested)
        ].copy()
    ledger = ledger.reset_index(drop=True)
    requested_traits = sorted(ledger["trait_name_canonical"].unique().tolist())
    external_split_identity: dict[str, object] = {}
    if args.split_manifest is not None:
        contract = verify_manifest_contract(args.split_manifest, args.split_contract)
        if contract.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise SystemExit("Evaluation manifest and protocol hashes do not match")
        if file_sha256(args.ledger) != contract.get("ledger_sha256"):
            raise SystemExit("Evaluation manifest was frozen against another ledger")
        split_manifest = pd.read_csv(args.split_manifest, sep="\t", dtype=str)
        train_index, val_index, test_index, omitted_index, leakage = assign_nested_split(
            ledger,
            split_manifest,
            scenario=args.evaluation_scenario,
            outer_fold=args.outer_fold,
            inner_fold=args.inner_fold,
        )
        canonical_split = SCENARIO_MODES[args.evaluation_scenario]
        external_split_identity = manifest_identity(args.split_manifest, args.split_contract)
        external_split_identity.update(
            {
                "scenario": args.evaluation_scenario,
                "outer_fold": args.outer_fold,
                "inner_fold": args.inner_fold,
                "omitted_rows": int(len(omitted_index)),
            }
        )
    else:
        canonical_split = canonical_split_mode(args.split, warn=True)
        group_col = split_group_column(canonical_split)
        train_index, val_index, test_index = make_split(
            ledger,
            canonical_split,
            args.seed,
            args.test_fraction,
            args.val_fraction,
            group_col,
        )
        leakage = split_leakage_record(
            ledger,
            args.seed,
            canonical_split,
            train_index,
            val_index,
            test_index,
            group_col,
        )
    if leakage["leakage_status"] != "pass":
        raise SystemExit(f"Split leakage detected: {leakage}")
    split_labels = np.full(len(ledger), "", dtype=object)
    split_labels[train_index] = "train"
    split_labels[val_index] = "val"
    split_labels[test_index] = "test"
    ledger["split"] = split_labels

    protected_outcome_rows_cleared = 0
    if args.evaluation_stage == "inner_selection":
        protected = ~ledger["split"].isin(["train", "val"])
        protected_outcome_rows_cleared = int(protected.sum())
        outcome_columns = [
            column
            for column in (
                "phenotype_value",
                "raw_mean",
                "raw_sd",
                "var_g_e",
                "weight_g_e",
            )
            if column in ledger.columns
        ]
        ledger.loc[protected, outcome_columns] = np.nan

    support = ledger.groupby(["trait_name_canonical", "split"]).size().unstack(fill_value=0)
    for column in ("train", "val", "test"):
        if column not in support:
            support[column] = 0
    retained = support[
        support["train"].ge(args.min_train_rows_per_trait)
        & support["val"].ge(args.min_eval_rows_per_trait)
        & support["test"].ge(args.min_eval_rows_per_trait)
    ].index.tolist()
    support_report = support[["train", "val", "test"]].reset_index()
    support_report["retained"] = support_report["trait_name_canonical"].isin(retained)
    if len(retained) < 2:
        raise SystemExit(f"Multi-trait training requires two supported traits: {retained}")
    retained_splits = ["train", "val"] if args.evaluation_stage == "inner_selection" else [
        "train",
        "val",
        "test",
    ]
    ledger = ledger[
        ledger["trait_name_canonical"].isin(retained)
        & ledger["split"].isin(retained_splits)
    ].copy().reset_index(drop=True)

    stage1_policy_applied = "existing_adjusted"
    if args.stage1_policy == "leakage_safe_by_scenario" and args.split_manifest is not None:
        if args.evaluation_scenario in {
            "unseen_genotypes",
            "unseen_genotypes_and_environments",
        }:
            required_raw = {"raw_mean", "raw_sd", "n_plot_records"}
            missing_raw = sorted(required_raw.difference(ledger.columns))
            if missing_raw:
                raise SystemExit(f"Raw plot summaries are missing: {missing_raw}")
            raw_mean = pd.to_numeric(ledger["raw_mean"], errors="coerce")
            if not np.isfinite(raw_mean.to_numpy(dtype=float)).all():
                raise SystemExit("raw_mean contains non-finite values")
            ledger["phenotype_value"] = raw_mean
            raw_sd = pd.to_numeric(ledger["raw_sd"], errors="coerce")
            raw_n = pd.to_numeric(ledger["n_plot_records"], errors="coerce")
            ledger["var_g_e"] = (np.square(raw_sd) / raw_n.where(raw_n.gt(0))).replace(
                [np.inf, -np.inf], np.nan
            )
            stage1_policy_applied = "genotype_environment_raw_mean_and_sampling_variance"
        else:
            stage1_policy_applied = "environment_isolated_stage1_adjustment"
    ledger["phenotype_value"] = pd.to_numeric(
        ledger["phenotype_value"], errors="coerce"
    )
    if not np.isfinite(ledger["phenotype_value"].to_numpy(dtype=float)).all():
        raise SystemExit("Phenotype values are non-finite")

    retained_order = trait_order[
        trait_order["trait_name_canonical"].isin(retained)
    ].copy().sort_values("trait_index")
    retained_order["source_trait_index"] = retained_order["trait_index"]
    retained_order["trait_index"] = np.arange(len(retained_order), dtype=np.int32)
    trait_map = dict(
        zip(retained_order["trait_name_canonical"], retained_order["trait_index"])
    )
    ledger["trait_index"] = ledger["trait_name_canonical"].map(trait_map).astype(np.int32)
    trait_names = retained_order["trait_name_canonical"].tolist()
    retained_upper = {value.upper() for value in trait_names}
    registry = registry[
        registry["eligible_traits"].map(
            lambda value: trait_set(value) is None
            or bool(trait_set(value) & retained_upper)
        )
    ].copy().reset_index(drop=True)
    ledger, expert_columns, coverage = add_expert_indices(ledger, registry)

    weight_parameters = pd.DataFrame()
    if args.fold_local_weights:
        weight_parameters = fit_precision_weight_transform(
            ledger[ledger["split"].eq("train")],
            floor_quantile=args.weight_var_floor_quantile,
            missing_variance_quantile=args.weight_missing_var_quantile,
            clip_quantile=args.weight_clip_quantile,
            weight_power=args.weight_power,
            min_effective_sample_fraction=args.weight_min_effective_sample_fraction,
            max_top_1pct_share=args.weight_max_top_1pct_share,
        )
        ledger = apply_precision_weight_transform(ledger, weight_parameters)
    for column in ("phenotype_value", "weight_g_e"):
        values = pd.to_numeric(ledger[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise SystemExit(f"Ledger column {column} contains non-finite values")
        ledger[column] = values
    if np.any(ledger["weight_g_e"].to_numpy(dtype=float) <= 0):
        raise SystemExit("Observation weights are non-positive")

    scaling_rows = []
    ledger["y_scaled"] = np.nan
    for trait, group in ledger.groupby("trait_name_canonical", sort=True):
        local_train = group[group["split"].eq("train")]
        mean, sd = weighted_mean_std(
            local_train["phenotype_value"].to_numpy(dtype=float),
            local_train["weight_g_e"].to_numpy(dtype=float),
        )
        ledger.loc[group.index, "y_scaled"] = (
            group["phenotype_value"] - mean
        ) / sd
        scaling_rows.append(
            {"trait_name_canonical": trait, "train_mean": mean, "train_sd": sd}
        )
    scaling = pd.DataFrame(scaling_rows)
    scale_map = scaling.set_index("trait_name_canonical").to_dict("index")
    train = ledger[ledger["split"].eq("train")].copy()
    val = ledger[ledger["split"].eq("val")].copy()
    test = ledger[ledger["split"].eq("test")].copy()
    train["loss_weight"] = trait_balanced_loss_weights(
        train["trait_index"].to_numpy(), train["weight_g_e"].to_numpy()
    )
    val["loss_weight"] = 1.0
    if not test.empty:
        test["loss_weight"] = 1.0

    covariance, covariance_sqrt, pair_counts = estimate_trait_covariance(
        train,
        trait_names,
        args.trait_covariance_shrinkage,
        args.trait_covariance_minimum_pairs,
    )
    covariance_frame = pd.DataFrame(covariance, index=trait_names, columns=trait_names)
    pair_count_frame = pd.DataFrame(pair_counts, index=trait_names, columns=trait_names)

    effective_mode = effective_factorization_mode(
        args.factorization_mode, canonical_split, warn=True
    )
    centered = not args.no_center_kernels
    active_positions = []
    train_ids_by_expert = []
    factor_configurations = []
    support_rows = []
    for expert_index, (_, spec) in enumerate(registry.iterrows()):
        column = expert_columns[expert_index]
        eligible = trait_set(spec["eligible_traits"])
        local = train if eligible is None else train[
            train["trait_name_canonical"].str.upper().isin(eligible)
        ]
        ids = np.unique(local.loc[local[column].ge(0), column].to_numpy(dtype=np.int32))
        minimum_value = spec.get("minimum_training_entities", 2)
        minimum = 2 if pd.isna(minimum_value) else max(2, int(minimum_value))
        supported, reason = factorization_training_support(
            ids, effective_mode, centered, minimum_ids=minimum
        )
        support_rows.append(
            {
                "kernel": str(spec["kernel"]),
                "axis": str(spec["axis"]),
                "eligible_training_rows": int(len(local)),
                "unique_training_kernel_ids": int(len(ids)),
                "minimum_training_kernel_ids": minimum,
                "effective_factorization_mode": effective_mode,
                "kernel_centered": centered,
                "fold_status": "ACTIVE" if supported else "DROPPED",
                "inactive_reason": reason,
            }
        )
        if not supported:
            continue
        active_positions.append(expert_index)
        train_ids = ids if effective_mode == "train_nystrom" else None
        train_ids_by_expert.append(train_ids)
        max_rank = (
            args.max_rank_genotype
            if str(spec["axis"]) == "genotype"
            else args.max_rank_environment
        )
        factor_configurations.append(
            {
                "kernel": str(spec["kernel"]),
                "identity": file_identity(Path(str(spec["kernel_path"]))),
                "rank": min(int(spec["rank"]), max_rank),
                "train_index_digest": index_digest(train_ids),
            }
        )
    fold_support = pd.DataFrame(support_rows)
    fold_support_path = args.out_dir / f"{args.prefix}_fold_expert_support.tsv"
    fold_support.to_csv(fold_support_path, sep="\t", index=False)
    registry = registry.iloc[active_positions].copy().reset_index(drop=True)
    expert_columns = [expert_columns[index] for index in active_positions]
    dropped_required = sorted(required_kernels - set(registry["kernel"].astype(str)))
    if dropped_required:
        raise SystemExit(
            "Fold support removed required reaction-norm kernels: "
            f"{dropped_required}"
        )
    if args.genotype_kernel not in set(registry["kernel"]):
        raise SystemExit("Fold filtering removed the canonical genotype kernel")
    if not set(registry["axis"]).issuperset({"genotype", "environment"}):
        raise SystemExit("Reaction-norm model requires genotype and environment axes")
    for frame in (ledger, train, val, test):
        assign_active_marker_coverage(frame, registry, expert_columns)

    cache_configuration = {
        "experts": factor_configurations,
        "effective_factorization_mode": effective_mode,
        "kernel_centered": centered,
    }
    cache_metadata_path = args.factor_cache.with_suffix(".json") if args.factor_cache else None
    factors: list[np.ndarray] = []
    factor_metadata: dict[str, object] = {}
    cache_loaded = False
    if args.factor_cache and args.factor_cache.exists() and cache_metadata_path.exists():
        cached_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        if cached_metadata.get("configuration") == cache_configuration:
            with np.load(args.factor_cache) as cached:
                factors = [
                    cached[f"factor__{safe_name(str(spec['kernel']))}"].astype(np.float32)
                    for _, spec in registry.iterrows()
                ]
            factor_metadata = cached_metadata["factorizations"]
            cache_loaded = True
    if not cache_loaded:
        arrays = {}
        for expert_index, (_, spec) in enumerate(registry.iterrows()):
            configuration = factor_configurations[expert_index]
            factor, metadata = kernel_factors(
                Path(str(spec["kernel_path"])),
                int(configuration["rank"]),
                train_ids_by_expert[expert_index],
                jitter=1e-6,
                center=centered,
            )
            factors.append(factor)
            name = str(spec["kernel"])
            arrays[f"factor__{safe_name(name)}"] = factor
            factor_metadata[name] = metadata
        if args.factor_cache:
            args.factor_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.factor_cache, **arrays)
            cache_metadata_path.write_text(
                json.dumps(
                    {"configuration": cache_configuration, "factorizations": factor_metadata},
                    indent=2,
                ),
                encoding="utf-8",
            )

    if len(factors) != len(registry):
        raise RuntimeError(
            f"Factor count mismatch: factors={len(factors)} registry={len(registry)}"
        )
    for factor, (_, spec) in zip(factors, registry.iterrows()):
        expected_rows = int(spec["dimension"])
        if factor.ndim != 2 or factor.shape[0] != expected_rows or factor.shape[1] < 1:
            raise RuntimeError(
                f"Invalid factor shape for {spec['kernel']}: "
                f"observed={factor.shape}; expected_rows={expected_rows}"
            )
        if not np.isfinite(factor).all():
            raise RuntimeError(f"Non-finite factor values for {spec['kernel']}")

    model = MultiTraitReactionNorm(
        registry.to_dict("records"),
        factors,
        trait_names,
        args.genotype_kernel,
        covariance_sqrt,
        args.reaction_rank,
        args.ridge_penalty,
        args.residual_scale_floor,
        args.seed,
    )
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    train_dataset = make_dataset(train, expert_columns, args.batch_size, True, args.seed)

    @tf.function
    def train_step(inputs, target, weight):
        with tf.GradientTape() as tape:
            prediction = model(inputs, training=True)
            trait_index = inputs[1]
            scale = tf.gather(model.residual_scales(), trait_index)
            nll = 0.5 * tf.square((prediction - target) / scale) + tf.math.log(scale)
            loss = tf.reduce_sum(weight * nll) / tf.reduce_sum(weight)
            loss += model.regularization_loss()
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    best_score = float("inf")
    best_weights = None
    stale = 0
    history_rows = []
    for epoch in range(1, args.epochs + 1):
        losses = [
            float(train_step(inputs, target, weight).numpy())
            for inputs, target, weight in train_dataset
        ]
        if not losses or not np.isfinite(losses).all():
            raise RuntimeError(f"Training produced a non-finite loss at epoch {epoch}")
        val_prediction = predict_scaled(model, val, expert_columns, args.batch_size)
        score = macro_standardized_rmse(val, val_prediction)
        history_rows.append(
            {
                "epoch": epoch,
                "train_gaussian_nll": float(np.mean(losses)),
                "val_macro_nrmse": float(score),
            }
        )
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(history_rows[-1]), flush=True)
        if score < best_score:
            best_score = score
            best_weights = model.get_weights()
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_weights is not None:
        model.set_weights(best_weights)

    metric_output = []
    prediction_outputs = []
    evaluation_frames = [("val", val)]
    if args.evaluation_stage != "inner_selection":
        evaluation_frames.append(("test", test))
    for split_name, frame in evaluation_frames:
        frame = frame.copy().reset_index(drop=True)
        frame["y_pred_scaled"] = predict_scaled(
            model, frame, expert_columns, args.batch_size
        )
        frame["y_pred"] = [
            value * scale_map[trait]["train_sd"] + scale_map[trait]["train_mean"]
            for value, trait in zip(frame["y_pred_scaled"], frame["trait_name_canonical"])
        ]
        frame["y_pred_train_mean"] = frame["trait_name_canonical"].map(
            {trait: values["train_mean"] for trait, values in scale_map.items()}
        )
        metric_output.extend(
            metric_rows(
                frame, split_name, model_name=args.model_label, prediction_col="y_pred"
            )
        )
        metric_output.extend(
            metric_rows(
                frame,
                split_name,
                model_name="train_mean",
                prediction_col="y_pred_train_mean",
            )
        )
        prediction_outputs.append(frame)
    metrics = pd.DataFrame(metric_output)
    macro = (
        metrics[metrics["coverage_group"].eq("all")]
        .groupby(["split", "model"])[
            ["weighted_rmse", "unweighted_rmse", "normalized_rmse", "pearson", "prediction_sd_ratio"]
        ]
        .mean()
        .reset_index()
        .rename(columns=lambda column: f"macro_{column}" if column not in {"split", "model"} else column)
    )
    model_metrics = metrics[metrics["model"].eq(args.model_label)]
    baseline_metrics = metrics[metrics["model"].eq("train_mean")]
    improvement = model_metrics.merge(
        baseline_metrics,
        on=["split", "coverage_group", "trait_name_canonical"],
        suffixes=("_model", "_train_mean"),
        validate="one_to_one",
    )
    for metric in ("weighted_rmse", "unweighted_rmse", "normalized_rmse"):
        improvement[f"{metric}_improvement"] = (
            improvement[f"{metric}_train_mean"] - improvement[f"{metric}_model"]
        )

    predictions = pd.concat(prediction_outputs, ignore_index=True)
    residual_scales = model.residual_scales().numpy()
    residual_frame = scaling.copy()
    residual_frame["residual_scale_standardized"] = residual_scales
    residual_frame["residual_scale_original_units"] = (
        residual_frame["residual_scale_standardized"] * residual_frame["train_sd"]
    )
    metrics.to_csv(args.out_dir / f"{args.prefix}_trait_metrics.tsv", sep="\t", index=False)
    macro.to_csv(args.out_dir / f"{args.prefix}_macro_metrics.tsv", sep="\t", index=False)
    improvement.to_csv(args.out_dir / f"{args.prefix}_vs_train_mean.tsv", sep="\t", index=False)
    pd.DataFrame(history_rows).to_csv(
        args.out_dir / f"{args.prefix}_history.tsv", sep="\t", index=False
    )
    scaling.to_csv(args.out_dir / f"{args.prefix}_trait_scaling.tsv", sep="\t", index=False)
    covariance_frame.to_csv(
        args.out_dir / f"{args.prefix}_trait_covariance.tsv", sep="\t", index=True
    )
    pair_count_frame.to_csv(
        args.out_dir / f"{args.prefix}_trait_covariance_pair_counts.tsv", sep="\t", index=True
    )
    residual_frame.to_csv(
        args.out_dir / f"{args.prefix}_trait_residual_scales.tsv", sep="\t", index=False
    )
    model.component_variance_frame().to_csv(
        args.out_dir / f"{args.prefix}_component_variance_proxies.tsv", sep="\t", index=False
    )
    if not weight_parameters.empty:
        weight_parameters.to_csv(
            args.out_dir / f"{args.prefix}_fold_weight_parameters.tsv", sep="\t", index=False
        )
    support_report.to_csv(
        args.out_dir / f"{args.prefix}_trait_split_support.tsv", sep="\t", index=False
    )
    retained_order.to_csv(
        args.out_dir / f"{args.prefix}_trait_order.tsv", sep="\t", index=False
    )
    coverage.to_csv(args.out_dir / f"{args.prefix}_kernel_coverage.tsv", sep="\t", index=False)
    registry.to_csv(
        args.out_dir / f"{args.prefix}_active_kernel_registry.tsv", sep="\t", index=False
    )
    pd.DataFrame([leakage]).to_csv(
        args.out_dir / f"{args.prefix}_split_leakage_qc.tsv", sep="\t", index=False
    )
    export = predictions.drop(columns=expert_columns, errors="ignore")
    write_table(export, args.out_dir / f"{args.prefix}_predictions.parquet")

    metadata = {
        "status": "PASS",
        "trainer_sha256": file_sha256(Path(__file__)),
        "reaction_protocol": {
            "path": str(args.reaction_protocol.resolve()),
            "sha256": file_sha256(args.reaction_protocol),
            "protocol_version": reaction_protocol["protocol_version"],
        },
        "outer_evaluation_protocol": (
            {
                "path": str(args.outer_evaluation_protocol.resolve()),
                "sha256": file_sha256(args.outer_evaluation_protocol),
                "protocol_version": outer_protocol["protocol_version"],
            }
            if outer_protocol is not None
            else {}
        ),
        "reaction_selection_lock": (
            {
                "path": str(args.reaction_selection_lock.resolve()),
                "sha256": file_sha256(args.reaction_selection_lock),
                "selected_candidate": selection_lock["selected_candidate"],
            }
            if selection_lock is not None
            else {}
        ),
        "kernel_factorization_sha256": file_sha256(
            Path(effective_factorization_mode.__code__.co_filename).resolve()
        ),
        "certification_summary_sha256": file_sha256(args.certification_summary),
        "seed": args.seed,
        "evaluation_stage": args.evaluation_stage,
        "evaluation_protocol": (
            {
                "protocol_version": protocol["protocol_version"],
                "protocol_sha256": protocol["protocol_sha256"],
            }
            if protocol is not None
            else {}
        ),
        "external_split": external_split_identity,
        "canonical_split_mode": canonical_split,
        "model_label": args.model_label,
        "hyperparameter_label": args.hyperparameter_label,
        "model_family": "penalized_multitrait_reaction_norm_mixed_model",
        "traits": trait_names,
        "requested_traits": requested_traits,
        "support_filtered_traits": sorted(set(requested_traits) - set(trait_names)),
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "active_kernels": registry["kernel"].tolist(),
        "training_input_identities": training_input_identities(
            certification, registry["kernel"].tolist()
        ),
        "phenotype_preprocessing": {
            "stage1_policy": stage1_policy_applied,
            "trait_scaling_fit_partition": "train",
            "trait_covariance_fit_partition": "train",
            "fold_local_weights": args.fold_local_weights,
            "weight_transform_fit_partition": "train" if args.fold_local_weights else "ledger",
            "outer_test_outcomes_used": False,
            "protected_outcome_rows_cleared_before_preprocessing": (
                protected_outcome_rows_cleared
            ),
        },
        "training_configuration": {
            "max_rank_genotype": args.max_rank_genotype,
            "max_rank_environment": args.max_rank_environment,
            "reaction_rank": args.reaction_rank,
            "trait_covariance_shrinkage": args.trait_covariance_shrinkage,
            "trait_covariance_minimum_pairs": args.trait_covariance_minimum_pairs,
            "ridge_penalty": args.ridge_penalty,
            "residual_scale_floor": args.residual_scale_floor,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "intra_op_threads": args.intra_op_threads,
            "inter_op_threads": args.inter_op_threads,
        },
        "requested_factorization_mode": args.factorization_mode,
        "effective_factorization_mode": effective_mode,
        "kernel_centered": centered,
        "factor_cache_loaded": cache_loaded,
        "factorizations": factor_metadata,
        "trait_covariance": {
            "shrinkage": args.trait_covariance_shrinkage,
            "minimum_pairs": args.trait_covariance_minimum_pairs,
            "minimum_eigenvalue": float(np.linalg.eigvalsh(covariance).min()),
        },
        "reaction_projection_digests": model.reaction_projection_digests,
        "best_validation_macro_nrmse": float(best_score),
        "epochs_completed": len(history_rows),
        "outer_test_metrics_read": args.evaluation_stage == "outer_evaluation",
        "final_holdout_outcomes_read": False,
    }
    (args.out_dir / f"{args.prefix}_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
