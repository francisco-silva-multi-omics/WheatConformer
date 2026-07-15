from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from .kernel_factorization import effective_factorization_mode, kernel_factors
    from .split_utils import canonical_split_mode, make_split, split_group_column, split_leakage_record
except ImportError:
    from kernel_factorization import effective_factorization_mode, kernel_factors
    from split_utils import canonical_split_mode, make_split, split_group_column, split_leakage_record


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except ImportError:
            fallback = path.with_suffix(".tsv.gz")
            if fallback.exists():
                return pd.read_csv(fallback, sep="\t", low_memory=False)
            raise
    return pd.read_csv(path, sep="\t", low_memory=False)


def write_table(frame: pd.DataFrame, path: Path, write_tsv: bool = True) -> None:
    try:
        frame.to_parquet(path, index=False)
    except ImportError:
        write_tsv = True
    if write_tsv:
        frame.to_csv(path.with_suffix(".tsv.gz"), sep="\t", index=False)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(path),
    }


def require_certified_file(path: Path, certified: dict[str, object], label: str) -> None:
    current = file_identity(path)
    expected = {key: certified.get(key) for key in current}
    if current != expected:
        raise SystemExit(
            f"{label} does not match the certified file identity. "
            f"current={current}; certified={expected}"
        )


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def trait_set(value: object) -> set[str] | None:
    text = str(value).strip()
    if text == "*":
        return None
    return {item.strip().upper() for item in text.split(",") if item.strip()}


def safe_weights(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(values) & (values > 0), values, 1.0)


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = safe_weights(weights)
    mean = float(np.sum(values * weights) / np.sum(weights))
    variance = float(np.sum(weights * np.square(values - mean)) / np.sum(weights))
    return mean, max(math.sqrt(variance), 1e-8)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weights = safe_weights(weights)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred, weights = y_true[valid], y_pred[valid], weights[valid]
    error = y_pred - y_true
    weighted_rmse = float(np.sqrt(np.sum(weights * np.square(error)) / np.sum(weights)))
    weighted_mae = float(np.sum(weights * np.abs(error)) / np.sum(weights))
    unweighted_rmse = float(np.sqrt(np.mean(np.square(error))))
    unweighted_mae = float(np.mean(np.abs(error)))
    pearson = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0
        else float("nan")
    )
    true_sd = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 0.0
    pred_sd = float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else 0.0
    return {
        "n": int(len(y_true)),
        "weighted_rmse": weighted_rmse,
        "weighted_mae": weighted_mae,
        "unweighted_rmse": unweighted_rmse,
        "unweighted_mae": unweighted_mae,
        "normalized_rmse": unweighted_rmse / true_sd if true_sd > 0 else float("nan"),
        "pearson": pearson,
        "true_sd": true_sd,
        "pred_sd": pred_sd,
        "prediction_sd_ratio": pred_sd / true_sd if true_sd > 0 else float("nan"),
    }


def trait_balanced_loss_weights(traits: np.ndarray, precision_weights: np.ndarray) -> np.ndarray:
    traits = np.asarray(traits, dtype=np.int32)
    precision_weights = safe_weights(precision_weights)
    output = np.zeros(len(traits), dtype=np.float64)
    for trait in np.unique(traits):
        mask = traits == trait
        local = precision_weights[mask]
        local = local / np.mean(local)
        output[mask] = local / np.sum(mask)
    output *= len(output) / np.sum(output)
    return output.astype(np.float32)


def index_digest(values: np.ndarray | None) -> str:
    if values is None:
        return "all"
    array = np.asarray(values, dtype=np.int32)
    return hashlib.sha256(array.tobytes()).hexdigest()


class MultiTraitKernelExperts(tf.keras.Model):
    def __init__(
        self,
        expert_specs: list[dict[str, object]],
        factors: list[np.ndarray],
        trait_names: list[str],
        latent_dim: int,
        include_genotype_main: bool,
        include_environment_main: bool,
        include_interaction: bool,
        learn_kernel_gates: bool,
        weight_decay: float,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        self.expert_specs = expert_specs
        self.factors = [tf.constant(value, dtype=tf.float32) for value in factors]
        self.trait_names = trait_names
        self.learn_kernel_gates = bool(learn_kernel_gates)
        self.weight_decay = float(weight_decay)
        self.initialization_seed = int(initialization_seed)
        self._initializer_index = 0
        self.intercept = self.add_weight(
            name="trait_intercept", shape=(len(trait_names),), initializer="zeros"
        )

        self.main_projection: list[tf.Variable | None] = []
        self.main_trait: list[tf.Variable | None] = []
        self.main_term_index: dict[int, int] = {}
        self.term_names: list[str] = []
        term_eligibility: list[np.ndarray] = []
        for expert_index, (spec, factor) in enumerate(zip(expert_specs, factors)):
            axis = str(spec["axis"])
            active = (axis == "genotype" and include_genotype_main) or (
                axis == "environment" and include_environment_main
            )
            name = safe_name(str(spec["kernel"]))
            if active:
                self.main_projection.append(
                    self.add_weight(
                        name=f"{name}_main_projection",
                        shape=(factor.shape[1], latent_dim),
                        initializer=self._random_normal_initializer(),
                    )
                )
                self.main_trait.append(
                    self.add_weight(
                        name=f"{name}_trait_main",
                        shape=(len(trait_names), latent_dim),
                        initializer=self._random_normal_initializer(),
                    )
                )
                self.main_term_index[expert_index] = len(self.term_names)
                self.term_names.append(str(spec["kernel"]))
                term_eligibility.append(self._eligibility_vector(spec))
            else:
                self.main_projection.append(None)
                self.main_trait.append(None)

        self.interaction_projection: list[tf.Variable | None] = [None] * len(expert_specs)
        self.interaction_pairs: list[tuple[int, int]] = []
        self.interaction_trait: list[tf.Variable] = []
        if include_interaction:
            genotype_indices = [
                i
                for i, spec in enumerate(expert_specs)
                if spec["axis"] == "genotype" and parse_bool(spec["interaction_enabled"])
            ]
            environment_indices = [
                i
                for i, spec in enumerate(expert_specs)
                if spec["axis"] == "environment" and parse_bool(spec["interaction_enabled"])
            ]
            participating = set(genotype_indices + environment_indices)
            for expert_index in participating:
                name = safe_name(str(expert_specs[expert_index]["kernel"]))
                self.interaction_projection[expert_index] = self.add_weight(
                    name=f"{name}_interaction_projection",
                    shape=(factors[expert_index].shape[1], latent_dim),
                    initializer=self._random_normal_initializer(),
                )
            for genotype_index in genotype_indices:
                for environment_index in environment_indices:
                    g_name = str(expert_specs[genotype_index]["kernel"])
                    e_name = str(expert_specs[environment_index]["kernel"])
                    pair_name = safe_name(f"{g_name}_x_{e_name}")
                    self.interaction_pairs.append((genotype_index, environment_index))
                    self.interaction_trait.append(
                        self.add_weight(
                            name=f"{pair_name}_trait_interaction",
                            shape=(len(trait_names), latent_dim),
                            initializer=self._random_normal_initializer(),
                        )
                    )
                    self.term_names.append(f"{g_name}x{e_name}")
                    term_eligibility.append(
                        self._eligibility_vector(expert_specs[genotype_index])
                        & self._eligibility_vector(expert_specs[environment_index])
                    )

        if not self.term_names:
            raise ValueError("At least one kernel term must be active")
        self.term_eligibility_np = np.stack(term_eligibility, axis=1).astype(bool)
        if not self.term_eligibility_np.any(axis=1).all():
            unsupported = [
                trait_names[i] for i in np.flatnonzero(~self.term_eligibility_np.any(axis=1))
            ]
            raise ValueError(f"Traits have no eligible kernel terms: {unsupported}")
        self.term_eligibility = tf.constant(self.term_eligibility_np)
        self.gate_logits = (
            self.add_weight(
                name="trait_kernel_gate_logits",
                shape=(len(trait_names), len(self.term_names)),
                initializer="zeros",
            )
            if self.learn_kernel_gates
            else None
        )

    def _random_normal_initializer(self) -> tf.keras.initializers.RandomNormal:
        self._initializer_index += 1
        return tf.keras.initializers.RandomNormal(
            stddev=0.02,
            seed=self.initialization_seed + self._initializer_index,
        )

    def _eligibility_vector(self, spec: dict[str, object]) -> np.ndarray:
        eligible = trait_set(spec["eligible_traits"])
        if eligible is None:
            return np.ones(len(self.trait_names), dtype=bool)
        return np.asarray([trait.upper() in eligible for trait in self.trait_names], dtype=bool)

    def call(self, inputs, training: bool = False):
        expert_indices, trait_index = inputs
        expert_available: list[tf.Tensor] = []
        main_latent: list[tf.Tensor | None] = []
        interaction_latent: list[tf.Tensor | None] = []
        for expert_index, factor in enumerate(self.factors):
            index = expert_indices[:, expert_index]
            available = index >= 0
            safe_index = tf.maximum(index, 0)
            gathered = tf.gather(factor, safe_index)
            expert_available.append(available)
            projection = self.main_projection[expert_index]
            main_latent.append(tf.matmul(gathered, projection) if projection is not None else None)
            interaction_projection = self.interaction_projection[expert_index]
            interaction_latent.append(
                tf.matmul(gathered, interaction_projection)
                if interaction_projection is not None
                else None
            )

        terms: list[tf.Tensor] = []
        available_terms: list[tf.Tensor] = []
        for expert_index in range(len(self.expert_specs)):
            projection = self.main_projection[expert_index]
            if projection is None:
                continue
            trait_effect = tf.gather(self.main_trait[expert_index], trait_index)
            terms.append(tf.reduce_sum(main_latent[expert_index] * trait_effect, axis=1))
            available_terms.append(expert_available[expert_index])

        for pair_index, (genotype_index, environment_index) in enumerate(self.interaction_pairs):
            trait_effect = tf.gather(self.interaction_trait[pair_index], trait_index)
            terms.append(
                tf.reduce_sum(
                    interaction_latent[genotype_index]
                    * interaction_latent[environment_index]
                    * trait_effect,
                    axis=1,
                )
            )
            available_terms.append(
                expert_available[genotype_index] & expert_available[environment_index]
            )

        term_values = tf.stack(terms, axis=1)
        row_available = tf.stack(available_terms, axis=1)
        trait_eligible = tf.gather(self.term_eligibility, trait_index)
        active = row_available & trait_eligible
        active_count = tf.reduce_sum(tf.cast(active, tf.float32), axis=1, keepdims=True)
        tf.debugging.assert_positive(active_count, message="An observation has no available kernel expert")
        if self.gate_logits is None:
            gates = tf.cast(active, tf.float32) / active_count
        else:
            logits = tf.gather(self.gate_logits, trait_index)
            masked_logits = tf.where(active, logits, tf.constant(-1e9, dtype=tf.float32))
            gates = tf.nn.softmax(masked_logits, axis=1)
        prediction = tf.reduce_sum(term_values * gates, axis=1)
        return tf.gather(self.intercept, trait_index) + prediction

    def regularization_loss(self) -> tf.Tensor:
        if self.weight_decay <= 0:
            return tf.constant(0.0, dtype=tf.float32)
        excluded = {id(self.intercept)}
        if self.gate_logits is not None:
            excluded.add(id(self.gate_logits))
        terms = [
            tf.reduce_sum(tf.square(variable))
            for variable in self.trainable_variables
            if id(variable) not in excluded
        ]
        return self.weight_decay * tf.add_n(terms)

    def kernel_gate_frame(self) -> pd.DataFrame:
        logits = (
            self.gate_logits.numpy()
            if self.gate_logits is not None
            else np.zeros((len(self.trait_names), len(self.term_names)), dtype=np.float32)
        )
        rows = []
        for trait_index, trait in enumerate(self.trait_names):
            eligible = self.term_eligibility_np[trait_index]
            local = np.zeros(len(self.term_names), dtype=np.float64)
            if self.gate_logits is None:
                local[eligible] = 1.0 / eligible.sum()
            else:
                shifted = logits[trait_index, eligible]
                shifted = shifted - shifted.max()
                probabilities = np.exp(shifted) / np.exp(shifted).sum()
                local[eligible] = probabilities
            for term_index, term in enumerate(self.term_names):
                rows.append(
                    {
                        "trait_name_canonical": trait,
                        "trait_index": trait_index,
                        "kernel_term": term,
                        "eligible": bool(eligible[term_index]),
                        "prior_gate": float(local[term_index]),
                    }
                )
        return pd.DataFrame(rows)


def add_expert_indices(
    ledger: pd.DataFrame, registry: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    ledger = ledger.copy()
    columns = []
    coverage_rows = []
    marker_columns = []
    for _, spec in registry.iterrows():
        name = str(spec["kernel"])
        axis = str(spec["axis"])
        order = pd.read_csv(Path(str(spec["order_path"])), sep="\t", dtype=str)
        id_col = str(spec["id_col"])
        compact = pd.to_numeric(order["compact_kernel_index"], errors="raise").astype(int)
        lookup = dict(zip(order[id_col].fillna("").astype(str), compact))
        ledger_id_col = "genotype_id" if axis == "genotype" else "environment_id"
        column = f"expert_index__{safe_name(name)}"
        ledger[column] = ledger[ledger_id_col].fillna("").astype(str).map(lookup).fillna(-1).astype(np.int32)
        columns.append(column)
        if name.startswith("K_G_"):
            marker_columns.append(column)
        eligible = trait_set(spec["eligible_traits"])
        for (split, trait), group in ledger.groupby(["split", "trait_name_canonical"], sort=True):
            trait_is_eligible = eligible is None or trait.upper() in eligible
            available = group[column].ge(0) & trait_is_eligible
            coverage_rows.append(
                {
                    "kernel": name,
                    "axis": axis,
                    "split": split,
                    "trait_name_canonical": trait,
                    "eligible_for_trait": trait_is_eligible,
                    "rows": len(group),
                    "available_rows": int(available.sum()),
                    "availability_fraction": float(available.mean()),
                }
            )
    ledger["has_marker_kernel"] = (
        ledger[marker_columns].ge(0).any(axis=1) if marker_columns else False
    )
    ledger["genomic_coverage_group"] = np.where(
        ledger["has_marker_kernel"], "marker_available", "pedigree_only"
    )
    return ledger, columns, pd.DataFrame(coverage_rows)


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
        dataset = dataset.shuffle(min(len(frame), 100_000), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def predict_scaled(
    model: tf.keras.Model, frame: pd.DataFrame, expert_columns: list[str], batch_size: int
) -> np.ndarray:
    dataset = make_dataset(frame, expert_columns, batch_size, False, 0)
    predictions = [model(inputs, training=False).numpy() for inputs, _, _ in dataset]
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)


def macro_standardized_rmse(frame: pd.DataFrame, predictions_scaled: np.ndarray) -> float:
    scores = []
    for _, group_index in frame.groupby("trait_index", sort=True).groups.items():
        positions = frame.index.get_indexer(group_index)
        values = regression_metrics(
            frame.loc[group_index, "y_scaled"].to_numpy(dtype=float),
            predictions_scaled[positions],
            frame.loc[group_index, "weight_g_e"].to_numpy(dtype=float),
        )
        scores.append(values["weighted_rmse"])
    return float(np.mean(scores)) if scores else float("inf")


def metric_rows(
    frame: pd.DataFrame, split: str, *, model_name: str, prediction_col: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    subsets = [("all", frame)] + [
        (name, group) for name, group in frame.groupby("genomic_coverage_group", sort=True)
    ]
    for coverage_group, subset in subsets:
        for trait, group in subset.groupby("trait_name_canonical", sort=True):
            values = regression_metrics(
                group["phenotype_value"].to_numpy(dtype=float),
                group[prediction_col].to_numpy(dtype=float),
                group["weight_g_e"].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "split": split,
                    "coverage_group": coverage_group,
                    "model": model_name,
                    "trait_name_canonical": trait,
                    **values,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a masked, trait-gated, multi-trait multi-kernel TensorFlow model."
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--trait-order", type=Path, required=True)
    parser.add_argument("--kernel-registry", type=Path, required=True)
    parser.add_argument("--certification-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="multitrait_kernel_experts")
    parser.add_argument("--model-label", default="multitrait_kernel_experts")
    parser.add_argument("--trait", action="append")
    parser.add_argument("--exclude-kernel", action="append", default=[])
    parser.add_argument("--include-disabled-kernel", action="append", default=[])
    parser.add_argument("--split", default="gho_environment")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-train-rows-per-trait", type=int, default=100)
    parser.add_argument("--min-eval-rows-per-trait", type=int, default=20)
    parser.add_argument("--max-rank-genotype", type=int, default=128)
    parser.add_argument("--max-rank-environment", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument(
        "--factorization-mode",
        choices=["full_transductive", "train_nystrom"],
        default="train_nystrom",
    )
    parser.add_argument("--factor-cache", type=Path)
    parser.add_argument("--no-center-kernels", action="store_true")
    parser.add_argument("--no-genotype-main", action="store_true")
    parser.add_argument("--no-environment-main", action="store_true")
    parser.add_argument("--no-interaction", action="store_true")
    parser.add_argument("--fixed-kernel-gates", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--intra-op-threads", type=int, default=16)
    parser.add_argument("--inter-op-threads", type=int, default=2)
    args = parser.parse_args()

    certification = json.loads(args.certification_summary.read_text(encoding="utf-8"))
    if certification.get("status") != "PASS":
        raise SystemExit(f"Kernel certification is not PASS: {args.certification_summary}")
    require_certified_file(args.ledger, certification.get("ledger_identity", {}), "Ledger")
    require_certified_file(
        args.kernel_registry, certification.get("registry_identity", {}), "Kernel registry"
    )
    registry = pd.read_csv(args.kernel_registry, sep="\t")
    enabled = registry["enabled_default"].map(parse_bool)
    if args.include_disabled_kernel:
        enabled |= registry["kernel"].isin(args.include_disabled_kernel)
    enabled &= ~registry["kernel"].isin(args.exclude_kernel)
    registry = registry[enabled].copy().reset_index(drop=True)
    if registry.empty:
        raise SystemExit("No kernel experts remain after registry filtering")
    certified_kernels = certification.get("kernel_identities", {})
    certified_orders = certification.get("order_identities", {})
    for _, spec in registry.iterrows():
        name = str(spec["kernel"])
        require_certified_file(Path(str(spec["kernel_path"])), certified_kernels.get(name, {}), name)
        require_certified_file(
            Path(str(spec["order_path"])), certified_orders.get(name, {}), f"{name} order"
        )

    if args.no_genotype_main and args.no_environment_main and args.no_interaction:
        raise SystemExit("At least one main effect or interaction must remain active")
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
    if ledger.empty:
        raise SystemExit("No observations remain after trait filtering")

    canonical_split = canonical_split_mode(args.split, warn=True)
    group_col = split_group_column(canonical_split)
    train_index, val_index, test_index = make_split(
        ledger, canonical_split, args.seed, args.test_fraction, args.val_fraction, group_col
    )
    split_labels = np.full(len(ledger), "", dtype=object)
    split_labels[train_index] = "train"
    split_labels[val_index] = "val"
    split_labels[test_index] = "test"
    ledger["split"] = split_labels
    leakage = split_leakage_record(
        ledger, args.seed, canonical_split, train_index, val_index, test_index, group_col
    )
    if leakage["leakage_status"] != "pass":
        raise SystemExit(f"Split leakage detected: {leakage}")

    support = ledger.groupby(["trait_name_canonical", "split"]).size().unstack(fill_value=0)
    for column in ["train", "val", "test"]:
        if column not in support:
            support[column] = 0
    retained = support[
        support["train"].ge(args.min_train_rows_per_trait)
        & support["val"].ge(args.min_eval_rows_per_trait)
        & support["test"].ge(args.min_eval_rows_per_trait)
    ].index.tolist()
    ledger = ledger[ledger["trait_name_canonical"].isin(retained)].copy().reset_index(drop=True)
    if len(retained) < 2:
        raise SystemExit(f"Multi-trait training requires at least two supported traits; found {retained}")

    retained_order = trait_order[trait_order["trait_name_canonical"].isin(retained)].copy()
    retained_order = retained_order.sort_values("trait_index").reset_index(drop=True)
    retained_order["source_trait_index"] = retained_order["trait_index"]
    retained_order["trait_index"] = np.arange(len(retained_order), dtype=np.int32)
    trait_map = dict(zip(retained_order["trait_name_canonical"], retained_order["trait_index"]))
    ledger["trait_index"] = ledger["trait_name_canonical"].map(trait_map).astype(np.int32)
    retained_trait_names = retained_order["trait_name_canonical"].tolist()
    retained_upper = {value.upper() for value in retained_trait_names}
    registry = registry[
        registry["eligible_traits"].map(
            lambda value: trait_set(value) is None or bool(trait_set(value) & retained_upper)
        )
    ].copy().reset_index(drop=True)
    ledger, expert_columns, coverage = add_expert_indices(ledger, registry)

    scaling_rows = []
    ledger["y_scaled"] = np.nan
    for trait, group in ledger.groupby("trait_name_canonical", sort=True):
        local_train = group[group["split"].eq("train")]
        mean, sd = weighted_mean_std(
            local_train["phenotype_value"].to_numpy(dtype=float),
            local_train["weight_g_e"].to_numpy(dtype=float),
        )
        ledger.loc[group.index, "y_scaled"] = (group["phenotype_value"] - mean) / sd
        scaling_rows.append({"trait_name_canonical": trait, "train_mean": mean, "train_sd": sd})
    scaling = pd.DataFrame(scaling_rows)
    scale_map = scaling.set_index("trait_name_canonical").to_dict("index")

    train = ledger[ledger["split"].eq("train")].copy()
    val = ledger[ledger["split"].eq("val")].copy()
    test = ledger[ledger["split"].eq("test")].copy()
    train["loss_weight"] = trait_balanced_loss_weights(
        train["trait_index"].to_numpy(), train["weight_g_e"].to_numpy()
    )
    val["loss_weight"] = 1.0
    test["loss_weight"] = 1.0

    effective_mode = effective_factorization_mode(
        args.factorization_mode, canonical_split, warn=True
    )
    centered = not args.no_center_kernels
    factor_configurations = []
    train_ids_by_expert = []
    for expert_index, (_, spec) in enumerate(registry.iterrows()):
        column = expert_columns[expert_index]
        eligible = trait_set(spec["eligible_traits"])
        local = train if eligible is None else train[
            train["trait_name_canonical"].str.upper().isin(eligible)
        ]
        train_ids = local.loc[local[column].ge(0), column].unique().astype(np.int32)
        if not len(train_ids):
            raise SystemExit(f"{spec['kernel']} has no eligible training IDs")
        if effective_mode != "train_nystrom":
            train_ids = None
        train_ids_by_expert.append(train_ids)
        max_rank = (
            args.max_rank_genotype if spec["axis"] == "genotype" else args.max_rank_environment
        )
        rank = min(int(spec["rank"]), max_rank)
        factor_configurations.append(
            {
                "kernel": str(spec["kernel"]),
                "identity": file_identity(Path(str(spec["kernel_path"]))),
                "rank": rank,
                "train_index_digest": index_digest(train_ids),
            }
        )
    cache_configuration = {
        "experts": factor_configurations,
        "effective_factorization_mode": effective_mode,
        "kernel_centered": centered,
    }
    cache_metadata_path = args.factor_cache.with_suffix(".json") if args.factor_cache else None
    cache_loaded = False
    factors: list[np.ndarray] = []
    factor_metadata: dict[str, object] = {}
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
            print(f"Loaded kernel factors from {args.factor_cache}", flush=True)
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

    expert_specs = registry.to_dict("records")
    model = MultiTraitKernelExperts(
        expert_specs,
        factors,
        trait_names=retained_trait_names,
        latent_dim=args.latent_dim,
        include_genotype_main=not args.no_genotype_main,
        include_environment_main=not args.no_environment_main,
        include_interaction=not args.no_interaction,
        learn_kernel_gates=not args.fixed_kernel_gates,
        weight_decay=args.weight_decay,
        initialization_seed=args.seed,
    )
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    train_dataset = make_dataset(train, expert_columns, args.batch_size, True, args.seed)

    @tf.function
    def train_step(inputs, target, weight):
        with tf.GradientTape() as tape:
            prediction = model(inputs, training=True)
            loss = tf.reduce_sum(weight * tf.square(prediction - target)) / tf.reduce_sum(weight)
            loss = loss + model.regularization_loss()
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
        val_prediction_scaled = predict_scaled(model, val, expert_columns, args.batch_size)
        val_score = macro_standardized_rmse(val.reset_index(drop=True), val_prediction_scaled)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_macro_standardized_rmse": val_score,
            }
        )
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(history_rows[-1]), flush=True)
        if val_score < best_score:
            best_score = val_score
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
    for split_name, frame in [("val", val), ("test", test)]:
        frame = frame.copy().reset_index(drop=True)
        frame["y_pred_scaled"] = predict_scaled(model, frame, expert_columns, args.batch_size)
        frame["y_pred"] = [
            value * scale_map[trait]["train_sd"] + scale_map[trait]["train_mean"]
            for value, trait in zip(frame["y_pred_scaled"], frame["trait_name_canonical"])
        ]
        frame["y_pred_train_mean"] = frame["trait_name_canonical"].map(
            {trait: values["train_mean"] for trait, values in scale_map.items()}
        )
        metric_output.extend(
            metric_rows(frame, split_name, model_name=args.model_label, prediction_col="y_pred")
        )
        metric_output.extend(
            metric_rows(
                frame, split_name, model_name="train_mean", prediction_col="y_pred_train_mean"
            )
        )
        prediction_outputs.append(frame)

    metrics_frame = pd.DataFrame(metric_output)
    all_metrics = metrics_frame[metrics_frame["coverage_group"].eq("all")]
    macro = (
        all_metrics.groupby(["split", "model"])[
            ["weighted_rmse", "unweighted_rmse", "normalized_rmse", "pearson", "prediction_sd_ratio"]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                column: f"macro_{column}"
                for column in [
                    "weighted_rmse",
                    "unweighted_rmse",
                    "normalized_rmse",
                    "pearson",
                    "prediction_sd_ratio",
                ]
            }
        )
    )
    model_metrics = metrics_frame[metrics_frame["model"].eq(args.model_label)]
    baseline_metrics = metrics_frame[metrics_frame["model"].eq("train_mean")]
    improvement = model_metrics.merge(
        baseline_metrics,
        on=["split", "coverage_group", "trait_name_canonical"],
        suffixes=("_model", "_train_mean"),
        validate="one_to_one",
    )
    improvement["weighted_rmse_improvement"] = (
        improvement["weighted_rmse_train_mean"] - improvement["weighted_rmse_model"]
    )
    improvement["unweighted_rmse_improvement"] = (
        improvement["unweighted_rmse_train_mean"] - improvement["unweighted_rmse_model"]
    )
    improvement["normalized_rmse_improvement"] = (
        improvement["normalized_rmse_train_mean"] - improvement["normalized_rmse_model"]
    )
    predictions = pd.concat(prediction_outputs, ignore_index=True)
    history = pd.DataFrame(history_rows)
    gate_frame = model.kernel_gate_frame()

    metrics_frame.to_csv(args.out_dir / f"{args.prefix}_trait_metrics.tsv", sep="\t", index=False)
    macro.to_csv(args.out_dir / f"{args.prefix}_macro_metrics.tsv", sep="\t", index=False)
    improvement.to_csv(args.out_dir / f"{args.prefix}_vs_train_mean.tsv", sep="\t", index=False)
    history.to_csv(args.out_dir / f"{args.prefix}_history.tsv", sep="\t", index=False)
    scaling.to_csv(args.out_dir / f"{args.prefix}_trait_scaling.tsv", sep="\t", index=False)
    retained_order.to_csv(args.out_dir / f"{args.prefix}_trait_order.tsv", sep="\t", index=False)
    gate_frame.to_csv(args.out_dir / f"{args.prefix}_kernel_gates.tsv", sep="\t", index=False)
    coverage.to_csv(args.out_dir / f"{args.prefix}_kernel_coverage.tsv", sep="\t", index=False)
    registry.to_csv(args.out_dir / f"{args.prefix}_active_kernel_registry.tsv", sep="\t", index=False)
    pd.DataFrame([leakage]).to_csv(
        args.out_dir / f"{args.prefix}_split_leakage_qc.tsv", sep="\t", index=False
    )
    prediction_export = predictions.drop(columns=expert_columns, errors="ignore")
    write_table(prediction_export, args.out_dir / f"{args.prefix}_predictions.parquet")
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_path = checkpoint.save(str(args.out_dir / f"{args.prefix}_ckpt"))
    run_metadata = {
        "tensorflow_version": tf.__version__,
        "seed": args.seed,
        "canonical_split_mode": canonical_split,
        "requested_factorization_mode": args.factorization_mode,
        "effective_factorization_mode": effective_mode,
        "kernel_centered": centered,
        "include_interaction": not args.no_interaction,
        "include_genotype_main": not args.no_genotype_main,
        "include_environment_main": not args.no_environment_main,
        "model_label": args.model_label,
        "learn_kernel_gates": not args.fixed_kernel_gates,
        "parameter_initialization": {
            "distribution": "RandomNormal",
            "stddev": 0.02,
            "seed_policy": "run_seed_plus_variable_index",
            "run_seed": args.seed,
        },
        "traits": retained_trait_names,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "active_kernels": registry["kernel"].tolist(),
        "factorizations": factor_metadata,
        "best_val_macro_standardized_rmse": best_score,
        "checkpoint": checkpoint_path,
        "factor_cache": str(args.factor_cache) if args.factor_cache else "",
        "factor_cache_loaded": cache_loaded,
    }
    (args.out_dir / f"{args.prefix}_run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    print(metrics_frame.to_string(index=False), flush=True)
    print(macro.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
