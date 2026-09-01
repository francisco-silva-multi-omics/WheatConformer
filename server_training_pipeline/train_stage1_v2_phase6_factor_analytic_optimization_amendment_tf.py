from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
import tensorflow as tf

from . import train_stage1_v2_phase6_factor_analytic_tf as legacy
from .stage1_v2_phase6_remediation import add_hierarchy_indices
from .train_stage1_v2_phase6_confirmation_tf import build_confirmation_factors
from .train_stage1_v2_phase6_remediation_tf import HierarchicalReactionNorm
from .train_stage1_v2_phase6_tf import (
    add_factor_indices,
    build_projection_environment,
    macro_nrmse,
    sha256_file,
    stable_seed,
)


PROTOCOL = Path(
    "server_training_pipeline/"
    "stage1_v2_phase6_factor_analytic_optimization_amendment_protocol_v1.json"
)
TRAINER = Path(
    "server_training_pipeline/"
    "train_stage1_v2_phase6_factor_analytic_optimization_amendment_tf.py"
)
PARENT_DECISION = Path(
    "model_kernels/stage1_v2_phase6_factor_analytic_screen_v1/phase_1/"
    "FACTOR_ANALYTIC_PHASE1_DECISION.json"
)
RUN_PROTOCOL = (
    "stage1_v2_phase6_normalized_direction_factor_analytic_optimization_tf_v1"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path) -> dict[str, Any]:
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    protocol = read_json(code_root / PROTOCOL)
    if protocol.get("protocol_version") != (
        "stage1_v2_phase6_factor_analytic_optimization_amendment_v1"
    ):
        raise ValueError("Unexpected FA optimization-amendment protocol")
    objective = protocol["objective_policy"]
    reporting = set(objective["training_likelihood_traits"])
    primary = set(objective["primary_macro_traits"])
    demoted = set(objective["demoted_from_primary_macro"])
    if len(reporting) != 7 or len(primary) != 6:
        raise ValueError("The amendment must retain seven traits and use a six-trait macro")
    if demoted != {"TEST_WEIGHT"} or reporting - primary != demoted:
        raise ValueError("Only TEST_WEIGHT may be demoted from the primary macro")
    if not all(
        bool(objective[key])
        for key in (
            "all_seven_traits_retained_in_training_rows",
            "test_weight_predictions_retained",
            "test_weight_trait_reporting_retained",
            "test_weight_training_only_huber_calibration_retained",
            "test_weight_exploratory_non_deterioration_guard_retained",
        )
    ):
        raise ValueError("TEST_WEIGHT retention contract is incomplete")
    architecture = protocol["architecture_policy"]
    if architecture.get("only_mutable_component") != (
        "covariate_linked_factor_analytic_optimization"
    ):
        raise ValueError("The amendment changes an unauthorized component")
    if architecture.get("free_environment_loadings_allowed") is not False:
        raise ValueError("Free environment loadings are forbidden")
    if int(protocol["projection_feature_contract"]["expected_feature_count"]) != 153:
        raise ValueError("The amendment is not bound to the 153-feature schema")
    ranks = {
        int(value["factor_analytic_rank"])
        for value in protocol["candidates"].values()
        if value.get("source_reuse") is False
    }
    if ranks != {2, 4}:
        raise ValueError(f"Unexpected bounded FA ranks: {sorted(ranks)}")
    parent = read_json(root / PARENT_DECISION)
    if parent.get("status") != protocol["parent_factor_analytic_status"]:
        raise ValueError("The terminal V1 FA decision is not the frozen parent")
    return protocol


class NormalizedDirectionFactorAnalyticHierarchicalReactionNorm(
    legacy.CovariateLinkedFactorAnalyticHierarchicalReactionNorm
):
    """Use normalized G/E directions and shrink only trait amplitudes."""

    def __init__(
        self,
        *args,
        trait_amplitude_penalty_multiplier: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, factor_penalty_multiplier=1.0, **kwargs)
        self.trait_amplitude_penalty_multiplier = float(
            trait_amplitude_penalty_multiplier
        )

    @staticmethod
    def _column_directions(values: tf.Tensor) -> tf.Tensor:
        norms = tf.norm(values, axis=0, keepdims=True)
        return values / tf.maximum(norms, tf.constant(1e-12, values.dtype))

    def factor_analytic_residual(self, inputs) -> tf.Tensor:
        genotype_indices = inputs[0]
        trait_index = inputs[3]
        projection_index = inputs[6]
        genotype_index = genotype_indices[:, 0]
        safe_genotype = tf.maximum(genotype_index, 0)
        safe_projection = tf.maximum(projection_index, 0)
        active = (
            (genotype_index >= 0)
            & (projection_index >= 0)
            & tf.gather(self.projection_available, safe_projection)
        )
        genotype_direction = self._column_directions(
            self.fa_genotype_coefficients
        )
        environment_direction = self._column_directions(
            self.fa_environment_coefficients
        )
        genotype_score = tf.matmul(
            tf.gather(self.genotype_factors[0], safe_genotype),
            genotype_direction,
        )
        environment_loading = tf.matmul(
            tf.gather(self.projection_design, safe_projection),
            environment_direction,
        )
        trait_amplitude = tf.gather(
            tf.transpose(self.fa_trait_loadings), trait_index
        )
        residual = tf.reduce_sum(
            genotype_score * environment_loading * trait_amplitude,
            axis=1,
        ) / math.sqrt(float(self.factor_analytic_rank))
        return residual * tf.cast(active, tf.float32)

    def call(self, inputs, training: bool = False):
        if len(inputs) != 7:
            raise ValueError("Normalized-direction FA requires seven input tensors")
        base = HierarchicalReactionNorm.call(self, inputs[:6], training=training)
        return base + self.factor_analytic_residual(inputs)

    def regularization_loss(self) -> tf.Tensor:
        value = HierarchicalReactionNorm.regularization_loss(self)
        direction_penalty = self.weight_decay * (
            tf.reduce_sum(tf.square(self.fa_genotype_coefficients))
            + tf.reduce_sum(tf.square(self.fa_environment_coefficients))
        )
        value -= direction_penalty
        if self.trait_amplitude_penalty_multiplier != 1.0:
            value += self.weight_decay * (
                self.trait_amplitude_penalty_multiplier - 1.0
            ) * tf.reduce_sum(tf.square(self.fa_trait_loadings))
        return value

    def activity_values(self) -> dict[str, np.ndarray]:
        genotype_raw = self.fa_genotype_coefficients.numpy()
        environment_raw = self.fa_environment_coefficients.numpy()
        trait = self.fa_trait_loadings.numpy()
        genotype_norm = np.linalg.norm(genotype_raw, axis=0)
        environment_norm = np.linalg.norm(environment_raw, axis=0)
        trait_norm = np.linalg.norm(trait, axis=1)
        genotype_normalized = genotype_raw / np.maximum(genotype_norm, 1e-12)
        environment_normalized = environment_raw / np.maximum(
            environment_norm, 1e-12
        )
        return {
            "genotype_raw_norm": genotype_norm,
            "environment_raw_norm": environment_norm,
            "trait_amplitude_norm": trait_norm,
            "genotype_normalized_norm": np.linalg.norm(
                genotype_normalized, axis=0
            ),
            "environment_normalized_norm": np.linalg.norm(
                environment_normalized, axis=0
            ),
        }


def primary_macro_nrmse(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    primary_macro_traits: Sequence[str],
) -> float:
    keep = frame["trait"].isin(primary_macro_traits).to_numpy()
    if not keep.any():
        return float("inf")
    local = frame.loc[keep].reset_index(drop=True)
    return macro_nrmse(local, np.asarray(prediction)[keep])


def predict_residual(
    model: NormalizedDirectionFactorAnalyticHierarchicalReactionNorm,
    frame: pd.DataFrame,
    genotype_columns: Sequence[str],
    environment_columns: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    dataset = legacy.make_dataset(
        frame,
        genotype_columns,
        environment_columns,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    values = [model.factor_analytic_residual(inputs) for inputs, _, _ in dataset]
    result = (
        tf.concat(values, axis=0).numpy()
        if values
        else np.empty(0, dtype=np.float32)
    )
    tf.debugging.assert_all_finite(result, "Non-finite normalized-direction residual")
    return result


def _activity_row(
    model: NormalizedDirectionFactorAnalyticHierarchicalReactionNorm,
    *,
    epoch: int,
    record_type: str,
    mean_loss: float,
    gradient_norms: np.ndarray,
    training_residual_rms: float,
    validation_residual_rms: float | None,
    validation_primary_nrmse: float | None,
) -> dict[str, object]:
    values = model.activity_values()
    return {
        "epoch": epoch,
        "record_type": record_type,
        "train_gaussian_nll_regularized": mean_loss,
        "validation_primary6_macro_normalized_rmse": validation_primary_nrmse,
        "training_FA_residual_rms": training_residual_rms,
        "validation_FA_residual_rms": validation_residual_rms,
        "genotype_gradient_norm": float(gradient_norms[0]),
        "environment_gradient_norm": float(gradient_norms[1]),
        "trait_amplitude_gradient_norm": float(gradient_norms[2]),
        "genotype_raw_direction_norm_min": float(
            values["genotype_raw_norm"].min()
        ),
        "genotype_raw_direction_norm_max": float(
            values["genotype_raw_norm"].max()
        ),
        "environment_raw_direction_norm_min": float(
            values["environment_raw_norm"].min()
        ),
        "environment_raw_direction_norm_max": float(
            values["environment_raw_norm"].max()
        ),
        "trait_amplitude_norm_min": float(values["trait_amplitude_norm"].min()),
        "trait_amplitude_norm_max": float(values["trait_amplitude_norm"].max()),
        "genotype_normalized_norm_max_error": float(
            np.max(np.abs(values["genotype_normalized_norm"] - 1.0))
        ),
        "environment_normalized_norm_max_error": float(
            np.max(np.abs(values["environment_normalized_norm"] - 1.0))
        ),
    }


def fit_normalized_direction_component(
    *,
    root: Path,
    state_id: str,
    configuration: dict[str, Any],
    frame: pd.DataFrame,
    trait_names: Sequence[str],
    seed: int,
    protocol: dict[str, Any],
    candidate_contract: dict[str, Any],
    replay_artifact_dir: Path,
) -> legacy.FactorAnalyticFit:
    (
        genotype,
        environment,
        reaction_design,
        reaction_available,
        environment_ids,
        reaction_enabled,
    ) = build_confirmation_factors(
        root,
        state_id,
        "historical_reaction_reference",
        configuration,
    )
    _, projection_design, projection_available, projection_ids = (
        build_projection_environment(root, state_id)
    )
    if projection_design.shape[1] != 153:
        raise ValueError("Split-bound projection feature count is not 153")
    if not np.array_equal(environment_ids.astype(str), projection_ids.astype(str)):
        raise ValueError("Historical and projection environment axes disagree")
    for block in environment:
        if not np.array_equal(block.entity_ids.astype(str), projection_ids.astype(str)):
            raise ValueError(f"Environment block axis differs from projection: {block.name}")

    local = frame.copy()
    genotype_columns, environment_columns = add_factor_indices(
        local, genotype, environment, environment_ids, reaction_available
    )
    projection_lookup = {
        value: index for index, value in enumerate(projection_ids.astype(str))
    }
    local["fa_environment_index"] = (
        local["environment_id"]
        .astype(str)
        .map(projection_lookup)
        .fillna(-1)
        .astype(np.int32)
    )
    local, trial_support, environment_support, hierarchy_support = (
        add_hierarchy_indices(local, trait_names, protocol)
    )
    training = local.loc[local["selection_role"].eq("TRAINING")].reset_index(
        drop=True
    )
    validation = local.loc[
        local["selection_role"].eq("INNER_VALIDATION")
    ].reset_index(drop=True)
    if training.empty or validation.empty:
        raise ValueError("FA amendment state lacks training or validation rows")
    expected_traits = set(protocol["objective_policy"]["training_likelihood_traits"])
    if set(training["trait"].astype(str)) != expected_traits:
        raise ValueError("FA amendment training likelihood lost a retained trait")
    if set(validation["trait"].astype(str)) != expected_traits:
        raise ValueError("FA amendment validation reporting lost a retained trait")

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    floors, multipliers = legacy.trait_regularization_vectors(protocol, trait_names)
    hierarchy = protocol["trial_environment_hierarchy"]
    model = NormalizedDirectionFactorAnalyticHierarchicalReactionNorm(
        genotype=genotype,
        environment=environment,
        reaction_design=reaction_design,
        trait_names=trait_names,
        latent_dim=int(configuration["latent_dim"]),
        reaction_rank=int(configuration["reaction_rank"]),
        residual_floor=0.05,
        weight_decay=float(configuration["weight_decay"]),
        seed=seed,
        reaction_enabled=reaction_enabled,
        trait_residual_floors=floors,
        trait_penalty_multipliers=multipliers,
        trial_support=trial_support,
        environment_support=environment_support,
        trial_penalty=float(hierarchy["trial_effect_penalty"]),
        environment_penalty=float(hierarchy["environment_effect_penalty"]),
        projection_design=projection_design,
        projection_available=projection_available,
        factor_analytic_rank=int(candidate_contract["factor_analytic_rank"]),
        trait_amplitude_penalty_multiplier=float(
            candidate_contract["trait_amplitude_penalty_multiplier"]
        ),
    )
    optimizer = tf.keras.optimizers.Adam(float(configuration["learning_rate"]))
    batch_size = int(configuration["batch_size"])
    training_dataset = legacy.make_dataset(
        training,
        genotype_columns,
        environment_columns,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    fa_names = [variable.name for variable in model.factor_analytic_variables]

    @tf.function
    def train_step(inputs, target, weight):
        with tf.GradientTape() as tape:
            prediction = model(inputs, training=True)
            residual = model.factor_analytic_residual(inputs)
            tf.debugging.assert_all_finite(prediction, "Non-finite amended prediction")
            tf.debugging.assert_all_finite(residual, "Non-finite amended FA residual")
            tf.debugging.assert_all_finite(target, "Non-finite amended target")
            tf.debugging.assert_all_finite(weight, "Non-finite amended weight")
            trait_index = inputs[3]
            scale = tf.gather(model.residual_scales(), trait_index)
            tf.debugging.assert_all_finite(scale, "Non-finite amended residual scale")
            nll = 0.5 * tf.square((target - prediction) / scale) + tf.math.log(scale)
            tf.debugging.assert_all_finite(nll, "Non-finite amended per-row NLL")
            denominator = tf.maximum(tf.reduce_sum(weight), 1e-6)
            regularization = model.regularization_loss()
            tf.debugging.assert_all_finite(
                regularization, "Non-finite amended regularization"
            )
            loss = tf.reduce_sum(nll * weight) / denominator + regularization
            tf.debugging.assert_all_finite(loss, "Non-finite amended scalar loss")
        gradients = tape.gradient(loss, model.trainable_variables)
        checked = []
        fa_gradient_norms: dict[str, tf.Tensor] = {}
        for gradient, variable in zip(gradients, model.trainable_variables):
            if gradient is None:
                raise RuntimeError(f"Missing amended gradient for {variable.name}")
            values = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
            tf.debugging.assert_all_finite(
                values, f"Non-finite amended gradient for {variable.name}"
            )
            checked.append(gradient)
            if variable.name in fa_names:
                fa_gradient_norms[variable.name] = tf.linalg.global_norm([values])
        if set(fa_gradient_norms) != set(fa_names):
            raise RuntimeError("Not every amended FA tensor received a gradient")
        optimizer.apply_gradients(zip(checked, model.trainable_variables))
        return (
            loss,
            tf.reduce_sum(tf.square(residual)),
            tf.cast(tf.size(residual), tf.float32),
            tf.stack([fa_gradient_norms[name] for name in fa_names]),
        )

    best_metric = float("inf")
    best_weights: list[np.ndarray] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    primary_traits = protocol["objective_policy"]["primary_macro_traits"]
    initial_training_residual = predict_residual(
        model,
        training,
        genotype_columns,
        environment_columns,
        batch_size,
    )
    initial_validation_residual = predict_residual(
        model,
        validation,
        genotype_columns,
        environment_columns,
        batch_size,
    )
    history.append(
        _activity_row(
            model,
            epoch=0,
            record_type="initialization",
            mean_loss=float("nan"),
            gradient_norms=np.zeros(3, dtype=float),
            training_residual_rms=float(
                np.sqrt(np.mean(np.square(initial_training_residual)))
            ),
            validation_residual_rms=float(
                np.sqrt(np.mean(np.square(initial_validation_residual)))
            ),
            validation_primary_nrmse=None,
        )
    )
    for epoch in range(1, int(configuration["epochs_max"]) + 1):
        outputs = [
            train_step(inputs, target, weight)
            for inputs, target, weight in training_dataset
        ]
        mean_loss = float(tf.reduce_mean(tf.stack([item[0] for item in outputs])).numpy())
        residual_ss = float(tf.add_n([item[1] for item in outputs]).numpy())
        residual_n = float(tf.add_n([item[2] for item in outputs]).numpy())
        gradient_norms = np.mean(
            np.stack([item[3].numpy() for item in outputs]), axis=0
        )
        training_residual_rms = math.sqrt(residual_ss / max(residual_n, 1.0))
        evaluate = epoch == 1 or epoch % 5 == 0
        validation_metric = None
        validation_residual_rms = None
        if evaluate:
            validation_prediction = legacy.predict(
                model,
                validation,
                genotype_columns,
                environment_columns,
                batch_size,
            )
            validation_metric = primary_macro_nrmse(
                validation, validation_prediction, primary_traits
            )
            validation_residual = predict_residual(
                model,
                validation,
                genotype_columns,
                environment_columns,
                batch_size,
            )
            validation_residual_rms = float(
                np.sqrt(np.mean(np.square(validation_residual)))
            )
        row = _activity_row(
            model,
            epoch=epoch,
            record_type="training_epoch",
            mean_loss=mean_loss,
            gradient_norms=gradient_norms,
            training_residual_rms=training_residual_rms,
            validation_residual_rms=validation_residual_rms,
            validation_primary_nrmse=validation_metric,
        )
        history.append(row)
        if evaluate:
            print(json.dumps(row), flush=True)
            assert validation_metric is not None
            if validation_metric < best_metric - 1e-7:
                best_metric = validation_metric
                best_weights = model.get_weights()
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 5 if epoch > 1 else 1
            if epochs_without_improvement >= int(
                configuration["early_stopping_patience"]
            ):
                break
    if best_weights is None:
        raise RuntimeError("FA optimization amendment produced no checkpoint")
    epochs_completed = int(history[-1]["epoch"])
    model.set_weights(best_weights)
    training_prediction = legacy.predict(
        model, training, genotype_columns, environment_columns, batch_size
    )
    validation_prediction = legacy.predict(
        model, validation, genotype_columns, environment_columns, batch_size
    )
    training_residual = predict_residual(
        model, training, genotype_columns, environment_columns, batch_size
    )
    validation_residual = predict_residual(
        model, validation, genotype_columns, environment_columns, batch_size
    )
    selected_row = _activity_row(
        model,
        epoch=best_epoch,
        record_type="selected_checkpoint",
        mean_loss=float("nan"),
        gradient_norms=np.max(
            np.asarray(
                [
                    [
                        row["genotype_gradient_norm"],
                        row["environment_gradient_norm"],
                        row["trait_amplitude_gradient_norm"],
                    ]
                    for row in history
                ]
            ),
            axis=0,
        ),
        training_residual_rms=float(np.sqrt(np.mean(np.square(training_residual)))),
        validation_residual_rms=float(
            np.sqrt(np.mean(np.square(validation_residual)))
        ),
        validation_primary_nrmse=best_metric,
    )
    history.append(selected_row)
    activity = pd.DataFrame(history)
    activity_path = replay_artifact_dir / "component_activity_history.tsv"
    activity.to_csv(activity_path, sep="\t", index=False, lineterminator="\n")
    replay_artifacts = legacy.persist_component_replay_artifacts(
        replay_artifact_dir,
        model,
        training["phase4_adjusted_row_id"].astype(str).to_numpy(),
        validation["phase4_adjusted_row_id"].astype(str).to_numpy(),
        training_prediction,
        validation_prediction,
    )
    replay_artifacts[activity_path.name] = sha256_file(activity_path)
    inventory = pd.DataFrame(
        [
            {
                "component_model": "historical_plus_normalized_direction_FA",
                "component": block.name,
                "axis": block.axis,
                "entities": len(block.entity_ids),
                "available_entities": int(block.available.sum()),
                "rank": int(block.values.shape[1]),
                "state_hash": block.state_hash,
            }
            for block in (*genotype, *environment)
        ]
        + [
            {
                "component_model": "historical_plus_normalized_direction_FA",
                "component": "E_PROJECTION_CORE_V1_STANDARDIZED_FEATURES",
                "axis": "environment",
                "entities": len(projection_ids),
                "available_entities": int(projection_available.sum()),
                "rank": int(projection_design.shape[1]),
                "state_hash": hashlib.sha256(
                    np.asarray(projection_design, dtype="<f4").tobytes(order="C")
                ).hexdigest(),
            }
        ]
    )
    active_training = training[genotype_columns[0]].ge(0).to_numpy()
    active_training &= training["fa_environment_index"].ge(0).to_numpy()
    active_training &= projection_available[
        np.maximum(training["fa_environment_index"].to_numpy(dtype=int), 0)
    ]
    active_validation = validation[genotype_columns[0]].ge(0).to_numpy()
    active_validation &= validation["fa_environment_index"].ge(0).to_numpy()
    active_validation &= projection_available[
        np.maximum(validation["fa_environment_index"].to_numpy(dtype=int), 0)
    ]
    fa_variables = list(model.factor_analytic_variables)
    result = legacy.FactorAnalyticFit(
        training_prediction=training_prediction,
        validation_prediction=validation_prediction,
        best_metric=best_metric,
        epochs_completed=epochs_completed,
        epoch_history=activity,
        factor_inventory=inventory,
        hierarchy_support=hierarchy_support,
        fa_rank=int(candidate_contract["factor_analytic_rank"]),
        fa_variable_count=len(fa_variables),
        fa_parameter_count=int(
            sum(np.prod(variable.shape.as_list()) for variable in fa_variables)
        ),
        fa_active_training_rows=int(active_training.sum()),
        fa_active_validation_rows=int(active_validation.sum()),
        replay_artifacts=replay_artifacts,
    )
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def _primary_summary(
    trait_metrics: pd.DataFrame, primary_traits: Sequence[str]
) -> dict[str, float]:
    local = trait_metrics.loc[
        trait_metrics["trait_name_canonical"].isin(primary_traits)
    ]
    if len(local) != len(primary_traits):
        raise ValueError("Primary six-trait metric is incomplete")
    return {
        "validation_macro_normalized_rmse": float(local["normalized_rmse"].mean()),
        "validation_macro_pearson": float(local["pearson"].mean()),
        "validation_macro_calibration_error": float(
            local["calibration_error"].mean()
        ),
    }


def _activity_summary(
    activity: pd.DataFrame, protocol: dict[str, Any]
) -> dict[str, object]:
    thresholds = protocol["activity_certification"]
    epochs = activity.loc[activity["record_type"].eq("training_epoch")]
    initialization = activity.loc[
        activity["record_type"].eq("initialization")
    ].iloc[-1]
    selected = activity.loc[
        activity["record_type"].eq("selected_checkpoint")
    ].iloc[-1]
    direction_floor = float(thresholds["minimum_raw_direction_column_norm"])
    norm_tolerance = float(thresholds["maximum_normalized_direction_norm_error"])
    gradient_floor = float(
        thresholds["minimum_observed_gradient_norm_per_FA_tensor"]
    )
    optimization_path = bool(
        initialization["training_FA_residual_rms"]
        >= float(thresholds["minimum_initial_training_residual_rms"])
        and epochs["genotype_gradient_norm"].max() >= gradient_floor
        and epochs["environment_gradient_norm"].max() >= gradient_floor
        and epochs["trait_amplitude_gradient_norm"].max() >= gradient_floor
        and selected["genotype_raw_direction_norm_min"] >= direction_floor
        and selected["environment_raw_direction_norm_min"] >= direction_floor
        and selected["genotype_normalized_norm_max_error"] <= norm_tolerance
        and selected["environment_normalized_norm_max_error"] <= norm_tolerance
    )
    final_rms = float(selected["validation_FA_residual_rms"])
    final_active = bool(
        final_rms
        >= float(
            thresholds[
                "minimum_final_validation_residual_rms_for_performance_eligibility"
            ]
        )
    )
    return {
        "FA_optimization_path_certified": optimization_path,
        "FA_final_component_active": final_active,
        "FA_selected_validation_residual_rms": final_rms,
        "FA_selected_training_residual_rms": float(
            selected["training_FA_residual_rms"]
        ),
        "FA_minimum_raw_genotype_direction_norm": float(
            selected["genotype_raw_direction_norm_min"]
        ),
        "FA_minimum_raw_environment_direction_norm": float(
            selected["environment_raw_direction_norm_min"]
        ),
        "FA_minimum_trait_amplitude_norm": float(
            selected["trait_amplitude_norm_min"]
        ),
        "FA_maximum_observed_genotype_gradient_norm": float(
            epochs["genotype_gradient_norm"].max()
        ),
        "FA_maximum_observed_environment_gradient_norm": float(
            epochs["environment_gradient_norm"].max()
        ),
        "FA_maximum_observed_trait_amplitude_gradient_norm": float(
            epochs["trait_amplitude_gradient_norm"].max()
        ),
    }


def train_amendment_run(
    *, root: Path, state_id: str, candidate: str, seed: int, out_dir: Path
) -> dict[str, object]:
    root = root.resolve()
    out_dir = out_dir.resolve()
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    protocol = load_protocol(root)

    original = {
        "PROTOCOL": legacy.PROTOCOL,
        "TRAINER": legacy.TRAINER,
        "RUN_PROTOCOL": legacy.RUN_PROTOCOL,
        "load_protocol": legacy.load_protocol,
        "fit": legacy.fit_factor_analytic_component,
    }
    try:
        legacy.PROTOCOL = PROTOCOL
        legacy.TRAINER = TRAINER
        legacy.RUN_PROTOCOL = RUN_PROTOCOL
        legacy.load_protocol = load_protocol
        legacy.fit_factor_analytic_component = fit_normalized_direction_component
        metadata = legacy.train_factor_analytic_run(
            root=root,
            state_id=state_id,
            candidate=candidate,
            seed=seed,
            out_dir=out_dir,
        )
    finally:
        legacy.PROTOCOL = original["PROTOCOL"]
        legacy.TRAINER = original["TRAINER"]
        legacy.RUN_PROTOCOL = original["RUN_PROTOCOL"]
        legacy.load_protocol = original["load_protocol"]
        legacy.fit_factor_analytic_component = original["fit"]

    trait_metrics = pd.read_csv(out_dir / "validation_trait_metrics.tsv", sep="\t")
    all_seven_nrmse = float(metadata["validation_macro_normalized_rmse"])
    all_seven_pearson = float(metadata["validation_macro_pearson"])
    all_seven_calibration = float(
        metadata["validation_macro_calibration_error"]
    )
    primary = _primary_summary(
        trait_metrics, protocol["objective_policy"]["primary_macro_traits"]
    )
    activity = pd.read_csv(out_dir / "component_activity_history.tsv", sep="\t")
    activity_summary = _activity_summary(activity, protocol)
    test_weight = trait_metrics.loc[
        trait_metrics["trait_name_canonical"].eq("TEST_WEIGHT")
    ]
    if len(test_weight) != 1 or int(test_weight.iloc[0]["rows"]) <= 0:
        raise ValueError("TEST_WEIGHT was lost from validation reporting")
    metadata.update(
        {
            "validation_all_seven_macro_normalized_rmse": all_seven_nrmse,
            "validation_all_seven_macro_pearson": all_seven_pearson,
            "validation_all_seven_macro_calibration_error": (
                all_seven_calibration
            ),
            **primary,
            **activity_summary,
            "primary_macro_trait_count": 6,
            "primary_macro_excludes_TEST_WEIGHT": True,
            "training_likelihood_trait_count": 7,
            "TEST_WEIGHT_training_rows_retained": True,
            "TEST_WEIGHT_validation_rows": int(test_weight.iloc[0]["rows"]),
            "TEST_WEIGHT_reporting_retained": True,
            "TEST_WEIGHT_calibration_retained": True,
            "normalized_direction_parameterization": True,
            "only_mutable_component": "covariate_linked_factor_analytic_optimization",
            "genotype_direction_L2_penalty": False,
            "environment_direction_L2_penalty": False,
            "trait_amplitude_loading_L2_penalty": True,
            "parent_factor_analytic_decision_sha256": sha256_file(
                root / PARENT_DECISION
            ),
            "trainer_sha256": sha256_file(code_root / TRAINER),
            "protocol_sha256": sha256_file(code_root / PROTOCOL),
        }
    )
    metadata["artifacts"]["component_activity_history.tsv"] = sha256_file(
        out_dir / "component_activity_history.tsv"
    )
    legacy.write_json(out_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one normalized-direction Stage-1 v2 FA amendment candidate"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    tf.config.experimental.enable_op_determinism()
    train_amendment_run(
        root=args.root,
        state_id=args.state_id,
        candidate=args.candidate,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
