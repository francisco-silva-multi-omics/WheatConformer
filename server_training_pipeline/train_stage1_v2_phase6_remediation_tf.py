from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

from .stage1_v2_phase6_remediation import (
    add_hierarchy_indices,
    apply_calibration,
    fit_positive_calibration,
)
from .stage1_v2_trainer_interface import load_selection_protocol, load_state_spec
from .train_stage1_v2_phase6_confirmation_tf import build_confirmation_factors
from .train_stage1_v2_phase6_tf import (
    FactorBlock,
    Stage1V2ReactionNorm,
    _candidate_marker_gids,
    _projection_active_environments,
    add_factor_indices,
    build_ka_factor,
    build_projection_environment,
    git_commit,
    identifier_signature,
    load_state_observations,
    macro_nrmse,
    prepare_targets,
    reporting_subset_masks,
    sha256_file,
    stable_seed,
    validation_metrics,
    write_json,
)


PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_remediation_protocol_v1.json"
)
CONFIRMATION_PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_confirmation_protocol_v1.json"
)
FACTOR_BUILDER = Path("server_training_pipeline/train_stage1_v2_phase6_tf.py")
TRAINER_INTERFACE = Path("server_training_pipeline/stage1_v2_trainer_interface.py")
RUN_PROTOCOL = "stage1_v2_phase6_structural_remediation_tf_v1"
REFERENCE = "historical_reaction_reference"
HIERARCHY = "known_environment_hierarchical_v2"
PROJECTION_ROUTE = "projection_output_routed_calibrated_v2"
MARKER_ROUTE = "marker_supported_output_routed_v2"


def load_remediation_protocol(root: Path) -> dict[str, Any]:
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    protocol = json.loads((code_root / PROTOCOL).read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "stage1_v2_phase6_structural_remediation_v1":
        raise ValueError("Unexpected Stage-1 v2 remediation protocol")
    if protocol.get("outer_test_metrics_read") is not False:
        raise ValueError("Remediation protocol has read outer-test metrics")
    if protocol.get("final_holdout_outcomes_read") is not False:
        raise ValueError("Remediation protocol has read final-holdout outcomes")
    return protocol


def marker_gids(root: Path, state_id: str) -> set[str]:
    parent = load_selection_protocol(root)
    return _candidate_marker_gids(
        root,
        state_id,
        "ka_seeds_cimmyt_historical_environment",
        parent,
    )


def reporting_masks(
    root: Path,
    state_id: str,
    frame: pd.DataFrame,
    protocol: dict[str, Any],
) -> dict[str, dict[str, pd.Series]]:
    projection_active = _projection_active_environments(root, state_id)
    markers = marker_gids(root, state_id)
    return {
        candidate: reporting_subset_masks(
            frame,
            marker_gids=markers if candidate == MARKER_ROUTE else set(),
            projection_active_environments=projection_active,
        )
        for candidate in protocol["phase_1"]["candidate_order"]
    }


def trait_regularization_vectors(
    protocol: dict[str, Any], trait_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    contract = protocol["trait_specific_regularization"]
    floors = np.full(
        len(trait_names), float(contract["default_residual_scale_floor"]), dtype=np.float32
    )
    multipliers = np.full(
        len(trait_names),
        float(contract["default_trait_loading_penalty_multiplier"]),
        dtype=np.float32,
    )
    for index, trait in enumerate(trait_names):
        override = contract["overrides"].get(trait)
        if override:
            floors[index] = float(override["residual_scale_floor"])
            multipliers[index] = float(override["trait_loading_penalty_multiplier"])
    return floors, multipliers


class TraitRegularizedReactionNorm(Stage1V2ReactionNorm):
    def __init__(
        self,
        *args,
        trait_residual_floors: np.ndarray,
        trait_penalty_multipliers: np.ndarray,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trait_residual_floors = tf.constant(trait_residual_floors, tf.float32)
        self.trait_penalty_multipliers = tf.constant(
            trait_penalty_multipliers, tf.float32
        )

    def residual_scales(self) -> tf.Tensor:
        return tf.nn.softplus(self.raw_residual) + self.trait_residual_floors

    def regularization_loss(self) -> tf.Tensor:
        base = super().regularization_loss()
        additional = tf.reduce_sum(
            tf.square(self.trait_loadings)
            * (self.trait_penalty_multipliers[tf.newaxis, :] - 1.0)
        )
        return base + self.weight_decay * additional


class HierarchicalReactionNorm(TraitRegularizedReactionNorm):
    def __init__(
        self,
        *args,
        trial_support: np.ndarray,
        environment_support: np.ndarray,
        trial_penalty: float,
        environment_penalty: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trial_support = tf.constant(trial_support, tf.bool)
        self.environment_support = tf.constant(environment_support, tf.bool)
        self.trial_penalty = float(trial_penalty)
        self.environment_penalty = float(environment_penalty)
        self.has_trial_effects = bool(trial_support.size)
        self.has_environment_effects = bool(environment_support.size)
        self.trial_effects = self.add_weight(
            name="trial_trait_intercepts",
            shape=trial_support.shape,
            initializer="zeros",
        )
        self.environment_intercepts = self.add_weight(
            name="environment_trait_intercepts",
            shape=environment_support.shape,
            initializer="zeros",
        )

    @staticmethod
    def _effect(
        coefficients: tf.Tensor,
        support: tf.Tensor,
        entity_index: tf.Tensor,
        trait_index: tf.Tensor,
    ) -> tf.Tensor:
        if coefficients.shape[0] == 0:
            return tf.zeros_like(tf.cast(entity_index, tf.float32))
        available = entity_index >= 0
        safe = tf.maximum(entity_index, 0)
        entity_coefficients = tf.gather(coefficients, safe)
        value = tf.gather(entity_coefficients, trait_index, batch_dims=1)
        entity_support = tf.gather(support, safe)
        eligible = tf.gather(entity_support, trait_index, batch_dims=1)
        return value * tf.cast(available & eligible, tf.float32)

    def call(self, inputs, training: bool = False):
        if len(inputs) != 6:
            raise ValueError("Hierarchical v2 model requires six input tensors")
        core = inputs[:4]
        prediction = super().call(core, training=training)
        prediction += self._effect(
            self.trial_effects, self.trial_support, inputs[4], inputs[3]
        )
        prediction += self._effect(
            self.environment_intercepts,
            self.environment_support,
            inputs[5],
            inputs[3],
        )
        return prediction

    def regularization_loss(self) -> tf.Tensor:
        value = super().regularization_loss()
        if self.has_trial_effects:
            value += self.trial_penalty * tf.reduce_mean(tf.square(self.trial_effects))
        if self.has_environment_effects:
            value += self.environment_penalty * tf.reduce_mean(
                tf.square(self.environment_intercepts)
            )
        return value


def make_component_dataset(
    frame: pd.DataFrame,
    genotype_columns: Sequence[str],
    environment_columns: Sequence[str],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    hierarchy: bool,
) -> tf.data.Dataset:
    inputs: tuple[np.ndarray, ...] = (
        frame[list(genotype_columns)].to_numpy(dtype=np.int32),
        frame[list(environment_columns)].to_numpy(dtype=np.int32),
        frame["reaction_environment_index"].to_numpy(dtype=np.int32),
        frame["trait_index"].to_numpy(dtype=np.int32),
    )
    if hierarchy:
        inputs += (
            frame["trial_hierarchy_index"].to_numpy(dtype=np.int32),
            frame["environment_hierarchy_index"].to_numpy(dtype=np.int32),
        )
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            inputs,
            frame["y_scaled"].to_numpy(dtype=np.float32),
            frame["loss_weight"].to_numpy(dtype=np.float32),
        )
    )
    if shuffle:
        dataset = dataset.shuffle(
            min(len(frame), 100_000), seed=seed, reshuffle_each_iteration=True
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def predict_component(
    model: tf.keras.Model,
    frame: pd.DataFrame,
    genotype_columns: Sequence[str],
    environment_columns: Sequence[str],
    batch_size: int,
    hierarchy: bool,
) -> np.ndarray:
    dataset = make_component_dataset(
        frame,
        genotype_columns,
        environment_columns,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        hierarchy=hierarchy,
    )
    values = [model(inputs, training=False) for inputs, _, _ in dataset]
    return tf.concat(values, axis=0).numpy() if values else np.empty(0, np.float32)


@dataclass
class ComponentFit:
    label: str
    training_ids: np.ndarray
    validation_ids: np.ndarray
    training_prediction: np.ndarray
    validation_prediction: np.ndarray
    best_metric: float
    epochs_completed: int
    epoch_history: pd.DataFrame
    factor_inventory: pd.DataFrame
    hierarchy_support: pd.DataFrame
    reaction_enabled: bool
    reaction_feature_count: int
    reaction_available_environment_count: int
    fit_training_rows: int
    checkpoint_validation_rows: int


def fit_component(
    *,
    root: Path,
    state_id: str,
    component: str,
    configuration: dict[str, Any],
    frame: pd.DataFrame,
    trait_names: Sequence[str],
    seed: int,
    protocol: dict[str, Any],
    hierarchy: bool,
    trait_regularization: bool,
    fit_scope: str = "all",
) -> ComponentFit:
    if component == "projection":
        genotype = (
            build_ka_factor(root, state_id, int(configuration["max_rank_genotype"])),
        )
        environment, reaction_design, reaction_available, reaction_ids = (
            build_projection_environment(root, state_id)
        )
        reaction_enabled = reaction_design.shape[1] > 0 and bool(
            reaction_available.any()
        )
    else:
        confirmation_candidate = {
            "historical": REFERENCE,
            "multikernel": "historical_v2_native_multikernel",
        }[component]
        (
            genotype,
            environment,
            reaction_design,
            reaction_available,
            reaction_ids,
            reaction_enabled,
        ) = build_confirmation_factors(
            root, state_id, confirmation_candidate, configuration
        )
    local = frame.copy()
    genotype_columns, environment_columns = add_factor_indices(
        local,
        genotype,
        environment,
        reaction_ids,
        reaction_available,
    )
    hierarchy_support = pd.DataFrame(
        columns=[
            "level",
            "entity_id",
            "trait_name_canonical",
            "positive_weight_training_rows",
            "minimum_rows",
            "eligible",
        ]
    )
    trial_support = np.zeros((0, len(trait_names)), dtype=bool)
    environment_support = np.zeros((0, len(trait_names)), dtype=bool)
    if hierarchy:
        local, trial_support, environment_support, hierarchy_support = (
            add_hierarchy_indices(local, trait_names, protocol)
        )
    training = local.loc[local["selection_role"].eq("TRAINING")].reset_index(drop=True)
    validation = local.loc[
        local["selection_role"].eq("INNER_VALIDATION")
    ].reset_index(drop=True)
    if fit_scope == "all":
        fit_training = training
        checkpoint_validation = validation
    elif fit_scope == "projection_active":
        fit_training = training.loc[training["projection_core_active"]].reset_index(drop=True)
        checkpoint_validation = validation.loc[
            validation["projection_core_active"]
        ].reset_index(drop=True)
    elif fit_scope == "marker_supported":
        fit_training = training.loc[
            training["candidate_marker_supported"]
        ].reset_index(drop=True)
        checkpoint_validation = validation.loc[
            validation["candidate_marker_supported"]
        ].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown component fit scope: {fit_scope}")
    if fit_training.empty or checkpoint_validation.empty:
        raise ValueError(
            f"Component {component} lacks {fit_scope} training or checkpoint rows"
        )
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    common: dict[str, Any] = {
        "genotype": genotype,
        "environment": environment,
        "reaction_design": reaction_design,
        "trait_names": trait_names,
        "latent_dim": int(configuration["latent_dim"]),
        "reaction_rank": int(configuration["reaction_rank"]),
        "residual_floor": 0.05,
        "weight_decay": float(configuration["weight_decay"]),
        "seed": seed,
        "reaction_enabled": reaction_enabled,
    }
    if trait_regularization:
        floors, multipliers = trait_regularization_vectors(protocol, trait_names)
        common.update(
            {
                "trait_residual_floors": floors,
                "trait_penalty_multipliers": multipliers,
            }
        )
        if hierarchy:
            contract = protocol["trial_environment_hierarchy"]
            model: tf.keras.Model = HierarchicalReactionNorm(
                **common,
                trial_support=trial_support,
                environment_support=environment_support,
                trial_penalty=float(contract["trial_effect_penalty"]),
                environment_penalty=float(contract["environment_effect_penalty"]),
            )
        else:
            model = TraitRegularizedReactionNorm(**common)
    else:
        if hierarchy:
            raise ValueError("Hierarchy requires the remediation model implementation")
        model = Stage1V2ReactionNorm(**common)

    optimizer = tf.keras.optimizers.Adam(float(configuration["learning_rate"]))
    batch_size = int(configuration["batch_size"])
    train_dataset = make_component_dataset(
        fit_training,
        genotype_columns,
        environment_columns,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        hierarchy=hierarchy,
    )

    @tf.function
    def train_step(inputs, target, weight):
        with tf.GradientTape() as tape:
            prediction = model(inputs, training=True)
            trait_index = inputs[3]
            scale = tf.gather(model.residual_scales(), trait_index)
            nll = 0.5 * tf.square((target - prediction) / scale) + tf.math.log(scale)
            denominator = tf.maximum(tf.reduce_sum(weight), 1e-6)
            loss = tf.reduce_sum(nll * weight) / denominator + model.regularization_loss()
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    best_metric = float("inf")
    best_weights: list[np.ndarray] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, int(configuration["epochs_max"]) + 1):
        losses = [train_step(inputs, target, weight) for inputs, target, weight in train_dataset]
        mean_loss = float(tf.reduce_mean(tf.stack(losses)).numpy())
        if epoch == 1 or epoch % 5 == 0:
            validation_prediction = predict_component(
                model,
                checkpoint_validation,
                genotype_columns,
                environment_columns,
                batch_size,
                hierarchy,
            )
            metric = macro_nrmse(checkpoint_validation, validation_prediction)
            row = {
                "component": component,
                "epoch": epoch,
                "train_gaussian_nll_regularized": mean_loss,
                "validation_macro_normalized_rmse": metric,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if metric < best_metric - 1e-7:
                best_metric = metric
                best_weights = model.get_weights()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 5 if epoch > 1 else 1
            if epochs_without_improvement >= int(
                configuration["early_stopping_patience"]
            ):
                break
    if best_weights is None:
        raise RuntimeError(f"Component {component} did not produce a checkpoint")
    model.set_weights(best_weights)
    training_prediction = predict_component(
        model,
        training,
        genotype_columns,
        environment_columns,
        batch_size,
        hierarchy,
    )
    validation_prediction = predict_component(
        model,
        validation,
        genotype_columns,
        environment_columns,
        batch_size,
        hierarchy,
    )
    inventory = pd.DataFrame(
        [
            {
                "component_model": component,
                "component": block.name,
                "axis": block.axis,
                "entities": len(block.entity_ids),
                "available_entities": int(block.available.sum()),
                "rank": block.values.shape[1],
                "state_hash": block.state_hash,
            }
            for block in (*genotype, *environment)
        ]
    )
    result = ComponentFit(
        label=component,
        training_ids=training["phase4_adjusted_row_id"].astype(str).to_numpy(),
        validation_ids=validation["phase4_adjusted_row_id"].astype(str).to_numpy(),
        training_prediction=training_prediction,
        validation_prediction=validation_prediction,
        best_metric=best_metric,
        epochs_completed=int(history[-1]["epoch"]),
        epoch_history=pd.DataFrame(history),
        factor_inventory=inventory,
        hierarchy_support=hierarchy_support,
        reaction_enabled=reaction_enabled,
        reaction_feature_count=int(reaction_design.shape[1]),
        reaction_available_environment_count=int(reaction_available.sum()),
        fit_training_rows=len(fit_training),
        checkpoint_validation_rows=len(checkpoint_validation),
    )
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def _assert_aligned(frame: pd.DataFrame, fit: ComponentFit, role: str) -> None:
    observed = (
        frame.loc[frame["selection_role"].eq(role), "phase4_adjusted_row_id"]
        .astype(str)
        .to_numpy()
    )
    expected = fit.training_ids if role == "TRAINING" else fit.validation_ids
    if not np.array_equal(observed, expected):
        raise ValueError(f"Component {fit.label} lost {role} observation order")


def train_remediation_run(
    *, root: Path, state_id: str, candidate: str, seed: int, out_dir: Path
) -> dict[str, object]:
    root = root.resolve()
    out_dir = out_dir.resolve()
    intra_op_threads = int(os.environ.get("STAGE1_V2_INTRA_OP_THREADS", "16"))
    inter_op_threads = int(os.environ.get("STAGE1_V2_INTER_OP_THREADS", "2"))
    tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
    protocol = load_remediation_protocol(root)
    if candidate not in protocol["candidates"]:
        raise ValueError(f"Unknown remediation candidate: {candidate}")
    candidate_contract = protocol["candidates"][candidate]
    state = load_state_spec(root, state_id, "ka_historical_environment")
    if state.state_level != "INNER" or state.outer_fold != 1:
        raise ValueError("Remediation Phase 1 is restricted to outer-fold-1 inner states")
    if state.scenario not in candidate_contract["eligible_scenarios"]:
        raise ValueError(f"Candidate {candidate} is not eligible for {state.scenario}")
    configuration_label = str(candidate_contract["configuration"])
    configuration = protocol["hyperparameter_configurations"][configuration_label]
    frame, role_metadata = load_state_observations(root, state_id)
    trait_names = [*protocol["primary_traits"], *protocol["exploratory_traits"]]
    frame, scaling = prepare_targets(frame, trait_names)
    full_masks = reporting_masks(root, state_id, frame, protocol)
    frame["candidate_marker_supported"] = full_masks[candidate]["MARKER_SUPPORTED"]
    frame["recovered_component_supported"] = full_masks[candidate][
        "RECOVERED_IDENTITY_OR_COMPONENT"
    ]
    frame["projection_core_active"] = full_masks[candidate]["PROJECTION_CORE_ACTIVE"]
    training = frame.loc[frame["selection_role"].eq("TRAINING")].reset_index(drop=True)
    validation = frame.loc[
        frame["selection_role"].eq("INNER_VALIDATION")
    ].reset_index(drop=True)
    validation_masks = reporting_masks(root, state_id, validation, protocol)

    fits: list[ComponentFit] = []
    calibration = pd.DataFrame(
        columns=[
            "trait_name_canonical",
            "training_rows",
            "intercept",
            "slope",
            "status",
            "validation_values_used",
        ]
    )
    if candidate == REFERENCE:
        fit = fit_component(
            root=root,
            state_id=state_id,
            component="historical",
            configuration=configuration,
            frame=frame,
            trait_names=trait_names,
            seed=seed,
            protocol=protocol,
            hierarchy=False,
            trait_regularization=False,
            fit_scope="all",
        )
        fits.append(fit)
        prediction = fit.validation_prediction
        active_validation = np.ones(len(validation), dtype=bool)
        active_training = np.ones(len(training), dtype=bool)
    elif candidate == HIERARCHY:
        fit = fit_component(
            root=root,
            state_id=state_id,
            component="historical",
            configuration=configuration,
            frame=frame,
            trait_names=trait_names,
            seed=seed,
            protocol=protocol,
            hierarchy=True,
            trait_regularization=True,
            fit_scope="all",
        )
        fits.append(fit)
        active_training = np.ones(len(training), dtype=bool)
        active_validation = np.ones(len(validation), dtype=bool)
        calibration = fit_positive_calibration(
            training,
            fit.training_prediction,
            active_training,
            trait_names,
            protocol,
        )
        prediction = apply_calibration(
            validation,
            fit.validation_prediction,
            active_validation,
            calibration,
        )
    else:
        fallback_configuration = protocol["hyperparameter_configurations"][
            "historical_capacity_16_batch8192"
        ]
        historical = fit_component(
            root=root,
            state_id=state_id,
            component="historical",
            configuration=fallback_configuration,
            frame=frame,
            trait_names=trait_names,
            seed=seed,
            protocol=protocol,
            hierarchy=False,
            trait_regularization=False,
            fit_scope="all",
        )
        active_component = "projection" if candidate == PROJECTION_ROUTE else "multikernel"
        active_fit = fit_component(
            root=root,
            state_id=state_id,
            component=active_component,
            configuration=configuration,
            frame=frame,
            trait_names=trait_names,
            seed=seed,
            protocol=protocol,
            hierarchy=False,
            trait_regularization=True,
            fit_scope=(
                "projection_active"
                if candidate == PROJECTION_ROUTE
                else "marker_supported"
            ),
        )
        fits.extend([historical, active_fit])
        if candidate == PROJECTION_ROUTE:
            active_training = training["projection_core_active"].to_numpy(dtype=bool)
            active_validation = validation["projection_core_active"].to_numpy(dtype=bool)
        else:
            active_training = training["candidate_marker_supported"].to_numpy(dtype=bool)
            active_validation = validation["candidate_marker_supported"].to_numpy(dtype=bool)
        calibration = fit_positive_calibration(
            training,
            active_fit.training_prediction,
            active_training,
            trait_names,
            protocol,
        )
        active_prediction = apply_calibration(
            validation,
            active_fit.validation_prediction,
            active_validation,
            calibration,
        )
        prediction = historical.validation_prediction.copy()
        prediction[active_validation] = active_prediction[active_validation]
        if not np.array_equal(
            prediction[~active_validation], historical.validation_prediction[~active_validation]
        ):
            raise ValueError("Output route changed historical fallback predictions")

    for fit in fits:
        _assert_aligned(frame, fit, "TRAINING")
        _assert_aligned(frame, fit, "INNER_VALIDATION")
    trait_metrics, subset_metrics, all_guard_metrics, summary = validation_metrics(
        validation,
        prediction,
        scaling,
        validation_masks,
        candidate,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    scaling.to_csv(out_dir / "trait_scaling.tsv", sep="\t", index=False)
    pd.concat([fit.epoch_history for fit in fits], ignore_index=True).to_csv(
        out_dir / "component_epoch_history.tsv", sep="\t", index=False
    )
    pd.concat([fit.factor_inventory for fit in fits], ignore_index=True).to_csv(
        out_dir / "active_component_factors.tsv", sep="\t", index=False
    )
    supports = [fit.hierarchy_support for fit in fits if not fit.hierarchy_support.empty]
    (pd.concat(supports, ignore_index=True) if supports else fits[0].hierarchy_support).to_csv(
        out_dir / "trial_environment_hierarchy_support.tsv", sep="\t", index=False
    )
    calibration.to_csv(out_dir / "training_only_calibration.tsv", sep="\t", index=False)
    trait_metrics.to_csv(out_dir / "validation_trait_metrics.tsv", sep="\t", index=False)
    subset_metrics.to_csv(out_dir / "validation_subset_metrics.tsv", sep="\t", index=False)
    all_guard_metrics.to_csv(
        out_dir / "validation_guard_metrics.tsv", sep="\t", index=False
    )
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    metadata = {
        "status": "PASS",
        "protocol_version": RUN_PROTOCOL,
        "stage1_version": "Stage-1 v2",
        "state_id": state_id,
        "scenario": state.scenario,
        "outer_fold": state.outer_fold,
        "inner_fold": state.inner_fold,
        "candidate": candidate,
        "configuration_label": configuration_label,
        "configuration": configuration,
        "seed": seed,
        "component_models": [fit.label for fit in fits],
        "component_best_validation_nrmse": {
            fit.label: fit.best_metric for fit in fits
        },
        "component_epochs_completed": {
            fit.label: fit.epochs_completed for fit in fits
        },
        "component_fit_training_rows": {
            fit.label: fit.fit_training_rows for fit in fits
        },
        "component_checkpoint_validation_rows": {
            fit.label: fit.checkpoint_validation_rows for fit in fits
        },
        "active_route_training_rows": int(active_training.sum()),
        "active_route_validation_rows": int(active_validation.sum()),
        "fallback_validation_rows": int((~active_validation).sum()),
        "fallback_predictions_preserved_exactly": candidate in {PROJECTION_ROUTE, MARKER_ROUTE},
        "positive_training_calibration_fitted": not calibration.empty,
        "calibration_validation_values_used": False,
        "trait_specific_regularization": bool(
            candidate_contract["trait_specific_regularization"]
        ),
        "hierarchy_fit_partition": (
            "inner_training_only" if candidate == HIERARCHY else "not_applicable"
        ),
        "reaction_enabled_components": {
            fit.label: fit.reaction_enabled for fit in fits
        },
        **role_metadata,
        **summary,
        "best_validation_macro_nrmse": float(summary["validation_macro_normalized_rmse"]),
        "selection_protocol_sha256": sha256_file(code_root / PROTOCOL),
        "confirmation_protocol_sha256": sha256_file(code_root / CONFIRMATION_PROTOCOL),
        "trainer_sha256": sha256_file(Path(__file__)),
        "factor_builder_sha256": sha256_file(code_root / FACTOR_BUILDER),
        "trainer_interface_sha256": sha256_file(code_root / TRAINER_INTERFACE),
        "code_commit": git_commit(root),
        "validation_observation_signature": identifier_signature(
            validation["phase4_adjusted_row_id"].tolist()
        ),
        "guard_mask_candidate_count": int(all_guard_metrics["mask_candidate"].nunique()),
        "guard_mask_observation_signatures_written": True,
        "phenotype_values_read": True,
        "inner_validation_metrics_read": True,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "model_training_performed": True,
    }
    write_json(out_dir / "run_metadata.json", metadata)
    gc.collect()
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one Stage-1 v2 structural-remediation inner run"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_remediation_run(
        root=args.root,
        state_id=args.state_id,
        candidate=args.candidate,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
