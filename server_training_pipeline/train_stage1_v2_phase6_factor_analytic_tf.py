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
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
import tensorflow as tf

from .stage1_v2_phase6_hierarchy_calibration_amendment_v2 import (
    fit_test_weight_calibration,
)
from .stage1_v2_phase6_remediation import apply_calibration, fit_positive_calibration
from .stage1_v2_trainer_interface import load_state_spec
from .train_stage1_v2_phase6_confirmation_tf import build_confirmation_factors
from .train_stage1_v2_phase6_hierarchy_calibration_amendment_tf import (
    shared_reporting_masks,
)
from .train_stage1_v2_phase6_private_heads_tf import (
    authoritative_mass_diagnostics,
    loss_weight_signature,
)
from .train_stage1_v2_phase6_remediation_tf import (
    HierarchicalReactionNorm,
    persist_component_replay_artifacts,
    trait_regularization_vectors,
)
from .train_stage1_v2_phase6_tf import (
    add_factor_indices,
    build_projection_environment,
    git_commit,
    identifier_signature,
    load_state_observations,
    macro_nrmse,
    prepare_targets,
    sha256_file,
    stable_seed,
    validation_metrics,
    write_json,
)


PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_factor_analytic_screen_protocol_v1.json"
)
TRAINER = Path(
    "server_training_pipeline/train_stage1_v2_phase6_factor_analytic_tf.py"
)
CALIBRATION_HELPER = Path(
    "server_training_pipeline/stage1_v2_phase6_hierarchy_calibration_amendment_v2.py"
)
CALIBRATION_TRAINER = Path(
    "server_training_pipeline/"
    "train_stage1_v2_phase6_hierarchy_calibration_amendment_tf.py"
)
REMEDIATION_HELPER = Path("server_training_pipeline/stage1_v2_phase6_remediation.py")
MODEL_BUILDER = Path("server_training_pipeline/train_stage1_v2_phase6_remediation_tf.py")
FACTOR_BUILDER = Path("server_training_pipeline/train_stage1_v2_phase6_tf.py")
TRAINER_INTERFACE = Path("server_training_pipeline/stage1_v2_trainer_interface.py")
PRIVATE_HEAD_DECISION = Path(
    "model_kernels/stage1_v2_phase6_private_head_screen_v1/phase_1/"
    "PRIVATE_HEAD_PHASE1_DECISION.json"
)
POST_HIERARCHY_PLAN = Path(
    "server_training_pipeline/stage1_v2_phase6_post_hierarchy_screen_plan_v2.json"
)
PROJECTION_PROTOCOL = Path(
    "server_training_pipeline/phase6a_split_bound_projection_inputs_protocol_v1.json"
)
RUN_PROTOCOL = "stage1_v2_phase6_covariate_linked_factor_analytic_tf_v1"
CALIBRATION_CANDIDATE = "hierarchy_test_weight_environment_oof_huber_v2"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path) -> dict[str, Any]:
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    protocol = read_json(code_root / PROTOCOL)
    if protocol.get("protocol_version") != (
        "stage1_v2_phase6_factor_analytic_screen_v1"
    ):
        raise ValueError("Unexpected factor-analytic protocol")
    policy = protocol["architecture_policy"]
    if policy.get("only_mutable_component") != (
        "covariate_linked_factor_analytic_residual"
    ):
        raise ValueError("Factor-analytic screen changes an unauthorized component")
    if policy.get("free_environment_loadings_allowed") is not False:
        raise ValueError("Factor-analytic screen permits free environment loadings")
    feature_contract = protocol["projection_feature_contract"]
    if int(feature_contract["expected_feature_count"]) != 153:
        raise ValueError("Factor-analytic screen does not bind the 153-feature schema")
    if int(protocol["fixed_configuration"]["batch_size"]) != 8192:
        raise ValueError("Factor-analytic screen changed the frozen batch size")
    ranks = {
        int(value["factor_analytic_rank"])
        for value in protocol["candidates"].values()
        if value.get("source_reuse") is False
    }
    if ranks != {2, 4}:
        raise ValueError(f"Unexpected bounded FA ranks: {sorted(ranks)}")
    return protocol


class CovariateLinkedFactorAnalyticHierarchicalReactionNorm(
    HierarchicalReactionNorm
):
    """Add a projection-safe low-rank GxE residual to the frozen hierarchy."""

    def __init__(
        self,
        *args,
        projection_design: np.ndarray,
        projection_available: np.ndarray,
        factor_analytic_rank: int,
        factor_penalty_multiplier: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        design = np.asarray(projection_design, dtype=np.float32)
        available = np.asarray(projection_available, dtype=bool)
        if design.ndim != 2 or design.shape[1] != 153:
            raise ValueError(
                f"Projection design must have 153 columns; observed={design.shape}"
            )
        if available.shape != (design.shape[0],):
            raise ValueError("Projection availability does not match its environment axis")
        if factor_analytic_rank not in {2, 4}:
            raise ValueError("The preregistered FA rank must be 2 or 4")
        if not self.genotype_blocks or "K_A" not in self.genotype_blocks[0].name:
            raise ValueError("The first genotype block must be the split-bound K_A factor")
        self.projection_design = tf.constant(design, dtype=tf.float32)
        self.projection_available = tf.constant(available, dtype=tf.bool)
        self.factor_analytic_rank = int(factor_analytic_rank)
        self.factor_penalty_multiplier = float(factor_penalty_multiplier)

        def initializer(label: str) -> tf.keras.initializers.Initializer:
            return tf.keras.initializers.RandomNormal(
                stddev=0.02, seed=stable_seed(int(kwargs["seed"]), label)
            )

        genotype_rank = int(self.genotype_blocks[0].values.shape[1])
        trait_count = len(self.trait_names)
        self.fa_genotype_coefficients = self.add_weight(
            name="covariate_linked_fa_genotype_coefficients",
            shape=(genotype_rank, self.factor_analytic_rank),
            initializer=initializer("fa_genotype_coefficients"),
        )
        self.fa_environment_coefficients = self.add_weight(
            name="covariate_linked_fa_environment_coefficients",
            shape=(design.shape[1], self.factor_analytic_rank),
            initializer=initializer("fa_environment_coefficients"),
        )
        self.fa_trait_loadings = self.add_weight(
            name="covariate_linked_fa_trait_loadings",
            shape=(self.factor_analytic_rank, trait_count),
            initializer=initializer("fa_trait_loadings"),
        )

    @property
    def factor_analytic_variables(self) -> tuple[tf.Variable, ...]:
        return (
            self.fa_genotype_coefficients,
            self.fa_environment_coefficients,
            self.fa_trait_loadings,
        )

    def call(self, inputs, training: bool = False):
        if len(inputs) != 7:
            raise ValueError("Covariate-linked FA model requires seven input tensors")
        prediction = super().call(inputs[:6], training=training)
        genotype_indices = inputs[0]
        trait_index = inputs[3]
        projection_index = inputs[6]

        genotype_index = genotype_indices[:, 0]
        genotype_available = genotype_index >= 0
        projection_row_available = projection_index >= 0
        safe_genotype = tf.maximum(genotype_index, 0)
        safe_projection = tf.maximum(projection_index, 0)
        projection_source_available = tf.gather(
            self.projection_available, safe_projection
        )
        active = (
            genotype_available
            & projection_row_available
            & projection_source_available
        )

        genotype_factor = tf.gather(
            self.genotype_factors[0], safe_genotype
        )
        genotype_score = tf.matmul(
            genotype_factor, self.fa_genotype_coefficients
        )
        environment_features = tf.gather(
            self.projection_design, safe_projection
        )
        environment_loading = tf.matmul(
            environment_features, self.fa_environment_coefficients
        )
        trait_loading = tf.gather(
            tf.transpose(self.fa_trait_loadings), trait_index
        )
        residual = tf.reduce_sum(
            genotype_score * environment_loading * trait_loading, axis=1
        ) / math.sqrt(float(self.factor_analytic_rank))
        return prediction + residual * tf.cast(active, tf.float32)

    def regularization_loss(self) -> tf.Tensor:
        value = super().regularization_loss()
        if self.factor_penalty_multiplier != 1.0:
            extra = tf.add_n(
                [tf.reduce_sum(tf.square(item)) for item in self.factor_analytic_variables]
            )
            value += self.weight_decay * (
                self.factor_penalty_multiplier - 1.0
            ) * extra
        return value


def make_dataset(
    frame: pd.DataFrame,
    genotype_columns: Sequence[str],
    environment_columns: Sequence[str],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    inputs = (
        frame[list(genotype_columns)].to_numpy(dtype=np.int32),
        frame[list(environment_columns)].to_numpy(dtype=np.int32),
        frame["reaction_environment_index"].to_numpy(dtype=np.int32),
        frame["trait_index"].to_numpy(dtype=np.int32),
        frame["trial_hierarchy_index"].to_numpy(dtype=np.int32),
        frame["environment_hierarchy_index"].to_numpy(dtype=np.int32),
        frame["fa_environment_index"].to_numpy(dtype=np.int32),
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
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options).batch(batch_size).prefetch(1)


def predict(
    model: tf.keras.Model,
    frame: pd.DataFrame,
    genotype_columns: Sequence[str],
    environment_columns: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    values = []
    for inputs, _, _ in make_dataset(
        frame,
        genotype_columns,
        environment_columns,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    ):
        prediction = model(inputs, training=False)
        tf.debugging.assert_all_finite(prediction, "Non-finite FA inference prediction")
        values.append(prediction)
    result = tf.concat(values, axis=0).numpy() if values else np.empty(0, np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite FA prediction escaped TensorFlow checks")
    return result


@dataclass
class FactorAnalyticFit:
    training_prediction: np.ndarray
    validation_prediction: np.ndarray
    best_metric: float
    epochs_completed: int
    epoch_history: pd.DataFrame
    factor_inventory: pd.DataFrame
    hierarchy_support: pd.DataFrame
    fa_rank: int
    fa_variable_count: int
    fa_parameter_count: int
    fa_active_training_rows: int
    fa_active_validation_rows: int
    replay_artifacts: dict[str, str]


def fit_factor_analytic_component(
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
) -> FactorAnalyticFit:
    from .stage1_v2_phase6_remediation import add_hierarchy_indices

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
    if projection_design.shape[1] != int(
        protocol["projection_feature_contract"]["expected_feature_count"]
    ):
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
        local["environment_id"].astype(str).map(projection_lookup).fillna(-1).astype(np.int32)
    )
    local, trial_support, environment_support, hierarchy_support = (
        add_hierarchy_indices(local, trait_names, protocol)
    )
    training = local.loc[local["selection_role"].eq("TRAINING")].reset_index(drop=True)
    validation = local.loc[
        local["selection_role"].eq("INNER_VALIDATION")
    ].reset_index(drop=True)
    if training.empty or validation.empty:
        raise ValueError("Factor-analytic state lacks training or validation rows")

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    floors, multipliers = trait_regularization_vectors(protocol, trait_names)
    hierarchy = protocol["trial_environment_hierarchy"]
    model = CovariateLinkedFactorAnalyticHierarchicalReactionNorm(
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
        factor_penalty_multiplier=float(candidate_contract["factor_penalty_multiplier"]),
    )
    optimizer = tf.keras.optimizers.Adam(float(configuration["learning_rate"]))
    batch_size = int(configuration["batch_size"])
    training_dataset = make_dataset(
        training,
        genotype_columns,
        environment_columns,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    @tf.function
    def train_step(inputs, target, weight):
        with tf.GradientTape() as tape:
            prediction = model(inputs, training=True)
            tf.debugging.assert_all_finite(prediction, "Non-finite FA training prediction")
            tf.debugging.assert_all_finite(target, "Non-finite FA training target")
            tf.debugging.assert_all_finite(weight, "Non-finite FA training weight")
            trait_index = inputs[3]
            scale = tf.gather(model.residual_scales(), trait_index)
            tf.debugging.assert_all_finite(scale, "Non-finite FA residual scale")
            nll = 0.5 * tf.square((target - prediction) / scale) + tf.math.log(scale)
            tf.debugging.assert_all_finite(nll, "Non-finite FA per-row NLL")
            denominator = tf.maximum(tf.reduce_sum(weight), 1e-6)
            regularization = model.regularization_loss()
            tf.debugging.assert_all_finite(regularization, "Non-finite FA regularization")
            loss = tf.reduce_sum(nll * weight) / denominator + regularization
            tf.debugging.assert_all_finite(loss, "Non-finite FA scalar loss")
        gradients = tape.gradient(loss, model.trainable_variables)
        checked = []
        for gradient, variable in zip(gradients, model.trainable_variables):
            if gradient is None:
                raise RuntimeError(f"Missing FA gradient for {variable.name}")
            values = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
            tf.debugging.assert_all_finite(values, f"Non-finite FA gradient for {variable.name}")
            checked.append(gradient)
        optimizer.apply_gradients(zip(checked, model.trainable_variables))
        return loss

    best_metric = float("inf")
    best_weights: list[np.ndarray] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, int(configuration["epochs_max"]) + 1):
        losses = [
            train_step(inputs, target, weight)
            for inputs, target, weight in training_dataset
        ]
        mean_loss = float(tf.reduce_mean(tf.stack(losses)).numpy())
        if epoch == 1 or epoch % 5 == 0:
            validation_prediction = predict(
                model, validation, genotype_columns, environment_columns, batch_size
            )
            metric = macro_nrmse(validation, validation_prediction)
            row = {
                "component": "historical_plus_covariate_linked_FA",
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
        raise RuntimeError("Factor-analytic component produced no checkpoint")
    model.set_weights(best_weights)
    training_prediction = predict(
        model, training, genotype_columns, environment_columns, batch_size
    )
    validation_prediction = predict(
        model, validation, genotype_columns, environment_columns, batch_size
    )
    replay_artifacts = persist_component_replay_artifacts(
        replay_artifact_dir,
        model,
        training["phase4_adjusted_row_id"].astype(str).to_numpy(),
        validation["phase4_adjusted_row_id"].astype(str).to_numpy(),
        training_prediction,
        validation_prediction,
    )
    inventory = pd.DataFrame(
        [
            {
                "component_model": "historical_plus_covariate_linked_FA",
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
                "component_model": "historical_plus_covariate_linked_FA",
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
    fa_variables = list(model.factor_analytic_variables)
    active_training = training["fa_environment_index"].ge(0).to_numpy()
    active_training &= projection_available[
        np.maximum(training["fa_environment_index"].to_numpy(dtype=int), 0)
    ]
    active_validation = validation["fa_environment_index"].ge(0).to_numpy()
    active_validation &= projection_available[
        np.maximum(validation["fa_environment_index"].to_numpy(dtype=int), 0)
    ]
    result = FactorAnalyticFit(
        training_prediction=training_prediction,
        validation_prediction=validation_prediction,
        best_metric=best_metric,
        epochs_completed=int(history[-1]["epoch"]),
        epoch_history=pd.DataFrame(history),
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


def train_factor_analytic_run(
    *, root: Path, state_id: str, candidate: str, seed: int, out_dir: Path
) -> dict[str, object]:
    root = root.resolve()
    out_dir = out_dir.resolve()
    tf.config.experimental.enable_op_determinism()
    intra_op_threads = int(os.environ.get("STAGE1_V2_INTRA_OP_THREADS", "16"))
    inter_op_threads = int(os.environ.get("STAGE1_V2_INTER_OP_THREADS", "2"))
    tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
    runtime_thread_configuration_sha256 = hashlib.sha256(
        json.dumps(
            {
                "intra_op_threads": intra_op_threads,
                "inter_op_threads": inter_op_threads,
                "tf_deterministic_ops": True,
                "tf_data_deterministic": True,
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "execution_backend": os.environ.get(
                    "STAGE1_V2_EXECUTION_BACKEND", "unknown"
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    protocol = load_protocol(root)
    contract = protocol["candidates"].get(candidate)
    if not contract or contract.get("source_reuse") is not False:
        raise ValueError(f"Unknown factor-analytic candidate: {candidate}")
    state = load_state_spec(root, state_id, "ka_historical_environment")
    projection_state = load_state_spec(root, state_id, "ka_projection_core")
    if (
        state.state_level != "INNER"
        or state.scenario != "GNEW_EOBS"
        or projection_state.state_id != state.state_id
    ):
        raise ValueError("FA training is restricted to matched GNEW_EOBS inner states")

    configuration = dict(protocol["fixed_configuration"])
    configuration_label = str(configuration.pop("label"))
    trait_names = [*protocol["primary_traits"], *protocol["exploratory_traits"]]
    frame, role_metadata = load_state_observations(root, state_id)
    frame, scaling = prepare_targets(frame, trait_names)
    authoritative_weight_sha256 = loss_weight_signature(frame)
    training = frame.loc[frame["selection_role"].eq("TRAINING")].reset_index(drop=True)
    validation = frame.loc[
        frame["selection_role"].eq("INNER_VALIDATION")
    ].reset_index(drop=True)
    mass_diagnostics = authoritative_mass_diagnostics(training)
    masks = shared_reporting_masks(root, state_id, validation, [candidate])
    out_dir.mkdir(parents=True, exist_ok=True)

    fit = fit_factor_analytic_component(
        root=root,
        state_id=state_id,
        configuration=configuration,
        frame=frame,
        trait_names=trait_names,
        seed=seed,
        protocol=protocol,
        candidate_contract=contract,
        replay_artifact_dir=out_dir,
    )
    if loss_weight_signature(frame) != authoritative_weight_sha256:
        raise ValueError("FA fit mutated authoritative row weights")
    if fit.fa_variable_count != 3 or fit.fa_parameter_count <= 0:
        raise ValueError("FA component does not expose the expected parameters")

    active_training = np.ones(len(training), dtype=bool)
    active_validation = np.ones(len(validation), dtype=bool)
    base_calibration = fit_positive_calibration(
        training,
        fit.training_prediction,
        active_training,
        trait_names,
        protocol,
    )
    calibration, crossfit = fit_test_weight_calibration(
        training,
        fit.training_prediction,
        active_training,
        trait_names,
        base_calibration,
        protocol,
        candidate=CALIBRATION_CANDIDATE,
        target_trait="TEST_WEIGHT",
    )
    prediction = apply_calibration(
        validation,
        fit.validation_prediction,
        active_validation,
        calibration,
    )
    tf.debugging.assert_all_finite(prediction, "Non-finite calibrated FA prediction")
    np.save(out_dir / "validation_predictions_calibrated.npy", prediction, allow_pickle=False)
    trait_metrics, subset_metrics, guard_metrics, summary = validation_metrics(
        validation, prediction, scaling, masks, candidate
    )
    fa_inventory = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "factor_analytic_rank": fit.fa_rank,
                "factor_analytic_variable_count": fit.fa_variable_count,
                "factor_analytic_parameter_count": fit.fa_parameter_count,
                "factor_analytic_active_training_rows": fit.fa_active_training_rows,
                "factor_analytic_active_validation_rows": fit.fa_active_validation_rows,
                "environment_feature_count": 153,
                "environment_loading_link": "linear",
                "free_environment_loadings": False,
                "projection_inactive_policy": "exact_zero_FA_residual",
                "authoritative_row_mass_changed": False,
                "historical_backbone_changed": False,
                "hierarchy_changed": False,
            }
        ]
    )

    tables = {
        "trait_scaling.tsv": scaling,
        "authoritative_row_mass_diagnostics.tsv": mass_diagnostics,
        "factor_analytic_parameter_inventory.tsv": fa_inventory,
        "training_only_calibration.tsv": calibration,
        "training_only_calibration_crossfit.tsv": crossfit,
        "validation_trait_metrics.tsv": trait_metrics,
        "validation_subset_metrics.tsv": subset_metrics,
        "validation_guard_metrics.tsv": guard_metrics,
        "component_epoch_history.tsv": fit.epoch_history,
        "active_component_factors.tsv": fit.factor_inventory,
        "trial_environment_hierarchy_support.tsv": fit.hierarchy_support,
    }
    for name, table in tables.items():
        table.to_csv(out_dir / name, sep="\t", index=False, lineterminator="\n")

    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    target_calibration = calibration.loc[
        calibration["trait_name_canonical"].eq("TEST_WEIGHT")
    ].iloc[0]
    artifact_hashes = {name: sha256_file(out_dir / name) for name in tables}
    artifact_hashes["validation_predictions_calibrated.npy"] = sha256_file(
        out_dir / "validation_predictions_calibrated.npy"
    )
    artifact_hashes.update(fit.replay_artifacts)
    metadata: dict[str, object] = {
        "status": "PASS",
        "protocol_version": RUN_PROTOCOL,
        "stage1_version": "Stage-1 v2",
        "selection_data": "nested_inner_validation_only",
        "state_id": state_id,
        "scenario": state.scenario,
        "outer_fold": int(state.outer_fold),
        "inner_fold": int(state.inner_fold),
        "candidate": candidate,
        "factor_analytic_rank": fit.fa_rank,
        "factor_analytic_variable_count": fit.fa_variable_count,
        "factor_analytic_parameter_count": fit.fa_parameter_count,
        "factor_analytic_active_training_rows": fit.fa_active_training_rows,
        "factor_analytic_active_validation_rows": fit.fa_active_validation_rows,
        "projection_feature_count": 153,
        "free_environment_loadings": False,
        "only_mutable_component": "covariate_linked_factor_analytic_residual",
        "authoritative_row_mass_changed": False,
        "historical_backbone_changed": False,
        "hierarchy_changed": False,
        "configuration_label": configuration_label,
        "configuration": protocol["fixed_configuration"],
        "seed": int(seed),
        "training_rows": len(training),
        "validation_rows": len(validation),
        "training_gid_signature": role_metadata["training_gid_signature"],
        "training_environment_signature": role_metadata[
            "training_environment_signature"
        ],
        "validation_observation_signature": identifier_signature(
            validation["phase4_adjusted_row_id"].astype(str)
        ),
        "authoritative_loss_weight_sha256": authoritative_weight_sha256,
        "test_weight_calibration_method": target_calibration["method"],
        "test_weight_calibration_slope": float(target_calibration["slope"]),
        "test_weight_calibration_intercept": float(target_calibration["intercept"]),
        "test_weight_crossfit_valid_folds": int(
            target_calibration["crossfit_valid_folds"]
        ),
        "component_best_validation_nrmse": {
            "historical_plus_covariate_linked_FA": fit.best_metric
        },
        "component_epochs_completed": {
            "historical_plus_covariate_linked_FA": fit.epochs_completed
        },
        **summary,
        "model_training_performed": True,
        "tensorflow_op_determinism_enabled": True,
        "deterministic_tf_data_order": True,
        "per_batch_finite_assertions_enabled": True,
        "factor_cache_expected_state_validation": True,
        "checkpoint_and_prediction_replay_artifacts_persisted": True,
        "runtime_thread_configuration_sha256": runtime_thread_configuration_sha256,
        "file_access_attestation_scope": "controlled_process_only",
        "os_level_complete_file_open_audit_performed": False,
        "phenotype_values_read": True,
        "inner_validation_metrics_read": True,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "future_SSP_values_read": False,
        "future_covariate_matrices_used": 0,
        "future_predictions_generated": 0,
        "code_commit": git_commit(code_root),
        "trainer_sha256": sha256_file(code_root / TRAINER),
        "calibration_helper_sha256": sha256_file(code_root / CALIBRATION_HELPER),
        "calibration_trainer_sha256": sha256_file(code_root / CALIBRATION_TRAINER),
        "remediation_helper_sha256": sha256_file(code_root / REMEDIATION_HELPER),
        "model_builder_sha256": sha256_file(code_root / MODEL_BUILDER),
        "factor_builder_sha256": sha256_file(code_root / FACTOR_BUILDER),
        "trainer_interface_sha256": sha256_file(code_root / TRAINER_INTERFACE),
        "protocol_sha256": sha256_file(code_root / PROTOCOL),
        "post_hierarchy_plan_sha256": sha256_file(code_root / POST_HIERARCHY_PLAN),
        "projection_protocol_sha256": sha256_file(code_root / PROJECTION_PROTOCOL),
        "private_head_decision_sha256": sha256_file(root / PRIVATE_HEAD_DECISION),
        "artifacts": artifact_hashes,
    }
    write_json(out_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one Stage-1 v2 covariate-linked FA candidate"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    train_factor_analytic_run(
        root=args.root,
        state_id=args.state_id,
        candidate=args.candidate,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
