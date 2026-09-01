from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest


tf = pytest.importorskip("tensorflow")

from server_training_pipeline.train_stage1_v2_phase6_factor_analytic_optimization_amendment_tf import (  # noqa: E402
    NormalizedDirectionFactorAnalyticHierarchicalReactionNorm,
    _activity_summary,
    _validated_recovered_component_fit,
    primary_macro_nrmse,
)
from server_training_pipeline.train_stage1_v2_phase6_tf import FactorBlock  # noqa: E402


def block(name: str, axis: str) -> FactorBlock:
    return FactorBlock(
        name=name,
        axis=axis,
        entity_ids=np.asarray(["a", "b"]),
        values=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        available=np.asarray([True, True]),
        state_hash=f"{name}-state",
    )


def build_model(rank: int = 2) -> NormalizedDirectionFactorAnalyticHierarchicalReactionNorm:
    design = np.zeros((2, 153), dtype=np.float32)
    design[:, :2] = [[1.0, 0.0], [0.0, 1.0]]
    return NormalizedDirectionFactorAnalyticHierarchicalReactionNorm(
        genotype=[block("K_A_CANONICAL_V3", "genotype")],
        environment=[block("K_E_WEATHER", "environment")],
        reaction_design=np.zeros((2, 0), dtype=np.float32),
        trait_names=["DAYS_TO_HEADING", "TEST_WEIGHT"],
        latent_dim=3,
        reaction_rank=2,
        residual_floor=0.05,
        weight_decay=0.0001,
        seed=17,
        reaction_enabled=False,
        trait_residual_floors=np.asarray([0.05, 0.1], dtype=np.float32),
        trait_penalty_multipliers=np.asarray([1.0, 4.0], dtype=np.float32),
        trial_support=np.zeros((0, 2), dtype=bool),
        environment_support=np.zeros((0, 2), dtype=bool),
        trial_penalty=0.01,
        environment_penalty=0.05,
        projection_design=design,
        projection_available=np.asarray([True, False]),
        factor_analytic_rank=rank,
        trait_amplitude_penalty_multiplier=1.0,
    )


def inputs() -> tuple[tf.Tensor, ...]:
    return (
        tf.constant([[0], [1]], dtype=tf.int32),
        tf.constant([[0], [1]], dtype=tf.int32),
        tf.constant([-1, -1], dtype=tf.int32),
        tf.constant([0, 1], dtype=tf.int32),
        tf.constant([-1, -1], dtype=tf.int32),
        tf.constant([-1, -1], dtype=tf.int32),
        tf.constant([0, 1], dtype=tf.int32),
    )


def test_directions_are_unit_normalized_and_inactive_projection_is_zero() -> None:
    model = build_model()
    model(inputs(), training=False)
    model.fa_genotype_coefficients.assign(
        np.asarray([[3.0, 0.0], [4.0, 2.0]], dtype=np.float32)
    )
    environment = np.zeros((153, 2), dtype=np.float32)
    environment[:2] = [[6.0, 0.0], [8.0, 5.0]]
    model.fa_environment_coefficients.assign(environment)
    model.fa_trait_loadings.assign(np.ones((2, 2), dtype=np.float32))
    values = model.activity_values()
    np.testing.assert_allclose(values["genotype_normalized_norm"], 1.0)
    np.testing.assert_allclose(values["environment_normalized_norm"], 1.0)
    residual = model.factor_analytic_residual(inputs()).numpy()
    assert residual[0] != 0.0
    assert residual[1] == 0.0


def test_regularization_shrinks_amplitudes_but_not_direction_coordinates() -> None:
    model = build_model()
    model(inputs(), training=False)
    with tf.GradientTape() as tape:
        loss = model.regularization_loss()
    gradients = tape.gradient(loss, model.factor_analytic_variables)
    np.testing.assert_allclose(gradients[0].numpy(), 0.0, atol=1e-8)
    np.testing.assert_allclose(gradients[1].numpy(), 0.0, atol=1e-8)
    assert np.linalg.norm(gradients[2].numpy()) > 0.0


def test_primary_macro_excludes_test_weight_without_dropping_its_rows() -> None:
    frame = pd.DataFrame(
        {
            "trait": ["DAYS_TO_HEADING", "DAYS_TO_HEADING", "TEST_WEIGHT"],
            "trait_index": [0, 0, 1],
            "y_scaled": [0.0, 1.0, 100.0],
        }
    )
    value = primary_macro_nrmse(
        frame,
        np.asarray([0.0, 1.0, -100.0]),
        ["DAYS_TO_HEADING"],
    )
    assert value == pytest.approx(0.0)
    assert len(frame.loc[frame["trait"].eq("TEST_WEIGHT")]) == 1


def test_activity_summary_requires_gradients_but_allows_a_valid_null_final() -> None:
    activity = pd.DataFrame(
        [
            {
                "record_type": "initialization",
                "training_FA_residual_rms": 0.1,
                "validation_FA_residual_rms": 0.1,
                "genotype_gradient_norm": 0.0,
                "environment_gradient_norm": 0.0,
                "trait_amplitude_gradient_norm": 0.0,
                "genotype_raw_direction_norm_min": 0.1,
                "environment_raw_direction_norm_min": 0.1,
                "trait_amplitude_norm_min": 0.1,
                "genotype_normalized_norm_max_error": 0.0,
                "environment_normalized_norm_max_error": 0.0,
            },
            {
                "record_type": "training_epoch",
                "training_FA_residual_rms": 0.01,
                "validation_FA_residual_rms": np.nan,
                "genotype_gradient_norm": 0.1,
                "environment_gradient_norm": 0.1,
                "trait_amplitude_gradient_norm": 0.1,
                "genotype_raw_direction_norm_min": 0.1,
                "environment_raw_direction_norm_min": 0.1,
                "trait_amplitude_norm_min": 0.01,
                "genotype_normalized_norm_max_error": 0.0,
                "environment_normalized_norm_max_error": 0.0,
            },
            {
                "record_type": "selected_checkpoint",
                "training_FA_residual_rms": 0.0,
                "validation_FA_residual_rms": 0.0,
                "genotype_gradient_norm": 0.1,
                "environment_gradient_norm": 0.1,
                "trait_amplitude_gradient_norm": 0.1,
                "genotype_raw_direction_norm_min": 0.1,
                "environment_raw_direction_norm_min": 0.1,
                "trait_amplitude_norm_min": 0.0,
                "genotype_normalized_norm_max_error": 0.0,
                "environment_normalized_norm_max_error": 0.0,
            },
        ]
    )
    summary = _activity_summary(
        activity,
        {
            "activity_certification": {
                "minimum_raw_direction_column_norm": 1e-8,
                "maximum_normalized_direction_norm_error": 1e-5,
                "minimum_observed_gradient_norm_per_FA_tensor": 1e-10,
                "minimum_initial_training_residual_rms": 1e-8,
                "minimum_final_validation_residual_rms_for_performance_eligibility": 1e-8,
            }
        },
    )
    assert summary["FA_optimization_path_certified"] is True
    assert summary["FA_final_component_active"] is False


def test_same_seed_replay_can_force_fresh_training(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("STAGE1_V2_DISABLE_COMPONENT_RECOVERY", "1")
    result = _validated_recovered_component_fit(
        root=tmp_path,
        state_id="GNEW_EOBS__OUTER1__INNER1",
        candidate="normalized_direction_FA_rank2",
        protocol={},
        candidate_contract={},
        replay_artifact_dir=tmp_path / "replay",
        training=pd.DataFrame(),
        validation=pd.DataFrame(),
        genotype=(),
        environment=(),
        genotype_columns=(),
        projection_design=np.empty((0, 153), dtype=np.float32),
        projection_available=np.empty(0, dtype=bool),
        projection_ids=np.empty(0, dtype=str),
        hierarchy_support=pd.DataFrame(),
        trait_names=(),
    )
    assert result is None


def test_completed_v1_postfit_artifacts_are_validated_and_recovered(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("STAGE1_V2_DISABLE_COMPONENT_RECOVERY", raising=False)
    state_id = "GNEW_EOBS__OUTER1__INNER1"
    candidate = "normalized_direction_FA_rank2"
    source = tmp_path / "source_runs" / state_id / candidate
    source.mkdir(parents=True)
    weights = source / "best_model_weights"
    weights.mkdir()

    def digest(path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    weight_rows = []
    arrays = {
        "covariate_linked_fa_genotype_coefficients:0": np.ones(
            (2, 2), dtype=np.float32
        ),
        "covariate_linked_fa_environment_coefficients:0": np.ones(
            (153, 2), dtype=np.float32
        ),
        "covariate_linked_fa_trait_loadings:0": np.ones(
            (2, 2), dtype=np.float32
        ),
    }
    for index, (name, array) in enumerate(arrays.items()):
        relative = f"best_model_weights/weight_{index:04d}.npy"
        path = source / relative
        np.save(path, array, allow_pickle=False)
        weight_rows.append(
            {
                "variable_name": name,
                "shape": json.dumps(list(array.shape)),
                "path": relative,
                "sha256": digest(path),
                "parameter_count": array.size,
            }
        )
    pd.DataFrame(weight_rows).to_csv(
        source / "best_model_weight_manifest.tsv", sep="\t", index=False
    )
    training_ids = np.asarray(["train-1", "train-2"])
    validation_ids = np.asarray(["valid-1"])
    training_prediction = np.asarray([0.1, 0.2], dtype=np.float32)
    validation_prediction = np.asarray([0.3], dtype=np.float32)
    for name, array in {
        "training_observation_ids.npy": training_ids,
        "validation_observation_ids.npy": validation_ids,
        "training_predictions_raw.npy": training_prediction,
        "validation_predictions_raw.npy": validation_prediction,
    }.items():
        np.save(source / name, array, allow_pickle=False)
    replay_rows = []
    for role, ids_name, prediction_name, rows in (
        (
            "training",
            "training_observation_ids.npy",
            "training_predictions_raw.npy",
            2,
        ),
        (
            "validation",
            "validation_observation_ids.npy",
            "validation_predictions_raw.npy",
            1,
        ),
    ):
        replay_rows.append(
            {
                "role": role,
                "rows": rows,
                "observation_signature": "not_used_by_recovery",
                "identifier_path": ids_name,
                "identifier_sha256": digest(source / ids_name),
                "prediction_path": prediction_name,
                "prediction_sha256": digest(source / prediction_name),
            }
        )
    pd.DataFrame(replay_rows).to_csv(
        source / "component_replay_manifest.tsv", sep="\t", index=False
    )
    activity = pd.DataFrame(
        [
            {
                "epoch": 0,
                "record_type": "initialization",
                "validation_primary6_macro_normalized_rmse": np.nan,
                "training_FA_residual_rms": 0.1,
                "validation_FA_residual_rms": 0.1,
                "genotype_gradient_norm": 0.0,
                "environment_gradient_norm": 0.0,
                "trait_amplitude_gradient_norm": 0.0,
                "genotype_raw_direction_norm_min": 0.1,
                "environment_raw_direction_norm_min": 0.1,
                "trait_amplitude_norm_min": 0.1,
                "genotype_normalized_norm_max_error": 0.0,
                "environment_normalized_norm_max_error": 0.0,
            },
            {
                "epoch": 1,
                "record_type": "training_epoch",
                "validation_primary6_macro_normalized_rmse": 0.6,
                "training_FA_residual_rms": 0.05,
                "validation_FA_residual_rms": 0.05,
                "genotype_gradient_norm": 0.1,
                "environment_gradient_norm": 0.1,
                "trait_amplitude_gradient_norm": 0.1,
                "genotype_raw_direction_norm_min": 0.1,
                "environment_raw_direction_norm_min": 0.1,
                "trait_amplitude_norm_min": 0.1,
                "genotype_normalized_norm_max_error": 0.0,
                "environment_normalized_norm_max_error": 0.0,
            },
            {
                "epoch": 1,
                "record_type": "selected_checkpoint",
                "validation_primary6_macro_normalized_rmse": 0.6,
                "training_FA_residual_rms": 0.05,
                "validation_FA_residual_rms": 0.05,
                "genotype_gradient_norm": 0.1,
                "environment_gradient_norm": 0.1,
                "trait_amplitude_gradient_norm": 0.1,
                "genotype_raw_direction_norm_min": 0.1,
                "environment_raw_direction_norm_min": 0.1,
                "trait_amplitude_norm_min": 0.1,
                "genotype_normalized_norm_max_error": 0.0,
                "environment_normalized_norm_max_error": 0.0,
            },
        ]
    )
    activity.to_csv(source / "component_activity_history.tsv", sep="\t", index=False)
    protocol = {
        "postfit_recovery_policy": {
            "protocol_version": "recovery-test-v1",
            "source_run_root": "source_runs",
            "reusable_artifacts": [
                "best_model_weight_manifest.tsv",
                "component_replay_manifest.tsv",
                "component_activity_history.tsv",
                "training_observation_ids.npy",
                "validation_observation_ids.npy",
                "training_predictions_raw.npy",
                "validation_predictions_raw.npy",
            ],
            "source_implementation_version": "v1",
            "source_code_commit": "source-commit",
            "source_failure_class": "postfit-calibration-contract",
        },
        "activity_certification": {
            "minimum_raw_direction_column_norm": 1e-8,
            "maximum_normalized_direction_norm_error": 1e-5,
            "minimum_observed_gradient_norm_per_FA_tensor": 1e-10,
            "minimum_initial_training_residual_rms": 1e-8,
            "minimum_final_validation_residual_rms_for_performance_eligibility": 1e-8,
        },
    }
    training = pd.DataFrame(
        {
            "phase4_adjusted_row_id": training_ids,
            "genotype_index_K_A": [0, 1],
            "fa_environment_index": [0, 1],
        }
    )
    validation = pd.DataFrame(
        {
            "phase4_adjusted_row_id": validation_ids,
            "genotype_index_K_A": [0],
            "fa_environment_index": [0],
        }
    )
    projection_design = np.zeros((2, 153), dtype=np.float32)
    recovered = _validated_recovered_component_fit(
        root=tmp_path,
        state_id=state_id,
        candidate=candidate,
        protocol=protocol,
        candidate_contract={"factor_analytic_rank": 2},
        replay_artifact_dir=tmp_path / "recovered",
        training=training,
        validation=validation,
        genotype=(block("K_A_CANONICAL_V3", "genotype"),),
        environment=(block("K_E_WEATHER", "environment"),),
        genotype_columns=("genotype_index_K_A",),
        projection_design=projection_design,
        projection_available=np.asarray([True, True]),
        projection_ids=np.asarray(["a", "b"]),
        hierarchy_support=pd.DataFrame(),
        trait_names=("DAYS_TO_HEADING", "TEST_WEIGHT"),
    )
    assert recovered is not None
    np.testing.assert_array_equal(
        recovered.validation_prediction, np.asarray([0.3], dtype=np.float32)
    )
    assert recovered.best_metric == pytest.approx(0.6)
    assert (tmp_path / "recovered/component_fit_recovery_provenance.json").is_file()
