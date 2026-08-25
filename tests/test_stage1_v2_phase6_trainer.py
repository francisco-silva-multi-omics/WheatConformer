from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2.run_stage1_v2_phase6_phase1 import (
    metadata_matches,
    phase1_grid,
    recommended_cpu_parallelism,
)
from server_training_pipeline.stage1_v2_trainer_interface import load_selection_protocol
from server_training_pipeline.train_stage1_v2_phase6_tf import (
    FactorBlock,
    Stage1V2ReactionNorm,
    centered_marker_random_features,
    reporting_subset_masks,
    state_role_masks,
    validation_reporting_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_phase1_grid_is_exact_and_uses_matched_fold_seeds() -> None:
    grid = phase1_grid(load_selection_protocol(ROOT))
    assert len(grid) == 120
    assert set(grid["state_id"]) == {
        f"GNEW_EOBS__OUTER1__INNER{fold}" for fold in range(1, 6)
    }
    assert grid.groupby("inner_fold")["seed"].nunique().eq(1).all()
    assert grid.groupby(["candidate", "configuration_label"]).size().eq(5).all()


def test_server_cpu_parallelism_is_bounded_by_physical_cores() -> None:
    assert recommended_cpu_parallelism(10) == (4, 2)
    assert recommended_cpu_parallelism(20) == (4, 5)
    workers, threads = recommended_cpu_parallelism(40)
    assert workers == 6
    assert workers * threads <= 40


def test_resume_rejects_superseded_execution_protocol(tmp_path: Path) -> None:
    row = pd.Series(
        {
            "state_id": "GNEW_EOBS__OUTER1__INNER1",
            "candidate": "ka_identity_location_baseline",
            "configuration_label": "compact",
            "seed": 62111,
        }
    )
    metadata = {
        "status": "PASS",
        **row.to_dict(),
        "code_commit": "commit",
        "selection_protocol_sha256": "selection",
        "trainer_sha256": "trainer",
        "execution_protocol_sha256": "old-execution",
        "execution_backend": "wsl_gpu",
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
    }
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert not metadata_matches(
        path,
        row,
        commit="commit",
        protocol_sha="selection",
        trainer_sha="trainer",
        execution_protocol_sha="new-execution",
        runtime_mode="server_cpu",
    )


def test_gnew_eobs_roles_are_disjoint_and_environment_observed() -> None:
    observations = pd.DataFrame(
        {
            "canonical_gid": ["g1", "g2", "g1", "g2"],
            "environment_id": ["e1", "e1", "e2", "e3"],
            "gnew_eobs_outer1_role": ["TRAIN", "TRAIN", "TEST", "TRAIN"],
        }
    )
    training, validation, embargo = state_role_masks(
        observations,
        scenario="GNEW_EOBS",
        outer_fold=1,
        inner_fold=1,
        training_gids={"g1"},
        training_environments={"e1", "e2"},
    )
    assert training.tolist() == [True, False, False, False]
    assert validation.tolist() == [False, True, False, False]
    assert embargo.tolist() == [False, False, False, True]


def test_gobs_enew_roles_require_an_observed_genotype() -> None:
    observations = pd.DataFrame(
        {
            "canonical_gid": ["g1", "g1", "g2"],
            "environment_id": ["e1", "e2", "e2"],
            "gobs_enew_outer1_role": ["TRAIN"] * 3,
        }
    )
    training, validation, embargo = state_role_masks(
        observations,
        scenario="GOBS_ENEW",
        outer_fold=1,
        inner_fold=1,
        training_gids={"g1"},
        training_environments={"e1"},
    )
    assert training.tolist() == [True, False, False]
    assert validation.tolist() == [False, True, False]
    assert embargo.tolist() == [False, False, True]


def test_temporal_roles_use_phase5_normalized_cycle_years() -> None:
    observations = pd.DataFrame(
        {
            "canonical_gid": ["g1", "g2", "g3", "g4"],
            "environment_id": ["e1", "e2", "e3", "e4"],
            "year": ["79-80", "80-81", "81-82", "1983"],
            "temporal_year_outer1_role": ["TRAIN"] * 4,
        }
    )
    assignments = pd.DataFrame(
        {
            "scenario": ["TEMPORAL_YEAR"] * 4,
            "outer_fold": ["1"] * 4,
            "inner_fold": ["1"] * 4,
            "entity_type": ["NORMALIZED_YEAR"] * 4,
            "entity_id": ["1980", "1981", "1982", "1983"],
            "assignment": [
                "TRAIN",
                "EMBARGO_ONE_YEAR",
                "INNER_VALIDATION_ID_ONLY",
                "NOT_AVAILABLE",
            ],
        }
    )
    training, validation, embargo = state_role_masks(
        observations,
        scenario="TEMPORAL_YEAR",
        outer_fold=1,
        inner_fold=1,
        training_gids={"g1"},
        training_environments={"e1"},
        assignments=assignments,
    )
    assert training.tolist() == [True, False, False, False]
    assert validation.tolist() == [False, False, True, False]
    assert embargo.tolist() == [False, True, False, True]


def test_marker_random_features_are_deterministic_and_impute_training_mean() -> None:
    dosage = np.array([[0, 1, 255], [2, 255, 0]], dtype=np.uint8)
    kwargs = {
        "marker_indices": np.array([0, 1]),
        "allele_frequency": np.array([0.25, 0.5]),
        "denominator": 1.75,
        "rank": 4,
        "seed": 42,
        "marker_major": False,
    }
    first = centered_marker_random_features(dosage, **kwargs)
    second = centered_marker_random_features(dosage, **kwargs)
    assert first.shape == (2, 4)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_stage1_v2_model_forward_and_gradient_are_finite() -> None:
    import tensorflow as tf

    genotype = (
        FactorBlock(
            name="K_A",
            axis="genotype",
            entity_ids=np.array(["g1", "g2"]),
            values=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            available=np.ones(2, dtype=bool),
        ),
    )
    environment = (
        FactorBlock(
            name="K_E",
            axis="environment",
            entity_ids=np.array(["e1", "e2"]),
            values=np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.float32),
            available=np.ones(2, dtype=bool),
        ),
    )
    model = Stage1V2ReactionNorm(
        genotype=genotype,
        environment=environment,
        reaction_design=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        trait_names=["T1", "T2"],
        latent_dim=3,
        reaction_rank=2,
        residual_floor=0.05,
        weight_decay=1e-4,
        seed=17,
    )
    inputs = (
        tf.constant([[0], [1]], dtype=tf.int32),
        tf.constant([[0], [1]], dtype=tf.int32),
        tf.constant([0, 1], dtype=tf.int32),
        tf.constant([0, 1], dtype=tf.int32),
    )
    target = tf.constant([0.25, -0.25], dtype=tf.float32)
    with tf.GradientTape() as tape:
        prediction = model(inputs, training=True)
        scale = tf.gather(model.residual_scales(), inputs[3])
        loss = tf.reduce_mean(
            0.5 * tf.square((prediction - target) / scale) + tf.math.log(scale)
        ) + model.regularization_loss()
    gradients = tape.gradient(loss, model.trainable_variables)
    assert np.isfinite(prediction.numpy()).all()
    assert np.isfinite(loss.numpy())
    assert all(gradient is not None for gradient in gradients)
    assert all(
        np.isfinite(tf.convert_to_tensor(gradient).numpy()).all()
        for gradient in gradients
    )


def test_guard_masks_use_direct_marker_support_and_candidate_independent_projection() -> None:
    frame = pd.DataFrame(
        {
            "canonical_gid": ["g1", "g2", "g3", "g4"],
            "environment_id": ["active", "inactive", "active", "inactive"],
            "pedigree_available": [True, True, False, False],
        }
    )
    masks = reporting_subset_masks(
        frame,
        marker_gids={"g2", "g3"},
        projection_active_environments={"active"},
    )
    assert masks["PEDIGREE_ONLY"].tolist() == [True, False, False, False]
    assert masks["MARKER_SUPPORTED"].tolist() == [False, True, True, False]
    assert masks["PEDIGREE_AND_MARKER"].tolist() == [False, True, False, False]
    assert masks["NEITHER_PRODUCTION_PEDIGREE_NOR_DENSE_MARKERS"].tolist() == [
        False,
        False,
        False,
        True,
    ]
    assert masks["PROJECTION_CORE_ACTIVE"].tolist() == [True, False, True, False]


def test_guard_metrics_write_exact_observation_signatures() -> None:
    frame = pd.DataFrame(
        {
            "phase4_adjusted_row_id": ["row-b", "row-a"],
            "canonical_gid": ["g1", "g2"],
            "environment_id": ["e1", "e1"],
            "trait": ["T1", "T1"],
            "adjusted_value": [1.0, 2.0],
            "prediction": [1.1, 1.9],
        }
    )
    scaling = pd.DataFrame(
        {"trait_name_canonical": ["T1"], "training_weighted_sd": [2.0]}
    )
    masks = {
        "candidate": {
            "MARKER_SUPPORTED": pd.Series([True, True]),
            "PEDIGREE_ONLY": pd.Series([False, False]),
        }
    }
    metrics = validation_reporting_metrics(frame, scaling, masks)
    assert len(metrics) == 2
    marker = metrics.loc[metrics["subset"].eq("MARKER_SUPPORTED")].iloc[0]
    assert marker["rows"] == 2
    import hashlib

    expected = hashlib.sha256(b"row-a\nrow-b\n").hexdigest()
    assert marker["observation_id_signature"] == expected
