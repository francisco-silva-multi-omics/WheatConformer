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
    state_role_masks,
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
