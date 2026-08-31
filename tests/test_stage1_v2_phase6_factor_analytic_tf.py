from __future__ import annotations

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from server_training_pipeline.train_stage1_v2_phase6_factor_analytic_tf import (  # noqa: E402
    CovariateLinkedFactorAnalyticHierarchicalReactionNorm,
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


def build_model(rank: int = 2) -> CovariateLinkedFactorAnalyticHierarchicalReactionNorm:
    design = np.zeros((2, 153), dtype=np.float32)
    design[:, 0] = [1.0, 2.0]
    return CovariateLinkedFactorAnalyticHierarchicalReactionNorm(
        genotype=[block("K_A_CANONICAL_V3", "genotype")],
        environment=[block("K_E_WEATHER", "environment")],
        reaction_design=np.zeros((2, 0), dtype=np.float32),
        trait_names=["DAYS_TO_HEADING", "DAYS_TO_MATURITY"],
        latent_dim=3,
        reaction_rank=2,
        residual_floor=0.05,
        weight_decay=0.0001,
        seed=17,
        reaction_enabled=False,
        trait_residual_floors=np.asarray([0.05, 0.05], dtype=np.float32),
        trait_penalty_multipliers=np.asarray([1.0, 1.0], dtype=np.float32),
        trial_support=np.zeros((0, 2), dtype=bool),
        environment_support=np.zeros((0, 2), dtype=bool),
        trial_penalty=0.01,
        environment_penalty=0.05,
        projection_design=design,
        projection_available=np.asarray([True, False]),
        factor_analytic_rank=rank,
        factor_penalty_multiplier=1.0,
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


def test_projection_inactive_rows_receive_exactly_zero_FA_residual() -> None:
    model = build_model()
    model(inputs(), training=False)
    for value in model.factor_analytic_variables:
        value.assign(np.zeros(value.shape, dtype=np.float32))
    baseline = model(inputs(), training=False).numpy()
    for value in model.factor_analytic_variables:
        value.assign(np.ones(value.shape, dtype=np.float32))
    changed = model(inputs(), training=False).numpy()
    assert changed[0] != baseline[0]
    assert changed[1] == baseline[1]


def test_FA_component_has_exactly_three_low_rank_parameter_blocks() -> None:
    model = build_model(rank=4)
    model(inputs(), training=False)
    variables = model.factor_analytic_variables
    assert len(variables) == 3
    assert variables[0].shape == (2, 4)
    assert variables[1].shape == (153, 4)
    assert variables[2].shape == (4, 2)
    assert np.isfinite(float(model.regularization_loss().numpy()))


def test_FA_model_rejects_a_noncertified_feature_schema() -> None:
    with pytest.raises(ValueError, match="153 columns"):
        CovariateLinkedFactorAnalyticHierarchicalReactionNorm(
            genotype=[block("K_A_CANONICAL_V3", "genotype")],
            environment=[block("K_E_WEATHER", "environment")],
            reaction_design=np.zeros((2, 0), dtype=np.float32),
            trait_names=["DAYS_TO_HEADING"],
            latent_dim=2,
            reaction_rank=2,
            residual_floor=0.05,
            weight_decay=0.0001,
            seed=17,
            reaction_enabled=False,
            trait_residual_floors=np.asarray([0.05], dtype=np.float32),
            trait_penalty_multipliers=np.asarray([1.0], dtype=np.float32),
            trial_support=np.zeros((0, 1), dtype=bool),
            environment_support=np.zeros((0, 1), dtype=bool),
            trial_penalty=0.01,
            environment_penalty=0.05,
            projection_design=np.zeros((2, 152), dtype=np.float32),
            projection_available=np.asarray([True, True]),
            factor_analytic_rank=2,
            factor_penalty_multiplier=1.0,
        )
