from __future__ import annotations

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from server_training_pipeline.train_stage1_v2_phase6_remediation_tf import (  # noqa: E402
    PrivateDecoderHierarchicalReactionNorm,
)
from server_training_pipeline.train_stage1_v2_phase6_tf import FactorBlock  # noqa: E402


def factor(name: str, axis: str) -> FactorBlock:
    return FactorBlock(
        name=name,
        axis=axis,
        entity_ids=np.asarray(["a", "b"]),
        values=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        available=np.asarray([True, True]),
    )


def build_model(mode: str) -> PrivateDecoderHierarchicalReactionNorm:
    return PrivateDecoderHierarchicalReactionNorm(
        genotype=[factor("g", "genotype")],
        environment=[factor("e", "environment")],
        reaction_design=np.asarray([[1.0], [2.0]], dtype=np.float32),
        trait_names=["DAYS_TO_HEADING", "DAYS_TO_MATURITY"],
        latent_dim=3,
        reaction_rank=2,
        residual_floor=0.05,
        weight_decay=0.0001,
        seed=17,
        reaction_enabled=True,
        trait_residual_floors=np.asarray([0.05, 0.05], dtype=np.float32),
        trait_penalty_multipliers=np.asarray([1.0, 1.0], dtype=np.float32),
        trial_support=np.zeros((0, 2), dtype=bool),
        environment_support=np.zeros((0, 2), dtype=bool),
        trial_penalty=0.01,
        environment_penalty=0.05,
        decoder_mode=mode,
        trait_family_indices=np.asarray([0, 0], dtype=np.int32),
        trait_private_penalty_multiplier=4.0,
        family_penalty_multiplier=1.0,
    )


def inputs() -> tuple[tf.Tensor, ...]:
    return (
        tf.constant([[0], [1]], dtype=tf.int32),
        tf.constant([[0], [1]], dtype=tf.int32),
        tf.constant([0, 1], dtype=tf.int32),
        tf.constant([0, 1], dtype=tf.int32),
        tf.constant([-1, -1], dtype=tf.int32),
        tf.constant([-1, -1], dtype=tf.int32),
    )


def test_trait_private_residuals_start_at_zero_and_are_active() -> None:
    model = build_model("trait_private_residual")
    baseline = model(inputs(), training=False).numpy()
    private = [
        value
        for value in model.trainable_variables
        if "private_trait_decoder_" in value.name
    ]
    family = [
        value for value in model.trainable_variables if "family_decoder_" in value.name
    ]
    assert private
    assert not family
    assert all(np.array_equal(value.numpy(), np.zeros(value.shape)) for value in private)

    model.private_genotype_loadings[0].assign(
        np.ones(model.private_genotype_loadings[0].shape, dtype=np.float32)
    )
    changed = model(inputs(), training=False).numpy()
    assert not np.allclose(changed, baseline)
    assert np.isfinite(float(model.regularization_loss().numpy()))


def test_family_candidate_contains_family_and_trait_residuals() -> None:
    model = build_model("family_shared_trait_private_residual")
    model(inputs(), training=False)
    private = [
        value
        for value in model.trainable_variables
        if "private_trait_decoder_" in value.name
    ]
    family = [
        value for value in model.trainable_variables if "family_decoder_" in value.name
    ]
    assert private
    assert family
    assert model.family_count == 1
    assert all(np.array_equal(value.numpy(), np.zeros(value.shape)) for value in family)
