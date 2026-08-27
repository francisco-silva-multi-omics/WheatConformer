from __future__ import annotations

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from server_training_pipeline.train_stage1_v2_phase6_remediation_tf import (  # noqa: E402
    HierarchicalReactionNorm,
)
from server_training_pipeline.train_stage1_v2_phase6_tf import FactorBlock  # noqa: E402


def test_hierarchy_model_handles_empty_training_only_support() -> None:
    genotype = FactorBlock(
        name="K_A",
        axis="genotype",
        entity_ids=np.asarray(["G1", "G2"]),
        values=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        available=np.ones(2, dtype=bool),
    )
    environment = FactorBlock(
        name="K_E",
        axis="environment",
        entity_ids=np.asarray(["E1", "E2"]),
        values=np.asarray([[1.0], [-1.0]], dtype=np.float32),
        available=np.ones(2, dtype=bool),
    )
    model = HierarchicalReactionNorm(
        genotype=(genotype,),
        environment=(environment,),
        reaction_design=np.zeros((2, 0), dtype=np.float32),
        trait_names=("GRAIN_YIELD",),
        latent_dim=2,
        reaction_rank=2,
        residual_floor=0.05,
        weight_decay=0.0001,
        seed=7,
        reaction_enabled=False,
        trait_residual_floors=np.asarray([0.05], dtype=np.float32),
        trait_penalty_multipliers=np.asarray([1.0], dtype=np.float32),
        trial_support=np.zeros((0, 1), dtype=bool),
        environment_support=np.zeros((0, 1), dtype=bool),
        trial_penalty=0.01,
        environment_penalty=0.05,
    )
    inputs = (
        np.asarray([[0], [1]], dtype=np.int32),
        np.asarray([[0], [1]], dtype=np.int32),
        np.asarray([-1, -1], dtype=np.int32),
        np.asarray([0, 0], dtype=np.int32),
        np.asarray([-1, -1], dtype=np.int32),
        np.asarray([-1, -1], dtype=np.int32),
    )
    prediction = model(inputs, training=False).numpy()
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()
    assert np.isfinite(float(model.regularization_loss().numpy()))
