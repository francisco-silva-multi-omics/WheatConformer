from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from server_training_pipeline.train_stage1_v2_phase6_hierarchy_calibration_tf import (  # noqa: E402
    effective_protocol,
)
from server_training_pipeline.train_stage1_v2_phase6_remediation_tf import (  # noqa: E402
    trait_regularization_vectors,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "server_training_pipeline/stage1_v2_phase6_hierarchy_calibration_protocol_v1.json"
)


def test_strong_head_changes_only_preregistered_test_weight_regularization() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    traits = [*protocol["primary_traits"], *protocol["exploratory_traits"]]
    base = effective_protocol(
        protocol, "hierarchy_test_weight_group_crossfit_calibration_v1"
    )
    strong = effective_protocol(
        protocol, "hierarchy_test_weight_group_crossfit_strong_head_v1"
    )
    base_floor, base_penalty = trait_regularization_vectors(base, traits)
    strong_floor, strong_penalty = trait_regularization_vectors(strong, traits)
    target = traits.index("TEST_WEIGHT")
    assert np.isclose(base_floor[target], 0.1)
    assert np.isclose(base_penalty[target], 4.0)
    assert np.isclose(strong_floor[target], 0.15)
    assert np.isclose(strong_penalty[target], 8.0)
    other = np.arange(len(traits)) != target
    assert np.array_equal(base_floor[other], strong_floor[other])
    assert np.array_equal(base_penalty[other], strong_penalty[other])
