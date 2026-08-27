from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from server_training_pipeline.stage1_v2_phase6_remediation import (
    add_hierarchy_indices,
    apply_calibration,
    fit_positive_calibration,
)


REFERENCE = "historical_reaction_reference"
HIERARCHY = "known_environment_hierarchical_v2"
PROJECTION_ROUTE = "projection_output_routed_calibrated_v2"
MARKER_ROUTE = "marker_supported_output_routed_v2"


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "server_training_pipeline/stage1_v2_phase6_remediation_protocol_v1.json"
)


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_remediation_protocol_freezes_bounded_structural_grid() -> None:
    value = protocol()
    assert value["protocol_version"] == "stage1_v2_phase6_structural_remediation_v1"
    assert value["phase_1"]["candidate_state_count"] == 70
    assert value["outer_test_metrics_read"] is False
    assert value["final_holdout_outcomes_read"] is False
    assert value["phase_2_optimizer_screen"]["status"] == (
        "blocked_until_phase_1_candidate_acceptance"
    )
    assert value["candidates"][HIERARCHY]["eligible_scenarios"] == ["GNEW_EOBS"]
    assert value["candidates"][PROJECTION_ROUTE]["output_route"] == (
        "projection_active_else_historical_reference"
    )
    assert value["candidates"][MARKER_ROUTE]["output_route"] == (
        "marker_supported_else_historical_reference"
    )
    assert value["candidates"][REFERENCE]["positive_training_calibration"] is False


def test_positive_calibration_is_training_only_and_preserves_inactive_rows() -> None:
    value = protocol()
    rows = 600
    prediction = np.linspace(-1.0, 1.0, rows, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "trait_index": np.zeros(rows, dtype=np.int32),
            "trait": ["1000_GRAIN_WEIGHT"] * rows,
            "y_scaled": 0.25 + 1.5 * prediction,
            "loss_weight": np.ones(rows),
        }
    )
    traits = ["1000_GRAIN_WEIGHT"]
    active = np.ones(rows, dtype=bool)
    calibration = fit_positive_calibration(
        frame, prediction, active, traits, value
    )
    row = calibration.iloc[0]
    assert row["validation_values_used"] in {False, np.bool_(False)}
    assert 0.05 <= float(row["slope"]) <= 2.0
    routed_active = np.zeros(rows, dtype=bool)
    routed_active[::2] = True
    calibrated = apply_calibration(
        frame, prediction, routed_active, calibration
    )
    assert np.array_equal(calibrated[~routed_active], prediction[~routed_active])
    assert not np.array_equal(calibrated[routed_active], prediction[routed_active])


def test_hierarchy_axes_are_fit_from_training_rows_only() -> None:
    value = protocol()
    trait_names = [*value["primary_traits"], *value["exploratory_traits"]]
    training = pd.DataFrame(
        {
            "selection_role": ["TRAINING"] * 20,
            "loss_weight": np.ones(20),
            "trial_id": ["TRIAL_TRAIN"] * 20,
            "environment_id": ["ENV_TRAIN"] * 20,
            "trait": ["1000_GRAIN_WEIGHT"] * 20,
        }
    )
    validation = pd.DataFrame(
        {
            "selection_role": ["INNER_VALIDATION"],
            "loss_weight": [1.0],
            "trial_id": ["TRIAL_VALIDATION_ONLY"],
            "environment_id": ["ENV_VALIDATION_ONLY"],
            "trait": ["1000_GRAIN_WEIGHT"],
        }
    )
    frame, trial_support, environment_support, support = add_hierarchy_indices(
        pd.concat([training, validation], ignore_index=True), trait_names, value
    )
    assert trial_support.shape == (1, len(trait_names))
    assert environment_support.shape == (1, len(trait_names))
    assert frame.loc[20, "trial_hierarchy_index"] == -1
    assert frame.loc[20, "environment_hierarchy_index"] == -1
    assert not support["entity_id"].eq("TRIAL_VALIDATION_ONLY").any()
    assert not support["entity_id"].eq("ENV_VALIDATION_ONLY").any()
