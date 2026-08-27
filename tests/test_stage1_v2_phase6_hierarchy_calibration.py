from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v2.run_stage1_v2_phase6_hierarchy_calibration import (
    MASK_CANDIDATE,
    pair_corrected_guards,
)
from server_training_pipeline.stage1_v2_phase6_remediation import (
    fit_group_crossfitted_trait_calibration,
    fit_positive_calibration,
    set_trait_identity_calibration,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "server_training_pipeline/stage1_v2_phase6_hierarchy_calibration_protocol_v1.json"
)
SOURCE_PROTOCOL_PATH = (
    ROOT / "server_training_pipeline/stage1_v2_phase6_remediation_protocol_v1.json"
)
SERVER_RUNNER_PATH = (
    ROOT / "scripts/v2/run_stage1_v2_phase6_hierarchy_calibration_server_cpu.sh"
)
SERVER_TEST_REQUIREMENTS_PATH = (
    ROOT / "requirements/stage1_v2_server_cpu_test_addons.txt"
)


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_protocol_keeps_architecture_optimizer_and_acceptance_fixed() -> None:
    value = protocol()
    source = json.loads(SOURCE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert value["protocol_version"] == "stage1_v2_phase6_hierarchy_calibration_v1"
    assert value["architecture_policy"]["fixed_candidate"] == (
        "known_environment_hierarchical_v2"
    )
    assert value["fixed_configuration"]["batch_size"] == 8192
    assert value["fixed_configuration"]["learning_rate"] == 0.001
    assert value["architecture_policy"]["batch_size_screen_performed"] is False
    assert value["phase_1"]["new_training_run_count"] == 15
    assert value["phase_1_acceptance"] == source["phase_1_acceptance"]
    assert value["outer_test_metrics_read"] is False
    assert value["final_holdout_outcomes_read"] is False
    assert value["full_confirmation_policy"]["automatic_launch"] is False


def test_server_full_suite_runs_from_code_root_with_pinned_addons() -> None:
    runner = SERVER_RUNNER_PATH.read_text(encoding="utf-8")
    requirements = SERVER_TEST_REQUIREMENTS_PATH.read_text(encoding="utf-8")
    assert 'cd "$CODE_ROOT"' in runner
    assert '"$PYTHON_BIN" -m pytest -q "$CODE_ROOT/tests"' in runner
    assert "validate_runtime(Path.cwd(), \"server_cpu\")" in runner
    for requirement in (
        "xarray==2026.7.0",
        "lxml==6.0.2",
        "cftime==1.6.5",
        "affine==2.4.0",
        "rasterio==1.4.3",
        "netCDF4==1.7.4",
    ):
        assert requirement in requirements


def calibration_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = 2500
    row = np.arange(rows)
    prediction = np.sin(row / 31.0).astype(np.float32)
    target = 0.15 + 1.25 * prediction + 0.02 * np.cos(row / 7.0)
    frame = pd.DataFrame(
        {
            "trait_index": np.zeros(rows, dtype=np.int32),
            "trait": ["TEST_WEIGHT"] * rows,
            "y_scaled": target,
            "loss_weight": np.ones(rows),
            "trial_id": [f"TRIAL_{value % 100:03d}" for value in row],
            "environment_id": [f"ENV_{value % 50:03d}" for value in row],
            "phase4_adjusted_row_id": [f"ROW_{value:05d}" for value in row],
        }
    )
    return frame, prediction, np.ones(rows, dtype=bool)


def test_group_crossfit_calibration_is_training_only_and_complete() -> None:
    frame, prediction, active = calibration_frame()
    calibration, evidence = fit_group_crossfitted_trait_calibration(
        frame,
        prediction,
        active,
        ["TEST_WEIGHT"],
        protocol(),
        target_trait="TEST_WEIGHT",
    )
    row = calibration.iloc[0]
    assert row["status"] == "FITTED_GROUP_CROSSFIT_MEDIAN"
    assert row["method"] == "group_crossfit_median"
    assert int(row["crossfit_valid_folds"]) == 5
    assert 0.05 <= float(row["slope"]) <= 2.0
    assert len(evidence) == 5
    assert evidence["status"].eq("PASS").all()
    assert not evidence["validation_values_used"].astype(bool).any()
    assert evidence["heldout_rows"].gt(0).all()


def test_identity_candidate_changes_only_target_trait_calibration() -> None:
    frame, prediction, active = calibration_frame()
    other = frame.iloc[:600].copy()
    other["trait_index"] = 1
    other["trait"] = "GRAIN_YIELD"
    combined = pd.concat([frame, other], ignore_index=True)
    combined_prediction = np.concatenate([prediction, prediction[:600]])
    calibration = fit_positive_calibration(
        combined,
        combined_prediction,
        np.ones(len(combined), dtype=bool),
        ["TEST_WEIGHT", "GRAIN_YIELD"],
        protocol(),
    )
    original_other = calibration.loc[
        calibration["trait_name_canonical"].eq("GRAIN_YIELD"), "slope"
    ].iloc[0]
    identity = set_trait_identity_calibration(calibration, "TEST_WEIGHT")
    target = identity.loc[identity["trait_name_canonical"].eq("TEST_WEIGHT")].iloc[0]
    assert target["intercept"] == 0.0
    assert target["slope"] == 1.0
    assert target["status"] == "IDENTITY_PREREGISTERED"
    assert identity.loc[
        identity["trait_name_canonical"].eq("GRAIN_YIELD"), "slope"
    ].iloc[0] == original_other


def test_corrected_guard_pairing_uses_one_frozen_marker_mask() -> None:
    rows = []
    for candidate, nrmse in (
        ("historical_reaction_reference", 1.0),
        ("candidate", 0.9),
    ):
        rows.append(
            {
                "state_id": "STATE",
                "scenario": "GNEW_EOBS",
                "candidate": candidate,
                "mask_candidate": MASK_CANDIDATE,
                "subset": "MARKER_SUPPORTED",
                "rows": 1000,
                "unique_genotypes": 100,
                "unique_environments": 10,
                "trait_count": 7,
                "observation_id_signature": "same-signature",
                "normalized_rmse_macro": nrmse,
                "pearson_macro": 0.5,
            }
        )
    paired = pair_corrected_guards(pd.DataFrame(rows))
    candidate = paired.loc[paired["candidate"].eq("candidate")].iloc[0]
    assert candidate["mask_candidate"] == MASK_CANDIDATE
    assert np.isclose(candidate["relative_nrmse_gain"], 0.1)
    assert candidate["rows"] == candidate["rows_reference"]
