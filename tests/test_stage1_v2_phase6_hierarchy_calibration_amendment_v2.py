from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from server_training_pipeline.stage1_v2_phase6_hierarchy_calibration_amendment_v2 import (
    fit_test_weight_calibration,
    non_target_calibration_signature,
)
from server_training_pipeline.stage1_v2_phase6_remediation import (
    fit_positive_calibration,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_hierarchy_calibration_amendment_protocol_v2.json"
)
SOURCE_PROTOCOL_PATH = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_hierarchy_full_confirmation_protocol_v1.json"
)
TRAINER_PATH = (
    ROOT
    / "server_training_pipeline"
    / "train_stage1_v2_phase6_hierarchy_calibration_amendment_tf.py"
)
RUNNER_PATH = (
    ROOT / "scripts" / "v2" / "run_stage1_v2_phase6_hierarchy_calibration_amendment.py"
)
ROUTE_FREEZE_PATH = (
    ROOT / "scripts" / "v2" / "freeze_stage1_v2_phase6_hierarchy_calibration_route.py"
)


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def calibration_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    rows = 4000
    index = np.arange(rows)
    trait = np.where(index < 3000, "TEST_WEIGHT", "GRAIN_YIELD")
    trait_index = np.where(index < 3000, 0, 1)
    prediction = np.sin(index / 29.0) + 0.1 * np.cos(index / 7.0)
    target = np.where(
        index < 3000,
        0.2 + 1.35 * prediction,
        -0.1 + 0.8 * prediction,
    )
    frame = pd.DataFrame(
        {
            "trait_index": trait_index,
            "trait": trait,
            "y_scaled": target,
            "loss_weight": np.ones(rows),
            "environment_id": [f"ENV_{value % 75:03d}" for value in index],
            "trial_id": [f"TRIAL_{value % 120:03d}" for value in index],
            "phase4_adjusted_row_id": [f"ROW_{value:05d}" for value in index],
        }
    )
    return frame, prediction, np.ones(rows, dtype=bool), ["TEST_WEIGHT", "GRAIN_YIELD"]


def base_calibration(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    active: np.ndarray,
    traits: list[str],
) -> pd.DataFrame:
    return fit_positive_calibration(
        frame, prediction, active, traits, protocol()
    )


def test_protocol_changes_only_test_weight_calibration() -> None:
    value = protocol()
    source = json.loads(SOURCE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert value["stage1_version"] == "Stage-1 v2"
    assert value["fixed_configuration"] == source["fixed_configuration"]
    assert value["trial_environment_hierarchy"] == source["trial_environment_hierarchy"]
    assert value["trait_specific_regularization"] == source["trait_specific_regularization"]
    assert value["positive_training_calibration"] == source["positive_training_calibration"]
    assert value["phase_1_acceptance"] == source["phase_1_acceptance"]
    assert value["phase_1_acceptance"]["maximum_absolute_macro_calibration_error"] == 0.2
    assert value["retrospective_threshold_change_allowed"] is False
    assert value["confirmation_scope"]["new_model_fit_count"] == 25
    assert value["confirmation_scope"]["derived_calibration_result_count"] == 75
    assert value["outer_test_policy"]["outer_evaluation_allowed"] is False


def test_identity_ridge_and_huber_are_training_only_positive_calibrators() -> None:
    frame, prediction, active, traits = calibration_data()
    base = base_calibration(frame, prediction, active, traits)
    signatures = set()
    methods = {}
    for candidate in protocol()["confirmation_scope"]["candidate_order"]:
        calibration, evidence = fit_test_weight_calibration(
            frame,
            prediction,
            active,
            traits,
            base,
            protocol(),
            candidate=candidate,
        )
        target = calibration.loc[
            calibration["trait_name_canonical"].eq("TEST_WEIGHT")
        ].iloc[0]
        assert 0.05 <= float(target["slope"]) <= 2.0
        assert not evidence["validation_values_used"].astype(bool).any()
        signatures.add(non_target_calibration_signature(calibration))
        methods[str(target["method"])] = (target, evidence)
    assert len(signatures) == 1
    assert float(methods["identity"][0]["slope"]) == 1.0
    for method in ("environment_oof_affine_ridge", "environment_oof_huber"):
        target, evidence = methods[method]
        assert target["status"] == "FITTED_ENVIRONMENT_OOF_COMPONENTWISE_MEDIAN"
        assert int(target["crossfit_valid_folds"]) == 5
        assert len(evidence) == 5
        assert evidence["status"].eq("PASS").all()
        assert evidence["fit_environment_count"].ge(8).all()
        assert evidence["heldout_environment_count"].ge(1).all()
        assert 1.1 < float(target["slope"]) < 1.6


def test_huber_reduces_outlier_leverage_and_keeps_positive_slope() -> None:
    frame, prediction, active, traits = calibration_data()
    outlier_rows = frame["trait"].eq("TEST_WEIGHT") & frame.index.to_series().mod(97).eq(0)
    frame.loc[outlier_rows, "y_scaled"] += 25.0
    base = base_calibration(frame, prediction, active, traits)
    fitted = {}
    for candidate in (
        "hierarchy_test_weight_environment_oof_ridge_v2",
        "hierarchy_test_weight_environment_oof_huber_v2",
    ):
        calibration, _ = fit_test_weight_calibration(
            frame,
            prediction,
            active,
            traits,
            base,
            protocol(),
            candidate=candidate,
        )
        fitted[candidate] = float(
            calibration.loc[
                calibration["trait_name_canonical"].eq("TEST_WEIGHT"), "slope"
            ].iloc[0]
        )
    ridge = fitted["hierarchy_test_weight_environment_oof_ridge_v2"]
    huber = fitted["hierarchy_test_weight_environment_oof_huber_v2"]
    assert huber > 0
    assert abs(huber - 1.35) < abs(ridge - 1.35)


def test_positive_affine_calibration_preserves_ordering() -> None:
    values = np.array([0.8, -0.2, 1.1, 0.0, 0.4])
    calibrated = -0.3 + 1.4 * values
    assert np.array_equal(np.argsort(values), np.argsort(calibrated))
    assert np.array_equal(
        np.sign(values[:, None] - values[None, :]),
        np.sign(calibrated[:, None] - calibrated[None, :]),
    )


def test_implementation_has_one_fit_per_state_and_separate_route_gate() -> None:
    trainer = TRAINER_PATH.read_text(encoding="utf-8")
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    route = ROUTE_FREEZE_PATH.read_text(encoding="utf-8")
    assert trainer.count("fit_component(") == 1
    assert "for candidate in candidates:" in trainer
    assert '"new_model_fit_count": 25' in runner
    assert '"route_freeze_allowed": selected is not None and not failed' in runner
    assert "Calibration amendment did not authorize a route freeze" in route
    assert '"outer_evaluation_allowed": False' in route
