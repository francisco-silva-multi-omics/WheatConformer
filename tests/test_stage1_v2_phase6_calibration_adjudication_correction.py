from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.v2.certify_stage1_v2_phase6_hierarchy_calibration_adjudication_correction import (
    corrected_summary,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_hierarchy_calibration_adjudication_correction_protocol_v1.json"
)
CERTIFIER = (
    ROOT
    / "scripts"
    / "v2"
    / "certify_stage1_v2_phase6_hierarchy_calibration_adjudication_correction.py"
)
ROUTE = (
    ROOT
    / "scripts"
    / "v2"
    / "freeze_stage1_v2_phase6_hierarchy_calibration_corrected_route.py"
)


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_macro_calibration_uses_run_macro_and_keeps_worst_trait_diagnostic() -> None:
    candidate = "hierarchy_test_weight_identity_calibration_v2"
    paired = pd.DataFrame(
        [
            {
                "state_id": "STATE_1",
                "scenario": "GNEW_EOBS",
                "candidate": candidate,
                "validation_macro_normalized_rmse": 0.60,
                "validation_macro_pearson": 0.80,
                "validation_macro_calibration_error": 0.064,
                "relative_nrmse_gain": 0.20,
                "nrmse_win": True,
                "pearson_gain": 0.10,
                "centered_spearman_gain": 0.01,
                "pairwise_accuracy_gain": 0.01,
            }
        ]
    )
    traits = []
    values = {
        "1000_GRAIN_WEIGHT": 0.02,
        "DAYS_TO_HEADING": 0.03,
        "DAYS_TO_MATURITY": 0.04,
        "GRAIN_YIELD": 0.05,
        "PLANT_HEIGHT": 0.06,
        "ABOVE_GROUND_BIOMASS": 0.047,
        "TEST_WEIGHT": 0.201,
    }
    for trait, error in values.items():
        traits.append(
            {
                "state_id": "STATE_1",
                "scenario": "GNEW_EOBS",
                "candidate": candidate,
                "trait_name_canonical": trait,
                "rows": 1000,
                "normalized_rmse": 0.6,
                "pearson": 0.8,
                "calibration_slope": 1.0 - error,
                "calibration_error": error,
                "relative_nrmse_gain": 0.2,
                "pearson_gain": 0.1,
            }
        )
    guards = pd.DataFrame(
        [
            {
                "state_id": "STATE_1",
                "scenario": "GNEW_EOBS",
                "candidate": candidate,
                "subset": subset,
                "rows": 1000,
                "relative_nrmse_gain": 0.2,
                "pearson_gain": 0.1,
            }
            for subset in [
                "PEDIGREE_ONLY",
                "MARKER_SUPPORTED",
                "PEDIGREE_AND_MARKER",
                "NEITHER_PRODUCTION_PEDIGREE_NOR_DENSE_MARKERS",
                "RECOVERED_IDENTITY_OR_COMPONENT",
                "PROJECTION_CORE_INACTIVE_814_ENVIRONMENTS",
            ]
        ]
    )
    decision, diagnostic = corrected_summary(
        protocol(), paired, pd.DataFrame(traits), guards
    )
    row = decision.iloc[0]
    assert row["absolute_macro_calibration_error_max"] == 0.064
    assert row["worst_trait_fold_calibration_error_max"] == 0.201
    assert bool(row["guard_macro_calibration"])
    assert bool(row["eligible_for_route_freeze"])
    assert diagnostic.iloc[0]["trait_name_canonical"] == "TEST_WEIGHT"
    assert diagnostic.iloc[0]["selection_role"] == "diagnostic_only"


def test_correction_is_reporting_only_and_route_remains_outer_blocked() -> None:
    value = protocol()
    assert value["operation"] == "reporting_only_semantic_correction"
    assert value["new_model_training_allowed"] is False
    assert value["new_prediction_generation_allowed"] is False
    assert value["acceptance"]["maximum_absolute_macro_calibration_error"] == 0.2
    assert value["metric_semantics"]["worst_trait_fold_calibration"] == "diagnostic_only"
    certifier = CERTIFIER.read_text(encoding="utf-8")
    route = ROUTE.read_text(encoding="utf-8")
    assert "validation_macro_calibration_error" in certifier
    assert "worst_trait_fold_calibration_diagnostic.tsv" in certifier
    assert '"outer_evaluation_allowed": False' in certifier
    assert '"outer_protocol_creation_allowed": False' in route
