from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_V1 = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_post_hierarchy_screen_plan_v1.json"
)
PLAN_V2 = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_post_hierarchy_screen_plan_v2.json"
)


def load_plan() -> dict:
    return json.loads(PLAN_V2.read_text(encoding="utf-8"))


def test_plan_v2_binds_parent_and_does_not_change_active_screen() -> None:
    plan = load_plan()
    assert hashlib.sha256(PLAN_V1.read_bytes()).hexdigest() == plan[
        "parent_plan_sha256"
    ]
    assert plan["active_trait_balance_screen_changed"] is False
    assert plan["active_trait_balance_metrics_read"] is False
    assert plan["outer_test_metrics_read"] is False
    assert plan["final_holdout_outcomes_read"] is False


def test_factorial_contains_exactly_the_four_preregistered_cells() -> None:
    factorial = load_plan()["factorial_ablation"]
    observed = {
        (
            row["model"],
            row["FA_decomposition"],
            row["phenology_alignment"],
            row["SINN_residual"],
        )
        for row in factorial["cells"]
    }
    assert observed == {
        ("FA", True, False, False),
        ("FA_PHENO", True, True, False),
        ("FA_SINN", True, False, True),
        ("PHENO_FA_SINN", True, True, True),
    }
    assert factorial["paired_estimands"]["interaction_on_NRMSE_scale"] == (
        "NRMSE_PHENO_FA_SINN - NRMSE_FA_PHENO - NRMSE_FA_SINN + NRMSE_FA"
    )
    assert "all 125" in factorial["screen_sequence"]["definitive_confirmation"]
    assert factorial["model_selection_independent_of_synergy_claim"] is True


def test_day_240_is_unselected_and_global_DTM_is_forbidden() -> None:
    horizon = load_plan()["daily_weather_horizon_policy"]
    assert horizon["current_authoritative_inclusive_endpoint_day"] == 179
    assert horizon["day_240_authorized"] is False
    assert horizon["withdrawn_justification"] == "global DTM quantiles"
    assert "DAYS_TO_MATURITY phenotype values" in horizon[
        "forbidden_horizon_evidence"
    ]
    assert horizon["feature_or_model_use_before_release_passes"] is False


def test_outer_protocol_waits_for_factorial_terminal_decision() -> None:
    plan = load_plan()
    assert plan["factorial_ablation"]["screen_sequence"][
        "outer_evaluation_allowed"
    ] is False
    assert plan["outer_protocol_policy"][
        "create_only_after_all_entered_inner_gates_and_factorial_are_terminal"
    ] is True
    assert plan["outer_protocol_policy"]["final_holdout_remains_sealed"] is True
