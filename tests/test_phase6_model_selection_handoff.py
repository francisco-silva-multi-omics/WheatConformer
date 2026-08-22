from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_aggregate_handoff_freezer_binds_every_authoritative_release() -> None:
    source = (ROOT / "scripts/v2/freeze_phase6_model_selection_handoff.py").read_text()
    required = [
        "phase5_split_bound",
        "phase5_parity",
        "ka_150_state",
        "regulatory_eligibility_v2",
        "projection_core_split_bound_historical",
        "projection_core_future_covariates",
        "cimmyt_pre_qc_split_local",
        "h_seeds_operator",
    ]
    for label in required:
        assert label in source


def test_aggregate_handoff_release_passes_when_present() -> None:
    release = ROOT / "audit/v2/phase6_model_selection_handoff_v1"
    decision_path = release / "PHASE6_MODEL_SELECTION_HANDOFF.json"
    if not decision_path.exists():
        return
    decision = json.loads(decision_path.read_text())
    validation = pd.read_csv(release / "validation_checks.tsv", sep="\t")
    inventory = pd.read_csv(release / "authoritative_release_inventory.tsv", sep="\t")
    assert decision["status"] == "PASS_READY_FOR_STAGE1_V2_PHASE6_INNER_MODEL_SELECTION"
    assert decision["inner_state_count"] == 125
    assert decision["projection_inactive_environment_count"] == 814
    assert decision["outer_evaluation_allowed"] is False
    assert validation["status"].eq("PASS").all()
    assert inventory["status"].eq("PASS").all()
