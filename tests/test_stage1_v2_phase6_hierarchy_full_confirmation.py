from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.v2.run_stage1_v2_phase6_hierarchy_full_confirmation import (
    MASK_CANDIDATE,
    REFERENCE,
    SELECTED,
    build_grid,
    pair_guards,
)
from scripts.v2.freeze_stage1_v2_phase6_hierarchy_full_confirmation import (
    expected_source_seed,
)
from server_training_pipeline.stage1_v2_trainer_interface import PARITY


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / (
    "server_training_pipeline/"
    "stage1_v2_phase6_hierarchy_full_confirmation_protocol_v1.json"
)


def test_protocol_retains_identity_hierarchy_and_identifier_routes() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["selected_hierarchy_candidate"] == SELECTED
    assert protocol["selected_candidate_contract"]["test_weight_calibration"] == "identity"
    assert protocol["routing_policy"]["GNEW_EOBS"]["candidate"] == SELECTED
    assert {
        route["candidate"]
        for scenario, route in protocol["routing_policy"].items()
        if scenario != "GNEW_EOBS"
    } == {REFERENCE}
    assert protocol["confirmation_grid"] == {
        "scenarios": [
            "GNEW_EOBS",
            "GOBS_ENEW",
            "GNEW_ENEW",
            "TEMPORAL_YEAR",
            "COUNTRY_HOLDOUT",
        ],
        "outer_folds": [1, 2, 3, 4, 5],
        "inner_folds": [1, 2, 3, 4, 5],
        "routed_state_count": 125,
        "active_hierarchy_state_count": 25,
        "exact_reference_reuse_state_count": 100,
        "matched_training_run_count": 50,
        "matched_seed_within_state": True,
    }
    assert protocol["outer_test_metrics_read"] is False
    assert protocol["final_holdout_outcomes_read"] is False


def test_grid_has_25_matched_states_and_50_runs(tmp_path: Path) -> None:
    rows = []
    for outer in range(1, 6):
        for inner in range(1, 6):
            rows.append(
                {
                    "state_id": f"GNEW_EOBS__OUTER{outer}__INNER{inner}",
                    "state_level": "INNER",
                    "scenario": "GNEW_EOBS",
                    "outer_fold": outer,
                    "inner_fold": inner,
                }
            )
    path = tmp_path / PARITY / "splits/state_registry.tsv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    grid = build_grid(tmp_path, protocol)
    assert len(grid) == 50
    assert grid["state_id"].nunique() == 25
    assert set(grid["candidate"]) == {REFERENCE, SELECTED}
    assert grid.groupby("state_id")["seed"].nunique().eq(1).all()


def test_guard_pairing_requires_shared_mask_and_identifiers() -> None:
    rows = []
    for candidate, rmse in ((REFERENCE, 1.0), (SELECTED, 0.8)):
        rows.append(
            {
                "state_id": "GNEW_EOBS__OUTER1__INNER1",
                "scenario": "GNEW_EOBS",
                "candidate": candidate,
                "mask_candidate": MASK_CANDIDATE,
                "subset": "MARKER_SUPPORTED",
                "rows": 1000,
                "observation_id_signature": "same",
                "normalized_rmse_macro": rmse,
                "pearson_macro": 0.5,
            }
        )
    paired = pair_guards(pd.DataFrame(rows))
    selected = paired.loc[paired["candidate"].eq(SELECTED)].iloc[0]
    assert selected["relative_nrmse_gain"] == pytest.approx(0.2)
    assert selected["rows"] == selected["rows_reference"]


def test_launcher_is_detached_and_resumable() -> None:
    launcher = (
        ROOT
        / "scripts/v2/run_stage1_v2_phase6_hierarchy_full_confirmation_server_cpu.sh"
    ).read_text(encoding="utf-8")
    assert "nohup setsid" in launcher
    assert "--resume" in launcher
    assert "certify_stage1_v2_phase6_hierarchy_guard_amendment" in launcher
    assert "freeze_stage1_v2_phase6_hierarchy_full_confirmation" in launcher


def test_source_seed_formula_includes_scenario_offsets() -> None:
    scenarios = [
        "GNEW_EOBS",
        "GOBS_ENEW",
        "GNEW_ENEW",
        "TEMPORAL_YEAR",
        "COUNTRY_HOLDOUT",
    ]
    assert expected_source_seed("GNEW_EOBS", 1, 1, scenarios) == 63111
    assert expected_source_seed("GOBS_ENEW", 1, 1, scenarios) == 73111
    assert expected_source_seed("GNEW_ENEW", 1, 1, scenarios) == 83111
    assert expected_source_seed("TEMPORAL_YEAR", 1, 1, scenarios) == 93111
    assert expected_source_seed("COUNTRY_HOLDOUT", 1, 1, scenarios) == 103111
