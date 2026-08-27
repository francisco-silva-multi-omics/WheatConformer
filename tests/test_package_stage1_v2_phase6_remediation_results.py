from __future__ import annotations

from scripts.v2.package_stage1_v2_phase6_remediation_results import (
    CODE_FILES,
    EXPECTED_CANDIDATES,
    EXPECTED_RUNS,
    EXPECTED_SCENARIOS,
    RUN_FILES,
    SUMMARY_FILES,
    expected_candidate_scenarios,
)


def test_remediation_export_allowlist_is_reporting_only() -> None:
    names = " ".join((*SUMMARY_FILES, *RUN_FILES, *CODE_FILES)).lower()
    for forbidden in ("prediction.parquet", "checkpoint", "phenotype.parquet", "outer_test"):
        assert forbidden not in names


def test_remediation_export_grid_contract() -> None:
    pairs = expected_candidate_scenarios()
    assert len(EXPECTED_CANDIDATES) == 4
    assert len(EXPECTED_SCENARIOS) == 5
    assert len(pairs) == 14
    assert EXPECTED_RUNS == 70
