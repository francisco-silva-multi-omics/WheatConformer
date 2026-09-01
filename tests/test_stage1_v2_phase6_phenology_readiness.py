from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.v2.freeze_stage1_v2_phase6_phenology_readiness import (
    extension_inventories,
    harvest_horizon_audit,
)
from scripts.v2.fetch_stage1_v2_phase6_phenology_horizon_extension import (
    MAXIMUM_CDS_WORKERS,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_phenology_readiness_protocol_v1.json"
)
PLAN = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_post_hierarchy_screen_plan_v3.json"
)


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_fa_is_terminal_and_phenology_is_the_only_active_readiness_gate() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    gates = {row["name"]: row for row in plan["ordered_gates"]}
    assert gates["covariate_linked_factor_analytic_decomposition"]["status"] == (
        "TERMINAL_ACTIVE_COMPONENT_NO_ADVANCE"
    )
    assert gates["covariate_linked_factor_analytic_decomposition"][
        "confirmation_allowed"
    ] is False
    assert gates["phenotype_safe_phenology_random_regression"]["status"] == (
        "ACTIVE_READINESS_GATE"
    )
    assert gates["phenotype_safe_phenology_random_regression"][
        "training_allowed"
    ] is False
    assert plan["retained_reference"] == "current_huber_authoritative_row_mass"


def test_horizon_policy_is_phenotype_blind_and_does_not_authorize_240() -> None:
    value = protocol()
    policy = value["horizon_policy"]
    assert policy["candidate_inclusive_endpoint_days"] == [179, 209, 239, 269, 299]
    assert 240 not in policy["candidate_inclusive_endpoint_days"]
    assert policy["minimum_valid_nonphenotypic_harvest_coverage"] == 0.99
    assert policy["global_DTH_or_DTM_quantiles_allowed"] is False
    assert policy["phenotype_values_allowed"] is False
    assert value["protected_access"]["future_SSP_values_read"] is False
    assert value["extension_contract"]["cross_provider_policy"][
        "new_Open_Meteo_extension_fetch_required"
    ] is False
    assert value["extension_contract"]["continuous_CDS_reference_reuse"][
        "new_fetch_policy"
    ] == "only_requests_not_fully_covered_by_the_certified_continuous_reference"


def test_horizon_selection_uses_finish_then_start_and_fixed_endpoints() -> None:
    weather = pd.DataFrame(
        {
            "env_id": [f"E{i}" for i in range(5)],
            "sowing_date": ["2000-01-01"] * 5,
            "harvest_finish_date": [
                "2000-05-30",
                "2000-06-29",
                "2000-07-29",
                "2000-08-28",
                None,
            ],
            "harvest_start_date": [None, None, None, None, "2000-09-27"],
        }
    )
    policy = protocol()["horizon_policy"] | {
        "minimum_valid_nonphenotypic_harvest_coverage": 0.8
    }
    audit, coverage, endpoint = harvest_horizon_audit(weather, policy)
    assert endpoint == 269
    assert audit.iloc[-1]["harvest_anchor_source"] == "harvest_start_date"
    assert coverage["inclusive_endpoint_day"].tolist() == [179, 209, 239, 269, 299]


def test_horizon_audit_collapses_duplicate_environment_metadata() -> None:
    weather = pd.DataFrame(
        {
            "env_id": ["E1", "E1", "E2"],
            "sowing_date": ["2000-01-01", "2000-01-01", "2000-01-01"],
            "harvest_finish_date": ["2000-07-01", "2000-07-01", None],
            "harvest_start_date": [None, None, "2000-07-15"],
        }
    )
    audit, _, _ = harvest_horizon_audit(weather, protocol()["horizon_policy"])
    assert len(audit) == 2
    assert audit["environment_id"].is_unique
    assert int(audit.set_index("environment_id").loc["E1", "source_row_count"]) == 2


def test_horizon_audit_rejects_conflicting_dates() -> None:
    weather = pd.DataFrame(
        {
            "env_id": ["E1", "E1"],
            "sowing_date": ["2000-01-01", "2000-01-02"],
            "harvest_finish_date": ["2000-07-01", "2000-07-01"],
            "harvest_start_date": [None, None],
        }
    )
    audit, _, endpoint = harvest_horizon_audit(
        weather, protocol()["horizon_policy"]
    )
    assert audit.iloc[0]["status"] == "CONFLICTING_SOWING"
    assert endpoint is None


def test_extension_inventory_is_exactly_sowing_plus_180_through_endpoint() -> None:
    environment_map = pd.DataFrame(
        {
            "environment_id": ["E1", "E2"],
            "latitude": [10.0, 10.0],
            "longitude": [20.0, 20.0],
            "sowing_date": ["2000-01-01", "2000-01-01"],
            "source_archive_status": [
                "READY_SOWING_RELATIVE_ARCHIVE",
                "READY_SOWING_RELATIVE_ARCHIVE",
            ],
        }
    )
    mapping, daily, reuse, masked, cds = extension_inventories(
        environment_map, 269, 180
    )
    assert len(mapping) == 2
    assert len(daily) == 1
    assert len(reuse) == 0
    assert len(masked) == 0
    assert len(cds) == 1
    assert daily.iloc[0]["request_start_date"] == "2000-06-29"
    assert daily.iloc[0]["request_end_date"] == "2000-09-26"
    assert int(daily.iloc[0]["mapped_environment_count"]) == 2
    assert daily["request_id"].is_unique


def test_extension_reuses_only_complete_reference_intervals() -> None:
    environment_map = pd.DataFrame(
        {
            "environment_id": ["E1", "E2"],
            "latitude": [10.0, 10.0],
            "longitude": [20.0, 20.0],
            "sowing_date": ["2010-01-01", "2010-06-01"],
            "source_archive_status": [
                "READY_SOWING_RELATIVE_ARCHIVE",
                "READY_SOWING_RELATIVE_ARCHIVE",
            ],
        }
    )
    reference = pd.DataFrame(
        {
            "latitude": [10.0],
            "longitude": [20.0],
            "reference_site_id": ["SITE"],
            "reference_request_id": ["REFERENCE"],
            "reference_start": ["1981-01-01"],
            "reference_end": ["2010-12-31"],
            "reference_daily_path": ["reference.parquet"],
            "reference_daily_sha256": ["a" * 64],
            "reference_metadata_path": ["reference.json"],
            "reference_metadata_sha256": ["b" * 64],
            "reference_required_climate_complete_days": [10957],
            "reference_required_climate_incomplete_days": [0],
        }
    )
    _, daily, reuse, masked, cds = extension_inventories(
        environment_map, 269, 180, reference
    )
    assert daily["source_route"].value_counts().to_dict() == {
        "REUSE_CERTIFIED_CDS_1981_2010": 1,
        "FETCH_CDS_EXTENSION": 1,
    }
    assert len(reuse) == 1
    assert len(masked) == 0
    assert len(cds) == 1
    assert reuse.iloc[0]["reference_site_id"] == "SITE"


def test_incomplete_continuous_reference_is_masked_not_reused_or_refetched() -> None:
    environment_map = pd.DataFrame(
        {
            "environment_id": ["E1"],
            "latitude": [10.0],
            "longitude": [20.0],
            "sowing_date": ["2000-01-01"],
            "source_archive_status": ["READY_SOWING_RELATIVE_ARCHIVE"],
        }
    )
    reference = pd.DataFrame(
        {
            "latitude": [10.0],
            "longitude": [20.0],
            "reference_site_id": ["SITE"],
            "reference_request_id": ["REFERENCE"],
            "reference_start": ["1981-01-01"],
            "reference_end": ["2010-12-31"],
            "reference_daily_path": ["reference.parquet"],
            "reference_daily_sha256": ["a" * 64],
            "reference_metadata_path": ["reference.json"],
            "reference_metadata_sha256": ["b" * 64],
            "reference_required_climate_complete_days": [0],
            "reference_required_climate_incomplete_days": [10957],
        }
    )
    _, daily, reuse, masked, cds = extension_inventories(
        environment_map, 269, 180, reference
    )
    assert daily.iloc[0]["source_route"] == "MASKED_NO_VALID_CDS_LAND_REFERENCE"
    assert len(reuse) == 0
    assert len(masked) == 1
    assert len(cds) == 0


def test_phase1_scope_remains_five_inner_states_and_outer_is_blocked() -> None:
    value = protocol()["prospective_phenology_contract"]
    assert value["phase_1_state_count"] == 5
    assert value["phase_1_scenario"] == "GNEW_EOBS"
    assert value["phase_1_inner_fold"] == 1
    assert value["outer_evaluation_allowed"] is False
    assert value["validation_target_values_allowed_as_inputs"] is False
    assert value["outer_test_target_values_allowed"] is False


def test_cds_concurrency_cannot_exceed_provider_queue_limit() -> None:
    assert MAXIMUM_CDS_WORKERS == 20
