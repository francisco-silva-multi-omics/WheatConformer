from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest


RELEASE_ID = "P6AEPR_20260809_V1_274E41DF"
PARENTS = {
    "P5SBK_20260808_V1_274E41DF",
    "P5PESP_20260809_V2_274E41DF",
    "P5REV2_20260809_V1_274E41DF",
    "P5KATC_20260809_V1_274E41DF",
}
ALLOWED_CLASSES = {
    "directly_reproducible_future",
    "physically_reconstructable",
    "static_future_available",
    "management_scenario_required",
    "historical_observational_proxy",
    "transport_validation_required",
    "irrecoverable",
}


def release_root() -> Path:
    configured = os.environ.get("PHASE6A_ENV_RELEASE_ROOT")
    if configured:
        return Path(configured)
    code_root = Path(__file__).resolve().parents[1]
    data_root = Path(os.environ.get("STAGE1_V2_DATA_ROOT", code_root)).resolve()
    return data_root / "audit/v2/phase6a_environmental_projection_readiness_v1"


@pytest.fixture(scope="module")
def release() -> Path:
    path = release_root()
    if not path.exists():
        pytest.skip("Phase-6A environmental projection-readiness release not built")
    return path


def test_opening_binds_four_parents_and_is_phenotype_blind(release: Path) -> None:
    opening = json.loads((release / "OPENING_RELEASE.json").read_text(encoding="utf-8"))
    assert opening["release_id"] == RELEASE_ID
    assert {row["release_id"] for row in opening["parents"]} == PARENTS
    assert opening["stage1_v2_only"] is True
    assert opening["frozen_state_registry_rows"] == 150
    assert opening["phenotype_blind"] is True
    assert opening["legacy_v1_metric_selected_architecture_inherited"] is False
    assert opening["protected_files_rendered"] == []


def test_every_feature_has_exactly_one_primary_class_and_two_explicit_contracts(
    release: Path,
) -> None:
    contract = pd.read_csv(release / "environmental_feature_contract.tsv", sep="\t")
    assert contract.feature_id.is_unique
    assert contract.primary_class.isin(ALLOWED_CLASSES).all()
    assert len(contract) == 193
    assert contract.origin.eq("AUTHORITATIVE_PARITY_V2").sum() == 163
    historical = set(contract.loc[contract.E_HISTORICAL_ENHANCED_V2.astype(bool), "feature_id"])
    projection = set(contract.loc[contract.E_PROJECTION_CORE_V2.astype(bool), "feature_id"])
    assert historical
    assert projection
    assert historical != projection
    historical_proxy = contract.primary_class.eq("historical_observational_proxy")
    assert not contract.loc[historical_proxy, "E_PROJECTION_CORE_V2"].astype(bool).any()


def test_legacy_fifteen_are_reconciled_without_metric_architecture(release: Path) -> None:
    crosswalk = pd.read_csv(release / "legacy_v1_to_v2_feature_crosswalk.tsv", sep="\t")
    assert len(crosswalk) == 15
    assert crosswalk.legacy_v1_feature.nunique() == 15
    assert set(crosswalk.reconciliation_status) == {
        "replaced",
        "outside_authoritative_v2_feature_set",
        "absent",
    }
    assert not crosswalk.legacy_exact_name_present_in_v2.astype(bool).any()
    assert not crosswalk.phenotype_or_harvest_anchor_allowed.astype(bool).any()
    assert not crosswalk.metric_selected_architecture_inherited.astype(bool).any()


def test_split_local_reconstruction_replays_all_150_states(release: Path) -> None:
    replay = pd.read_csv(
        release / "split_local_reconstruction_certification.tsv", sep="\t"
    )
    assert len(replay) == 150
    assert replay.state_id.nunique() == 150
    assert replay.status.eq("PASS").all()
    assert replay.parameter_mismatches.eq(0).all()
    assert replay.registry_mismatches.eq(0).all()
    assert replay.expected_parameter_rows.eq(replay.replayed_parameter_rows).all()
    assert replay.fit_scope.eq("TRAINING_ENVIRONMENTS_ONLY").all()


def test_backcast_records_partial_success_and_required_blockers(release: Path) -> None:
    backcast = pd.read_csv(release / "historical_backcast_validation.tsv", sep="\t")
    assert backcast.status.eq("PASS_EXACT_ALIAS").any()
    assert backcast.status.eq("PASS_STATIC_BACKCAST").any()
    assert backcast.status.eq(
        "PARTIAL_PASS_AGGREGATE_REPLAY_DAILY_ARCHIVE_BLOCKED"
    ).any()
    water = backcast[backcast.component.eq("E_PROJECTION_CORE_WATER_BALANCE")]
    assert len(water) == 5
    assert water.finite_environments.eq(0).all()
    assert water.status.eq("BLOCKED_NO_ANTECEDENT_DAILY_PET_SOIL_BACKCAST").all()


def test_climate_generations_and_ood_actions_are_separate_and_frozen(release: Path) -> None:
    climate = json.loads(
        (release / "climate_source_and_unit_contract.json").read_text(encoding="utf-8")
    )
    assert climate["primary_interface"] == "CMIP6_SSP_DAILY_MEMBER_RESOLVED"
    assert climate["future_covariates_inspected"] is False
    assert climate["ensemble_policy"]["no_ensemble_averaging_before_feature_derivation"] is True
    compatibility = json.loads(
        (release / "cmip5_cmip6_compatibility_contract.json").read_text(encoding="utf-8")
    )
    assert compatibility["pool_generations"] is False
    assert compatibility["average_rcp_and_ssp_labels"] is False
    ood = json.loads(
        (release / "applicability_domain_protocol.json").read_text(encoding="utf-8")
    )
    assert ood["frozen_before_future_covariate_inspection"] is True
    assert ood["clipping_allowed"] is False
    assert set(ood["actions"]) == {
        "IN_DOMAIN",
        "LIMITED_EXTRAPOLATION",
        "OUT_OF_DOMAIN",
    }


def test_leakage_protected_access_hashes_and_tests_pass(release: Path) -> None:
    leakage = pd.read_csv(release / "leakage_audit.tsv", sep="\t")
    assert leakage.status.eq("PASS").all()
    assert not leakage.phenotype_value_accessed.astype(bool).any()
    assert not leakage.protected_outcome_accessed.astype(bool).any()
    protected = pd.read_csv(
        release / "protected_outcome_access_audit.tsv", sep="\t", dtype=str
    )
    assert not protected.decision.eq("DENY").any()
    assert not protected.relative_path.str.contains(
        "selection_lock.json|inner.*metric|validation.*metric|prediction|outer.*outcome|final.*holdout",
        case=False,
        regex=True,
    ).any()
    closing = pd.read_csv(release / "CLOSING_HASH_MANIFEST.tsv", sep="\t")
    assert closing.status.eq("PASS").all()
    tests = pd.read_csv(release / "tests/test_summary.tsv", sep="\t")
    assert tests.status.isin(["PASS", "SKIP"]).all()


def test_atomic_decision_if_finalized(release: Path) -> None:
    path = release / "PHASE6A_PROJECTION_READINESS_DECISION.json"
    if not path.exists():
        pytest.skip("Predecision test run")
    decision = json.loads(path.read_text(encoding="utf-8"))
    assert decision["release_id"] == RELEASE_ID
    assert decision["status"] == (
        "BLOCKED_PHASE6A_PROJECTION_READINESS_INCOMPLETE_DAILY_BACKCAST_AND_WATER_BALANCE"
    )
    assert decision["split_local_states_passing"] == 150
    assert decision["historical_daily_backcast_passed"] is False
    assert decision["antecedent_pet_soil_water_backcast_passed"] is False
    assert decision["projection_core_training_authorized"] is False
    assert decision["future_covariate_generation_authorized"] is False
    assert decision["future_prediction_authorized"] is False
    assert decision["phenotype_values_accessed"] is False
    assert decision["model_training_performed"] is False
