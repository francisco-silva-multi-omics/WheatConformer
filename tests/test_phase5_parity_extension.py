from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest


RELEASE_ID = "P5PESP_20260809_V2_274E41DF"
UPSTREAM_ID = "P5SBK_20260808_V1_274E41DF"
FOUR_LOCKS = {
    "server_phase5_parity_bundle/artifacts/audit/reaction_norm_explicit_environment_v2_frozen/reaction_norm_environment_selection_lock.json",
    "server_phase5_parity_bundle/artifacts/audit/reaction_norm_explicit_environment_v2_frozen/reaction_norm_selection_lock.json",
    "server_phase5_parity_bundle/artifacts/audit/reaction_norm_explicit_environment_v3_frozen/reaction_norm_environment_selection_lock.json",
    "server_phase5_parity_bundle/artifacts/audit/reaction_norm_explicit_environment_v3_frozen/reaction_norm_selection_lock.json",
}


def release_root() -> Path:
    configured = os.environ.get("PHASE5_PARITY_RELEASE_ROOT")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "audit/v2/phase5_panel_environment_scenario_parity_extension_v2"


@pytest.fixture(scope="module")
def release() -> Path:
    path = release_root()
    if not path.exists():
        pytest.skip("Phase-5 parity extension v2 has not been built")
    return path


def test_opening_contract_is_clean_v2(release: Path) -> None:
    opening = json.loads((release / "OPENING_RELEASE.json").read_text(encoding="utf-8"))
    assert opening["release_id"] == RELEASE_ID
    assert opening["authoritative_phase5_release"] == UPSTREAM_ID
    assert opening["v1_attempt_disposition"] == "TERMINALLY_BLOCKED_INCIDENT_ONLY_NO_SCIENTIFIC_DECISIONS_INHERITED"
    assert opening["phenotype_blind"] is True
    assert opening["protected_files_rendered"] == []


def test_protected_locks_are_metadata_only(release: Path) -> None:
    opening = pd.read_csv(release / "OPENING_HASH_MANIFEST.tsv", sep="\t", dtype=str)
    locks = opening[opening.relative_path.isin(FOUR_LOCKS)]
    assert set(locks.relative_path) == FOUR_LOCKS
    assert locks.access.eq("APPROVED_SHA256_METADATA_ONLY").all()
    assert locks.sha256.str.fullmatch(r"[0-9a-f]{64}").all()


def test_primary_panel_overlap_is_exact(release: Path) -> None:
    overlap = pd.read_csv(release / "genomic/panel_stage1_overlap.tsv", sep="\t")
    expected = {
        "frozen_hmp_v1": (5187, 1_173_132),
        "cimmyt_bread_gbs_2013_2018": (4512, 721_033),
        "seeds_of_discovery_dartseq": (3212, 801_276),
        "eyt_haplotype_blocks_2011_2018": (2612, 520_592),
        "dartag_panel2": (1931, 280_688),
        "mas_45ibwsn": (334, 158_464),
        "hibap35k": (95, 52_397),
        "mexican_landrace_dartseq": (0, 0),
    }
    by_panel = overlap.set_index("panel_id")
    for panel, values in expected.items():
        assert tuple(by_panel.loc[panel, ["primary_stage1_gids", "primary_stage1_rows"]].astype(int)) == values
    assert overlap.status.eq("PASS").all()


def test_scenario_extension_has_150_training_states(release: Path) -> None:
    states = pd.read_csv(release / "splits/state_registry.tsv", sep="\t")
    assert len(states) == 150
    assert set(states.scenario) == {"GNEW_EOBS", "GOBS_ENEW", "GNEW_ENEW", "TEMPORAL_YEAR", "COUNTRY_HOLDOUT"}
    assert states.training_gid_signature.str.fullmatch(r"[0-9a-f]{64}").all()
    assert states.training_environment_signature.str.fullmatch(r"[0-9a-f]{64}").all()
    leakage = pd.read_csv(release / "splits/scenario_leakage_report.tsv", sep="\t")
    assert leakage.status.eq("PASS").all()
    roles = pq.ParquetFile(release / "splits/scenario_observation_roles.parquet")
    assert roles.metadata.num_rows == 2_242_863


def test_seeds_protocol_and_states_are_split_local(release: Path) -> None:
    protocols = pd.read_csv(release / "genomic/panel_qc_protocols.tsv", sep="\t", dtype=str)
    seeds = protocols[protocols.panel_id.eq("seeds_of_discovery_dartseq")].iloc[0]
    assert seeds.status == "FROZEN_BEFORE_CALL_INSPECTION"
    assert ">=10000" in seeds.replicate_qc
    assert ">=0.995" in seeds.replicate_qc
    stream = json.loads((release / "genomic/seeds_stream_audit.json").read_text(encoding="utf-8"))
    assert stream["source_markers"] == 102_474
    assert stream["selected_primary_sample_instances"] == 5_256
    assert stream["complete_matrix_string_dataframe_materialized"] is False
    registry = pd.read_csv(release / "genomic/seeds_component_registry.tsv", sep="\t")
    preprocessing = pd.read_csv(release / "genomic/seeds_fold_preprocessing_registry.tsv", sep="\t")
    assert len(registry) == 150
    assert len(preprocessing) == 150
    active = preprocessing[preprocessing.status.eq("PASS")]
    assert len(active) > 0
    assert active.allele_frequency_fit_scope.eq("TRAINING_GIDS_ONLY").all()
    assert active.imputation_fit_scope.eq("TRAINING_GIDS_ONLY_2P").all()
    diagnostics = pd.read_csv(release / "genomic/seeds_state_diagnostics.tsv", sep="\t")
    assert diagnostics.status.eq("PASS").all()


def test_hmp_is_not_promoted_from_imputed_export(release: Path) -> None:
    trace = json.loads((release / "genomic/hmp_source_trace.json").read_text(encoding="utf-8"))
    assert trace["all_rows_marked_imputed"] is True
    assert trace["observed_missing_call_mask_recoverable"] is False
    assert trace["production_disposition"] == "BLOCKED_TRACED_TO_GLOBALLY_IMPUTED_CIMMYT_EXPORT"
    panels = pd.read_csv(release / "genomic/panel_source_axis_audit.tsv", sep="\t", dtype=str)
    hmp = panels[panels.panel_id.eq("frozen_hmp_v1")].iloc[0]
    assert hmp.v2_terminal_disposition == "BLOCKED_TRACED_TO_GLOBALLY_IMPUTED_CIMMYT_EXPORT"


def test_environment_states_are_training_local_and_outcome_free(release: Path) -> None:
    provenance = pd.read_csv(release / "environment/environment_feature_provenance.tsv", sep="\t")
    assert not provenance.phenology_outcome_used.astype(bool).any()
    assert not provenance.feature.str.contains("heading|maturity|phenotype|yield|metric|prediction", case=False, regex=True).any()
    parameters = pd.read_csv(release / "environment/environment_preprocessing_parameters.tsv", sep="\t")
    assert parameters.fit_scope.eq("TRAINING_ENVIRONMENTS_ONLY").all()
    registry = pd.read_csv(release / "environment/environment_component_registry.tsv", sep="\t")
    assert "E_REACTION_NORM" in set(registry.component)
    diagnostics = pd.read_csv(release / "environment/environment_state_certifications.tsv", sep="\t")
    assert diagnostics.status.eq("PASS").all()
    protocol = json.loads((release / "environment/reaction_norm_protocol.json").read_text(encoding="utf-8"))
    assert protocol["historical_v1_metric_selected_architecture_inherited"] is False
    assert protocol["protected_selection_locks_opened"] is False


def test_component_absence_is_mask_not_row_deletion(release: Path) -> None:
    masks = pq.ParquetFile(release / "masks/observation_component_masks.parquet")
    assert masks.metadata.num_rows == 3_193_677
    summary = pd.read_csv(release / "masks/component_mask_summary.tsv", sep="\t")
    assert summary.master_rows.eq(3_193_677).all()
    assert summary.rows_deleted.eq(0).all()
    assert summary.status.eq("PASS_EXPLICIT_MASK_NO_ROW_DELETION").all()
    views = pd.read_csv(release / "view_preservation_audit.tsv", sep="\t")
    assert views.difference.eq(0).all()
    assert views.status.eq("PASS").all()


def test_inputs_close_immutable_and_replay_passes(release: Path) -> None:
    closing = pd.read_csv(release / "CLOSING_HASH_MANIFEST.tsv", sep="\t", dtype=str)
    assert not closing.status.str.startswith("FAIL").any()
    replay = pd.read_csv(release / "deterministic_replay_validation.tsv", sep="\t", dtype=str)
    assert replay.status.eq("PASS").all()


def test_atomic_decision_if_finalized(release: Path) -> None:
    path = release / "PHASE5_PARITY_EXTENSION_DECISION.json"
    if not path.exists():
        pytest.skip("Predecision test run")
    decision = json.loads(path.read_text(encoding="utf-8"))
    assert decision["release_id"] == RELEASE_ID
    assert decision["status"] == "PASS_PHASE5_PARITY_EXTENSION_WITH_EXPLICIT_COMPONENT_BLOCKERS"
    assert decision["inner_validation_metrics_accessed"] is False
    assert decision["outer_test_outcomes_accessed"] is False
    assert decision["final_holdout_accessed"] is False
    assert decision["model_training_performed"] is False
