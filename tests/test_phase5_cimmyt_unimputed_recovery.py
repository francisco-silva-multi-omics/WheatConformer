from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest


RELEASE_ID = "P5CUG_20260809_V4_274E41DF"
PARITY_ID = "P5PESP_20260809_V2_274E41DF"
PHASE5_ID = "P5SBK_20260808_V1_274E41DF"


def release_root() -> Path:
    configured = os.environ.get("PHASE5_CIMMYT_UNIMPUTED_RELEASE_ROOT")
    if configured:
        return Path(configured)
    return (
        Path(__file__).resolve().parents[1]
        / "audit/v2/phase5_cimmyt_unimputed_recovery_v4"
    )


@pytest.fixture(scope="module")
def release() -> Path:
    path = release_root()
    if not path.exists():
        pytest.skip("CIMMYT unimputed recovery release has not been built")
    return path


def test_opening_contract_and_frozen_protocol(release: Path) -> None:
    opening = json.loads((release / "OPENING_RELEASE.json").read_text(encoding="utf-8"))
    assert opening["release_id"] == RELEASE_ID
    assert opening["authoritative_phase5_release"] == PHASE5_ID
    assert opening["authoritative_parity_release"] == PARITY_ID
    assert opening["v1_attempt_disposition"] == (
        "TERMINALLY_BLOCKED_INCIDENT_ONLY_NO_SCIENTIFIC_DECISIONS_INHERITED"
    )
    assert opening["qc_protocol_frozen_before_genotype_call_inspection"] is True
    assert opening["bundle_content_accessed"] is False
    assert opening["protected_files_rendered"] == []
    protocol = json.loads(
        (release / "genomic/CIMMYT_UNIMPUTED_QC_PROTOCOL.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["frozen_before_genotype_call_inspection"] is True
    assert protocol["sample_call_rate_minimum"] == 0.50
    assert protocol["marker_call_rate_minimum"] == 0.80
    assert protocol["minor_allele_frequency_minimum"] == 0.01
    assert protocol["marker_heterozygosity_maximum"] == 0.10
    assert protocol["training_state_minimum_gids"] == 20


def test_sources_are_distinct_and_axes_are_exact(release: Path) -> None:
    opening = pd.read_csv(release / "OPENING_HASH_MANIFEST.tsv", sep="\t", dtype=str)
    sources = opening[opening.scope.isin(["NEW_CIMMYT_SOURCE", "CIMMYT_IMPUTED_COMPARATOR"])]
    assert len(sources) == 2
    assert sources["sha256"].nunique() == 2
    assert sources["size"].astype(int).nunique() == 1
    axis = json.loads(
        (release / "genomic/cimmyt_source_axis_audit.json").read_text(encoding="utf-8")
    )
    assert axis["status"] == "PASS_EXACT_RAW_IMPUTED_AXES"
    assert axis["sample_columns"] == 50_363
    assert axis["marker_rows"] > 0
    assert axis["marker_identity_metadata_mismatches"] == 0
    assert axis["incompatible_allele_set_mismatches"] == 0
    assert axis["allele_orientation_reversals_harmonized_to_unimputed_order"] == 79
    markers = pq.ParquetFile(release / "genomic/cimmyt_marker_axis.parquet")
    assert markers.metadata.num_rows == axis["marker_rows"]


def test_stage1_overlap_is_exact(release: Path) -> None:
    overlap = pd.read_csv(release / "genomic/cimmyt_stage1_overlap.tsv", sep="\t")
    assert len(overlap) == 1
    row = overlap.iloc[0]
    assert row.primary_stage1_gids == 4_512
    assert row.primary_stage1_rows == 721_033
    assert row.discrepancy_gids == 0
    assert row.discrepancy_rows == 0
    assert row.status == "PASS_EXACT"
    samples = pd.read_csv(release / "genomic/cimmyt_primary_sample_qc.tsv", sep="\t")
    assert len(samples) == 4_512
    assert samples.accepted_canonical_gid.nunique() == 4_512


def test_missing_call_mask_is_recovered_without_overclaim(release: Path) -> None:
    comparison = json.loads(
        (release / "genomic/cimmyt_unimputed_imputed_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert comparison["status"] == (
        "PASS_CALL_MASK_RECOVERY_WITH_GLOBAL_MARKER_UNIVERSE_BLOCKER"
    )
    assert comparison["unimputed_missing_cells"] > 0
    assert comparison["unimputed_missing_filled_by_imputed"] > 0
    assert comparison[
        "lossless_observed_missing_call_mask_recovered_for_retained_marker_universe"
    ] is True
    assert comparison["pre_qc_marker_universe_recovered"] is False
    assert comparison["strict_production_disposition"] == (
        "BLOCKED_GLOBALLY_PREFILTERED_MARKER_UNIVERSE"
    )


def test_all_150_states_have_training_local_parameters_or_masks(release: Path) -> None:
    preprocessing = pd.read_csv(
        release / "genomic/cimmyt_fold_preprocessing_registry.tsv", sep="\t"
    )
    registry = pd.read_csv(release / "genomic/cimmyt_component_registry.tsv", sep="\t")
    assert len(preprocessing) == 150
    assert len(registry) == 150
    assert preprocessing.state_id.nunique() == 150
    candidates = preprocessing[preprocessing.diagnostic_candidate_available.astype(bool)]
    assert len(candidates) == 121
    assert candidates.marker_qc_fit_scope.eq("TRAINING_GIDS_ONLY").all()
    assert candidates.allele_frequency_fit_scope.eq("TRAINING_GIDS_ONLY").all()
    assert candidates.imputation_fit_scope.eq("TRAINING_GIDS_ONLY_2P_ON_DEMAND").all()
    assert candidates.training_local_retained_markers.gt(0).all()
    assert not registry.strict_production_component_available.astype(bool).any()
    assert registry.strict_production_mask.astype(bool).all()
    diagnostics = pd.read_csv(
        release / "genomic/cimmyt_state_diagnostics.tsv", sep="\t"
    )
    assert len(diagnostics) == 121
    assert diagnostics.status.eq("PASS_DIAGNOSTIC").all()
    assert diagnostics.psd_by_factor_construction.astype(bool).all()


def test_master_rows_are_preserved_with_explicit_masks(release: Path) -> None:
    masks = pq.ParquetFile(release / "masks/cimmyt_observation_component_masks.parquet")
    assert masks.metadata.num_rows == 3_193_677
    summary = pd.read_csv(
        release / "masks/cimmyt_observation_mask_summary.tsv", sep="\t"
    )
    assert summary.master_rows.eq(3_193_677).all()
    assert summary.rows_deleted.eq(0).all()
    assert summary.strict_production_kg_available_rows.eq(0).all()
    assert summary.status.eq("PASS_EXPLICIT_MASK_NO_ROW_DELETION").all()
    state_masks = pd.read_csv(
        release / "masks/cimmyt_state_component_masks.tsv", sep="\t"
    )
    assert len(state_masks) == 150
    assert state_masks.strict_production_mask.astype(bool).all()


def test_inputs_are_immutable_and_protected_content_is_unread(release: Path) -> None:
    closing = pd.read_csv(release / "CLOSING_HASH_MANIFEST.tsv", sep="\t", dtype=str)
    assert closing.status.eq("PASS").all()
    access = pd.read_csv(release / "protected_outcome_access_audit.tsv", sep="\t", dtype=str)
    assert not access.relative_path.str.startswith("server_phase5_parity_bundle/").any()
    assert not access.decision.eq("DENY").any()
    replay = pd.read_csv(release / "deterministic_replay_validation.tsv", sep="\t")
    assert replay.status.eq("PASS").all()


def test_atomic_decision_if_finalized(release: Path) -> None:
    path = release / "PHASE5_CIMMYT_UNIMPUTED_DECISION.json"
    if not path.exists():
        pytest.skip("Predecision test run")
    decision = json.loads(path.read_text(encoding="utf-8"))
    assert decision["release_id"] == RELEASE_ID
    assert decision["status"] == (
        "PASS_CIMMYT_UNIMPUTED_ANALYSIS_WITH_GLOBAL_MARKER_UNIVERSE_BLOCKER"
    )
    assert decision["diagnostic_split_local_states_constructed"] == 121
    assert decision["explicit_support_or_qc_state_masks"] == 29
    assert decision["strict_production_states_activated"] == 0
    assert decision["pre_qc_marker_universe_recovered"] is False
    assert decision["model_training_performed"] is False
    assert decision["inner_validation_metrics_accessed"] is False
    assert decision["outer_test_outcomes_accessed"] is False
    assert decision["final_holdout_accessed"] is False
