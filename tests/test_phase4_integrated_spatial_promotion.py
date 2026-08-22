from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from scripts.v2.phase4_coordinate_recovery import semantic
from scripts.v2.phase4_integrated_build_promotion import stable_id


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "audit/v2/phase4_integrated_spatial_promotion_release_v1"
TRAIN = "P4ISP_20260802_V1_274E41DF"


def j(name: str):
    return json.loads((RELEASE / name).read_text(encoding="utf-8"))


def t(name: str):
    return pd.read_csv(RELEASE / name, sep="\t", low_memory=False)


def test_01_phase4_v1_starting_state_reproduced():
    x = t("phase4_v1_starting_state_reproduction.tsv")
    assert len(x) >= 20 and set(x.status) == {"PASS"}


def test_02_complete_raw_source_coordinate_search():
    x = j("coordinate_scan_summary.json")
    assert x["raw_top_level_artifacts"] == x["raw_top_level_artifacts_accounted"] == 2662


def test_03_coordinate_provenance_and_plot_join_validity():
    x = t("coordinate_coverage_summary.tsv").iloc[0]
    assert x.phase4_plot_records == 4_226_848
    assert pq.ParquetFile(RELEASE / "plot_coordinate_crosswalk.parquet").metadata.num_rows > 0


def test_04_arbitrary_plot_order_reshape_rejected():
    assert j("coordinate_scan_summary.json")["arbitrary_plot_reshape_performed"] is False
    assert "floor" not in j("coordinate_adjudication.json")["accepted_transformation_rules"]


def test_05_serpentine_reset_incomplete_conflict_not_assumed():
    a = j("coordinate_adjudication.json")
    required = {"SERPENTINE_UNKNOWN", "RESET_RULE_UNKNOWN", "INCOMPLETE_GRID_UNKNOWN", "CONFLICTS_NOT_SILENTLY_RESOLVED"}
    assert required.issubset(set(a["rejected_inference_reason_codes"]))


def test_06_conditional_authoritative_candidate_selection():
    a, p = j("coordinate_adjudication.json"), j("authoritative_phase4_pointer.json")
    assert a["coordinate_outcome"] == "NO_VALID_COORDINATES_FOUND"
    assert p["phenotype_correction_required"] is False


def test_07_no_v1_fallback_when_valid_coordinate_requires_correction():
    source = (ROOT / "scripts/v2/phase4_integrated_build_promotion.py").read_text(encoding="utf-8")
    assert "Valid coordinates require a complete corrected Phase-4 candidate" in source


def test_08_all_adjusted_records_conserved():
    assert pq.ParquetFile(RELEASE / "promoted_phenotypes.parquet").metadata.num_rows == 3_193_677


def test_09_all_groups_accounted():
    assert pq.ParquetFile(RELEASE / "group_promotion_ledger.parquet").metadata.num_rows == 37_206


def test_10_unaffected_records_exact():
    x = t("phase4_v1_to_candidate_comparison.tsv").iloc[0]
    assert x.unaffected_records_exactly_identical == 3_193_677 and x.changed_adjusted_values == 0


def test_11_affected_change_reporting_complete():
    x = t("phase4_v1_to_candidate_comparison.tsv").iloc[0]
    assert x.changed_adjusted_values == x.changed_uncertainty_records == x.changed_model_groups == 0


def test_12_adjusted_values_preserved():
    x = t("phase4_to_promoted_row_reconciliation.tsv").iloc[0]
    assert x.changed_adjusted_values == 0


def test_13_deregression_prevented():
    x = t("phase4_to_promoted_row_reconciliation.tsv").iloc[0]
    assert x.deregressed_recommended_targets == 0


def test_14_status_difference_2176_resolved():
    x = t("ranking_status_reconciliation.tsv")
    overlap = int(x.loc[x.intersection_state.eq("OVERLAP_CEILING_ESTIMABLE_BUT_UNSUITABLE"), "groups"].sum())
    neither = int(x.loc[x.intersection_state.eq("CEILING_NOT_ESTIMABLE_BUT_NOT_CLASSIFIED_UNSUITABLE"), "groups"].sum())
    assert (overlap, neither, neither - overlap) == (4342, 6518, 2176)


def test_15_unresolved_identity_never_canonical_eligible():
    x = t("unresolved_identity_phase4_footprint.tsv")
    overall = x[(x.scope_type == "OVERALL") & (x.scope_value == "ALL")].iloc[0]
    assert overall.inadvertently_promoted_to_canonical_gid == 0


def test_16_phase3g_v1_not_consumed():
    opening = t("OPENING_HASH_MANIFEST.tsv")
    assert not opening.path.str.contains("phase3g_all_panel_genotype_linkage_audit_v1", regex=False).any()


def test_17_ranking_unsuitable_never_ranking_eligible():
    definitions = j("promotion_view_definitions.json")
    assert any(v["view"] == "RANKING_EVALUATION" for v in definitions["views"])
    x = t("trial_trait_status_crosswalk.tsv")
    bad = x.ranking_status.str.startswith("TOO_UNRELIABLE", na=False) & (x.ranking_evaluation_eligible_records > 0)
    assert not bad.any()


def test_18_continuous_error_distinct_from_ranking():
    s = t("promotion_view_population_summary.tsv")
    s = s[s.summary_scope == "OVERALL"].set_index("view")
    assert s.loc["CONTINUOUS_ERROR_EVALUATION", "rows"] > s.loc["RANKING_EVALUATION", "rows"]


def test_19_no_invented_reliability_weight():
    p = j("promotion_policy.json")
    assert p["reliability_threshold"].startswith("none invented")


def test_20_invalid_uncertainty_explicitly_restricted():
    d = t("restriction_reason_dictionary.tsv")
    assert {"PEV_NONESTIMABLE", "PEV_NEGATIVE", "RELIABILITY_NONESTIMABLE", "RELIABILITY_OUT_OF_BOUNDS"}.issubset(set(d.reason_code))


def test_21_unadjusted_means_not_rejected_by_class():
    x = t("uncertainty_metadata_validation.tsv")
    u = x[x.selected_model == "UNADJUSTED_GENOTYPE_MEANS"].iloc[0]
    assert u.uncertainty_weight_eligible > 0


def test_22_check_codes_are_contextual():
    x = t("check_code_conflict_impact.tsv")
    assert x.status_is_contextual.all() and not x.affected_primary_estimation.any()


def test_23_huber_nonconvergence_does_not_overwrite_primary():
    x = t("huber_nonconvergence_impact.tsv")
    m = x[x.huber_status == "MAX_ITER"].iloc[0]
    assert m.groups == 1357 and not m.overwrote_primary_adjusted_value


def test_24_stable_deterministic_row_ids():
    assert stable_id("X_", "a", 1) == stable_id("X_", "a", 1)
    x = t("phase4_to_promoted_row_reconciliation.tsv").iloc[0]
    assert x.duplicate_promoted_ids == 0


def test_25_complete_reason_code_coverage():
    d = set(t("restriction_reason_dictionary.tsv").reason_code)
    x = t("restriction_overlap_summary.tsv")
    marginal = set(x.loc[x.summary_type == "MARGINAL_REASON", "reason_code"])
    assert marginal.issubset(d)


def test_26_deterministic_view_regeneration_contract():
    x = j("promotion_view_definitions.json")
    assert len(x["views"]) == 8 and x["source"] == "promoted_phenotypes.parquet"


def test_27_source_artifacts_unmodified():
    d = j("RELEASE_DECISION.json")
    assert d["protected_source_hashes_match"] is True


def test_28_opening_closing_hashes_agree():
    d = j("RELEASE_DECISION.json")
    assert d["opening_closing_hash_mismatch_count"] == 0


def test_29_no_protected_outcome_access():
    m = j("run_manifest.json")
    assert not m["outer_test_content_accessed"] and not m["final_holdout_content_accessed"]


def test_30_deterministic_replay_passed():
    d = j("RELEASE_DECISION.json")
    assert d["deterministic_replay_passed"] is True


def test_31_release_train_id_consistent():
    assert j("run_manifest.json")["release_train_id"] == j("RELEASE_DECISION.json")["release_train_id"] == TRAIN


def test_32_no_independent_component_pass_statuses():
    d = j("RELEASE_DECISION.json")
    assert d["component_authoritative_statuses_emitted"] is False


def test_coordinate_semantics_are_case_and_separator_stable():
    assert semantic("FIELD_ROW") == "FIELD_ROW"
    assert semantic("Plot-Col") == "FIELD_COLUMN"


def test_pointer_does_not_mix_versions():
    assert j("authoritative_phase4_pointer.json")["mixed_version"] is False


def test_phase5_not_started():
    assert j("run_manifest.json")["phase5_started"] is False
