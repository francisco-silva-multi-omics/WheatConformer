"""Build review-incorporated Phase-2 closure tables from immutable diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq


SELECTED = {
    "1000_GRAIN_WEIGHT", "ABOVE_GROUND_BIOMASS", "DAYS_TO_HEADING",
    "DAYS_TO_MATURITY", "GRAIN_YIELD", "PLANT_HEIGHT", "TEST_WEIGHT",
}


def clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def norm(series: pd.Series) -> pd.Series:
    return clean(series).str.upper().str.replace(r"\s+", " ", regex=True)


def cycle_year(series: pd.Series) -> pd.Series:
    raw = clean(series)
    return raw.str.extract(r"(\d{4})", expand=False).fillna(raw)


def clean_id(series: pd.Series) -> pd.Series:
    return clean(series).str.replace(r"\.0$", "", regex=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)

    forensic = json.loads((root / "phase2_forensic_summary.json").read_text(encoding="utf-8"))
    raw_id = json.loads((root / "identity_amendment_v1/raw_row_id_correction_summary.json").read_text(encoding="utf-8"))
    doi = json.loads((root / "doi_glis_audit_v3/doi_glis_audit_summary.json").read_text(encoding="utf-8"))
    immutability = json.loads((root / "raw_phase1_to_phase2_comparison_summary.json").read_text(encoding="utf-8"))

    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str, low_memory=False).fillna("")
    manifest["resolver_key"] = (
        norm(manifest["trial_name"]) + "\x1f" + cycle_year(manifest["cycle"]) + "\x1f"
        + clean(manifest["occ"]) + "\x1f" + clean_id(manifest["CID"]) + "\x1f" + clean_id(manifest["SID"])
    )
    chosen = manifest.drop_duplicates("resolver_key", keep="first").copy()
    chosen["conflict"] = norm(chosen["fieldbook_glis_gid_conflict"]).isin({"TRUE", "1", "YES"})
    conflict_map = chosen.set_index("resolver_key")["conflict"]
    raw_table = pq.read_table(
        root / "raw_row_disposition_ledger.parquet",
        columns=[
            "resolver_key", "final_raw_disposition", "genotype_id_class",
            "expected_stage1_observation_id", "trait_name_canonical",
        ],
    )
    retained = raw_table.filter(
        pc.and_(
            pc.equal(raw_table["final_raw_disposition"], "RETAINED_CONTRIBUTES_TO_STAGE1"),
            pc.equal(raw_table["genotype_id_class"], "MANIFEST_RESOLVED"),
        )
    ).to_pandas()
    retained["fieldbook_glis_conflict"] = retained["resolver_key"].map(conflict_map).fillna(False)
    conflict_rows = retained[retained["fieldbook_glis_conflict"]]
    conflict_raw = len(conflict_rows)
    conflict_stage1 = int(conflict_rows["expected_stage1_observation_id"].nunique())
    conflict_selected = int(
        conflict_rows.loc[norm(conflict_rows["trait_name_canonical"]).isin(SELECTED), "expected_stage1_observation_id"].nunique()
    )

    defects = pd.read_csv(root / "refinement_v2/confirmed_pipeline_defects_final.tsv", sep="\t", dtype=str).fillna("")
    defects.loc[defects["defect_id"].eq("D2-010"), "status"] = "CONFIRMED_POLICY_GAP_NO_SELECTED_TRAIT_ROWS_AFFECTED"
    defects.loc[defects["defect_id"].eq("D2-010"), "defect"] = (
        "No trait-specific zero/sentinel policy; 55,878 eligible all-trait rows require a rule, "
        "but selected-trait Stage-1 contributors contain zero such rows"
    )
    additions = pd.DataFrame([
        {
            "defect_id": "D2-020",
            "status": "CONFIRMED_DIAGNOSTIC_DEFECT_CORRECTED_IN_FINAL_LEDGER",
            "affected_rows": raw_id["provisional_excess_duplicate_ids"],
            "defect": "Provisional RAW2 ID omitted logical source path; byte-identical source files collided",
            "required_rebuild_action": "Use protocol amendment 001 and the collision-free final raw ledger; include logical source path.",
        },
        {
            "defect_id": "D2-021",
            "status": "CONFIRMED_REPRODUCIBILITY_GAP",
            "affected_rows": doi["stage1_observations_resolved_by_glis_doi"],
            "defect": f"GLIS/DOI-derived identity contributes {doi['stage1_observations_resolved_by_glis_doi']:,} Stage-1 observations ({doi['selected_stage1_observations_resolved_by_glis_doi']:,} selected), but the referenced resolve_all_trial_gids.py producer is absent",
            "required_rebuild_action": "Recover/version the exact resolver and bind local DOI input, GLIS response provenance, code hash and response timestamp/hash.",
        },
        {
            "defect_id": "D2-022",
            "status": "CONFIRMED_OVER_SPECIFIC_IDENTITY_JOIN_REVIEW_REQUIRED",
            "affected_rows": doi["candidate_rows_after_relaxing_occurrence_only"],
            "defect": "The genotype resolver requires occurrence although 1,530,430 unresolved numeric rows have a unique valid-DOI GID at identical trial/cycle/CID/SID when only occurrence is relaxed",
            "required_rebuild_action": "Domain-approve trial-wide DOI identity scope, then join at the approved grain and ledger every recovered row; do not auto-accept in Phase 2.",
        },
        {
            "defect_id": "D2-023",
            "status": "CONFIRMED_DOI_INPUT_COVERAGE_GAP_REVIEW_REQUIRED",
            "affected_rows": doi["doi_records_without_manifest_match"],
            "defect": f"Only {doi['doi_source_files_represented_in_manifest']} of {doi['doi_files']} local Germplasm DOI files are represented in the manifest; {doi['doi_records_without_manifest_match']:,} DOI rows lack exact file/CID/SID linkage",
            "required_rebuild_action": "Account for all 127 DOI files and classify every row as applied, duplicate, conflicting, unsupported, or not applicable.",
        },
        {
            "defect_id": "D2-024",
            "status": "CONFIRMED_AMBIGUITY",
            "affected_rows": doi["doi_to_multiple_gid_conflicts"],
            "defect": "Ninety-five syntactically valid DOI values are associated with multiple resolved GIDs in the manifest",
            "required_rebuild_action": "Adjudicate DOI-to-GID conflicts with preserved GLIS response and fieldbook evidence; fail closed meanwhile.",
        },
        {
            "defect_id": "D2-025",
            "status": "CONFIRMED_POLICY_AMBIGUITY_ZERO_CURRENT_STAGE1_ROWS" if conflict_stage1 == 0 else "CONFIRMED_AMBIGUITY_REACHES_STAGE1",
            "affected_rows": conflict_stage1,
            "defect": f"The manifest flags 90 fieldbook-versus-GLIS GID conflicts; exact legacy keys reach {conflict_stage1:,} Stage-1 observations ({conflict_selected:,} selected)",
            "required_rebuild_action": "Place conflicts in a human-review state before Stage-1; do not accept the fieldbook GID silently.",
        },
    ])
    defects = pd.concat([defects, additions], ignore_index=True)
    defects.to_csv(result_dir / "confirmed_pipeline_defects_final.tsv", sep="\t", index=False)

    review = pd.read_csv(root / "refinement_v2/unresolved_human_review_final.tsv", sep="\t", dtype=str).fillna("")
    review = pd.concat([
        review,
        pd.DataFrame([
            {
                "priority": "P0", "review_class": "DOI_IDENTITY_OCCURRENCE_SCOPE",
                "items": doi["candidate_key_groups_after_relaxing_cycle_occ"],
                "affected_rows": doi["candidate_rows_after_relaxing_occurrence_only"],
                "detail_artifact": "doi_glis_audit_v3/doi_glis_unresolved_candidate_ledger.tsv",
                "decision_needed": "Confirm whether a trial/cycle/CID/SID DOI-to-GID mapping is trial-wide across occurrences.",
            },
            {
                "priority": "P1" if conflict_stage1 == 0 else "P0", "review_class": "FIELDBOOK_GLIS_GID_CONFLICT",
                "items": 90, "affected_rows": conflict_stage1,
                "detail_artifact": "doi_glis_audit_v3/doi_to_gid_conflicts.tsv; manifest fieldbook_glis_gid_conflict",
                "decision_needed": "Adjudicate fieldbook versus GLIS GID; current keep-first fieldbook result must not be treated as v2 truth.",
            },
            {
                "priority": "P0", "review_class": "DOI_TO_MULTIPLE_GID",
                "items": doi["doi_to_multiple_gid_conflicts"], "affected_rows": doi["doi_to_multiple_gid_conflicts"],
                "detail_artifact": "doi_glis_audit_v3/doi_to_gid_conflicts.tsv",
                "decision_needed": "Determine authoritative GID per DOI or mark unusable; no silent resolution.",
            },
            {
                "priority": "P0", "review_class": "DOI_FILE_MANIFEST_COVERAGE",
                "items": doi["doi_files"] - doi["doi_source_files_represented_in_manifest"],
                "affected_rows": doi["doi_records_without_manifest_match"],
                "detail_artifact": "doi_glis_audit_v3/doi_file_inventory.tsv; doi_to_manifest_linkage.parquet",
                "decision_needed": "Classify why 72 DOI files are absent from exact manifest linkage and whether each is applicable.",
            },
            {
                "priority": "P0", "review_class": "GLIS_RESOLVER_PROVENANCE",
                "items": doi["manifest_rows_resolved_by_glis_doi"],
                "affected_rows": doi["stage1_observations_resolved_by_glis_doi"],
                "detail_artifact": "doi_glis_audit_v3/doi_glis_stage1_impact.tsv",
                "decision_needed": "Supply/version the missing resolver code and immutable GLIS response evidence for the 484 GLIS-resolved manifest rows.",
            },
        ]),
    ], ignore_index=True)
    review.to_csv(result_dir / "unresolved_human_review_final.tsv", sep="\t", index=False)

    joins = pd.read_csv(root / "join_cardinality_report.tsv", sep="\t", dtype=str).fillna("")
    joins = pd.concat([
        joins,
        pd.DataFrame([
            {
                "join_name": "local_DOI_files_to_phase1_hash_inventory", "left_rows": doi["doi_files"],
                "right_rows": 2662, "left_unique_keys": doi["doi_files"], "right_unique_keys": 2662,
                "expected_cardinality": "m:1", "matched_left_rows": doi["doi_files"], "unmatched_left_rows": 0,
                "right_duplicate_keys": 0, "output_rows": doi["doi_files"],
            },
            {
                "join_name": "local_DOI_records_to_manifest_file_CID_SID", "left_rows": doi["doi_records"],
                "right_rows": doi["manifest_rows"], "left_unique_keys": "record locator", "right_unique_keys": "grouped join key",
                "expected_cardinality": "m:1", "matched_left_rows": doi["doi_records"] - doi["doi_records_without_manifest_match"],
                "unmatched_left_rows": doi["doi_records_without_manifest_match"], "right_duplicate_keys": "grouped before join",
                "output_rows": doi["doi_records"],
            },
            {
                "join_name": "unresolved_numeric_to_valid_DOI_trial_cycle_CID_SID_ignoring_occurrence_DIAGNOSTIC_ONLY",
                "left_rows": forensic["raw_to_stage1_reconciliation"]["unresolved_gid_rows_after_numeric_filter"],
                "right_rows": doi["manifest_rows_with_valid_DOI"], "left_unique_keys": "not materialized",
                "right_unique_keys": "trial/cycle/CID/SID registry", "expected_cardinality": "m:0..1 review only",
                "matched_left_rows": doi["candidate_rows_after_relaxing_occurrence_only"],
                "unmatched_left_rows": forensic["raw_to_stage1_reconciliation"]["unresolved_gid_rows_after_numeric_filter"] - doi["candidate_rows_after_relaxing_occurrence_only"],
                "right_duplicate_keys": 0, "output_rows": forensic["raw_to_stage1_reconciliation"]["unresolved_gid_rows_after_numeric_filter"],
            },
        ]),
    ], ignore_index=True)
    joins.to_csv(result_dir / "join_cardinality_report_final.tsv", sep="\t", index=False)

    waterfall = pd.read_csv(root / "attrition_waterfall.tsv", sep="\t", dtype=str).fillna("")
    diagnostic_waterfall = pd.DataFrame([
        {"pipeline": "doi_identity_review", "step_order": 1, "step": "numeric_rows_unresolved_by_legacy_identity", "row_grain": "raw row", "rows": 6691857, "lost_from_prior": 0},
        {"pipeline": "doi_identity_review", "step_order": 2, "step": "unique_valid_DOI_GID_candidate_when_occurrence_only_relaxed", "row_grain": "raw row; not accepted", "rows": doi["candidate_rows_after_relaxing_occurrence_only"], "lost_from_prior": 6691857 - doi["candidate_rows_after_relaxing_occurrence_only"]},
        {"pipeline": "doi_identity_review", "step_order": 3, "step": "remaining_without_this_DOI_candidate", "row_grain": "raw row", "rows": 6691857 - doi["candidate_rows_after_relaxing_occurrence_only"], "lost_from_prior": doi["candidate_rows_after_relaxing_occurrence_only"]},
    ])
    pd.concat([waterfall, diagnostic_waterfall], ignore_index=True).to_csv(
        result_dir / "attrition_waterfall_final.tsv", sep="\t", index=False
    )

    expected_observed = pd.DataFrame([
        ("raw_rows", 7836162, forensic["raw_rows"]),
        ("numeric_rows", 7273254, forensic["numeric_rows"]),
        ("eligible_stage1_raw_rows", 581397, forensic["eligible_stage1_raw_rows"]),
        ("stage1_rows_all_traits", 433626, forensic["stage1_rows"]),
        ("stage1_rows_selected_traits", 278001, forensic["selected_stage1_rows"]),
        ("canonical_rows_all_traits", 2938384, forensic["canonical_rows"]),
        ("canonical_rows_selected_traits", 2022291, forensic["canonical_selected_rows"]),
        ("canonical_permanent_ids_distinct", 2938384, forensic["canonical_distinct_permanent_row_ids"]),
        ("final_raw_permanent_ids_distinct", 7836162, raw_id["final_distinct_ids"]),
        ("stage1_id_count_reconstructed", 433626, forensic["raw_to_stage1_reconciliation"]["distinct_reconstructed_stage1_ids"]),
        ("stage1_plot_record_count_mismatches", 0, forensic["raw_to_stage1_reconciliation"]["stage1_plot_count_mismatches"]),
        ("doi_files_parsed", 127, doi["doi_files"] - doi["doi_file_parse_failures"]),
        ("raw_files_immutable", 2754, sum(item["status_counts"].get("MATCH", 0) for item in immutability)),
    ], columns=["metric", "expected", "observed"])
    expected_observed["status"] = expected_observed["expected"].eq(expected_observed["observed"]).map({True: "PASS", False: "FAIL"})
    expected_observed.to_csv(result_dir / "phase2_expected_vs_observed_counts.tsv", sep="\t", index=False)

    legitimate = pd.DataFrame([
        {"category": "TEXTUAL_OR_BLANK_MISSING_VALUE", "rows": 555238, "classification": "MODEL_EXCLUSION_WITH_LEDGER", "rationale": "No numeric phenotype is available; raw token remains traceable."},
        {"category": "UNRECOGNIZED_NONNUMERIC_VALUE", "rows": 7670, "classification": "HUMAN_REVIEW_NOT_YET_LEGITIMATE", "rationale": "Categorical or malformed tokens require source/trait semantics."},
        {"category": "UNRESOLVED_GENOTYPE", "rows": 6691857, "classification": "FAIL_CLOSED_PENDING_IDENTITY_REVIEW", "rationale": "No GID may be invented; 1,530,430 rows have occurrence-scope DOI candidates but remain unaccepted."},
        {"category": "RAW_TO_STAGE1_MANY_TO_ONE_ADJUSTMENT", "rows": 147771, "classification": "LEGITIMATE_AGGREGATION_WITH_CONTRIBUTION_BRIDGE", "rationale": "581,397 contributor rows yield 433,626 Stage-1 rows and exact n_plot_records reconciliation."},
        {"category": "TRAIT_OUTSIDE_SELECTED_SEVEN", "rows": 155625, "classification": "LEGITIMATE_MODEL_SCOPE_ONLY", "rationale": "Rows remain in all-trait Stage 1."},
        {"category": "INVALID_OR_NONPOSITIVE_WEIGHT", "rows": 59, "classification": "NOT_A_PHENOTYPE_EXCLUSION", "rationale": "Retain outcome; weight treatment is fold-local."},
        {"category": "PEDIGREE_OR_MARKER_UNAVAILABLE", "rows": 0, "classification": "NO_CURRENT_STAGE1_ATTRITION", "rationale": "All selected Stage-1 rows are pedigree-and-marker supported."},
        {"category": "OUTLIER_REMOVAL", "rows": 0, "classification": "NOT_PRESENT_IN_LEGACY_STAGE1", "rationale": "No legacy outlier filter was found."},
    ])
    legitimate.to_csv(result_dir / "legitimate_exclusion_categories_final.tsv", sep="\t", index=False)

    protected = {
        "certified_v1_artifacts_modified": False,
        "outer_test_content_read": False,
        "outer_test_information_used_for_selection": False,
        "final_holdout_identifiers_or_membership_read": False,
        "final_holdout_outcomes_predictions_or_summaries_read": False,
        "stage1_rebuilt": False,
        "models_trained": False,
    }
    (result_dir / "protected_access_report.json").write_text(json.dumps(protected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "status": "PASS_PHASE2_DIAGNOSTIC_CLOSURE_TABLES_CREATED",
        "confirmed_or_policy_defect_rows": len(defects),
        "human_review_classes": len(review),
        "fieldbook_glis_conflict_contributing_raw_rows": conflict_raw,
        "fieldbook_glis_conflict_stage1_observations": conflict_stage1,
        "fieldbook_glis_conflict_selected_stage1_observations": conflict_selected,
        "all_expected_counts_pass": bool(expected_observed["status"].eq("PASS").all()),
        "raw_immutability_pass": all(set(item["status_counts"]) == {"MATCH"} for item in immutability),
        **protected,
    }
    (result_dir / "phase2_closure_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not summary["all_expected_counts_pass"] or not summary["raw_immutability_pass"]:
        raise RuntimeError("Phase-2 closure validation failed")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
