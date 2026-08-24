"""Refine Phase-2 classifications without modifying the initial forensic outputs."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v2.phase2_forensic_stage1_audit import (
    BatchParquetWriter,
    SELECTED_TRAITS,
    canonical_gid,
    clean_text,
    natural_key,
    norm_text,
    write_tsv,
)


def compact_key(*values: pd.Series) -> pd.Series:
    output = values[0].astype(str)
    for value in values[1:]:
        output = output + "\x1f" + value.astype(str)
    return output


def truth(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return norm_text(series, upper=True).isin({"TRUE", "1", "YES", "Y"})


def group_dimension(parts: list[pd.DataFrame], frame: pd.DataFrame) -> None:
    dimensions = {
        "source_class": "phenotype_source",
        "trial": "trial_name",
        "cycle": "cycle",
        "occurrence": "occ",
        "trait": "trait_name_canonical",
        "environment": "env_kernel_id",
        "genotype_id_class": "genotype_modality_class",
        "transformation_step": "final_canonical_disposition_v2",
    }
    for dimension, column in dimensions.items():
        if column == "final_canonical_disposition_v2":
            grouped = (
                frame.groupby(column, dropna=False, sort=False)
                .size().rename("rows").reset_index()
                .rename(columns={column: "disposition"})
            )
            grouped["dimension_value"] = grouped["disposition"]
        else:
            grouped = (
                frame.groupby([column, "final_canonical_disposition_v2"], dropna=False, sort=False)
                .size().rename("rows").reset_index()
                .rename(columns={column: "dimension_value", "final_canonical_disposition_v2": "disposition"})
            )
        grouped.insert(0, "dimension", dimension)
        grouped.insert(0, "ledger_scope", "canonical")
        parts.append(grouped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=200_000)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    result_dir = args.result_dir.resolve() if args.result_dir else out_dir
    result_dir.mkdir(parents=True, exist_ok=False)

    raw_ledger_path = out_dir / "raw_row_disposition_ledger.parquet"
    raw_columns = [
        "source_file", "trial_name", "cycle", "occ", "env_kernel_id",
        "resolved_gid", "genotype_id_class", "trait_name_original",
        "trait_name_canonical", "trait_mapping_status", "raw_unit", "unit",
        "unit_conflict_with_selected_mapping", "raw_value_token", "numeric_zero",
        "numeric_sentinel_candidate", "identifier_normalization_changed",
        "final_raw_disposition",
    ]
    raw_table = pq.read_table(raw_ledger_path, columns=raw_columns)
    retained_table = raw_table.filter(
        pc.equal(raw_table["final_raw_disposition"], "RETAINED_CONTRIBUTES_TO_STAGE1")
    )
    retained = retained_table.to_pandas()
    del raw_table, retained_table
    retained["selected_trait"] = norm_text(retained["trait_name_canonical"], upper=True).isin(SELECTED_TRAITS)
    retained["trait_mapping_ambiguous"] = clean_text(retained["trait_mapping_status"]).str.startswith("AMBIGUOUS")
    retained["unit_conflict"] = truth(retained["unit_conflict_with_selected_mapping"])
    retained["numeric_zero"] = truth(retained["numeric_zero"])
    retained["numeric_sentinel_candidate"] = truth(retained["numeric_sentinel_candidate"])

    trait_unit_summary = (
        retained.groupby(
            ["trait_name_canonical", "trait_mapping_status", "raw_unit", "unit"],
            dropna=False,
            sort=True,
        )
        .agg(
            stage1_input_rows=("resolved_gid", "size"),
            unique_genotypes=("resolved_gid", "nunique"),
            unique_environments=("env_kernel_id", "nunique"),
            source_files=("source_file", "nunique"),
            zero_rows=("numeric_zero", "sum"),
            numeric_sentinel_rows=("numeric_sentinel_candidate", "sum"),
            unit_conflict_rows=("unit_conflict", "sum"),
        )
        .reset_index()
    )
    write_tsv(result_dir / "stage1_input_trait_unit_diagnostics.tsv", trait_unit_summary)

    selected_zero = (
        retained[retained["selected_trait"] & retained["numeric_zero"]]
        .groupby(
            ["source_file", "trial_name", "cycle", "occ", "trait_name_canonical", "unit"],
            dropna=False,
            sort=True,
        )
        .agg(rows=("resolved_gid", "size"), unique_genotypes=("resolved_gid", "nunique"), unique_environments=("env_kernel_id", "nunique"))
        .reset_index()
    )
    write_tsv(result_dir / "selected_trait_zero_review.tsv", selected_zero)
    selected_sentinel = (
        retained[retained["selected_trait"] & retained["numeric_sentinel_candidate"]]
        .groupby(
            ["source_file", "trial_name", "cycle", "occ", "trait_name_canonical", "unit", "raw_value_token"],
            dropna=False,
            sort=True,
        )
        .agg(rows=("resolved_gid", "size"), unique_genotypes=("resolved_gid", "nunique"), unique_environments=("env_kernel_id", "nunique"))
        .reset_index()
    )
    write_tsv(result_dir / "selected_trait_numeric_sentinel_review.tsv", selected_sentinel)

    numeric_failures = pd.read_csv(out_dir / "numeric_parsing_failures.tsv", sep="\t", dtype=str).fillna("")
    numeric_failures["rows"] = pd.to_numeric(numeric_failures["rows"], errors="raise").astype("int64")
    numeric_summary = (
        numeric_failures.groupby("value_token_class", sort=True)
        .agg(
            rows=("rows", "sum"),
            distinct_tokens=("raw_value_token", "nunique"),
            source_files=("source_file", "nunique"),
            trials=("trial_name", "nunique"),
            traits=("trait_name_original", "nunique"),
        )
        .reset_index()
    )
    write_tsv(result_dir / "numeric_parsing_summary.tsv", numeric_summary)

    wide = pd.read_csv(out_dir / "wide_to_long_omission_candidates.tsv", sep="\t", dtype=str).fillna("")
    auxiliary = re.compile(
        r"^(TRAIT_NO|GEN_NO|GID|ENTRY|PLOT)(\.\d+)?$|^UNNAMED_?\d*$", re.I
    )
    wide["final_assessment"] = np.where(
        wide["column_name"].map(lambda value: bool(auxiliary.match(value.replace(" ", "_")))),
        "AUXILIARY_IDENTIFIER_OR_ROW_METADATA_NOT_A_WIDE_PHENOTYPE",
        "POSSIBLE_WIDE_PHENOTYPE_REQUIRES_HUMAN_REVIEW",
    )
    write_tsv(result_dir / "wide_to_long_omission_assessment_final.tsv", wide)

    unresolved = pd.read_csv(
        out_dir / "unresolved_genotype_alias_candidates.tsv", sep="\t", dtype=str,
        low_memory=False,
    ).fillna("")
    unresolved["rows"] = pd.to_numeric(unresolved["rows"], errors="raise").astype("int64")
    priority = (
        unresolved.groupby(["source_file", "trial_name", "cycle", "occ"], dropna=False, sort=True)
        .agg(
            unresolved_numeric_rows=("rows", "sum"),
            unresolved_candidate_groups=("rows", "size"),
            distinct_CID=("CID", lambda x: clean_text(x).replace("", np.nan).nunique()),
            distinct_SID=("SID", lambda x: clean_text(x).replace("", np.nan).nunique()),
            distinct_genotype_names=("genotype_name", lambda x: clean_text(x).replace("", np.nan).nunique()),
            rows_with_any_raw_gid=("raw_gid", lambda x: int(clean_text(x).ne("").sum())),
            rows_with_ignored_alternate_gid=("alternate_gid_ignored", lambda x: int(clean_text(x).ne("").sum())),
        )
        .reset_index()
        .sort_values(["unresolved_numeric_rows", "source_file"], ascending=[False, True])
        .reset_index(drop=True)
    )
    write_tsv(result_dir / "unresolved_genotype_priority.tsv", priority)
    del unresolved

    stage1_path = root / "server_phase1_bundle/artifacts/phenotypes/stage1_adjusted_phenotypes.parquet"
    stage1 = pd.read_parquet(
        stage1_path,
        columns=[
            "canonical_observation_id", "canonical_germplasm_key", "env_kernel_id",
            "env_id_pheno", "trait_name_canonical", "trait_name_original", "unit",
        ],
    )
    stage1["gid_norm"] = canonical_gid(stage1["canonical_germplasm_key"])
    stage1["env_norm"] = norm_text(stage1["env_kernel_id"])
    stage1["env_pheno_norm"] = norm_text(stage1["env_id_pheno"])
    stage1["trait_norm"] = norm_text(stage1["trait_name_canonical"], upper=True)
    stage1["original_norm"] = norm_text(stage1["trait_name_original"], upper=True)
    stage1["unit_norm"] = norm_text(stage1["unit"], upper=True)
    stage1["analysis_natural_key"] = natural_key(stage1)
    stage1_gid = set(stage1["gid_norm"])
    stage1_env = set(stage1["env_norm"])
    stage1_gid_env = set(compact_key(stage1["gid_norm"], stage1["env_norm"]))
    stage1_gid_env_trait = set(compact_key(stage1["gid_norm"], stage1["env_norm"], stage1["trait_norm"]))
    stage1_no_unit = set(compact_key(stage1["gid_norm"], stage1["env_norm"], stage1["trait_norm"], stage1["original_norm"]))
    stage1_gid_trait = set(compact_key(stage1["gid_norm"], stage1["trait_norm"]))

    canonical_initial = out_dir / "canonical_row_disposition_ledger.parquet"
    canonical_final = result_dir / "canonical_row_disposition_ledger_final.parquet"
    writer = BatchParquetWriter(canonical_final)
    canonical_keys: set[str] = set()
    reason_counts: Counter[str] = Counter()
    canonical_dimension_parts: list[pd.DataFrame] = []
    parquet_file = pq.ParquetFile(canonical_initial)
    for batch in parquet_file.iter_batches(batch_size=args.batch_size):
        frame = batch.to_pandas()
        canonical_keys.update(clean_text(frame["analysis_natural_key"]))
        gid = canonical_gid(frame["canonical_germplasm_key"])
        env = norm_text(frame["env_kernel_id"])
        trait = norm_text(frame["trait_name_canonical"], upper=True)
        original = norm_text(frame["trait_name_original"], upper=True)
        unit = norm_text(frame["unit"], upper=True)
        gid_env = compact_key(gid, env)
        gid_env_trait = compact_key(gid, env, trait)
        no_unit = compact_key(gid, env, trait, original)
        gid_trait = compact_key(gid, trait)
        matched = truth(frame["stage1_key_available"])
        source_level = norm_text(frame["source_level"], upper=True)
        raw_numeric = pd.to_numeric(frame["raw_numeric_records"], errors="coerce").fillna(0)
        raw_linked_unmatched = ~matched & ~source_level.eq("SUMMARY_LEVEL") & raw_numeric.gt(0)
        near_reason = np.select(
            [
                ~gid.isin(stage1_gid),
                ~env.isin(stage1_env),
                no_unit.isin(stage1_no_unit),
                gid_env_trait.isin(stage1_gid_env_trait),
                gid_env.isin(stage1_gid_env),
                gid_trait.isin(stage1_gid_trait),
            ],
            [
                "GENOTYPE_NOT_PRESENT_IN_STAGE1",
                "ENVIRONMENT_NOT_PRESENT_IN_STAGE1",
                "UNIT_MISMATCH",
                "ORIGINAL_TRAIT_OR_UNIT_MISMATCH",
                "TRAIT_MAPPING_MISMATCH",
                "ENVIRONMENT_KEY_MISMATCH",
            ],
            default="NO_STAGE1_NEAR_MATCH",
        )
        nonmatch_reason = np.select(
            [
                matched,
                source_level.eq("SUMMARY_LEVEL"),
                raw_numeric.le(0),
                raw_linked_unmatched,
            ],
            [
                "NOT_APPLICABLE_STAGE1_KEY_MATCHED",
                "SUMMARY_LEVEL_PARALLEL_BRANCH",
                "NO_NUMERIC_RAW_STAGE1_INPUT",
                near_reason,
            ],
            default="UNRESOLVED_NONMATCH_REASON",
        )
        frame["stage1_nonmatch_reason"] = nonmatch_reason
        existing = clean_text(frame["final_canonical_disposition"])
        frame["final_canonical_disposition_v2"] = np.where(
            matched,
            existing,
            np.where(
                source_level.eq("SUMMARY_LEVEL"),
                "NOT_RECONSTRUCTED_SUMMARY_LEVEL_PARALLEL_BRANCH",
                np.where(
                    raw_numeric.le(0),
                    "NOT_RECONSTRUCTED_NO_NUMERIC_RAW_STAGE1_INPUT",
                    "NOT_RECONSTRUCTED_RAW_LINKED_" + pd.Series(near_reason, index=frame.index),
                ),
            ),
        )
        reason_counts.update(frame["final_canonical_disposition_v2"])
        group_dimension(canonical_dimension_parts, frame)
        writer.write(frame)
    writer.close()

    missing_canonical = stage1[~stage1["analysis_natural_key"].isin(canonical_keys)].copy()
    missing_canonical["selected_trait"] = missing_canonical["trait_norm"].isin(SELECTED_TRAITS)
    write_tsv(
        result_dir / "stage1_without_canonical_natural_key.tsv",
        missing_canonical.sort_values(["selected_trait", "trait_norm", "canonical_observation_id"], ascending=[False, True, True]),
    )

    final_summary = pd.DataFrame([
        {"final_canonical_disposition_v2": key, "canonical_rows": count}
        for key, count in sorted(reason_counts.items())
    ]).sort_values("canonical_rows", ascending=False)
    write_tsv(result_dir / "canonical_final_disposition_summary.tsv", final_summary)

    canonical_dims = pd.concat(canonical_dimension_parts, ignore_index=True)
    canonical_dims["dimension_value"] = clean_text(canonical_dims["dimension_value"])
    canonical_dims = (
        canonical_dims.groupby(
            ["ledger_scope", "dimension", "dimension_value", "disposition"],
            dropna=False,
            sort=True,
        )["rows"].sum().reset_index()
    )
    initial_dims = pd.read_csv(out_dir / "attrition_by_dimension.tsv", sep="\t", dtype=str).fillna("")
    raw_dims = initial_dims[initial_dims["ledger_scope"].eq("raw_input")].copy()
    raw_dims["rows"] = pd.to_numeric(raw_dims["rows"], errors="raise").astype("int64")
    write_tsv(result_dir / "attrition_by_dimension_final.tsv", pd.concat([raw_dims, canonical_dims], ignore_index=True))

    duplicates = pd.read_csv(out_dir / "raw_plot_duplicate_groups.tsv", sep="\t", dtype=str).fillna("")
    duplicate_excess = int((pd.to_numeric(duplicates["records_with_same_plot_key"], errors="raise") - 1).sum())
    conflicting_duplicate_groups = int(truth(duplicates["conflicting_value_tokens"]).sum())
    resolver_audit = pd.read_csv(out_dir / "genotype_resolver_key_audit.tsv", sep="\t", dtype=str).fillna("")
    resolver_conflicts = int(resolver_audit["resolver_key_status"].str.startswith("DUPLICATE_CONFLICTING").sum())
    eligible_trait_ambiguity = int(retained["trait_mapping_ambiguous"].sum())
    eligible_unit_conflict = int(retained["unit_conflict"].sum())
    eligible_zero = int(retained["numeric_zero"].sum())
    selected_zero_rows = int((retained["selected_trait"] & retained["numeric_zero"]).sum())
    eligible_sentinel = int(retained["numeric_sentinel_candidate"].sum())
    selected_sentinel_rows = int((retained["selected_trait"] & retained["numeric_sentinel_candidate"]).sum())

    defects = pd.DataFrame([
        {"defect_id": "D2-001", "status": "CONFIRMED", "affected_rows": 7836162, "defect": "Legacy concatenation omits persisted source member and physical row", "required_rebuild_action": "Persist immutable source locator and RAW2 ID at ingestion."},
        {"defect_id": "D2-002", "status": "CONFIRMED", "affected_rows": 562908, "defect": "Numeric parse failures are filtered without a row exclusion ledger", "required_rebuild_action": "Retain raw token, parser status, reason, and terminal disposition."},
        {"defect_id": "D2-003", "status": "CONFIRMED_POLICY_DEFECT_ZERO_CURRENT_RAW_ROWS", "affected_rows": 0, "defect": f"Resolver keep-first can adjudicate conflicting keys ({resolver_conflicts} manifest keys)", "required_rebuild_action": "Reject conflicting keys into an ambiguity queue before joining."},
        {"defect_id": "D2-004", "status": "CONFIRMED", "affected_rows": eligible_trait_ambiguity, "defect": "Trait/unit registry keep-first ambiguity reaches Stage-1 inputs", "required_rebuild_action": "Use explicit versioned trait+unit rules and fail on collisions."},
        {"defect_id": "D2-005", "status": "CONFIRMED", "affected_rows": eligible_unit_conflict, "defect": "Mapped units override disagreeing raw units", "required_rebuild_action": "Convert with an approved rule or retain as unresolved; never relabel values."},
        {"defect_id": "D2-006", "status": "CONFIRMED", "affected_rows": duplicate_excess, "defect": f"Plot-key duplicates enter fitting without adjudication; {conflicting_duplicate_groups} groups have conflicting values", "required_rebuild_action": "Emit exact/concordant/conflicting duplicate states before fitting."},
        {"defect_id": "D2-007", "status": "CONFIRMED", "affected_rows": 22609, "defect": "Legacy environment membership filtering precedes the certified alias registry", "required_rebuild_action": "Resolve environment identity before any kernel/order membership filter."},
        {"defect_id": "D2-008", "status": "CONFIRMED", "affected_rows": 59, "defect": "Positive-weight filter removes observations before fold-local recovery", "required_rebuild_action": "Retain outcomes; mark weight missing and fit recovery on inner training rows only."},
        {"defect_id": "D2-009", "status": "NO_AFFECTED_ROWS", "affected_rows": 0, "defect": "No Stage-1 input has missing trial/cycle/occurrence/location key components", "required_rebuild_action": "Keep a mandatory completeness assertion."},
        {"defect_id": "D2-010", "status": "CONFIRMED_POLICY_GAP", "affected_rows": eligible_zero + eligible_sentinel, "defect": f"No trait-specific zero/sentinel policy; selected-trait review contains {selected_zero_rows} zeros and {selected_sentinel_rows} numeric sentinels", "required_rebuild_action": "Freeze per-trait valid range and missing-code registry; never infer from value alone."},
        {"defect_id": "D2-011", "status": "NO_AFFECTED_ROWS", "affected_rows": 0, "defect": "Ignored alternate GID columns do not recover any numeric unresolved row in this input", "required_rebuild_action": "Still inventory all identifier columns explicitly."},
        {"defect_id": "D2-012", "status": "NO_WIDE_PHENOTYPE_OMISSION_CONFIRMED", "affected_rows": 0, "defect": "All ignored numeric columns are Trait_No, Gen_no, Entry, GID, Plot, or unnamed auxiliary metadata", "required_rebuild_action": "Declare long/wide schema per source and fail on unknown value columns."},
        {"defect_id": "D2-013", "status": "CONFIRMED", "affected_rows": 2938384, "defect": "Canonical summaries and raw Stage-1 records are parallel branches without a persisted contribution bridge", "required_rebuild_action": "Persist canonical-to-raw and raw-to-Stage-1 bridge tables."},
        {"defect_id": "D2-014", "status": "NO_OUTLIER_REMOVAL_FOUND", "affected_rows": 0, "defect": "No outlier filter occurs in the original Stage-1 code", "required_rebuild_action": "Any future outlier policy requires immutable exclusion rows and training-only thresholds."},
        {"defect_id": "D2-015", "status": "CONFIRMED", "affected_rows": 7836162, "defect": "Repository-wide source discovery and fixed output paths are not reproducible/fail-safe", "required_rebuild_action": "Use the hash-bound 284-file allowlist and fail-if-exists outputs."},
        {"defect_id": "D2-016", "status": "NO_LITERAL_INNER_JOIN_LOSS", "affected_rows": 0, "defect": "Stage-1 identity and trait merges are left joins", "required_rebuild_action": "Retain cardinality assertions because post-join filters cause equivalent loss."},
        {"defect_id": "D2-017", "status": "CONFIRMED", "affected_rows": 14162, "defect": "Legacy genotype-order membership removes fully pedigree-and-marker-supported observations", "required_rebuild_action": "Bind the intended genotype registry/order before phenotype filtering."},
        {"defect_id": "D2-018", "status": "CONFIRMED_FILTER_NOT_JOIN", "affected_rows": 59, "defect": "Weight attrition is a direct filter, not a weight join", "required_rebuild_action": "Make the filter an explicit disposition and retain the phenotype row."},
        {"defect_id": "D2-019", "status": "CONFIRMED_PARALLEL_BRANCH_RISK", "affected_rows": 57310, "defect": "Summary duplicates are mean-collapsed before canonical construction and that table supplies the Stage-1 trait/unit registry", "required_rebuild_action": "Separate trait registry construction from phenotype aggregation and ledger every duplicate."},
    ])
    write_tsv(result_dir / "confirmed_pipeline_defects_final.tsv", defects)

    review = pd.DataFrame([
        {"priority": "P0", "review_class": "UNRESOLVED_GENOTYPE_IDENTITY", "items": int(priority["unresolved_candidate_groups"].sum()), "affected_rows": int(priority["unresolved_numeric_rows"].sum()), "detail_artifact": "unresolved_genotype_priority.tsv", "decision_needed": "Authoritative GID/SID/CID/alias/pedigree adjudication; no automatic acceptance."},
        {"priority": "P0", "review_class": "CONFLICTING_RAW_PLOT_DUPLICATES", "items": conflicting_duplicate_groups, "affected_rows": duplicate_excess, "detail_artifact": "raw_plot_duplicate_groups.tsv", "decision_needed": "Define whether rows are repeated measures, transcription conflicts, or duplicates."},
        {"priority": "P0", "review_class": "TRAIT_UNIT_AMBIGUITY", "items": int(retained.loc[retained["trait_mapping_ambiguous"], "trait_name_canonical"].nunique()), "affected_rows": eligible_trait_ambiguity, "detail_artifact": "stage1_input_trait_unit_diagnostics.tsv", "decision_needed": "Approve canonical unit rules; date-like 5-Jan/10-Jan labels indicate spreadsheet coercion."},
        {"priority": "P0", "review_class": "RAW_UNIT_OVERRIDE", "items": int(retained.loc[retained["unit_conflict"], ["trait_name_canonical", "raw_unit", "unit"]].drop_duplicates().shape[0]), "affected_rows": eligible_unit_conflict, "detail_artifact": "stage1_input_trait_unit_diagnostics.tsv", "decision_needed": "Approve conversion, equivalence, or exclusion rule per raw/canonical unit pair."},
        {"priority": "P0", "review_class": "SELECTED_TRAIT_ZERO_OR_SENTINEL", "items": len(selected_zero) + len(selected_sentinel), "affected_rows": selected_zero_rows + selected_sentinel_rows, "detail_artifact": "selected_trait_zero_review.tsv; selected_trait_numeric_sentinel_review.tsv", "decision_needed": "Confirm biological validity versus missing codes per trait/source."},
        {"priority": "P1", "review_class": "ENVIRONMENT_ALIAS_COLLISION", "items": 4, "affected_rows": 1271, "detail_artifact": "environment_alias_collision_review.tsv", "decision_needed": "Domain confirmation of the four dominant-trial collision resolutions."},
        {"priority": "P1", "review_class": "MANIFEST_CONFLICTING_KEY_LATENT", "items": resolver_conflicts, "affected_rows": 0, "detail_artifact": "genotype_resolver_key_audit.tsv", "decision_needed": "Resolve before future data can match these keys."},
        {"priority": "P1", "review_class": "STAGE1_WITHOUT_CANONICAL_NATURAL_KEY", "items": len(missing_canonical), "affected_rows": int(missing_canonical["selected_trait"].sum()), "detail_artifact": "stage1_without_canonical_natural_key.tsv", "decision_needed": "Explain selected-trait key mismatch and nonselected raw-only rows."},
    ])
    write_tsv(result_dir / "unresolved_human_review_final.tsv", review)

    print(defects.to_string(index=False))
    print(review.to_string(index=False))


if __name__ == "__main__":
    main()
