"""Audit local Germplasm DOI files and GLIS-derived GIDs against Stage-1 lineage."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SELECTED_TRAITS = {
    "1000_GRAIN_WEIGHT", "ABOVE_GROUND_BIOMASS", "DAYS_TO_HEADING",
    "DAYS_TO_MATURITY", "GRAIN_YIELD", "PLANT_HEIGHT", "TEST_WEIGHT",
}


def clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def norm(series: pd.Series) -> pd.Series:
    return clean(series).str.upper().str.replace(r"\s+", " ", regex=True)


def cycle_year(series: pd.Series) -> pd.Series:
    raw = clean(series)
    extracted = raw.str.extract(r"(\d{4})", expand=False)
    return extracted.fillna(raw)


def clean_id(series: pd.Series) -> pd.Series:
    return clean(series).str.replace(r"\.0$", "", regex=True)


def valid_doi(series: pd.Series) -> pd.Series:
    return clean(series).str.match(r"^10\.\d{4,9}/\S+$", case=False, na=False)


def normalized_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_"): column
        for column in frame.columns
    }


def pick(frame: pd.DataFrame, columns: dict[str, str], *names: str) -> pd.Series:
    for name in names:
        if name in columns:
            return clean(frame[columns[name]])
    return pd.Series("", index=frame.index, dtype="object")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-root", type=Path, required=True)
    parser.add_argument("--phase1-inventory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-ledger", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    inventory = pd.read_csv(args.phase1_inventory, sep="\t", dtype=str).fillna("")
    doi_inventory = inventory[
        inventory["relative_path"].str.contains(r"Germplasm_DOIs", case=False, regex=True)
        & inventory["suffix"].str.lower().isin({".csv", ".tab"})
    ].copy()
    file_rows: list[dict[str, object]] = []
    records: list[pd.DataFrame] = []
    for item in doi_inventory.sort_values("relative_path").itertuples(index=False):
        path = args.trial_root / item.relative_path
        sep = "\t" if path.suffix.lower() == ".tab" else ","
        status = "PARSED"
        error = ""
        try:
            frame = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding_errors="replace")
        except Exception as exc:  # retain the file in the denominator
            frame = pd.DataFrame()
            status = "PARSE_FAILED"
            error = f"{type(exc).__name__}: {exc}"
        columns = normalized_columns(frame)
        cid = pick(frame, columns, "cid")
        sid = pick(frame, columns, "sid")
        doi = pick(frame, columns, "doi", "germplasm_doi")
        url = pick(frame, columns, "doi_glis_url", "glis_url")
        file_rows.append({
            "relative_path": item.relative_path,
            "sha256": item.sha256,
            "bytes": int(item.bytes),
            "delimiter": "TAB" if sep == "\t" else "COMMA",
            "parse_status": status,
            "error": error,
            "rows": len(frame),
            "columns": ";".join(map(str, frame.columns)),
            "rows_with_CID": int(cid.ne("").sum()),
            "rows_with_SID": int(sid.ne("").sum()),
            "rows_with_DOI": int(doi.ne("").sum()),
            "rows_with_valid_DOI": int(valid_doi(doi).sum()),
            "rows_with_GLIS_url": int(url.ne("").sum()),
        })
        if frame.empty:
            continue
        record = pd.DataFrame({
            "doi_source_file": item.relative_path,
            "doi_source_file_sha256": item.sha256,
            "doi_source_physical_row": np.arange(2, len(frame) + 2, dtype=np.int64),
            "trial_file_token": re.sub(r"_Germplasm_DOIs.*$", "", path.stem, flags=re.I),
            "CID": clean_id(cid),
            "SID": clean_id(sid),
            "cross_name": pick(frame, columns, "cross_name", "crossname"),
            "selection_history": pick(frame, columns, "selection_history", "selectionhistory"),
            "DOI": doi,
            "DOI_valid_syntax": valid_doi(doi),
            "DOI_GLIS_url": url,
        })
        records.append(record)

    file_audit = pd.DataFrame(file_rows)
    file_audit.to_csv(result_dir / "doi_file_inventory.tsv", sep="\t", index=False)
    doi_records = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    pq.write_table(pa.Table.from_pandas(doi_records, preserve_index=False), result_dir / "doi_record_ledger.parquet", compression="zstd")

    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str, low_memory=False).fillna("")
    manifest["trial_key"] = norm(manifest["trial_name"])
    manifest["cycle_norm"] = cycle_year(manifest["cycle"])
    manifest["occ_norm"] = clean(manifest["occ"])
    manifest["CID_norm"] = clean_id(manifest["CID"])
    manifest["SID_norm"] = clean_id(manifest["SID"])
    manifest["resolver_key"] = (
        manifest["trial_key"] + "\x1f" + manifest["cycle_norm"] + "\x1f"
        + manifest["occ_norm"] + "\x1f" + manifest["CID_norm"] + "\x1f" + manifest["SID_norm"]
    )
    manifest["doi_file_norm"] = clean(manifest["doi_file"]).str.replace("\\", "/", regex=False).str.lower()
    manifest["DOI_valid_syntax"] = valid_doi(manifest["DOI"])
    chosen = manifest.drop_duplicates("resolver_key", keep="first").copy()
    resolver_meta = chosen.set_index("resolver_key")[[
        "gid_source", "gid_resolution_status", "DOI", "doi_gid", "glis_gid",
        "resolved_gid", "fieldbook_glis_gid_conflict", "doi_file",
    ]]

    # Connect each local DOI row to its manifest and GLIS-resolution outcomes.
    manifest_doi = manifest.copy()
    manifest_doi["doi_join_key"] = (
        manifest_doi["doi_file_norm"] + "\x1f" + manifest_doi["CID_norm"] + "\x1f" + manifest_doi["SID_norm"]
    )
    manifest_link = (
        manifest_doi.groupby("doi_join_key", dropna=False, sort=False)
        .agg(
            manifest_rows=("resolver_key", "size"),
            manifest_trials=("trial_name", "nunique"),
            manifest_resolved_gids=("resolved_gid", lambda x: ";".join(sorted(set(clean(x)) - {""}))),
            manifest_glis_gids=("glis_gid", lambda x: ";".join(sorted(set(clean(x)) - {""}))),
            manifest_gid_sources=("gid_source", lambda x: ";".join(sorted(set(clean(x)) - {""}))),
            manifest_resolution_statuses=("gid_resolution_status", lambda x: ";".join(sorted(set(clean(x)) - {""}))),
            manifest_fieldbook_glis_conflicts=("fieldbook_glis_gid_conflict", lambda x: int(norm(x).isin({"TRUE", "1", "YES"}).sum())),
        )
        .reset_index()
    )
    doi_records["doi_file_norm"] = clean(doi_records["doi_source_file"]).str.lower()
    doi_records["doi_join_key"] = doi_records["doi_file_norm"] + "\x1f" + doi_records["CID"] + "\x1f" + doi_records["SID"]
    linkage = doi_records.merge(manifest_link, on="doi_join_key", how="left", validate="m:1")
    linkage["manifest_rows"] = pd.to_numeric(linkage["manifest_rows"], errors="coerce").fillna(0).astype("int64")
    linkage["doi_manifest_link_status"] = np.select(
        [
            linkage["manifest_rows"].eq(0),
            clean(linkage["manifest_glis_gids"]).ne(""),
            clean(linkage["manifest_resolved_gids"]).ne(""),
        ],
        ["NO_MANIFEST_MATCH", "MATCH_WITH_GLIS_GID", "MATCH_WITH_NON_GLIS_RESOLVED_GID"],
        default="MATCH_WITHOUT_RESOLVED_GID",
    )
    pq.write_table(pa.Table.from_pandas(linkage, preserve_index=False), result_dir / "doi_to_manifest_linkage.parquet", compression="zstd")

    # Quantify DOI/GLIS-derived identity in the actual legacy Stage-1 contributors.
    raw = pq.read_table(args.raw_ledger, columns=[
        "resolver_key", "final_raw_disposition", "genotype_id_class", "resolved_gid",
        "trait_name_canonical", "expected_stage1_observation_id", "env_kernel_id",
        "numeric_parse_pass", "trial_key", "cycle", "occ", "CID_normalized", "SID_normalized",
    ]).to_pandas()
    retained = raw[
        raw["final_raw_disposition"].eq("RETAINED_CONTRIBUTES_TO_STAGE1")
        & raw["genotype_id_class"].eq("MANIFEST_RESOLVED")
    ].copy()
    retained = retained.join(resolver_meta, on="resolver_key", rsuffix="_manifest")
    retained["selected_trait"] = norm(retained["trait_name_canonical"]).isin(SELECTED_TRAITS)
    impact = (
        retained.groupby(["gid_source", "selected_trait", "trait_name_canonical"], dropna=False, sort=True)
        .agg(
            contributing_raw_rows=("resolver_key", "size"),
            stage1_observations=("expected_stage1_observation_id", "nunique"),
            resolved_gids=("resolved_gid", "nunique"),
            environments=("env_kernel_id", "nunique"),
            rows_with_DOI=("DOI", lambda x: int(clean(x).ne("").sum())),
            rows_with_GLIS_GID=("glis_gid", lambda x: int(clean(x).ne("").sum())),
        )
        .reset_index()
    )
    impact.to_csv(result_dir / "doi_glis_stage1_impact.tsv", sep="\t", index=False)

    # Potential DOI/GLIS candidates for legacy-unresolved numeric rows. These are
    # evidence queues only because cycle/occurrence or trial context is relaxed.
    unresolved = raw[
        raw["numeric_parse_pass"].astype(str).str.upper().isin({"TRUE", "1"})
        & raw["genotype_id_class"].eq("UNRESOLVED")
    ].copy()
    unresolved["CID_norm"] = clean_id(unresolved["CID_normalized"])
    unresolved["SID_norm"] = clean_id(unresolved["SID_normalized"])
    unresolved["cycle_norm"] = cycle_year(unresolved["cycle"])
    unresolved["occ_norm"] = clean(unresolved["occ"])
    unresolved_grouped = (
        unresolved.groupby(["trial_key", "cycle_norm", "occ_norm", "CID_norm", "SID_norm"], dropna=False, sort=False)
        .size().rename("unresolved_numeric_rows").reset_index()
    )
    unresolved_grouped = unresolved_grouped[
        unresolved_grouped["CID_norm"].ne("") | unresolved_grouped["SID_norm"].ne("")
    ]
    doi_manifest_candidates = manifest[
        manifest["DOI_valid_syntax"] & clean(manifest["resolved_gid"]).ne("")
    ].copy()
    candidate_registry_no_occ = (
        doi_manifest_candidates.groupby(["trial_key", "cycle_norm", "CID_norm", "SID_norm"], dropna=False, sort=False)
        .agg(
            candidate_manifest_rows=("resolver_key", "size"),
            candidate_DOIs=("DOI", lambda x: ";".join(sorted(set(clean(x)) - {""}))),
            candidate_resolved_gids=("resolved_gid", lambda x: ";".join(sorted(set(clean(x)) - {""}))),
            candidate_glis_gids=("glis_gid", lambda x: ";".join(sorted(set(clean(x)) - {""}))),
            candidate_gid_sources=("gid_source", lambda x: ";".join(sorted(set(clean(x)) - {""}))),
        )
        .reset_index()
    )
    candidates_no_occ = unresolved_grouped.merge(
        candidate_registry_no_occ,
        on=["trial_key", "cycle_norm", "CID_norm", "SID_norm"],
        how="inner",
        validate="m:1",
    )
    candidates_no_occ["relaxed_components"] = "OCCURRENCE_ONLY"

    matched_keys = set(
        candidates_no_occ[["trial_key", "cycle_norm", "occ_norm", "CID_norm", "SID_norm"]]
        .astype(str).agg("\x1f".join, axis=1)
    )
    unresolved_grouped["_full_key"] = (
        unresolved_grouped[["trial_key", "cycle_norm", "occ_norm", "CID_norm", "SID_norm"]]
        .astype(str).agg("\x1f".join, axis=1)
    )
    remaining = unresolved_grouped[~unresolved_grouped["_full_key"].isin(matched_keys)].drop(columns="_full_key")
    candidate_registry_no_cycle_occ = (
        doi_manifest_candidates.groupby(["trial_key", "CID_norm", "SID_norm"], dropna=False, sort=False)
        .agg(
            candidate_manifest_rows=("resolver_key", "size"),
            candidate_DOIs=("DOI", lambda x: ";".join(sorted(set(clean(x)) - {""}))),
            candidate_resolved_gids=("resolved_gid", lambda x: ";".join(sorted(set(clean(x)) - {""}))),
            candidate_glis_gids=("glis_gid", lambda x: ";".join(sorted(set(clean(x)) - {""}))),
            candidate_gid_sources=("gid_source", lambda x: ";".join(sorted(set(clean(x)) - {""}))),
        )
        .reset_index()
    )
    candidates_no_cycle_occ = remaining.merge(
        candidate_registry_no_cycle_occ,
        on=["trial_key", "CID_norm", "SID_norm"],
        how="inner",
        validate="m:1",
    )
    candidates_no_cycle_occ["relaxed_components"] = "CYCLE_AND_OCCURRENCE"
    candidates = pd.concat([candidates_no_occ, candidates_no_cycle_occ], ignore_index=True)
    candidates["candidate_unique_gid_count"] = candidates["candidate_resolved_gids"].map(
        lambda value: len([part for part in str(value).split(";") if part])
    )
    candidates["candidate_status"] = np.where(
        candidates["candidate_unique_gid_count"].eq(1),
        "UNIQUE_GID_AFTER_RELAXING_CYCLE_OCCURRENCE_REQUIRES_REVIEW",
        "AMBIGUOUS_GID_AFTER_RELAXING_CYCLE_OCCURRENCE",
    )
    candidates.sort_values(["unresolved_numeric_rows", "trial_key"], ascending=[False, True]).to_csv(
        result_dir / "doi_glis_unresolved_candidate_ledger.tsv", sep="\t", index=False
    )

    doi_gid_conflicts = (
        manifest[manifest["DOI_valid_syntax"]]
        .groupby("DOI", dropna=False)
        .agg(
            manifest_rows=("resolver_key", "size"),
            resolved_gid_count=("resolved_gid", lambda x: len(set(clean(x)) - {""})),
            resolved_gids=("resolved_gid", lambda x: ";".join(sorted(set(clean(x)) - {""}))),
            glis_gid_count=("glis_gid", lambda x: len(set(clean(x)) - {""})),
            glis_gids=("glis_gid", lambda x: ";".join(sorted(set(clean(x)) - {""}))),
        )
        .reset_index()
    )
    doi_gid_conflicts = doi_gid_conflicts[
        doi_gid_conflicts["resolved_gid_count"].gt(1) | doi_gid_conflicts["glis_gid_count"].gt(1)
    ]
    doi_gid_conflicts.to_csv(result_dir / "doi_to_gid_conflicts.tsv", sep="\t", index=False)

    gid_source_counts = manifest.groupby(["gid_source", "gid_resolution_status"], dropna=False).size().reset_index(name="manifest_rows")
    gid_source_counts.to_csv(result_dir / "manifest_gid_source_summary.tsv", sep="\t", index=False)
    glis_retained = retained[retained["gid_source"].eq("glis_doi_resolver")]
    summary = {
        "status": "PASS_LOCAL_DOI_GLIS_AUDIT_COMPLETE",
        "external_GLIS_queries_performed": 0,
        "doi_files": len(file_audit),
        "doi_file_parse_failures": int(file_audit["parse_status"].ne("PARSED").sum()),
        "doi_records": len(doi_records),
        "doi_records_with_DOI": int(clean(doi_records["DOI"]).ne("").sum()),
        "doi_records_with_valid_DOI": int(valid_doi(doi_records["DOI"]).sum()),
        "doi_records_with_non_DOI_text": int((clean(doi_records["DOI"]).ne("") & ~valid_doi(doi_records["DOI"])).sum()),
        "doi_records_without_manifest_match": int(linkage["doi_manifest_link_status"].eq("NO_MANIFEST_MATCH").sum()),
        "doi_source_files_represented_in_manifest": int(manifest.loc[clean(manifest["doi_file"]).ne(""), "doi_file_norm"].nunique()),
        "manifest_rows": len(manifest),
        "manifest_rows_with_DOI": int(clean(manifest["DOI"]).ne("").sum()),
        "manifest_rows_with_valid_DOI": int(manifest["DOI_valid_syntax"].sum()),
        "manifest_rows_with_glis_gid": int(clean(manifest["glis_gid"]).ne("").sum()),
        "manifest_rows_resolved_by_glis_doi": int(manifest["gid_source"].eq("glis_doi_resolver").sum()),
        "manifest_fieldbook_glis_conflicts": int(norm(manifest["fieldbook_glis_gid_conflict"]).isin({"TRUE", "1", "YES"}).sum()),
        "stage1_contributing_raw_rows_resolved_by_glis_doi": len(glis_retained),
        "stage1_observations_resolved_by_glis_doi": int(glis_retained["expected_stage1_observation_id"].nunique()),
        "selected_stage1_observations_resolved_by_glis_doi": int(glis_retained.loc[glis_retained["selected_trait"], "expected_stage1_observation_id"].nunique()),
        "unresolved_numeric_rows": len(unresolved),
        "unresolved_numeric_rows_with_same_trial_CID_SID_DOI_candidate_after_relaxing_cycle_occ": int(candidates["unresolved_numeric_rows"].sum()),
        "candidate_key_groups_after_relaxing_cycle_occ": len(candidates),
        "candidate_key_groups_with_unique_gid": int(candidates["candidate_unique_gid_count"].eq(1).sum()),
        "candidate_rows_after_relaxing_occurrence_only": int(candidates.loc[candidates["relaxed_components"].eq("OCCURRENCE_ONLY"), "unresolved_numeric_rows"].sum()),
        "candidate_rows_after_relaxing_cycle_and_occurrence": int(candidates.loc[candidates["relaxed_components"].eq("CYCLE_AND_OCCURRENCE"), "unresolved_numeric_rows"].sum()),
        "doi_to_multiple_gid_conflicts": len(doi_gid_conflicts),
    }
    (result_dir / "doi_glis_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
