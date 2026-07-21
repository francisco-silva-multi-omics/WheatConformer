from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import io
import json
import re
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import pandas as pd

from server_genotype_recovery.archive_utils import iter_7z_members
from server_genotype_recovery.dataverse_crop_scope import (
    AMBIGUOUS_REVIEW,
    NON_WHEAT_EXCLUDED,
    WHEAT_CONFIRMED,
    classify_crop_scope,
)
from server_genotype_recovery.fetch_brapi_pedigree_markers import (
    build_query_terms,
    clean,
    normalized_identifier,
    read_table,
    sha256_file,
    write_json_atomic,
)


EVIDENCE_COLUMNS = [
    "query_id",
    "query_kind",
    "query_text",
    "normalized_query",
    "resolver_term_gid_count",
    "evidence_class",
    "individual_identity_level",
    "marker_bridge_class",
    "crop_scope",
    "crop_scope_evidence",
    "dataset_name",
    "dataset_persistent_id",
    "datafile_id",
    "candidate_role",
    "source_subtype",
    "filename",
    "local_path",
    "source_part",
    "source_row",
    "source_column",
    "cell_value",
    "row_context_json",
]

LARGE_STRUCTURED_BYTES = 512 * 1024**2
LARGE_MATRIX_AXIS_ROWS = 32


def _first_nonempty(values: pd.Series) -> str:
    for value in values:
        value = clean(value)
        if value:
            return value
    return ""


def annotate_download_crop_scope(
    downloads: pd.DataFrame,
    search_results: pd.DataFrame,
) -> pd.DataFrame:
    local = downloads.copy()
    if "dataset_name" in local.columns:
        local["download_dataset_name"] = local["dataset_name"].fillna("")
        local = local.drop(columns="dataset_name")
    else:
        local["download_dataset_name"] = ""
    if search_results.empty:
        dataset_names = pd.DataFrame(
            columns=["dataset_persistent_id", "dataset_name"]
        )
    else:
        search = search_results.copy()
        for column in ("dataset_persistent_id", "global_id", "dataset_name"):
            if column not in search.columns:
                search[column] = ""
        search["dataset_persistent_id"] = search["dataset_persistent_id"].fillna("")
        missing = search["dataset_persistent_id"].map(clean).eq("")
        search.loc[missing, "dataset_persistent_id"] = search.loc[missing, "global_id"]
        dataset_names = (
            search.groupby("dataset_persistent_id", dropna=False)["dataset_name"]
            .agg(_first_nonempty)
            .reset_index()
        )
    local = local.merge(dataset_names, on="dataset_persistent_id", how="left")
    search_name = local["dataset_name"].fillna("")
    existing_name = local["download_dataset_name"].fillna("")
    local["dataset_name"] = search_name.where(
        search_name.map(clean).ne(""), existing_name
    )
    local = local.drop(columns="download_dataset_name")
    crop = local.apply(
        lambda row: classify_crop_scope(
            row.get("dataset_name"), row.get("filename"), row.get("description")
        ),
        axis=1,
        result_type="expand",
    )
    local[["crop_scope", "crop_scope_evidence"]] = crop
    return local


@lru_cache(maxsize=250_000)
def _normalized_clean_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def evidence_class(query_kind: object, resolver_term_gid_count: int) -> tuple[str, str]:
    kind = clean(query_kind)
    unique = resolver_term_gid_count == 1
    if kind == "sample_id":
        return "direct_gid_exact", "individual_direct"
    if kind == "selection_history":
        if unique:
            return "selection_history_exact_unique", "individual_candidate"
        return "selection_history_exact_shared", "family_or_alias_ambiguous"
    if kind == "bcid":
        return "bcid_exact_unique" if unique else "bcid_exact_shared", "family_or_batch"
    if kind == "cross_name":
        return "cross_exact_unique" if unique else "cross_exact_shared", "family_only"
    if kind in {"parent1", "parent2"}:
        return "parent_exact", "ancestor_only"
    return "other_exact", "unclassified"


def source_subtype(filename: object, description: object) -> str:
    text = f"{clean(filename)} {clean(description)}".lower()
    mapping_terms = ("sampleidvsgid", "sample_id", "germplasm", "pedigree", "passport", "doi")
    matrix_terms = ("pop_axiom", "transposed", "genotypic_report", "snp call", ".vcf", "hapmap", "dosage")
    if any(term in text for term in matrix_terms):
        return "marker_matrix_candidate"
    if any(term in text for term in mapping_terms):
        return "germplasm_or_sample_mapping"
    return "other_candidate"


def marker_bridge_class(
    query_kind: object,
    identity_level: str,
    candidate_role: object,
    subtype: str,
) -> str:
    if "marker" not in clean(candidate_role):
        return "no_marker_source"
    kind = clean(query_kind)
    if subtype == "marker_matrix_candidate" and kind == "sample_id":
        return "candidate_direct_gid_in_marker_matrix"
    if subtype == "germplasm_or_sample_mapping" and kind == "sample_id":
        return "candidate_direct_gid_to_sample_mapping"
    if subtype == "marker_matrix_candidate" and identity_level == "individual_candidate":
        return "candidate_unique_line_in_marker_matrix"
    if subtype == "germplasm_or_sample_mapping" and identity_level == "individual_candidate":
        return "candidate_unique_line_to_sample_mapping"
    return "family_context_only"


def _read_delimited(
    source: object,
    suffix: str,
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    compression = "gzip" if suffix.endswith(".gz") else "infer"
    try:
        return pd.read_csv(
            source,
            sep=None,
            engine="python",
            header=None,
            dtype=str,
            compression=compression,
            on_bad_lines="skip",
            nrows=nrows,
        )
    except (UnicodeDecodeError, pd.errors.ParserError):
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(
            source,
            sep="\t",
            header=None,
            dtype=str,
            compression=compression,
            encoding="latin-1",
            on_bad_lines="skip",
            nrows=nrows,
        )


def _excel_engine(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".gz"):
        lower = lower[:-3]
    if lower.endswith(".xls"):
        return "xlrd"
    if lower.endswith((".xlsx", ".xlsm")):
        return "openpyxl"
    raise ValueError(f"unsupported Excel format: {filename}")


def _read_excel_sheets(path: Path) -> dict[str, pd.DataFrame]:
    lower = path.name.lower()
    engine = _excel_engine(lower)
    payload: bytes | None = None
    if lower.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            payload = handle.read()
            source: object = io.BytesIO(payload)
    else:
        source = path
        if engine == "xlrd":
            payload = path.read_bytes()
    if engine == "xlrd" and payload is not None:
        ole_header = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        sample = payload[:8192]
        text_like = b"\x00" not in sample and any(
            delimiter in sample for delimiter in (b"\t", b",", b";")
        )
        if not payload.startswith(ole_header) and text_like:
            return {
                "text_fallback": _read_delimited(io.BytesIO(payload), ".txt")
            }
    return pd.read_excel(
        source,
        sheet_name=None,
        header=None,
        dtype=str,
        engine=engine,
    )


def structured_parts(path: Path) -> Iterator[tuple[str, pd.DataFrame]]:
    lower = path.name.lower()
    if lower.endswith(
        (".xlsx", ".xlsm", ".xls", ".xlsx.gz", ".xlsm.gz", ".xls.gz")
    ):
        sheets = _read_excel_sheets(path)
        for name, frame in sheets.items():
            yield f"sheet:{name}", frame
        return
    if lower.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_lower = member.filename.lower()
                if not member_lower.endswith((".txt", ".tsv", ".tab", ".csv")):
                    continue
                bounded = member.file_size >= LARGE_STRUCTURED_BYTES
                part_kind = "archive_axis_preview" if bounded else "archive"
                with archive.open(member) as handle:
                    yield f"{part_kind}:{member.filename}", _read_delimited(
                        handle,
                        member_lower,
                        nrows=LARGE_MATRIX_AXIS_ROWS if bounded else None,
                    )
        return
    if lower.endswith(".7z"):
        for member_name, member_path in iter_7z_members(path):
            member_lower = member_name.lower()
            if not member_lower.endswith(
                (
                    ".txt",
                    ".tsv",
                    ".tab",
                    ".csv",
                    ".txt.gz",
                    ".tsv.gz",
                    ".csv.gz",
                    ".xls",
                    ".xlsx",
                    ".xlsm",
                )
            ):
                continue
            for source_part, frame in structured_parts(member_path):
                yield f"archive:{member_name}:{source_part}", frame
        return
    if lower.endswith((".txt", ".tsv", ".tab", ".csv", ".txt.gz", ".tsv.gz", ".csv.gz")):
        bounded = path.stat().st_size >= LARGE_STRUCTURED_BYTES
        yield "file_axis_preview" if bounded else "file", _read_delimited(
            path,
            lower,
            nrows=LARGE_MATRIX_AXIS_ROWS if bounded else None,
        )
        return
    raise ValueError(f"unsupported structured format: {path.name}")


def term_index_from_resolver(
    resolver: pd.DataFrame,
) -> tuple[dict[str, list[dict[str, object]]], int]:
    _, terms = build_query_terms(resolver, len(resolver), 0)
    index: dict[str, list[dict[str, object]]] = {}
    for term in terms:
        normalized = normalized_identifier(term["query_text"])
        if normalized:
            index.setdefault(normalized, []).append(term)
    for normalized, rows in index.items():
        gid_count = len({clean(row["query_id"]) for row in rows if clean(row["query_id"])})
        for row in rows:
            row["resolver_term_gid_count"] = gid_count
            row["normalized_query"] = normalized
    return index, len(terms)


def scan_frame(
    frame: pd.DataFrame,
    term_index: dict[str, list[dict[str, object]]],
    source: dict[str, object],
    source_part: str,
) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    local = frame.fillna("")
    subtype = source_subtype(source.get("filename"), source.get("description"))
    for row_position, values in enumerate(local.itertuples(index=False, name=None)):
        text_values = [clean(value) for value in values]
        normalized_cells: dict[str, list[int]] = {}
        for column_position, value in enumerate(text_values):
            normalized = _normalized_clean_text(value)
            if normalized in term_index:
                normalized_cells.setdefault(normalized, []).append(column_position)
        if not normalized_cells:
            continue
        context_positions = set(range(min(3, len(text_values))))
        for positions in normalized_cells.values():
            for position in positions:
                context_positions.update(
                    range(max(0, position - 2), min(len(text_values), position + 3))
                )
        context = {
            str(position): text_values[position][:500]
            for position in sorted(context_positions)
            if text_values[position]
        }
        for normalized, positions in normalized_cells.items():
            for term in term_index[normalized]:
                count = int(term["resolver_term_gid_count"])
                evidence, identity = evidence_class(term["query_kind"], count)
                bridge = marker_bridge_class(
                    term["query_kind"], identity, source.get("candidate_role"), subtype
                )
                for position in positions:
                    hits.append(
                        {
                            "query_id": term["query_id"],
                            "query_kind": term["query_kind"],
                            "query_text": term["query_text"],
                            "normalized_query": normalized,
                            "resolver_term_gid_count": count,
                            "evidence_class": evidence,
                            "individual_identity_level": identity,
                            "marker_bridge_class": bridge,
                            "crop_scope": source.get("crop_scope", ""),
                            "crop_scope_evidence": source.get("crop_scope_evidence", ""),
                            "dataset_name": source.get("dataset_name", ""),
                            "dataset_persistent_id": source.get("dataset_persistent_id", ""),
                            "datafile_id": source.get("datafile_id", ""),
                            "candidate_role": source.get("candidate_role", ""),
                            "source_subtype": subtype,
                            "filename": source.get("filename", ""),
                            "local_path": source.get("local_path", ""),
                            "source_part": source_part,
                            "source_row": row_position,
                            "source_column": position,
                            "cell_value": text_values[position][:2000],
                            "row_context_json": json.dumps(context, sort_keys=True),
                        }
                    )
    return hits


def requires_full_structured_scan(filename: object) -> bool:
    lower = clean(filename).lower()
    return lower.endswith(
        (".xls", ".xls.gz", ".xlsx.gz", ".xlsm.gz", ".7z")
    )


def content_term_index(
    content_matches: pd.DataFrame,
    full_term_index: dict[str, list[dict[str, object]]],
) -> tuple[dict[str, set[str]], set[str]]:
    terms_by_path: dict[str, set[str]] = {}
    scan_error_paths: set[str] = set()
    if content_matches.empty or "path" not in content_matches.columns:
        return terms_by_path, scan_error_paths
    for row in content_matches.to_dict("records"):
        path = clean(row.get("path"))
        if not path:
            continue
        if clean(row.get("query_kind")) == "scan_error":
            scan_error_paths.add(path)
            continue
        normalized = normalized_identifier(row.get("query_text"))
        if normalized in full_term_index:
            terms_by_path.setdefault(path, set()).add(normalized)
    return terms_by_path, scan_error_paths


def summarize_gid_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return pd.DataFrame(
            columns=[
                "query_id",
                "structured_match_rows",
                "direct_gid_exact",
                "unique_selection_history_exact",
                "family_only_evidence",
                "marker_bridge_candidate",
                "strongest_evidence",
                "direct_marker_assignment_ready",
                "source_file_count",
                "source_files",
            ]
        )
    rows: list[dict[str, object]] = []
    for query_id, group in evidence.groupby("query_id", sort=True):
        classes = set(group["evidence_class"])
        bridges = set(group["marker_bridge_class"])
        direct = "direct_gid_exact" in classes
        selection = "selection_history_exact_unique" in classes
        family = any(
            value.startswith(("bcid_", "cross_", "parent_"))
            or value == "selection_history_exact_shared"
            for value in classes
        )
        marker_candidate = any(value.startswith("candidate_") for value in bridges)
        strongest = (
            "direct_gid_exact"
            if direct
            else "selection_history_exact_unique"
            if selection
            else "family_only"
            if family
            else "other"
        )
        files = sorted(set(group["filename"].dropna().astype(str)))
        rows.append(
            {
                "query_id": query_id,
                "structured_match_rows": len(group),
                "direct_gid_exact": direct,
                "unique_selection_history_exact": selection,
                "family_only_evidence": family,
                "marker_bridge_candidate": marker_candidate,
                "strongest_evidence": strongest,
                # This audit identifies candidates only. Sample-axis and call
                # concordance certification are separate required steps.
                "direct_marker_assignment_ready": False,
                "source_file_count": len(files),
                "source_files": ";".join(files),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create row/cell-certified evidence from downloaded CIMMYT Dataverse files."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--recovery-dir",
        type=Path,
        default=Path("genotype_panels/cimmyt_dataverse_recovery_v1/batch_00000_00010_ranked"),
    )
    parser.add_argument(
        "--resolver-query",
        type=Path,
        default=Path("genotype_panels/germplasm_resolver/germplasm_cross_query.tsv"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    recovery_dir = args.recovery_dir if args.recovery_dir.is_absolute() else root / args.recovery_dir
    resolver_path = args.resolver_query if args.resolver_query.is_absolute() else root / args.resolver_query
    out_dir = args.out_dir or recovery_dir / "structured_evidence"
    out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_status_path = out_dir / "dataverse_structured_evidence_run_status.json"
    write_json_atomic(
        {
            "status": "INCOMPLETE",
            "detail": "Audit started; prior result files may remain until completion.",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        run_status_path,
    )
    downloads_path = recovery_dir / "dataverse_downloads.tsv"
    content_matches_path = recovery_dir / "dataverse_content_matches.tsv"
    search_results_path = recovery_dir / "dataverse_search_results.tsv"
    local_reuse_path = (
        recovery_dir / "tier2_inventory/dataverse_tier2_local_reuse_manifest.tsv"
    )
    if not downloads_path.is_file():
        raise FileNotFoundError(downloads_path)
    if not resolver_path.is_file():
        raise FileNotFoundError(resolver_path)

    resolver = read_table(resolver_path)
    term_index, term_count = term_index_from_resolver(resolver)
    downloads = read_table(downloads_path)
    downloads = downloads[downloads["download_status"].isin(["DOWNLOADED", "REUSED"])].copy()
    local_reuse_count = 0
    if local_reuse_path.is_file():
        local_reuse = read_table(local_reuse_path)
        required_reuse_columns = {
            "dataset_persistent_id",
            "datafile_id",
            "filename",
            "local_path",
            "download_status",
            "crop_scope",
            "local_reconciliation_status",
        }
        missing_reuse_columns = sorted(
            required_reuse_columns - set(local_reuse.columns)
        )
        if missing_reuse_columns:
            raise ValueError(
                "Local reuse manifest is stale or incomplete; missing columns: "
                f"{missing_reuse_columns}"
            )
        invalid_reuse = local_reuse[
            ~local_reuse["crop_scope"].eq(WHEAT_CONFIRMED)
            | ~local_reuse["local_reconciliation_status"].eq(
                "LOCAL_EXACT_CHECKSUM"
            )
        ]
        if not invalid_reuse.empty:
            raise ValueError(
                "Local reuse manifest contains entries that are not certified "
                "wheat checksum matches"
            )
        local_reuse_count = len(local_reuse)
        downloads = pd.concat([downloads, local_reuse], ignore_index=True)
        downloads = downloads.drop_duplicates(
            ["dataset_persistent_id", "datafile_id"], keep="last"
        )
    search_results = (
        read_table(search_results_path)
        if search_results_path.is_file()
        else pd.DataFrame()
    )
    downloads = annotate_download_crop_scope(downloads, search_results)
    source_crop_audit = downloads[
        [
            "dataset_persistent_id",
            "datafile_id",
            "dataset_name",
            "filename",
            "local_path",
            "crop_scope",
            "crop_scope_evidence",
        ]
    ].copy()
    source_crop_audit.to_csv(
        out_dir / "dataverse_structured_source_crop_scope.tsv", sep="\t", index=False
    )
    content_matches = (
        read_table(content_matches_path)
        if content_matches_path.is_file()
        else pd.DataFrame()
    )
    terms_by_path, scan_error_paths = content_term_index(content_matches, term_index)
    use_content_prefilter = content_matches_path.is_file()
    evidence_rows: list[dict[str, object]] = []
    parse_rows: list[dict[str, object]] = []
    records = downloads.to_dict("records")
    for file_number, record in enumerate(records, start=1):
        path = Path(clean(record.get("local_path")))
        filename = clean(record.get("filename")) or path.name
        started = time.monotonic()
        print(
            f"[{file_number}/{len(records)}] structured evidence: {filename}",
            flush=True,
        )
        crop_scope = clean(record.get("crop_scope"))
        if crop_scope != WHEAT_CONFIRMED:
            status = (
                "EXCLUDED_NON_WHEAT_CROP"
                if crop_scope == NON_WHEAT_EXCLUDED
                else "DEFERRED_AMBIGUOUS_CROP"
            )
            parse_rows.append(
                {
                    "filename": filename,
                    "status": status,
                    "parts": 0,
                    "rows": 0,
                    "detail": clean(record.get("crop_scope_evidence")),
                }
            )
            print(f"  {status}; skipped", flush=True)
            continue
        if not path.is_file():
            parse_rows.append({"filename": record.get("filename", ""), "status": "MISSING", "parts": 0, "rows": 0, "detail": str(path)})
            continue
        path_key = str(path)
        full_scan = (
            requires_full_structured_scan(filename)
            or path_key in scan_error_paths
            or clean(record.get("local_reconciliation_status"))
            == "LOCAL_EXACT_CHECKSUM"
        )
        indexed_terms = terms_by_path.get(path_key, set())
        if use_content_prefilter and not full_scan and not indexed_terms:
            parse_rows.append(
                {
                    "filename": filename,
                    "status": "INDEXED_NO_IDENTIFIER_MATCH",
                    "parts": 0,
                    "rows": 0,
                    "detail": "Skipped by the complete raw-content identifier index",
                }
            )
            print("  indexed no-match; skipped", flush=True)
            continue
        local_term_index = (
            term_index
            if full_scan or not use_content_prefilter
            else {key: term_index[key] for key in indexed_terms}
        )
        parts = 0
        parsed_rows = 0
        bounded_axis_scan = False
        file_evidence_rows: list[dict[str, object]] = []
        try:
            for source_part, frame in structured_parts(path):
                parts += 1
                parsed_rows += len(frame)
                bounded_axis_scan |= "axis_preview" in source_part
                file_evidence_rows.extend(
                    scan_frame(frame, local_term_index, record, source_part)
                )
            if bounded_axis_scan:
                status = "PASS_BOUNDED_AXIS"
                detail = (
                    "Large marker matrix: scanned only the first "
                    f"{LARGE_MATRIX_AXIS_ROWS} identifier/header rows; genotype "
                    "cells were not materialized."
                )
            else:
                status, detail = "PASS", ""
        except Exception as exc:
            status, detail = "SKIPPED_OR_FAILED", f"{type(exc).__name__}: {exc}"
        evidence_rows.extend(file_evidence_rows)
        parse_rows.append({"filename": record.get("filename", ""), "status": status, "parts": parts, "rows": parsed_rows, "detail": detail})
        print(
            f"  {status}; parts={parts}; rows={parsed_rows}; "
            f"matches={len(file_evidence_rows)}; seconds={time.monotonic() - started:.2f}",
            flush=True,
        )

    evidence = pd.DataFrame(evidence_rows, columns=EVIDENCE_COLUMNS)
    evidence_path = out_dir / "dataverse_structured_evidence.tsv.gz"
    evidence.to_csv(evidence_path, sep="\t", index=False, compression="gzip")
    gid_summary = summarize_gid_evidence(evidence)
    gid_summary.to_csv(out_dir / "dataverse_structured_gid_summary.tsv", sep="\t", index=False)
    parse_log = pd.DataFrame(parse_rows)
    parse_log.to_csv(out_dir / "dataverse_structured_parse_log.tsv", sep="\t", index=False)
    if evidence.empty:
        file_summary = pd.DataFrame(columns=["filename", "query_kind", "evidence_class", "matched_gids", "match_rows"])
    else:
        file_summary = (
            evidence.groupby(["filename", "query_kind", "evidence_class"], dropna=False)
            .agg(matched_gids=("query_id", "nunique"), match_rows=("query_id", "size"))
            .reset_index()
        )
    file_summary.to_csv(out_dir / "dataverse_structured_file_summary.tsv", sep="\t", index=False)
    bridge = evidence[evidence["marker_bridge_class"].str.startswith("candidate_", na=False)].copy()
    bridge.to_csv(out_dir / "dataverse_marker_bridge_candidates.tsv", sep="\t", index=False)

    qc = pd.DataFrame(
        [
            {"metric": "resolver_rows", "value": len(resolver)},
            {"metric": "resolver_terms", "value": term_count},
            {"metric": "downloaded_files_considered", "value": len(downloads)},
            {"metric": "verified_local_reuse_files_considered", "value": local_reuse_count},
            {"metric": "wheat_confirmed_downloaded_files", "value": int(downloads["crop_scope"].eq(WHEAT_CONFIRMED).sum())},
            {"metric": "non_wheat_downloaded_files_excluded", "value": int(downloads["crop_scope"].eq(NON_WHEAT_EXCLUDED).sum())},
            {"metric": "ambiguous_crop_downloaded_files_deferred", "value": int(downloads["crop_scope"].eq(AMBIGUOUS_REVIEW).sum())},
            {"metric": "files_parsed", "value": int(parse_log["status"].str.startswith("PASS").sum()) if not parse_log.empty else 0},
            {"metric": "large_matrices_scanned_as_bounded_axes", "value": int(parse_log["status"].eq("PASS_BOUNDED_AXIS").sum()) if not parse_log.empty else 0},
            {"metric": "files_skipped_by_complete_content_index", "value": int((parse_log["status"] == "INDEXED_NO_IDENTIFIER_MATCH").sum()) if not parse_log.empty else 0},
            {"metric": "structured_match_rows", "value": len(evidence)},
            {"metric": "matched_query_ids", "value": evidence["query_id"].nunique() if not evidence.empty else 0},
            {"metric": "direct_gid_exact_query_ids", "value": evidence.loc[evidence["evidence_class"] == "direct_gid_exact", "query_id"].nunique() if not evidence.empty else 0},
            {"metric": "unique_selection_history_query_ids", "value": evidence.loc[evidence["evidence_class"] == "selection_history_exact_unique", "query_id"].nunique() if not evidence.empty else 0},
            {"metric": "marker_bridge_candidate_query_ids", "value": bridge["query_id"].nunique() if not bridge.empty else 0},
            {"metric": "direct_marker_assignment_ready", "value": False},
            {"metric": "phenotype_values_read", "value": False},
            {"metric": "outer_test_metrics_read", "value": False},
            {"metric": "final_holdout_outcomes_read", "value": False},
        ]
    )
    qc.to_csv(out_dir / "dataverse_structured_evidence_qc.tsv", sep="\t", index=False)
    provenance = {
        "status": "complete",
        "selection_data": "resolver_identifiers_and_downloaded_repository_files_only",
        "resolver_query": {"path": str(resolver_path), "sha256": sha256_file(resolver_path)},
        "downloads_manifest": {"path": str(downloads_path), "sha256": sha256_file(downloads_path)},
        "content_prefilter": {
            "enabled": use_content_prefilter,
            "path": str(content_matches_path) if use_content_prefilter else "",
            "sha256": sha256_file(content_matches_path) if use_content_prefilter else "",
            "full_scan_formats": [
                ".xls",
                ".xls.gz",
                ".xlsx.gz",
                ".xlsm.gz",
                ".7z",
            ],
            "large_matrix_strategy": {
                "minimum_uncompressed_bytes": LARGE_STRUCTURED_BYTES,
                "axis_rows_scanned": LARGE_MATRIX_AXIS_ROWS,
                "genotype_cells_materialized": False,
            },
        },
        "crop_selection": {
            "policy": "only WHEAT_CONFIRMED downloaded sources are parsed",
            "search_results_path": str(search_results_path) if search_results_path.is_file() else "",
            "search_results_sha256": sha256_file(search_results_path) if search_results_path.is_file() else "",
            "source_crop_audit": str(out_dir / "dataverse_structured_source_crop_scope.tsv"),
        },
        "local_reuse_manifest": {
            "path": str(local_reuse_path) if local_reuse_path.is_file() else "",
            "sha256": sha256_file(local_reuse_path) if local_reuse_path.is_file() else "",
            "verified_reuse_rows": local_reuse_count,
        },
        "direct_marker_assignment_ready": False,
        "required_next_certification": "external_sample_axis_and_marker_call_concordance",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
    }
    write_json_atomic(provenance, out_dir / "dataverse_structured_evidence_provenance.json")
    write_json_atomic(
        {
            "status": "COMPLETE",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "files_considered": len(downloads),
            "files_parsed": int(parse_log["status"].str.startswith("PASS").sum())
            if not parse_log.empty
            else 0,
            "structured_match_rows": len(evidence),
        },
        run_status_path,
    )
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
