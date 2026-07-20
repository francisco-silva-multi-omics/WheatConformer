from __future__ import annotations

import argparse
import gzip
import io
import json
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd

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


def _read_delimited(source: object, suffix: str) -> pd.DataFrame:
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
        )
    except (UnicodeDecodeError, pd.errors.ParserError):
        return pd.read_csv(
            source,
            sep="\t",
            header=None,
            dtype=str,
            compression=compression,
            encoding="latin-1",
            on_bad_lines="skip",
        )


def structured_parts(path: Path) -> Iterator[tuple[str, pd.DataFrame]]:
    lower = path.name.lower()
    if lower.endswith((".xlsx", ".xlsm")):
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
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
                payload = archive.read(member)
                yield f"archive:{member.filename}", _read_delimited(
                    io.BytesIO(payload), member_lower
                )
        return
    if lower.endswith((".txt", ".tsv", ".tab", ".csv", ".txt.gz", ".tsv.gz", ".csv.gz")):
        yield "file", _read_delimited(path, lower)
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
        normalized_cells: dict[str, list[int]] = {}
        for column_position, value in enumerate(values):
            normalized = normalized_identifier(value)
            if normalized in term_index:
                normalized_cells.setdefault(normalized, []).append(column_position)
        if not normalized_cells:
            continue
        context_positions = set(range(min(3, len(values))))
        for positions in normalized_cells.values():
            for position in positions:
                context_positions.update(
                    range(max(0, position - 2), min(len(values), position + 3))
                )
        context = {
            str(position): clean(values[position])[:500]
            for position in sorted(context_positions)
            if clean(values[position])
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
                            "dataset_persistent_id": source.get("dataset_persistent_id", ""),
                            "datafile_id": source.get("datafile_id", ""),
                            "candidate_role": source.get("candidate_role", ""),
                            "source_subtype": subtype,
                            "filename": source.get("filename", ""),
                            "local_path": source.get("local_path", ""),
                            "source_part": source_part,
                            "source_row": row_position,
                            "source_column": position,
                            "cell_value": clean(values[position])[:2000],
                            "row_context_json": json.dumps(context, sort_keys=True),
                        }
                    )
    return hits


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
    downloads_path = recovery_dir / "dataverse_downloads.tsv"
    if not downloads_path.is_file():
        raise FileNotFoundError(downloads_path)
    if not resolver_path.is_file():
        raise FileNotFoundError(resolver_path)

    resolver = read_table(resolver_path)
    term_index, term_count = term_index_from_resolver(resolver)
    downloads = read_table(downloads_path)
    downloads = downloads[downloads["download_status"].isin(["DOWNLOADED", "REUSED"])].copy()
    evidence_rows: list[dict[str, object]] = []
    parse_rows: list[dict[str, object]] = []
    for record in downloads.to_dict("records"):
        path = Path(clean(record.get("local_path")))
        if not path.is_file():
            parse_rows.append({"filename": record.get("filename", ""), "status": "MISSING", "parts": 0, "rows": 0, "detail": str(path)})
            continue
        parts = 0
        parsed_rows = 0
        try:
            for source_part, frame in structured_parts(path):
                parts += 1
                parsed_rows += len(frame)
                evidence_rows.extend(scan_frame(frame, term_index, record, source_part))
            status, detail = "PASS", ""
        except Exception as exc:
            status, detail = "SKIPPED_OR_FAILED", f"{type(exc).__name__}: {exc}"
        parse_rows.append({"filename": record.get("filename", ""), "status": status, "parts": parts, "rows": parsed_rows, "detail": detail})

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
            {"metric": "files_parsed", "value": int((parse_log["status"] == "PASS").sum()) if not parse_log.empty else 0},
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
        "direct_marker_assignment_ready": False,
        "required_next_certification": "external_sample_axis_and_marker_call_concordance",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
    }
    write_json_atomic(provenance, out_dir / "dataverse_structured_evidence_provenance.json")
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
