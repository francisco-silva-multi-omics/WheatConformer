from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from server_genotype_recovery.audit_dataverse_structured_evidence import (
    source_subtype,
    structured_parts,
)
from server_genotype_recovery.fetch_brapi_pedigree_markers import (
    clean,
    normalized_identifier,
    read_table,
    sha256_file,
    write_json_atomic,
)

MAPPING_IDENTITY_CLASSES = {
    "direct_gid_exact",
    "selection_history_exact_unique",
}

BRIDGE_COLUMNS = [
    "query_id",
    "query_text",
    "dataset_persistent_id",
    "external_alias",
    "normalized_external_alias",
    "alias_trial_gid_count",
    "alias_mapping_record_count",
    "alias_marker_location_count",
    "mapping_filename",
    "mapping_source_part",
    "mapping_source_row",
    "mapping_source_column",
    "mapping_column_header",
    "mapping_header_row",
    "marker_filename",
    "marker_source_part",
    "marker_source_row",
    "marker_source_column",
    "marker_column_header",
    "marker_axis_candidate",
    "bridge_confidence",
    "direct_marker_assignment_ready",
]


def plausible_external_alias(value: object, column_header: object = "") -> bool:
    text = clean(value)
    normalized = normalized_identifier(text)
    header = normalized_identifier(column_header)
    numeric_identifier_header = any(
        token in header for token in ("GID", "ENT", "ENTRY", "SID", "SAMPLE")
    )
    if normalized.isdigit():
        return numeric_identifier_header and int(normalized) > 0
    if not 4 <= len(normalized) <= 100:
        return False
    if normalized in {"TRUE", "FALSE", "NULL", "NONE", "MISSING", "SAMPLE", "GERMPLASM"}:
        return False
    if not re.search(r"[A-Z]", normalized) or not re.search(r"\d", normalized):
        return False
    if normalized in {"AA", "AB", "BA", "BB", "AABB", "BBAA"}:
        return False
    return True


def marker_axis_candidate(row: int, column: int, row_descriptor: object = "") -> str:
    descriptor = normalized_identifier(row_descriptor)
    if "GID" in descriptor and "IDENTIFIER" in descriptor:
        return "gid_metadata_row_sample_candidate"
    if "ENTRYNUMBER" in descriptor:
        return "entry_metadata_row_sample_candidate"
    if any(token in descriptor for token in ("RSALLELES", "SNPID", "MARKERID")):
        return "sample_label_header_row_candidate"
    if row == 0 and column == 0:
        return "corner_header_ambiguous"
    if row == 0:
        return "header_column_sample_candidate"
    if column == 0:
        return "first_column_sample_candidate"
    return "interior_cell_not_sample_axis"


def bridge_confidence(
    trial_gid_count: int,
    mapping_record_count: int,
    marker_location_count: int,
    axis_candidate: str,
) -> str:
    axis_supported = axis_candidate in {
        "header_column_sample_candidate",
        "first_column_sample_candidate",
        "gid_metadata_row_sample_candidate",
        "entry_metadata_row_sample_candidate",
        "sample_label_header_row_candidate",
    }
    if (
        trial_gid_count == 1
        and mapping_record_count == 1
        and marker_location_count == 1
        and axis_supported
    ):
        return "high_candidate_requires_call_concordance"
    if trial_gid_count == 1 and marker_location_count <= 5 and axis_supported:
        return "moderate_candidate_requires_disambiguation"
    return "ambiguous_or_non_axis"


def infer_mapping_headers(frame: pd.DataFrame) -> tuple[int | None, dict[int, str]]:
    expected = {
        "ENT",
        "ENTRY",
        "GID",
        "CID",
        "SID",
        "CROSSNAME",
        "SELECTIONHISTORY",
        "ORIGIN",
        "SAMPLE35K",
        "SAMPLEID",
    }
    best_row: int | None = None
    best_score = 0
    for row_number in range(min(30, len(frame))):
        values = frame.iloc[row_number].tolist()
        score = sum(normalized_identifier(value) in expected for value in values)
        if score > best_score:
            best_row, best_score = row_number, score
    if best_row is None or best_score < 2:
        return None, {}
    return best_row, {
        column: clean(value) for column, value in enumerate(frame.iloc[best_row].tolist())
    }


def infer_marker_sample_header_row(frame: pd.DataFrame) -> int | None:
    best_row: int | None = None
    best_score = 0
    expected = {"RS", "ALLELES", "CHROM", "POS", "STRAND", "SNPID", "MARKERID"}
    for row_number in range(min(30, len(frame))):
        normalized = [normalized_identifier(value) for value in frame.iloc[row_number].tolist()]
        score = sum(value in expected for value in normalized)
        if score > best_score:
            best_row, best_score = row_number, score
    return best_row if best_score >= 2 else None


def load_frames(
    downloads: pd.DataFrame,
    required_parts: set[tuple[str, str]] | None = None,
    all_parts_paths: set[str] | None = None,
    progress: bool = False,
) -> tuple[dict[tuple[str, str], pd.DataFrame], list[dict[str, object]]]:
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    log: list[dict[str, object]] = []
    all_parts_paths = all_parts_paths or set()
    records = downloads.to_dict("records")
    for file_number, record in enumerate(records, start=1):
        path = Path(clean(record.get("local_path")))
        if not path.is_file():
            continue
        if progress:
            print(
                f"[{file_number}/{len(records)}] load structured frames: "
                f"{clean(record.get('filename')) or path.name}",
                flush=True,
            )
        try:
            part_count = 0
            for part, frame in structured_parts(path):
                key = (str(path), part)
                if (
                    required_parts is None
                    or key in required_parts
                    or str(path) in all_parts_paths
                ):
                    frames[key] = frame.fillna("")
                part_count += 1
            log.append(
                {
                    "filename": record.get("filename", ""),
                    "status": "PASS",
                    "parts": part_count,
                    "detail": "",
                }
            )
        except Exception as exc:
            log.append(
                {
                    "filename": record.get("filename", ""),
                    "status": "SKIPPED_OR_FAILED",
                    "parts": 0,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
    return frames, log


def select_two_hop_downloads(
    downloads: pd.DataFrame,
    evidence: pd.DataFrame,
) -> tuple[pd.DataFrame, set[tuple[str, str]], set[str], set[str]]:
    mapping = evidence[
        evidence["evidence_class"].isin(MAPPING_IDENTITY_CLASSES)
        & evidence["source_subtype"].eq("germplasm_or_sample_mapping")
    ]
    mapping_parts = {
        (clean(row["local_path"]), clean(row["source_part"]))
        for row in mapping[["local_path", "source_part"]].to_dict("records")
        if clean(row["local_path"]) and clean(row["source_part"])
    }
    subtype = downloads.apply(
        lambda row: source_subtype(row.get("filename"), row.get("description")),
        axis=1,
    )
    mapping_candidate_mask = subtype.eq("germplasm_or_sample_mapping")
    marker_candidate_mask = subtype.eq("marker_matrix_candidate")
    mapping_datasets = set(mapping["dataset_persistent_id"].map(clean)) - {""}
    explicit_mapping_datasets = set(
        downloads.loc[mapping_candidate_mask, "dataset_persistent_id"].map(clean)
    )
    marker_datasets = set(
        downloads.loc[marker_candidate_mask, "dataset_persistent_id"].map(clean)
    )
    bridge_datasets = (mapping_datasets | explicit_mapping_datasets) & marker_datasets
    marker_mask = marker_candidate_mask & downloads["dataset_persistent_id"].map(
        clean
    ).isin(bridge_datasets)
    direct_mapping_mask = mapping_candidate_mask & downloads[
        "dataset_persistent_id"
    ].map(clean).isin(bridge_datasets)
    marker_paths = set(downloads.loc[marker_mask, "local_path"].map(clean)) - {""}
    direct_mapping_paths = set(
        downloads.loc[direct_mapping_mask, "local_path"].map(clean)
    ) - {""}
    required_paths = (
        {path for path, _ in mapping_parts}
        | direct_mapping_paths
        | marker_paths
    )
    selected = downloads[downloads["local_path"].map(clean).isin(required_paths)].copy()
    return selected, mapping_parts, direct_mapping_paths, marker_paths


def collect_mapping_aliases(
    evidence: pd.DataFrame,
    frames: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    selected = evidence[
        evidence["evidence_class"].isin(MAPPING_IDENTITY_CLASSES)
        & (evidence["source_subtype"] == "germplasm_or_sample_mapping")
    ].copy()
    rows: list[dict[str, object]] = []
    keys = [
        "query_id",
        "query_text",
        "dataset_persistent_id",
        "filename",
        "local_path",
        "source_part",
        "source_row",
    ]
    for values, group in selected.groupby(keys, dropna=False, sort=False):
        record = dict(zip(keys, values))
        frame = frames.get((clean(record["local_path"]), clean(record["source_part"])))
        row_number = int(record["source_row"])
        if frame is None or row_number >= len(frame):
            continue
        row_values = frame.iloc[row_number].tolist()
        header_row, headers = infer_mapping_headers(frame)
        matched_norms = {normalized_identifier(value) for value in group["query_text"]}
        for column, value in enumerate(row_values):
            normalized = normalized_identifier(value)
            header = headers.get(column, "")
            if normalized in matched_norms or not plausible_external_alias(value, header):
                continue
            rows.append(
                {
                    "query_id": record["query_id"],
                    "query_text": record["query_text"],
                    "dataset_persistent_id": record["dataset_persistent_id"],
                    "external_alias": clean(value),
                    "normalized_external_alias": normalized,
                    "mapping_filename": record["filename"],
                    "mapping_source_part": record["source_part"],
                    "mapping_source_row": row_number,
                    "mapping_source_column": column,
                    "mapping_column_header": header,
                    "mapping_header_row": header_row,
                }
            )
    if not rows:
        return pd.DataFrame(columns=[
            "query_id", "query_text", "dataset_persistent_id", "external_alias",
            "normalized_external_alias", "mapping_filename", "mapping_source_part",
            "mapping_source_row", "mapping_source_column", "mapping_column_header",
            "mapping_header_row",
        ])
    return pd.DataFrame(rows).drop_duplicates()


def canonical_gid(value: object) -> str:
    normalized = normalized_identifier(value)
    match = re.fullmatch(r"(?:GID)?0*([1-9][0-9]*)", normalized)
    return f"GID{match.group(1)}" if match else ""


def collect_direct_gid_mapping_aliases(
    resolver_gids: set[str],
    downloads: pd.DataFrame,
    frames: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    mapping_downloads = downloads[
        downloads.apply(
            lambda row: source_subtype(row.get("filename"), row.get("description"))
            == "germplasm_or_sample_mapping",
            axis=1,
        )
    ]
    by_path = {
        clean(row["local_path"]): row for row in mapping_downloads.to_dict("records")
    }
    for (path, part), frame in frames.items():
        source = by_path.get(path)
        if source is None:
            continue
        header_row, headers = infer_mapping_headers(frame)
        if header_row is None:
            continue
        normalized_headers = {
            column: normalized_identifier(header) for column, header in headers.items()
        }
        gid_columns = [
            column
            for column, header in normalized_headers.items()
            if header in {"GID", "CIMMYTGID", "GENERALIDENTIFIER"}
        ]
        alias_columns = [
            column
            for column, header in normalized_headers.items()
            if header in {"SAMPLEID", "SAMPLE35K", "DNAID", "DARTSAMPLEID"}
        ]
        if not gid_columns or not alias_columns:
            continue
        for row_number in range(header_row + 1, len(frame)):
            values = frame.iloc[row_number].tolist()
            gids = {
                canonical_gid(values[column])
                for column in gid_columns
                if column < len(values)
            } & resolver_gids
            if not gids:
                continue
            for alias_column in alias_columns:
                if alias_column >= len(values):
                    continue
                alias = clean(values[alias_column])
                normalized_alias = normalized_identifier(alias)
                if not plausible_external_alias(
                    alias, headers.get(alias_column, "")
                ):
                    continue
                for gid in gids:
                    rows.append(
                        {
                            "query_id": gid,
                            "query_text": gid,
                            "dataset_persistent_id": source.get(
                                "dataset_persistent_id", ""
                            ),
                            "external_alias": alias,
                            "normalized_external_alias": normalized_alias,
                            "mapping_filename": source.get("filename", ""),
                            "mapping_source_part": part,
                            "mapping_source_row": row_number,
                            "mapping_source_column": alias_column,
                            "mapping_column_header": headers.get(alias_column, ""),
                            "mapping_header_row": header_row,
                        }
                    )
    columns = [
        "query_id",
        "query_text",
        "dataset_persistent_id",
        "external_alias",
        "normalized_external_alias",
        "mapping_filename",
        "mapping_source_part",
        "mapping_source_row",
        "mapping_source_column",
        "mapping_column_header",
        "mapping_header_row",
    ]
    return (
        pd.DataFrame(rows, columns=columns).drop_duplicates()
        if rows
        else pd.DataFrame(columns=columns)
    )


def collect_marker_locations(
    aliases: set[str],
    downloads: pd.DataFrame,
    frames: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    marker_downloads = downloads[
        downloads.apply(
            lambda row: source_subtype(row.get("filename"), row.get("description"))
            == "marker_matrix_candidate",
            axis=1,
        )
    ]
    by_path = {clean(row["local_path"]): row for row in marker_downloads.to_dict("records")}
    for (path, part), frame in frames.items():
        source = by_path.get(path)
        if source is None:
            continue
        sample_header_row = infer_marker_sample_header_row(frame)
        for row_number, values in enumerate(frame.itertuples(index=False, name=None)):
            descriptor = " ".join(clean(value) for value in values[:12] if clean(value))
            for column, value in enumerate(values):
                normalized = normalized_identifier(value)
                if normalized not in aliases:
                    continue
                rows.append(
                    {
                        "dataset_persistent_id": source.get("dataset_persistent_id", ""),
                        "normalized_external_alias": normalized,
                        "marker_filename": source.get("filename", ""),
                        "marker_source_part": part,
                        "marker_source_row": row_number,
                        "marker_source_column": column,
                        "marker_column_header": (
                            clean(frame.iloc[sample_header_row, column])
                            if sample_header_row is not None
                            else ""
                        ),
                        "marker_axis_candidate": marker_axis_candidate(
                            row_number, column, descriptor
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_bridges(aliases: pd.DataFrame, marker_locations: pd.DataFrame) -> pd.DataFrame:
    if aliases.empty or marker_locations.empty:
        return pd.DataFrame(columns=BRIDGE_COLUMNS)
    alias_gid_counts = aliases.groupby("normalized_external_alias")["query_id"].nunique()
    alias_record_counts = (
        aliases[
            [
                "dataset_persistent_id",
                "normalized_external_alias",
                "mapping_filename",
                "mapping_source_part",
                "mapping_source_row",
            ]
        ]
        .drop_duplicates()
        .groupby(["dataset_persistent_id", "normalized_external_alias"])
        .size()
    )
    marker_counts = marker_locations.groupby(
        ["dataset_persistent_id", "normalized_external_alias"]
    ).size()
    merged = aliases.merge(
        marker_locations,
        on=["dataset_persistent_id", "normalized_external_alias"],
        how="inner",
    )
    rows: list[dict[str, object]] = []
    for row in merged.to_dict("records"):
        normalized = row["normalized_external_alias"]
        dataset_key = (row["dataset_persistent_id"], normalized)
        gid_count = int(alias_gid_counts.get(normalized, 0))
        mapping_count = int(alias_record_counts.get(dataset_key, 0))
        marker_count = int(marker_counts.get(dataset_key, 0))
        confidence = bridge_confidence(
            gid_count, mapping_count, marker_count, row["marker_axis_candidate"]
        )
        rows.append(
            {
                **row,
                "alias_trial_gid_count": gid_count,
                "alias_mapping_record_count": mapping_count,
                "alias_marker_location_count": marker_count,
                "bridge_confidence": confidence,
                "direct_marker_assignment_ready": False,
            }
        )
    return pd.DataFrame(rows, columns=BRIDGE_COLUMNS).drop_duplicates()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit dataset-local selection-history to marker-matrix alias bridges."
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
    structured_dir = recovery_dir / "structured_evidence"
    out_dir = args.out_dir or structured_dir / "two_hop_marker_bridges"
    out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    downloads_path = recovery_dir / "dataverse_downloads.tsv"
    evidence_path = structured_dir / "dataverse_structured_evidence.tsv.gz"
    source_manifest_path = structured_dir / "dataverse_structured_source_crop_scope.tsv"
    run_status_path = structured_dir / "dataverse_structured_evidence_run_status.json"
    local_reuse_path = (
        recovery_dir / "tier2_inventory/dataverse_tier2_local_reuse_manifest.tsv"
    )
    resolver_path = (
        args.resolver_query
        if args.resolver_query.is_absolute()
        else root / args.resolver_query
    )
    required = [
        downloads_path,
        evidence_path,
        source_manifest_path,
        run_status_path,
        resolver_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Completed structured evidence inputs are required: {missing}"
        )
    run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
    if run_status.get("status") != "COMPLETE":
        raise ValueError(
            "Structured evidence is incomplete; rerun it before the two-hop audit"
        )

    source_manifest = read_table(source_manifest_path)
    if not source_manifest["crop_scope"].eq("WHEAT_CONFIRMED").all():
        source_manifest = source_manifest[
            source_manifest["crop_scope"].eq("WHEAT_CONFIRMED")
        ].copy()
    metadata = read_table(downloads_path)
    metadata = metadata[
        metadata["download_status"].isin(["DOWNLOADED", "REUSED"])
    ].copy()
    if local_reuse_path.is_file():
        metadata = pd.concat(
            [metadata, read_table(local_reuse_path)], ignore_index=True
        )
    for column in ("description", "candidate_role"):
        if column not in metadata.columns:
            metadata[column] = ""
    metadata = metadata.drop_duplicates(
        ["dataset_persistent_id", "datafile_id"], keep="last"
    )
    downloads = source_manifest.merge(
        metadata[
            [
                "dataset_persistent_id",
                "datafile_id",
                "description",
                "candidate_role",
            ]
        ],
        on=["dataset_persistent_id", "datafile_id"],
        how="left",
        validate="one_to_one",
    )
    downloads[["description", "candidate_role"]] = downloads[
        ["description", "candidate_role"]
    ].fillna("")
    evidence = pd.read_csv(evidence_path, sep="\t", dtype=str)
    if "crop_scope" not in evidence.columns:
        raise ValueError(
            "Structured evidence is stale and lacks crop_scope; rerun the "
            "wheat-gated structured evidence audit"
        )
    invalid_crop = evidence[~evidence["crop_scope"].eq("WHEAT_CONFIRMED")]
    if not invalid_crop.empty:
        raise ValueError(
            "Structured evidence contains non-wheat or ambiguous rows; rerun the "
            "wheat-gated structured evidence audit"
        )
    evidence["source_row"] = pd.to_numeric(evidence["source_row"], errors="coerce").fillna(-1).astype(int)
    resolver = read_table(resolver_path)
    resolver_gid_columns = [
        column
        for column in (
            "sample_id",
            "query_id",
            "panel_sample_id_expected",
            "gid",
        )
        if column in resolver.columns
    ]
    resolver_gids = {
        canonical_gid(value)
        for column in resolver_gid_columns
        for value in resolver[column]
        if canonical_gid(value)
    }
    if not resolver_gids:
        raise ValueError("No canonical GIDs were found in the resolver query")
    (
        selected_downloads,
        mapping_parts,
        direct_mapping_paths,
        marker_paths,
    ) = select_two_hop_downloads(downloads, evidence)
    frames, parse_log = load_frames(
        selected_downloads,
        required_parts=mapping_parts,
        all_parts_paths=direct_mapping_paths | marker_paths,
        progress=True,
    )
    evidence_aliases = collect_mapping_aliases(evidence, frames)
    direct_gid_aliases = collect_direct_gid_mapping_aliases(
        resolver_gids,
        selected_downloads,
        frames,
    )
    aliases = pd.concat(
        [evidence_aliases, direct_gid_aliases], ignore_index=True
    ).drop_duplicates()
    locations = collect_marker_locations(
        set(aliases["normalized_external_alias"]) if not aliases.empty else set(),
        selected_downloads,
        frames,
    )
    bridges = build_bridges(aliases, locations)

    aliases.to_csv(out_dir / "dataverse_mapping_alias_candidates.tsv", sep="\t", index=False)
    locations.to_csv(out_dir / "dataverse_marker_alias_locations.tsv", sep="\t", index=False)
    bridges.to_csv(out_dir / "dataverse_two_hop_marker_bridges.tsv", sep="\t", index=False)
    pd.DataFrame(parse_log).to_csv(out_dir / "dataverse_two_hop_parse_log.tsv", sep="\t", index=False)
    if bridges.empty:
        summary = pd.DataFrame(columns=["bridge_confidence", "candidate_gids", "bridge_rows"])
    else:
        summary = (
            bridges.groupby("bridge_confidence")
            .agg(candidate_gids=("query_id", "nunique"), bridge_rows=("query_id", "size"))
            .reset_index()
        )
    summary.to_csv(out_dir / "dataverse_two_hop_marker_bridge_summary.tsv", sep="\t", index=False)
    high = bridges[bridges["bridge_confidence"] == "high_candidate_requires_call_concordance"] if not bridges.empty else bridges
    qc = pd.DataFrame(
        [
            {"metric": "structured_unique_selection_gids", "value": evidence.loc[evidence["evidence_class"] == "selection_history_exact_unique", "query_id"].nunique()},
            {"metric": "resolver_canonical_gids", "value": len(resolver_gids)},
            {"metric": "structured_source_files_considered", "value": len(downloads)},
            {"metric": "certified_local_reuse_files_considered", "value": int(downloads["source_origin"].eq("certified_local_reuse").sum()) if "source_origin" in downloads.columns else 0},
            {"metric": "dataset_local_files_parsed", "value": len(selected_downloads)},
            {"metric": "direct_gid_mapping_alias_rows", "value": len(direct_gid_aliases)},
            {"metric": "mapping_alias_candidate_rows", "value": len(aliases)},
            {"metric": "marker_alias_location_rows", "value": len(locations)},
            {"metric": "two_hop_bridge_rows", "value": len(bridges)},
            {"metric": "two_hop_candidate_gids", "value": bridges["query_id"].nunique() if not bridges.empty else 0},
            {"metric": "high_candidate_gids", "value": high["query_id"].nunique() if not high.empty else 0},
            {"metric": "direct_marker_assignment_ready", "value": False},
            {"metric": "phenotype_values_read", "value": False},
            {"metric": "outer_test_metrics_read", "value": False},
            {"metric": "final_holdout_outcomes_read", "value": False},
        ]
    )
    qc.to_csv(out_dir / "dataverse_two_hop_marker_bridge_qc.tsv", sep="\t", index=False)
    provenance = {
        "status": "complete",
        "selection_data": "downloaded_repository_identifiers_only",
        "downloads_manifest_sha256": sha256_file(downloads_path),
        "structured_source_manifest_sha256": sha256_file(source_manifest_path),
        "structured_run_status_sha256": sha256_file(run_status_path),
        "structured_evidence_sha256": sha256_file(evidence_path),
        "resolver_query_sha256": sha256_file(resolver_path),
        "direct_marker_assignment_ready": False,
        "required_next_certification": "marker_sample_axis_and_call_concordance",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
    }
    write_json_atomic(provenance, out_dir / "dataverse_two_hop_marker_bridge_provenance.json")
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
