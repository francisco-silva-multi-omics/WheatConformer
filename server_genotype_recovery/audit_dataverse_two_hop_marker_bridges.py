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
    "marker_filename",
    "marker_source_part",
    "marker_source_row",
    "marker_source_column",
    "marker_column_header",
    "marker_axis_candidate",
    "bridge_confidence",
    "direct_marker_assignment_ready",
]


def plausible_external_alias(value: object) -> bool:
    text = clean(value)
    normalized = normalized_identifier(text)
    if not 4 <= len(normalized) <= 100:
        return False
    if normalized in {"TRUE", "FALSE", "NULL", "NONE", "MISSING", "SAMPLE", "GERMPLASM"}:
        return False
    if not re.search(r"[A-Z]", normalized) or not re.search(r"\d", normalized):
        return False
    if normalized in {"AA", "AB", "BA", "BB", "AABB", "BBAA"}:
        return False
    return True


def marker_axis_candidate(row: int, column: int) -> str:
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


def load_frames(downloads: pd.DataFrame) -> tuple[dict[tuple[str, str], pd.DataFrame], list[dict[str, object]]]:
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    log: list[dict[str, object]] = []
    for record in downloads.to_dict("records"):
        path = Path(clean(record.get("local_path")))
        if not path.is_file():
            continue
        try:
            part_count = 0
            for part, frame in structured_parts(path):
                frames[(str(path), part)] = frame.fillna("")
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


def collect_mapping_aliases(
    evidence: pd.DataFrame,
    frames: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    selected = evidence[
        (evidence["evidence_class"] == "selection_history_exact_unique")
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
        matched_norms = {normalized_identifier(value) for value in group["query_text"]}
        for column, value in enumerate(row_values):
            normalized = normalized_identifier(value)
            if normalized in matched_norms or not plausible_external_alias(value):
                continue
            header = clean(frame.iloc[0, column]) if len(frame) else ""
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
                }
            )
    if not rows:
        return pd.DataFrame(columns=[
            "query_id", "query_text", "dataset_persistent_id", "external_alias",
            "normalized_external_alias", "mapping_filename", "mapping_source_part",
            "mapping_source_row", "mapping_source_column", "mapping_column_header",
        ])
    return pd.DataFrame(rows).drop_duplicates()


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
        for row_number, values in enumerate(frame.itertuples(index=False, name=None)):
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
                        "marker_column_header": clean(frame.iloc[0, column]) if len(frame) else "",
                        "marker_axis_candidate": marker_axis_candidate(row_number, column),
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
    if not downloads_path.is_file() or not evidence_path.is_file():
        raise FileNotFoundError("Structured evidence and Dataverse downloads are required")

    downloads = read_table(downloads_path)
    downloads = downloads[downloads["download_status"].isin(["DOWNLOADED", "REUSED"])].copy()
    evidence = pd.read_csv(evidence_path, sep="\t", dtype=str)
    evidence["source_row"] = pd.to_numeric(evidence["source_row"], errors="coerce").fillna(-1).astype(int)
    frames, parse_log = load_frames(downloads)
    aliases = collect_mapping_aliases(evidence, frames)
    locations = collect_marker_locations(
        set(aliases["normalized_external_alias"]) if not aliases.empty else set(),
        downloads,
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
        "structured_evidence_sha256": sha256_file(evidence_path),
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
