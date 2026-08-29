#!/usr/bin/env python3
"""Resolve CIMMYT 130K metadata evidence for Stage-1 v2 GIDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROTOCOL_VERSION = "cimmyt_130k_identifier_metadata_resolution_v1"
GID_SEARCH = re.compile(r"GID[0-9]+", re.IGNORECASE)
GID_EXACT = re.compile(r"^GID[0-9]+$", re.IGNORECASE)
WGE_EXACT = re.compile(r"^WGE[0-9]+$", re.IGNORECASE)


def normalize(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).upper()


def identifier_class(value: object) -> str:
    identifier = normalize(value)
    if not identifier:
        return "EMPTY"
    if GID_EXACT.fullmatch(identifier):
        return "GID"
    if WGE_EXACT.fullmatch(identifier):
        return "WGE"
    if identifier in {"BLANK", "GIDNA", "NA", "N/A"}:
        return "CONTROL_OR_PLACEHOLDER"
    return "OTHER"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_hmp_axis(path: Path) -> tuple[list[str], set[str], set[str]]:
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if not name.startswith("__MACOSX"))
        header = archive.open(member).readline().decode("utf-8-sig", "replace")
    labels = [normalize(value) for value in header.rstrip("\r\n").split("\t")[11:]]
    gids = {
        match.group(0).upper()
        for label in labels
        if (match := GID_SEARCH.search(label))
    }
    wges = {label for label in labels if WGE_EXACT.fullmatch(label)}
    return labels, gids, wges


def terminal_class(rows: pd.DataFrame) -> str:
    classes = set(rows["crosswalk_class"])
    submitted = {value for value in rows["submitted_identifier"] if value}
    dryad = rows.iloc[0]["dryad_identifier"]
    if "CONFLICTING_GID_FOR_BARCODE" in classes or any(
        value not in {dryad, "GIDNA"} for value in submitted
    ):
        return "CONFLICTING_SUBMITTED_IDENTIFIER"
    exact = int((rows["crosswalk_class"] == "EXACT_IDENTIFIER_AND_BARCODE").sum())
    dryad_only = int((rows["crosswalk_class"] == "DRYAD_ONLY_BARCODE").sum())
    placeholder = int((rows["submitted_identifier"] == "GIDNA").sum())
    if exact and not dryad_only and not placeholder:
        return "EXACT_SUBMISSION_ABSENT_FINAL_MATRIX"
    if exact and dryad_only and not placeholder:
        return "PARTIAL_EXACT_SUBMISSION_ABSENT_FINAL_MATRIX"
    if placeholder and not exact:
        return "SUBMITTED_PLACEHOLDER_ABSENT_FINAL_MATRIX"
    if dryad_only and not exact:
        return "PUBLIC_SUBMISSION_METADATA_UNINFORMATIVE"
    return "MIXED_OR_INCOMPLETE_METADATA_REVIEW"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def count_markdown(counts: dict[str, int], value_label: str) -> str:
    rows = [f"| class | {value_label} |", "|---|---:|"]
    rows.extend(f"| {key} | {value:,} |" for key, value in counts.items())
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--crosswalk",
        type=Path,
        default=Path(
            "audit/v2/cimmyt_130k_identifier_metadata_fetch_v1/"
            "dryad_ncbi_barcode_crosswalk.tsv.gz"
        ),
    )
    parser.add_argument(
        "--stage1-overlap",
        type=Path,
        default=Path(
            "audit/v2/phase3h_external_genotype_panel_search_v1/"
            "dryad_130k_2013_2023_gid_overlap.tsv"
        ),
    )
    parser.add_argument(
        "--hmp",
        type=Path,
        default=Path("GENOTYPIC_DATA/CIMMYT_Filtered.130K.GIDs.hmp.txt.zip"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("audit/v2/cimmyt_130k_identifier_metadata_resolution_v1"),
    )
    args = parser.parse_args()

    root = args.root.resolve()

    def resolved(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    crosswalk_path = resolved(args.crosswalk)
    overlap_path = resolved(args.stage1_overlap)
    hmp_path = resolved(args.hmp)
    out_dir = resolved(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crosswalk = pd.read_csv(crosswalk_path, sep="\t", dtype=str).fillna("")
    overlap = pd.read_csv(overlap_path, sep="\t", dtype=str).fillna("")
    matrix_labels, matrix_gids, matrix_wges = read_hmp_axis(hmp_path)
    matrix_axis = set(matrix_labels)
    secondary = {
        normalize(value)
        for value in overlap.loc[
            overlap["in_ordered_secondary_universe"].str.lower().eq("true"),
            "external_gid",
        ]
    }
    missing = secondary - matrix_gids
    present = secondary & matrix_gids

    for column in ("dryad_identifier", "submitted_identifier", "crosswalk_class"):
        crosswalk[column] = crosswalk[column].map(normalize)
    target_rows = crosswalk[crosswalk["dryad_identifier"].isin(missing)].copy()

    gid_rows = []
    for gid, rows in target_rows.groupby("dryad_identifier", sort=True):
        class_counts = rows["crosswalk_class"].value_counts()
        submitted = sorted({value for value in rows["submitted_identifier"] if value})
        gid_rows.append(
            {
                "gid": gid,
                "terminal_metadata_class": terminal_class(rows),
                "key_rows": len(rows),
                "run_count": rows["run_accession"].nunique(),
                "runs": ";".join(sorted(rows["run_accession"].unique())),
                "library_count": rows["run_library_key"].nunique(),
                "libraries": ";".join(sorted(rows["run_library_key"].unique())),
                "exact_identifier_barcode_rows": int(
                    class_counts.get("EXACT_IDENTIFIER_AND_BARCODE", 0)
                ),
                "dryad_only_barcode_rows": int(class_counts.get("DRYAD_ONLY_BARCODE", 0)),
                "other_identifier_mismatch_rows": int(
                    class_counts.get("OTHER_IDENTIFIER_MISMATCH", 0)
                ),
                "submitted_identifiers": ";".join(submitted),
                "submitted_identifier_in_matrix_axis": any(
                    value in matrix_axis for value in submitted
                ),
                "direct_gid_in_matrix_axis": gid in matrix_gids,
            }
        )
    gid_summary = pd.DataFrame(gid_rows).sort_values("gid")

    target_runs = set(target_rows["run_accession"])
    target_run_rows = crosswalk[crosswalk["run_accession"].isin(target_runs)].copy()
    target_run_rows["dryad_gid"] = target_run_rows["dryad_identifier"].where(
        target_run_rows["dryad_identifier"].map(lambda value: bool(GID_EXACT.fullmatch(value)))
    )
    run_rows = []
    for run, rows in target_run_rows.groupby("run_accession", sort=True):
        target = rows[rows["dryad_identifier"].isin(missing)]
        dryad_gids = {value for value in rows["dryad_gid"] if value}
        retained_gids = dryad_gids & matrix_gids
        classes = rows["crosswalk_class"].value_counts()
        target_classes = target["crosswalk_class"].value_counts()
        exact_target = int(target_classes.get("EXACT_IDENTIFIER_AND_BARCODE", 0))
        if exact_target and not retained_gids:
            interpretation = "WHOLE_LIBRARY_AXIS_ABSENCE_AFTER_EXACT_SUBMISSION"
        elif exact_target:
            interpretation = "MIXED_RETENTION_WITH_EXACT_SUBMISSION_EVIDENCE"
        else:
            interpretation = "PUBLIC_SUBMISSION_METADATA_UNINFORMATIVE"
        run_rows.append(
            {
                "run_accession": run,
                "run_library_key": ";".join(
                    sorted(value for value in rows["run_library_key"].unique() if value)
                ),
                "dryad_key_rows": int(rows["dryad_identifier"].ne("").sum()),
                "dryad_gid_count": len(dryad_gids),
                "retained_matrix_gid_count": len(retained_gids),
                "missing_stage1_gid_count": target["dryad_identifier"].nunique(),
                "target_exact_identifier_barcode_rows": exact_target,
                "target_dryad_only_barcode_rows": int(
                    target_classes.get("DRYAD_ONLY_BARCODE", 0)
                ),
                "all_exact_identifier_barcode_rows": int(
                    classes.get("EXACT_IDENTIFIER_AND_BARCODE", 0)
                ),
                "all_dryad_only_barcode_rows": int(classes.get("DRYAD_ONLY_BARCODE", 0)),
                "all_ncbi_only_barcode_rows": int(classes.get("NCBI_ONLY_BARCODE", 0)),
                "interpretation": interpretation,
            }
        )
    run_summary = pd.DataFrame(run_rows).sort_values("run_accession")

    mismatches = crosswalk[
        crosswalk["crosswalk_class"] == "OTHER_IDENTIFIER_MISMATCH"
    ].copy()
    mismatches["dryad_identifier_class"] = mismatches["dryad_identifier"].map(identifier_class)
    mismatches["submitted_identifier_class"] = mismatches["submitted_identifier"].map(identifier_class)
    mismatches["submitted_identifier_in_matrix_axis_recomputed"] = mismatches[
        "submitted_identifier"
    ].isin(matrix_axis)
    mismatches["dryad_identifier_in_stage1_missing_set"] = mismatches[
        "dryad_identifier"
    ].isin(missing)

    namespace_rows = []
    for source_column in ("dryad_identifier", "submitted_identifier"):
        values = crosswalk[source_column].map(normalize)
        for namespace, count in values.map(identifier_class).value_counts().sort_index().items():
            namespace_rows.append(
                {
                    "source_column": source_column,
                    "identifier_class": namespace,
                    "rows": int(count),
                    "unique_identifiers": int(values[values.map(identifier_class) == namespace].nunique()),
                    "identifiers_in_matrix_axis": int(
                        values[
                            (values.map(identifier_class) == namespace)
                            & values.isin(matrix_axis)
                        ].nunique()
                    ),
                }
            )
    namespace_summary = pd.DataFrame(namespace_rows)

    gid_summary.to_csv(out_dir / "stage1_missing_gid_metadata_classification.tsv", sep="\t", index=False)
    run_summary.to_csv(out_dir / "stage1_missing_gid_run_summary.tsv", sep="\t", index=False)
    mismatches.to_csv(out_dir / "all_identifier_mismatches.tsv", sep="\t", index=False)
    namespace_summary.to_csv(out_dir / "identifier_namespace_summary.tsv", sep="\t", index=False)

    terminal_counts = gid_summary["terminal_metadata_class"].value_counts().sort_index().to_dict()
    run_counts = run_summary["interpretation"].value_counts().sort_index().to_dict()
    exact_supported_gids = int(gid_summary["exact_identifier_barcode_rows"].gt(0).sum())
    uninformative_gids = int(gid_summary["exact_identifier_barcode_rows"].eq(0).sum())
    decision = {
        "status": "PASS_METADATA_AUDIT_NO_GID_WGE_CROSSWALK_RAW_QC_REQUIRED",
        "protocol_version": PROTOCOL_VERSION,
        "selection_data": "public_identifiers_and_genotype_axis_only",
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "sequence_payloads_read": False,
        "secondary_stage1_gids": len(secondary),
        "secondary_stage1_gids_in_matrix": len(present),
        "secondary_stage1_gids_absent_matrix": len(missing),
        "absent_gids_classified": len(gid_summary),
        "target_run_count": len(target_runs),
        "target_key_rows": len(target_rows),
        "target_gids_with_exact_submission_evidence": exact_supported_gids,
        "target_gids_with_uninformative_public_submission_metadata": uninformative_gids,
        "target_gids_with_exact_submission_evidence_fraction": (
            exact_supported_gids / len(gid_summary)
        ),
        "matrix_axis_rows": len(matrix_labels),
        "matrix_embedded_gid_count": len(matrix_gids),
        "matrix_wge_axis_count": len(matrix_wges),
        "direct_gid_wge_crosswalk_candidates": int(
            crosswalk["crosswalk_class"].isin(
                {"GID_TO_WGE_ALIAS_CANDIDATE", "WGE_TO_GID_ALIAS_CANDIDATE"}
            ).sum()
        ),
        "identifier_mismatch_rows": len(mismatches),
        "identifier_mismatch_stage1_target_rows": int(
            mismatches["dryad_identifier_in_stage1_missing_set"].sum()
        ),
        "terminal_metadata_class_counts": terminal_counts,
        "target_run_interpretation_counts": run_counts,
        "conclusion": (
            "Public SRA metadata does not explain the missing Stage-1 GIDs as WGE aliases. "
            "Exact submitted barcode evidence localizes downstream omission for the supported subset; "
            "runs without public barcode tables remain metadata-inconclusive and require raw demultiplexing/QC."
        ),
        "inputs": {
            "crosswalk": {"path": str(crosswalk_path), "sha256": sha256_file(crosswalk_path)},
            "stage1_overlap": {"path": str(overlap_path), "sha256": sha256_file(overlap_path)},
            "hapmap": {"path": str(hmp_path), "sha256": sha256_file(hmp_path)},
        },
    }
    write_json(out_dir / "identifier_metadata_resolution_decision.json", decision)

    report = f"""# CIMMYT 130K Identifier Metadata Resolution

Status: `{decision['status']}`

- Stage-1 v2 secondary GIDs: {len(secondary):,}
- Directly present in the released matrix: {len(present):,}
- Absent and classified: {len(missing):,}
- Implicated runs: {len(target_runs):,}
- Target key rows: {len(target_rows):,}
- GIDs with exact submitted GID/barcode evidence: {exact_supported_gids:,} ({exact_supported_gids / len(gid_summary):.2%})
- GIDs with uninformative public submission metadata: {uninformative_gids:,}
- Released WGE sample labels: {len(matrix_wges):,}
- Direct GID-WGE crosswalks recovered: {decision['direct_gid_wge_crosswalk_candidates']:,}
- Identifier mismatches affecting target GIDs: {decision['identifier_mismatch_stage1_target_rows']:,}

## Missing-GID metadata classes

{count_markdown(terminal_counts, 'gids')}

## Target-run interpretations

{count_markdown(run_counts, 'runs')}

## Decision

The public SRA submission metadata does not support the hypothesis that the
missing Stage-1 GIDs were renamed to WGE identifiers. Exact run/barcode/GID
submission evidence is available for a subset and places their loss downstream
of submission. Runs without public barcode tables remain inconclusive. No
genotype assignment is authorized from metadata alone; raw demultiplexing and
sample-level QC are required for unresolved or omitted samples.
"""
    (out_dir / "IDENTIFIER_METADATA_RESOLUTION_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
