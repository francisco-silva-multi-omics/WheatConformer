#!/usr/bin/env python3
"""Exhaustively inventory coordinate/design fields in the raw trial corpus.

The scanner reads every direct tabular artifact and every worksheet, and safely
extracts tabular members from zip/7z/gzip archives into a private temporary tree.
It only inventories evidence.  It never infers a grid from plot order.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import py7zr


RELEASE_TRAIN_ID = "P4ISP_20260802_V1_274E41DF"
VERSION = "v1"
# Exhaustive means every physical row/cell in every worksheet or delimited
# member, not merely a conventional header window.
SCAN_ROWS: int | None = None
TABULAR = {".xls", ".xlsx", ".csv", ".txt", ".tab", ".tsv"}
ARCHIVES = {".7z", ".zip", ".gz"}

SEMANTICS = {
    "FIELD_ROW": {"row", "field row", "plot row", "fieldrow", "plotrow"},
    "ROW_LIKE_AMBIGUOUS": {"range", "tier", "bed"},
    "FIELD_COLUMN": {
        "column", "col", "field column", "field col", "plot column", "plot col",
        "fieldcol", "plotcol",
    },
    "COLUMN_LIKE_AMBIGUOUS": {"pass"},
    "PLOT": {"plot", "plot no", "plot number", "plotno", "plotnumber", "field plot", "fieldplot"},
    "REPLICATION": {"rep", "replication", "replicate"},
    "BLOCK": {"block", "sub block", "subblock", "blk"},
    "OCCURRENCE": {"occ", "occurrence"},
    "LOCATION": {"location", "location no", "loc", "loc no", "locationnumber", "locno"},
}


def norm(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    text = re.sub(r"[_\-./]+", " ", text.strip().lower())
    return re.sub(r"\s+", " ", text).strip()


def semantic(value: Any) -> str:
    token = norm(value)
    for label, accepted in SEMANTICS.items():
        if token in accepted:
            return label
    return ""


def candidate_rows(
    frame: pd.DataFrame, source_file: str, archive_member: str, sheet: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_number, series in frame.iterrows():
        matches: list[tuple[int, str, str]] = []
        for column_number, value in enumerate(series.tolist()):
            label = semantic(value)
            if label:
                matches.append((column_number, str(value), label))
        if not matches:
            continue
        all_headers = ["" if pd.isna(v) else str(v) for v in series.tolist()]
        labels = {item[2] for item in matches}
        header_context = bool(labels & {"PLOT", "REPLICATION", "BLOCK", "OCCURRENCE", "LOCATION"})
        for column_number, raw_header, label in matches:
            sample_values = []
            nonempty_values: list[str] = []
            if column_number < frame.shape[1]:
                below = frame.iloc[row_number + 1 :, column_number].tolist()
                nonempty_values = [str(value).strip() for value in below if not pd.isna(value) and str(value).strip()]
                for value in below[:10]:
                    if not pd.isna(value) and str(value).strip():
                        sample_values.append(str(value))
            rows.append({
                "release_train_id": RELEASE_TRAIN_ID,
                "integrated_release_version": VERSION,
                "source_file": source_file,
                "archive_member": archive_member,
                "source_sheet": sheet,
                "candidate_header_row_zero_based": int(row_number),
                "candidate_column_zero_based": int(column_number),
                "raw_column_name": raw_header,
                "normalized_column_name": norm(raw_header),
                "semantic_class": label,
                "header_row_has_design_context": header_context,
                "header_row_semantic_classes": ";".join(sorted(labels)),
                "header_row_values_json": json.dumps(all_headers, ensure_ascii=False),
                "sample_values_json": json.dumps(sample_values, ensure_ascii=False),
                "nonempty_values_below_header": len(nonempty_values),
                "distinct_nonempty_values_below_header": len(set(nonempty_values)),
                "scan_depth_rows": "ALL_ROWS" if SCAN_ROWS is None else SCAN_ROWS,
            })
    return rows


def read_delimited(path: Path) -> tuple[pd.DataFrame, str]:
    raw = path.read_bytes()
    text = None
    encoding = ""
    for enc in ("utf-8-sig", "latin-1", "utf-16"):
        try:
            text = raw.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise UnicodeError("no supported text encoding")
    lines = text.splitlines()[:SCAN_ROWS]
    if not lines:
        return pd.DataFrame(), encoding
    counts = {sep: sum(line.count(sep) for line in lines[:20]) for sep in ("\t", ",", ";", "|")}
    sep = max(counts, key=counts.get)
    parsed = list(csv.reader(lines, delimiter=sep))
    width = max((len(row) for row in parsed), default=0)
    padded = [row + [""] * (width - len(row)) for row in parsed]
    return pd.DataFrame(padded), f"{encoding};delimiter={repr(sep)}"


def scan_tabular(path: Path, logical_source: str, archive_member: str = "") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        try:
            book = pd.ExcelFile(path)
        except Exception as excel_exc:
            # The legacy corpus intentionally contains many tab-delimited files
            # with an .xls suffix.  This is the same representation recorded as
            # <tabular_text> by the Stage-1 lineage layer.
            try:
                frame, details = read_delimited(path)
                found = candidate_rows(frame, logical_source, archive_member, "<tabular_text_mislabelled_xls>")
                candidates.extend(found)
                rows_scanned, columns_scanned = frame.shape
                source_rows.append({
                    "release_train_id": RELEASE_TRAIN_ID,
                    "integrated_release_version": VERSION,
                    "source_file": logical_source,
                    "archive_member": archive_member,
                    "source_sheet": "<tabular_text_mislabelled_xls>",
                    "source_format": suffix,
                    "scan_status": "SCANNED_TABULAR_TEXT_FALLBACK",
                    "rows_scanned": rows_scanned,
                    "columns_scanned": columns_scanned,
                    "coordinate_candidate_columns": len(found),
                    "scan_error": f"Excel parser rejected legacy text representation ({type(excel_exc).__name__}); {details}",
                })
                return source_rows, candidates
            except Exception as text_exc:
                source_rows.append({
                    "release_train_id": RELEASE_TRAIN_ID,
                    "integrated_release_version": VERSION,
                    "source_file": logical_source,
                    "archive_member": archive_member,
                    "source_sheet": "",
                    "source_format": suffix,
                    "scan_status": "SCAN_ERROR",
                    "rows_scanned": 0,
                    "columns_scanned": 0,
                    "coordinate_candidate_columns": 0,
                    "scan_error": f"Excel={type(excel_exc).__name__}: {excel_exc}; text={type(text_exc).__name__}: {text_exc}",
                })
                return source_rows, candidates
        for sheet in book.sheet_names:
            status = "SCANNED"
            error = ""
            try:
                frame = pd.read_excel(book, sheet_name=sheet, header=None, nrows=SCAN_ROWS, dtype=object)
                found = candidate_rows(frame, logical_source, archive_member, str(sheet))
                candidates.extend(found)
                rows_scanned, columns_scanned = frame.shape
            except Exception as exc:  # source evidence must retain parser failures
                status = "SCAN_ERROR"
                error = f"{type(exc).__name__}: {exc}"
                rows_scanned = columns_scanned = 0
            source_rows.append({
                "release_train_id": RELEASE_TRAIN_ID,
                "integrated_release_version": VERSION,
                "source_file": logical_source,
                "archive_member": archive_member,
                "source_sheet": str(sheet),
                "source_format": suffix,
                "scan_status": status,
                "rows_scanned": rows_scanned,
                "columns_scanned": columns_scanned,
                "coordinate_candidate_columns": len([r for r in candidates if r["source_file"] == logical_source and r["archive_member"] == archive_member and r["source_sheet"] == str(sheet)]),
                "scan_error": error,
            })
    else:
        status = "SCANNED"
        error = ""
        try:
            frame, details = read_delimited(path)
            found = candidate_rows(frame, logical_source, archive_member, "<tabular_text>")
            candidates.extend(found)
            rows_scanned, columns_scanned = frame.shape
        except Exception as exc:
            status = "SCAN_ERROR"
            error = f"{type(exc).__name__}: {exc}"
            details = ""
            rows_scanned = columns_scanned = 0
        source_rows.append({
            "release_train_id": RELEASE_TRAIN_ID,
            "integrated_release_version": VERSION,
            "source_file": logical_source,
            "archive_member": archive_member,
            "source_sheet": "<tabular_text>",
            "source_format": suffix,
            "scan_status": status,
            "rows_scanned": rows_scanned,
            "columns_scanned": columns_scanned,
            "coordinate_candidate_columns": len(candidates),
            "scan_error": error or details,
        })
    return source_rows, candidates


def scan_direct(task: tuple[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    absolute, relative = task
    return scan_tabular(Path(absolute), relative)


def safe_members(base: Path) -> list[Path]:
    resolved = base.resolve()
    return [p for p in base.rglob("*") if p.is_file() and p.resolve().is_relative_to(resolved)]


def scan_archive(path: Path, relative: str, temp_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = temp_root / re.sub(r"[^A-Za-z0-9_.-]+", "_", relative)
    target.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    try:
        if path.suffix.lower() == ".7z":
            with py7zr.SevenZipFile(path, mode="r") as archive:
                archive.extractall(target)
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    member = Path(info.filename)
                    if info.is_dir() or member.is_absolute() or ".." in member.parts:
                        continue
                    archive.extract(info, target)
        elif path.suffix.lower() == ".gz":
            output_name = path.stem
            with gzip.open(path, "rb") as source, (target / output_name).open("wb") as output:
                shutil.copyfileobj(source, output)
        members = safe_members(target)
        for member_path in members:
            member_name = member_path.relative_to(target).as_posix()
            if member_path.suffix.lower() in TABULAR:
                src, cand = scan_tabular(member_path, relative, member_name)
                rows.extend(src)
                candidates.extend(cand)
            else:
                rows.append({
                    "release_train_id": RELEASE_TRAIN_ID,
                    "integrated_release_version": VERSION,
                    "source_file": relative,
                    "archive_member": member_name,
                    "source_sheet": "",
                    "source_format": member_path.suffix.lower(),
                    "scan_status": "ARCHIVE_MEMBER_NON_TABULAR_INVENTORIED",
                    "rows_scanned": 0,
                    "columns_scanned": 0,
                    "coordinate_candidate_columns": 0,
                    "scan_error": "",
                })
        if not members:
            raise ValueError("archive contained no safe file members")
    except Exception as exc:
        rows.append({
            "release_train_id": RELEASE_TRAIN_ID,
            "integrated_release_version": VERSION,
            "source_file": relative,
            "archive_member": "",
            "source_sheet": "",
            "source_format": path.suffix.lower(),
            "scan_status": "ARCHIVE_SCAN_ERROR",
            "rows_scanned": 0,
            "columns_scanned": 0,
            "coordinate_candidate_columns": 0,
            "scan_error": f"{type(exc).__name__}: {exc}",
        })
    return rows, candidates


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else ["release_train_id", "integrated_release_version"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "TRIALS_AND_NURSERIES_DATA"
    release = root / "audit" / "v2" / f"phase4_integrated_spatial_promotion_release_{VERSION}"
    manifest = json.loads((release / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["release_train_id"] == RELEASE_TRAIN_ID
    files = sorted(p for p in raw.rglob("*") if p.is_file())
    direct = [(str(p), p.relative_to(raw).as_posix()) for p in files if p.suffix.lower() in TABULAR]
    archives = [(p, p.relative_to(raw).as_posix()) for p in files if p.suffix.lower() in ARCHIVES]
    inventory: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan_direct, task): task[1] for task in direct}
        for future in as_completed(futures):
            try:
                src, cand = future.result()
            except Exception as exc:
                src = [{
                    "release_train_id": RELEASE_TRAIN_ID,
                    "integrated_release_version": VERSION,
                    "source_file": futures[future], "archive_member": "", "source_sheet": "",
                    "source_format": Path(futures[future]).suffix.lower(), "scan_status": "SCAN_ERROR",
                    "rows_scanned": 0, "columns_scanned": 0, "coordinate_candidate_columns": 0,
                    "scan_error": f"{type(exc).__name__}: {exc}",
                }]
                cand = []
            inventory.extend(src)
            candidates.extend(cand)

    # On NTFS mounted through WSL, py7zr may leave handles briefly visible after
    # close.  Keep the private scratch tree through validation instead of risking
    # a cleanup exception before inventories are written.
    temp_root = Path(tempfile.mkdtemp(prefix="phase4_coordinate_archives_", dir=release / "logs"))
    for path, relative in archives:
        src, cand = scan_archive(path, relative, temp_root)
        inventory.extend(src)
        candidates.extend(cand)

    scanned_top = {row["source_file"] for row in inventory}
    for path in files:
        relative = path.relative_to(raw).as_posix()
        if relative not in scanned_top:
            inventory.append({
                "release_train_id": RELEASE_TRAIN_ID,
                "integrated_release_version": VERSION,
                "source_file": relative,
                "archive_member": "",
                "source_sheet": "",
                "source_format": path.suffix.lower(),
                "scan_status": "NON_TABULAR_ARTIFACT_INVENTORIED",
                "rows_scanned": 0,
                "columns_scanned": 0,
                "coordinate_candidate_columns": 0,
                "scan_error": "",
            })
    inventory.sort(key=lambda r: (r["source_file"], r["archive_member"], r["source_sheet"]))
    candidates.sort(key=lambda r: (r["source_file"], r["archive_member"], r["source_sheet"], r["candidate_header_row_zero_based"], r["candidate_column_zero_based"]))
    write_tsv(release / "coordinate_source_inventory.tsv", inventory)
    write_tsv(release / "coordinate_column_candidate_inventory.tsv", candidates)

    summary = {
        "release_train_id": RELEASE_TRAIN_ID,
        "integrated_release_version": VERSION,
        "raw_top_level_artifacts": len(files),
        "raw_top_level_artifacts_accounted": len(scanned_top),
        "source_sheet_or_member_inventory_rows": len(inventory),
        "scan_error_rows": sum("ERROR" in row["scan_status"] for row in inventory),
        "candidate_column_rows": len(candidates),
        "field_row_candidate_rows": sum(row["semantic_class"] == "FIELD_ROW" for row in candidates),
        "field_column_candidate_rows": sum(row["semantic_class"] == "FIELD_COLUMN" for row in candidates),
        "candidate_rows_with_row_and_column": len({
            (r["source_file"], r["archive_member"], r["source_sheet"], r["candidate_header_row_zero_based"])
            for r in candidates if r["semantic_class"] == "FIELD_ROW"
        } & {
            (r["source_file"], r["archive_member"], r["source_sheet"], r["candidate_header_row_zero_based"])
            for r in candidates if r["semantic_class"] == "FIELD_COLUMN"
        }),
        "arbitrary_plot_reshape_performed": False,
        "component_status": "DIAGNOSTIC_SCAN_COMPLETE_REQUIRES_EVIDENCE_ADJUDICATION",
    }
    (release / "coordinate_scan_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
