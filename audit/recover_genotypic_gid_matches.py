from __future__ import annotations

import argparse
import csv
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

from genotype_recovery import (
    canonical_gid,
    load_canonical_catalog,
    load_explicit_sample_gid_mappings,
    normalize_identifier,
)


NON_DATA_SUFFIXES = {".pdf", ".md5", ".gz", ".zip", ".7z", ".ini"}
NON_DATA_NAMES = ("manifest", "readme", "protocol", "dictionary", "agreement", "md5")


def dataset_name(path: Path, genotypic_root: Path) -> str:
    return path.relative_to(genotypic_root).parts[0]


def resolve_identifier(
    identifier: str,
    *,
    explicit: dict[str, set[str]],
    aliases: dict[str, set[str]],
    canonical_ids: set[str],
) -> tuple[set[str], str]:
    text = normalize_identifier(identifier)
    direct = canonical_gid(text)
    if direct and direct in canonical_ids:
        return {direct}, "exact_canonical_gid"
    mapped = explicit.get(text.upper(), set()) & canonical_ids
    if mapped:
        return mapped, "explicit_sample_gid_sidecar"
    mapped = aliases.get(text.upper(), set()) & canonical_ids
    return mapped, "authoritative_canonical_alias"


def add_axis_identifier(
    rows: list[dict[str, object]],
    *,
    path: Path,
    genotypic_root: Path,
    identifier: str,
    source_locator: str,
    parser: str,
) -> None:
    value = normalize_identifier(identifier)
    if not value or value in {"*", "-"}:
        return
    rows.append(
        {
            "dataset": dataset_name(path, genotypic_root),
            "file_path": str(path),
            "source_locator": source_locator,
            "sample_identifier": value,
            "parser": parser,
            "matrix_backed": True,
        }
    )


def scan_preamble(path: Path, genotypic_root: Path, max_lines: int = 16) -> tuple[list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            lines = [handle.readline() for _ in range(max_lines)]
    except OSError as exc:
        return rows, f"read_error:{type(exc).__name__}"
    lines = [line for line in lines if line]
    for line_number, line in enumerate(lines, start=1):
        delimiter = "\t" if line.count("\t") >= line.count(",") else ","
        fields = line.rstrip("\r\n").split(delimiter)
        if len(fields) < 2:
            continue
        first = normalize_identifier(fields[0])
        if re.match(r"(?i)^GID(?:\s|$|\()", first):
            for value in fields[1:]:
                if canonical_gid(value):
                    add_axis_identifier(
                        rows,
                        path=path,
                        genotypic_root=genotypic_root,
                        identifier=value,
                        source_locator=f"preamble_row_{line_number}",
                        parser="explicit_gid_axis",
                    )
        for column_number, value in enumerate(fields[1:], start=2):
            if re.fullmatch(r"(?i)GID[0-9]+(?:_[0-9]+)?", normalize_identifier(value)):
                add_axis_identifier(
                    rows,
                    path=path,
                    genotypic_root=genotypic_root,
                    identifier=value.split("_", 1)[0],
                    source_locator=f"preamble_row_{line_number}_column_{column_number}",
                    parser="gid_labeled_header_axis",
                )
    return rows, "parsed_preamble"


def scan_80k_vendor_axis(path: Path, genotypic_root: Path) -> tuple[list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            preamble = []
            for line_number in range(1, 33):
                line = handle.readline()
                if not line:
                    break
                fields = line.rstrip("\r\n").split(",")
                if fields and fields[0] == "AlleleID":
                    break
                preamble.append((line_number, fields))
    except OSError as exc:
        return rows, f"read_error:{type(exc).__name__}"
    candidate_rows = []
    for line_number, fields in preamble:
        values = [normalize_identifier(value) for value in fields]
        sample_like = sum(bool(re.match(r"(?i)^SEED[A-Z0-9&_.-]+$", value)) for value in values)
        candidate_rows.append((sample_like, line_number, values))
    if not candidate_rows or max(candidate_rows)[0] == 0:
        return rows, "80k_sample_axis_not_found"
    _, line_number, values = max(candidate_rows)
    for column_number, value in enumerate(values, start=1):
        if re.match(r"(?i)^SEED[A-Z0-9&_.-]+$", value):
            add_axis_identifier(
                rows,
                path=path,
                genotypic_root=genotypic_root,
                identifier=value,
                source_locator=f"sample_metadata_row_{line_number}_column_{column_number}",
                parser="80k_vendor_sample_axis",
            )
    return rows, "parsed_80k_vendor_sample_axis"


def scan_sidecar_as_matrix_axis(path: Path, genotypic_root: Path) -> tuple[list[dict[str, object]], str]:
    separator = "\t" if path.suffix.lower() in {".txt", ".tab", ".tsv"} else ","
    frame = pd.read_csv(path, sep=separator, dtype=str, low_memory=False)
    sample_col = next((column for column in frame.columns if column.lower() == "sampleid"), None)
    gid_col = next((column for column in frame.columns if column.lower() == "gid"), None)
    rows: list[dict[str, object]] = []
    if sample_col:
        for row_number, value in enumerate(frame[sample_col], start=2):
            add_axis_identifier(
                rows,
                path=path,
                genotypic_root=genotypic_root,
                identifier=value,
                source_locator=f"sidecar_row_{row_number}",
                parser="complete_sample_gid_sidecar_axis",
            )
    elif gid_col:
        for row_number, value in enumerate(frame[gid_col], start=2):
            add_axis_identifier(
                rows,
                path=path,
                genotypic_root=genotypic_root,
                identifier=value,
                source_locator=f"sidecar_row_{row_number}",
                parser="complete_gid_sidecar_axis",
            )
    return rows, "parsed_complete_sidecar" if rows else "sidecar_without_sample_or_gid"


def scan_marker_by_sample_header(
    path: Path, genotypic_root: Path
) -> tuple[list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            header_line = handle.readline()
    except OSError as exc:
        return rows, f"read_error:{type(exc).__name__}"
    delimiter = "\t" if header_line.count("\t") >= header_line.count(",") else ","
    fields = header_line.rstrip("\r\n").split(delimiter)
    if not fields or normalize_identifier(fields[0]).lower() != "markerid":
        return rows, "marker_by_sample_header_not_found"
    for column_number, value in enumerate(fields[1:], start=2):
        add_axis_identifier(
            rows,
            path=path,
            genotypic_root=genotypic_root,
            identifier=value,
            source_locator=f"matrix_header_column_{column_number}",
            parser="marker_by_sample_header_axis",
        )
    return rows, "parsed_marker_by_sample_header_axis"


def scan_first_column_gid(path: Path, genotypic_root: Path) -> tuple[list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="," if path.suffix.lower() == ".csv" else "\t")
        header = next(reader, [])
        if not header or normalize_identifier(header[0]).upper() != "GID":
            return rows, "first_column_is_not_gid"
        for row_number, values in enumerate(reader, start=2):
            if values:
                add_axis_identifier(
                    rows,
                    path=path,
                    genotypic_root=genotypic_root,
                    identifier=values[0],
                    source_locator=f"row_{row_number}",
                    parser="complete_first_column_gid_axis",
                )
    return rows, "parsed_complete_first_column_gid_axis"


def excel_column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference.upper())
    if not letters:
        return 0
    value = 0
    for letter in letters.group(0):
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def read_xlsx_rows(path: Path) -> list[list[str]]:
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rows: list[list[str]] = []
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{namespace}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{namespace}t")))
        sheets = sorted(
            name
            for name in archive.namelist()
            if re.fullmatch(r"xl/worksheets/sheet[0-9]+\.xml", name)
        )
        for sheet in sheets:
            root = ElementTree.fromstring(archive.read(sheet))
            for row in root.iter(f"{namespace}row"):
                values: dict[int, str] = {}
                for cell in row.findall(f"{namespace}c"):
                    index = excel_column_index(cell.attrib.get("r", "A1"))
                    cell_type = cell.attrib.get("t", "")
                    value_node = cell.find(f"{namespace}v")
                    if cell_type == "inlineStr":
                        inline = cell.find(f"{namespace}is")
                        value = "" if inline is None else "".join(
                            node.text or "" for node in inline.iter(f"{namespace}t")
                        )
                    elif value_node is None:
                        value = ""
                    elif cell_type == "s":
                        shared_index = int(value_node.text or "0")
                        value = shared[shared_index] if shared_index < len(shared) else ""
                    else:
                        value = value_node.text or ""
                    values[index] = value
                if values:
                    width = max(values) + 1
                    rows.append([values.get(index, "") for index in range(width)])
    return rows


def scan_spreadsheet_axis(path: Path, genotypic_root: Path) -> tuple[list[dict[str, object]], str]:
    if path.suffix.lower() != ".xlsx":
        return [], "legacy_xls_requires_xlrd"
    try:
        sheet_rows = read_xlsx_rows(path)
    except (OSError, ValueError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        return [], f"spreadsheet_read_error:{type(exc).__name__}"
    candidate_columns: set[int] = set()
    header_tokens = ("gid", "sampleid", "sample_id", "study sample", "studysample", "cimmyt id", "doi")
    for row in sheet_rows[:100]:
        for index, value in enumerate(row):
            lower = normalize_identifier(value).lower()
            if any(token in lower for token in header_tokens):
                candidate_columns.add(index)
    rows: list[dict[str, object]] = []
    seen: set[tuple[int, int, str]] = set()
    for row_number, values in enumerate(sheet_rows, start=1):
        for column_number, value in enumerate(values, start=1):
            text = normalize_identifier(value)
            if not text:
                continue
            recognizable = bool(
                re.fullmatch(r"(?i)GID[0-9]+", text)
                or re.match(r"(?i)^10\.18730/", text)
                or re.match(r"(?i)^SEED[A-Z0-9&_.-]+$", text)
                or (column_number - 1 in candidate_columns and len(text) <= 200)
            )
            key = (row_number, column_number, text)
            if recognizable and key not in seen:
                seen.add(key)
                add_axis_identifier(
                    rows,
                    path=path,
                    genotypic_root=genotypic_root,
                    identifier=text,
                    source_locator=f"xlsx_row_{row_number}_column_{column_number}",
                    parser="xlsx_identifier_axis",
                )
    return rows, "parsed_xlsx_identifier_axes"


def discover_matrix_axes(genotypic_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    axis_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for path in sorted(item for item in genotypic_root.rglob("*") if item.is_file()):
        relative = path.relative_to(genotypic_root)
        lower = path.name.lower()
        rows: list[dict[str, object]] = []
        status = "not_a_sample_axis_file"
        if path.suffix.lower() in NON_DATA_SUFFIXES or any(token in lower for token in NON_DATA_NAMES):
            status = "skipped_documentation_or_archive"
        elif relative.parts[0] == "80k" and path.suffix.lower() == ".csv":
            rows, status = scan_80k_vendor_axis(path, genotypic_root)
        elif path.name in {
            "SEQ_SNPs_Extract_45610samples_102474markers.txt",
            "SEQ_SNPs_Extract_mexican_8584samples_102474markers.txt",
        }:
            rows, status = scan_marker_by_sample_header(path, genotypic_root)
        elif path.name in {"SampleIDvsGID_45610samples.txt", "Mexican_landrace_samples_for_Germinate.txt"}:
            status = "parsed_explicit_mapping_sidecar"
        elif path.name == "Haplotype_blocks_EYT2011-12_to_EYT2017-18.csv":
            rows, status = scan_first_column_gid(path, genotypic_root)
        elif path.suffix.lower() == ".flapjack":
            try:
                with path.open("rb") as handle:
                    signature = handle.read(16)
            except OSError as exc:
                status = f"read_error:{type(exc).__name__}"
            else:
                status = (
                    "sqlite_flapjack_project_container"
                    if signature.startswith(b"SQLite format 3")
                    else "unrecognized_flapjack_container"
                )
        elif path.suffix.lower() in {".csv", ".txt", ".tab", ".tsv", ".hmp"}:
            rows, status = scan_preamble(path, genotypic_root)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            rows, status = scan_spreadsheet_axis(path, genotypic_root)
        axis_rows.extend(rows)
        status_rows.append(
            {
                "dataset": dataset_name(path, genotypic_root),
                "file_path": str(path),
                "bytes": path.stat().st_size,
                "parser_status": status,
                "matrix_axis_identifiers": len(rows),
            }
        )
    axis = pd.DataFrame(axis_rows).drop_duplicates(
        ["dataset", "file_path", "sample_identifier", "source_locator"]
    )
    return axis, pd.DataFrame(status_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustively recover matrix-backed canonical GIDs.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--genotypic-dir", type=Path, default=Path("GENOTYPIC_DATA"))
    parser.add_argument(
        "--canonical-catalog",
        type=Path,
        default=Path("audit/canonical_genotype_mapping_audited.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("audit/genotypic_recovery"))
    args = parser.parse_args()

    root = args.root.resolve()
    genotypic_root = (root / args.genotypic_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog, aliases = load_canonical_catalog((root / args.canonical_catalog).resolve())
    canonical_ids = set(catalog["canonical_gid"])
    explicit, mapping_evidence = load_explicit_sample_gid_mappings(genotypic_root)
    matrix_axis, file_status = discover_matrix_axes(genotypic_root)

    matches: list[dict[str, object]] = []
    ambiguous: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []
    for row in matrix_axis.itertuples(index=False):
        gids, method = resolve_identifier(
            row.sample_identifier,
            explicit=explicit,
            aliases=aliases,
            canonical_ids=canonical_ids,
        )
        base = row._asdict()
        base["resolution_method"] = method
        if len(gids) == 1:
            base["canonical_gid"] = next(iter(gids))
            matches.append(base)
        elif len(gids) > 1:
            base["candidate_canonical_gids"] = ";".join(sorted(gids))
            ambiguous.append(base)
        else:
            unmatched.append(base)

    match_frame = pd.DataFrame(matches)
    ambiguous_frame = pd.DataFrame(ambiguous)
    unmatched_frame = pd.DataFrame(unmatched)
    if not match_frame.empty:
        match_frame = match_frame.drop_duplicates(
            ["dataset", "file_path", "sample_identifier", "canonical_gid"]
        )
    flags = catalog.set_index("canonical_gid")
    candidates = (
        match_frame.groupby(["dataset", "canonical_gid"], as_index=False)
        .agg(
            matrix_files=("file_path", "nunique"),
            matrix_sample_identifiers=("sample_identifier", "nunique"),
            resolution_methods=("resolution_method", lambda x: ";".join(sorted(set(x)))),
        )
        if not match_frame.empty
        else pd.DataFrame(columns=["dataset", "canonical_gid"])
    )
    for column in [
        "marker_available_hmp_qc",
        "pedigree_available",
        "canonical_observation_rows",
        "audit_genotypic_match",
    ]:
        if column in flags.columns and not candidates.empty:
            candidates[column] = candidates["canonical_gid"].map(flags[column])

    summary_rows = []
    datasets = sorted(set(file_status["dataset"]))
    for dataset in datasets:
        subset = candidates[candidates["dataset"].eq(dataset)] if not candidates.empty else candidates
        status_subset = file_status[file_status["dataset"].eq(dataset)]
        hmp = subset.get("marker_available_hmp_qc", pd.Series(dtype=str)).astype(str).str.lower().eq("true")
        preview = subset.get("audit_genotypic_match", pd.Series(dtype=str)).astype(str).str.lower().eq("true")
        summary_rows.append(
            {
                "dataset": dataset,
                "files_inventoried": len(status_subset),
                "files_with_matrix_axis_identifiers": int(status_subset["matrix_axis_identifiers"].astype(int).gt(0).sum()),
                "matrix_axis_identifiers": int(matrix_axis["dataset"].eq(dataset).sum()),
                "matched_unique_canonical_gids": int(subset["canonical_gid"].nunique()) if len(subset) else 0,
                "additional_to_hmp_qc": int((~hmp).sum()) if len(subset) else 0,
                "missed_by_preview_audit": int((~preview).sum()) if len(subset) else 0,
                "ambiguous_identifiers": int(ambiguous_frame["dataset"].eq(dataset).sum()) if len(ambiguous_frame) else 0,
                "unmatched_identifiers": int(unmatched_frame["dataset"].eq(dataset).sum()) if len(unmatched_frame) else 0,
            }
        )
    summary = pd.DataFrame(summary_rows)

    matrix_axis.to_csv(out_dir / "matrix_sample_axis_catalog.tsv.gz", sep="\t", index=False, compression="gzip")
    mapping_evidence.to_csv(out_dir / "explicit_sample_gid_mappings.tsv.gz", sep="\t", index=False, compression="gzip")
    match_frame.to_csv(out_dir / "matrix_backed_gid_match_evidence.tsv.gz", sep="\t", index=False, compression="gzip")
    ambiguous_frame.to_csv(out_dir / "matrix_backed_gid_ambiguous.tsv", sep="\t", index=False)
    unmatched_frame.to_csv(out_dir / "matrix_sample_identifiers_unmatched.tsv.gz", sep="\t", index=False, compression="gzip")
    candidates.to_csv(out_dir / "matrix_backed_gid_candidates.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "matrix_backed_gid_dataset_summary.tsv", sep="\t", index=False)
    file_status.to_csv(out_dir / "genotypic_file_parser_status.tsv", sep="\t", index=False)

    union = candidates.drop_duplicates("canonical_gid") if len(candidates) else candidates
    marker_flag = union.get("marker_available_hmp_qc", pd.Series(dtype=str)).astype(str).str.lower().eq("true")
    preview_flag = union.get("audit_genotypic_match", pd.Series(dtype=str)).astype(str).str.lower().eq("true")
    observation_rows = pd.to_numeric(
        union.get("canonical_observation_rows", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    union_summary = pd.DataFrame(
        [
            {"metric": "genotypic_files_inventoried", "value": len(file_status)},
            {"metric": "files_with_recognized_matrix_axis", "value": int(file_status["matrix_axis_identifiers"].astype(int).gt(0).sum())},
            {"metric": "files_skipped_documentation_or_archive", "value": int(file_status["parser_status"].eq("skipped_documentation_or_archive").sum())},
            {"metric": "sqlite_flapjack_project_containers", "value": int(file_status["parser_status"].eq("sqlite_flapjack_project_container").sum())},
            {"metric": "explicit_mapping_sidecars", "value": int(file_status["parser_status"].eq("parsed_explicit_mapping_sidecar").sum())},
            {"metric": "parser_errors_or_unrecognized_containers", "value": int(file_status["parser_status"].str.contains(r"read_error|unrecognized|requires_xlrd", regex=True).sum())},
            {"metric": "matrix_backed_unique_canonical_gids", "value": len(union)},
            {"metric": "matrix_backed_additional_to_hmp_qc", "value": int((~marker_flag).sum())},
            {"metric": "matrix_backed_missed_by_preview_audit", "value": int((~preview_flag).sum())},
            {"metric": "matrix_backed_missed_by_preview_and_hmp", "value": int((~preview_flag & ~marker_flag).sum())},
            {"metric": "potential_canonical_observation_rows_additional_to_hmp_qc", "value": int(observation_rows[~marker_flag].sum())},
            {"metric": "ambiguous_matrix_identifiers", "value": len(ambiguous_frame)},
            {"metric": "files_without_recognized_matrix_axis", "value": int(file_status["matrix_axis_identifiers"].astype(int).eq(0).sum())},
        ]
    )
    union_summary.to_csv(out_dir / "matrix_backed_gid_union_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print("\n", union_summary.to_string(index=False))


if __name__ == "__main__":
    main()
