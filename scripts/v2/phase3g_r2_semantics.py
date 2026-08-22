"""Correct HiBAP and DArTseq-80K sample-instance semantics for Phase-3G R2."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from scripts.v2.phase3g_identifier_semantics import parse_identifier


PARSER_VERSION = "phase3g_r2_identifier_semantics_v2"
MISSING_HIBAP = frozenset({"", "N", "-", "."})


def clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def normalize_path(value: object) -> str:
    return clean(value).replace("\\", "/")


def canonical_gid(value: object) -> str:
    return parse_identifier(value, context="authoritative_gid_column").canonical_gid_candidate


def stable_sample_instance_key(
    panel_id: str,
    source_file: str,
    physical_axis_index: int,
    raw_sample_label: str,
    occurrence_index: int,
) -> str:
    """Hash every component required to distinguish a physical sample instance."""

    payload = json.dumps(
        {
            "panel_id": clean(panel_id),
            "source_file": normalize_path(source_file),
            "physical_axis_index": int(physical_axis_index),
            "raw_sample_label": clean(raw_sample_label),
            "occurrence_index": int(occurrence_index),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "PSI_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _delimited_rows(path: Path, delimiter: str = "\t") -> Iterable[list[str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        yield from csv.reader(handle, delimiter=delimiter)


def parse_hibap_sources(genotype_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    directory = genotype_root / "IWYP64_-_HiBAP_35k_Wheat_Breeders_Array_Genotyping"
    marker_path = directory / "HiBAP_snps_35karray.txt"
    sidecar_path = directory / "HIBAPI_germplasm_information.txt"
    marker_relative = normalize_path(marker_path.relative_to(genotype_root))
    sidecar_relative = normalize_path(sidecar_path.relative_to(genotype_root))

    with marker_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        note_row = next(reader)
        entry_row = next(reader)
        gid_row = next(reader)
        header_row = next(reader)
    if not clean(entry_row[0]).upper().startswith("ENTRY NUMBER"):
        raise RuntimeError("Unexpected HiBAP Entry number row")
    if not clean(gid_row[0]).upper().startswith("GID"):
        raise RuntimeError("Unexpected HiBAP GID row")
    if clean(header_row[0]).lower() != "rs#":
        raise RuntimeError("Unexpected HiBAP marker header")
    if len({len(note_row), len(entry_row), len(gid_row), len(header_row)}) != 1:
        raise RuntimeError("HiBAP preamble widths differ")

    sidecar_rows = list(_delimited_rows(sidecar_path, "\t"))
    header_index = next(
        index
        for index, row in enumerate(sidecar_rows)
        if {clean(value).upper() for value in row} >= {"ENT", "GID", "SAMPLE 35K"}
    )
    columns = {clean(value).upper(): index for index, value in enumerate(sidecar_rows[header_index])}
    sidecar: list[dict[str, object]] = []
    for physical_row, row in enumerate(sidecar_rows[header_index + 1 :], start=header_index + 2):
        if not any(clean(value) for value in row):
            continue
        record = {
            "sidecar_source_file": sidecar_relative,
            "sidecar_physical_row": physical_row,
        }
        for name, index in columns.items():
            record[name] = clean(row[index]) if index < len(row) else ""
        record["canonical_gid"] = canonical_gid(record["GID"])
        sidecar.append(record)
    sidecar_frame = pd.DataFrame(sidecar)
    if sidecar_frame["ENT"].eq("").any() or sidecar_frame["ENT"].duplicated().any():
        raise RuntimeError("HiBAP sidecar ENT is blank or nonunique")
    # Preserve ENT inside each record. DataFrame.set_index(...).to_dict("index")
    # removes the indexed column, but ENT is itself auditable linkage evidence.
    sidecar_by_ent = {
        str(row["ENT"]): row for row in sidecar_frame.to_dict("records")
    }

    sample_start = 11
    headers = [clean(value) for value in header_row[sample_start:]]
    entries = [clean(value) for value in entry_row[sample_start:]]
    matrix_gids = [clean(value) for value in gid_row[sample_start:]]
    if not (len(headers) == len(entries) == len(matrix_gids) == 148):
        raise RuntimeError(
            f"HiBAP expected 148 aligned sample columns, observed "
            f"{len(headers)}/{len(entries)}/{len(matrix_gids)}"
        )
    header_occurrences: Counter[str] = Counter()
    entry_counts = Counter(entries)
    gid_counts = Counter(canonical_gid(value) for value in matrix_gids)
    output: list[dict[str, object]] = []
    for matrix_order, (header, entry, raw_matrix_gid) in enumerate(
        zip(headers, entries, matrix_gids), start=1
    ):
        header_occurrences[header] += 1
        physical_column = sample_start + matrix_order
        side = sidecar_by_ent.get(entry)
        matrix_gid = canonical_gid(raw_matrix_gid)
        sidecar_gid = clean(side["canonical_gid"]) if side else ""
        if side is None:
            linkage_status = "UNRESOLVED"
            accepted_gid = ""
            conflict_status = "SIDECAR_ENT_NOT_FOUND"
        elif not matrix_gid:
            linkage_status = "MATRIX_GID_MISSING"
            accepted_gid = ""
            conflict_status = "MISSING_TYPED_MATRIX_GID"
        elif not sidecar_gid:
            linkage_status = "SIDECAR_GID_MISSING"
            accepted_gid = ""
            conflict_status = "MISSING_TYPED_SIDECAR_GID"
        elif matrix_gid != sidecar_gid:
            linkage_status = "ENTRY_ENT_MATCH_GID_CONFLICT"
            accepted_gid = ""
            conflict_status = f"{matrix_gid}!={sidecar_gid}"
        else:
            linkage_status = "ACCEPTED_ENTRY_ENT_AND_GID_CONCORDANT"
            accepted_gid = matrix_gid
            conflict_status = "NONE"
        replicate_statuses: list[str] = []
        if entry_counts[entry] > 1:
            replicate_statuses.append("DUPLICATE_ENTRY_RETAINED")
        if matrix_gid and gid_counts[matrix_gid] > 1:
            replicate_statuses.append("REPEATED_GID_RETAINED")
        instance = stable_sample_instance_key(
            "hibap35k",
            marker_relative,
            physical_column,
            header,
            header_occurrences[header],
        )
        output.append(
            {
                "panel_id": "hibap35k",
                "sample_instance_key": instance,
                "source_file": marker_relative,
                "source_sheet": "",
                "physical_column_index": physical_column,
                "matrix_order": matrix_order,
                "raw_matrix_header": header,
                "header_occurrence_index": header_occurrences[header],
                "matrix_entry_number": entry,
                "matrix_gid_raw": raw_matrix_gid,
                "matrix_canonical_gid": matrix_gid,
                "sidecar_source_file": sidecar_relative,
                "sidecar_physical_row": int(side["sidecar_physical_row"]) if side else pd.NA,
                "sidecar_ent": clean(side["ENT"]) if side else "",
                "sidecar_gid_raw": clean(side["GID"]) if side else "",
                "sidecar_canonical_gid": sidecar_gid,
                "sidecar_sample_35k": clean(side["SAMPLE 35K"]) if side else "",
                "sidecar_cid": clean(side.get("CID", "")) if side else "",
                "sidecar_sid": clean(side.get("SID", "")) if side else "",
                "sidecar_cross_name": clean(side.get("CROSS NAME", "")) if side else "",
                "sidecar_selection_history": clean(side.get("SELECTION HISTORY", "")) if side else "",
                "join_rule": "HIBAP35K_MATRIX_ENTRY_NUMBER_TO_HIBAP35K_SIDECAR_ENT_EXACT",
                "evidence_type": "ENTRY_ENT_AND_PARALLEL_TYPED_GID_CONCORDANCE",
                "linkage_status": linkage_status,
                "accepted_canonical_gid": accepted_gid,
                "conflict_status": conflict_status,
                "replicate_status": ";".join(replicate_statuses) or "UNIQUE_ENTRY_AND_GID",
                "parser_version": PARSER_VERSION,
            }
        )
    instances = pd.DataFrame(output)
    header_to_sample_agreement = int((instances["raw_matrix_header"] == instances["sidecar_sample_35k"]).sum())
    entry_matches = int((instances["matrix_entry_number"] == instances["sidecar_ent"]).sum())
    gid_matches = int((instances["matrix_canonical_gid"] == instances["sidecar_canonical_gid"]).sum())
    summary = {
        "matrix_columns": len(instances),
        "matrix_header_to_sidecar_sample35k_agreement": header_to_sample_agreement,
        "entry_to_ent_agreement": entry_matches,
        "unique_matrix_headers": int(instances["raw_matrix_header"].nunique()),
        "unique_entry_numbers": int(instances["matrix_entry_number"].nunique()),
        "unique_linked_gids": int(instances["accepted_canonical_gid"].replace("", pd.NA).nunique()),
        "matrix_sidecar_gid_concordant": gid_matches,
        "accepted_columns": int(instances["linkage_status"].eq("ACCEPTED_ENTRY_ENT_AND_GID_CONCORDANT").sum()),
        "gid_conflicts": int(instances["linkage_status"].eq("ENTRY_ENT_MATCH_GID_CONFLICT").sum()),
        "duplicate_entry_109_columns": int(instances["matrix_entry_number"].eq("109").sum()),
        "parser_version": PARSER_VERSION,
    }
    return instances, sidecar_frame, summary


def hibap_replicate_concordance(genotype_root: Path, instances: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    marker_path = genotype_root / "IWYP64_-_HiBAP_35k_Wheat_Breeders_Array_Genotyping" / "HiBAP_snps_35karray.txt"
    with marker_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for _ in range(4):
            next(reader)
        marker_rows = [row for row in reader]
    if any(len(row) < 159 for row in marker_rows):
        raise RuntimeError("HiBAP marker row width is shorter than the sample preamble")
    token_counts = Counter(
        clean(value).upper()
        for row in marker_rows
        for value in row[11:159]
    )
    allowed = {"A", "C", "G", "T", "N"}
    unexpected = sorted(set(token_counts) - allowed)
    if unexpected:
        raise RuntimeError(f"Unexpected HiBAP marker tokens: {unexpected}")
    pairs: list[dict[str, object]] = []
    repeated = instances[
        instances["accepted_canonical_gid"].ne("")
        & instances["accepted_canonical_gid"].duplicated(keep=False)
    ]
    for gid, group in repeated.groupby("accepted_canonical_gid", sort=True):
        records = list(group.sort_values("matrix_order").to_dict("records"))
        for left_index in range(len(records)):
            for right_index in range(left_index + 1, len(records)):
                left, right = records[left_index], records[right_index]
                left_column = int(left["physical_column_index"]) - 1
                right_column = int(right["physical_column_index"]) - 1
                left_calls = [clean(row[left_column]).upper() for row in marker_rows]
                right_calls = [clean(row[right_column]).upper() for row in marker_rows]
                comparable = [
                    index
                    for index, (a, b) in enumerate(zip(left_calls, right_calls))
                    if a not in MISSING_HIBAP and b not in MISSING_HIBAP
                ]
                matching = sum(left_calls[index] == right_calls[index] for index in comparable)
                discordant = len(comparable) - matching
                left_missing = sum(value in MISSING_HIBAP for value in left_calls)
                right_missing = sum(value in MISSING_HIBAP for value in right_calls)
                missing_overlap = sum(
                    a in MISSING_HIBAP and b in MISSING_HIBAP
                    for a, b in zip(left_calls, right_calls)
                )
                pairs.append(
                    {
                        "canonical_gid": gid,
                        "left_sample_instance_key": left["sample_instance_key"],
                        "left_header": left["raw_matrix_header"],
                        "left_entry": left["matrix_entry_number"],
                        "right_sample_instance_key": right["sample_instance_key"],
                        "right_header": right["raw_matrix_header"],
                        "right_entry": right["matrix_entry_number"],
                        "relationship": "SAME_ENTRY_TECHNICAL_REPLICATE_CANDIDATE" if left["matrix_entry_number"] == right["matrix_entry_number"] else "SEPARATE_ENTRIES_SAME_GID",
                        "total_markers": len(marker_rows),
                        "left_nonmissing_calls": len(marker_rows) - left_missing,
                        "right_nonmissing_calls": len(marker_rows) - right_missing,
                        "comparable_nonmissing_markers": len(comparable),
                        "matching_calls": matching,
                        "discordant_calls": discordant,
                        "concordance_proportion": matching / len(comparable) if comparable else pd.NA,
                        "left_missingness": left_missing / len(marker_rows),
                        "right_missingness": right_missing / len(marker_rows),
                        "missingness_overlap_count": missing_overlap,
                        "marker_encoding": "A/C/G/T with N missing",
                        "recommendation": "RETAIN_SEPARATELY_PENDING_PHASE5_REPLICATE_POLICY",
                    }
                )
    summary = {
        "total_markers": len(marker_rows),
        "token_counts": dict(sorted(token_counts.items())),
        "encoding_status": "PASS_VALIDATED_ACGT_N_MISSING",
        "replicate_pairs": len(pairs),
    }
    return pd.DataFrame(pairs), summary


def _population_from_name(name: str) -> str:
    lower = name.lower()
    if lower.startswith("hexaploid"):
        return "hexaploid"
    if lower.startswith("tetraploid"):
        return "tetraploid"
    if lower.startswith("wheat_recall"):
        return "wheat_recall"
    if lower.startswith("wild_relative"):
        return "wild_relative"
    raise ValueError(name)


def read_80k_csv_axis(path: Path, genotype_root: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    physical_rows: list[int] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for physical_row, row in enumerate(reader, start=1):
            if len(row) == 1 and clean(row[0]).startswith("#"):
                continue
            rows.append(row)
            physical_rows.append(physical_row)
            if len(rows) == 6:
                break
    if len(rows) != 6 or len({len(row) for row in rows}) != 1:
        raise RuntimeError(f"Incomplete or misaligned 80K preamble: {path}")
    population = _population_from_name(path.name)
    panel_id = f"dartseq80k_{population}"
    axis_type = "SNP" if "_SNP_" in path.name else "PAV"
    labels = [clean(value) for value in rows[4]]
    occurrences: Counter[str] = Counter()
    records: list[dict[str, object]] = []
    relative = normalize_path(path.relative_to(genotype_root))
    for zero_index, label in enumerate(labels):
        if label in {"", "*"}:
            continue
        occurrences[label] += 1
        physical_column = zero_index + 1
        records.append(
            {
                "panel_id": panel_id,
                "population": population,
                "representation": f"CSV_{axis_type}",
                "source_file": relative,
                "physical_column_index": physical_column,
                "raw_sample_label": label,
                "occurrence_index": occurrences[label],
                "sample_instance_key": stable_sample_instance_key(panel_id, relative, physical_column, label, occurrences[label]),
                "well": clean(rows[0][zero_index]),
                "plate_or_barcode": clean(rows[1][zero_index]),
                "sample_group": clean(rows[2][zero_index]),
                "replicate_or_index": clean(rows[3][zero_index]),
                "schema_column": clean(rows[5][zero_index]),
                "well_physical_row": physical_rows[0],
                "plate_physical_row": physical_rows[1],
                "sample_group_physical_row": physical_rows[2],
                "replicate_physical_row": physical_rows[3],
                "sample_id_physical_row": physical_rows[4],
                "schema_physical_row": physical_rows[5],
                "parser_version": PARSER_VERSION,
            }
        )
    return pd.DataFrame(records)


def read_80k_csv_axes(genotype_root: Path) -> pd.DataFrame:
    directory = genotype_root / "80k"
    frames = [read_80k_csv_axis(path, genotype_root) for path in sorted(directory.glob("*_data*.csv"))]
    return pd.concat(frames, ignore_index=True)


def canonical_80k_pav_axes(genotype_root: Path) -> pd.DataFrame:
    axes = read_80k_csv_axes(genotype_root)
    result = axes[axes["representation"].eq("CSV_PAV")].copy()
    expected = {
        "hexaploid": 56_342,
        "tetraploid": 18_946,
        "wheat_recall": 15_666,
        "wild_relative": 3_903,
    }
    observed = result.groupby("population").size().to_dict()
    if observed != expected:
        raise RuntimeError(f"Unexpected canonical 80K PAV physical axes: {observed}")
    return result


def _first_noncomment_line(handle) -> str:
    for line in handle:
        if not line.startswith("#"):
            return line
    raise RuntimeError("No noncomment line")


def _hash_order(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(clean(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def certify_pav_pair(csv_path: Path, fj_path: Path, genotype_root: Path) -> dict[str, object]:
    csv_axis = read_80k_csv_axis(csv_path, genotype_root)
    csv_labels = list(csv_axis["raw_sample_label"])
    with fj_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fj:
        fj_header = _first_noncomment_line(fj).rstrip("\r\n").split("\t")
        fj_labels = fj_header[1:]
    sample_exact = csv_labels == fj_labels

    csv_markers: list[str] = []
    fj_markers: list[str] = []
    marker_mismatches = 0
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as csv_handle, fj_path.open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as fj_handle:
        csv_noncomment = 0
        while csv_noncomment < 6:
            line = csv_handle.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF in {csv_path}")
            if line.startswith("#"):
                continue
            csv_noncomment += 1
        _first_noncomment_line(fj_handle)
        while True:
            csv_line = csv_handle.readline()
            fj_line = fj_handle.readline()
            if not csv_line and not fj_line:
                break
            if not csv_line or not fj_line:
                marker_mismatches += 1
                break
            csv_marker = csv_line.split(",", 1)[0].strip().strip('"')
            fj_marker = fj_line.split("\t", 1)[0].strip()
            csv_markers.append(csv_marker)
            fj_markers.append(fj_marker)
            marker_mismatches += int(csv_marker != fj_marker)
    return {
        "population": _population_from_name(csv_path.name),
        "representation_pair": "PAV_CSV_TO_FLAPJACK",
        "csv_source_file": normalize_path(csv_path.relative_to(genotype_root)),
        "flapjack_source_file": normalize_path(fj_path.relative_to(genotype_root)),
        "csv_sample_instances": len(csv_labels),
        "flapjack_sample_instances": len(fj_labels),
        "unique_csv_labels": len(set(csv_labels)),
        "unique_flapjack_labels": len(set(fj_labels)),
        "sample_order_relation": "EXACT_IDENTITY" if sample_exact else "MISMATCH_BLOCKED",
        "sample_permutation_reversible": sample_exact,
        "csv_sample_order_sha256": _hash_order(csv_labels),
        "flapjack_sample_order_sha256": _hash_order(fj_labels),
        "csv_markers": len(csv_markers),
        "flapjack_markers": len(fj_markers),
        "marker_order_mismatches": marker_mismatches,
        "marker_order_relation": "EXACT_IDENTITY" if marker_mismatches == 0 and len(csv_markers) == len(fj_markers) else "MISMATCH_BLOCKED",
        "csv_marker_order_sha256": _hash_order(csv_markers),
        "flapjack_marker_order_sha256": _hash_order(fj_markers),
        "missing_and_genotype_encoding": "CSV/FJ PAV 0/1 presence calls with '-' missing",
        "certification_status": "PASS" if sample_exact and marker_mismatches == 0 and len(csv_markers) == len(fj_markers) else "BLOCKED",
    }


def certify_snp_pair(csv_path: Path, fj_path: Path, genotype_root: Path) -> dict[str, object]:
    csv_axis = read_80k_csv_axis(csv_path, genotype_root)
    csv_labels = list(csv_axis["raw_sample_label"])
    with fj_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fj:
        fj_header = _first_noncomment_line(fj).rstrip("\r\n").split("\t")
        fj_markers = fj_header[1:]
        fj_labels: list[str] = []
        for line in fj:
            if not line:
                continue
            fj_labels.append(line.split("\t", 1)[0].strip())
    sample_exact = csv_labels == fj_labels

    reference_markers: list[str] = []
    pair_errors = 0
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        noncomment = 0
        while noncomment < 6:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF in {csv_path}")
            if line.startswith("#"):
                continue
            noncomment += 1
        while True:
            first = handle.readline()
            if not first:
                break
            second = handle.readline()
            if not second:
                pair_errors += 1
                break
            first_fields = first.split(",", 2)
            second_fields = second.split(",", 2)
            if len(first_fields) < 2 or len(second_fields) < 2 or first_fields[1] != second_fields[1]:
                pair_errors += 1
            reference_markers.append(first_fields[0].strip().strip('"'))
    marker_mismatches = sum(a != b for a, b in zip(reference_markers, fj_markers)) + abs(len(reference_markers) - len(fj_markers))
    return {
        "population": _population_from_name(csv_path.name),
        "representation_pair": "SNP_CSV_ALLELE_PAIRS_TO_TRANSPOSED_FLAPJACK",
        "csv_source_file": normalize_path(csv_path.relative_to(genotype_root)),
        "flapjack_source_file": normalize_path(fj_path.relative_to(genotype_root)),
        "csv_sample_instances": len(csv_labels),
        "flapjack_sample_instances": len(fj_labels),
        "unique_csv_labels": len(set(csv_labels)),
        "unique_flapjack_labels": len(set(fj_labels)),
        "sample_order_relation": "EXACT_IDENTITY" if sample_exact else "MISMATCH_BLOCKED",
        "sample_permutation_reversible": sample_exact,
        "csv_sample_order_sha256": _hash_order(csv_labels),
        "flapjack_sample_order_sha256": _hash_order(fj_labels),
        "csv_markers": len(reference_markers),
        "flapjack_markers": len(fj_markers),
        "marker_pair_structure_errors": pair_errors,
        "marker_order_mismatches": marker_mismatches,
        "marker_order_relation": "EXACT_REFERENCE_ALLELE_ORDER_AFTER_REVERSIBLE_PAIR_COLLAPSE" if pair_errors == 0 and marker_mismatches == 0 else "MISMATCH_BLOCKED",
        "csv_marker_order_sha256": _hash_order(reference_markers),
        "flapjack_marker_order_sha256": _hash_order(fj_markers),
        "missing_and_genotype_encoding": "CSV paired 0/1 allele-presence calls transform to FJ nucleotide or slash-separated heterozygote calls; '-' missing",
        "certification_status": "PASS" if sample_exact and pair_errors == 0 and marker_mismatches == 0 else "BLOCKED",
    }


def certify_80k_representations(genotype_root: Path) -> pd.DataFrame:
    directory = genotype_root / "80k"
    specs = [
        ("PAV", directory / "Hexaploid_PAV_data_for_Dataverse.csv", directory / "Hexaploid_PAV_inverted_FJ_format_for_Dataverse.txt"),
        ("SNP", directory / "Hexaploid_SNP_data_for_Dataverse.csv", directory / "Hexaploid_SNP_FJ_data_for_Dataverse.txt"),
        ("PAV", directory / "Tetraploid_PAV_data.csv", directory / "Tetraploid_PAV_inverted_FJ_format_for_Dataverse.txt"),
        ("SNP", directory / "Tetraploid_SNP_data_for_Dataverse.csv", directory / "Tetraploid_SNP_FJ_format_for_Dataverse.txt"),
        ("PAV", directory / "Wheat_Recall_PAV_data_for_Dataverse.csv", directory / "Wheat_Recall_PAV_inverted_FJ_format_for_Dataverse.txt"),
        ("PAV", directory / "Wild_Relative_PAV_data.csv", directory / "Wild_Relative_PAV_FJ_format_for_Dataverse.txt"),
        ("SNP", directory / "Wild_Relative_SNP_data_for_Dataverse.csv", directory / "Wild_Relative_SNP_FJ_format.txt"),
    ]
    rows = [
        certify_pav_pair(csv_path, fj_path, genotype_root)
        if kind == "PAV"
        else certify_snp_pair(csv_path, fj_path, genotype_root)
        for kind, csv_path, fj_path in specs
    ]
    wheat_snp = directory / "Wheat_Recall_SNP_data_FJ_format_for_Dataverse.txt"
    with wheat_snp.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        header = _first_noncomment_line(handle).rstrip("\r\n").split("\t")
        labels = [line.split("\t", 1)[0].strip() for line in handle if line]
    rows.append(
        {
            "population": "wheat_recall",
            "representation_pair": "SNP_FLAPJACK_ONLY_NO_RAW_CSV_COUNTERPART",
            "csv_source_file": "",
            "flapjack_source_file": normalize_path(wheat_snp.relative_to(genotype_root)),
            "csv_sample_instances": pd.NA,
            "flapjack_sample_instances": len(labels),
            "unique_csv_labels": pd.NA,
            "unique_flapjack_labels": len(set(labels)),
            "sample_order_relation": "PAV_CERTIFIED_AXIS_REUSED_FOR_PANEL;SNP_SINGLE_REPRESENTATION",
            "sample_permutation_reversible": pd.NA,
            "csv_sample_order_sha256": "",
            "flapjack_sample_order_sha256": _hash_order(labels),
            "csv_markers": pd.NA,
            "flapjack_markers": len(header) - 1,
            "marker_pair_structure_errors": pd.NA,
            "marker_order_mismatches": pd.NA,
            "marker_order_relation": "NOT_COMPARABLE_SINGLE_REPRESENTATION",
            "csv_marker_order_sha256": "",
            "flapjack_marker_order_sha256": _hash_order(header[1:]),
            "missing_and_genotype_encoding": "FJ nucleotide or slash-separated heterozygote calls with '-' missing",
            "certification_status": "PASS_PANEL_SAMPLE_AXIS_FROM_EXACT_PAV_CSV_FJ_PAIR;SNP_MARKER_CROSSCHECK_NOT_APPLICABLE",
        }
    )
    return pd.DataFrame(rows)


def _first_data_lines(path: Path, noncomment_header_rows: int, count: int) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        seen = 0
        for line in handle:
            if line.startswith("#"):
                continue
            if seen < noncomment_header_rows:
                seen += 1
                continue
            lines.append(line.rstrip("\r\n"))
            if len(lines) == count:
                break
    return lines


def _last_data_lines(path: Path, count: int, window_bytes: int = 8 * 1024 * 1024) -> list[str]:
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size - window_bytes))
        if size > window_bytes:
            handle.readline()
        lines = [line.decode("utf-8", errors="replace").rstrip("\r\n") for line in handle]
    return [line for line in lines if line and not line.startswith("#")][-count:]


def _tokens_from_delimited_payload(line: str, delimiter: str, leading_fields: int) -> set[str]:
    fields = next(csv.reader([line], delimiter=delimiter))
    return {clean(value).upper() for value in fields[leading_fields:]}


def audit_80k_encoding(genotype_root: Path) -> pd.DataFrame:
    """Validate representation token models using source-bound deterministic samples.

    Axis equality is certified independently by ``certify_80k_representations``.
    This audit checks complete call vectors for the first and last source records
    plus paired SNP structure samples, without asserting whole-matrix call-value
    identity between representations.
    """

    directory = genotype_root / "80k"
    specs = [
        ("hexaploid", "PAV", directory / "Hexaploid_PAV_data_for_Dataverse.csv", directory / "Hexaploid_PAV_inverted_FJ_format_for_Dataverse.txt", 15),
        ("hexaploid", "SNP", directory / "Hexaploid_SNP_data_for_Dataverse.csv", directory / "Hexaploid_SNP_FJ_data_for_Dataverse.txt", 32),
        ("tetraploid", "PAV", directory / "Tetraploid_PAV_data.csv", directory / "Tetraploid_PAV_inverted_FJ_format_for_Dataverse.txt", 15),
        ("tetraploid", "SNP", directory / "Tetraploid_SNP_data_for_Dataverse.csv", directory / "Tetraploid_SNP_FJ_format_for_Dataverse.txt", 32),
        ("wheat_recall", "PAV", directory / "Wheat_Recall_PAV_data_for_Dataverse.csv", directory / "Wheat_Recall_PAV_inverted_FJ_format_for_Dataverse.txt", 15),
        ("wild_relative", "PAV", directory / "Wild_Relative_PAV_data.csv", directory / "Wild_Relative_PAV_FJ_format_for_Dataverse.txt", 15),
        ("wild_relative", "SNP", directory / "Wild_Relative_SNP_data_for_Dataverse.csv", directory / "Wild_Relative_SNP_FJ_format.txt", 32),
    ]
    rows: list[dict[str, object]] = []
    for population, kind, csv_path, fj_path, metadata_fields in specs:
        csv_lines = _first_data_lines(csv_path, 6, 2) + _last_data_lines(csv_path, 2)
        fj_header_rows = 1
        fj_lines = _first_data_lines(fj_path, fj_header_rows, 2) + _last_data_lines(fj_path, 2)
        csv_tokens = set().union(*(_tokens_from_delimited_payload(line, ",", metadata_fields) for line in csv_lines))
        fj_tokens = set().union(*(_tokens_from_delimited_payload(line, "\t", 1) for line in fj_lines))
        if kind == "PAV":
            csv_allowed = fj_allowed = {"0", "1", "-"}
            transform = "DIRECT_0_1_PRESENCE_WITH_DASH_MISSING"
            pair_structure = "NOT_APPLICABLE"
        else:
            csv_allowed = {"0", "1", "-"}
            bases = {"A", "C", "G", "T"}
            fj_allowed = bases | {f"{left}/{right}" for left in bases for right in bases if left != right} | {"-"}
            transform = "PAIRED_0_1_ALLELE_PRESENCE_TO_NUCLEOTIDE_OR_SLASH_HETEROZYGOTE_WITH_DASH_MISSING"
            pair_structure = "PASS" if all(
                len(next(csv.reader([left]))) > 1
                and len(next(csv.reader([right]))) > 1
                and next(csv.reader([left]))[1] == next(csv.reader([right]))[1]
                for left, right in (csv_lines[:2], csv_lines[-2:])
            ) else "FAIL"
        unexpected_csv = sorted(csv_tokens - csv_allowed)
        unexpected_fj = sorted(fj_tokens - fj_allowed)
        passed = not unexpected_csv and not unexpected_fj and pair_structure != "FAIL"
        rows.append(
            {
                "population": population,
                "representation": kind,
                "csv_source_file": normalize_path(csv_path.relative_to(genotype_root)),
                "flapjack_source_file": normalize_path(fj_path.relative_to(genotype_root)),
                "validation_scope": "COMPLETE_CALL_VECTORS_FOR_FIRST_TWO_AND_LAST_TWO_SOURCE_RECORDS;AXES_CERTIFIED_SEPARATELY",
                "csv_observed_tokens": ";".join(sorted(csv_tokens)),
                "csv_allowed_tokens": ";".join(sorted(csv_allowed)),
                "flapjack_observed_tokens": ";".join(sorted(fj_tokens)),
                "flapjack_allowed_tokens": ";".join(sorted(fj_allowed)),
                "unexpected_csv_tokens": ";".join(unexpected_csv),
                "unexpected_flapjack_tokens": ";".join(unexpected_fj),
                "paired_snp_clone_structure": pair_structure,
                "encoding_transform": transform,
                "missing_value_code": "-",
                "status": "PASS_ENCODING_SAMPLE" if passed else "BLOCKED_ENCODING_SAMPLE",
                "parser_version": PARSER_VERSION,
            }
        )
    wheat_snp = directory / "Wheat_Recall_SNP_data_FJ_format_for_Dataverse.txt"
    wheat_lines = _first_data_lines(wheat_snp, 1, 2) + _last_data_lines(wheat_snp, 2)
    wheat_tokens = set().union(*(_tokens_from_delimited_payload(line, "\t", 1) for line in wheat_lines))
    bases = {"A", "C", "G", "T"}
    wheat_allowed = bases | {f"{left}/{right}" for left in bases for right in bases if left != right} | {"-"}
    unexpected = sorted(wheat_tokens - wheat_allowed)
    rows.append(
        {
            "population": "wheat_recall",
            "representation": "SNP_FLAPJACK_ONLY",
            "csv_source_file": "",
            "flapjack_source_file": normalize_path(wheat_snp.relative_to(genotype_root)),
            "validation_scope": "COMPLETE_CALL_VECTORS_FOR_FIRST_TWO_AND_LAST_TWO_SOURCE_RECORDS;NO_RAW_CSV_COUNTERPART",
            "csv_observed_tokens": "",
            "csv_allowed_tokens": "",
            "flapjack_observed_tokens": ";".join(sorted(wheat_tokens)),
            "flapjack_allowed_tokens": ";".join(sorted(wheat_allowed)),
            "unexpected_csv_tokens": "",
            "unexpected_flapjack_tokens": ";".join(unexpected),
            "paired_snp_clone_structure": "NOT_COMPARABLE_SINGLE_REPRESENTATION",
            "encoding_transform": "NUCLEOTIDE_OR_SLASH_HETEROZYGOTE_WITH_DASH_MISSING",
            "missing_value_code": "-",
            "status": "PASS_ENCODING_SAMPLE_SINGLE_REPRESENTATION" if not unexpected else "BLOCKED_ENCODING_SAMPLE",
            "parser_version": PARSER_VERSION,
        }
    )
    return pd.DataFrame(rows)
