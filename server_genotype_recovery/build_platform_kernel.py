from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np
import pandas as pd

from genotype_recovery import (
    IUPAC_TO_BASES,
    canonical_gid,
    genotype_call_to_dosage,
    load_canonical_catalog,
    load_explicit_sample_gid_mappings,
    marker_alleles,
    normalize_identifier,
    rbf_from_linear_kernel,
    validate_kernel,
)


PLATFORM_DEFAULTS = {
    "80k_hexaploid": {
        "matrix": Path("GENOTYPIC_DATA/80k/Hexaploid_SNP_FJ_data_for_Dataverse.txt"),
        "prefix": "K_G_80K_HEXAPLOID",
        "role": "80k_hexaploid_marker",
        "format": "sample_by_marker",
    },
    "seeds_dartseq": {
        "matrix": Path(
            "GENOTYPIC_DATA/Seeds_of_Discovery_-_MasAgro_Biodiversidad_Wheat_DArTseq-Derived_SNP_Data_Beta_Recall_Results_From_2011-2014/SEQ_SNPs_Extract_45610samples_102474markers.txt"
        ),
        "prefix": "K_G_SEEDS_DARTSEQ",
        "role": "Seeds_of_Discovery_DArTseq_marker",
        "format": "marker_by_sample",
    },
    "iwyp35k": {
        "matrix": Path(
            "GENOTYPIC_DATA/IWYP64_-_HiBAP_35k_Wheat_Breeders_Array_Genotyping/HiBAP_snps_35karray.txt"
        ),
        "prefix": "K_G_IWYP35K",
        "role": "IWYP_HiBAP_35k_marker",
        "format": "iwyp",
    },
    "dartag": {
        "matrix": Path(
            "GENOTYPIC_DATA/Genotypic_data_(DArTAG_panel_2)_for_the_IBWSN_and_SAWSN/DArTAG_numeric.csv"
        ),
        "matrix_extra": [
            Path(
                "GENOTYPIC_DATA/Genotypic_data_(DArTAG_panel_2)_for_the_IBWSN_and_SAWSN/DArTAG_2moreOrders_numeric.csv"
            )
        ],
        "prefix": "K_G_DARTAG",
        "role": "DArTAG_IBWSN_SAWSN_marker",
        "format": "dartag_numeric",
        "sample_heterozygosity_max": 1.0,
    },
}


def resolve_matrix_path(
    root: Path,
    configured_path: Path,
    *,
    discover_by_basename: bool,
    allow_gzip: bool,
) -> Path:
    requested = configured_path.resolve() if configured_path.is_absolute() else (root / configured_path).resolve()
    if requested.is_file() and requested.stat().st_size > 0:
        return requested
    if not discover_by_basename:
        raise SystemExit(f"Genotype matrix is missing or empty: {requested}")

    genotypic_root = root / "GENOTYPIC_DATA"
    names = {configured_path.name}
    if allow_gzip:
        names.add(f"{configured_path.name}.gz")
    matches = sorted(
        path.resolve()
        for path in genotypic_root.rglob("*")
        if path.is_file() and path.name in names and path.stat().st_size > 0
    ) if genotypic_root.is_dir() else []
    if len(matches) == 1:
        print(f"Resolved moved/compressed genotype matrix: {requested} -> {matches[0]}", flush=True)
        return matches[0]
    if len(matches) > 1:
        joined = "\n  ".join(str(path) for path in matches)
        raise SystemExit(
            f"Genotype matrix is absent at {requested} and basename discovery is ambiguous:\n  {joined}"
        )
    accepted = ", ".join(sorted(names))
    raise SystemExit(
        f"Genotype matrix is missing or empty: {requested}. "
        f"No unique fallback named [{accepted}] exists below {genotypic_root}"
    )


@contextmanager
def open_text_matrix(path: Path) -> Iterator[TextIO]:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            yield handle


def portable_output_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def heterozygous_code(ref: str, alt: str) -> int:
    wanted = {ref, alt}
    for code, bases in IUPAC_TO_BASES.items():
        if bases == wanted:
            return ord(code)
    return -1


def decode_call_bytes(
    calls: np.ndarray, refs: np.ndarray, alts: np.ndarray, hetero: np.ndarray
) -> np.ndarray:
    dosage = np.full(len(refs), -1, dtype=np.int8)
    dosage[calls == refs] = 0
    dosage[calls == alts] = 2
    valid_hetero = hetero >= 0
    dosage[valid_hetero & (calls == hetero)] = 1
    return dosage


def decode_single_character_calls(raw: bytes, refs: np.ndarray, alts: np.ndarray) -> np.ndarray:
    marker_count = len(refs)
    stripped = raw.rstrip(b"\r\n")
    if len(stripped) == marker_count * 2 - 1 and np.all(
        np.frombuffer(stripped, dtype=np.uint8)[1::2] == ord("\t")
    ):
        calls = np.frombuffer(stripped, dtype=np.uint8)[::2]
        hetero = np.fromiter(
            (heterozygous_code(chr(ref), chr(alt)) for ref, alt in zip(refs, alts)),
            dtype=np.int16,
            count=marker_count,
        )
        return decode_call_bytes(calls, refs, alts, hetero)
    values = stripped.decode("utf-8", errors="replace").split("\t")
    if len(values) != marker_count:
        raise ValueError(f"Expected {marker_count} calls but found {len(values)}")
    return np.fromiter(
        (
            genotype_call_to_dosage(value, chr(int(ref)), chr(int(alt)))
            for value, ref, alt in zip(values, refs, alts)
        ),
        dtype=np.int8,
        count=marker_count,
    )


def target_sample_lookup(
    *,
    root: Path,
    canonical_catalog_path: Path,
) -> tuple[set[str], dict[str, set[str]]]:
    catalog, aliases = load_canonical_catalog(canonical_catalog_path)
    canonical_ids = set(catalog["canonical_gid"])
    explicit, _ = load_explicit_sample_gid_mappings(root / "GENOTYPIC_DATA")
    lookup: dict[str, set[str]] = defaultdict(set)
    for identifier, gids in explicit.items():
        lookup[identifier].update(gids & canonical_ids)
    for identifier, gids in aliases.items():
        lookup[identifier].update(gids & canonical_ids)
    for gid in canonical_ids:
        lookup[gid.upper()].add(gid)
    return canonical_ids, lookup


def parse_sample_by_marker(
    path: Path,
    lookup: dict[str, set[str]],
) -> tuple[np.ndarray, list[str], list[str], list[str], list[str]]:
    with path.open("rb") as handle:
        header_line = b""
        while True:
            line = handle.readline()
            if not line:
                raise SystemExit(f"Could not find MarkerID header in {path}")
            if line.startswith(b"MarkerID\t"):
                header_line = line
                break
        marker_ids = header_line.decode("utf-8-sig", errors="replace").rstrip("\r\n").split("\t")[1:]
        parsed_alleles = [marker_alleles(marker) for marker in marker_ids]
        valid_marker = np.array([alleles is not None for alleles in parsed_alleles], dtype=bool)
        marker_ids = [marker for marker, keep in zip(marker_ids, valid_marker) if keep]
        alleles = [alleles for alleles in parsed_alleles if alleles is not None]
        refs = np.fromiter((ord(pair[0]) for pair in alleles), dtype=np.uint8, count=len(alleles))
        alts = np.fromiter((ord(pair[1]) for pair in alleles), dtype=np.uint8, count=len(alleles))
        hetero = np.fromiter(
            (heterozygous_code(pair[0], pair[1]) for pair in alleles),
            dtype=np.int16,
            count=len(alleles),
        )
        source_marker_count = len(valid_marker)
        vectors: list[np.ndarray] = []
        gids: list[str] = []
        source_samples: list[str] = []
        matched_rows = 0
        for matrix_row, line in enumerate(handle, start=1):
            if matrix_row % 5000 == 0:
                print(
                    f"[{path.name}] scanned_sample_rows={matrix_row} matched_rows={matched_rows}",
                    flush=True,
                )
            separator = line.find(b"\t")
            if separator < 0:
                continue
            sample_raw = line[:separator]
            sample = sample_raw.decode("utf-8", errors="replace").strip()
            candidates = lookup.get(sample.upper(), set())
            if len(candidates) != 1:
                continue
            calls = line[separator + 1 :]
            stripped = calls.rstrip(b"\r\n")
            dosage = None
            if len(stripped) == source_marker_count * 2 - 1:
                raw_bytes = np.frombuffer(stripped, dtype=np.uint8)
                if np.all(raw_bytes[1::2] == ord("\t")):
                    dosage = decode_call_bytes(raw_bytes[::2][valid_marker], refs, alts, hetero)
            if dosage is None:
                values = calls.rstrip(b"\r\n").decode("utf-8", errors="replace").split("\t")
                if len(values) != source_marker_count:
                    raise SystemExit(
                        f"{path}: sample {sample} has {len(values)} calls; expected {source_marker_count}"
                    )
                kept_values = [value for value, keep in zip(values, valid_marker) if keep]
                dosage = np.fromiter(
                    (
                        genotype_call_to_dosage(value, pair[0], pair[1])
                        for value, pair in zip(kept_values, alleles)
                    ),
                    dtype=np.int8,
                    count=len(marker_ids),
                )
            vectors.append(dosage)
            gids.append(next(iter(candidates)))
            source_samples.append(sample)
            matched_rows += 1
    if not vectors:
        raise SystemExit(f"No canonical samples from {path} were found in the matrix")
    return np.vstack(vectors), gids, source_samples, marker_ids, [f"{a}/{b}" for a, b in alleles]


def parse_marker_by_sample(
    path: Path,
    lookup: dict[str, set[str]],
    *,
    expected_sha256: str | None = None,
    forced_sample_columns: dict[int, tuple[str, str, str]] | None = None,
) -> tuple[np.ndarray, list[str], list[str], list[str], list[str]]:
    """Stream a marker-by-sample text matrix and retain only resolved trial samples."""
    digest = hashlib.sha256() if expected_sha256 else None
    with path.open("rb") as handle:
        header_line = handle.readline()
        if digest is not None:
            digest.update(header_line)
        while header_line and not header_line.startswith(b"MarkerID\t"):
            header_line = handle.readline()
            if digest is not None:
                digest.update(header_line)
        if not header_line:
            raise SystemExit(f"Could not find MarkerID header in {path}")

        header = header_line.decode("utf-8-sig", errors="replace").rstrip("\r\n").split("\t")
        forced = forced_sample_columns or {}
        invalid_columns = sorted(column for column in forced if column < 1 or column >= len(header))
        if invalid_columns:
            raise SystemExit(
                f"Adjudicated marker columns are outside the current matrix header: {invalid_columns[:10]}"
            )
        selected: list[tuple[int, str, str]] = []
        observed_forced: set[int] = set()
        for field_index, raw_sample in enumerate(header[1:], start=1):
            sample = normalize_identifier(raw_sample)
            candidates = lookup.get(sample.upper(), set())
            if field_index in forced:
                forced_gid, forced_display, forced_normalized = forced[field_index]
                if axis_identifier(raw_sample) != forced_normalized:
                    raise SystemExit(
                        "Adjudicated marker column no longer contains the expected sample: "
                        f"column={field_index}; expected={forced_display!r}/"
                        f"{forced_normalized}; observed={raw_sample!r}/"
                        f"{axis_identifier(raw_sample)}"
                    )
                if candidates and candidates != {forced_gid}:
                    raise SystemExit(
                        "Adjudicated marker column conflicts with an existing sample mapping: "
                        f"column={field_index}; sample={raw_sample!r}; "
                        f"existing={sorted(candidates)}; accepted={forced_gid}"
                    )
                selected.append((field_index, forced_gid, sample))
                observed_forced.add(field_index)
            elif len(candidates) == 1:
                selected.append((field_index, next(iter(candidates)), sample))
        missing_forced = sorted(set(forced) - observed_forced)
        if missing_forced:
            raise SystemExit(
                f"Adjudicated marker columns were not observed: {missing_forced[:10]}"
            )
        if not selected:
            raise SystemExit(f"No canonical samples from {path} were found in the matrix header")

        selected_fields = np.asarray([item[0] for item in selected], dtype=np.int64)
        gids = [item[1] for item in selected]
        source_samples = [item[2] for item in selected]
        expected_fields = len(header)
        marker_ids: list[str] = []
        marker_allele_values: list[str] = []
        marker_vectors: list[np.ndarray] = []

        for line_number, line in enumerate(handle, start=2):
            if digest is not None:
                digest.update(line)
            stripped = line.rstrip(b"\r\n")
            if not stripped:
                continue
            raw = np.frombuffer(stripped, dtype=np.uint8)
            tabs = np.flatnonzero(raw == ord("\t"))
            if len(tabs) != expected_fields - 1:
                raise SystemExit(
                    f"{path}: line {line_number} has {len(tabs) + 1} fields; expected {expected_fields}"
                )
            marker_id = stripped[: int(tabs[0])].decode("utf-8", errors="replace")
            pair = marker_alleles(marker_id)
            if pair is None:
                continue

            starts = tabs[selected_fields - 1] + 1
            ends = np.empty_like(selected_fields)
            nonfinal = selected_fields < len(tabs)
            ends[nonfinal] = tabs[selected_fields[nonfinal]]
            ends[~nonfinal] = len(raw)
            widths = ends - starts
            dosage = np.full(len(selected_fields), -1, dtype=np.int8)

            single = widths == 1
            if single.any():
                calls = raw[starts[single]]
                target = np.flatnonzero(single)
                ref_code = ord(pair[0])
                alt_code = ord(pair[1])
                hetero_code = heterozygous_code(pair[0], pair[1])
                dosage[target[calls == ref_code]] = 0
                dosage[target[calls == alt_code]] = 2
                if hetero_code >= 0:
                    dosage[target[calls == hetero_code]] = 1

            for selected_index in np.flatnonzero(~single):
                start = int(starts[selected_index])
                end = int(ends[selected_index])
                value = stripped[start:end].decode("utf-8", errors="replace")
                dosage[selected_index] = genotype_call_to_dosage(value, pair[0], pair[1])

            marker_ids.append(marker_id)
            marker_allele_values.append(f"{pair[0]}/{pair[1]}")
            marker_vectors.append(dosage)
            if line_number % 5000 == 0:
                print(
                    f"[{path.name}] scanned_marker_rows={line_number - 1} "
                    f"biallelic_markers={len(marker_vectors)} selected_samples={len(selected)}",
                    flush=True,
                )

    if not marker_vectors:
        raise SystemExit(f"No biallelic markers were parsed from {path}")
    if digest is not None:
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise SystemExit(
                "Marker matrix changed after identity adjudication: "
                f"expected={expected_sha256}; observed={observed_sha256}; path={path}"
            )
    return np.vstack(marker_vectors).T, gids, source_samples, marker_ids, marker_allele_values


def parse_iwyp(
    path: Path,
    canonical_ids: set[str],
) -> tuple[np.ndarray, list[str], list[str], list[str], list[str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        note = handle.readline()
        entry_row = handle.readline().rstrip("\r\n").split("\t")
        gid_row = handle.readline().rstrip("\r\n").split("\t")
        header = handle.readline().rstrip("\r\n").split("\t")
        if not note or not entry_row or not gid_row or not header:
            raise SystemExit(f"Incomplete IWYP preamble in {path}")
        selected: list[tuple[int, str, str]] = []
        for index in range(min(len(header), len(gid_row))):
            gid = canonical_gid(gid_row[index])
            if gid and gid in canonical_ids:
                selected.append((index, gid, normalize_identifier(header[index])))
        if not selected:
            raise SystemExit(f"No canonical IWYP GIDs found in {path}")
        marker_ids: list[str] = []
        marker_allele_values: list[str] = []
        marker_vectors: list[np.ndarray] = []
        reader = csv.reader(handle, delimiter="\t")
        for values in reader:
            if len(values) < 2:
                continue
            pair = marker_alleles(values[0], values[1])
            if pair is None:
                continue
            dosage = np.fromiter(
                (
                    genotype_call_to_dosage(values[index] if index < len(values) else "", pair[0], pair[1])
                    for index, _, _ in selected
                ),
                dtype=np.int8,
                count=len(selected),
            )
            marker_ids.append(values[0])
            marker_allele_values.append(f"{pair[0]}/{pair[1]}")
            marker_vectors.append(dosage)
    matrix = np.vstack(marker_vectors).T
    return matrix, [item[1] for item in selected], [item[2] for item in selected], marker_ids, marker_allele_values


def parse_dartag_numeric(
    paths: list[Path],
    canonical_ids: set[str],
) -> tuple[np.ndarray, list[str], list[str], list[str], list[str]]:
    """Read the two DArTAG numeric exports without assuming identical sample batches."""
    batches: list[tuple[np.ndarray, list[str], list[str], list[str]]] = []
    marker_union: list[str] = []
    marker_seen: set[str] = set()

    for path in paths:
        with open_text_matrix(path) as handle:
            reader = csv.reader(handle)
            first = next(reader, [])
            if not first:
                raise SystemExit(f"Empty DArTAG matrix: {path}")
            if normalize_identifier(first[0]).upper() == "SUBJECT_ID":
                gid_row = next(reader, [])
                if not gid_row or normalize_identifier(gid_row[0]).upper() != "GID":
                    raise SystemExit(f"DArTAG Subject_ID header is not followed by a GID row: {path}")
                source_ids = [normalize_identifier(value) for value in first[1:]]
                raw_gids = gid_row[1:]
            elif normalize_identifier(first[0]).upper() == "GID":
                raw_gids = first[1:]
                source_ids = [normalize_identifier(value) for value in raw_gids]
            else:
                raise SystemExit(f"Unrecognized DArTAG numeric header in {path}: {first[0]!r}")

            selected: list[tuple[int, str, str]] = []
            for index, raw_gid in enumerate(raw_gids, start=1):
                gid = canonical_gid(raw_gid)
                if gid and gid in canonical_ids:
                    selected.append((index, gid, source_ids[index - 1]))
            if not selected:
                print(f"[{path.name}] no canonical DArTAG samples; batch skipped", flush=True)
                continue

            selected_fields = np.asarray([item[0] for item in selected], dtype=np.int64)
            markers: list[str] = []
            marker_vectors: list[np.ndarray] = []
            for row_number, values in enumerate(reader, start=2):
                if not values:
                    continue
                marker = normalize_identifier(values[0])
                if not marker:
                    continue
                dosage = np.full(len(selected), -1, dtype=np.int8)
                for target_index, field_index in enumerate(selected_fields):
                    raw = normalize_identifier(values[field_index] if field_index < len(values) else "")
                    if raw in {"0", "1", "2"}:
                        dosage[target_index] = int(raw)
                markers.append(marker)
                marker_vectors.append(dosage)
                if marker not in marker_seen:
                    marker_seen.add(marker)
                    marker_union.append(marker)
                if row_number % 1000 == 0:
                    print(
                        f"[{path.name}] marker_rows={row_number - 1} selected_samples={len(selected)}",
                        flush=True,
                    )
            if not marker_vectors:
                raise SystemExit(f"No DArTAG marker rows were parsed from {path}")
            if len(markers) != len(set(markers)):
                raise SystemExit(f"DArTAG matrix contains duplicate marker IDs: {path}")
            batches.append(
                (
                    np.vstack(marker_vectors).T,
                    [item[1] for item in selected],
                    [f"{path.name}:{item[2]}" for item in selected],
                    markers,
                )
            )

    if not batches:
        raise SystemExit("No canonical DArTAG samples were found in any numeric export")
    marker_index = {marker: index for index, marker in enumerate(marker_union)}
    matrices: list[np.ndarray] = []
    gids: list[str] = []
    source_samples: list[str] = []
    for local, local_gids, local_sources, local_markers in batches:
        expanded = np.full((len(local), len(marker_union)), -1, dtype=np.int8)
        target = np.fromiter((marker_index[marker] for marker in local_markers), dtype=np.int64)
        expanded[:, target] = local
        matrices.append(expanded)
        gids.extend(local_gids)
        source_samples.extend(local_sources)
    return (
        np.vstack(matrices),
        gids,
        source_samples,
        marker_union,
        ["numeric_0_1_2"] * len(marker_union),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boolean_values(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    invalid = sorted(set(normalized) - {"", "0", "1", "false", "true", "no", "yes"})
    if invalid:
        raise ValueError(f"Cannot interpret boolean values: {invalid[:10]}")
    return normalized.isin({"1", "true", "yes"})


def axis_identifier(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", normalize_identifier(value).upper())


def load_accepted_identity_mappings(
    *,
    identity_dir: Path,
    panel_ids: set[str],
    canonical_ids: set[str],
    matrix_path: Path,
) -> tuple[
    pd.DataFrame,
    dict[int, tuple[str, str, str]],
    dict[str, set[str]],
    str,
    dict[str, str],
]:
    candidates_path = identity_dir / "marker_identity_candidate_paths.tsv.gz"
    pairs_path = identity_dir / "marker_identity_pairwise_concordance.tsv.gz"
    validation_path = identity_dir / "marker_identity_validation.tsv"
    provenance_path = identity_dir / "marker_identity_adjudication_provenance.json"
    required_paths = [candidates_path, pairs_path, validation_path, provenance_path]
    missing = [str(path) for path in required_paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit(f"Accepted identity evidence is incomplete: {missing}")

    validation = pd.read_csv(validation_path, sep="\t", dtype=str)
    if not {"check", "status"}.issubset(validation.columns):
        raise SystemExit(f"Identity validation has an invalid schema: {validation_path}")
    failed = validation[~validation["status"].eq("PASS")]
    required_checks = {
        "classification_evidence_preserved",
        "pairwise_concordance_evidence_preserved",
        "regulatory_overlay_is_gated",
    }
    observed_checks = set(validation["check"])
    if not failed.empty or not required_checks.issubset(observed_checks):
        raise SystemExit(
            "Accepted identity evidence is not a reconciled PASS artifact: "
            f"failed={failed.to_dict('records')}; missing_checks={sorted(required_checks - observed_checks)}"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("status") != "PASS"
        or provenance.get("classification_evidence_reused") is not True
        or provenance.get("classification_or_concordance_modified") is not False
        or provenance.get("marker_calls_read") is not False
        or provenance.get("phenotype_values_read") is not False
        or provenance.get("outer_test_metrics_read") is not False
        or provenance.get("final_holdout_outcomes_read") is not False
    ):
        raise SystemExit(f"Identity reconciliation provenance contract failed: {provenance_path}")

    candidates = pd.read_csv(candidates_path, sep="\t", dtype=str)
    required_columns = {
        "trial_gid",
        "panel_id",
        "sample_id",
        "normalized_sample_id",
        "marker_matrix_path",
        "marker_matrix_sha256",
        "marker_matrix_axis_index",
        "marker_axis_match_count",
        "classification",
        "direct_marker_assignment_ready",
        "existing_certified_in_panel",
        "existing_certified_in_any_panel",
    }
    missing_columns = sorted(required_columns - set(candidates.columns))
    if missing_columns:
        raise SystemExit(f"Identity candidates are missing columns: {missing_columns}")
    candidates["direct_marker_assignment_ready"] = boolean_values(
        candidates["direct_marker_assignment_ready"]
    )
    candidates["existing_certified_in_panel"] = boolean_values(
        candidates["existing_certified_in_panel"]
    )
    candidates["existing_certified_in_any_panel"] = boolean_values(
        candidates["existing_certified_in_any_panel"]
    )
    accepted = candidates[
        candidates["panel_id"].isin(panel_ids)
        & candidates["direct_marker_assignment_ready"]
    ].copy()
    if accepted.empty:
        raise SystemExit(f"No accepted marker identities found for panels {sorted(panel_ids)}")
    accepted["trial_gid"] = accepted["trial_gid"].map(canonical_gid)
    if accepted["trial_gid"].eq("").any():
        raise SystemExit("Accepted identity artifact contains noncanonical trial GIDs")
    outside_catalog = sorted(set(accepted["trial_gid"]) - canonical_ids)
    if outside_catalog:
        raise SystemExit(
            f"Accepted identity artifact contains {len(outside_catalog)} GIDs outside the canonical catalog"
        )
    accepted["sample_id"] = accepted["sample_id"].map(normalize_identifier)
    accepted["normalized_sample_id"] = accepted["normalized_sample_id"].map(axis_identifier)
    if accepted["sample_id"].eq("").any():
        raise SystemExit("Accepted identity artifact contains empty physical marker sample IDs")
    if accepted["normalized_sample_id"].eq("").any():
        raise SystemExit("Accepted identity artifact contains empty marker sample IDs")
    axis_matches = pd.to_numeric(accepted["marker_axis_match_count"], errors="coerce")
    if not axis_matches.eq(1).all():
        raise SystemExit("Accepted identities must map to exactly one marker-matrix axis")

    expected_paths = {str(Path(value).resolve()) for value in accepted["marker_matrix_path"]}
    if expected_paths != {str(matrix_path.resolve())}:
        raise SystemExit(
            "Accepted identities were adjudicated against a different marker matrix: "
            f"expected={sorted(expected_paths)}; current={matrix_path.resolve()}"
        )
    matrix_hashes = set(accepted["marker_matrix_sha256"].fillna("").str.strip()) - {""}
    if len(matrix_hashes) != 1:
        raise SystemExit(f"Accepted identities do not share one marker matrix hash: {matrix_hashes}")
    expected_matrix_sha256 = next(iter(matrix_hashes))

    accepted_columns: dict[int, tuple[str, str, str]] = {}
    for row in accepted.itertuples(index=False):
        try:
            column = int(float(row.marker_matrix_axis_index))
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Accepted identity has a nonnumeric marker column: {row.marker_matrix_axis_index!r}"
            ) from exc
        value = (str(row.trial_gid), str(row.sample_id), str(row.normalized_sample_id))
        existing = accepted_columns.get(column)
        if existing is not None and existing != value:
            raise SystemExit(
                f"Accepted marker column maps to multiple identities: column={column}; "
                f"first={existing}; second={value}"
            )
        accepted_columns[column] = value

    replicate_groups: dict[str, set[str]] = {}
    replicate_rows = accepted[
        accepted["classification"].eq("accepted_concordant_replicates")
    ]
    for gid, group in replicate_rows.groupby("trial_gid", sort=True):
        samples = set(group["sample_id"].str.upper()) - {""}
        if len(samples) < 2:
            raise SystemExit(f"Accepted replicate identity has fewer than two samples: {gid}")
        replicate_groups[gid] = samples

    pairs = pd.read_csv(pairs_path, sep="\t", dtype=str)
    pair_required = {
        "trial_gid",
        "panel_id",
        "sample_id_left",
        "sample_id_right",
        "overlap_pass",
        "concordance_pass",
        "pair_status",
    }
    if not pair_required.issubset(pairs.columns):
        raise SystemExit(f"Pairwise evidence is missing columns: {sorted(pair_required - set(pairs.columns))}")
    pairs["overlap_pass"] = boolean_values(pairs["overlap_pass"])
    pairs["concordance_pass"] = boolean_values(pairs["concordance_pass"])
    normalized_replicate_groups = {
        gid: set(group["normalized_sample_id"].str.upper()) - {""}
        for gid, group in replicate_rows.groupby("trial_gid", sort=True)
    }
    for gid, samples in normalized_replicate_groups.items():
        local = pairs[pairs["trial_gid"].map(canonical_gid).eq(gid) & pairs["panel_id"].isin(panel_ids)]
        expected_pairs = len(samples) * (len(samples) - 1) // 2
        if (
            len(local) != expected_pairs
            or not local["overlap_pass"].all()
            or not local["concordance_pass"].all()
            or not local["pair_status"].eq("PASS").all()
        ):
            raise SystemExit(f"Accepted replicate evidence is incomplete or nonpassing: {gid}")

    input_hashes = {
        "candidates": sha256_file(candidates_path),
        "pairs": sha256_file(pairs_path),
        "validation": sha256_file(validation_path),
        "provenance": sha256_file(provenance_path),
    }
    return accepted, accepted_columns, replicate_groups, expected_matrix_sha256, input_hashes


def collapse_accepted_replicates(
    matrix: np.ndarray,
    gids: list[str],
    source_samples: list[str],
    replicate_groups: dict[str, set[str]],
) -> tuple[np.ndarray, list[str], list[str], pd.DataFrame]:
    if not replicate_groups:
        return matrix, gids, source_samples, pd.DataFrame(
            columns=[
                "sample_id",
                "source_samples",
                "source_sample_count",
                "unanimous_multisample_markers",
                "single_observed_sample_markers",
                "discordant_observed_markers_set_missing",
                "all_missing_markers",
                "collapse_status",
            ]
        )
    normalized_sources = [normalize_identifier(value).upper() for value in source_samples]
    consumed: set[int] = set()
    consensus_rows: list[np.ndarray] = []
    consensus_gids: list[str] = []
    consensus_sources: list[str] = []
    audit_rows: list[dict[str, object]] = []
    for gid, expected_samples in sorted(replicate_groups.items()):
        indices = [
            index
            for index, (local_gid, sample) in enumerate(zip(gids, normalized_sources))
            if local_gid == gid and sample in expected_samples
        ]
        observed_samples = {normalized_sources[index] for index in indices}
        if observed_samples != expected_samples or len(indices) != len(expected_samples):
            raise SystemExit(
                f"Accepted replicate samples were not found exactly once in the matrix: "
                f"gid={gid}; expected={sorted(expected_samples)}; observed={sorted(observed_samples)}"
            )
        local = matrix[indices]
        observed_count = (local >= 0).sum(axis=0)
        consensus = np.full(matrix.shape[1], -1, dtype=np.int8)
        for dosage in (0, 1, 2):
            unanimous = (local == dosage).sum(axis=0) == observed_count
            consensus[unanimous & (observed_count > 0)] = dosage
        discordant = (observed_count >= 2) & (consensus < 0)
        consumed.update(indices)
        consensus_rows.append(consensus)
        consensus_gids.append(gid)
        consensus_source = "CONSENSUS:" + "|".join(sorted(expected_samples))
        consensus_sources.append(consensus_source)
        audit_rows.append(
            {
                "sample_id": gid,
                "source_samples": ";".join(sorted(expected_samples)),
                "source_sample_count": len(expected_samples),
                "unanimous_multisample_markers": int(
                    ((observed_count >= 2) & (consensus >= 0)).sum()
                ),
                "single_observed_sample_markers": int((observed_count == 1).sum()),
                "discordant_observed_markers_set_missing": int(discordant.sum()),
                "all_missing_markers": int((observed_count == 0).sum()),
                "collapse_status": "accepted_concordant_replicates_collapsed",
            }
        )
    retained = [index for index in range(len(gids)) if index not in consumed]
    matrices = [matrix[retained]] if retained else []
    matrices.append(np.vstack(consensus_rows))
    return (
        np.vstack(matrices),
        [gids[index] for index in retained] + consensus_gids,
        [source_samples[index] for index in retained] + consensus_sources,
        pd.DataFrame(audit_rows),
    )


def duplicate_call_concordance(
    matrix: np.ndarray,
    gids: list[str],
    source_samples: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_gid: dict[str, list[int]] = defaultdict(list)
    for index, gid in enumerate(gids):
        by_gid[gid].append(index)
    for gid, indices in sorted(by_gid.items()):
        if len(indices) < 2:
            continue
        for left_pos, left in enumerate(indices[:-1]):
            for right in indices[left_pos + 1 :]:
                observed = (matrix[left] >= 0) & (matrix[right] >= 0)
                overlap = int(observed.sum())
                rows.append(
                    {
                        "sample_id": gid,
                        "source_sample_left": source_samples[left],
                        "source_sample_right": source_samples[right],
                        "overlapping_observed_markers": overlap,
                        "call_concordance": (
                            float(np.mean(matrix[left, observed] == matrix[right, observed]))
                            if overlap
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "source_sample_left",
            "source_sample_right",
            "overlapping_observed_markers",
            "call_concordance",
        ],
    )


def qc_samples(
    matrix: np.ndarray,
    gids: list[str],
    source_samples: list[str],
    *,
    missing_max: float,
    heterozygosity_max: float,
) -> tuple[np.ndarray, list[str], list[str], pd.DataFrame, pd.DataFrame]:
    missing = (matrix < 0).mean(axis=1)
    observed = matrix >= 0
    heterozygous = np.divide(
        (matrix == 1).sum(axis=1),
        observed.sum(axis=1),
        out=np.ones(len(matrix), dtype=float),
        where=observed.sum(axis=1) > 0,
    )
    records = pd.DataFrame(
        {
            "source_row": np.arange(len(matrix)),
            "sample_id": gids,
            "source_sample_id": source_samples,
            "missingness": missing,
            "heterozygosity": heterozygous,
        }
    )
    records["passes_thresholds"] = (
        records["missingness"].le(missing_max)
        & records["heterozygosity"].le(heterozygosity_max)
    )
    passing = records[records["passes_thresholds"]].sort_values(
        ["sample_id", "missingness", "heterozygosity", "source_sample_id"], kind="stable"
    )
    selected = passing.drop_duplicates("sample_id", keep="first")
    if len(selected) < 2:
        raise SystemExit(
            "Fewer than two unique canonical samples passed sample QC; a relationship kernel cannot be built"
        )
    selected_rows = selected["source_row"].to_numpy(dtype=int)
    records["selected_for_kernel"] = records["source_row"].isin(selected_rows)
    duplicate_resolution = passing[passing.duplicated("sample_id", keep=False)].copy()
    duplicate_resolution["duplicate_rank"] = duplicate_resolution.groupby("sample_id").cumcount() + 1
    duplicate_resolution["selected_for_kernel"] = duplicate_resolution["duplicate_rank"].eq(1)
    return (
        matrix[selected_rows],
        selected["sample_id"].tolist(),
        selected["source_sample_id"].tolist(),
        records,
        duplicate_resolution,
    )


def qc_markers(
    matrix: np.ndarray,
    marker_ids: list[str],
    marker_alleles_values: list[str],
    *,
    missing_max: float,
    maf_min: float,
    heterozygosity_max: float,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    observed = matrix >= 0
    observed_count = observed.sum(axis=0)
    missingness = 1.0 - observed_count / len(matrix)
    dosage_sum = np.where(observed, matrix, 0).sum(axis=0, dtype=np.float64)
    frequency = np.divide(
        dosage_sum,
        2.0 * observed_count,
        out=np.full(matrix.shape[1], np.nan),
        where=observed_count > 0,
    )
    maf = np.minimum(frequency, 1.0 - frequency)
    heterozygosity = np.divide(
        (matrix == 1).sum(axis=0),
        observed_count,
        out=np.ones(matrix.shape[1], dtype=float),
        where=observed_count > 0,
    )
    keep = (
        np.isfinite(maf)
        & (missingness <= missing_max)
        & (maf >= maf_min)
        & (heterozygosity <= heterozygosity_max)
    )
    qc = pd.DataFrame(
        {
            "marker_id": marker_ids,
            "alleles": marker_alleles_values,
            "missingness": missingness,
            "allele_frequency": frequency,
            "maf": maf,
            "heterozygosity": heterozygosity,
            "retained": keep,
        }
    )
    reasons = np.full(len(qc), "retained", dtype=object)
    reasons[missingness > missing_max] = "high_missingness"
    reasons[np.isfinite(maf) & (maf < maf_min)] = "low_maf"
    reasons[heterozygosity > heterozygosity_max] = "high_heterozygosity"
    reasons[~np.isfinite(maf)] = "no_observed_calls"
    qc["removal_reason"] = reasons
    if not keep.any():
        raise SystemExit("All markers failed platform QC")
    return matrix[:, keep], qc, frequency[keep]


def vanraden_chunked(matrix: np.ndarray, frequency: np.ndarray, chunk_size: int) -> tuple[np.ndarray, float]:
    denominator = float(np.sum(2.0 * frequency * (1.0 - frequency)))
    if not np.isfinite(denominator) or denominator <= 0:
        raise SystemExit(f"Invalid VanRaden denominator: {denominator}")
    kernel = np.zeros((len(matrix), len(matrix)), dtype=np.float64)
    for start in range(0, matrix.shape[1], chunk_size):
        stop = min(start + chunk_size, matrix.shape[1])
        block = matrix[:, start:stop].astype(np.float32)
        p = frequency[start:stop].astype(np.float32)
        means = 2.0 * p
        missing = block < 0
        if missing.any():
            block[missing] = np.broadcast_to(means, block.shape)[missing]
        block -= means
        kernel += block @ block.T
    kernel /= denominator
    kernel = (kernel + kernel.T) * 0.5
    mean_diagonal = float(np.mean(np.diag(kernel)))
    if not np.isfinite(mean_diagonal) or mean_diagonal <= 0:
        raise SystemExit(f"Invalid genomic kernel mean diagonal: {mean_diagonal}")
    return (kernel / mean_diagonal).astype(np.float32), denominator


def baseline_kernel_comparison(
    *,
    baseline_kernel_path: Path,
    baseline_order_path: Path,
    candidate_kernel: np.ndarray,
    candidate_gids: list[str],
    minimum_correlation: float,
) -> pd.DataFrame:
    if not baseline_kernel_path.is_file() or not baseline_order_path.is_file():
        raise SystemExit(
            "Identity-recovered kernel requires the certified baseline kernel and order: "
            f"kernel={baseline_kernel_path}; order={baseline_order_path}"
        )
    baseline = np.load(baseline_kernel_path, mmap_mode="r")
    order = pd.read_csv(baseline_order_path, sep="\t", dtype=str)
    if "sample_id" not in order.columns:
        raise SystemExit(f"Baseline sample order lacks sample_id: {baseline_order_path}")
    baseline_gids = [canonical_gid(value) for value in order["sample_id"]]
    if baseline.shape != (len(baseline_gids), len(baseline_gids)):
        raise SystemExit(
            f"Baseline kernel/order mismatch: shape={baseline.shape}; order={len(baseline_gids)}"
        )
    candidate_index = {gid: index for index, gid in enumerate(candidate_gids)}
    missing = sorted(set(baseline_gids) - set(candidate_index))
    if missing:
        raise SystemExit(
            f"Identity-recovered panel lost {len(missing)} certified baseline GIDs; "
            f"examples={missing[:10]}"
        )
    candidate_positions = np.asarray([candidate_index[gid] for gid in baseline_gids], dtype=int)
    candidate_shared = candidate_kernel[np.ix_(candidate_positions, candidate_positions)]
    triangle = np.triu_indices(len(baseline_gids), k=1)
    baseline_values = np.asarray(baseline[triangle], dtype=np.float64)
    candidate_values = np.asarray(candidate_shared[triangle], dtype=np.float64)
    finite = np.isfinite(baseline_values) & np.isfinite(candidate_values)
    correlation = (
        float(np.corrcoef(baseline_values[finite], candidate_values[finite])[0, 1])
        if finite.sum() >= 2
        else float("nan")
    )
    rmse = (
        float(np.sqrt(np.mean((baseline_values[finite] - candidate_values[finite]) ** 2)))
        if finite.any()
        else float("nan")
    )
    passed = bool(np.isfinite(correlation) and correlation >= minimum_correlation)
    result = pd.DataFrame(
        [
            {
                "baseline_kernel": str(baseline_kernel_path),
                "baseline_order": str(baseline_order_path),
                "baseline_gid_count": len(baseline_gids),
                "candidate_gid_count": len(candidate_gids),
                "new_candidate_gids": len(set(candidate_gids) - set(baseline_gids)),
                "shared_offdiagonal_finite_pairs": int(finite.sum()),
                "shared_offdiagonal_correlation": correlation,
                "shared_offdiagonal_rmse": rmse,
                "minimum_required_correlation": minimum_correlation,
                "status": "PASS" if passed else "FAIL",
            }
        ]
    )
    if not passed:
        raise SystemExit(
            "Identity-recovered kernel is not concordant with the certified Seeds baseline: "
            f"correlation={correlation}; minimum={minimum_correlation}"
        )
    return result


def identity_recovery_status(
    *,
    accepted: pd.DataFrame,
    raw_gids: list[str],
    raw_source_samples: list[str],
    collapse_audit: pd.DataFrame,
    sample_qc: pd.DataFrame,
    final_gids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_sources: dict[str, set[str]] = defaultdict(set)
    for gid, source in zip(raw_gids, raw_source_samples):
        raw_sources[gid].add(normalize_identifier(source))
    collapsed = set(collapse_audit.get("sample_id", pd.Series(dtype=str)))
    selected = set(final_gids)
    rows: list[dict[str, object]] = []
    for gid, group in accepted.groupby("trial_gid", sort=True):
        classes = sorted(set(group["classification"]))
        expected_samples = sorted(set(group["sample_id"]) - {""})
        found_samples = sorted(raw_sources.get(gid, set()))
        all_found = set(sample.upper() for sample in expected_samples).issubset(
            {sample.upper() for sample in found_samples}
        )
        if not all_found:
            raise SystemExit(
                f"Accepted marker samples disappeared during matrix parsing: gid={gid}; "
                f"expected={expected_samples}; found={found_samples}"
            )
        local_qc = sample_qc[sample_qc["sample_id"].eq(gid)]
        included = gid in selected
        in_panel = boolean_values(group["existing_certified_in_panel"]).all()
        in_any = boolean_values(group["existing_certified_in_any_panel"]).all()
        rows.append(
            {
                "trial_gid": gid,
                "classification": ";".join(classes),
                "accepted_sample_ids": ";".join(expected_samples),
                "accepted_sample_count": len(expected_samples),
                "matrix_sample_ids_found": ";".join(found_samples),
                "matrix_sample_count_found": len(found_samples),
                "all_accepted_samples_found": all_found,
                "replicate_consensus_materialized": gid in collapsed,
                "any_sample_passed_qc": bool(local_qc["passes_thresholds"].any()),
                "included_in_candidate_kernel": included,
                "existing_certified_in_reference_panel": in_panel,
                "existing_certified_in_any_panel": in_any,
                "panel_coverage_status": (
                    "existing_panel_gid_recertified"
                    if included and in_panel
                    else "new_panel_gid_certified"
                    if included
                    else "identity_accepted_but_failed_genotype_qc"
                ),
            }
        )
    status = pd.DataFrame(rows)
    included = status["included_in_candidate_kernel"]
    summary = pd.DataFrame(
        [
            {"metric": "accepted_identity_gids", "value": len(status)},
            {
                "metric": "accepted_identity_gids_in_candidate_kernel",
                "value": int(included.sum()),
            },
            {
                "metric": "accepted_identity_gids_failed_genotype_qc",
                "value": int((~included).sum()),
            },
            {
                "metric": "accepted_new_to_reference_panel_gids",
                "value": int((~status["existing_certified_in_reference_panel"]).sum()),
            },
            {
                "metric": "certified_new_to_reference_panel_gids",
                "value": int(
                    (included & ~status["existing_certified_in_reference_panel"]).sum()
                ),
            },
            {
                "metric": "accepted_new_to_any_certified_panel_gids",
                "value": int((~status["existing_certified_in_any_panel"]).sum()),
            },
            {
                "metric": "certified_new_to_any_prior_panel_gids",
                "value": int((included & ~status["existing_certified_in_any_panel"]).sum()),
            },
            {
                "metric": "accepted_replicate_consensus_gids",
                "value": int(status["replicate_consensus_materialized"].sum()),
            },
            {"metric": "phenotype_values_read", "value": False},
            {"metric": "outer_test_metrics_read", "value": False},
            {"metric": "final_holdout_outcomes_read", "value": False},
        ]
    )
    return status, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a separately certified raw-platform genomic kernel.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--platform", choices=sorted(PLATFORM_DEFAULTS), required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--matrix-extra", type=Path, action="append", default=[])
    parser.add_argument("--canonical-catalog", type=Path, default=Path("audit/canonical_genotype_mapping_audited.csv"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--prefix")
    parser.add_argument("--sample-missing-max", type=float, default=0.20)
    parser.add_argument("--sample-heterozygosity-max", type=float)
    parser.add_argument("--marker-missing-max", type=float, default=0.20)
    parser.add_argument("--marker-heterozygosity-max", type=float, default=0.20)
    parser.add_argument("--maf-min", type=float, default=0.01)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--save-dosage", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--identity-adjudication-dir", type=Path)
    parser.add_argument("--identity-panel", action="append", default=[])
    parser.add_argument("--baseline-kernel", type=Path)
    parser.add_argument("--baseline-order", type=Path)
    parser.add_argument("--minimum-baseline-kernel-correlation", type=float, default=0.90)
    args = parser.parse_args()

    root = args.root.resolve()
    defaults = PLATFORM_DEFAULTS[args.platform]
    sample_heterozygosity_max = (
        args.sample_heterozygosity_max
        if args.sample_heterozygosity_max is not None
        else float(defaults.get("sample_heterozygosity_max", 0.20))
    )
    matrix_inputs = (
        [args.matrix]
        if args.matrix
        else [defaults["matrix"], *defaults.get("matrix_extra", [])]
    )
    matrix_inputs.extend(args.matrix_extra)
    matrix_paths = [
        resolve_matrix_path(
            root,
            path,
            discover_by_basename=args.matrix is None,
            allow_gzip=defaults["format"] == "dartag_numeric",
        )
        for path in matrix_inputs
    ]
    if args.preflight_only:
        for matrix_path in matrix_paths:
            print(f"PASS platform={args.platform} matrix={matrix_path} bytes={matrix_path.stat().st_size}")
        return
    prefix = args.prefix or str(defaults["prefix"])
    out_dir = (root / (args.out_dir or Path("genotype_panels/recovered") / args.platform)).resolve()
    identity_dir = (
        (args.identity_adjudication_dir.resolve() if args.identity_adjudication_dir.is_absolute()
         else (root / args.identity_adjudication_dir).resolve())
        if args.identity_adjudication_dir
        else None
    )
    if identity_dir is not None:
        default_out = (root / "genotype_panels/recovered" / args.platform).resolve()
        if defaults["format"] != "marker_by_sample":
            raise SystemExit("Accepted sample-axis identity recovery currently requires marker_by_sample data")
        if not args.identity_panel:
            raise SystemExit("--identity-panel is required with --identity-adjudication-dir")
        if out_dir == default_out or prefix == str(defaults["prefix"]):
            raise SystemExit(
                "Identity-recovered kernels must use an isolated output directory and prefix"
            )
        if args.baseline_kernel is None or args.baseline_order is None:
            raise SystemExit(
                "Identity-recovered kernels require --baseline-kernel and --baseline-order"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_catalog_path = (root / args.canonical_catalog).resolve()
    if not canonical_catalog_path.is_file() or canonical_catalog_path.stat().st_size == 0:
        raise SystemExit(f"Canonical genotype catalog is missing or empty: {canonical_catalog_path}")
    canonical_ids, lookup = target_sample_lookup(root=root, canonical_catalog_path=canonical_catalog_path)
    accepted_identities = pd.DataFrame()
    accepted_sample_columns: dict[int, tuple[str, str, str]] = {}
    accepted_replicates: dict[str, set[str]] = {}
    expected_matrix_sha256: str | None = None
    identity_input_hashes: dict[str, str] = {}
    if identity_dir is not None:
        (
            accepted_identities,
            accepted_sample_columns,
            accepted_replicates,
            expected_matrix_sha256,
            identity_input_hashes,
        ) = load_accepted_identity_mappings(
            identity_dir=identity_dir,
            panel_ids=set(args.identity_panel),
            canonical_ids=canonical_ids,
            matrix_path=matrix_paths[0],
        )

    if defaults["format"] == "sample_by_marker":
        matrix, gids, source_samples, marker_ids, allele_values = parse_sample_by_marker(matrix_paths[0], lookup)
    elif defaults["format"] == "marker_by_sample":
        matrix, gids, source_samples, marker_ids, allele_values = parse_marker_by_sample(
            matrix_paths[0],
            lookup,
            expected_sha256=expected_matrix_sha256,
            forced_sample_columns=accepted_sample_columns,
        )
    elif defaults["format"] == "iwyp":
        matrix, gids, source_samples, marker_ids, allele_values = parse_iwyp(matrix_paths[0], canonical_ids)
    else:
        matrix, gids, source_samples, marker_ids, allele_values = parse_dartag_numeric(
            matrix_paths, canonical_ids
        )
    raw_shape = matrix.shape
    raw_gids = list(gids)
    raw_source_samples = list(source_samples)
    duplicate_concordance = duplicate_call_concordance(matrix, gids, source_samples)
    matrix, gids, source_samples, collapse_audit = collapse_accepted_replicates(
        matrix, gids, source_samples, accepted_replicates
    )
    matrix, gids, source_samples, sample_qc, duplicates = qc_samples(
        matrix,
        gids,
        source_samples,
        missing_max=args.sample_missing_max,
        heterozygosity_max=sample_heterozygosity_max,
    )
    matrix, marker_qc, frequency = qc_markers(
        matrix,
        marker_ids,
        allele_values,
        missing_max=args.marker_missing_max,
        maf_min=args.maf_min,
        heterozygosity_max=args.marker_heterozygosity_max,
    )
    retained_marker_ids = marker_qc.loc[marker_qc["retained"], "marker_id"].tolist()
    kernel, denominator = vanraden_chunked(matrix, frequency, args.chunk_size)
    rbf, gamma = rbf_from_linear_kernel(kernel)
    linear_qc = validate_kernel(kernel, name=f"{prefix}_LINEAR")
    rbf_qc = validate_kernel(rbf, name=f"{prefix}_RBF")

    identity_status = pd.DataFrame()
    identity_summary = pd.DataFrame()
    baseline_comparison = pd.DataFrame()
    if identity_dir is not None:
        identity_status, identity_summary = identity_recovery_status(
            accepted=accepted_identities,
            raw_gids=raw_gids,
            raw_source_samples=raw_source_samples,
            collapse_audit=collapse_audit,
            sample_qc=sample_qc,
            final_gids=gids,
        )
        baseline_kernel_path = (
            args.baseline_kernel.resolve()
            if args.baseline_kernel.is_absolute()
            else (root / args.baseline_kernel).resolve()
        )
        baseline_order_path = (
            args.baseline_order.resolve()
            if args.baseline_order.is_absolute()
            else (root / args.baseline_order).resolve()
        )
        baseline_comparison = baseline_kernel_comparison(
            baseline_kernel_path=baseline_kernel_path,
            baseline_order_path=baseline_order_path,
            candidate_kernel=kernel,
            candidate_gids=gids,
            minimum_correlation=args.minimum_baseline_kernel_correlation,
        )

    linear_path = out_dir / f"{prefix}_LINEAR.npy"
    rbf_path = out_dir / f"{prefix}_RBF.npy"
    order_path = out_dir / f"{prefix}_sample_order.tsv"
    np.save(linear_path, kernel)
    np.save(rbf_path, rbf)
    pd.DataFrame(
        {
            "compact_kernel_index": np.arange(len(gids), dtype=int),
            "sample_id": gids,
            "source_sample_id": source_samples,
            "platform": args.platform,
        }
    ).to_csv(order_path, sep="\t", index=False)
    sample_qc.to_csv(out_dir / f"{prefix}_sample_qc.tsv", sep="\t", index=False)
    duplicates.to_csv(out_dir / f"{prefix}_duplicate_gid_resolution.tsv", sep="\t", index=False)
    duplicate_concordance.to_csv(
        out_dir / f"{prefix}_duplicate_call_concordance.tsv", sep="\t", index=False
    )
    collapse_audit.to_csv(
        out_dir / f"{prefix}_accepted_replicate_collapse.tsv", sep="\t", index=False
    )
    marker_qc.to_csv(out_dir / f"{prefix}_marker_qc.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame({"marker_id": retained_marker_ids}).to_csv(
        out_dir / f"{prefix}_retained_marker_order.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    if args.save_dosage:
        np.save(out_dir / f"{prefix}_QC_dosage.npy", matrix.astype(np.int8))

    certification = pd.DataFrame([linear_qc, rbf_qc])
    certification.to_csv(out_dir / f"{prefix}_kernel_certification.tsv", sep="\t", index=False)
    summary = pd.DataFrame(
        [
            {"metric": "platform", "value": args.platform},
            {"metric": "matrix_format", "value": defaults["format"]},
            {"metric": "matrix_path", "value": ";".join(map(str, matrix_paths))},
            {"metric": "matrix_bytes", "value": sum(path.stat().st_size for path in matrix_paths)},
            {"metric": "raw_matched_sample_rows", "value": raw_shape[0]},
            {"metric": "raw_biallelic_markers", "value": raw_shape[1]},
            {"metric": "samples_after_qc_and_gid_deduplication", "value": len(gids)},
            {"metric": "markers_after_qc", "value": matrix.shape[1]},
            {"metric": "sample_missing_max", "value": args.sample_missing_max},
            {"metric": "sample_heterozygosity_max", "value": sample_heterozygosity_max},
            {
                "metric": "sample_heterozygosity_qc_note",
                "value": (
                    "audited_not_excluded_for_polyploid_targeted_calls"
                    if args.platform == "dartag"
                    else "threshold_applied"
                ),
            },
            {"metric": "marker_missing_max", "value": args.marker_missing_max},
            {"metric": "marker_heterozygosity_max", "value": args.marker_heterozygosity_max},
            {"metric": "maf_min", "value": args.maf_min},
            {"metric": "vanraden_denominator", "value": denominator},
            {"metric": "rbf_gamma_median_heuristic", "value": gamma},
        ]
    )
    if not identity_summary.empty:
        summary = pd.concat([summary, identity_summary], ignore_index=True)
    summary.to_csv(out_dir / f"{prefix}_summary.tsv", sep="\t", index=False)
    if identity_dir is not None:
        identity_status.to_csv(
            out_dir / f"{prefix}_identity_recovery_status.tsv", sep="\t", index=False
        )
        identity_summary.to_csv(
            out_dir / f"{prefix}_identity_recovery_summary.tsv", sep="\t", index=False
        )
        baseline_comparison.to_csv(
            out_dir / f"{prefix}_baseline_kernel_comparison.tsv", sep="\t", index=False
        )
    registry = pd.DataFrame(
        [
            {
                "kernel": f"{prefix}_LINEAR",
                "biological_role": f"{defaults['role']}_linear_genomic_relationship",
                "kernel_path": portable_output_path(linear_path, root),
                "order_path": portable_output_path(order_path, root),
                "source_id_col": "sample_id",
                "eligible_traits": "*",
                "enabled_default": False,
                "interaction_enabled": True,
                "rank": min(128, len(gids)),
                "minimum_ledger_coverage": 0.001,
            },
            {
                "kernel": f"{prefix}_RBF",
                "biological_role": f"{defaults['role']}_gaussian_RBF",
                "kernel_path": portable_output_path(rbf_path, root),
                "order_path": portable_output_path(order_path, root),
                "source_id_col": "sample_id",
                "eligible_traits": "*",
                "enabled_default": False,
                "interaction_enabled": True,
                "rank": min(128, len(gids)),
                "minimum_ledger_coverage": 0.001,
            },
        ]
    )
    registry.to_csv(out_dir / f"{prefix}_registry_fragment.tsv", sep="\t", index=False)
    if identity_dir is not None:
        output_artifact_hashes = {
            path.name: sha256_file(path)
            for path in sorted(out_dir.glob(f"{prefix}_*"))
            if path.is_file()
            and path.name
            not in {
                f"{prefix}_identity_recovery_provenance.json",
                f"{prefix}_artifacts.sha256",
            }
        }
        provenance = {
            "status": "PASS",
            "selection_data": "accepted_marker_identity_and_genotype_calls_only",
            "identity_adjudication_dir": str(identity_dir),
            "identity_panels": sorted(set(args.identity_panel)),
            "identity_input_hashes": identity_input_hashes,
            "marker_matrix": str(matrix_paths[0]),
            "marker_matrix_sha256": expected_matrix_sha256,
            "baseline_kernel": str(baseline_kernel_path),
            "baseline_kernel_sha256": sha256_file(baseline_kernel_path),
            "baseline_order": str(baseline_order_path),
            "baseline_order_sha256": sha256_file(baseline_order_path),
            "certified_baseline_artifacts_overwritten": False,
            "candidate_kernel_enabled_default": False,
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "output_artifact_hashes": output_artifact_hashes,
            "phenotype_values_read": False,
            "outer_test_metrics_read": False,
            "final_holdout_outcomes_read": False,
        }
        provenance_path = out_dir / f"{prefix}_identity_recovery_provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        checksum_path = out_dir / f"{prefix}_artifacts.sha256"
        checksum_artifacts = sorted(
            path
            for path in out_dir.glob(f"{prefix}_*")
            if path.is_file() and path != checksum_path
        )
        checksum_path.write_text(
            "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_artifacts),
            encoding="utf-8",
        )
        print(f"Artifact checksum manifest: {checksum_path}")
    print(summary.to_string(index=False))
    print(certification.to_string(index=False))
    if not baseline_comparison.empty:
        print("\n=== BASELINE KERNEL COMPARISON ===")
        print(baseline_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
