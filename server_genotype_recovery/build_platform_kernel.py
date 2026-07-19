from __future__ import annotations

import argparse
import csv
import gzip
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
) -> tuple[np.ndarray, list[str], list[str], list[str], list[str]]:
    """Stream a marker-by-sample text matrix and retain only resolved trial samples."""
    with path.open("rb") as handle:
        header_line = handle.readline()
        while header_line and not header_line.startswith(b"MarkerID\t"):
            header_line = handle.readline()
        if not header_line:
            raise SystemExit(f"Could not find MarkerID header in {path}")

        header = header_line.decode("utf-8-sig", errors="replace").rstrip("\r\n").split("\t")
        selected: list[tuple[int, str, str]] = []
        for field_index, raw_sample in enumerate(header[1:], start=1):
            sample = normalize_identifier(raw_sample)
            candidates = lookup.get(sample.upper(), set())
            if len(candidates) == 1:
                selected.append((field_index, next(iter(candidates)), sample))
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
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_catalog_path = (root / args.canonical_catalog).resolve()
    if not canonical_catalog_path.is_file() or canonical_catalog_path.stat().st_size == 0:
        raise SystemExit(f"Canonical genotype catalog is missing or empty: {canonical_catalog_path}")
    canonical_ids, lookup = target_sample_lookup(root=root, canonical_catalog_path=canonical_catalog_path)

    if defaults["format"] == "sample_by_marker":
        matrix, gids, source_samples, marker_ids, allele_values = parse_sample_by_marker(matrix_paths[0], lookup)
    elif defaults["format"] == "marker_by_sample":
        matrix, gids, source_samples, marker_ids, allele_values = parse_marker_by_sample(matrix_paths[0], lookup)
    elif defaults["format"] == "iwyp":
        matrix, gids, source_samples, marker_ids, allele_values = parse_iwyp(matrix_paths[0], canonical_ids)
    else:
        matrix, gids, source_samples, marker_ids, allele_values = parse_dartag_numeric(
            matrix_paths, canonical_ids
        )
    raw_shape = matrix.shape
    duplicate_concordance = duplicate_call_concordance(matrix, gids, source_samples)
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
    summary.to_csv(out_dir / f"{prefix}_summary.tsv", sep="\t", index=False)
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
    print(summary.to_string(index=False))
    print(certification.to_string(index=False))


if __name__ == "__main__":
    main()
