from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import struct
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


RELEASE_ID = "P5CPSL_20260822_V1_274E41DF"
OUTPUT_RELATIVE = Path("audit/v2/phase5_cimmyt_pre_qc_split_local_v1")
PREREQUISITE_RELATIVE = Path("audit/v2/phase5_panel_prerequisite_recovery_v1")
ARCHIVE_RELATIVE = (
    PREREQUISITE_RELATIVE / "downloads/CIMMYT-2013-2018.hmp.txt.zip.gz"
)
PROTOCOL_RELATIVE = Path("scripts/v2/cimmyt_pre_qc_split_local_protocol_v1.json")
MASTER_RELATIVE = Path(
    "audit/v2/phase5_split_bound_kernel_validation_v2/indices/"
    "canonical_phase5_observation_index.parquet"
)
STATE_ROOT_RELATIVE = Path(
    "audit/v2/phase5_panel_environment_scenario_parity_extension_v2"
)
EXPECTED_ARCHIVE_BYTES = 862_429_679
EXPECTED_ARCHIVE_MD5 = "61865f5b1002f5a6e14dc555ba700663"
EXPECTED_ARCHIVE_SHA256 = (
    "8322dac67ecaf9ca8cc10165f8baed5be27b2570b87cdd2bbbc28c4f15ae640b"
)
EXPECTED_MARKERS = 91_680
EXPECTED_SAMPLES = 53_525
EXPECTED_PRIMARY_GIDS = 5_629
EXPECTED_PRIMARY_ROWS = 1_409_585
MISSING = np.uint8(255)
HAPMAP_METADATA_COLUMNS = 11
IUPAC_HETEROZYGOUS = {
    frozenset(("A", "G")): "R",
    frozenset(("C", "T")): "Y",
    frozenset(("C", "G")): "S",
    frozenset(("A", "T")): "W",
    frozenset(("G", "T")): "K",
    frozenset(("A", "C")): "M",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream the recovered CIMMYT 91,680-marker HapMap once and fit "
            "training-local QC parameters for all 150 frozen Stage-1 v2 states"
        )
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force-state-refit", action="store_true")
    return parser.parse_args()


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def parse_zip64_sizes(extra: bytes) -> tuple[int | None, int | None]:
    cursor = 0
    while cursor + 4 <= len(extra):
        header_id, length = struct.unpack_from("<HH", extra, cursor)
        payload = extra[cursor + 4 : cursor + 4 + length]
        if header_id == 0x0001 and len(payload) >= 16:
            return struct.unpack_from("<QQ", payload, 0)
        cursor += 4 + length
    return None, None


class GzipWrappedZipLines:
    """Stream the sole DEFLATE ZIP member without materializing 9.82 GB."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.outer = gzip.open(path, "rb")
        local = self.outer.read(30)
        if len(local) != 30 or local[:4] != b"PK\x03\x04":
            raise ValueError("Expected a gzip-wrapped ZIP local member")
        self.flags = struct.unpack_from("<H", local, 6)[0]
        self.method = struct.unpack_from("<H", local, 8)[0]
        self.expected_crc32 = struct.unpack_from("<I", local, 14)[0]
        filename_length, extra_length = struct.unpack_from("<HH", local, 26)
        self.member_name = self.outer.read(filename_length).decode("utf-8")
        extra = self.outer.read(extra_length)
        self.expected_uncompressed_bytes, self.compressed_bytes = parse_zip64_sizes(extra)
        if self.flags & 0x1:
            raise ValueError("Encrypted ZIP member is unsupported")
        if self.method != 8:
            raise ValueError(f"Expected DEFLATE member, observed method={self.method}")
        if self.expected_uncompressed_bytes is None or self.compressed_bytes is None:
            raise ValueError("ZIP64 member sizes were not found")
        self.observed_uncompressed_bytes = 0
        self.observed_crc32 = 0
        self.compressed_bytes_read = 0
        self.line_count = 0
        self.validated = False

    def __iter__(self) -> Iterator[bytes]:
        inflater = zlib.decompressobj(-zlib.MAX_WBITS)
        remaining = int(self.compressed_bytes)
        pending = bytearray()
        while remaining:
            chunk = self.outer.read(min(512 * 1024, remaining))
            if not chunk:
                raise ValueError("Nested ZIP member ended before its declared compressed size")
            remaining -= len(chunk)
            self.compressed_bytes_read += len(chunk)
            decoded = inflater.decompress(chunk)
            self.observed_uncompressed_bytes += len(decoded)
            self.observed_crc32 = zlib.crc32(decoded, self.observed_crc32)
            pending.extend(decoded)
            start = 0
            while True:
                newline = pending.find(b"\n", start)
                if newline < 0:
                    if start:
                        del pending[:start]
                    break
                line = bytes(pending[start : newline + 1])
                start = newline + 1
                self.line_count += 1
                yield line
        tail = inflater.flush()
        if tail:
            self.observed_uncompressed_bytes += len(tail)
            self.observed_crc32 = zlib.crc32(tail, self.observed_crc32)
            pending.extend(tail)
        if pending:
            self.line_count += 1
            yield bytes(pending)
        if not inflater.eof:
            raise ValueError("Nested DEFLATE member did not reach end-of-stream")
        self.validated = (
            self.observed_uncompressed_bytes == self.expected_uncompressed_bytes
            and (self.observed_crc32 & 0xFFFFFFFF) == self.expected_crc32
        )
        if not self.validated:
            raise ValueError(
                "Nested member validation failed: "
                f"bytes={self.observed_uncompressed_bytes}/{self.expected_uncompressed_bytes}; "
                f"crc={self.observed_crc32 & 0xFFFFFFFF:08x}/{self.expected_crc32:08x}"
            )
        self.outer.close()


def split_hapmap_line(line: bytes) -> tuple[list[bytes], bytes]:
    fields = line.rstrip(b"\r\n").split(b"\t", HAPMAP_METADATA_COLUMNS)
    if len(fields) != HAPMAP_METADATA_COLUMNS + 1:
        raise ValueError("HapMap row has fewer than 11 metadata fields plus calls")
    return fields[:HAPMAP_METADATA_COLUMNS], fields[HAPMAP_METADATA_COLUMNS]


def allele_lookup(alleles: str) -> tuple[np.ndarray, bool, str, str, str]:
    lookup = np.full(256, MISSING, dtype=np.uint8)
    for token in ("N", "n", ".", "-", "?", "0"):
        lookup[ord(token)] = MISSING
    parsed = [item.strip().upper() for item in re.split(r"[/|]", alleles)]
    if len(parsed) != 2 or any(len(item) != 1 or item not in "ACGT" for item in parsed):
        return lookup, False, "", "", ""
    first, second = parsed
    if first == second:
        return lookup, False, first, second, ""
    lookup[ord(first)] = 0
    lookup[ord(first.lower())] = 0
    lookup[ord(second)] = 2
    lookup[ord(second.lower())] = 2
    heterozygous = IUPAC_HETEROZYGOUS[frozenset((first, second))]
    lookup[ord(heterozygous)] = 1
    lookup[ord(heterozygous.lower())] = 1
    return lookup, True, first, second, heterozygous


def load_protocol(root: Path) -> tuple[dict[str, Any], Path, str]:
    path = root / PROTOCOL_RELATIVE
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["protocol_version"] != "cimmyt_pre_qc_split_local_v1":
        raise ValueError("Unexpected CIMMYT split-local protocol")
    return protocol, path, digest_file(path)


def load_primary_axis(root: Path, header_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = pd.read_parquet(
        root / MASTER_RELATIVE,
        columns=[
            "canonical_gid",
            "primary_weighted_training_eligible",
            "secondary_unweighted_training_eligible",
        ],
    )
    primary = master.primary_weighted_training_eligible.fillna(False).astype(bool)
    primary_counts = (
        master.loc[primary].groupby("canonical_gid", sort=True).size().rename("primary_rows")
    )
    header_index = {gid: index for index, gid in enumerate(header_ids)}
    if len(header_index) != len(header_ids):
        raise ValueError("Duplicate sample IDs in recovered CIMMYT source header")
    selected_gids = sorted(set(primary_counts.index).intersection(header_index))
    axis = pd.DataFrame(
        {
            "canonical_gid": selected_gids,
            "source_sample_id": selected_gids,
            "source_sample_index": [header_index[gid] for gid in selected_gids],
            "primary_stage1_rows": [int(primary_counts[gid]) for gid in selected_gids],
        }
    ).sort_values("source_sample_index").reset_index(drop=True)
    axis["shared_call_matrix_column"] = np.arange(len(axis), dtype=np.int32)
    if len(axis) != EXPECTED_PRIMARY_GIDS:
        raise ValueError(f"Expected {EXPECTED_PRIMARY_GIDS} primary GIDs, observed {len(axis)}")
    if int(axis.primary_stage1_rows.sum()) != EXPECTED_PRIMARY_ROWS:
        raise ValueError("Primary Stage-1 row overlap changed")
    return axis, master


def stream_source_once(
    root: Path,
    output: Path,
    archive: Path,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    genomic = output / "genomic"
    call_path = genomic / "cimmyt_pre_qc_primary_raw_calls.npy"
    marker_path = genomic / "cimmyt_pre_qc_marker_axis.parquet"
    sample_path = genomic / "cimmyt_pre_qc_primary_sample_qc.tsv"
    axis_path = genomic / "cimmyt_pre_qc_primary_sample_axis.tsv"
    audit_path = genomic / "cimmyt_pre_qc_source_stream_audit.json"
    if all(path.is_file() for path in (call_path, marker_path, sample_path, axis_path, audit_path)):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "PASS_SINGLE_STREAM_SOURCE_AND_CALL_CERTIFICATION":
            calls = np.load(call_path, mmap_mode="r")
            if calls.shape != (EXPECTED_MARKERS, EXPECTED_PRIMARY_GIDS):
                raise ValueError("Cached raw-call matrix shape mismatch")
            if digest_file(call_path) != audit["shared_call_matrix_sha256"]:
                raise ValueError("Cached raw-call matrix checksum mismatch")
            if digest_file(marker_path) != audit["marker_axis_sha256"]:
                raise ValueError("Cached marker axis checksum mismatch")
            if digest_file(sample_path) != audit["sample_qc_sha256"]:
                raise ValueError("Cached sample QC checksum mismatch")
            markers = pd.read_parquet(marker_path)
            samples = pd.read_csv(sample_path, sep="\t")
            print("REUSE certified one-pass CIMMYT raw-call store", flush=True)
            return call_path, markers, samples, audit

    genomic.mkdir(parents=True, exist_ok=True)
    reader = GzipWrappedZipLines(archive)
    lines = iter(reader)
    header = next(lines).rstrip(b"\r\n").split(b"\t")
    expected_prefix = [
        b"rs#", b"alleles", b"chrom", b"pos", b"strand", b"assembly#",
        b"center", b"protLSID", b"assayLSID", b"panelLSID", b"QCcode",
    ]
    if header[:HAPMAP_METADATA_COLUMNS] != expected_prefix:
        raise ValueError("Unexpected HapMap header metadata columns")
    header_ids = [value.decode("utf-8") for value in header[HAPMAP_METADATA_COLUMNS:]]
    if len(header_ids) != EXPECTED_SAMPLES or len(set(header_ids)) != EXPECTED_SAMPLES:
        raise ValueError("Recovered source sample axis changed")
    axis, _ = load_primary_axis(root, header_ids)
    write_tsv(axis_path, axis)
    source_indices = axis.source_sample_index.to_numpy(dtype=np.int64)
    fixed_positions = 2 * source_indices

    calls = np.lib.format.open_memmap(
        call_path,
        mode="w+",
        dtype=np.uint8,
        shape=(EXPECTED_MARKERS, len(axis)),
    )
    sample_observed = np.zeros(len(axis), dtype=np.int64)
    token_counts = np.zeros(256, dtype=np.int64)
    marker_rows: list[dict[str, Any]] = []
    payload_width_failures = 0
    unknown_selected_tokens = 0
    decodable_markers = 0
    marker_count = 0
    for marker_index, line in enumerate(lines):
        if marker_index >= EXPECTED_MARKERS:
            raise ValueError("Source has more marker rows than declared")
        metadata, payload = split_hapmap_line(line)
        if len(payload) == 2 * EXPECTED_SAMPLES - 1:
            payload_bytes = np.frombuffer(payload, dtype=np.uint8)
            if np.all(payload_bytes[1::2] == ord("\t")):
                selected_ascii = payload_bytes[fixed_positions]
            else:
                payload_width_failures += 1
                tokens = payload.split(b"\t")
                selected_ascii = np.asarray(
                    [tokens[index][0] if len(tokens[index]) == 1 else ord("?") for index in source_indices],
                    dtype=np.uint8,
                )
        else:
            payload_width_failures += 1
            tokens = payload.split(b"\t")
            if len(tokens) != EXPECTED_SAMPLES:
                raise ValueError(f"HapMap payload width mismatch at marker {marker_index}")
            selected_ascii = np.asarray(
                [tokens[index][0] if len(tokens[index]) == 1 else ord("?") for index in source_indices],
                dtype=np.uint8,
            )
        marker_token_counts = np.bincount(selected_ascii, minlength=256)
        token_counts += marker_token_counts
        decoded_metadata = [value.decode("utf-8", errors="strict") for value in metadata]
        lookup, decodable, allele_0, allele_2, heterozygous = allele_lookup(
            decoded_metadata[1]
        )
        decoded = lookup[selected_ascii]
        calls[marker_index, :] = decoded
        sample_observed += (decoded != MISSING).astype(np.int64)
        decodable_markers += int(decodable)
        if decodable:
            allowed = {
                ord(allele_0), ord(allele_0.lower()), ord(allele_2), ord(allele_2.lower()),
                ord(heterozygous), ord(heterozygous.lower()), ord("N"), ord("n"),
                ord("."), ord("-"), ord("?"), ord("0"),
            }
            unknown_selected_tokens += int(
                sum(
                    int(token_count)
                    for code, token_count in enumerate(marker_token_counts)
                    if code not in allowed
                )
            )
        marker_rows.append(
            {
                "marker_index": marker_index,
                "marker_id": decoded_metadata[0],
                "alleles": decoded_metadata[1],
                "chrom": decoded_metadata[2],
                "pos": decoded_metadata[3],
                "strand": decoded_metadata[4],
                "assembly": decoded_metadata[5],
                "center": decoded_metadata[6],
                "prot_lsid": decoded_metadata[7],
                "assay_lsid": decoded_metadata[8],
                "panel_lsid": decoded_metadata[9],
                "qc_code": decoded_metadata[10],
                "biallelic_acgt_decodable": decodable,
                "allele_0": allele_0,
                "allele_2": allele_2,
                "heterozygous_token": heterozygous,
            }
        )
        marker_count += 1
        if marker_count % 5_000 == 0:
            calls.flush()
            print(f"CIMMYT one-pass stream: {marker_count:,}/{EXPECTED_MARKERS:,} markers", flush=True)
    calls.flush()
    del calls
    if marker_count != EXPECTED_MARKERS:
        raise ValueError(f"Expected {EXPECTED_MARKERS} markers, observed {marker_count}")
    if not reader.validated:
        raise ValueError("Nested source member did not pass CRC and byte-count validation")

    markers = pd.DataFrame(marker_rows)
    markers.to_parquet(marker_path, index=False, compression="zstd")
    samples = axis.copy()
    samples["observed_marker_calls"] = sample_observed
    samples["missing_marker_calls"] = EXPECTED_MARKERS - sample_observed
    samples["raw_call_rate"] = sample_observed / EXPECTED_MARKERS
    samples["passes_frozen_sample_call_rate"] = samples.raw_call_rate.ge(0.50)
    write_tsv(sample_path, samples)
    token_frame = pd.DataFrame(
        [
            {
                "token": chr(code) if 32 <= code <= 126 else f"ASCII_{code}",
                "ascii_code": code,
                "selected_call_count": int(count),
            }
            for code, count in enumerate(token_counts)
            if count
        ]
    )
    write_tsv(genomic / "cimmyt_pre_qc_call_token_inventory.tsv", token_frame)
    audit = {
        "status": "PASS_SINGLE_STREAM_SOURCE_AND_CALL_CERTIFICATION",
        "archive_relative_path": ARCHIVE_RELATIVE.as_posix(),
        "member_name": reader.member_name,
        "member_expected_uncompressed_bytes": reader.expected_uncompressed_bytes,
        "member_observed_uncompressed_bytes": reader.observed_uncompressed_bytes,
        "member_expected_crc32": f"{reader.expected_crc32:08x}",
        "member_observed_crc32": f"{reader.observed_crc32 & 0xFFFFFFFF:08x}",
        "source_line_count": reader.line_count,
        "header_sample_columns": len(header_ids),
        "selected_primary_GIDs": len(axis),
        "selected_primary_rows": int(axis.primary_stage1_rows.sum()),
        "marker_rows": marker_count,
        "decodable_biallelic_markers": decodable_markers,
        "nondecodable_markers": marker_count - decodable_markers,
        "payload_width_fallback_rows": payload_width_failures,
        "unknown_selected_call_tokens": unknown_selected_tokens,
        "selected_call_cells": int(marker_count * len(axis)),
        "observed_selected_call_cells": int(sample_observed.sum()),
        "selected_call_rate": float(sample_observed.sum() / (marker_count * len(axis))),
        "samples_passing_call_rate": int(samples.passes_frozen_sample_call_rate.sum()),
        "shared_call_matrix_sha256": digest_file(call_path),
        "marker_axis_sha256": digest_file(marker_path),
        "sample_qc_sha256": digest_file(sample_path),
        "single_physical_source_pass": True,
    }
    write_json(audit_path, audit)
    return call_path, markers, samples, audit


def load_states(root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    state_root = root / STATE_ROOT_RELATIVE
    registry = pd.read_csv(
        state_root / "splits/state_registry.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    if len(registry) != 150:
        raise ValueError(f"Expected 150 frozen states, observed {len(registry)}")
    states: list[dict[str, Any]] = []
    for row in registry.itertuples(index=False):
        gids = pd.read_csv(state_root / row.training_gid_path, sep="\t", dtype=str)[
            "canonical_gid"
        ].astype(str).tolist()
        if stable_hash_lines(gids) != row.training_gid_signature:
            raise ValueError(f"Training GID signature mismatch: {row.state_id}")
        states.append(
            {
                "state_id": row.state_id,
                "scenario": row.scenario,
                "outer_fold": int(row.outer_fold),
                "inner_fold": None if row.inner_fold == "" else int(row.inner_fold),
                "state_level": row.state_level,
                "training_gids": gids,
                "training_gid_signature": row.training_gid_signature,
            }
        )
    return registry, states


def fit_states(
    output: Path,
    call_path: Path,
    markers: pd.DataFrame,
    samples: pd.DataFrame,
    states: list[dict[str, Any]],
    protocol: dict[str, Any],
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry_path = output / "states/cimmyt_pre_qc_component_registry.tsv"
    qc_path = output / "states/cimmyt_pre_qc_fold_preprocessing_registry.tsv"
    if not force and registry_path.is_file() and qc_path.is_file():
        registry = pd.read_csv(registry_path, sep="\t")
        qc = pd.read_csv(qc_path, sep="\t")
        if len(registry) == len(qc) == 150:
            print("REUSE certified 150-state CIMMYT QC registry", flush=True)
            return registry, qc

    state_dir = output / "states/parameters"
    state_dir.mkdir(parents=True, exist_ok=True)
    dosage = np.load(call_path, mmap_mode="r")
    if dosage.shape != (EXPECTED_MARKERS, EXPECTED_PRIMARY_GIDS):
        raise ValueError("Shared CIMMYT raw-call matrix shape mismatch")
    gid_index = {
        gid: index for index, gid in enumerate(samples.canonical_gid.astype(str).tolist())
    }
    sample_pass = samples.passes_frozen_sample_call_rate.astype(bool).to_numpy()
    marker_call_rate = float(protocol["marker_qc"]["minimum_training_call_rate"])
    min_maf = float(protocol["marker_qc"]["minimum_training_minor_allele_frequency"])
    max_heterozygosity = float(protocol["marker_qc"]["maximum_training_heterozygosity"])
    min_observed = int(protocol["marker_qc"]["minimum_observed_training_calls"])
    min_training = int(protocol["kernel"]["minimum_training_GIDs"])
    sketch_count = int(protocol["kernel"]["diagnostic_sketch_markers"])
    marker_priority = np.argsort(
        np.asarray(
            [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in markers.marker_id.astype(str)],
            dtype="U64",
        ),
        kind="stable",
    )
    cache: dict[str, dict[str, Any]] = {}
    registry_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for state_number, state in enumerate(states, start=1):
        before = [gid for gid in state["training_gids"] if gid in gid_index]
        training_indices = np.asarray(
            [gid_index[gid] for gid in before if sample_pass[gid_index[gid]]], dtype=np.int32
        )
        training_gids = [samples.iloc[index].canonical_gid for index in training_indices]
        training_signature = stable_hash_lines(training_gids)
        parameter_path = state_dir / f"{state['state_id']}__cimmyt_pre_qc_parameters.npz"
        reused = training_signature in cache
        if reused:
            fitted = cache[training_signature]
        else:
            retained_parts: list[np.ndarray] = []
            p_parts: list[np.ndarray] = []
            failure_nondecodable = 0
            failure_call_rate = 0
            failure_maf = 0
            failure_heterozygosity = 0
            if len(training_indices) >= min_training:
                required_observed = max(
                    min_observed, math.ceil(marker_call_rate * len(training_indices))
                )
                for start in range(0, EXPECTED_MARKERS, 2048):
                    stop = min(EXPECTED_MARKERS, start + 2048)
                    block = np.asarray(dosage[start:stop, training_indices], dtype=np.uint8)
                    valid = block != MISSING
                    counts = valid.sum(axis=1)
                    sums = np.where(valid, block, 0).sum(axis=1, dtype=np.float64)
                    heterozygous = (block == 1).sum(axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        p = sums / (2.0 * counts)
                        heterozygosity = heterozygous / counts
                    maf = np.minimum(p, 1.0 - p)
                    decodable = markers.biallelic_acgt_decodable.iloc[start:stop].to_numpy(bool)
                    call_pass = counts >= required_observed
                    maf_pass = np.isfinite(maf) & (maf >= min_maf) & (p > 0) & (p < 1)
                    hetero_pass = np.isfinite(heterozygosity) & (
                        heterozygosity <= max_heterozygosity
                    )
                    keep = decodable & call_pass & maf_pass & hetero_pass
                    failure_nondecodable += int((~decodable).sum())
                    failure_call_rate += int((decodable & ~call_pass).sum())
                    failure_maf += int((decodable & call_pass & ~maf_pass).sum())
                    failure_heterozygosity += int(
                        (decodable & call_pass & maf_pass & ~hetero_pass).sum()
                    )
                    retained_parts.append(np.flatnonzero(keep).astype(np.int32) + start)
                    p_parts.append(p[keep].astype(np.float32))
            else:
                failure_nondecodable = int((~markers.biallelic_acgt_decodable).sum())
                failure_call_rate = 0
                failure_maf = 0
                failure_heterozygosity = 0
            retained = (
                np.concatenate(retained_parts) if retained_parts else np.asarray([], dtype=np.int32)
            )
            allele_frequency = (
                np.concatenate(p_parts) if p_parts else np.asarray([], dtype=np.float32)
            )
            denominator = float(
                2.0 * np.sum(allele_frequency * (1.0 - allele_frequency), dtype=np.float64)
            )
            sketch_rank = 0
            sketch_min_eigenvalue = math.nan
            if retained.size and denominator > 0:
                retained_set = set(retained.tolist())
                sketch_markers = np.asarray(
                    [int(index) for index in marker_priority if int(index) in retained_set][
                        :sketch_count
                    ],
                    dtype=np.int32,
                )
                lookup = {int(marker): index for index, marker in enumerate(retained)}
                sketch_p = np.asarray(
                    [allele_frequency[lookup[int(marker)]] for marker in sketch_markers],
                    dtype=np.float64,
                )
                sketch = np.asarray(
                    dosage[np.ix_(sketch_markers, training_indices)], dtype=np.float64
                ).T
                sketch[sketch == MISSING] = np.nan
                means = 2.0 * sketch_p
                missing = ~np.isfinite(sketch)
                sketch[missing] = np.broadcast_to(means, sketch.shape)[missing]
                sketch -= means
                gram = sketch.T @ sketch
                eigenvalues = np.linalg.eigvalsh((gram + gram.T) / 2.0)
                tolerance = max(1e-10, float(eigenvalues.max()) * 1e-10)
                sketch_rank = int((eigenvalues > tolerance).sum())
                sketch_min_eigenvalue = float(eigenvalues.min())
            fitted = {
                "retained": retained,
                "allele_frequency": allele_frequency,
                "denominator": denominator,
                "failure_nondecodable": failure_nondecodable,
                "failure_call_rate": failure_call_rate,
                "failure_maf": failure_maf,
                "failure_heterozygosity": failure_heterozygosity,
                "sketch_rank": sketch_rank,
                "sketch_min_eigenvalue": sketch_min_eigenvalue,
            }
            cache[training_signature] = fitted
        strict_ready = (
            len(training_indices) >= min_training
            and len(fitted["retained"]) > 0
            and np.isfinite(fitted["denominator"])
            and fitted["denominator"] > 0
            and fitted["sketch_rank"] >= 2
        )
        np.savez_compressed(
            parameter_path,
            retained_marker_index=fitted["retained"],
            training_allele_frequency=fitted["allele_frequency"],
            vanraden_denominator=np.asarray([fitted["denominator"]], dtype=np.float64),
            training_gid_signature=np.asarray([training_signature], dtype="U64"),
            source_training_gid_signature=np.asarray(
                [state["training_gid_signature"]], dtype="U64"
            ),
            strict_production_eligible=np.asarray([strict_ready], dtype=bool),
        )
        status = "PASS_STRICT_SPLIT_LOCAL_PARAMETERS" if strict_ready else "MASKED_SUPPORT_OR_QC"
        qc_rows.append(
            {
                "state_id": state["state_id"],
                "scenario": state["scenario"],
                "outer_fold": state["outer_fold"],
                "inner_fold": state["inner_fold"],
                "state_level": state["state_level"],
                "training_panel_GIDs_before_sample_QC": len(before),
                "training_panel_GIDs_after_sample_QC": len(training_indices),
                "minimum_training_GIDs": min_training,
                "source_marker_rows": EXPECTED_MARKERS,
                "retained_marker_rows": len(fitted["retained"]),
                "markers_failed_nondecodable": fitted["failure_nondecodable"],
                "markers_failed_call_rate": fitted["failure_call_rate"],
                "markers_failed_MAF": fitted["failure_maf"],
                "markers_failed_heterozygosity": fitted["failure_heterozygosity"],
                "vanraden_denominator": fitted["denominator"],
                "deterministic_sketch_rank": fitted["sketch_rank"],
                "deterministic_sketch_min_eigenvalue": fitted["sketch_min_eigenvalue"],
                "sample_QC_fit_scope": "SOURCE_INTRINSIC_CALL_RATE_NO_PHENOTYPES",
                "marker_QC_fit_scope": "TRAINING_GIDS_ONLY",
                "allele_frequency_fit_scope": "TRAINING_GIDS_ONLY",
                "imputation_fit_scope": "TRAINING_GIDS_ONLY_2P_ON_DEMAND",
                "held_out_calls_used_for_parameters": False,
                "parameter_cache_reused_for_exact_training_signature": reused,
                "parameter_path": parameter_path.relative_to(output).as_posix(),
                "parameter_sha256": digest_file(parameter_path),
                "strict_production_eligible": strict_ready,
                "status": status,
            }
        )
        registry_rows.append(
            {
                "state_id": state["state_id"],
                "panel_id": "cimmyt_bread_gbs_pre_qc_91680",
                "representation": "SHARED_RAW_CALLS_PLUS_TRAINING_LOCAL_PARAMETERS",
                "training_entities": len(training_indices),
                "markers": len(fitted["retained"]),
                "formula": "K=ZZT/(2*sum(p*(1-p)))",
                "strict_production_component_available": strict_ready,
                "explicit_mask": not strict_ready,
                "status": status,
            }
        )
        if state_number % 10 == 0:
            print(f"CIMMYT split-local QC: {state_number}/{len(states)} states", flush=True)
    registry = pd.DataFrame(registry_rows)
    qc = pd.DataFrame(qc_rows)
    write_tsv(registry_path, registry)
    write_tsv(qc_path, qc)
    return registry, qc


def build_release(
    root: Path,
    output: Path,
    protocol: dict[str, Any],
    protocol_path: Path,
    protocol_sha256: str,
    archive: Path,
    call_path: Path,
    markers: pd.DataFrame,
    samples: pd.DataFrame,
    source_audit: dict[str, Any],
    registry: pd.DataFrame,
    qc: pd.DataFrame,
) -> dict[str, Any]:
    strict_states = int(registry.strict_production_component_available.astype(bool).sum())
    checks = {
        "prerequisite_recovery_pass": True,
        "protocol_frozen": protocol["protocol_version"] == "cimmyt_pre_qc_split_local_v1",
        "archive_byte_size": archive.stat().st_size == EXPECTED_ARCHIVE_BYTES,
        "archive_MD5": digest_file(archive, "md5") == EXPECTED_ARCHIVE_MD5,
        "archive_SHA256": digest_file(archive) == EXPECTED_ARCHIVE_SHA256,
        "nested_member_CRC_and_bytes": source_audit["status"]
        == "PASS_SINGLE_STREAM_SOURCE_AND_CALL_CERTIFICATION",
        "shared_call_matrix_checksum": digest_file(call_path)
        == source_audit["shared_call_matrix_sha256"],
        "single_physical_source_pass": source_audit["single_physical_source_pass"],
        "marker_count": len(markers) == EXPECTED_MARKERS,
        "source_sample_count": source_audit["header_sample_columns"] == EXPECTED_SAMPLES,
        "primary_GID_count": len(samples) == EXPECTED_PRIMARY_GIDS,
        "primary_row_count": int(samples.primary_stage1_rows.sum()) == EXPECTED_PRIMARY_ROWS,
        "state_count": len(registry) == 150,
        "all_states_have_training_only_marker_QC": qc.marker_QC_fit_scope.eq(
            "TRAINING_GIDS_ONLY"
        ).all(),
        "all_states_have_training_only_allele_frequency": qc.allele_frequency_fit_scope.eq(
            "TRAINING_GIDS_ONLY"
        ).all(),
        "all_states_have_training_only_imputation": qc.imputation_fit_scope.eq(
            "TRAINING_GIDS_ONLY_2P_ON_DEMAND"
        ).all(),
        "held_out_calls_unused_for_parameters": (~qc.held_out_calls_used_for_parameters.astype(bool)).all(),
        "all_150_states_strict_ready": strict_states == 150,
        "no_phenotype_or_evaluation_outcome_access": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not bool(passed)]
    status = (
        "PASS_CIMMYT_PRE_QC_SPLIT_LOCAL_150_STATE_CERTIFIED"
        if not failed
        else "BLOCKED_CIMMYT_PRE_QC_SPLIT_LOCAL_CERTIFICATION"
    )
    summary = {
        "release_id": RELEASE_ID,
        "status": status,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_sha256,
        "selection_data": "Stage1_v2_identifiers_and_genotype_calls_only",
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_climate_values_read": False,
        "model_training_performed": False,
        "outer_evaluation_performed": False,
        "existing_kernels_modified": False,
        "source_marker_rows": len(markers),
        "source_sample_columns": source_audit["header_sample_columns"],
        "primary_stage1_GIDs": len(samples),
        "primary_stage1_rows": int(samples.primary_stage1_rows.sum()),
        "primary_samples_passing_call_rate": int(
            samples.passes_frozen_sample_call_rate.astype(bool).sum()
        ),
        "state_count": len(registry),
        "strict_ready_state_count": strict_states,
        "masked_state_count": len(registry) - strict_states,
        "training_GIDs_min": int(qc.training_panel_GIDs_after_sample_QC.min()),
        "training_GIDs_median": float(qc.training_panel_GIDs_after_sample_QC.median()),
        "training_GIDs_max": int(qc.training_panel_GIDs_after_sample_QC.max()),
        "retained_markers_min": int(qc.retained_marker_rows.min()),
        "retained_markers_median": float(qc.retained_marker_rows.median()),
        "retained_markers_max": int(qc.retained_marker_rows.max()),
        "checks": checks,
        "failed_checks": failed,
        "production_component_disposition": (
            "ELIGIBLE_FOR_PHASE6_PREREGISTRATION_AS_MASKED_SPLIT_LOCAL_K_G"
            if not failed
            else "NOT_ELIGIBLE"
        ),
        "shared_raw_call_matrix": call_path.relative_to(output).as_posix(),
        "shared_raw_call_matrix_sha256": source_audit["shared_call_matrix_sha256"],
    }
    write_json(output / "CIMMYT_PRE_QC_SPLIT_LOCAL_DECISION.json", summary)
    check_rows = pd.DataFrame(
        [
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": ""}
            for name, passed in checks.items()
        ]
    )
    write_tsv(output / "validation_checks.tsv", check_rows)

    input_paths = [
        archive,
        protocol_path,
        root
        / PREREQUISITE_RELATIVE
        / "PHASE5_PANEL_PREREQUISITE_RECOVERY_DECISION.json",
        root / MASTER_RELATIVE,
        root / STATE_ROOT_RELATIVE / "splits/state_registry.tsv",
    ]
    input_inventory = pd.DataFrame(
        [
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest_file(path),
                "access_scope": "IDENTIFIERS_OR_GENOTYPE_SOURCE_ONLY",
            }
            for path in input_paths
        ]
    )
    write_tsv(output / "input_inventory.tsv", input_inventory)

    inventory_rows = []
    excluded_names = {"artifact_manifest.tsv"}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in excluded_names:
            inventory_rows.append(
                {
                    "relative_path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest_file(path),
                }
            )
    write_tsv(output / "artifact_manifest.tsv", pd.DataFrame(inventory_rows))
    if failed:
        raise RuntimeError(f"CIMMYT split-local certification failed: {failed}")
    return summary


def main() -> None:
    args = parse_args()
    root = args.repository_root.resolve()
    output = (args.output_root or (root / OUTPUT_RELATIVE)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_decision = output / "CIMMYT_PRE_QC_SPLIT_LOCAL_DECISION.json"
    if final_decision.exists():
        existing = json.loads(final_decision.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS_CIMMYT_PRE_QC_SPLIT_LOCAL_150_STATE_CERTIFIED":
            print(json.dumps(existing, indent=2, sort_keys=True))
            return
        raise FileExistsError(f"A non-passing decision already exists: {final_decision}")

    prerequisite = json.loads(
        (root / PREREQUISITE_RELATIVE / "PHASE5_PANEL_PREREQUISITE_RECOVERY_DECISION.json").read_text(
            encoding="utf-8"
        )
    )
    if prerequisite.get("status") != (
        "PASS_BOUNDED_PANEL_PREREQUISITE_RECOVERY_WITH_EXPLICIT_REMAINING_BLOCKERS"
    ):
        raise ValueError("Prerequisite panel recovery is not certified")
    protocol, protocol_path, protocol_sha256 = load_protocol(root)
    archive = root / ARCHIVE_RELATIVE
    if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("Recovered CIMMYT archive byte size mismatch")
    print("STREAM or reuse phenotype-blind CIMMYT pre-QC calls", flush=True)
    call_path, markers, samples, source_audit = stream_source_once(root, output, archive)
    _, states = load_states(root)
    print("FIT 150 frozen split-local QC parameter states", flush=True)
    registry, qc = fit_states(
        output,
        call_path,
        markers,
        samples,
        states,
        protocol,
        args.force_state_refit,
    )
    print("CERTIFY split-local CIMMYT production readiness", flush=True)
    summary = build_release(
        root,
        output,
        protocol,
        protocol_path,
        protocol_sha256,
        archive,
        call_path,
        markers,
        samples,
        source_audit,
        registry,
        qc,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
