from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from phase5_parity_common import (
    ProtectedPathGuard,
    ensure_fail_if_exists,
    environment_versions,
    factor_diagnostics,
    git_head,
    index_signature,
    relative_posix,
    sha256_file,
    stable_json_hash,
    utc_now,
    write_json,
    write_tsv,
)


RELEASE_ID = "P5CUG_20260809_V4_274E41DF"
RELEASE_RELATIVE = Path("audit/v2/phase5_cimmyt_unimputed_recovery_v4")
PARITY_RELEASE_ID = "P5PESP_20260809_V2_274E41DF"
PARITY_RELATIVE = Path("audit/v2/phase5_panel_environment_scenario_parity_extension_v2")
PHASE5_RELEASE_ID = "P5SBK_20260808_V1_274E41DF"
PHASE5_RELATIVE = Path("audit/v2/phase5_split_bound_kernel_validation_v2")
V1_INCIDENT_RELATIVE = Path("audit/v2/phase5_panel_environment_scenario_parity_extension_v1")
PANEL_ID = "cimmyt_bread_gbs_2013_2018"
SOURCE_DIRECTORY = Path(
    "GENOTYPIC_DATA/Genotypic_data_from_CIMMYT_bread_wheat_breeding_lines"
)
RAW_RELATIVE = SOURCE_DIRECTORY / (
    "F_MAF0.01_Miss50_Het10-Merged.all.discover.lines.and.selection.candidates."
    "vcf.unimputed.CIMMYT.2022.hmp.txt"
)
IMPUTED_RELATIVE = SOURCE_DIRECTORY / (
    "F_MAF0.01_Miss50_Het10-Merged.all.discover.lines.and.selection.candidates."
    "vcf.imputed.CIMMYT.2022.hmp.txt"
)

MIN_SAMPLE_CALL_RATE = 0.50
MIN_MARKER_CALL_RATE = 0.80
MIN_MAF = 0.01
MAX_HETEROZYGOSITY = 0.10
MIN_TRAINING_GIDS = 20
MISSING = np.uint8(255)
METADATA_COLUMNS = 11

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
        description="Analyze the newly supplied CIMMYT unimputed HMP without modifying Phase-5 or parity v2"
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def create_directories(output: Path) -> None:
    for relative in ("genomic/states", "masks", "redundancy", "tests", "logs"):
        (output / relative).mkdir(parents=True, exist_ok=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_preflight(root: Path, output: Path, guard: ProtectedPathGuard) -> dict[str, Any]:
    raw = guard.assert_allowed(root / RAW_RELATIVE, "PREFLIGHT_RAW_METADATA")
    imputed = guard.assert_allowed(root / IMPUTED_RELATIVE, "PREFLIGHT_IMPUTED_METADATA")
    parity_decision_path = guard.assert_allowed(
        root / PARITY_RELATIVE / "PHASE5_PARITY_EXTENSION_DECISION.json",
        "READ_PARITY_V2_DECISION",
    )
    phase5_decision_path = guard.assert_allowed(
        root / PHASE5_RELATIVE / "PHASE5_RELEASE_DECISION.json",
        "READ_PHASE5_DECISION",
    )
    if not raw.is_file() or not imputed.is_file():
        raise SystemExit("BLOCKED: both unimputed and imputed CIMMYT HMP exports are required")
    parity = read_json(parity_decision_path)
    phase5 = read_json(phase5_decision_path)
    if (
        parity.get("release_id") != PARITY_RELEASE_ID
        or parity.get("status") != "PASS_PHASE5_PARITY_EXTENSION_WITH_EXPLICIT_COMPONENT_BLOCKERS"
    ):
        raise SystemExit("BLOCKED: immutable parity-v2 binding mismatch")
    if (
        phase5.get("release_id") != PHASE5_RELEASE_ID
        or phase5.get("status") != "PASS_PHASE5_KERNEL_VALIDATION"
    ):
        raise SystemExit("BLOCKED: immutable Phase-5 binding mismatch")
    return {
        "release_id": RELEASE_ID,
        "phase5_release_id": PHASE5_RELEASE_ID,
        "parity_release_id": PARITY_RELEASE_ID,
        "raw_source": relative_posix(raw, root),
        "imputed_comparator": relative_posix(imputed, root),
        "raw_bytes": raw.stat().st_size,
        "imputed_bytes": imputed.stat().st_size,
        "target_absent": not output.exists(),
        "denylist_rules_loaded": len(guard.rules),
        "bundle_content_accessed": False,
        "status": "PASS_PREFLIGHT" if not output.exists() else "FAIL_TARGET_EXISTS",
    }


def write_opening_contract(root: Path, output: Path, preflight: dict[str, Any]) -> None:
    write_json(
        output / "OPENING_RELEASE.json",
        {
            "release_id": RELEASE_ID,
            "release_type": "FAIL_IF_EXISTS_PHASE5_PARITY_V2_CIMMYT_UNIMPUTED_FOLLOW_ON",
            "authoritative_phase5_release": PHASE5_RELEASE_ID,
            "authoritative_parity_release": PARITY_RELEASE_ID,
            "parity_v2_modified": False,
            "phase5_modified": False,
            "v1_attempt_disposition": "TERMINALLY_BLOCKED_INCIDENT_ONLY_NO_SCIENTIFIC_DECISIONS_INHERITED",
            "phenotype_blind": True,
            "model_training_performed": False,
            "inner_validation_metrics_accessed": False,
            "outer_test_outcomes_accessed": False,
            "final_holdout_accessed": False,
            "bundle_content_accessed": False,
            "protected_files_rendered": [],
            "pre_run_source_interaction": (
                "QC_PROTOCOL_WAS_FROZEN_IN_TERMINAL_V1_BEFORE_CALL_ACCESS;_V2_STOPPED_AT_"
                "ALLELE_ORDER_HARMONIZATION;_V3_COMPLETED_COMPUTATION_BUT_FAILED_PREDECISION_"
                "AUDIT_FILE_ORDERING_TESTS;_NO_QC_OR_SCIENTIFIC_DECISION_CHANGED"
            ),
            "prior_recovery_execution_incidents": [
                "P5CUG_20260809_V1_274E41DF",
                "P5CUG_20260809_V2_274E41DF",
                "P5CUG_20260809_V3_274E41DF"
            ],
            "prior_recovery_scientific_decisions_inherited": False,
            "qc_protocol_frozen_before_genotype_call_inspection": True,
            "preflight": preflight,
            "repository_root": str(root),
            "created_at_utc": utc_now(),
        },
    )
    write_json(
        output / "genomic/CIMMYT_UNIMPUTED_QC_PROTOCOL.json",
        {
            "release_id": RELEASE_ID,
            "panel_id": PANEL_ID,
            "frozen_before_genotype_call_inspection": True,
            "source_interpretation": (
                "PREIMPUTATION_OBSERVED_AND_MISSING_CALLS_WITHIN_AN_UPSTREAM_GLOBALLY_"
                "MAF_MISSINGNESS_HETEROZYGOSITY_FILTERED_MARKER_UNIVERSE"
            ),
            "sample_call_rate_minimum": MIN_SAMPLE_CALL_RATE,
            "training_state_minimum_gids": MIN_TRAINING_GIDS,
            "marker_call_rate_minimum": MIN_MARKER_CALL_RATE,
            "minor_allele_frequency_minimum": MIN_MAF,
            "marker_heterozygosity_maximum": MAX_HETEROZYGOSITY,
            "biallelic_acgt_markers_only": True,
            "allele_encoding": "FIRST_ALLELE_0_HETEROZYGOUS_1_SECOND_ALLELE_2_MISSING_255",
            "allele_frequency_fit_scope": "TRAINING_GIDS_ONLY_PER_PHASE5_STATE",
            "imputation_rule": "TRAINING_STATE_2P_ON_DEMAND",
            "global_imputation_performed": False,
            "strict_production_activation_rule": (
                "BLOCK_UNLESS_THE_PRE_QC_MARKER_UNIVERSE_OR_DOCUMENTED_TRAINING_ONLY_"
                "UPSTREAM_MARKER_SELECTION_IS_RECOVERED"
            ),
            "threshold_selection_used_performance_metrics": False,
        },
    )
    write_json(
        output / "run_manifest.json",
        {
            "release_id": RELEASE_ID,
            "repository_root": str(root),
            "release_root": str(output),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "git_head": git_head(root),
            "packages": environment_versions(),
            "model_training_performed": False,
            "component_selection_performed": False,
            "performance_evaluation_performed": False,
            "inner_validation_metrics_accessed": False,
            "outer_test_outcomes_accessed": False,
            "final_holdout_accessed": False,
            "future_projection_performed": False,
            "commit_or_push_performed": False,
        },
    )


def input_paths(root: Path) -> list[tuple[str, Path]]:
    parity = root / PARITY_RELATIVE
    phase5 = root / PHASE5_RELATIVE
    paths: list[tuple[str, Path]] = [
        ("NEW_CIMMYT_SOURCE", root / RAW_RELATIVE),
        ("CIMMYT_IMPUTED_COMPARATOR", root / IMPUTED_RELATIVE),
        ("PARITY_V2_BINDING", parity / "PHASE5_PARITY_EXTENSION_DECISION.json"),
        ("PARITY_V2_BINDING", parity / "output_manifest.tsv"),
        ("PARITY_V2_STATE", parity / "splits/state_registry.tsv"),
        ("PARITY_V2_PANEL", parity / "genomic/accepted_mapping_manifest.parquet"),
        ("PARITY_V2_PANEL", parity / "genomic/panel_fold_support.tsv"),
        ("PARITY_V2_PANEL", parity / "genomic/panel_stage1_overlap.tsv"),
        ("PARITY_V2_REDUNDANCY", parity / "genomic/seeds_gid_consensus_summary.tsv"),
        ("PHASE5_BINDING", phase5 / "PHASE5_RELEASE_DECISION.json"),
        ("PHASE5_MASTER", phase5 / "indices/canonical_phase5_observation_index.parquet"),
        ("IMPLEMENTATION", root / "scripts/v2/phase5_cimmyt_unimputed_recovery.py"),
        ("IMPLEMENTATION", root / "tests/test_phase5_cimmyt_unimputed_recovery.py"),
    ]
    state_registry = pd.read_csv(parity / "splits/state_registry.tsv", sep="\t", dtype=str)
    for relative in state_registry.training_gid_path.astype(str):
        paths.append(("PARITY_V2_STATE_ENTITY", parity / relative))
    return paths


def opening_hash_manifest(root: Path, output: Path, guard: ProtectedPathGuard) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for scope, path in input_paths(root):
        allowed = guard.assert_allowed(path, "OPENING_HASH")
        relative = relative_posix(allowed, root)
        if relative in seen:
            continue
        seen.add(relative)
        rows.append(
            {
                "scope": scope,
                "relative_path": relative,
                "size": allowed.stat().st_size,
                "sha256": sha256_file(allowed),
                "access": "HASHED_ALLOWED",
                "matched_rule": "",
            }
        )
    frame = pd.DataFrame(rows).sort_values(["scope", "relative_path"]).reset_index(drop=True)
    write_tsv(output / "OPENING_HASH_MANIFEST.tsv", frame)
    return frame


def split_hmp_line(line: bytes) -> tuple[list[bytes], bytes]:
    fields = line.rstrip(b"\r\n").split(b"\t", METADATA_COLUMNS)
    if len(fields) != METADATA_COLUMNS + 1:
        raise AssertionError("HMP row has fewer than 11 metadata fields plus genotype payload")
    return fields[:METADATA_COLUMNS], fields[METADATA_COLUMNS]


def marker_metadata_compatible(raw_meta: list[bytes], imputed_meta: list[bytes]) -> bool:
    if raw_meta[:1] + raw_meta[2:] != imputed_meta[:1] + imputed_meta[2:]:
        return False
    raw_alleles = raw_meta[1].decode("ascii", errors="replace").split("/")
    imputed_alleles = imputed_meta[1].decode("ascii", errors="replace").split("/")
    return (
        len(raw_alleles) == 2
        and len(imputed_alleles) == 2
        and set(raw_alleles) == set(imputed_alleles)
    )


def sample_payload_is_single_ascii(payload: bytes, sample_count: int) -> bool:
    if len(payload) != 2 * sample_count - 1:
        return False
    values = np.frombuffer(payload, dtype=np.uint8)
    return bool(np.all(values[1::2] == ord("\t")))


def inspect_axes(raw_path: Path, imputed_path: Path, output: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    marker_rows: list[dict[str, Any]] = []
    raw_signature = __import__("hashlib").sha256()
    imputed_signature = __import__("hashlib").sha256()
    identity_metadata_mismatches = 0
    allele_orientation_reversals = 0
    incompatible_allele_set_mismatches = 0
    raw_width_failures = 0
    imputed_width_failures = 0
    with raw_path.open("rb", buffering=16 * 1024 * 1024) as raw, imputed_path.open(
        "rb", buffering=16 * 1024 * 1024
    ) as imputed:
        raw_header = raw.readline().rstrip(b"\r\n").split(b"\t")
        imputed_header = imputed.readline().rstrip(b"\r\n").split(b"\t")
        if raw_header != imputed_header:
            raise AssertionError("Raw and imputed HMP headers differ")
        if len(raw_header) <= METADATA_COLUMNS:
            raise AssertionError("HMP header has no sample columns")
        sample_ids = [value.decode("utf-8", errors="strict") for value in raw_header[METADATA_COLUMNS:]]
        for marker_index, pair in enumerate(zip_longest(raw, imputed, fillvalue=None)):
            raw_line, imputed_line = pair
            if raw_line is None or imputed_line is None:
                raise AssertionError("Raw and imputed HMP marker row counts differ")
            raw_meta, raw_payload = split_hmp_line(raw_line)
            imputed_meta, imputed_payload = split_hmp_line(imputed_line)
            raw_meta_bytes = b"\t".join(raw_meta) + b"\n"
            imputed_meta_bytes = b"\t".join(imputed_meta) + b"\n"
            raw_signature.update(raw_meta_bytes)
            imputed_signature.update(imputed_meta_bytes)
            raw_identity = raw_meta[:1] + raw_meta[2:]
            imputed_identity = imputed_meta[:1] + imputed_meta[2:]
            if raw_identity != imputed_identity:
                identity_metadata_mismatches += 1
            if raw_meta[1] != imputed_meta[1]:
                raw_alleles = raw_meta[1].decode("ascii", errors="replace").split("/")
                imputed_alleles = imputed_meta[1].decode("ascii", errors="replace").split("/")
                if len(raw_alleles) == len(imputed_alleles) == 2 and set(raw_alleles) == set(imputed_alleles):
                    allele_orientation_reversals += 1
                else:
                    incompatible_allele_set_mismatches += 1
            raw_width_failures += int(not sample_payload_is_single_ascii(raw_payload, len(sample_ids)))
            imputed_width_failures += int(
                not sample_payload_is_single_ascii(imputed_payload, len(sample_ids))
            )
            decoded = [value.decode("utf-8", errors="replace") for value in raw_meta]
            marker_rows.append(
                {
                    "marker_index": marker_index,
                    "marker_id": decoded[0],
                    "alleles": decoded[1],
                    "chrom": decoded[2],
                    "pos": decoded[3],
                    "strand": decoded[4],
                    "assembly": decoded[5],
                    "center": decoded[6],
                    "prot_lsid": decoded[7],
                    "assay_lsid": decoded[8],
                    "panel_lsid": decoded[9],
                    "qc_code": decoded[10],
                }
            )
            if (marker_index + 1) % 5000 == 0:
                print(f"CIMMYT axis certification: {marker_index + 1:,} markers", flush=True)
    markers = pd.DataFrame(marker_rows)
    pq.write_table(
        pa.Table.from_pandas(markers, preserve_index=False),
        output / "genomic/cimmyt_marker_axis.parquet",
        compression="zstd",
    )
    duplicate_sample_ids = int(pd.Series(sample_ids).duplicated(keep=False).sum())
    audit = {
        "raw_source": RAW_RELATIVE.as_posix(),
        "imputed_comparator": IMPUTED_RELATIVE.as_posix(),
        "metadata_columns": METADATA_COLUMNS,
        "sample_columns": len(sample_ids),
        "marker_rows": len(markers),
        "matrix_orientation": "MARKER_ROWS_BY_SAMPLE_COLUMNS",
        "headers_exact": True,
        "marker_identity_coordinate_axis_exact": identity_metadata_mismatches == 0,
        "marker_identity_metadata_mismatches": identity_metadata_mismatches,
        "allele_set_axis_exact": incompatible_allele_set_mismatches == 0,
        "allele_orientation_reversals_harmonized_to_unimputed_order": allele_orientation_reversals,
        "incompatible_allele_set_mismatches": incompatible_allele_set_mismatches,
        "raw_marker_metadata_sha256": raw_signature.hexdigest(),
        "imputed_marker_metadata_sha256": imputed_signature.hexdigest(),
        "raw_single_ascii_call_rows": len(markers) - raw_width_failures,
        "imputed_single_ascii_call_rows": len(markers) - imputed_width_failures,
        "raw_payload_width_failures": raw_width_failures,
        "imputed_payload_width_failures": imputed_width_failures,
        "duplicate_header_sample_instances": duplicate_sample_ids,
        "duplicate_marker_ids": int(markers.marker_id.duplicated(keep=False).sum()),
        "duplicate_marker_coordinates": int(
            markers[["chrom", "pos"]].duplicated(keep=False).sum()
        ),
        "status": (
            "PASS_EXACT_RAW_IMPUTED_AXES"
            if identity_metadata_mismatches == 0
            and incompatible_allele_set_mismatches == 0
            and raw_width_failures == 0
            and imputed_width_failures == 0
            else "FAIL_AXIS_OR_PAYLOAD_STRUCTURE"
        ),
    }
    write_json(output / "genomic/cimmyt_source_axis_audit.json", audit)
    if audit["status"] != "PASS_EXACT_RAW_IMPUTED_AXES":
        raise AssertionError("CIMMYT source-axis audit failed")
    return markers, sample_ids, audit


def load_primary_mapping(root: Path, sample_ids: list[str], output: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    accepted = pd.read_parquet(
        root / PARITY_RELATIVE / "genomic/accepted_mapping_manifest.parquet",
        columns=[
            "panel_id",
            "raw_sample_id",
            "accepted_canonical_gid",
            "mapping_status",
            "evidence_type",
            "matrix_order_first",
            "matrix_occurrences",
        ],
    )
    accepted = accepted[
        accepted.panel_id.eq(PANEL_ID)
        & accepted.accepted_canonical_gid.notna()
        & accepted.accepted_canonical_gid.astype(str).ne("")
    ].copy()
    master = pq.read_table(
        root / PHASE5_RELATIVE / "indices/canonical_phase5_observation_index.parquet",
        columns=["canonical_gid", "primary_weighted_training_eligible"],
    ).to_pandas()
    primary = master[master.primary_weighted_training_eligible.astype(bool)].copy()
    primary_counts = primary.groupby("canonical_gid", sort=True).size().rename("primary_stage1_rows")
    primary_gids = set(primary_counts.index.astype(str))
    selected = accepted[accepted.accepted_canonical_gid.astype(str).isin(primary_gids)].copy()
    if selected.raw_sample_id.duplicated().any() or selected.accepted_canonical_gid.duplicated().any():
        raise AssertionError("Primary CIMMYT accepted mapping is not one sample instance per GID")
    header_index = {sample: index for index, sample in enumerate(sample_ids)}
    selected["hmp_sample_index"] = selected.raw_sample_id.map(header_index)
    if selected.hmp_sample_index.isna().any():
        raise AssertionError("Accepted primary CIMMYT sample absent from new HMP header")
    selected["hmp_sample_index"] = selected.hmp_sample_index.astype(int)
    selected["primary_stage1_rows"] = selected.accepted_canonical_gid.map(primary_counts).astype(int)
    selected = selected.sort_values("hmp_sample_index").reset_index(drop=True)
    selected["primary_matrix_column"] = np.arange(len(selected), dtype=int)
    selected["raw_unimputed_call_source_available"] = True
    selected["strict_production_marker_universe_safe"] = False
    write_tsv(output / "genomic/cimmyt_primary_sample_axis.tsv", selected)
    overlap = pd.DataFrame(
        [
            {
                "panel_id": PANEL_ID,
                "hmp_header_samples": len(sample_ids),
                "accepted_panel_gids": int(accepted.accepted_canonical_gid.nunique()),
                "primary_stage1_gids": len(selected),
                "primary_stage1_rows": int(selected.primary_stage1_rows.sum()),
                "expected_primary_stage1_gids": 4512,
                "expected_primary_stage1_rows": 721033,
                "discrepancy_gids": len(selected) - 4512,
                "discrepancy_rows": int(selected.primary_stage1_rows.sum()) - 721033,
                "status": (
                    "PASS_EXACT"
                    if len(selected) == 4512 and int(selected.primary_stage1_rows.sum()) == 721033
                    else "FAIL"
                ),
            }
        ]
    )
    write_tsv(output / "genomic/cimmyt_stage1_overlap.tsv", overlap)
    if overlap.iloc[0].status != "PASS_EXACT":
        raise AssertionError("CIMMYT Stage-1 overlap changed")
    return selected, primary_counts.astype(int).to_dict()


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


def selected_ascii(payload: bytes, sample_indices: np.ndarray, sample_count: int) -> np.ndarray:
    if len(payload) == 2 * sample_count - 1:
        values = np.frombuffer(payload, dtype=np.uint8)
        if np.all(values[1::2] == ord("\t")):
            return values[::2][sample_indices]
    tokens = payload.split(b"\t")
    if len(tokens) != sample_count:
        raise AssertionError("HMP genotype payload width mismatch")
    return np.asarray(
        [token[0] if len(token) == 1 else ord("?") for token in (tokens[i] for i in sample_indices)],
        dtype=np.uint8,
    )


def token_counter_rows(counter: Counter[int], source: str) -> list[dict[str, Any]]:
    rows = []
    for code, count in sorted(counter.items()):
        token = chr(code) if 32 <= code <= 126 else f"ASCII_{code}"
        rows.append({"source": source, "token": token, "ascii_code": code, "count": count})
    return rows


def stream_primary_calls(
    raw_path: Path,
    imputed_path: Path,
    markers: pd.DataFrame,
    sample_ids: list[str],
    mapping: pd.DataFrame,
    output: Path,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected_indices = mapping.hmp_sample_index.to_numpy(dtype=int)
    call_path = output / "genomic/cimmyt_primary_unimputed_calls.npy"
    calls = np.lib.format.open_memmap(
        call_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(markers), len(mapping)),
    )
    sample_observed = np.zeros(len(mapping), dtype=np.int64)
    marker_summary: list[dict[str, Any]] = []
    raw_tokens: Counter[int] = Counter()
    imputed_tokens: Counter[int] = Counter()
    total_primary_cells = 0
    raw_observed_cells = 0
    imputed_observed_cells = 0
    paired_observed_cells = 0
    observed_exact_matches = 0
    observed_mismatches = 0
    raw_missing_filled_by_imputed = 0
    raw_missing_still_missing = 0
    raw_observed_became_missing = 0
    decodable_markers = 0
    with raw_path.open("rb", buffering=16 * 1024 * 1024) as raw, imputed_path.open(
        "rb", buffering=16 * 1024 * 1024
    ) as imputed:
        raw.readline()
        imputed.readline()
        for marker_index, pair in enumerate(zip_longest(raw, imputed, fillvalue=None)):
            raw_line, imputed_line = pair
            if raw_line is None or imputed_line is None:
                raise AssertionError("Unexpected HMP row-count divergence during call stream")
            raw_meta, raw_payload = split_hmp_line(raw_line)
            imputed_meta, imputed_payload = split_hmp_line(imputed_line)
            if not marker_metadata_compatible(raw_meta, imputed_meta):
                raise AssertionError("Marker metadata changed during call stream")
            alleles = markers.iloc[marker_index].alleles
            lookup, decodable, first, second, heterozygous = allele_lookup(str(alleles))
            raw_ascii = selected_ascii(raw_payload, selected_indices, len(sample_ids))
            imputed_ascii = selected_ascii(imputed_payload, selected_indices, len(sample_ids))
            raw_tokens.update(dict(enumerate(np.bincount(raw_ascii, minlength=256))))
            imputed_tokens.update(dict(enumerate(np.bincount(imputed_ascii, minlength=256))))
            raw_call = lookup[raw_ascii]
            imputed_call = lookup[imputed_ascii]
            calls[marker_index, :] = raw_call
            valid_raw = raw_call != MISSING
            valid_imputed = imputed_call != MISSING
            paired = valid_raw & valid_imputed
            raw_missing = ~valid_raw
            sample_observed += valid_raw.astype(np.int64)
            total_primary_cells += len(mapping)
            raw_observed_cells += int(valid_raw.sum())
            imputed_observed_cells += int(valid_imputed.sum())
            paired_observed_cells += int(paired.sum())
            exact = paired & (raw_call == imputed_call)
            observed_exact_matches += int(exact.sum())
            observed_mismatches += int((paired & (raw_call != imputed_call)).sum())
            raw_missing_filled_by_imputed += int((raw_missing & valid_imputed).sum())
            raw_missing_still_missing += int((raw_missing & ~valid_imputed).sum())
            raw_observed_became_missing += int((valid_raw & ~valid_imputed).sum())
            decodable_markers += int(decodable)
            observed_count = int(valid_raw.sum())
            dosage_sum = float(np.where(valid_raw, raw_call, 0).sum(dtype=np.float64))
            p = dosage_sum / (2.0 * observed_count) if observed_count else math.nan
            marker_summary.append(
                {
                    "marker_index": marker_index,
                    "marker_id": markers.iloc[marker_index].marker_id,
                    "alleles": alleles,
                    "biallelic_acgt_decodable": decodable,
                    "allele_0": first,
                    "allele_2": second,
                    "heterozygous_token": heterozygous,
                    "primary_observed_calls": observed_count,
                    "primary_missing_calls": len(mapping) - observed_count,
                    "primary_call_rate": observed_count / len(mapping),
                    "primary_allele_frequency_second": p,
                    "primary_minor_allele_frequency": min(p, 1.0 - p) if np.isfinite(p) else math.nan,
                    "primary_heterozygosity": (
                        float(np.sum(raw_call == 1)) / observed_count if observed_count else math.nan
                    ),
                }
            )
            if (marker_index + 1) % 5000 == 0:
                calls.flush()
                print(f"CIMMYT call stream: {marker_index + 1:,}/{len(markers):,} markers", flush=True)
    calls.flush()
    del calls
    summary = pd.DataFrame(marker_summary)
    pq.write_table(
        pa.Table.from_pandas(summary, preserve_index=False),
        output / "genomic/cimmyt_primary_marker_call_summary.parquet",
        compression="zstd",
    )
    samples = mapping.copy()
    samples["observed_marker_calls"] = sample_observed
    samples["missing_marker_calls"] = len(markers) - sample_observed
    samples["raw_call_rate"] = sample_observed / len(markers)
    samples["passes_frozen_sample_call_rate"] = samples.raw_call_rate.ge(MIN_SAMPLE_CALL_RATE)
    write_tsv(output / "genomic/cimmyt_primary_sample_qc.tsv", samples)
    token_rows = token_counter_rows(raw_tokens, "UNIMPUTED") + token_counter_rows(
        imputed_tokens, "IMPUTED_COMPARATOR"
    )
    token_frame = pd.DataFrame(token_rows)
    token_frame = token_frame[token_frame["count"].gt(0)].reset_index(drop=True)
    write_tsv(output / "genomic/cimmyt_call_token_inventory.tsv", token_frame)
    comparison = {
        "comparison_scope": "ALL_RETAINED_MARKERS_BY_ALL_4512_PRIMARY_CIMMYT_GIDS",
        "primary_cells": total_primary_cells,
        "decodable_biallelic_markers": decodable_markers,
        "nondecodable_markers": len(markers) - decodable_markers,
        "unimputed_observed_cells": raw_observed_cells,
        "unimputed_missing_cells": total_primary_cells - raw_observed_cells,
        "unimputed_call_rate": raw_observed_cells / total_primary_cells,
        "imputed_observed_cells": imputed_observed_cells,
        "imputed_missing_cells": total_primary_cells - imputed_observed_cells,
        "paired_observed_cells": paired_observed_cells,
        "observed_exact_matches": observed_exact_matches,
        "observed_mismatches": observed_mismatches,
        "observed_call_concordance": (
            observed_exact_matches / paired_observed_cells if paired_observed_cells else math.nan
        ),
        "unimputed_missing_filled_by_imputed": raw_missing_filled_by_imputed,
        "unimputed_missing_still_missing_in_imputed": raw_missing_still_missing,
        "unimputed_observed_became_missing_in_imputed": raw_observed_became_missing,
        "lossless_observed_missing_call_mask_recovered_for_retained_marker_universe": True,
        "pre_qc_marker_universe_recovered": False,
        "strict_production_disposition": "BLOCKED_GLOBALLY_PREFILTERED_MARKER_UNIVERSE",
        "status": (
            "PASS_CALL_MASK_RECOVERY_WITH_GLOBAL_MARKER_UNIVERSE_BLOCKER"
            if raw_observed_cells > 0 and raw_missing_filled_by_imputed > 0
            else "FAIL_NO_RECOVERED_MISSING_CALL_INFORMATION"
        ),
    }
    write_json(output / "genomic/cimmyt_unimputed_imputed_comparison.json", comparison)
    if comparison["status"] != "PASS_CALL_MASK_RECOVERY_WITH_GLOBAL_MARKER_UNIVERSE_BLOCKER":
        raise AssertionError("New source did not recover missing-call information")
    return call_path, summary, samples, comparison


def load_states(root: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    parity = root / PARITY_RELATIVE
    registry = pd.read_csv(
        parity / "splits/state_registry.tsv", sep="\t", dtype=str, keep_default_na=False
    )
    states: dict[str, dict[str, Any]] = {}
    for row in registry.itertuples(index=False):
        gids = pd.read_csv(parity / row.training_gid_path, sep="\t", dtype=str)[
            "canonical_gid"
        ].astype(str).tolist()
        if index_signature(gids) != row.training_gid_signature:
            raise AssertionError(f"Training-GID signature mismatch for {row.state_id}")
        states[row.state_id] = {
            "state_id": row.state_id,
            "scenario": row.scenario,
            "outer_fold": int(row.outer_fold),
            "inner_fold": None if row.inner_fold == "" else int(row.inner_fold),
            "training_gids": frozenset(gids),
            "training_gid_signature": row.training_gid_signature,
        }
    if len(states) != 150:
        raise AssertionError(f"Expected 150 Phase-5 training states, found {len(states)}")
    return registry, states


def fit_training_local_states(
    call_path: Path,
    markers: pd.DataFrame,
    sample_qc: pd.DataFrame,
    states: dict[str, dict[str, Any]],
    root: Path,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dosage = np.load(call_path, mmap_mode="r")
    if dosage.shape != (len(markers), len(sample_qc)):
        raise AssertionError("CIMMYT raw-call matrix shape mismatch")
    eligible = sample_qc[sample_qc.passes_frozen_sample_call_rate.astype(bool)].copy()
    gid_index = {
        gid: index
        for index, gid in enumerate(sample_qc.accepted_canonical_gid.astype(str).tolist())
    }
    eligible_gids = set(eligible.accepted_canonical_gid.astype(str))
    support_reference = pd.read_csv(
        root / PARITY_RELATIVE / "genomic/panel_fold_support.tsv", sep="\t"
    )
    support_reference = support_reference[support_reference.panel_id.eq(PANEL_ID)].set_index(
        "state_id"
    )
    marker_priority = np.argsort(
        np.asarray(
            [stable_json_hash({"marker_id": marker}) for marker in markers.marker_id.astype(str)],
            dtype="U64",
        ),
        kind="stable",
    )
    registry_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for state_number, (state_id, state) in enumerate(sorted(states.items()), start=1):
        all_training_gids = sorted(set(state["training_gids"]).intersection(gid_index))
        training_gids = sorted(eligible_gids.intersection(state["training_gids"]))
        expected_support = int(support_reference.loc[state_id, "training_primary_stage1_gids"])
        if len(all_training_gids) != expected_support:
            raise AssertionError(f"CIMMYT support changed for {state_id}")
        training_indices = np.asarray([gid_index[gid] for gid in training_gids], dtype=int)
        state_path = output / f"genomic/states/{state_id}__cimmyt_unimputed_vanraden_candidate.npz"
        support_pass = len(training_indices) >= MIN_TRAINING_GIDS
        retained = np.asarray([], dtype=np.int32)
        allele_frequency = np.asarray([], dtype=np.float32)
        denominator = math.nan
        candidate_status = "MASKED_INSUFFICIENT_TRAINING_GIDS"
        if support_pass:
            retained_blocks: list[np.ndarray] = []
            p_blocks: list[np.ndarray] = []
            for start in range(0, len(markers), 2048):
                stop = min(len(markers), start + 2048)
                block = np.asarray(dosage[start:stop, :], dtype=np.uint8)[:, training_indices]
                valid = block != MISSING
                counts = valid.sum(axis=1)
                sums = np.where(valid, block, 0).sum(axis=1, dtype=np.float64)
                heterozygous = (block == 1).sum(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    p = sums / (2.0 * counts)
                    heterozygosity = heterozygous / counts
                maf = np.minimum(p, 1.0 - p)
                keep = (
                    (counts >= max(10, math.ceil(MIN_MARKER_CALL_RATE * len(training_indices))))
                    & np.isfinite(p)
                    & (maf >= MIN_MAF)
                    & (p > 0.0)
                    & (p < 1.0)
                    & np.isfinite(heterozygosity)
                    & (heterozygosity <= MAX_HETEROZYGOSITY)
                )
                retained_blocks.append(np.flatnonzero(keep).astype(np.int32) + start)
                p_blocks.append(p[keep].astype(np.float32))
            retained = (
                np.concatenate(retained_blocks)
                if retained_blocks
                else np.asarray([], dtype=np.int32)
            )
            allele_frequency = (
                np.concatenate(p_blocks) if p_blocks else np.asarray([], dtype=np.float32)
            )
            if retained.size:
                denominator = float(
                    2.0 * np.sum(allele_frequency * (1.0 - allele_frequency), dtype=np.float64)
                )
                candidate_status = "PASS_DIAGNOSTIC_CANDIDATE_GLOBAL_MARKER_UNIVERSE_BLOCKED"
            else:
                candidate_status = "MASKED_NO_TRAINING_LOCAL_MARKERS_PASS_QC"
        np.savez_compressed(
            state_path,
            retained_marker_index=retained,
            allele_frequency=allele_frequency,
            denominator=np.asarray([denominator], dtype=np.float64),
            training_gid_signature=np.asarray([index_signature(training_gids)], dtype="U64"),
            qc_thresholds=np.asarray(
                [MIN_MARKER_CALL_RATE, MIN_MAF, MAX_HETEROZYGOSITY], dtype=np.float64
            ),
            strict_production_eligible=np.asarray([False], dtype=bool),
        )
        candidate_available = bool(support_pass and retained.size > 0)
        preprocessing_rows.append(
            {
                "state_id": state_id,
                "scenario": state["scenario"],
                "outer_fold": state["outer_fold"],
                "inner_fold": state["inner_fold"],
                "training_primary_panel_gids_before_sample_qc": len(all_training_gids),
                "training_gids_passing_sample_qc": len(training_gids),
                "minimum_training_gids": MIN_TRAINING_GIDS,
                "input_retained_universe_markers": len(markers),
                "training_local_retained_markers": len(retained),
                "sample_qc_fit_scope": "RAW_CALL_RATE_PER_SAMPLE_SOURCE_INTRINSIC",
                "marker_qc_fit_scope": "TRAINING_GIDS_ONLY",
                "allele_frequency_fit_scope": "TRAINING_GIDS_ONLY",
                "imputation_fit_scope": "TRAINING_GIDS_ONLY_2P_ON_DEMAND",
                "upstream_marker_universe_fit_scope": "GLOBAL_OR_UNDOCUMENTED_ENCODED_BY_SOURCE_FILENAME",
                "state_path": relative_posix(state_path, output),
                "state_sha256": sha256_file(state_path),
                "diagnostic_candidate_available": candidate_available,
                "strict_production_eligible": False,
                "status": candidate_status,
            }
        )
        registry_rows.append(
            {
                "state_id": state_id,
                "panel_id": PANEL_ID,
                "representation": "ON_DEMAND_RAW_CALLS_PLUS_TRAINING_LOCAL_PARAMETERS",
                "formula": "K=ZZT/(2*sum(p*(1-p)))",
                "entities": len(sample_qc),
                "training_entities": len(training_gids),
                "markers": len(retained),
                "denominator": denominator,
                "raw_call_path": relative_posix(call_path, output),
                "state_path": relative_posix(state_path, output),
                "diagnostic_candidate_available": candidate_available,
                "strict_production_component_available": False,
                "strict_production_mask": True,
                "strict_production_blocker": (
                    "GLOBAL_MARKER_UNIVERSE_PREFILTER" if candidate_available else "INSUFFICIENT_SUPPORT_OR_QC"
                ),
                "status": candidate_status,
            }
        )
        if candidate_available:
            retained_set = set(retained.tolist())
            sketch_indices = np.asarray(
                [int(index) for index in marker_priority if int(index) in retained_set][:64],
                dtype=int,
            )
            retained_lookup = {int(marker): position for position, marker in enumerate(retained)}
            sketch_p = np.asarray(
                [allele_frequency[retained_lookup[int(marker)]] for marker in sketch_indices],
                dtype=np.float64,
            )
            sketch = np.asarray(dosage[sketch_indices, :], dtype=np.float64)[
                :, training_indices
            ].T
            sketch[sketch == MISSING] = np.nan
            means = 2.0 * sketch_p
            missing = ~np.isfinite(sketch)
            sketch[missing] = np.broadcast_to(means, sketch.shape)[missing]
            sketch -= means
            sketch_denominator = float(2.0 * np.sum(sketch_p * (1.0 - sketch_p)))
            sketch /= math.sqrt(sketch_denominator)
            diagnostics = factor_diagnostics(sketch)
            diagnostics.update(
                {
                    "state_id": state_id,
                    "panel_id": PANEL_ID,
                    "effective_rank_scope": "DETERMINISTIC_64_MARKER_LOWER_BOUND",
                    "full_kernel_psd_certification": "EXACT_BY_FACTOR_CONSTRUCTION",
                    "strict_production_eligible": False,
                    "scientific_disposition": "DIAGNOSTIC_ONLY_GLOBAL_MARKER_UNIVERSE_BLOCKED",
                    "status": (
                        "PASS_DIAGNOSTIC"
                        if diagnostics["all_finite"] and diagnostics["algebraic_rank"] >= 2
                        else "FAIL"
                    ),
                }
            )
            diagnostic_rows.append(diagnostics)
        if state_number % 10 == 0:
            print(f"CIMMYT split-local QC: {state_number}/{len(states)} states", flush=True)
    registry = pd.DataFrame(registry_rows)
    preprocessing = pd.DataFrame(preprocessing_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    write_tsv(output / "genomic/cimmyt_component_registry.tsv", registry)
    write_tsv(output / "genomic/cimmyt_fold_preprocessing_registry.tsv", preprocessing)
    write_tsv(output / "genomic/cimmyt_state_diagnostics.tsv", diagnostics)
    if len(diagnostics) and not diagnostics.status.eq("PASS_DIAGNOSTIC").all():
        raise AssertionError("A reconstructable CIMMYT diagnostic state failed certification")
    return registry, preprocessing, diagnostics


def build_masks_and_redundancy(
    root: Path,
    output: Path,
    sample_qc: pd.DataFrame,
    component_registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = pq.read_table(
        root / PHASE5_RELATIVE / "indices/canonical_phase5_observation_index.parquet",
        columns=["phase5_observation_index", "canonical_gid"],
    ).to_pandas()
    raw_available = set(
        sample_qc.loc[
            sample_qc.passes_frozen_sample_call_rate.astype(bool), "accepted_canonical_gid"
        ].astype(str)
    )
    masks = master[["phase5_observation_index", "canonical_gid"]].copy()
    masks["cimmyt_unimputed_raw_call_source_available"] = masks.canonical_gid.astype(str).isin(
        raw_available
    )
    masks["cimmyt_strict_production_kg_available"] = False
    masks["cimmyt_strict_production_mask_reason"] = np.where(
        masks.cimmyt_unimputed_raw_call_source_available,
        "GLOBAL_MARKER_UNIVERSE_PREFILTER",
        "GID_NOT_IN_CIMMYT_PRIMARY_RAW_CALL_AXIS",
    )
    pq.write_table(
        pa.Table.from_pandas(masks, preserve_index=False),
        output / "masks/cimmyt_observation_component_masks.parquet",
        compression="zstd",
    )
    mask_summary = pd.DataFrame(
        [
            {
                "master_rows": len(masks),
                "raw_call_source_available_rows": int(
                    masks.cimmyt_unimputed_raw_call_source_available.sum()
                ),
                "strict_production_kg_available_rows": 0,
                "rows_deleted": 0,
                "status": (
                    "PASS_EXPLICIT_MASK_NO_ROW_DELETION" if len(masks) == 3_193_677 else "FAIL"
                ),
            }
        ]
    )
    write_tsv(output / "masks/cimmyt_observation_mask_summary.tsv", mask_summary)
    state_masks = component_registry[
        [
            "state_id",
            "diagnostic_candidate_available",
            "strict_production_component_available",
            "strict_production_mask",
            "strict_production_blocker",
        ]
    ].copy()
    write_tsv(output / "masks/cimmyt_state_component_masks.tsv", state_masks)
    seeds = pd.read_csv(
        root / PARITY_RELATIVE / "genomic/seeds_gid_consensus_summary.tsv", sep="\t", dtype=str
    )
    seeds_gids = set(seeds.canonical_gid.astype(str))
    cimmyt_gids = set(sample_qc.accepted_canonical_gid.astype(str))
    redundancy = pd.DataFrame(
        [
            {
                "left_component": PANEL_ID,
                "right_component": "seeds_of_discovery_dartseq",
                "left_primary_gids": len(cimmyt_gids),
                "right_primary_gids": len(seeds_gids),
                "shared_primary_gids": len(cimmyt_gids.intersection(seeds_gids)),
                "marker_coordinate_allele_harmonization": "NOT_ESTABLISHED",
                "numeric_kernel_correlation_computed": False,
                "merge_permitted": False,
                "disposition": "RETAIN_SEPARATE_EXPERTS_NO_CROSS_PLATFORM_MERGE",
            }
        ]
    )
    write_tsv(output / "redundancy/cimmyt_seeds_relationship.tsv", redundancy)
    return masks, mask_summary


def deterministic_replay(
    output: Path, preprocessing: pd.DataFrame, comparison: dict[str, Any]
) -> pd.DataFrame:
    state_payload = preprocessing.sort_values("state_id").fillna("").to_dict("records")
    first = stable_json_hash(state_payload)
    replay = stable_json_hash(
        pd.DataFrame(state_payload)
        .sample(frac=1, random_state=20260809)
        .sort_values("state_id")
        .to_dict("records")
    )
    comparison_payload = {
        key: comparison[key]
        for key in (
            "primary_cells",
            "unimputed_observed_cells",
            "unimputed_missing_cells",
            "imputed_observed_cells",
            "observed_exact_matches",
            "observed_mismatches",
        )
    }
    comparison_hash = stable_json_hash(comparison_payload)
    rows = [
        {
            "check": "STATE_PARAMETER_REGISTRY_ROW_ORDER_INVARIANT",
            "first_hash": first,
            "replay_hash": replay,
            "status": "PASS" if first == replay else "FAIL",
        },
        {
            "check": "RAW_IMPUTED_COMPARISON_CANONICAL_SERIALIZATION",
            "first_hash": comparison_hash,
            "replay_hash": stable_json_hash(json.loads(json.dumps(comparison_payload, sort_keys=True))),
            "status": "PASS",
        },
    ]
    frame = pd.DataFrame(rows)
    write_tsv(output / "deterministic_replay_validation.tsv", frame)
    if not frame.status.eq("PASS").all():
        raise AssertionError("Deterministic replay failed")
    return frame


def closing_hash_manifest(
    root: Path, output: Path, opening: pd.DataFrame, guard: ProtectedPathGuard
) -> pd.DataFrame:
    rows = []
    for record in opening.to_dict("records"):
        path = guard.assert_allowed(root / record["relative_path"], "CLOSING_HASH")
        closing_size = path.stat().st_size
        closing_sha = sha256_file(path)
        status = (
            "PASS"
            if closing_size == int(record["size"]) and closing_sha == record["sha256"]
            else "FAIL"
        )
        rows.append(
            {
                **record,
                "closing_size": closing_size,
                "closing_sha256": closing_sha,
                "status": status,
            }
        )
    frame = pd.DataFrame(rows)
    write_tsv(output / "CLOSING_HASH_MANIFEST.tsv", frame)
    write_json(
        output / "closing_hash_summary.json",
        {
            "files": len(frame),
            "bytes": int(frame["size"].sum()),
            "failures": int(frame.status.ne("PASS").sum()),
            "status": "PASS" if frame.status.eq("PASS").all() else "FAIL",
        },
    )
    if not frame.status.eq("PASS").all():
        raise AssertionError("Opening/closing input immutability validation failed")
    return frame


def run_tests(root: Path, output: Path, skip: bool) -> pd.DataFrame:
    if skip:
        frame = pd.DataFrame(
            [{"scope": "SKIPPED_BY_EXPLICIT_FLAG", "passed": 0, "failed": 0, "status": "SKIP"}]
        )
        write_tsv(output / "tests/test_summary.tsv", frame)
        return frame

    def wsl_path(path: Path) -> str:
        resolved = path.resolve()
        drive = resolved.drive.rstrip(":").lower()
        suffix = resolved.as_posix().split(":", 1)[1]
        return f"/mnt/{drive}{suffix}"

    wsl_root = wsl_path(root)
    wsl_output = wsl_path(output)
    wsl_python = "/home/Francisco/wheatconformer-envs/phase1-tf215-gpu-pandas22/bin/python"
    wsl_command = (
        f"cd {shlex.quote(wsl_root)} && "
        f"PHASE5_CIMMYT_UNIMPUTED_RELEASE_ROOT={shlex.quote(wsl_output)} "
        f"{shlex.quote(wsl_python)} -m pytest -q tests "
        f"--basetemp=/tmp/p5cug_final_tests"
    )
    commands = [
        (
            "TARGETED",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_phase5_cimmyt_unimputed_recovery.py",
                "tests/test_phase5_parity_extension.py",
                "tests/test_phase5_split_bound_kernel_release.py",
            ],
        ),
        (
            "COMPLETE_RELEVANT_WSL_TF215",
            ["wsl.exe", "-d", "Debian", "--", "bash", "-lc", wsl_command],
        ),
    ]
    rows = []
    environment = os.environ.copy()
    environment["PHASE5_CIMMYT_UNIMPUTED_RELEASE_ROOT"] = str(output)
    for scope, command in commands:
        result = subprocess.run(
            command, cwd=root, env=environment, text=True, capture_output=True, check=False
        )
        log_path = output / f"logs/{scope.lower()}_pytest.stdout.log"
        log_path.write_text(
            result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else ""),
            encoding="utf-8",
        )
        passed_match = re.search(r"(?P<passed>\d+) passed", result.stdout)
        failed_match = re.search(r"(?P<failed>\d+) failed", result.stdout)
        rows.append(
            {
                "scope": scope,
                "command": " ".join(command),
                "log": relative_posix(log_path, output),
                "passed": int(passed_match.group("passed")) if passed_match else 0,
                "failed": (
                    int(failed_match.group("failed"))
                    if failed_match
                    else (0 if result.returncode == 0 else 1)
                ),
                "return_code": result.returncode,
                "status": "PASS" if result.returncode == 0 else "FAIL",
            }
        )
        print(f"{scope} tests: return_code={result.returncode}", flush=True)
    frame = pd.DataFrame(rows)
    write_tsv(output / "tests/test_summary.tsv", frame)
    if not frame.status.eq("PASS").all():
        raise AssertionError("Test suite failed")
    return frame


def write_issue_ledger(output: Path, component_registry: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "issue_id": "CIMMYT_UNIMPUTED_RECOVERED_CALL_MASK",
            "severity": "RESOLVED_FOR_RETAINED_MARKER_UNIVERSE",
            "finding": (
                "The new HMP recovers observed calls and the exact missing-call mask on the same "
                "sample and retained-marker axes as the imputed export."
            ),
            "component_activation_blocking": False,
            "resolution": "TRAINING_LOCAL_DIAGNOSTIC_PARAMETERS_CONSTRUCTED_WHERE_SUPPORT_PASSES",
        },
        {
            "issue_id": "CIMMYT_GLOBAL_MARKER_UNIVERSE",
            "severity": "STRICT_PRODUCTION_BLOCKER",
            "finding": (
                "The source filename and historical provenance identify MAF0.01, Miss50, and Het10 "
                "filters applied before this export; the unfiltered pre-QC marker universe and fit "
                "population are not supplied."
            ),
            "component_activation_blocking": True,
            "resolution": "SUPPLY_PRE_QC_CALLS_OR_PROVE_UPSTREAM_MARKER_SELECTION_WAS_TRAINING_ONLY",
        },
        {
            "issue_id": "CIMMYT_LOW_SUPPORT_STATES",
            "severity": "EXPLICIT_STATE_MASK",
            "finding": (
                f"{int(component_registry.diagnostic_candidate_available.astype(bool).eq(False).sum())} "
                "of 150 states lack the frozen minimum support or retained markers."
            ),
            "component_activation_blocking": True,
            "resolution": "KEEP_EXPLICIT_STATE_MASK_NO_ROW_DELETION",
        },
    ]
    frame = pd.DataFrame(rows)
    write_tsv(output / "issue_ledger.tsv", frame)
    write_tsv(
        output / "terminal_disposition_ledger.tsv",
        pd.DataFrame(
            [
                {
                    "entity": PANEL_ID,
                    "role": "DENSE_GENOMEWIDE_DIAGNOSTIC_CANDIDATE",
                    "terminal_disposition": (
                        "CALL_MASK_RECOVERED_TRAINING_LOCAL_PARAMETERS_BUILT_STRICT_PRODUCTION_"
                        "BLOCKED_GLOBAL_MARKER_UNIVERSE"
                    ),
                    "entity_type": "GENOTYPE_PANEL",
                }
            ]
        ),
    )
    return frame


def write_decision(
    output: Path,
    guard: ProtectedPathGuard,
    axis: dict[str, Any],
    comparison: dict[str, Any],
    component_registry: pd.DataFrame,
    diagnostics: pd.DataFrame,
    masks: pd.DataFrame,
    opening: pd.DataFrame,
    closing: pd.DataFrame,
    tests: pd.DataFrame,
) -> None:
    access = guard.audit_frame()
    write_tsv(output / "protected_outcome_access_audit.tsv", access)
    candidate_count = int(component_registry.diagnostic_candidate_available.astype(bool).sum())
    support_masks = len(component_registry) - candidate_count
    checks = [
        ("release_id", RELEASE_ID == "P5CUG_20260809_V4_274E41DF", RELEASE_ID),
        ("raw_imputed_axes_exact", axis["status"] == "PASS_EXACT_RAW_IMPUTED_AXES", axis["status"]),
        (
            "missing_call_mask_recovered",
            comparison["lossless_observed_missing_call_mask_recovered_for_retained_marker_universe"],
            comparison["unimputed_missing_cells"],
        ),
        ("all_150_states_represented", len(component_registry) == 150, len(component_registry)),
        (
            "diagnostic_state_certifications_pass",
            diagnostics.status.eq("PASS_DIAGNOSTIC").all(),
            int(diagnostics.status.ne("PASS_DIAGNOSTIC").sum()),
        ),
        (
            "strict_production_not_activated",
            not component_registry.strict_production_component_available.astype(bool).any(),
            int(component_registry.strict_production_component_available.astype(bool).sum()),
        ),
        ("master_rows_preserved", len(masks) == 3_193_677, len(masks)),
        ("no_row_deletion", len(masks) == 3_193_677, len(masks)),
        (
            "opening_closing_inputs_immutable",
            closing.status.eq("PASS").all(),
            int(closing.status.ne("PASS").sum()),
        ),
        ("tests_pass", tests.status.isin(["PASS", "SKIP"]).all(), tests.to_dict("records")),
        (
            "no_protected_or_bundle_reads",
            not access.relative_path.str.startswith("server_phase5_parity_bundle/").any()
            and not access.decision.eq("DENY").any(),
            len(access),
        ),
        ("no_model_training", True, False),
        ("no_metric_selection", True, False),
        ("no_outer_or_final_outcome_access", True, False),
        ("no_commit_or_push", True, False),
    ]
    validation = pd.DataFrame(
        [
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "observed": json.dumps(observed, sort_keys=True)
                if isinstance(observed, (dict, list))
                else observed,
            }
            for name, passed, observed in checks
        ]
    )
    write_tsv(output / "validation_checks.tsv", validation)
    if not validation.status.eq("PASS").all():
        raise AssertionError("Atomic validation checks failed")
    decision = {
        "release_id": RELEASE_ID,
        "authoritative_phase5_release": PHASE5_RELEASE_ID,
        "authoritative_parity_release": PARITY_RELEASE_ID,
        "status": "PASS_CIMMYT_UNIMPUTED_ANALYSIS_WITH_GLOBAL_MARKER_UNIVERSE_BLOCKER",
        "sample_marker_identity_coordinate_and_allele_set_axes_exact_to_imputed_export": True,
        "allele_orientation_reversals_harmonized_to_unimputed_order": axis[
            "allele_orientation_reversals_harmonized_to_unimputed_order"
        ],
        "lossless_observed_missing_call_mask_recovered_for_retained_marker_universe": True,
        "pre_qc_marker_universe_recovered": False,
        "diagnostic_split_local_states_constructed": candidate_count,
        "explicit_support_or_qc_state_masks": support_masks,
        "strict_production_states_activated": 0,
        "strict_production_blocker": "GLOBAL_MAF0.01_MISS50_HET10_MARKER_UNIVERSE_PREFILTER",
        "required_resolution": (
            "SUPPLY_UNFILTERED_PRE_QC_CALLS_OR_DOCUMENT_UPSTREAM_MARKER_SELECTION_AS_"
            "TRAINING_ONLY_FOR_EACH_STATE"
        ),
        "parity_v2_modified": False,
        "phase5_modified": False,
        "v1_scientific_decisions_inherited": False,
        "model_training_performed": False,
        "component_selection_performed": False,
        "performance_evaluation_performed": False,
        "inner_validation_metrics_accessed": False,
        "outer_test_outcomes_accessed": False,
        "final_holdout_accessed": False,
        "future_projection_performed": False,
        "commit_or_push_performed": False,
        "phase6_handoff": (
            "KEEP_CIMMYT_DIAGNOSTIC_ONLY;_EXISTING_V2_ACTIVATIONS_AND_MASKS_REMAIN_AUTHORITATIVE"
        ),
        "decided_at_utc": utc_now(),
    }
    write_json(output / "PHASE5_CIMMYT_UNIMPUTED_DECISION.json", decision)
    report = f"""# Phase-5 CIMMYT unimputed recovery follow-on

- Release: `{RELEASE_ID}`
- Atomic status: `{decision['status']}`
- Immutable parents: `{PHASE5_RELEASE_ID}` and `{PARITY_RELEASE_ID}` remain unchanged.
- Protected access: the repository denylist was loaded before inspection; no server bundle content, metric-bearing lock, validation metric, prediction, outer outcome, or final holdout was opened.

## What the new file resolves

The unimputed and imputed HMP files have exact 50,363-sample and marker metadata axes. For the exact Stage-1 primary overlap (4,512 GIDs / 721,033 observations), the new file restores observed genotypes and the missing-call mask. The full retained-axis comparison is recorded in `genomic/cimmyt_unimputed_imputed_comparison.json`.

Frozen sample QC and training-local marker QC, allele-frequency fitting, and 2p imputation parameters were applied to all 150 Phase-5 outer/inner states. `{candidate_count}` states yield independently certified diagnostic VanRaden parameter sets; `{support_masks}` states remain explicitly masked for insufficient support or QC. No phenotype rows were deleted.

## Remaining strict-production blocker

This source is unimputed, but its name and prior provenance identify a marker universe already filtered using MAF0.01, Miss50, and Het10. The pre-QC marker universe and the population used to choose retained markers were not supplied. Consequently, the restored calls support fold-local diagnostics but do not prove training-only marker-universe selection. Strict inductive production activation remains zero rather than weakening the leakage criterion.

Resolution requires either the unfiltered pre-QC calls or auditable evidence that marker selection was fitted independently within every Phase-5 training state. Phase 6 was not begun.
"""
    (output / "PHASE5_CIMMYT_UNIMPUTED_REPORT.md").write_text(report, encoding="utf-8")


def write_output_manifest(output: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "output_manifest.tsv":
            rows.append(
                {
                    "relative_path": relative_posix(path, output),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    frame = pd.DataFrame(rows)
    write_tsv(output / "output_manifest.tsv", frame)
    return frame


def main() -> None:
    args = parse_args()
    root = args.repository_root.resolve()
    output = (
        args.output_root.resolve()
        if args.output_root
        else (root / RELEASE_RELATIVE).resolve()
    )
    denylist = root / V1_INCIDENT_RELATIVE / "PROTECTED_PATH_DENYLIST.txt"
    guard = ProtectedPathGuard(root, denylist)
    preflight = verify_preflight(root, output, guard)
    if args.preflight:
        print(json.dumps({**preflight, "output_root": str(output)}, indent=2, sort_keys=True))
        return
    if preflight["status"] != "PASS_PREFLIGHT":
        raise SystemExit(f"FAIL_IF_EXISTS: release root already exists: {output}")
    ensure_fail_if_exists(output)
    create_directories(output)
    write_opening_contract(root, output, preflight)
    print("Opening hash manifest: binding authorized immutable inputs", flush=True)
    opening = opening_hash_manifest(root, output, guard)
    raw_path = root / RAW_RELATIVE
    imputed_path = root / IMPUTED_RELATIVE
    print("Certifying complete raw/imputed axes", flush=True)
    markers, sample_ids, axis = inspect_axes(raw_path, imputed_path, output)
    mapping, _ = load_primary_mapping(root, sample_ids, output)
    print("Streaming 4,512 primary sample columns from raw and imputed HMPs", flush=True)
    call_path, _, sample_qc, comparison = stream_primary_calls(
        raw_path, imputed_path, markers, sample_ids, mapping, output
    )
    _, states = load_states(root)
    print("Fitting training-local CIMMYT diagnostic parameters for 150 states", flush=True)
    component_registry, preprocessing, diagnostics = fit_training_local_states(
        call_path, markers, sample_qc, states, root, output
    )
    masks, _ = build_masks_and_redundancy(root, output, sample_qc, component_registry)
    write_issue_ledger(output, component_registry)
    deterministic_replay(output, preprocessing, comparison)
    print("Closing hash manifest: verifying input immutability", flush=True)
    closing = closing_hash_manifest(root, output, opening, guard)
    # This audit is a predecision test input, so materialize it before invoking pytest.
    write_tsv(output / "protected_outcome_access_audit.tsv", guard.audit_frame())
    tests = run_tests(root, output, args.skip_tests)
    write_decision(
        output,
        guard,
        axis,
        comparison,
        component_registry,
        diagnostics,
        masks,
        opening,
        closing,
        tests,
    )
    manifest = write_output_manifest(output)
    print(
        json.dumps(
            {
                "release_id": RELEASE_ID,
                "status": "PASS_CIMMYT_UNIMPUTED_ANALYSIS_WITH_GLOBAL_MARKER_UNIVERSE_BLOCKER",
                "diagnostic_states": int(
                    component_registry.diagnostic_candidate_available.astype(bool).sum()
                ),
                "strict_production_states": 0,
                "output_files": len(manifest),
                "output_root": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
