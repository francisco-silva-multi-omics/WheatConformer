from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.v2.audit_cimmyt_pre_qc_split_local import (
    GzipWrappedZipLines,
    HAPMAP_METADATA_COLUMNS,
    MISSING,
    allele_lookup,
    load_states,
    split_hapmap_line,
    stable_hash_lines,
)


PROTOCOL_RELATIVE = Path(
    "server_training_pipeline/cimmyt_pre_qc_production_kg_protocol_v2.json"
)
SOURCE_RELEASE = Path("audit/v2/phase5_cimmyt_pre_qc_split_local_v1")
FILTERED_RELEASE = Path("audit/v2/phase5_cimmyt_unimputed_recovery_v4")
PREREQUISITE_RELEASE = Path("audit/v2/phase5_panel_prerequisite_recovery_v1")
SOURCE_ORDER_RELATIVE = (
    PREREQUISITE_RELEASE / "results/cimmyt_pre_qc_hapmap_sample_order.parquet"
)
ARCHIVE_RELATIVE = PREREQUISITE_RELEASE / "downloads/CIMMYT-2013-2018.hmp.txt.zip.gz"
OUTPUT_RELATIVE = Path("audit/v2/phase5_cimmyt_pre_qc_production_kg_v2")
DECISION_NAME = "CIMMYT_PRE_QC_PRODUCTION_KG_V2_DECISION.json"
REPORT_NAME = "CIMMYT_PRE_QC_PRODUCTION_KG_V2_REPORT.md"

SOURCE_CALLS = SOURCE_RELEASE / "genomic/cimmyt_pre_qc_primary_raw_calls.npy"
SOURCE_MARKERS = SOURCE_RELEASE / "genomic/cimmyt_pre_qc_marker_axis.parquet"
SOURCE_SAMPLES = SOURCE_RELEASE / "genomic/cimmyt_pre_qc_primary_sample_axis.tsv"
SOURCE_DECISION = SOURCE_RELEASE / "CIMMYT_PRE_QC_SPLIT_LOCAL_DECISION.json"
FILTERED_MARKERS = FILTERED_RELEASE / "genomic/cimmyt_marker_axis.parquet"

QC_RETAINED = np.uint8(0)
QC_INCOMPATIBLE_ALLELES = np.uint8(1)
QC_LOW_CALL_RATE = np.uint8(2)
QC_MONOMORPHIC = np.uint8(3)
QC_LOW_MAF = np.uint8(4)
QC_HIGH_HETEROZYGOSITY = np.uint8(5)
QC_REASON_NAMES = {
    int(QC_RETAINED): "RETAINED",
    int(QC_INCOMPATIBLE_ALLELES): "INCOMPATIBLE_ALLELE_SET",
    int(QC_LOW_CALL_RATE): "LOW_TRAINING_CALL_RATE",
    int(QC_MONOMORPHIC): "MONOMORPHIC_IN_TRAINING",
    int(QC_LOW_MAF): "LOW_TRAINING_MAF",
    int(QC_HIGH_HETEROZYGOSITY): "HIGH_TRAINING_HETEROZYGOSITY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exact split-local CIMMYT pre-QC production K_G artifacts"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_RELATIVE)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def git_commit(code_root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else "UNAVAILABLE"


def exact_allele_relation(pre_qc: str, filtered: str | None) -> str:
    if filtered is None or not str(filtered).strip():
        return "PRE_QC_ONLY_REFERENCE_ORIENTATION"
    left = [value.strip().upper() for value in str(pre_qc).replace("|", "/").split("/")]
    right = [value.strip().upper() for value in str(filtered).replace("|", "/").split("/")]
    if len(left) != 2 or len(right) != 2 or set(left) != set(right):
        return "INCOMPATIBLE_ALLELE_SET"
    if left == right:
        return "SAME_ORDER"
    if left == right[::-1]:
        return "REVERSED_ORDER"
    return "INCOMPATIBLE_ALLELE_SET"


def estimate_sample_call_rate_threshold(
    training_rates: Iterable[float],
    *,
    floor: float,
    ceiling: float,
    mad_multiplier: float,
) -> dict[str, float]:
    values = np.asarray(list(training_rates), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "threshold": float(floor),
            "median": math.nan,
            "mad": math.nan,
            "robust_lower_fence": math.nan,
        }
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    lower_fence = median - mad_multiplier * 1.4826 * mad
    threshold = float(np.clip(max(floor, lower_fence), floor, ceiling))
    return {
        "threshold": threshold,
        "median": median,
        "mad": mad,
        "robust_lower_fence": float(lower_fence),
    }


def fit_marker_qc(
    dosage: np.ndarray,
    training_indices: np.ndarray,
    structural_eligible: np.ndarray,
    *,
    minimum_call_rate: float,
    minimum_observed: int,
    minimum_maf: float,
    maximum_heterozygosity: float,
    block_size: int = 2048,
) -> dict[str, np.ndarray | int]:
    marker_count = dosage.shape[0]
    observed = np.zeros(marker_count, dtype=np.int32)
    allele_frequency = np.full(marker_count, np.nan, dtype=np.float32)
    heterozygosity = np.full(marker_count, np.nan, dtype=np.float32)
    required_observed = max(
        int(minimum_observed), int(math.ceil(minimum_call_rate * len(training_indices)))
    )
    for start in range(0, marker_count, block_size):
        stop = min(marker_count, start + block_size)
        block = np.asarray(dosage[start:stop, training_indices], dtype=np.uint8)
        valid = block != MISSING
        counts = valid.sum(axis=1, dtype=np.int32)
        sums = np.where(valid, block, 0).sum(axis=1, dtype=np.float64)
        het = (block == 1).sum(axis=1, dtype=np.int32)
        observed[start:stop] = counts
        with np.errstate(divide="ignore", invalid="ignore"):
            allele_frequency[start:stop] = (sums / (2.0 * counts)).astype(np.float32)
            heterozygosity[start:stop] = (het / counts).astype(np.float32)

    p = allele_frequency.astype(np.float64)
    maf = np.minimum(p, 1.0 - p)
    reasons = np.full(marker_count, QC_RETAINED, dtype=np.uint8)
    reasons[~structural_eligible] = QC_INCOMPATIBLE_ALLELES
    eligible = structural_eligible.copy()
    low_call = eligible & (observed < required_observed)
    reasons[low_call] = QC_LOW_CALL_RATE
    eligible &= ~low_call
    monomorphic = eligible & (~np.isfinite(p) | (p <= 0.0) | (p >= 1.0))
    reasons[monomorphic] = QC_MONOMORPHIC
    eligible &= ~monomorphic
    low_maf = eligible & (maf < minimum_maf)
    reasons[low_maf] = QC_LOW_MAF
    eligible &= ~low_maf
    high_het = eligible & (
        ~np.isfinite(heterozygosity) | (heterozygosity > maximum_heterozygosity)
    )
    reasons[high_het] = QC_HIGH_HETEROZYGOSITY
    retained = np.flatnonzero(reasons == QC_RETAINED).astype(np.int32)
    return {
        "observed": observed,
        "allele_frequency_all": allele_frequency,
        "heterozygosity": heterozygosity,
        "reasons": reasons,
        "retained": retained,
        "retained_frequency": allele_frequency[retained].astype(np.float32),
        "required_observed": required_observed,
    }


def build_exact_kernel(
    dosage: np.ndarray,
    retained_markers: np.ndarray,
    allele_frequency: np.ndarray,
    projection_indices: np.ndarray,
    training_indices: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    if not retained_markers.size:
        raise ValueError("Cannot construct K_G without retained markers")
    projection_lookup = {int(value): index for index, value in enumerate(projection_indices)}
    training_positions = np.asarray(
        [projection_lookup[int(value)] for value in training_indices], dtype=np.int32
    )
    z = np.asarray(
        dosage[np.ix_(retained_markers.astype(np.int64), projection_indices)],
        dtype=np.float64,
    ).T
    means = 2.0 * allele_frequency.astype(np.float64)
    missing = z == float(MISSING)
    z[missing] = np.broadcast_to(means, z.shape)[missing]
    z -= means
    denominator = float(
        2.0
        * np.sum(
            allele_frequency.astype(np.float64)
            * (1.0 - allele_frequency.astype(np.float64)),
            dtype=np.float64,
        )
    )
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("VanRaden denominator is not finite and positive")
    relationship = (z @ z.T) / denominator
    raw_training = relationship[np.ix_(training_positions, training_positions)]
    diagonal_scale = float(np.mean(np.diag(raw_training)))
    if not np.isfinite(diagonal_scale) or diagonal_scale <= 0.0:
        raise ValueError("Training K_G mean diagonal is not finite and positive")
    relationship /= diagonal_scale
    training_kernel = relationship[np.ix_(training_positions, training_positions)]
    projection_to_training = relationship[:, training_positions]
    training_center_residual = float(
        np.max(np.abs(z[training_positions].mean(axis=0)))
    )
    stored_training = training_kernel.astype(np.float32)
    stored_projection = projection_to_training.astype(np.float32)
    stored_eigenvalues = np.linalg.eigvalsh(stored_training.astype(np.float64))
    largest = float(stored_eigenvalues[-1])
    return {
        "training_positions": training_positions,
        "training_kernel": stored_training,
        "projection_to_training": stored_projection,
        "denominator": denominator,
        "diagonal_scale": diagonal_scale,
        "training_center_residual": training_center_residual,
        "eigenvalues": stored_eigenvalues,
        "largest_eigenvalue": largest,
    }


def build_orientation_audit(
    pre_qc: pd.DataFrame, filtered: pd.DataFrame
) -> pd.DataFrame:
    right = filtered[["marker_id", "alleles", "chrom", "pos"]].rename(
        columns={
            "alleles": "filtered_alleles",
            "chrom": "filtered_chrom",
            "pos": "filtered_pos",
        }
    )
    audit = pre_qc[["marker_index", "marker_id", "alleles", "chrom", "pos"]].merge(
        right, on="marker_id", how="left", validate="one_to_one"
    )
    audit = audit.rename(
        columns={"alleles": "pre_qc_alleles", "chrom": "pre_qc_chrom", "pos": "pre_qc_pos"}
    )
    audit["allele_relation"] = [
        exact_allele_relation(left, None if pd.isna(right_value) else str(right_value))
        for left, right_value in zip(audit.pre_qc_alleles, audit.filtered_alleles)
    ]
    audit["globally_filtered_hmp_member"] = audit.filtered_alleles.notna()
    audit["production_orientation"] = "PRE_QC_SOURCE_ORDER"
    audit["production_marker_structurally_eligible"] = ~audit.allele_relation.eq(
        "INCOMPATIBLE_ALLELE_SET"
    )
    return audit


def extract_replicate_candidate_calls(
    archive: Path,
    source_ids: list[str],
    candidate_ids: list[str],
    marker_count: int,
    output_path: Path,
) -> np.ndarray:
    positions = [source_ids.index(value) for value in candidate_ids]
    fixed_positions = 2 * np.asarray(positions, dtype=np.int64)
    calls = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.uint8, shape=(marker_count, len(candidate_ids))
    )
    reader = GzipWrappedZipLines(archive)
    lines = iter(reader)
    header = next(lines).rstrip(b"\r\n").split(b"\t")
    observed_ids = [value.decode("utf-8") for value in header[HAPMAP_METADATA_COLUMNS:]]
    if observed_ids != source_ids:
        raise ValueError("CIMMYT source header changed before replicate extraction")
    observed_markers = 0
    for marker_index, line in enumerate(lines):
        metadata, payload = split_hapmap_line(line)
        if len(payload) == 2 * len(source_ids) - 1:
            payload_bytes = np.frombuffer(payload, dtype=np.uint8)
            if np.all(payload_bytes[1::2] == ord("\t")):
                selected_ascii = payload_bytes[fixed_positions]
            else:
                tokens = payload.split(b"\t")
                selected_ascii = np.asarray([tokens[index][0] for index in positions])
        else:
            tokens = payload.split(b"\t")
            if len(tokens) != len(source_ids):
                raise ValueError("CIMMYT replicate extraction payload width mismatch")
            selected_ascii = np.asarray([tokens[index][0] for index in positions])
        lookup, _, _, _, _ = allele_lookup(metadata[1].decode("utf-8"))
        calls[marker_index] = lookup[selected_ascii]
        observed_markers += 1
        if observed_markers % 10_000 == 0:
            calls.flush()
            print(f"CIMMYT replicate stream: {observed_markers:,}/{marker_count:,}", flush=True)
    calls.flush()
    if observed_markers != marker_count or not reader.validated:
        raise ValueError("CIMMYT replicate candidate extraction did not certify source stream")
    return calls


def read_archive_header_ids(archive: Path) -> list[str]:
    reader = GzipWrappedZipLines(archive)
    header = next(iter(reader)).rstrip(b"\r\n").split(b"\t")
    if header[:2] != [b"rs#", b"alleles"]:
        raise ValueError("Unexpected CIMMYT pre-QC HapMap header")
    return [value.decode("utf-8") for value in header[HAPMAP_METADATA_COLUMNS:]]


def build_identity_and_replicate_audit(
    root: Path,
    output: Path,
    protocol: dict[str, Any],
    primary_axis: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source_ids = read_archive_header_ids(root / ARCHIVE_RELATIVE)
    recovered_set = set(
        pd.read_parquet(root / SOURCE_ORDER_RELATIVE).iloc[:, 0].astype(str)
    )
    if set(source_ids) != recovered_set:
        raise ValueError("Physical CIMMYT header IDs disagree with prerequisite sample set")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Duplicate source sample IDs in pre-QC panel")
    exact_pattern = re.compile(r"^GID\d+$")
    suffix_pattern = re.compile(r"^(GID\d+)_([^\s]+)$")
    primary_gids = set(primary_axis.canonical_gid.astype(str))
    rows: list[dict[str, object]] = []
    candidate_groups: dict[str, list[str]] = {}
    for source_index, source_id in enumerate(source_ids):
        exact = bool(exact_pattern.fullmatch(source_id))
        suffix = suffix_pattern.fullmatch(source_id)
        candidate_gid = source_id if exact else (suffix.group(1) if suffix else "")
        identity_class = "UNASSIGNED_NON_GID_LABEL"
        if exact:
            identity_class = "EXACT_UNIQUE_GID"
        elif suffix:
            identity_class = "SUFFIX_REPLICATE_CANDIDATE_NOT_IDENTITY"
        if suffix:
            candidate_groups.setdefault(candidate_gid, []).append(source_id)
        rows.append(
            {
                "source_sample_index": source_index,
                "source_sample_id": source_id,
                "candidate_canonical_gid": candidate_gid,
                "identity_class": identity_class,
                "exact_primary_stage1_mapping": exact and source_id in primary_gids,
                "production_mapping_eligible": exact and source_id in primary_gids,
            }
        )
    identity = pd.DataFrame(rows)
    exact_by_gid = {
        value for value in identity.loc[identity.identity_class.eq("EXACT_UNIQUE_GID"), "source_sample_id"]
    }
    for gid in list(candidate_groups):
        if gid in exact_by_gid:
            candidate_groups[gid].insert(0, gid)
    candidate_groups = {gid: values for gid, values in candidate_groups.items() if len(values) >= 2}
    candidate_ids = [value for gid in sorted(candidate_groups) for value in candidate_groups[gid]]
    candidate_ids = list(dict.fromkeys(candidate_ids))
    replicate_call_path = output / "identity/cimmyt_replicate_candidate_calls.npy"
    replicate_call_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_calls = extract_replicate_candidate_calls(
        root / ARCHIVE_RELATIVE,
        source_ids,
        candidate_ids,
        int(protocol["source_marker_count"]),
        replicate_call_path,
    )
    candidate_index = {value: index for index, value in enumerate(candidate_ids)}
    minimum_overlap = int(
        protocol["identity_policy"]["minimum_replicate_shared_nonmissing_markers"]
    )
    minimum_concordance = float(protocol["identity_policy"]["minimum_replicate_concordance"])
    pair_rows: list[dict[str, object]] = []
    for gid, labels in sorted(candidate_groups.items()):
        for first_index in range(len(labels)):
            for second_index in range(first_index + 1, len(labels)):
                first = labels[first_index]
                second = labels[second_index]
                left = np.asarray(candidate_calls[:, candidate_index[first]])
                right = np.asarray(candidate_calls[:, candidate_index[second]])
                shared = (left != MISSING) & (right != MISSING)
                overlap = int(shared.sum())
                concordance = float(np.mean(left[shared] == right[shared])) if overlap else math.nan
                if "wrong" in first.lower() or "wrong" in second.lower():
                    disposition = "QUARANTINED_LABEL_CONFLICT"
                elif overlap >= minimum_overlap and concordance >= minimum_concordance:
                    disposition = "CONCORDANT_REPLICATE_CANDIDATE_METADATA_REVIEW"
                else:
                    disposition = "CONFLICTING_REPLICATE_CANDIDATE"
                pair_rows.append(
                    {
                        "candidate_canonical_gid": gid,
                        "sample_a": first,
                        "sample_b": second,
                        "shared_nonmissing_markers": overlap,
                        "genotype_call_concordance": concordance,
                        "minimum_required_overlap": minimum_overlap,
                        "minimum_required_concordance": minimum_concordance,
                        "disposition": disposition,
                        "used_in_production_axis": False,
                    }
                )
    replicate_audit = pd.DataFrame(pair_rows)
    identity_summary = {
        "source_sample_count": len(identity),
        "duplicate_source_sample_ids": int(identity.source_sample_id.duplicated(False).sum()),
        "exact_gid_source_samples": int(identity.identity_class.eq("EXACT_UNIQUE_GID").sum()),
        "suffix_replicate_candidate_samples": int(
            identity.identity_class.eq("SUFFIX_REPLICATE_CANDIDATE_NOT_IDENTITY").sum()
        ),
        "exact_primary_stage1_mappings": int(identity.exact_primary_stage1_mapping.sum()),
        "replicate_candidate_groups": len(candidate_groups),
        "replicate_candidate_pairs": len(replicate_audit),
        "replicate_pairs_used_in_production": int(
            replicate_audit.used_in_production_axis.sum() if len(replicate_audit) else 0
        ),
        "replicate_call_matrix_sha256": sha256_file(replicate_call_path),
    }
    return identity, replicate_audit, identity_summary


def state_fit_signature(
    candidate_gids: list[str], threshold: float, protocol_sha256: str
) -> str:
    return stable_hash_lines(
        ["CIMMYT_PRE_QC_PRODUCTION_KG_V2", protocol_sha256, f"{threshold:.17g}", *candidate_gids]
    )


def build_states(
    root: Path,
    output: Path,
    protocol: dict[str, Any],
    protocol_sha256: str,
    dosage: np.ndarray,
    markers: pd.DataFrame,
    primary_axis: pd.DataFrame,
    sample_rates: np.ndarray,
    orientation: pd.DataFrame,
) -> pd.DataFrame:
    _, states = load_states(root)
    gid_index = {
        gid: index for index, gid in enumerate(primary_axis.canonical_gid.astype(str).tolist())
    }
    structural_eligible = orientation.production_marker_structurally_eligible.to_numpy(bool)
    globally_filtered = orientation.globally_filtered_hmp_member.to_numpy(bool)
    sample_protocol = protocol["sample_qc"]
    marker_protocol = protocol["marker_qc"]
    kernel_protocol = protocol["kernel"]
    kernel_dir = output / "states/by_training_signature"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict[str, Any]] = {}
    registry_rows: list[dict[str, object]] = []
    for number, state in enumerate(states, start=1):
        state_training = set(state["training_gids"])
        candidate_indices = np.asarray(
            [index for gid, index in gid_index.items() if gid in state_training], dtype=np.int32
        )
        candidate_gids = primary_axis.iloc[candidate_indices].canonical_gid.astype(str).tolist()
        threshold_fit = estimate_sample_call_rate_threshold(
            sample_rates[candidate_indices],
            floor=float(sample_protocol["minimum_call_rate_floor"]),
            ceiling=float(sample_protocol["maximum_call_rate_threshold"]),
            mad_multiplier=float(sample_protocol["robust_mad_multiplier"]),
        )
        threshold = float(threshold_fit["threshold"])
        projection_indices = np.flatnonzero(sample_rates >= threshold).astype(np.int32)
        training_indices = candidate_indices[sample_rates[candidate_indices] >= threshold]
        signature = state_fit_signature(candidate_gids, threshold, protocol_sha256)
        reused = signature in cache
        if reused:
            fitted = cache[signature]
        else:
            marker_fit = fit_marker_qc(
                dosage,
                training_indices,
                structural_eligible,
                minimum_call_rate=float(marker_protocol["minimum_training_call_rate"]),
                minimum_observed=int(marker_protocol["minimum_observed_training_calls"]),
                minimum_maf=float(
                    marker_protocol["minimum_training_minor_allele_frequency"]
                ),
                maximum_heterozygosity=float(
                    marker_protocol["maximum_training_heterozygosity"]
                ),
            )
            retained = np.asarray(marker_fit["retained"], dtype=np.int32)
            retained_frequency = np.asarray(
                marker_fit["retained_frequency"], dtype=np.float32
            )
            kernel = build_exact_kernel(
                dosage,
                retained,
                retained_frequency,
                projection_indices,
                training_indices,
            )
            eigenvalues = np.asarray(kernel["eigenvalues"], dtype=np.float64)
            largest = float(kernel["largest_eigenvalue"])
            psd_tolerance = max(
                float(kernel_protocol["psd_absolute_tolerance"]),
                largest * float(kernel_protocol["psd_relative_tolerance"]),
            )
            rank_tolerance = max(
                float(kernel_protocol["psd_absolute_tolerance"]),
                largest * float(kernel_protocol["effective_rank_relative_tolerance"]),
            )
            effective_rank = int((eigenvalues > rank_tolerance).sum())
            training_positions = np.asarray(kernel["training_positions"], dtype=np.int32)
            training_kernel = np.asarray(kernel["training_kernel"], dtype=np.float32)
            projection_kernel = np.asarray(
                kernel["projection_to_training"], dtype=np.float32
            )
            projection_residual = float(
                np.max(np.abs(projection_kernel[training_positions] - training_kernel))
            )
            reasons = np.asarray(marker_fit["reasons"], dtype=np.uint8)
            reason_counts = {
                name: int((reasons == code).sum()) for code, name in QC_REASON_NAMES.items()
            }
            artifact_path = kernel_dir / f"{signature}.npz"
            temporary = artifact_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                protocol_version=np.asarray([protocol["protocol_version"]]),
                fit_signature=np.asarray([signature]),
                source_marker_count=np.asarray([dosage.shape[0]], dtype=np.int32),
                source_sample_count=np.asarray([dosage.shape[1]], dtype=np.int32),
                sample_call_rate_threshold=np.asarray([threshold], dtype=np.float64),
                all_sample_support=(sample_rates >= threshold),
                training_sample_index=training_indices,
                projection_sample_index=projection_indices,
                training_gid=primary_axis.iloc[training_indices].canonical_gid.astype(str).to_numpy(),
                projection_gid=primary_axis.iloc[projection_indices].canonical_gid.astype(str).to_numpy(),
                training_axis_position=training_positions,
                marker_observed_training_calls=np.asarray(marker_fit["observed"], dtype=np.int32),
                marker_training_allele_frequency=np.asarray(
                    marker_fit["allele_frequency_all"], dtype=np.float32
                ),
                marker_training_heterozygosity=np.asarray(
                    marker_fit["heterozygosity"], dtype=np.float32
                ),
                marker_qc_reason_code=reasons,
                retained_marker_index=retained,
                training_allele_frequency=retained_frequency,
                vanraden_denominator=np.asarray([kernel["denominator"]], dtype=np.float64),
                mean_training_diagonal_scale=np.asarray(
                    [kernel["diagonal_scale"]], dtype=np.float64
                ),
                training_kernel=training_kernel,
                projection_to_training_kernel=projection_kernel,
                training_kernel_eigenvalues=eigenvalues,
            )
            temporary.replace(artifact_path)
            fitted = {
                "artifact_path": artifact_path,
                "artifact_sha256": sha256_file(artifact_path),
                "retained": retained,
                "reason_counts": reason_counts,
                "training_count": len(training_indices),
                "projection_count": len(projection_indices),
                "denominator": float(kernel["denominator"]),
                "diagonal_scale": float(kernel["diagonal_scale"]),
                "training_diag_mean": float(np.mean(np.diag(training_kernel))),
                "symmetry_residual": float(np.max(np.abs(training_kernel - training_kernel.T))),
                "minimum_eigenvalue": float(eigenvalues[0]),
                "psd_tolerance": psd_tolerance,
                "effective_rank": effective_rank,
                "projection_residual": projection_residual,
                "training_center_residual": float(kernel["training_center_residual"]),
                "retained_pre_qc_only": int((~globally_filtered[retained]).sum()),
                "retained_globally_filtered": int(globally_filtered[retained].sum()),
            }
            cache[signature] = fitted
        strict = (
            fitted["training_count"] >= int(kernel_protocol["minimum_training_GIDs"])
            and len(fitted["retained"]) > 0
            and fitted["minimum_eigenvalue"] >= -fitted["psd_tolerance"]
            and fitted["effective_rank"] >= 2
            and abs(fitted["training_diag_mean"] - 1.0) <= 1e-5
            and fitted["projection_residual"] <= 1e-6
            and fitted["reason_counts"]["INCOMPATIBLE_ALLELE_SET"]
            == len(protocol["allele_policy"]["incompatible_shared_markers"])
            and fitted["retained_pre_qc_only"] > 0
        )
        registry_rows.append(
            {
                "state_id": state["state_id"],
                "scenario": state["scenario"],
                "outer_fold": state["outer_fold"],
                "inner_fold": state["inner_fold"],
                "state_level": state["state_level"],
                "fit_signature": signature,
                "fit_reused": reused,
                "training_panel_GIDs_before_sample_QC": len(candidate_indices),
                "training_panel_GIDs_after_sample_QC": fitted["training_count"],
                "sample_call_rate_training_median": threshold_fit["median"],
                "sample_call_rate_training_MAD": threshold_fit["mad"],
                "sample_call_rate_robust_lower_fence": threshold_fit["robust_lower_fence"],
                "sample_call_rate_threshold": threshold,
                "projection_supported_GIDs": fitted["projection_count"],
                "held_out_projection_GIDs": fitted["projection_count"]
                - fitted["training_count"],
                "source_marker_rows_considered": dosage.shape[0],
                "retained_marker_rows": len(fitted["retained"]),
                "markers_incompatible_alleles": fitted["reason_counts"][
                    "INCOMPATIBLE_ALLELE_SET"
                ],
                "markers_failed_call_rate": fitted["reason_counts"][
                    "LOW_TRAINING_CALL_RATE"
                ],
                "markers_monomorphic": fitted["reason_counts"][
                    "MONOMORPHIC_IN_TRAINING"
                ],
                "markers_failed_MAF": fitted["reason_counts"]["LOW_TRAINING_MAF"],
                "markers_failed_heterozygosity": fitted["reason_counts"][
                    "HIGH_TRAINING_HETEROZYGOSITY"
                ],
                "retained_pre_qc_only_markers": fitted["retained_pre_qc_only"],
                "retained_globally_filtered_HMP_markers": fitted[
                    "retained_globally_filtered"
                ],
                "vanraden_denominator": fitted["denominator"],
                "mean_training_diagonal_scale": fitted["diagonal_scale"],
                "certified_training_diagonal_mean": fitted["training_diag_mean"],
                "training_kernel_symmetry_residual": fitted["symmetry_residual"],
                "training_kernel_minimum_eigenvalue": fitted["minimum_eigenvalue"],
                "training_kernel_PSD_tolerance": fitted["psd_tolerance"],
                "training_kernel_effective_rank": fitted["effective_rank"],
                "training_center_residual": fitted["training_center_residual"],
                "projection_training_block_residual": fitted["projection_residual"],
                "sample_QC_fit_scope": "TRAINING_PARTITION_PANEL_SAMPLES_ONLY",
                "marker_QC_fit_scope": "SAMPLE_QC_PASSING_TRAINING_GIDS_ONLY",
                "imputation_fit_scope": "TRAINING_MARKER_MEAN_2P_ONLY",
                "held_out_calls_used_for_fit": False,
                "globally_filtered_marker_availability_used": False,
                "kernel_artifact_path": fitted["artifact_path"].relative_to(output).as_posix(),
                "kernel_artifact_sha256": fitted["artifact_sha256"],
                "strict_production_eligible": strict,
                "status": "PASS_STRICT_SPLIT_LOCAL_K_G" if strict else "MASKED_QC_OR_SUPPORT",
            }
        )
        if number == 1 or number % 10 == 0:
            print(f"CIMMYT production K_G: {number}/{len(states)} states", flush=True)
    return pd.DataFrame(registry_rows)


def artifact_manifest(output: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.tsv", "artifacts.sha256"}:
            rows.append(
                {
                    "relative_path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    code_root = (
        args.code_root or Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root))
    ).resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"CIMMYT production K_G release already exists: {output}")
        if not output.is_relative_to(root / "audit/v2"):
            raise ValueError(f"Refusing to replace output outside audit/v2: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    protocol_path = code_root / PROTOCOL_RELATIVE
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_version"] != "cimmyt_pre_qc_production_kg_v2":
        raise ValueError("Unexpected CIMMYT production K_G protocol")
    protocol_sha256 = sha256_file(protocol_path)
    source_decision = json.loads((root / SOURCE_DECISION).read_text(encoding="utf-8"))
    if source_decision["status"] != "PASS_CIMMYT_PRE_QC_SPLIT_LOCAL_150_STATE_CERTIFIED":
        raise ValueError("Source CIMMYT pre-QC release is not certified")
    source_call_path = root / SOURCE_CALLS
    source_call_hash_before = sha256_file(source_call_path)
    if source_call_hash_before != protocol["source_call_matrix_sha256"]:
        raise ValueError("Certified CIMMYT raw-call matrix checksum changed")
    dosage = np.load(source_call_path, mmap_mode="r")
    if dosage.shape != (
        int(protocol["source_marker_count"]),
        int(protocol["primary_stage1_gid_count"]),
    ):
        raise ValueError(f"Unexpected CIMMYT raw-call shape: {dosage.shape}")
    markers = pd.read_parquet(root / SOURCE_MARKERS)
    primary_axis = pd.read_csv(root / SOURCE_SAMPLES, sep="\t", dtype=str)
    if primary_axis.canonical_gid.duplicated().any():
        raise ValueError("Duplicate canonical GIDs in CIMMYT primary production axis")

    print("AUDIT CIMMYT sample identity and technical-replicate candidates", flush=True)
    identity, replicate_audit, identity_summary = build_identity_and_replicate_audit(
        root, output, protocol, primary_axis
    )
    write_tsv(output / "identity/cimmyt_source_sample_identity.tsv", identity)
    write_tsv(output / "identity/cimmyt_technical_replicate_concordance.tsv", replicate_audit)
    write_json(output / "identity/cimmyt_identity_summary.json", identity_summary)

    print("AUDIT pre-QC versus filtered-HMP allele orientation", flush=True)
    filtered = pd.read_parquet(root / FILTERED_MARKERS)
    orientation = build_orientation_audit(markers, filtered)
    (output / "markers").mkdir(parents=True, exist_ok=True)
    orientation.to_parquet(
        output / "markers/cimmyt_marker_allele_orientation.parquet",
        index=False,
        compression="zstd",
    )
    incompatible = sorted(
        orientation.loc[
            orientation.allele_relation.eq("INCOMPATIBLE_ALLELE_SET"), "marker_id"
        ].astype(str)
    )
    known_incompatible = sorted(protocol["allele_policy"]["incompatible_shared_markers"])
    if incompatible != known_incompatible:
        raise ValueError(
            f"Known incompatible CIMMYT markers changed: {incompatible} != {known_incompatible}"
        )

    structural = orientation.production_marker_structurally_eligible.to_numpy(bool)
    observed_counts = np.zeros(dosage.shape[1], dtype=np.int64)
    for start in range(0, dosage.shape[0], 4096):
        stop = min(dosage.shape[0], start + 4096)
        eligible = structural[start:stop]
        if eligible.any():
            block = np.asarray(dosage[start:stop][eligible], dtype=np.uint8)
            observed_counts += (block != MISSING).sum(axis=0, dtype=np.int64)
    sample_rates = observed_counts / int(structural.sum())
    sample_qc = primary_axis.copy()
    sample_qc["structurally_eligible_marker_count"] = int(structural.sum())
    sample_qc["observed_structurally_eligible_calls"] = observed_counts
    sample_qc["structurally_eligible_call_rate"] = sample_rates
    (output / "samples").mkdir(parents=True, exist_ok=True)
    write_tsv(output / "samples/cimmyt_primary_sample_call_rates.tsv", sample_qc)

    print("FIT and certify exact split-local K_G artifacts", flush=True)
    registry = build_states(
        root,
        output,
        protocol,
        protocol_sha256,
        dosage,
        markers,
        primary_axis,
        sample_rates,
        orientation,
    )
    write_tsv(output / "states/cimmyt_production_kg_state_registry.tsv", registry)
    source_call_hash_after = sha256_file(source_call_path)

    orientation_counts = orientation.allele_relation.value_counts().to_dict()
    strict_states = int(registry.strict_production_eligible.astype(bool).sum())
    checks = {
        "protocol_frozen": protocol["protocol_version"]
        == "cimmyt_pre_qc_production_kg_v2",
        "source_release_certified": source_decision["status"]
        == "PASS_CIMMYT_PRE_QC_SPLIT_LOCAL_150_STATE_CERTIFIED",
        "source_call_matrix_exact": source_call_hash_before
        == protocol["source_call_matrix_sha256"],
        "source_call_matrix_unchanged": source_call_hash_after == source_call_hash_before,
        "source_sample_ids_unique": identity_summary["duplicate_source_sample_ids"] == 0,
        "exact_primary_axis_one_to_one": identity_summary["exact_primary_stage1_mappings"]
        == len(primary_axis)
        == int(protocol["primary_stage1_gid_count"]),
        "replicate_candidates_audited": identity_summary["replicate_candidate_pairs"]
        == len(replicate_audit)
        and len(replicate_audit) > 0,
        "replicate_candidates_not_used_for_production": identity_summary[
            "replicate_pairs_used_in_production"
        ]
        == 0,
        "known_incompatible_alleles_exact": incompatible == known_incompatible,
        "all_source_markers_enter_before_qc": registry.source_marker_rows_considered.eq(
            int(protocol["source_marker_count"])
        ).all(),
        "all_states_training_only_sample_qc": registry.sample_QC_fit_scope.eq(
            "TRAINING_PARTITION_PANEL_SAMPLES_ONLY"
        ).all(),
        "all_states_training_only_marker_qc": registry.marker_QC_fit_scope.eq(
            "SAMPLE_QC_PASSING_TRAINING_GIDS_ONLY"
        ).all(),
        "all_states_training_only_imputation": registry.imputation_fit_scope.eq(
            "TRAINING_MARKER_MEAN_2P_ONLY"
        ).all(),
        "held_out_calls_unused_for_fit": (~registry.held_out_calls_used_for_fit.astype(bool)).all(),
        "no_global_filter_controls_availability": (
            ~registry.globally_filtered_marker_availability_used.astype(bool)
        ).all()
        and registry.retained_pre_qc_only_markers.gt(0).all(),
        "five_incompatible_markers_excluded_in_every_state": registry.markers_incompatible_alleles.eq(
            5
        ).all(),
        "state_count_exact": len(registry) == int(protocol["required_state_count"]),
        "all_states_strict_production_ready": strict_states
        == int(protocol["required_state_count"]),
        "all_kernel_axes_unique": registry.training_panel_GIDs_after_sample_QC.le(
            registry.projection_supported_GIDs
        ).all(),
        "all_kernels_scaled": np.allclose(
            registry.certified_training_diagonal_mean.to_numpy(float), 1.0, atol=1e-5
        ),
        "all_kernels_symmetric": registry.training_kernel_symmetry_residual.le(1e-6).all(),
        "all_kernels_PSD": (
            registry.training_kernel_minimum_eigenvalue
            >= -registry.training_kernel_PSD_tolerance
        ).all(),
        "all_kernels_have_rank": registry.training_kernel_effective_rank.ge(2).all(),
        "all_held_out_projections_certified": registry.projection_training_block_residual.le(
            1e-6
        ).all(),
        "current_frozen_models_unchanged": protocol["current_frozen_models_modified"]
        is False,
        "phenotype_and_evaluation_outcomes_unread": all(
            protocol[key] is False
            for key in [
                "phenotype_values_read",
                "inner_validation_metrics_read",
                "outer_test_outcomes_read",
                "outer_test_metrics_read",
                "final_holdout_outcomes_read",
            ]
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    check_frame = pd.DataFrame(
        [
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": ""}
            for name, passed in checks.items()
        ]
    )
    write_tsv(output / "validation_checks.tsv", check_frame)

    summary = {
        "status": (
            "PASS_CIMMYT_PRE_QC_PRODUCTION_KG_V2"
            if not failed
            else "BLOCKED_CIMMYT_PRE_QC_PRODUCTION_KG_V2"
        ),
        "protocol_version": protocol["protocol_version"],
        "selection_data": protocol["selection_data"],
        "code_commit": git_commit(code_root),
        "protocol_sha256": protocol_sha256,
        "source_call_matrix_sha256": source_call_hash_before,
        "source_call_cells": int(np.prod(dosage.shape)),
        "source_marker_count": int(dosage.shape[0]),
        "source_sample_columns": identity_summary["source_sample_count"],
        "primary_stage1_GIDs": len(primary_axis),
        "technical_replicate_candidate_groups": identity_summary[
            "replicate_candidate_groups"
        ],
        "technical_replicate_candidate_pairs": identity_summary["replicate_candidate_pairs"],
        "allele_orientation_counts": {
            key: int(value) for key, value in sorted(orientation_counts.items())
        },
        "known_incompatible_markers_excluded": incompatible,
        "state_count": len(registry),
        "unique_training_fits": int(registry.fit_signature.nunique()),
        "strict_ready_states": strict_states,
        "masked_states": len(registry) - strict_states,
        "sample_call_rate_threshold_min": float(registry.sample_call_rate_threshold.min()),
        "sample_call_rate_threshold_median": float(
            registry.sample_call_rate_threshold.median()
        ),
        "sample_call_rate_threshold_max": float(registry.sample_call_rate_threshold.max()),
        "training_supported_GIDs_min": int(
            registry.training_panel_GIDs_after_sample_QC.min()
        ),
        "training_supported_GIDs_median": float(
            registry.training_panel_GIDs_after_sample_QC.median()
        ),
        "training_supported_GIDs_max": int(
            registry.training_panel_GIDs_after_sample_QC.max()
        ),
        "retained_markers_min": int(registry.retained_marker_rows.min()),
        "retained_markers_median": float(registry.retained_marker_rows.median()),
        "retained_markers_max": int(registry.retained_marker_rows.max()),
        "effective_rank_min": int(registry.training_kernel_effective_rank.min()),
        "effective_rank_median": float(registry.training_kernel_effective_rank.median()),
        "effective_rank_max": int(registry.training_kernel_effective_rank.max()),
        "current_frozen_models_modified": False,
        "eligible_for_current_frozen_model_retrofit": False,
        "production_disposition": (
            "READY_FOR_NEW_PREREGISTERED_PHASE6_CANDIDATE"
            if not failed
            else "NOT_READY"
        ),
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "checks": checks,
        "failed_checks": failed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / DECISION_NAME, summary)
    report = f"""# CIMMYT pre-QC production K_G v2

Status: `{summary['status']}`

This phenotype-blind release constructs exact, training-local VanRaden relationship
kernels from the certified {summary['source_call_cells']:,}-call pre-QC array. It does
not modify any frozen model or inspect phenotype values or evaluation outcomes.

## Identity

- Source sample columns: {summary['source_sample_columns']:,}
- Exact Stage-1 v2 GID mappings: {summary['primary_stage1_GIDs']:,}
- Candidate technical-replicate groups: {summary['technical_replicate_candidate_groups']}
- Candidate replicate pairs audited: {summary['technical_replicate_candidate_pairs']}
- Candidate suffix mappings admitted to production: 0

## Alleles and marker QC

- Pre-QC markers: {summary['source_marker_count']:,}
- Shared reversed-order markers: {orientation_counts.get('REVERSED_ORDER', 0):,}
- Incompatible shared markers excluded in every state: {len(incompatible)}
- Retained markers by state: {summary['retained_markers_min']:,} to
  {summary['retained_markers_max']:,} (median {summary['retained_markers_median']:,.0f})

All marker missingness, MAF, heterozygosity, monomorphism, allele frequency,
imputation means, VanRaden denominators and diagonal scales were fitted from the
sample-QC-passing training GIDs in each frozen state. The globally filtered HMP
membership was not used to determine marker availability.

## State-local K_G

- Frozen states: {summary['state_count']}
- Unique training fits: {summary['unique_training_fits']}
- Strict-ready states: {summary['strict_ready_states']}
- Masked states: {summary['masked_states']}
- Training support: {summary['training_supported_GIDs_min']} to
  {summary['training_supported_GIDs_max']} GIDs
- Effective rank: {summary['effective_rank_min']} to {summary['effective_rank_max']}

Every artifact contains the exact training K_G, its training and transformed
projection axes, the held-out-to-training relationship block, sample support mask,
complete marker QC reason vector, training allele frequencies and imputation
parameters. Symmetry, PSD, mean-diagonal scaling, rank and projection consistency
were certified after float32 storage.

Disposition: `{summary['production_disposition']}`. This component may enter only a
new preregistered model release; it is not retroactively inserted into frozen results.
"""
    (output / REPORT_NAME).write_text(report, encoding="utf-8")

    input_paths = [
        protocol_path,
        root / SOURCE_DECISION,
        source_call_path,
        root / SOURCE_MARKERS,
        root / SOURCE_SAMPLES,
        root / FILTERED_MARKERS,
        root / SOURCE_ORDER_RELATIVE,
        root / ARCHIVE_RELATIVE,
    ]
    input_inventory = pd.DataFrame(
        [
            {
                "relative_path": path.relative_to(root).as_posix()
                if path.is_relative_to(root)
                else str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "access_scope": "IDENTIFIERS_OR_GENOTYPE_CALLS_ONLY",
            }
            for path in input_paths
        ]
    )
    write_tsv(output / "input_inventory.tsv", input_inventory)
    manifest = artifact_manifest(output)
    write_tsv(output / "artifact_manifest.tsv", manifest)
    (output / "artifacts.sha256").write_text(
        "".join(f"{row.sha256}  {row.relative_path}\n" for row in manifest.itertuples()),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"CIMMYT production K_G v2 certification failed: {failed}")


if __name__ == "__main__":
    main()
