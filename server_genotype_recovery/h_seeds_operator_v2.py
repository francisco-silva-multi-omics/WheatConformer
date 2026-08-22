from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


MISSING_DOSAGE = 255


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def index_signature(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def stable_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "pass"}
    )


@dataclass(frozen=True)
class HSeedsProtocol:
    genomic_blend_weight: float = 0.95
    minimum_training_overlap: int = 20
    diagnostic_sample_size: int = 96
    marker_block_size: int = 1024

    def validate(self) -> None:
        if not 0.0 < self.genomic_blend_weight < 1.0:
            raise ValueError("Genomic blend weight must be strictly between zero and one")
        if self.minimum_training_overlap < 2:
            raise ValueError("Minimum training overlap must be at least two")
        if self.diagnostic_sample_size < 2:
            raise ValueError("Diagnostic sample size must be at least two")
        if self.marker_block_size < 1:
            raise ValueError("Marker block size must be positive")


def observed_pedigree_axis(node_registry: pd.DataFrame) -> pd.DataFrame:
    required = {
        "node_index",
        "node_id",
        "is_observed_gid",
        "raw_relationship_diagonal",
    }
    missing = sorted(required.difference(node_registry.columns))
    if missing:
        raise ValueError(f"Pedigree node registry lacks columns: {missing}")
    local = node_registry.copy()
    local["node_index"] = pd.to_numeric(local["node_index"], errors="raise").astype(int)
    local["raw_relationship_diagonal"] = pd.to_numeric(
        local["raw_relationship_diagonal"], errors="raise"
    ).astype(float)
    local["is_observed_gid"] = boolean_series(local["is_observed_gid"])
    observed = local.loc[local["is_observed_gid"]].copy()
    observed["canonical_gid"] = observed["node_id"].astype(str)
    if observed["canonical_gid"].duplicated().any():
        raise ValueError("Observed pedigree GIDs are not unique")
    if not observed["canonical_gid"].str.startswith("GID").all():
        raise ValueError("Observed pedigree axis contains a non-canonical GID")
    return observed.sort_values("canonical_gid").reset_index(drop=True)


def seeds_axis(consensus_summary: pd.DataFrame) -> pd.DataFrame:
    required = {"canonical_gid", "retained_for_component"}
    missing = sorted(required.difference(consensus_summary.columns))
    if missing:
        raise ValueError(f"Seeds consensus summary lacks columns: {missing}")
    local = consensus_summary.copy().reset_index(drop=True)
    local["seeds_row_index"] = np.arange(len(local), dtype=np.int32)
    local["retained_for_component"] = boolean_series(local["retained_for_component"])
    if local["canonical_gid"].astype(str).duplicated().any():
        raise ValueError("Seeds consensus GID axis is not unique")
    return local.loc[local["retained_for_component"]].copy()


def relationship_block(
    factor: sparse.csr_matrix,
    d_values: np.ndarray,
    node_indices: Sequence[int],
    scale: float,
) -> np.ndarray:
    selected = factor[np.asarray(node_indices, dtype=np.int64), :]
    weighted = selected.multiply(np.sqrt(d_values))
    block = np.asarray((weighted @ weighted.T).toarray(), dtype=np.float64)
    return block / float(scale)


def centered_genomic_rows(
    dosage: np.ndarray,
    row_indices: Sequence[int],
    marker_indices: np.ndarray,
    allele_frequency: np.ndarray,
    denominator: float,
    *,
    marker_block_size: int,
) -> np.ndarray:
    rows = np.asarray(row_indices, dtype=np.int64)
    if len(marker_indices) != len(allele_frequency):
        raise ValueError("Marker index and allele-frequency vectors disagree")
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("VanRaden denominator must be finite and positive")
    output = np.empty((len(rows), len(marker_indices)), dtype=np.float64)
    for start in range(0, len(marker_indices), marker_block_size):
        stop = min(len(marker_indices), start + marker_block_size)
        markers = marker_indices[start:stop]
        p = allele_frequency[start:stop]
        block = np.asarray(dosage[np.ix_(rows, markers)], dtype=np.float64)
        missing = block == MISSING_DOSAGE
        means = 2.0 * p
        block[missing] = np.broadcast_to(means, block.shape)[missing]
        block -= means
        output[:, start:stop] = block / np.sqrt(denominator)
    return output


def genomic_diagonal_mean(
    dosage: np.ndarray,
    row_indices: Sequence[int],
    marker_indices: np.ndarray,
    allele_frequency: np.ndarray,
    denominator: float,
    *,
    marker_block_size: int,
) -> float:
    rows = np.asarray(row_indices, dtype=np.int64)
    squared_norm = np.zeros(len(rows), dtype=np.float64)
    for start in range(0, len(marker_indices), marker_block_size):
        stop = min(len(marker_indices), start + marker_block_size)
        markers = marker_indices[start:stop]
        p = allele_frequency[start:stop]
        block = np.asarray(dosage[np.ix_(rows, markers)], dtype=np.float64)
        missing = block == MISSING_DOSAGE
        means = 2.0 * p
        block[missing] = np.broadcast_to(means, block.shape)[missing]
        block -= means
        squared_norm += np.einsum("ij,ij->i", block, block)
    value = float(np.mean(squared_norm / denominator))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Training genomic mean diagonal is not finite and positive")
    return value


def build_state_binding(
    *,
    state_id: str,
    scenario: str,
    state_level: str,
    training_gids: set[str],
    observed_axis: pd.DataFrame,
    panel_axis: pd.DataFrame,
    dosage: np.ndarray,
    marker_indices: np.ndarray,
    allele_frequency: np.ndarray,
    denominator: float,
    pedigree_factor: sparse.csr_matrix,
    mendelian_variance: np.ndarray,
    ka_scale: float,
    ka_state_hash: str,
    seeds_state_hash: str,
    protocol: HSeedsProtocol,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    protocol.validate()
    observed = observed_axis.set_index("canonical_gid", drop=False)
    panel = panel_axis.set_index("canonical_gid", drop=False)
    overlap_gids = sorted(set(observed.index).intersection(panel.index))
    training_overlap = [gid for gid in overlap_gids if gid in training_gids]
    operator_pedigree_nodes = observed.loc[overlap_gids, "node_index"].astype(int).to_numpy()
    operator_seeds_rows = panel.loc[overlap_gids, "seeds_row_index"].astype(int).to_numpy()
    if len(training_overlap) < protocol.minimum_training_overlap:
        arrays = {
            "pedigree_node_indices": operator_pedigree_nodes.astype(np.int32),
            "seeds_row_indices": operator_seeds_rows.astype(np.int32),
            "training_overlap_mask": np.asarray(
                [gid in training_gids for gid in overlap_gids], dtype=np.bool_
            ),
            "genomic_scale": np.asarray([np.nan], dtype=np.float64),
            "genomic_blend_weight": np.asarray(
                [protocol.genomic_blend_weight], dtype=np.float64
            ),
            "pedigree_training_scale": np.asarray([ka_scale], dtype=np.float64),
            "overlap_gid_signature": np.asarray(
                [index_signature(overlap_gids)], dtype="U64"
            ),
        }
        row = {
            "state_id": state_id,
            "scenario": scenario,
            "state_level": state_level,
            "operator_representation": "K_A_BACKBONE_WITH_MASKED_SINGLE_STEP_CORRECTION",
            "formula": "H^-1=A^-1 when Seeds training overlap is below the frozen minimum",
            "panel_id": "seeds_of_discovery_dartseq",
            "operator_overlap_gids": len(overlap_gids),
            "training_overlap_gids": len(training_overlap),
            "retained_markers": len(marker_indices),
            "pedigree_training_scale": ka_scale,
            "genomic_training_diagonal_mean_before_alignment": np.nan,
            "pedigree_training_diagonal_mean": np.nan,
            "genomic_alignment_scale": np.nan,
            "genomic_blend_weight": protocol.genomic_blend_weight,
            "pedigree_blend_weight": 1.0 - protocol.genomic_blend_weight,
            "sample_dimension": 0,
            "sample_minimum_blended_eigenvalue": np.nan,
            "sample_maximum_symmetry_error": np.nan,
            "ka_state_hash": ka_state_hash,
            "seeds_state_hash": seeds_state_hash,
            "overlap_gid_signature": index_signature(overlap_gids),
            "state_hash": stable_json_hash(
                {
                    "state_id": state_id,
                    "ka_state_hash": ka_state_hash,
                    "seeds_state_hash": seeds_state_hash,
                    "overlap": overlap_gids,
                    "training_overlap": training_overlap,
                    "minimum_training_overlap": protocol.minimum_training_overlap,
                    "component_available": False,
                }
            ),
            "component_available": False,
            "absence_mask": "SEEDS_TRAINING_PEDIGREE_OVERLAP_LT20_KA_BACKBONE_RETAINED",
            "status": "PASS_MASKED_INSUFFICIENT_TRAINING_OVERLAP",
        }
        return row, arrays
    training_seed_rows = panel.loc[training_overlap, "seeds_row_index"].astype(int).to_numpy()
    training_nodes = observed.loc[training_overlap, "node_index"].astype(int).to_numpy()
    a_training_diagonal = (
        observed.loc[training_overlap, "raw_relationship_diagonal"].astype(float).to_numpy()
        / float(ka_scale)
    )
    a_diagonal_mean = float(np.mean(a_training_diagonal))
    g_diagonal_mean = genomic_diagonal_mean(
        dosage,
        training_seed_rows,
        marker_indices,
        allele_frequency,
        denominator,
        marker_block_size=protocol.marker_block_size,
    )
    genomic_scale = a_diagonal_mean / g_diagonal_mean
    if not np.isfinite(genomic_scale) or genomic_scale <= 0:
        raise ValueError(f"H_SEEDS state {state_id} has invalid genomic scale")

    sample_gids = training_overlap[: protocol.diagnostic_sample_size]
    sample_seed_rows = panel.loc[sample_gids, "seeds_row_index"].astype(int).to_numpy()
    sample_nodes = observed.loc[sample_gids, "node_index"].astype(int).to_numpy()
    z_sample = centered_genomic_rows(
        dosage,
        sample_seed_rows,
        marker_indices,
        allele_frequency,
        denominator,
        marker_block_size=protocol.marker_block_size,
    )
    g_sample = genomic_scale * (z_sample @ z_sample.T)
    a_sample = relationship_block(
        factor=pedigree_factor,
        d_values=mendelian_variance,
        node_indices=sample_nodes,
        scale=ka_scale,
    )
    blended = (
        protocol.genomic_blend_weight * g_sample
        + (1.0 - protocol.genomic_blend_weight) * a_sample
    )
    blended = (blended + blended.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(blended)
    minimum_eigenvalue = float(np.min(eigenvalues))
    maximum_symmetry_error = float(np.max(np.abs(blended - blended.T)))
    if not np.isfinite(blended).all() or maximum_symmetry_error > 1e-10:
        raise ValueError(f"H_SEEDS state {state_id} sampled blend is invalid")
    if minimum_eigenvalue <= 0:
        raise ValueError(f"H_SEEDS state {state_id} sampled blend is not positive definite")

    arrays = {
        "pedigree_node_indices": operator_pedigree_nodes.astype(np.int32),
        "seeds_row_indices": operator_seeds_rows.astype(np.int32),
        "training_overlap_mask": np.asarray(
            [gid in training_gids for gid in overlap_gids], dtype=np.bool_
        ),
        "genomic_scale": np.asarray([genomic_scale], dtype=np.float64),
        "genomic_blend_weight": np.asarray(
            [protocol.genomic_blend_weight], dtype=np.float64
        ),
        "pedigree_training_scale": np.asarray([ka_scale], dtype=np.float64),
        "overlap_gid_signature": np.asarray([index_signature(overlap_gids)], dtype="U64"),
    }
    row = {
        "state_id": state_id,
        "scenario": scenario,
        "state_level": state_level,
        "operator_representation": "SINGLE_STEP_PRECISION_UPDATE_ON_DEMAND",
        "formula": "H^-1=A^-1+S'(G_blend^-1-A22^-1)S",
        "panel_id": "seeds_of_discovery_dartseq",
        "operator_overlap_gids": len(overlap_gids),
        "training_overlap_gids": len(training_overlap),
        "retained_markers": len(marker_indices),
        "pedigree_training_scale": ka_scale,
        "genomic_training_diagonal_mean_before_alignment": g_diagonal_mean,
        "pedigree_training_diagonal_mean": a_diagonal_mean,
        "genomic_alignment_scale": genomic_scale,
        "genomic_blend_weight": protocol.genomic_blend_weight,
        "pedigree_blend_weight": 1.0 - protocol.genomic_blend_weight,
        "sample_dimension": len(sample_gids),
        "sample_minimum_blended_eigenvalue": minimum_eigenvalue,
        "sample_maximum_symmetry_error": maximum_symmetry_error,
        "ka_state_hash": ka_state_hash,
        "seeds_state_hash": seeds_state_hash,
        "overlap_gid_signature": index_signature(overlap_gids),
        "state_hash": stable_json_hash(
            {
                "state_id": state_id,
                "ka_state_hash": ka_state_hash,
                "seeds_state_hash": seeds_state_hash,
                "overlap": overlap_gids,
                "training_overlap": training_overlap,
                "genomic_scale": genomic_scale,
                "genomic_blend_weight": protocol.genomic_blend_weight,
            }
        ),
        "component_available": True,
        "absence_mask": "GID_NOT_IN_PEDIGREE_AND_ACCEPTED_SEEDS_OVERLAP",
        "status": "PASS",
    }
    return row, arrays
