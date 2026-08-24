from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


def stable_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def index_signature(values: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "pass"}
    )


def optional_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def observed_pedigree_registry(node_registry: pd.DataFrame) -> pd.DataFrame:
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
    if local["node_index"].tolist() != list(range(len(local))):
        raise ValueError("Pedigree node indices are not the compact stored operator order")
    if local["node_id"].fillna("").astype(str).duplicated().any():
        raise ValueError("Pedigree node IDs are not unique")
    local["is_observed_gid"] = boolean_series(local["is_observed_gid"])
    observed = local.loc[local["is_observed_gid"]].copy()
    observed["node_id"] = observed["node_id"].astype(str)
    if not observed["node_id"].str.startswith("GID").all():
        raise ValueError("Observed pedigree registry contains non-canonical GID nodes")
    return observed.sort_values("node_id").reset_index(drop=True)


def build_state_binding(
    state: Mapping[str, object],
    training_gids: set[str],
    observed_registry: pd.DataFrame,
    *,
    entity_order_path: str,
    raw_operator_factor: str,
    raw_operator_d: str,
) -> tuple[pd.DataFrame, dict[str, object], list[int]]:
    observed_gids = observed_registry["node_id"].astype(str).tolist()
    observed_set = set(observed_gids)
    training = sorted(observed_set.intersection(training_gids))
    application = sorted(observed_set.difference(training))
    diagonal_by_gid = observed_registry.set_index("node_id")[
        "raw_relationship_diagonal"
    ].astype(float)
    if not training:
        raise ValueError(f"K_A state has no training-supported GIDs: {state['state_id']}")
    scalar = float(diagonal_by_gid.loc[training].mean())
    if not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"K_A state has invalid training scale: {state['state_id']}")

    entity_frame = pd.DataFrame(
        {
            "entity_index": np.arange(len(observed_gids), dtype=np.int64),
            "canonical_gid": observed_gids,
        }
    )
    entity_frame["partition"] = np.where(
        entity_frame["canonical_gid"].isin(training), "TRAINING", "APPLICATION"
    )
    row = {
        "state_id": str(state["state_id"]),
        "scenario": str(state["scenario"]),
        "outer_fold": str(state["outer_fold"]),
        "inner_fold": optional_text(state.get("inner_fold")),
        "training_observed_gids": len(training),
        "application_observed_gids": len(application),
        "raw_operator_factor": raw_operator_factor,
        "raw_operator_d": raw_operator_d,
        "training_scale_mean_diagonal": scalar,
        "entity_order_path": entity_order_path,
        "entity_order_signature": index_signature(entity_frame["canonical_gid"].tolist()),
        "state_hash": stable_json_hash(
            {
                "state": str(state["state_id"]),
                "training": training,
                "application": application,
                "scale": scalar,
            }
        ),
        "status": "PASS",
    }
    training_node_indices = (
        observed_registry.set_index("node_id").loc[training, "node_index"].astype(int).tolist()
    )
    return entity_frame, row, training_node_indices


def relationship_block(
    factor: sparse.csr_matrix,
    d_values: np.ndarray,
    indices: Sequence[int],
) -> np.ndarray:
    block_factor = factor[np.asarray(indices, dtype=int), :]
    weighted = block_factor.multiply(np.sqrt(d_values))
    return np.asarray((weighted @ weighted.T).toarray(), dtype=np.float64)


def diagnose_state_binding(
    state_id: str,
    factor: sparse.csr_matrix,
    d_values: np.ndarray,
    training_node_indices: Sequence[int],
    scale: float,
) -> dict[str, object]:
    sample = list(training_node_indices[: min(256, len(training_node_indices))])
    block = relationship_block(factor, d_values, sample) / scale
    symmetric = (block + block.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetric)
    all_finite = bool(np.isfinite(block).all())
    max_symmetry_error = float(np.max(np.abs(block - block.T)))
    minimum_eigenvalue = float(np.min(eigenvalues))
    status = (
        "PASS"
        if sample
        and all_finite
        and max_symmetry_error <= 1e-10
        and minimum_eigenvalue >= -1e-8
        else "FAIL"
    )
    return {
        "state_id": state_id,
        "sample_dimension": len(sample),
        "all_finite": all_finite,
        "max_symmetry_error": max_symmetry_error,
        "minimum_eigenvalue": minimum_eigenvalue,
        "mean_diagonal": float(np.mean(np.diag(block))),
        "raw_training_scale": scale,
        "status": status,
    }


def certify_frozen_operator(
    factor: sparse.csr_matrix,
    d_values: np.ndarray,
    node_registry: pd.DataFrame,
    observed_registry: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": detail})

    add(
        "operator_dimensions",
        factor.shape == (len(node_registry), len(node_registry)) and len(d_values) == len(node_registry),
        f"factor={factor.shape}; d={len(d_values)}; nodes={len(node_registry)}",
    )
    add(
        "operator_values_finite",
        bool(np.isfinite(factor.data).all() and np.isfinite(d_values).all()),
        f"factor_nonfinite={int((~np.isfinite(factor.data)).sum())}; d_nonfinite={int((~np.isfinite(d_values)).sum())}",
    )
    add(
        "mendelian_variances_positive",
        bool((d_values > 0).all()),
        f"minimum={float(np.min(d_values)):.12g}; maximum={float(np.max(d_values)):.12g}",
    )
    add(
        "observed_pedigree_gid_count",
        len(observed_registry) == 8762,
        f"observed={len(observed_registry)}; expected=8762",
    )
    sample_indices = observed_registry["node_index"].astype(int).tolist()[:256]
    block = relationship_block(factor, d_values, sample_indices)
    expected_diagonal = observed_registry["raw_relationship_diagonal"].astype(float).to_numpy()[:256]
    diagonal_delta = float(np.max(np.abs(np.diag(block) - expected_diagonal)))
    add(
        "stored_diagonal_reproduced",
        diagonal_delta <= 1e-12,
        f"sample=256; max_abs_delta={diagonal_delta:.12g}",
    )
    eigenvalue = float(np.min(np.linalg.eigvalsh((block + block.T) / 2.0)))
    add(
        "sampled_relationship_psd",
        eigenvalue >= -1e-8,
        f"sample=256; minimum_eigenvalue={eigenvalue:.12g}",
    )
    return pd.DataFrame(checks)


def compare_replayed_binding(
    generated_row: Mapping[str, object],
    generated_entities: pd.DataFrame,
    frozen_row: Mapping[str, object],
    frozen_entities: pd.DataFrame,
) -> dict[str, object]:
    scalar_delta = abs(
        float(generated_row["training_scale_mean_diagonal"])
        - float(frozen_row["training_scale_mean_diagonal"])
    )
    entity_match = generated_entities.equals(frozen_entities)
    checks = {
        "training_count_match": int(generated_row["training_observed_gids"])
        == int(frozen_row["training_observed_gids"]),
        "application_count_match": int(generated_row["application_observed_gids"])
        == int(frozen_row["application_observed_gids"]),
        "scale_match": scalar_delta <= 1e-15,
        "order_signature_match": str(generated_row["entity_order_signature"])
        == str(frozen_row["entity_order_signature"]),
        "state_hash_match": str(generated_row["state_hash"]) == str(frozen_row["state_hash"]),
        "entity_partition_match": entity_match,
    }
    return {
        "state_id": str(generated_row["state_id"]),
        **checks,
        "scale_absolute_delta": scalar_delta,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def validate_combined_registry(combined: pd.DataFrame, state_registry: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": detail})

    required_ids = set(state_registry["state_id"].astype(str))
    observed_ids = set(combined["state_id"].astype(str))
    add(
        "combined_state_grid",
        len(combined) == 150 and combined["state_id"].is_unique and observed_ids == required_ids,
        f"rows={len(combined)}; unique={combined['state_id'].nunique()}; required={len(required_ids)}",
    )
    counts = combined.groupby("scenario")["state_id"].size().to_dict()
    add(
        "scenario_state_counts",
        len(counts) == 5 and set(counts.values()) == {30},
        json.dumps(counts, sort_keys=True),
    )
    source_counts = combined.groupby("binding_source")["state_id"].size().to_dict()
    add(
        "binding_source_counts",
        source_counts == {"IMMUTABLE_PHASE5": 90, "TEMPORAL_COUNTRY_EXTENSION": 60},
        json.dumps(source_counts, sort_keys=True),
    )
    add(
        "all_bindings_pass",
        bool(combined["status"].eq("PASS").all()),
        f"passing={int(combined['status'].eq('PASS').sum())}/{len(combined)}",
    )
    scales = pd.to_numeric(combined["training_scale_mean_diagonal"], errors="coerce")
    add(
        "training_scales_positive_finite",
        bool(np.isfinite(scales).all() and (scales > 0).all()),
        f"minimum={float(scales.min()):.12g}; maximum={float(scales.max()):.12g}",
    )
    add(
        "common_operator_identity",
        combined["raw_operator_factor_root_relative"].nunique() == 1
        and combined["raw_operator_d_root_relative"].nunique() == 1,
        f"factor_paths={combined['raw_operator_factor_root_relative'].nunique()}; d_paths={combined['raw_operator_d_root_relative'].nunique()}",
    )
    return pd.DataFrame(checks)
