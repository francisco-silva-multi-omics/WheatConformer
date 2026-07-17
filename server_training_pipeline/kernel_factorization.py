from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


STRICT_INDUCTIVE_SPLIT_MODES = {
    "gho_environment",
    "cv1_genotype",
    "cv1_environment",
    "cv0_genotype_environment",
}


def effective_factorization_mode(requested_mode: str, split_mode: str, warn: bool = False) -> str:
    if requested_mode == "train_nystrom" and split_mode not in STRICT_INDUCTIVE_SPLIT_MODES:
        if warn:
            warnings.warn(
                f"train_nystrom is restricted to held-out genotype/environment benchmarking; "
                f"using full_transductive for {split_mode!r}.",
                stacklevel=2,
            )
        return "full_transductive"
    return requested_mode


def factorization_training_support(
    train_ids: np.ndarray,
    factorization_mode: str,
    center: bool,
    minimum_ids: int = 2,
) -> tuple[bool, str]:
    """Return whether a kernel expert is estimable from a fold's training entities."""
    unique_ids = np.unique(np.asarray(train_ids, dtype=np.int32))
    if unique_ids.size == 0:
        return False, "no_eligible_training_kernel_ids"
    required_ids = max(2, int(minimum_ids))
    if unique_ids.size < required_ids:
        if required_ids > 2:
            return False, f"expert_requires_at_least_{required_ids}_training_ids"
        if factorization_mode == "train_nystrom" and center:
            return False, "centered_train_nystrom_requires_at_least_two_training_ids"
        return False, "expert_requires_at_least_two_training_ids"
    return True, ""


def kernel_factors(
    path: Path,
    rank: int,
    train_ids: np.ndarray | None = None,
    jitter: float = 0.0,
    center: bool = False,
) -> tuple[np.ndarray, dict[str, int | str]]:
    K = np.load(path).astype(np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Kernel must be square: {path} has shape {K.shape}")
    if not np.isfinite(K).all():
        raise ValueError(f"Kernel contains non-finite values: {path}")
    K = (K + K.T) / 2.0
    if jitter > 0:
        K.flat[:: K.shape[0] + 1] += jitter
    if train_ids is None:
        train_ids = np.arange(K.shape[0], dtype=np.int32)
        K_train_raw = K
        factorization_mode = "full_transductive"
    else:
        train_ids = np.unique(np.asarray(train_ids, dtype=np.int32))
        if train_ids.size == 0:
            raise ValueError("train_nystrom requires at least one training kernel ID")
        if train_ids.min() < 0 or train_ids.max() >= K.shape[0]:
            raise ValueError(f"Training kernel IDs are outside kernel dimensions for {path}")
        K_train_raw = K[np.ix_(train_ids, train_ids)]
        factorization_mode = "train_nystrom"

    if center:
        train_column_mean = K_train_raw.mean(axis=0)
        train_grand_mean = float(K_train_raw.mean())
        K_train = (
            K_train_raw
            - K_train_raw.mean(axis=1, keepdims=True)
            - train_column_mean[None, :]
            + train_grand_mean
        )
    else:
        train_column_mean = np.zeros(K_train_raw.shape[1], dtype=np.float64)
        train_grand_mean = 0.0
        K_train = K_train_raw

    vals, vecs = np.linalg.eigh(K_train)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    maximum_eigenvalue = float(vals[0]) if vals.size else float("nan")
    keep = vals > 1e-8
    vals = vals[keep][:rank]
    vecs = vecs[:, keep][:, :rank]
    if vals.size == 0:
        raise ValueError(
            f"Kernel has no positive eigenvalues above tolerance: {path}; "
            f"factorization_mode={factorization_mode}; "
            f"train_kernel_dimension={K_train.shape[0]}; centered={center}; "
            f"jitter={jitter}; maximum_eigenvalue={maximum_eigenvalue:.8g}"
        )
    if factorization_mode == "full_transductive":
        factors = vecs * np.sqrt(vals)[None, :]
    else:
        cross_kernel = K[:, train_ids]
        if center:
            cross_kernel = (
                cross_kernel
                - cross_kernel.mean(axis=1, keepdims=True)
                - train_column_mean[None, :]
                + train_grand_mean
            )
        factors = cross_kernel @ (vecs / np.sqrt(vals)[None, :])
    metadata = {
        "factorization_mode": factorization_mode,
        "rank_requested": int(rank),
        "rank_retained": int(vals.size),
        "train_kernel_dimension": int(train_ids.size),
        "kernel_dimension": int(K.shape[0]),
        "kernel_centered": str(bool(center)).lower(),
    }
    factors = factors.astype(np.float32)
    if not np.isfinite(factors).all():
        raise ValueError(f"Kernel factorization produced non-finite values: {path}")
    return factors, metadata


def top_factors(path: Path, rank: int, jitter: float = 0.0) -> np.ndarray:
    factors, _ = kernel_factors(path, rank, jitter=jitter)
    return factors


def retained_eigenvalues(factors: np.ndarray, train_ids: np.ndarray | None = None) -> np.ndarray:
    rows = factors if train_ids is None else factors[np.asarray(train_ids, dtype=np.int32)]
    return np.sum(np.square(rows.astype(np.float64)), axis=0).astype(np.float32)
