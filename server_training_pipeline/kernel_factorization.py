from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


STRICT_INDUCTIVE_SPLIT_MODES = {
    "cv1_genotype",
    "cv1_environment",
    "cv0_genotype_environment",
}


def effective_factorization_mode(requested_mode: str, split_mode: str, warn: bool = False) -> str:
    if requested_mode == "train_nystrom" and split_mode not in STRICT_INDUCTIVE_SPLIT_MODES:
        if warn:
            warnings.warn(
                f"train_nystrom is restricted to CV1/CV0 benchmarking; using full_transductive for {split_mode!r}.",
                stacklevel=2,
            )
        return "full_transductive"
    return requested_mode


def kernel_factors(
    path: Path,
    rank: int,
    train_ids: np.ndarray | None = None,
    jitter: float = 0.0,
) -> tuple[np.ndarray, dict[str, int | str]]:
    K = np.load(path).astype(np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Kernel must be square: {path} has shape {K.shape}")
    K = (K + K.T) / 2.0
    if jitter > 0:
        K.flat[:: K.shape[0] + 1] += jitter
    if train_ids is None:
        train_ids = np.arange(K.shape[0], dtype=np.int32)
        K_train = K
        factorization_mode = "full_transductive"
    else:
        train_ids = np.unique(np.asarray(train_ids, dtype=np.int32))
        if train_ids.size == 0:
            raise ValueError("train_nystrom requires at least one training kernel ID")
        if train_ids.min() < 0 or train_ids.max() >= K.shape[0]:
            raise ValueError(f"Training kernel IDs are outside kernel dimensions for {path}")
        K_train = K[np.ix_(train_ids, train_ids)]
        factorization_mode = "train_nystrom"

    vals, vecs = np.linalg.eigh(K_train)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    keep = vals > 1e-8
    vals = vals[keep][:rank]
    vecs = vecs[:, keep][:, :rank]
    if vals.size == 0:
        raise ValueError(f"Kernel has no positive eigenvalues above tolerance: {path}")
    if factorization_mode == "full_transductive":
        factors = vecs * np.sqrt(vals)[None, :]
    else:
        factors = K[:, train_ids] @ (vecs / np.sqrt(vals)[None, :])
    metadata = {
        "factorization_mode": factorization_mode,
        "rank_requested": int(rank),
        "rank_retained": int(vals.size),
        "train_kernel_dimension": int(train_ids.size),
        "kernel_dimension": int(K.shape[0]),
    }
    return factors.astype(np.float32), metadata


def top_factors(path: Path, rank: int, jitter: float = 0.0) -> np.ndarray:
    factors, _ = kernel_factors(path, rank, jitter=jitter)
    return factors


def retained_eigenvalues(factors: np.ndarray, train_ids: np.ndarray | None = None) -> np.ndarray:
    rows = factors if train_ids is None else factors[np.asarray(train_ids, dtype=np.int32)]
    return np.sum(np.square(rows.astype(np.float64)), axis=0).astype(np.float32)
