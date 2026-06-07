from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sampled_indices(n: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0 or n <= sample_size:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=sample_size, replace=False).astype(np.int64))


def squared_distance_block(K: np.ndarray, diag: np.ndarray, start: int, end: int) -> np.ndarray:
    block = np.asarray(K[start:end, :], dtype=np.float64)
    d2 = diag[start:end, None] + diag[None, :] - 2.0 * block
    np.maximum(d2, 0.0, out=d2)
    return d2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Gaussian/RBF genomic kernel from a linear genomic relationship kernel."
    )
    parser.add_argument(
        "--linear-kernel",
        type=Path,
        default=Path("genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy"),
    )
    parser.add_argument(
        "--sample-order",
        type=Path,
        default=Path("genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv"),
    )
    parser.add_argument("--sample-order-col", default="sample_id")
    parser.add_argument(
        "--out-kernel",
        type=Path,
        default=Path("genotype_panels/hmp/K_HMP.QCfiltered.gaussian.npy"),
    )
    parser.add_argument(
        "--out-qc",
        type=Path,
        default=Path("genotype_panels/hmp/K_HMP.QCfiltered.gaussian.qc.json"),
    )
    parser.add_argument("--gamma", type=float, help="Explicit RBF gamma. Default: multiplier / median sampled squared distance.")
    parser.add_argument("--gamma-multiplier", type=float, default=1.0)
    parser.add_argument("--median-sample-size", type=int, default=2048)
    parser.add_argument("--psd-sample-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.gamma is not None and (not np.isfinite(args.gamma) or args.gamma <= 0):
        raise SystemExit("--gamma must be finite and positive")
    if not np.isfinite(args.gamma_multiplier) or args.gamma_multiplier <= 0:
        raise SystemExit("--gamma-multiplier must be finite and positive")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")

    K = np.load(args.linear_kernel, mmap_mode="r")
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise SystemExit(f"Linear kernel must be square; found {K.shape}")
    n = K.shape[0]
    order = pd.read_csv(args.sample_order, sep="\t", dtype=str)
    if args.sample_order_col not in order.columns:
        raise SystemExit(f"{args.sample_order} lacks column {args.sample_order_col}")
    if len(order) != n:
        raise SystemExit(f"Kernel dimension {n} does not match sample-order rows {len(order)}")
    if order[args.sample_order_col].fillna("").astype(str).str.strip().duplicated().any():
        raise SystemExit(f"{args.sample_order_col} contains duplicated IDs")

    diag = np.asarray(K[np.arange(n), np.arange(n)], dtype=np.float64)
    if not np.all(np.isfinite(diag)):
        raise SystemExit("Linear-kernel diagonal contains non-finite values")

    median_idx = sampled_indices(n, args.median_sample_size, args.seed)
    sampled_K = np.asarray(K[np.ix_(median_idx, median_idx)], dtype=np.float64)
    sampled_diag = diag[median_idx]
    sampled_d2 = sampled_diag[:, None] + sampled_diag[None, :] - 2.0 * sampled_K
    sampled_d2 = np.maximum(sampled_d2, 0.0)
    upper = sampled_d2[np.triu_indices(len(median_idx), k=1)]
    positive = upper[np.isfinite(upper) & (upper > 0)]
    if positive.size == 0:
        raise SystemExit("Could not estimate Gaussian bandwidth: no positive finite sampled distances")
    median_d2 = float(np.median(positive))
    gamma = float(args.gamma) if args.gamma is not None else float(args.gamma_multiplier / median_d2)

    args.out_kernel.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(args.out_kernel, mode="w+", dtype=np.float32, shape=(n, n))
    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        d2 = squared_distance_block(K, diag, start, end)
        out[start:end, :] = np.exp(-gamma * d2).astype(np.float32)
        print(f"Gaussian-kernel rows: {end:,}/{n:,}", flush=True)
    out[np.arange(n), np.arange(n)] = 1.0
    out.flush()

    qc_idx = sampled_indices(n, args.psd_sample_size, args.seed + 1)
    qc_block = np.asarray(out[np.ix_(qc_idx, qc_idx)], dtype=np.float64)
    max_symmetry_error = float(np.max(np.abs(qc_block - qc_block.T)))
    qc_block = (qc_block + qc_block.T) / 2.0
    eigvals = np.linalg.eigvalsh(qc_block)
    qc = {
        "linear_kernel": str(args.linear_kernel),
        "sample_order": str(args.sample_order),
        "output_kernel": str(args.out_kernel),
        "samples": n,
        "gamma": gamma,
        "gamma_source": "explicit" if args.gamma is not None else "median_squared_distance",
        "gamma_multiplier": float(args.gamma_multiplier),
        "sampled_median_squared_distance": median_d2,
        "sampled_positive_distances": int(positive.size),
        "kernel_mean_diagonal": float(np.mean(np.diag(out))),
        "kernel_sample_min": float(np.min(qc_block)),
        "kernel_sample_max": float(np.max(qc_block)),
        "kernel_sample_mean": float(np.mean(qc_block)),
        "sampled_symmetry_max_abs": max_symmetry_error,
        "sampled_min_eigenvalue": float(eigvals[0]),
        "sampled_max_eigenvalue": float(eigvals[-1]),
        "psd_sample_size": int(len(qc_idx)),
    }
    args.out_qc.parent.mkdir(parents=True, exist_ok=True)
    args.out_qc.write_text(json.dumps(qc, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(qc, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
