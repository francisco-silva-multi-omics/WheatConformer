from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sampled_indices(n: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0 or n <= sample_size:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(n, size=sample_size, replace=False))


def build_epi2(
    linear_path: Path,
    sample_order: Path,
    sample_order_col: str,
    out_kernel: Path,
    chunk_size: int,
    psd_sample_size: int,
    seed: int,
) -> dict[str, object]:
    linear = np.load(linear_path, mmap_mode="r")
    if linear.ndim != 2 or linear.shape[0] != linear.shape[1]:
        raise SystemExit(f"Linear kernel must be square; found {linear.shape}")
    order = pd.read_csv(sample_order, sep="\t", dtype=str)
    if sample_order_col not in order.columns or len(order) != linear.shape[0]:
        raise SystemExit(f"Sample order {sample_order} does not match kernel dimension {linear.shape[0]}")
    n = linear.shape[0]
    diagonal = np.asarray(linear[np.arange(n), np.arange(n)], dtype=np.float64) ** 2
    mean_diagonal = float(diagonal.mean())
    if not np.isfinite(mean_diagonal) or mean_diagonal <= 0:
        raise SystemExit("K_G hadamard K_G has a non-positive or non-finite mean diagonal")
    out_kernel.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(out_kernel, mode="w+", dtype=np.float32, shape=(n, n))
    minimum = np.inf
    maximum = -np.inf
    symmetry_error = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = np.asarray(linear[start:end], dtype=np.float64)
        scaled = block * block / mean_diagonal
        out[start:end] = scaled.astype(np.float32)
        minimum = min(minimum, float(scaled.min()))
        maximum = max(maximum, float(scaled.max()))
        symmetry_error = max(
            symmetry_error,
            float(
                np.max(
                    np.abs(
                        scaled
                        - (np.asarray(linear[:, start:end], dtype=np.float64).T ** 2 / mean_diagonal)
                    )
                )
            ),
        )
        print(f"EPI2-kernel rows: {end:,}/{n:,}", flush=True)
    out.flush()
    idx = sampled_indices(n, psd_sample_size, seed)
    sampled = np.asarray(out[np.ix_(idx, idx)], dtype=np.float64)
    sampled = (sampled + sampled.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(sampled)
    return {
        "linear_kernel": str(linear_path),
        "sample_order": str(sample_order),
        "output_kernel": str(out_kernel),
        "shape": list(out.shape),
        "mean_diagonal_before_scaling": mean_diagonal,
        "mean_diagonal_after_scaling": float(np.mean(np.diag(out))),
        "min": float(minimum),
        "max": float(maximum),
        "symmetry_error": symmetry_error,
        "sampled_min_eigenvalue": float(eigenvalues[0]),
        "sampled_max_eigenvalue": float(eigenvalues[-1]),
        "psd_sample_size": int(len(idx)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build optional second-order additive-by-additive genomic kernel.")
    parser.add_argument("--linear-kernel", type=Path, required=True)
    parser.add_argument("--sample-order", type=Path, required=True)
    parser.add_argument("--sample-order-col", default="sample_id")
    parser.add_argument("--out-kernel", type=Path, default=Path("genotype_panels/hmp/K_HMP.QCfiltered.epi2.npy"))
    parser.add_argument("--out-qc", type=Path, default=Path("genotype_panels/hmp/K_HMP.QCfiltered.epi2.qc.json"))
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--psd-sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    qc = build_epi2(
        args.linear_kernel,
        args.sample_order,
        args.sample_order_col,
        args.out_kernel,
        args.chunk_size,
        args.psd_sample_size,
        args.seed,
    )
    args.out_qc.parent.mkdir(parents=True, exist_ok=True)
    args.out_qc.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
