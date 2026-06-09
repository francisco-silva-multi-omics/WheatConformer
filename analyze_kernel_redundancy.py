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


def diagonal_summary(kernel: np.ndarray) -> dict[str, float]:
    diagonal = np.asarray(kernel[np.arange(kernel.shape[0]), np.arange(kernel.shape[0])], dtype=np.float64)
    return {
        "min": float(np.min(diagonal)),
        "mean": float(np.mean(diagonal)),
        "median": float(np.median(diagonal)),
        "max": float(np.max(diagonal)),
    }


def effective_rank(eigenvalues: np.ndarray) -> float:
    positive = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    total = positive.sum()
    if total <= 0:
        return 0.0
    probabilities = positive[positive > 0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def blockwise_alignment(
    linear: np.ndarray,
    rbf: np.ndarray,
    chunk_size: int,
) -> tuple[float, float, float, float, float, float]:
    n = linear.shape[0]
    linear_row_mean = np.empty(n, dtype=np.float64)
    rbf_row_mean = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        linear_row_mean[start:end] = np.asarray(linear[start:end], dtype=np.float64).mean(axis=1)
        rbf_row_mean[start:end] = np.asarray(rbf[start:end], dtype=np.float64).mean(axis=1)
    linear_global = float(linear_row_mean.mean())
    rbf_global = float(rbf_row_mean.mean())

    cka_cross = cka_linear = cka_rbf = 0.0
    linear_symmetry_error = rbf_symmetry_error = 0.0
    count = sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        a = np.asarray(linear[start:end], dtype=np.float64)
        b = np.asarray(rbf[start:end], dtype=np.float64)
        linear_symmetry_error = max(
            linear_symmetry_error,
            float(np.max(np.abs(a - np.asarray(linear[:, start:end], dtype=np.float64).T))),
        )
        rbf_symmetry_error = max(
            rbf_symmetry_error,
            float(np.max(np.abs(b - np.asarray(rbf[:, start:end], dtype=np.float64).T))),
        )
        ca = a - linear_row_mean[start:end, None] - linear_row_mean[None, :] + linear_global
        cb = b - rbf_row_mean[start:end, None] - rbf_row_mean[None, :] + rbf_global
        cka_cross += float(np.sum(ca * cb))
        cka_linear += float(np.sum(ca * ca))
        cka_rbf += float(np.sum(cb * cb))

        mask = np.ones_like(a, dtype=bool)
        local_rows = np.arange(start, end)
        mask[np.arange(end - start), local_rows] = False
        x = a[mask]
        y = b[mask]
        count += x.size
        sum_x += float(x.sum())
        sum_y += float(y.sum())
        sum_x2 += float(x @ x)
        sum_y2 += float(y @ y)
        sum_xy += float(x @ y)

    cka = cka_cross / max(np.sqrt(cka_linear * cka_rbf), np.finfo(float).eps)
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = sum_x2 - sum_x * sum_x / count
    variance_y = sum_y2 - sum_y * sum_y / count
    correlation = covariance / max(np.sqrt(variance_x * variance_y), np.finfo(float).eps)
    return (
        float(cka),
        float(correlation),
        linear_global,
        rbf_global,
        linear_symmetry_error,
        rbf_symmetry_error,
    )


def eigen_summary(name: str, kernel: np.ndarray, indices: np.ndarray) -> tuple[dict[str, object], pd.DataFrame]:
    block = np.asarray(kernel[np.ix_(indices, indices)], dtype=np.float64)
    symmetry_error = float(np.max(np.abs(block - block.T)))
    eigenvalues = np.linalg.eigvalsh((block + block.T) / 2.0)
    positive = eigenvalues[eigenvalues > max(float(eigenvalues[-1]) * 1e-10, 1e-12)]
    condition = float(eigenvalues[-1] / positive[0]) if positive.size else None
    summary = {
        "sample_size": int(len(indices)),
        "sampled_min_eigenvalue": float(eigenvalues[0]),
        "sampled_max_eigenvalue": float(eigenvalues[-1]),
        "sampled_negative_eigenvalues": int((eigenvalues < -1e-8).sum()),
        "sampled_effective_rank": effective_rank(eigenvalues),
        "sampled_condition_number_estimate": condition,
        "sampled_symmetry_error": symmetry_error,
    }
    spectrum = pd.DataFrame(
        {
            "kernel": name,
            "eigen_rank_descending": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues[::-1],
            "proportion_of_positive_sum": np.clip(eigenvalues[::-1], 0, None)
            / max(float(np.clip(eigenvalues, 0, None).sum()), np.finfo(float).eps),
        }
    )
    return summary, spectrum


def analyze(
    linear_path: Path,
    rbf_path: Path,
    sample_order: Path,
    sample_order_col: str,
    eigen_sample_size: int,
    quantile_sample_size: int,
    chunk_size: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    linear = np.load(linear_path, mmap_mode="r")
    rbf = np.load(rbf_path, mmap_mode="r")
    if linear.ndim != 2 or linear.shape[0] != linear.shape[1] or rbf.shape != linear.shape:
        raise SystemExit(f"Kernel shape mismatch: linear={linear.shape}, rbf={rbf.shape}")
    order = pd.read_csv(sample_order, sep="\t", dtype=str)
    if sample_order_col not in order.columns or len(order) != linear.shape[0]:
        raise SystemExit(f"Sample order {sample_order} does not match kernel dimension {linear.shape[0]}")
    n = linear.shape[0]
    eigen_idx = sampled_indices(n, eigen_sample_size, seed)
    quantile_idx = sampled_indices(n, quantile_sample_size, seed + 1)
    linear_sample = np.asarray(linear[np.ix_(quantile_idx, quantile_idx)], dtype=np.float64)
    rbf_sample = np.asarray(rbf[np.ix_(quantile_idx, quantile_idx)], dtype=np.float64)
    upper = np.triu_indices(len(quantile_idx), k=1)
    quantiles = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    linear_eigen, linear_spectrum = eigen_summary("linear", linear, eigen_idx)
    rbf_eigen, rbf_spectrum = eigen_summary("rbf", rbf, eigen_idx)
    cka, correlation, linear_mean, rbf_mean, linear_symmetry, rbf_symmetry = blockwise_alignment(linear, rbf, chunk_size)
    result = {
        "linear_kernel": str(linear_path),
        "rbf_kernel": str(rbf_path),
        "sample_order": str(sample_order),
        "shape_consistent": True,
        "shape": list(linear.shape),
        "linear_diagonal": diagonal_summary(linear),
        "rbf_diagonal": diagonal_summary(rbf),
        "linear_off_diagonal_quantiles": dict(zip(map(str, quantiles), map(float, np.quantile(linear_sample[upper], quantiles)))),
        "rbf_off_diagonal_quantiles": dict(zip(map(str, quantiles), map(float, np.quantile(rbf_sample[upper], quantiles)))),
        "off_diagonal_pearson_correlation": correlation,
        "centered_kernel_alignment": cka,
        "linear_symmetry_error": linear_symmetry,
        "rbf_symmetry_error": rbf_symmetry,
        "linear_global_mean": linear_mean,
        "rbf_global_mean": rbf_mean,
        "eigen_sample_size": int(len(eigen_idx)),
        "quantile_sample_size": int(len(quantile_idx)),
        "linear_eigen_summary": linear_eigen,
        "rbf_eigen_summary": rbf_eigen,
    }
    return result, pd.concat([linear_spectrum, rbf_spectrum], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze redundancy between additive and Gaussian genomic kernels.")
    parser.add_argument("--linear-kernel", type=Path, required=True)
    parser.add_argument("--rbf-kernel", type=Path, required=True)
    parser.add_argument("--sample-order", type=Path, required=True)
    parser.add_argument("--sample-order-col", default="sample_id")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--eigen-sample-size", type=int, default=1024)
    parser.add_argument("--quantile-sample-size", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    result, spectrum = analyze(
        args.linear_kernel,
        args.rbf_kernel,
        args.sample_order,
        args.sample_order_col,
        args.eigen_sample_size,
        args.quantile_sample_size,
        args.chunk_size,
        args.seed,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    spectrum.to_csv(args.out_tsv, sep="\t", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
