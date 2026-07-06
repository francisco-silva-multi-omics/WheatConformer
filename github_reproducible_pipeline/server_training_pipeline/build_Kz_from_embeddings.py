from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_embeddings(args) -> tuple[list[str], np.ndarray]:
    if args.embedding_npy:
        Z = np.load(args.embedding_npy).astype(np.float32)
        order = pd.read_csv(args.order, sep="\t", dtype=str)
        if args.id_col not in order.columns:
            raise SystemExit(f"Order file missing --id-col {args.id_col}")
        ids = order[args.id_col].astype(str).tolist()
        return ids, Z
    df = pd.read_parquet(args.embeddings) if "".join(args.embeddings.suffixes).endswith(".parquet") else pd.read_csv(args.embeddings, sep="\t")
    if args.id_col not in df.columns:
        raise SystemExit(f"Embedding table missing --id-col {args.id_col}")
    ids = df[args.id_col].astype(str).tolist()
    num_cols = args.embedding_col or [c for c in df.columns if c != args.id_col and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise SystemExit("No numeric embedding columns found")
    return ids, df[num_cols].to_numpy(dtype=np.float32)


def standardize(Z: np.ndarray) -> np.ndarray:
    Z = Z.astype(np.float64)
    col_mean = np.nanmean(Z, axis=0)
    inds = np.where(~np.isfinite(Z))
    Z[inds] = np.take(col_mean, inds[1])
    sd = np.std(Z, axis=0)
    keep = sd > 1e-12
    Z = Z[:, keep]
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    return Z.astype(np.float32)


def pca_reduce(Z: np.ndarray, n_components: int) -> np.ndarray:
    if n_components <= 0 or n_components >= Z.shape[1]:
        return Z
    U, S, _ = np.linalg.svd(Z.astype(np.float64), full_matrices=False)
    return (U[:, :n_components] * S[:n_components]).astype(np.float32)


def rbf_kernel(Z: np.ndarray, sample_size: int, gamma: float | None) -> tuple[np.ndarray, float]:
    sq = np.sum(Z * Z, axis=1, keepdims=True)
    D = np.maximum(sq + sq.T - 2.0 * (Z @ Z.T), 0.0)
    if gamma is None or gamma <= 0:
        n = D.shape[0]
        rng = np.random.default_rng(2026)
        idx = rng.choice(n, size=min(sample_size, n), replace=False)
        vals = D[np.ix_(idx, idx)]
        med = float(np.median(vals[vals > 0])) if np.any(vals > 0) else 1.0
        gamma = 1.0 / max(med, 1e-8)
    return np.exp(-gamma * D).astype(np.float32), float(gamma)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build functional kernel K_z from regulatory embeddings.")
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--embedding-npy", type=Path)
    parser.add_argument("--order", type=Path)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--embedding-col", action="append")
    parser.add_argument("--out-dir", type=Path, default=Path("model_kernels"))
    parser.add_argument("--prefix", default="K_z")
    parser.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    parser.add_argument("--pca-components", type=int, default=0)
    parser.add_argument("--rbf-gamma", type=float, default=0.0)
    parser.add_argument("--median-sample-size", type=int, default=2000)
    parser.add_argument("--scale-mean-diagonal", action="store_true", default=True)
    parser.add_argument("--embedding-source", default="reference_regulatory_embeddings")
    parser.add_argument("--coordinate-system", default="IWGSC_RefSeq_v1.0")
    parser.add_argument("--graph-derived", action="store_true")
    args = parser.parse_args()

    if not args.embeddings and not (args.embedding_npy and args.order):
        raise SystemExit("Provide --embeddings or both --embedding-npy and --order")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids, Z = read_embeddings(args)
    Z = standardize(Z)
    Z = pca_reduce(Z, args.pca_components)
    if args.kernel == "linear":
        K = (Z @ Z.T) / max(Z.shape[1], 1)
        gamma = np.nan
    else:
        K, gamma = rbf_kernel(Z, args.median_sample_size, args.rbf_gamma)
    K = ((K + K.T) / 2.0).astype(np.float32)
    if args.scale_mean_diagonal:
        mean_diag = float(np.mean(np.diag(K)))
        if mean_diag > 0:
            K = (K / mean_diag).astype(np.float32)
    np.save(args.out_dir / f"{args.prefix}.npy", K)
    pd.DataFrame({args.id_col: ids}).to_csv(args.out_dir / f"{args.prefix}_sample_order.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "metric": ["samples", "embedding_dim_used", "kernel", "mean_diagonal", "rbf_gamma"],
            "value": [len(ids), Z.shape[1], args.kernel, float(np.mean(np.diag(K))), gamma],
        }
    ).to_csv(args.out_dir / f"{args.prefix}_qc.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "field": ["embedding_source", "coordinate_system", "graph_derived", "kernel", "samples", "embedding_dim_used"],
            "value": [
                args.embedding_source,
                args.coordinate_system,
                str(bool(args.graph_derived)),
                args.kernel,
                len(ids),
                Z.shape[1],
            ],
        }
    ).to_csv(args.out_dir / f"{args.prefix}_provenance.tsv", sep="\t", index=False)
    print(K.shape, K.dtype)
    print("Mean diagonal:", float(np.mean(np.diag(K))))


if __name__ == "__main__":
    main()
