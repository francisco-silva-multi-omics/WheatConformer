from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genotype-matrix", type=Path, required=True)
    parser.add_argument("--prior-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="K_80kWeighted")
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--marker-col", default="marker_id")
    parser.add_argument("--orientation", choices=["auto", "sample_by_marker", "marker_by_sample"], default="auto")
    parser.add_argument("--marker-id-col", default="marker_id")
    parser.add_argument("--weight-col", default="marker_weight")
    parser.add_argument("--maf-min", type=float, default=0.01)
    parser.add_argument("--missing-max", type=float, default=0.50)
    parser.add_argument("--weight-min", type=float, default=0.05)
    parser.add_argument("--weight-max", type=float, default=10.0)
    parser.add_argument("--no-mean-diag-scale", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    priors = read_table(args.prior_table)
    priors[args.marker_id_col] = priors[args.marker_id_col].astype(str)
    priors[args.weight_col] = pd.to_numeric(priors[args.weight_col], errors="coerce")
    weights = (
        priors.dropna(subset=[args.weight_col])
        .groupby(args.marker_id_col)[args.weight_col]
        .mean()
        .clip(args.weight_min, args.weight_max)
    )

    X = pd.read_parquet(args.genotype_matrix)
    orientation = args.orientation
    if orientation == "auto":
        if args.sample_col in X.columns:
            orientation = "sample_by_marker"
        elif args.marker_col in X.columns:
            orientation = "marker_by_sample"
        else:
            raise SystemExit(
                f"Could not infer orientation. Expected {args.sample_col!r} or {args.marker_col!r} in columns."
            )

    if orientation == "sample_by_marker":
        if args.sample_col not in X.columns:
            raise SystemExit(f"{args.sample_col!r} not found in {args.genotype_matrix}")
        sample_ids = X[args.sample_col].astype(str).reset_index(drop=True)
        marker_cols = [c for c in X.columns if c != args.sample_col]
        keep_markers = [m for m in marker_cols if m in weights.index]
        if not keep_markers:
            raise SystemExit("No genotype markers overlap the 80k prior table")
        M = X[keep_markers].astype(np.float32).replace(-9, np.nan)
    else:
        if args.marker_col not in X.columns:
            raise SystemExit(f"{args.marker_col!r} not found in {args.genotype_matrix}")
        X[args.marker_col] = X[args.marker_col].astype(str)
        keep_rows = X[args.marker_col].isin(weights.index)
        keep_markers = X.loc[keep_rows, args.marker_col].tolist()
        if not keep_markers:
            raise SystemExit("No genotype markers overlap the 80k prior table")
        sample_ids = pd.Series([c for c in X.columns if c != args.marker_col], name=args.sample_col).astype(str)
        M = X.loc[keep_rows, sample_ids.tolist()].T
        M.columns = keep_markers
        M = M.astype(np.float32).replace(-9, np.nan)

    if not keep_markers:
        raise SystemExit("No genotype markers overlap the 80k prior table")
    n_overlap_before_qc = len(keep_markers)

    missing = M.isna().mean(axis=0)
    p = M.mean(axis=0) / 2.0
    maf = np.minimum(p, 1.0 - p)
    qc = (missing <= args.missing_max) & (maf >= args.maf_min) & (M.var(axis=0, skipna=True) > 0)
    keep_markers = list(qc[qc].index)
    if not keep_markers:
        raise SystemExit("All overlapping markers failed QC")

    M = M[keep_markers]
    M = M.apply(lambda col: col.fillna(col.mean()), axis=0)
    p = M.mean(axis=0) / 2.0
    Z = M - (2.0 * p)

    w = weights.loc[keep_markers].astype(np.float32)
    w = w / float(w.mean())
    sqrt_w = np.sqrt(w.to_numpy(dtype=np.float32))
    Zw = Z.to_numpy(dtype=np.float32) * sqrt_w
    denom = float(np.sum(w.to_numpy(dtype=np.float32) * 2.0 * p.to_numpy(dtype=np.float32) * (1.0 - p.to_numpy(dtype=np.float32))))
    if denom <= 0:
        raise SystemExit("VanRaden denominator is zero after weighting")

    K = (Zw @ Zw.T) / denom
    K = K.astype(np.float32)
    mean_diag_before_scaling = float(np.mean(np.diag(K)))
    if not args.no_mean_diag_scale:
        if mean_diag_before_scaling <= 0 or not np.isfinite(mean_diag_before_scaling):
            raise SystemExit("Cannot scale kernel: mean diagonal is not positive and finite")
        K = (K / mean_diag_before_scaling).astype(np.float32)
    mean_diag_after_scaling = float(np.mean(np.diag(K)))

    np.save(args.out_dir / f"{args.prefix}.npy", K)
    pd.DataFrame({args.sample_col: sample_ids}).to_csv(
        args.out_dir / f"{args.prefix}_sample_order.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {
            "marker_id": keep_markers,
            "marker_weight": w.loc[keep_markers].to_numpy(),
            "missingness": missing.loc[keep_markers].to_numpy(),
            "maf": maf.loc[keep_markers].to_numpy(),
        }
    ).to_csv(args.out_dir / f"{args.prefix}_marker_weights_used.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {"metric": "samples", "value": K.shape[0]},
            {"metric": "orientation", "value": orientation},
            {"metric": "markers_overlap_before_qc", "value": n_overlap_before_qc},
            {"metric": "markers_used_after_qc", "value": len(keep_markers)},
            {"metric": "mean_diagonal_before_scaling", "value": mean_diag_before_scaling},
            {"metric": "mean_diagonal_after_scaling", "value": mean_diag_after_scaling},
            {"metric": "mean_diagonal_scaled", "value": not args.no_mean_diag_scale},
            {"metric": "denominator", "value": denom},
        ]
    ).to_csv(args.out_dir / f"{args.prefix}_summary.tsv", sep="\t", index=False)
    print(K.shape, K.dtype)
    print("Markers used:", len(keep_markers))
    print("Mean diagonal before scaling:", mean_diag_before_scaling)
    print("Mean diagonal after scaling:", mean_diag_after_scaling)


if __name__ == "__main__":
    main()
