from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from train_enformer_like_tf import TransformerBlock, make_one_hot


def read_table(path: Path) -> pd.DataFrame:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def write_table(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        df.to_csv(path.with_suffix(".tsv.gz"), sep="\t", index=False)


def numeric_matrix(df: pd.DataFrame, marker_cols: list[str]) -> np.ndarray:
    M = df[marker_cols].replace(-9, np.nan).astype(np.float32)
    means = M.mean(axis=0)
    M = M.fillna(means)
    return M.to_numpy(dtype=np.float32)


def detect_col(df: pd.DataFrame, candidates: list[str], required_name: str) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise SystemExit(f"Could not detect {required_name}; use explicit columns. Available: {df.columns.tolist()[:40]}")


def build_embedding_model(model: tf.keras.Model, layer_name: str | None, layer_index: int) -> tf.keras.Model:
    if layer_name:
        layer = model.get_layer(layer_name)
        return tf.keras.Model(model.input, layer.output)
    return tf.keras.Model(model.input, model.layers[layer_index].output)


def extract_window_embeddings(args) -> tuple[pd.DataFrame, np.ndarray]:
    custom = {"TransformerBlock": TransformerBlock}
    model = tf.keras.models.load_model(args.model, custom_objects=custom, compile=False)
    emb_model = build_embedding_model(model, args.layer_name, args.layer_index)
    print(f"TensorFlow: {tf.__version__}; embedding output: {emb_model.output_shape}", flush=True)

    intervals = read_table(args.intervals)
    with h5py.File(args.h5, "r") as h5:
        n = h5["seq"].shape[0]
        idx = np.arange(n)
        if args.max_windows and args.max_windows < n:
            idx = idx[: args.max_windows]
        embeddings = []
        for start in range(0, len(idx), args.batch_size):
            batch_idx = idx[start : start + args.batch_size]
            seq = h5["seq"][batch_idx].astype(np.int64)
            x = np.stack([make_one_hot(s) for s in seq], axis=0)
            z = emb_model.predict(x, verbose=0)
            if z.ndim == 3:
                if args.pooling == "center":
                    z = z[:, z.shape[1] // 2, :]
                else:
                    z = z.mean(axis=1)
            embeddings.append(z.astype(np.float32))
            if (start + len(batch_idx)) % 1000 == 0:
                print(f"embedded windows: {start + len(batch_idx):,}/{len(idx):,}", flush=True)
    Z = np.vstack(embeddings).astype(np.float32)
    intervals = intervals.iloc[idx].reset_index(drop=True).copy()
    intervals["window_embedding_index"] = np.arange(len(intervals), dtype=np.int64)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / f"{args.prefix}_window_embeddings.npy", Z)
    intervals.to_csv(args.out_dir / f"{args.prefix}_window_embedding_order.tsv", sep="\t", index=False)
    return intervals, Z


def marker_embeddings_from_windows(args, intervals: pd.DataFrame, Z_window: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    marker = read_table(args.marker_metadata)
    marker_col = args.marker_col or detect_col(marker, ["marker", "marker_id", "rs", "SNP"], "marker column")
    chrom_col = args.marker_chrom_col or detect_col(marker, ["chrom", "chr", "chromosome"], "marker chromosome")
    pos_col = args.marker_pos_col or detect_col(marker, ["pos", "position", "physical_position", "bp"], "marker position")
    marker = marker[[marker_col, chrom_col, pos_col]].copy()
    marker.columns = ["marker_id", "chrom", "pos"]
    marker["pos"] = pd.to_numeric(marker["pos"], errors="coerce")
    marker = marker[marker["marker_id"].notna() & marker["chrom"].notna() & marker["pos"].notna()].copy()

    rows = []
    by_chrom = {str(c): g for c, g in intervals.groupby(intervals["chrom"].astype(str))}
    for rec in marker.itertuples(index=False):
        chrom = str(rec.chrom)
        if chrom not in by_chrom:
            continue
        hits = by_chrom[chrom]
        hit = hits[(hits["start"].astype(int) <= int(rec.pos)) & (hits["end"].astype(int) > int(rec.pos))]
        if hit.empty:
            continue
        best = hit.iloc[0]
        rows.append(
            {
                "marker_id": str(rec.marker_id),
                "chrom": chrom,
                "pos": int(rec.pos),
                "window_embedding_index": int(best["window_embedding_index"]),
            }
        )
    marker_order = pd.DataFrame(rows).drop_duplicates("marker_id")
    if marker_order.empty:
        raise SystemExit("No markers overlapped embedded windows")
    Z_marker = Z_window[marker_order["window_embedding_index"].to_numpy(dtype=np.int64)]
    np.save(args.out_dir / f"{args.prefix}_marker_embeddings.npy", Z_marker.astype(np.float32))
    marker_order.to_csv(args.out_dir / f"{args.prefix}_marker_embedding_order.tsv", sep="\t", index=False)
    return marker_order, Z_marker.astype(np.float32)


def genotype_embeddings(args, marker_order: pd.DataFrame, Z_marker: np.ndarray) -> None:
    X = read_table(args.genotype_matrix)
    if args.sample_col not in X.columns:
        raise SystemExit(f"Genotype matrix missing sample column {args.sample_col}")
    common = [m for m in marker_order["marker_id"].astype(str) if m in X.columns]
    if not common:
        raise SystemExit("No embedded markers are present as columns in genotype matrix")
    marker_index = {m: i for i, m in enumerate(marker_order["marker_id"].astype(str))}
    Zm = Z_marker[[marker_index[m] for m in common], :]
    M = numeric_matrix(X, common)
    M = M - np.nanmean(M, axis=0, keepdims=True)
    denom = max(np.sqrt(M.shape[1]), 1.0)
    Zg = (M @ Zm) / denom
    out = pd.DataFrame({"sample_id": X[args.sample_col].astype(str)})
    for j in range(Zg.shape[1]):
        out[f"z{j:04d}"] = Zg[:, j].astype(np.float32)
    write_table(out, args.out_dir / f"{args.prefix}_genotype_regulatory_embeddings.parquet")
    np.save(args.out_dir / f"{args.prefix}_genotype_regulatory_embeddings.npy", Zg.astype(np.float32))
    pd.DataFrame({"sample_id": X[args.sample_col].astype(str)}).to_csv(
        args.out_dir / f"{args.prefix}_genotype_regulatory_embedding_order.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {
            "metric": ["markers_with_embeddings", "genotype_samples", "embedding_dim"],
            "value": [len(common), len(X), Zg.shape[1]],
        }
    ).to_csv(args.out_dir / f"{args.prefix}_genotype_embedding_qc.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export regulatory embeddings from a trained TensorFlow Enformer-like model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("regulatory_model/embeddings"))
    parser.add_argument("--prefix", default="regulatory")
    parser.add_argument("--layer-name")
    parser.add_argument("--layer-index", type=int, default=-2)
    parser.add_argument("--pooling", choices=["mean", "center"], default="mean")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--marker-metadata", type=Path)
    parser.add_argument("--marker-col")
    parser.add_argument("--marker-chrom-col")
    parser.add_argument("--marker-pos-col")
    parser.add_argument("--genotype-matrix", type=Path)
    parser.add_argument("--sample-col", default="sample_id")
    args = parser.parse_args()

    intervals, Z_window = extract_window_embeddings(args)
    if args.marker_metadata:
        marker_order, Z_marker = marker_embeddings_from_windows(args, intervals, Z_window)
        if args.genotype_matrix:
            genotype_embeddings(args, marker_order, Z_marker)
    print(f"Wrote embeddings under: {args.out_dir}")


if __name__ == "__main__":
    main()
