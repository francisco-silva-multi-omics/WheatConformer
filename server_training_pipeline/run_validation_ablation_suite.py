from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .trait_isolation import select_single_trait
except ImportError:
    from trait_isolation import select_single_trait


DEFAULT_ABLATIONS = [
    "G",
    "E",
    "G+E",
    "G+E+GE",
    "RBF",
    "RBF+E",
    "RBF+E+RBFE",
    "G+RBF+E",
    "G+RBF+E+GE+RBFE",
]


def read_table(path: Path) -> pd.DataFrame:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def top_factors(path: Path, rank: int) -> np.ndarray:
    K = np.load(path).astype(np.float64)
    K = (K + K.T) / 2.0
    vals, vecs = np.linalg.eigh(K)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    keep = vals > 1e-8
    vals = vals[keep][:rank]
    vecs = vecs[:, keep][:, :rank]
    return (vecs * np.sqrt(vals)[None, :]).astype(np.float32)


def map_compact(obs: pd.DataFrame, index_col: str, order_path: Path) -> np.ndarray:
    order = pd.read_csv(order_path, sep="\t")
    if {"source_kernel_index", "compact_kernel_index"}.issubset(order.columns):
        mapper = dict(zip(order["source_kernel_index"].astype(int), order["compact_kernel_index"].astype(int)))
        out = obs[index_col].astype(int).map(mapper)
    else:
        out = obs[index_col].astype(int)
    if out.isna().any():
        raise SystemExit(f"Could not map all rows through {order_path}")
    return out.to_numpy(dtype=np.int32)


def weighted_standardize(y: np.ndarray, w: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, float, float]:
    wt = np.where(np.isfinite(w[train]) & (w[train] > 0), w[train], 1.0)
    mu = float(np.sum(wt * y[train]) / np.sum(wt))
    sd = float(np.sqrt(np.sum(wt * (y[train] - mu) ** 2) / np.sum(wt)))
    return ((y - mu) / max(sd, 1e-8)).astype(np.float32), mu, max(sd, 1e-8)


def make_split(df: pd.DataFrame, mode: str, seed: int, test_fraction: float, val_fraction: float, group_col: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(df)
    if mode == "cv2":
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(n * test_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]
    if group_col is None or group_col not in df.columns:
        raise SystemExit(f"Split {mode} requires group column {group_col}")
    groups = np.asarray(df[group_col].fillna("").astype(str).unique(), dtype=object).copy()
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_fraction)))
    n_val = max(1, int(round(len(groups) * val_fraction)))
    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test : n_test + n_val])
    g = df[group_col].fillna("").astype(str)
    test = np.where(g.isin(test_groups))[0]
    val = np.where(g.isin(val_groups))[0]
    train = np.where(~g.isin(test_groups | val_groups))[0]
    return train, val, test


def split_leakage_record(
    df: pd.DataFrame,
    repeat: int,
    split_mode: str,
    group_col: str | None,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> dict[str, object]:
    train_rows, val_rows, test_rows = set(train.tolist()), set(val.tolist()), set(test.tolist())
    row_overlaps = {
        "train_val_row_overlap": len(train_rows & val_rows),
        "train_test_row_overlap": len(train_rows & test_rows),
        "val_test_row_overlap": len(val_rows & test_rows),
    }
    group_overlaps = {
        "train_val_group_overlap": 0,
        "train_test_group_overlap": 0,
        "val_test_group_overlap": 0,
    }
    if group_col is not None:
        groups = df[group_col].fillna("").astype(str)
        train_groups = set(groups.iloc[train])
        val_groups = set(groups.iloc[val])
        test_groups = set(groups.iloc[test])
        group_overlaps = {
            "train_val_group_overlap": len(train_groups & val_groups),
            "train_test_group_overlap": len(train_groups & test_groups),
            "val_test_group_overlap": len(val_groups & test_groups),
        }
    leakage = any(value > 0 for value in [*row_overlaps.values(), *group_overlaps.values()])
    return {
        "repeat": repeat,
        "split_mode": split_mode,
        "group_column": group_col or "",
        "train_n": len(train),
        "val_n": len(val),
        "test_n": len(test),
        **row_overlaps,
        **group_overlaps,
        "leakage_status": "fail" if leakage else "pass",
    }


def build_features(
    ablation: str,
    G: np.ndarray,
    E: np.ndarray,
    G_RBF: np.ndarray | None,
    gi: np.ndarray,
    ei: np.ndarray,
    rank_ge_g: int,
    rank_ge_e: int,
) -> np.ndarray:
    terms = set(ablation.split("+"))
    parts = [np.ones((len(gi), 1), dtype=np.float32)]
    if "G" in terms:
        parts.append(G[gi])
    if "RBF" in terms:
        if G_RBF is None:
            raise ValueError("Ablation requests RBF but --k-g-rbf-unique was not supplied")
        parts.append(G_RBF[gi])
    if "E" in terms:
        parts.append(E[ei])
    if "GE" in terms:
        Gs = G[gi, : min(rank_ge_g, G.shape[1])]
        Es = E[ei, : min(rank_ge_e, E.shape[1])]
        parts.append((Gs[:, :, None] * Es[:, None, :]).reshape(len(gi), -1))
    if "RBFE" in terms:
        if G_RBF is None:
            raise ValueError("Ablation requests RBFE but --k-g-rbf-unique was not supplied")
        Gs = G_RBF[gi, : min(rank_ge_g, G_RBF.shape[1])]
        Es = E[ei, : min(rank_ge_e, E.shape[1])]
        parts.append((Gs[:, :, None] * Es[:, None, :]).reshape(len(gi), -1))
    return np.hstack(parts).astype(np.float32)


def fit_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, train: np.ndarray, lam: float) -> np.ndarray:
    Xt = X[train].astype(np.float64)
    yt = y[train].astype(np.float64)
    wt = np.sqrt(np.where(np.isfinite(w[train]) & (w[train] > 0), w[train], 1.0)).astype(np.float64)
    Xw = Xt * wt[:, None]
    yw = yt * wt
    penalty = np.eye(X.shape[1], dtype=np.float64) * lam
    penalty[0, 0] = 0.0
    return np.linalg.solve(Xw.T @ Xw + penalty, Xw.T @ yw).astype(np.float32)


def metric_rows(y: np.ndarray, pred: np.ndarray, w: np.ndarray, idx: np.ndarray, split: str) -> dict[str, float | str | int]:
    yy = y[idx]
    pp = pred[idx]
    ww = np.where(np.isfinite(w[idx]) & (w[idx] > 0), w[idx], 1.0)
    err = pp - yy
    rmse = float(np.sqrt(np.sum(ww * err * err) / np.sum(ww)))
    mae = float(np.sum(ww * np.abs(err)) / np.sum(ww))
    corr = float(np.corrcoef(yy, pp)[0, 1]) if len(idx) > 2 and np.std(yy) > 0 and np.std(pp) > 0 else np.nan
    return {"split": split, "n": int(len(idx)), "rmse": rmse, "mae": mae, "pearson": corr}


def family_group(df: pd.DataFrame, fallback_col: str) -> pd.Series:
    if fallback_col in df.columns:
        raw = df[fallback_col].fillna("").astype(str)
    elif "canonical_germplasm_key" in df.columns:
        raw = df["canonical_germplasm_key"].fillna("").astype(str)
    else:
        raw = df["panel_sample_id"].fillna("").astype(str)
    return raw.str.replace(r"\s+", " ", regex=True).str.split(r"[/\\]|//| X | x |-", regex=True).str[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scalable validation and ablation suite for low-rank multikernel baseline.")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--k-g-unique", type=Path, required=True)
    parser.add_argument("--k-g-rbf-unique", type=Path)
    parser.add_argument("--k-e-unique", type=Path, required=True)
    parser.add_argument("--k-g-order", type=Path, required=True)
    parser.add_argument("--k-e-order", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("trained_models/validation_ablation"))
    parser.add_argument("--prefix", default="validation_ablation")
    parser.add_argument("--trait", action="append")
    parser.add_argument("--rank-g", type=int, default=96)
    parser.add_argument("--rank-g-rbf", type=int, default=96)
    parser.add_argument("--rank-e", type=int, default=48)
    parser.add_argument("--rank-ge-g", type=int, default=16)
    parser.add_argument("--rank-ge-e", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-observations", type=int, default=0)
    parser.add_argument("--lofo-col", default="canonical_germplasm_key")
    parser.add_argument("--ablation", action="append")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    obs = read_table(args.observations)
    obs, selected_trait = select_single_trait(obs, args.trait)
    print(f"Selected trait: {selected_trait}; rows before response filtering: {len(obs):,}", flush=True)
    obs["phenotype_value"] = pd.to_numeric(obs["phenotype_value"], errors="coerce")
    obs["weight_g_e"] = pd.to_numeric(obs["weight_g_e"], errors="coerce").fillna(1.0)
    obs = obs[obs["phenotype_value"].notna()].reset_index(drop=True)
    if obs.empty:
        raise SystemExit(f"Selected trait has zero finite phenotype rows: {selected_trait}")
    print(f"Selected trait: {selected_trait}; finite phenotype rows: {len(obs):,}", flush=True)
    if args.max_observations and len(obs) > args.max_observations:
        obs = obs.sample(args.max_observations, random_state=args.seed).reset_index(drop=True)

    gi = map_compact(obs, "geno_kernel_index", args.k_g_order)
    ei = map_compact(obs, "env_kernel_index", args.k_e_order)
    G = top_factors(args.k_g_unique, args.rank_g)
    G_RBF = top_factors(args.k_g_rbf_unique, args.rank_g_rbf) if args.k_g_rbf_unique else None
    E = top_factors(args.k_e_unique, args.rank_e)
    ablations = args.ablation or (DEFAULT_ABLATIONS if G_RBF is not None else DEFAULT_ABLATIONS[:4])
    y_raw = obs["phenotype_value"].to_numpy(dtype=np.float32)
    w = obs["weight_g_e"].to_numpy(dtype=np.float32)
    obs["_lofo_group"] = family_group(obs, args.lofo_col)

    split_cols = {
        "cv2": None,
        "loeo": "env_kernel_id",
        "loyo": "cycle",
        "loto": "trial_name",
        "loco": "country",
        "lofo": "_lofo_group",
    }
    rows = []
    leakage_rows = []
    for repeat in range(args.repeats):
        for split_mode, group_col in split_cols.items():
            try:
                train, val, test = make_split(obs, split_mode, args.seed + repeat, args.test_fraction, args.val_fraction, group_col)
            except SystemExit as exc:
                rows.append({"repeat": repeat, "split_mode": split_mode, "ablation": "NA", "split": "skipped", "n": 0, "rmse": np.nan, "mae": np.nan, "pearson": np.nan, "note": str(exc)})
                leakage_rows.append({"repeat": repeat, "split_mode": split_mode, "group_column": group_col or "", "leakage_status": "skipped", "note": str(exc)})
                continue
            leakage_rows.append(split_leakage_record(obs, repeat, split_mode, group_col, train, val, test))
            if len(train) == 0 or len(test) == 0:
                rows.append({"repeat": repeat, "split_mode": split_mode, "ablation": "NA", "split": "skipped", "n": 0, "rmse": np.nan, "mae": np.nan, "pearson": np.nan, "note": "empty train/test"})
                continue
            y, mu, sd = weighted_standardize(y_raw, w, train)
            for ablation in ablations:
                X = build_features(ablation, G, E, G_RBF, gi, ei, args.rank_ge_g, args.rank_ge_e)
                beta = fit_ridge(X, y, w, train, args.ridge)
                pred = (X @ beta) * sd + mu
                for label, idx in [("train", train), ("val", val), ("test", test)]:
                    rec = metric_rows(y_raw, pred, w, idx, label)
                    rec.update({"repeat": repeat, "split_mode": split_mode, "ablation": ablation, "note": ""})
                    rows.append(rec)
                print(f"repeat={repeat} split={split_mode} ablation={ablation} test_n={len(test)}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / f"{args.prefix}_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(leakage_rows).to_csv(args.out_dir / "split_leakage_qc.tsv", sep="\t", index=False)
    summary = (
        result[result["split"].eq("test")]
        .groupby(["split_mode", "ablation"], dropna=False)
        .agg(n_mean=("n", "mean"), rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), pearson_mean=("pearson", "mean"), pearson_sd=("pearson", "std"))
        .reset_index()
    )
    summary.to_csv(args.out_dir / f"{args.prefix}_summary.tsv", sep="\t", index=False)
    config = vars(args) | {"selected_trait": selected_trait, "observations_used": int(len(obs)), "ablations_used": ablations}
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, default=str, indent=2)
    if args.prefix != "validation_ablation":
        with (args.out_dir / f"{args.prefix}_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, default=str, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
