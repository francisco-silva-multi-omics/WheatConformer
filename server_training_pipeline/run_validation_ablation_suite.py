from __future__ import annotations

import argparse
import json
import warnings
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

EPI2_ABLATIONS = [
    "G+EPI2+E",
    "G+EPI2+E+GE+EPI2E",
    "G+RBF+EPI2+E+GE+RBFE+EPI2E",
]

SPLIT_ALIASES = {
    "cv2": "cv2_random_observation",
    "loeo": "gho_environment",
    "loyo": "gho_cycle",
    "loto": "gho_trial",
    "loco": "gho_country",
    "lofo": "gho_family",
}

DEFAULT_SPLIT_MODES = [
    "cv2_random_observation",
    "gho_environment",
    "gho_cycle",
    "gho_trial",
    "gho_country",
    "gho_family",
    "cv1_genotype",
    "cv1_environment",
    "cv0_genotype_environment",
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


def canonical_split_mode(mode: str, warn: bool = False) -> str:
    normalized = mode.strip().lower()
    canonical = SPLIT_ALIASES.get(normalized, normalized)
    if warn and canonical != normalized:
        warnings.warn(
            f"Split mode {mode!r} is a backward-compatible alias; recording canonical mode {canonical!r}.",
            stacklevel=2,
        )
    return canonical


def grouped_holdout(
    df: pd.DataFrame,
    group_col: str,
    seed: int,
    test_fraction: float,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if group_col not in df.columns:
        raise SystemExit(f"Grouped holdout requires group column {group_col}")
    rng = np.random.default_rng(seed)
    groups = np.asarray(df[group_col].fillna("").astype(str).unique(), dtype=object).copy()
    if len(groups) < 3:
        raise SystemExit(f"Grouped holdout requires at least three groups in {group_col}; found {len(groups)}")
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_fraction)))
    n_val = max(1, int(round(len(groups) * val_fraction)))
    if n_test + n_val >= len(groups):
        n_val = 1
        n_test = 1
    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test : n_test + n_val])
    g = df[group_col].fillna("").astype(str)
    test = np.where(g.isin(test_groups))[0]
    val = np.where(g.isin(val_groups))[0]
    train = np.where(~g.isin(test_groups | val_groups))[0]
    return train, val, test


def cv0_split(
    df: pd.DataFrame,
    seed: int,
    test_fraction: float,
    val_fraction: float,
    genotype_col: str = "panel_sample_id",
    environment_col: str = "env_kernel_id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = [genotype_col, environment_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"cv0_genotype_environment requires columns: {missing}")
    rng = np.random.default_rng(seed)
    geno = df[genotype_col].fillna("").astype(str)
    env = df[environment_col].fillna("").astype(str)

    def partition(values: pd.Series) -> tuple[set[str], set[str]]:
        groups = np.asarray(values.unique(), dtype=object).copy()
        if len(groups) < 3:
            raise SystemExit("cv0_genotype_environment requires at least three groups on each axis")
        rng.shuffle(groups)
        n_test = max(1, int(round(len(groups) * test_fraction)))
        n_val = max(1, int(round(len(groups) * val_fraction)))
        if n_test + n_val >= len(groups):
            n_test = n_val = 1
        return set(groups[:n_test]), set(groups[n_test : n_test + n_val])

    test_g, val_g = partition(geno)
    test_e, val_e = partition(env)
    test = np.where(geno.isin(test_g) & env.isin(test_e))[0]
    val = np.where(geno.isin(val_g) & env.isin(val_e))[0]
    train = np.where(~geno.isin(test_g | val_g) & ~env.isin(test_e | val_e))[0]
    return train, val, test


def make_split(df: pd.DataFrame, mode: str, seed: int, test_fraction: float, val_fraction: float, group_col: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode = canonical_split_mode(mode)
    rng = np.random.default_rng(seed)
    n = len(df)
    if mode == "cv2_random_observation":
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(n * test_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]
    if mode == "cv0_genotype_environment":
        return cv0_split(df, seed, test_fraction, val_fraction)
    if group_col is None or group_col not in df.columns:
        raise SystemExit(f"Split {mode} requires group column {group_col}")
    return grouped_holdout(df, group_col, seed, test_fraction, val_fraction)


def group_kfold_splits(
    df: pd.DataFrame,
    group_col: str,
    splits: int,
    seed: int,
    val_fraction: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if group_col not in df.columns:
        raise SystemExit(f"group_kfold requires group column {group_col}")
    groups = np.asarray(sorted(df[group_col].fillna("").astype(str).unique()), dtype=object)
    if splits < 2 or len(groups) < splits:
        raise SystemExit(f"group_kfold requires 2 <= splits <= unique groups; got {splits} splits and {len(groups)} groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    folds = np.array_split(groups, splits)
    values = df[group_col].fillna("").astype(str)
    out = []
    for fold, test_values in enumerate(folds):
        test_groups = set(test_values)
        remaining = np.asarray([value for value in groups if value not in test_groups], dtype=object)
        fold_rng = np.random.default_rng(seed + fold + 1)
        fold_rng.shuffle(remaining)
        n_val = max(1, int(round(len(remaining) * val_fraction)))
        val_groups = set(remaining[:n_val])
        test = np.where(values.isin(test_groups))[0]
        val = np.where(values.isin(val_groups))[0]
        train = np.where(~values.isin(test_groups | val_groups))[0]
        out.append((train, val, test))
    return out


try:
    from .split_utils import canonical_split_mode, grouped_holdout, cv0_split, make_split, split_leakage_record
except ImportError:
    from split_utils import canonical_split_mode, grouped_holdout, cv0_split, make_split, split_leakage_record


def split_leakage_record(
    df: pd.DataFrame,
    repeat: int,
    split_mode: str,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    group_col: str | None = None,
) -> dict[str, object]:
    split_mode = canonical_split_mode(split_mode)
    geno_col = "panel_sample_id" if "panel_sample_id" in df.columns else "geno_kernel_index"
    env_col = "env_kernel_id" if "env_kernel_id" in df.columns else "env_kernel_index"
    geno = df[geno_col].fillna("").astype(str)
    env = df[env_col].fillna("").astype(str)
    train_g, val_g, test_g = set(geno.iloc[train]), set(geno.iloc[val]), set(geno.iloc[test])
    train_e, val_e, test_e = set(env.iloc[train]), set(env.iloc[val]), set(env.iloc[test])
    geno_overlap = len(train_g & test_g)
    env_overlap = len(train_e & test_e)
    expected_geno = "zero" if split_mode in {"cv1_genotype", "cv0_genotype_environment"} else "allowed"
    expected_env = "zero" if split_mode in {"gho_environment", "cv1_environment", "cv0_genotype_environment"} else "allowed"
    if split_mode == "group_kfold":
        if group_col in {"panel_sample_id", "geno_kernel_index"}:
            expected_geno = "zero"
        if group_col in {"env_kernel_id", "env_kernel_index"}:
            expected_env = "zero"
    leakage = (expected_geno == "zero" and geno_overlap > 0) or (expected_env == "zero" and env_overlap > 0)
    return {
        "repeat": repeat,
        "split_mode": split_mode,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_unique_genotypes": len(train_g),
        "val_unique_genotypes": len(val_g),
        "test_unique_genotypes": len(test_g),
        "train_unique_environments": len(train_e),
        "val_unique_environments": len(val_e),
        "test_unique_environments": len(test_e),
        "geno_overlap_train_test": geno_overlap,
        "env_overlap_train_test": env_overlap,
        "expected_geno_overlap": expected_geno,
        "expected_env_overlap": expected_env,
        "leakage_status": "fail" if leakage else "pass",
    }


try:
    from .split_utils import split_leakage_record
except ImportError:
    from split_utils import split_leakage_record


def build_features(
    ablation: str,
    G: np.ndarray,
    E: np.ndarray,
    G_RBF: np.ndarray | None,
    G_EPI2: np.ndarray | None,
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
    if "EPI2" in terms:
        if G_EPI2 is None:
            raise ValueError("Ablation requests EPI2 but --k-g-epi2-unique was not supplied")
        parts.append(G_EPI2[gi])
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
    if "EPI2E" in terms:
        if G_EPI2 is None:
            raise ValueError("Ablation requests EPI2E but --k-g-epi2-unique was not supplied")
        Gs = G_EPI2[gi, : min(rank_ge_g, G_EPI2.shape[1])]
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
    parser.add_argument("--k-g-epi2-unique", type=Path)
    parser.add_argument("--k-e-unique", type=Path, required=True)
    parser.add_argument("--k-g-order", type=Path, required=True)
    parser.add_argument("--k-e-order", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("trained_models/validation_ablation"))
    parser.add_argument("--prefix", default="validation_ablation")
    parser.add_argument("--trait", action="append")
    parser.add_argument("--rank-g", type=int, default=96)
    parser.add_argument("--rank-g-rbf", type=int, default=96)
    parser.add_argument("--rank-g-epi2", type=int, default=96)
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
    parser.add_argument("--group-kfold-col", default="env_kernel_id")
    parser.add_argument("--group-kfold-splits", type=int, default=5)
    parser.add_argument(
        "--split-mode",
        action="append",
        help="Split mode to run; can be repeated. Default: run all available modes.",
    )
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
    G_EPI2 = top_factors(args.k_g_epi2_unique, args.rank_g_epi2) if args.k_g_epi2_unique else None
    E = top_factors(args.k_e_unique, args.rank_e)
    ablations = args.ablation or (DEFAULT_ABLATIONS if G_RBF is not None else DEFAULT_ABLATIONS[:4])
    if args.ablation is None and G_EPI2 is not None:
        ablations = ablations + EPI2_ABLATIONS
    y_raw = obs["phenotype_value"].to_numpy(dtype=np.float32)
    w = obs["weight_g_e"].to_numpy(dtype=np.float32)
    obs["_lofo_group"] = family_group(obs, args.lofo_col)

    split_cols = {
        "cv2_random_observation": None,
        "gho_environment": "env_kernel_id",
        "gho_cycle": "cycle",
        "gho_trial": "trial_name",
        "gho_country": "country",
        "gho_family": "_lofo_group",
        "cv1_genotype": "panel_sample_id",
        "cv1_environment": "env_kernel_id",
        "cv0_genotype_environment": None,
        "group_kfold": args.group_kfold_col,
    }
    if args.split_mode:
        selected_modes = [canonical_split_mode(mode, warn=True) for mode in args.split_mode]
        unknown = sorted(set(selected_modes).difference(split_cols))
        if unknown:
            raise SystemExit(f"Unknown --split-mode values: {unknown}; choose from {sorted(split_cols)}")
    else:
        selected_modes = DEFAULT_SPLIT_MODES
    split_cols = {mode: split_cols[mode] for mode in selected_modes}
    rows = []
    leakage_rows = []
    for split_mode, group_col in split_cols.items():
        if split_mode == "group_kfold":
            split_runs = group_kfold_splits(obs, group_col, args.group_kfold_splits, args.seed, args.val_fraction)
        else:
            split_runs = []
            for repeat in range(args.repeats):
                try:
                    split_runs.append(make_split(obs, split_mode, args.seed + repeat, args.test_fraction, args.val_fraction, group_col))
                except SystemExit as exc:
                    rows.append({"repeat": repeat, "split_mode": split_mode, "ablation": "NA", "split": "skipped", "n": 0, "rmse": np.nan, "mae": np.nan, "pearson": np.nan, "note": str(exc)})
                    leakage_rows.append({"repeat": repeat, "split_mode": split_mode, "leakage_status": "skipped", "note": str(exc)})
        for repeat, (train, val, test) in enumerate(split_runs):
            try:
                leakage_rows.append(split_leakage_record(obs, repeat, split_mode, train, val, test, group_col=group_col))
            except SystemExit as exc:
                leakage_rows.append({"repeat": repeat, "split_mode": split_mode, "leakage_status": "skipped", "note": str(exc)})
                continue
            if len(train) == 0 or len(val) == 0 or len(test) == 0:
                note = "empty train/validation/test partition"
                rows.append({"repeat": repeat, "split_mode": split_mode, "ablation": "NA", "split": "skipped", "n": 0, "rmse": np.nan, "mae": np.nan, "pearson": np.nan, "note": note})
                leakage_rows[-1]["leakage_status"] = "skipped"
                leakage_rows[-1]["note"] = note
                continue
            y, mu, sd = weighted_standardize(y_raw, w, train)
            for ablation in ablations:
                X = build_features(ablation, G, E, G_RBF, G_EPI2, gi, ei, args.rank_ge_g, args.rank_ge_e)
                beta = fit_ridge(X, y, w, train, args.ridge)
                pred = (X @ beta) * sd + mu
                for label, idx in [("train", train), ("val", val), ("test", test)]:
                    rec = metric_rows(y_raw, pred, w, idx, label)
                    rec.update({"repeat": repeat, "split_mode": split_mode, "ablation": ablation, "note": ""})
                    rows.append(rec)
                print(f"repeat={repeat} split={split_mode} ablation={ablation} test_n={len(test)}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / f"{args.prefix}_metrics.tsv", sep="\t", index=False)
    leakage_df = pd.DataFrame(leakage_rows)
    leakage_df.to_csv(args.out_dir / "split_leakage_qc.tsv", sep="\t", index=False)
    leakage_summary = (
        leakage_df.assign(
            passed=leakage_df["leakage_status"].eq("pass"),
            failed=leakage_df["leakage_status"].eq("fail"),
            skipped=leakage_df["leakage_status"].eq("skipped"),
        )
        .groupby("split_mode", dropna=False)
        .agg(repeats_attempted=("repeat", "count"), repeats_passed=("passed", "sum"),
             repeats_failed=("failed", "sum"), repeats_skipped=("skipped", "sum"))
        .reset_index()
    )
    leakage_summary.insert(0, "trait", selected_trait)
    leakage_summary["leakage_free"] = leakage_summary["repeats_failed"].eq(0)
    leakage_summary.to_csv(args.out_dir / "split_leakage_summary.tsv", sep="\t", index=False)
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
