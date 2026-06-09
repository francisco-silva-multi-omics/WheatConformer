from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .trait_isolation import select_single_trait
    from .split_utils import canonical_split_mode, group_kfold_splits, make_split, split_leakage_record
    from .kernel_factorization import effective_factorization_mode, kernel_factors
except ImportError:
    from trait_isolation import select_single_trait
    from split_utils import canonical_split_mode, group_kfold_splits, make_split, split_leakage_record
    from kernel_factorization import effective_factorization_mode, kernel_factors


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


def factorization_columns(
    factorization_mode: str,
    metadata: dict[str, dict[str, int | str] | None],
) -> dict[str, int | str | None]:
    columns: dict[str, int | str | None] = {"factorization_mode": factorization_mode}
    for label, details in metadata.items():
        for field in ("rank_requested", "rank_retained", "train_kernel_dimension"):
            columns[f"{label}_{field}"] = details[field] if details is not None else None
    return columns


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


def fold_skip_reason(
    leakage_record: dict[str, object],
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> str | None:
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        return "empty train/validation/test partition"
    if leakage_record["leakage_status"] != "pass":
        return "split leakage detected"
    return None


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
        "--factorization-mode",
        choices=["full_transductive", "train_nystrom"],
        default="full_transductive",
        help="Use complete-kernel factors, or train-only Nyström factors for CV1/CV0 splits.",
    )
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
    full_factors: tuple[
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray,
        dict[str, dict[str, int | str] | None],
    ] | None = None
    ablations = args.ablation or (DEFAULT_ABLATIONS if args.k_g_rbf_unique else DEFAULT_ABLATIONS[:4])
    if args.ablation is None and args.k_g_epi2_unique:
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
                leakage_record = split_leakage_record(obs, repeat, split_mode, train, val, test, group_col=group_col)
                leakage_rows.append(leakage_record)
            except SystemExit as exc:
                leakage_rows.append({"repeat": repeat, "split_mode": split_mode, "leakage_status": "skipped", "note": str(exc)})
                continue
            note = fold_skip_reason(leakage_record, train, val, test)
            if note is not None:
                rows.append({"repeat": repeat, "split_mode": split_mode, "ablation": "NA", "split": "skipped", "n": 0, "rmse": np.nan, "mae": np.nan, "pearson": np.nan, "note": note})
                if note == "empty train/validation/test partition":
                    leakage_rows[-1]["leakage_status"] = "skipped"
                leakage_rows[-1]["note"] = note
                continue
            fold_factorization_mode = effective_factorization_mode(args.factorization_mode, split_mode)
            if fold_factorization_mode == "train_nystrom":
                train_g_ids = np.unique(gi[train])
                train_e_ids = np.unique(ei[train])
                fold_G, fold_G_metadata = kernel_factors(args.k_g_unique, args.rank_g, train_g_ids)
                if args.k_g_rbf_unique:
                    fold_G_RBF, fold_G_RBF_metadata = kernel_factors(
                        args.k_g_rbf_unique, args.rank_g_rbf, train_g_ids
                    )
                else:
                    fold_G_RBF, fold_G_RBF_metadata = None, None
                if args.k_g_epi2_unique:
                    fold_G_EPI2, fold_G_EPI2_metadata = kernel_factors(
                        args.k_g_epi2_unique, args.rank_g_epi2, train_g_ids
                    )
                else:
                    fold_G_EPI2, fold_G_EPI2_metadata = None, None
                fold_E, fold_E_metadata = kernel_factors(args.k_e_unique, args.rank_e, train_e_ids)
                fold_metadata = {
                    "g": fold_G_metadata,
                    "g_rbf": fold_G_RBF_metadata,
                    "g_epi2": fold_G_EPI2_metadata,
                    "e": fold_E_metadata,
                }
            else:
                if full_factors is None:
                    fold_G, fold_G_metadata = kernel_factors(args.k_g_unique, args.rank_g)
                    if args.k_g_rbf_unique:
                        fold_G_RBF, fold_G_RBF_metadata = kernel_factors(
                            args.k_g_rbf_unique, args.rank_g_rbf
                        )
                    else:
                        fold_G_RBF, fold_G_RBF_metadata = None, None
                    if args.k_g_epi2_unique:
                        fold_G_EPI2, fold_G_EPI2_metadata = kernel_factors(
                            args.k_g_epi2_unique, args.rank_g_epi2
                        )
                    else:
                        fold_G_EPI2, fold_G_EPI2_metadata = None, None
                    fold_E, fold_E_metadata = kernel_factors(args.k_e_unique, args.rank_e)
                    fold_metadata = {
                        "g": fold_G_metadata,
                        "g_rbf": fold_G_RBF_metadata,
                        "g_epi2": fold_G_EPI2_metadata,
                        "e": fold_E_metadata,
                    }
                    full_factors = fold_G, fold_G_RBF, fold_G_EPI2, fold_E, fold_metadata
                else:
                    fold_G, fold_G_RBF, fold_G_EPI2, fold_E, fold_metadata = full_factors
            factor_columns = factorization_columns(fold_factorization_mode, fold_metadata)
            y, mu, sd = weighted_standardize(y_raw, w, train)
            for ablation in ablations:
                X = build_features(
                    ablation,
                    fold_G,
                    fold_E,
                    fold_G_RBF,
                    fold_G_EPI2,
                    gi,
                    ei,
                    args.rank_ge_g,
                    args.rank_ge_e,
                )
                beta = fit_ridge(X, y, w, train, args.ridge)
                pred = (X @ beta) * sd + mu
                for label, idx in [("train", train), ("val", val), ("test", test)]:
                    rec = metric_rows(y_raw, pred, w, idx, label)
                    rec.update(
                        {
                            "repeat": repeat,
                            "split_mode": split_mode,
                            "ablation": ablation,
                            "note": "",
                            **factor_columns,
                        }
                    )
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
            skipped_empty=leakage_df.get("note", pd.Series("", index=leakage_df.index)).eq("empty train/validation/test partition"),
        )
        .groupby("split_mode", dropna=False)
        .agg(repeats_attempted=("repeat", "count"), repeats_passed=("passed", "sum"),
             repeats_failed=("failed", "sum"), repeats_skipped=("skipped", "sum"),
             repeats_skipped_empty=("skipped_empty", "sum"))
        .reset_index()
    )
    leakage_summary.insert(0, "trait", selected_trait)
    leakage_summary["leakage_free"] = leakage_summary["repeats_failed"].eq(0)
    leakage_summary.to_csv(args.out_dir / "split_leakage_summary.tsv", sep="\t", index=False)
    summary = (
        result[result["split"].eq("test") & result["ablation"].ne("NA")]
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
