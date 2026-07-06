from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
except Exception:  # pragma: no cover - server env controls availability
    HistGradientBoostingRegressor = None
    Ridge = None

try:
    from .split_utils import canonical_split_mode, make_split, split_group_column, split_leakage_record
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from split_utils import canonical_split_mode, make_split, split_group_column, split_leakage_record


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def weighted_mean(y: np.ndarray, w: np.ndarray) -> float:
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    return float(np.sum(y * w) / np.sum(w))


def metrics(y: np.ndarray, pred: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred)
    y = y[ok]
    pred = pred[ok]
    if w is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(w, dtype=float)[ok]
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    err = pred - y
    corr = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 2 and np.std(y) > 0 and np.std(pred) > 0 else np.nan
    return {
        "rmse": float(np.sqrt(np.sum(w * err * err) / np.sum(w))),
        "mae": float(np.sum(w * np.abs(err)) / np.sum(w)),
        "pearson": corr,
        "pred_sd": float(np.std(pred, ddof=1)) if len(pred) > 1 else np.nan,
    }


def map_compact(obs: pd.DataFrame, order_path: Path, index_col: str) -> np.ndarray:
    order = pd.read_csv(order_path, sep="\t")
    if {"source_kernel_index", "compact_kernel_index"}.issubset(order.columns):
        mapper = dict(zip(order["source_kernel_index"].astype(int), order["compact_kernel_index"].astype(int)))
        mapped = obs[index_col].astype(int).map(mapper)
    else:
        mapped = obs[index_col].astype(int)
    if mapped.isna().any():
        raise SystemExit(f"Could not map all rows with {order_path}")
    return mapped.to_numpy(dtype=np.int32)


def fit_kernel_ridge(K: np.ndarray, train_env: np.ndarray, y_env: np.ndarray, global_mu: float, lambdas: list[float]) -> list[tuple[str, np.ndarray]]:
    preds = []
    for lam in lambdas:
        A = K[np.ix_(train_env, train_env)].astype(float)
        A.flat[:: A.shape[0] + 1] += lam
        alpha = np.linalg.solve(A, y_env - global_mu)
        pred_all = global_mu + K[:, train_env] @ alpha
        preds.append((f"kernel_ridge_lambda_{lam:g}", pred_all))
    return preds


def fit_feature_models(
    X: np.ndarray,
    train_env: np.ndarray,
    y_env: np.ndarray,
    w_env: np.ndarray,
    lambdas: list[float],
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    preds: list[tuple[str, np.ndarray]] = []
    if Ridge is not None:
        for lam in lambdas:
            model = Ridge(alpha=lam, random_state=seed)
            model.fit(X[train_env], y_env, sample_weight=w_env)
            preds.append((f"ridge_lambda_{lam:g}", model.predict(X)))
    if HistGradientBoostingRegressor is not None and len(train_env) >= 20:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=0.1,
            random_state=seed,
        )
        model.fit(X[train_env], y_env, sample_weight=w_env)
        preds.append(("hist_gradient_boosting", model.predict(X)))
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DTH environment-mean baselines across seeds.")
    parser.add_argument("--model-dir", type=Path, default=Path("model_kernels/stage1_pedigree_env_dth_v2"))
    parser.add_argument("--prefix", default="stage1_pedigree_env")
    parser.add_argument("--trait", default="DAYS_TO_HEADING")
    parser.add_argument("--split", default="loeo")
    parser.add_argument("--seeds", default="2026,2027,2028,2029")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--ridge-grid", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--out-dir", type=Path, default=Path("trained_models/dth_env_baseline_v2"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    lambdas = [float(x) for x in args.ridge_grid.split(",") if x.strip()]
    canonical_split = canonical_split_mode(args.split, warn=True)
    group_col = split_group_column(canonical_split)

    obs = read_table(args.model_dir / f"{args.prefix}_model_ready_stage1_observations.parquet")
    obs = obs[obs["trait_name_canonical"].astype(str).str.upper().eq(args.trait.upper())].copy()
    obs["phenotype_value"] = pd.to_numeric(obs["phenotype_value"], errors="coerce")
    obs["weight_g_e"] = pd.to_numeric(obs["weight_g_e"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    obs["weight_g_e"] = np.where(obs["weight_g_e"] > 0, obs["weight_g_e"], 1.0)
    obs = obs[obs["phenotype_value"].notna()].reset_index(drop=True)
    obs["e_compact"] = map_compact(obs, args.model_dir / f"{args.prefix}_K_E_unique_order.tsv", "env_kernel_index")

    features = pd.read_parquet(args.model_dir / f"{args.prefix}_DTH_env_features_v2.parquet")
    X = features.drop(columns=["env_id"], errors="ignore").to_numpy(dtype=np.float32)
    K = np.load(args.model_dir / f"{args.prefix}_K_E_unique.npy").astype(float)
    K = (K + K.T) / 2

    rows = []
    for seed in seeds:
        train_idx, val_idx, test_idx = make_split(obs, canonical_split, seed, args.test_fraction, args.val_fraction, group_col)
        leakage = split_leakage_record(obs, 0, canonical_split, train_idx, val_idx, test_idx, group_col)
        if leakage["leakage_status"] != "pass":
            raise SystemExit(f"Split leakage detected: {leakage}")

        train = obs.iloc[train_idx].copy()
        env_train = (
            train.groupby("e_compact")
            .apply(
                lambda x: pd.Series(
                    {
                        "y_mean": weighted_mean(x["phenotype_value"].to_numpy(float), x["weight_g_e"].to_numpy(float)),
                        "w_sum": float(x["weight_g_e"].sum()),
                    }
                )
            )
            .reset_index()
        )
        train_env = env_train["e_compact"].to_numpy(dtype=int)
        y_env = env_train["y_mean"].to_numpy(dtype=float)
        w_env = env_train["w_sum"].to_numpy(dtype=float)
        global_mu = weighted_mean(train["phenotype_value"].to_numpy(float), train["weight_g_e"].to_numpy(float))

        candidates: list[tuple[str, np.ndarray]] = [("train_mean", np.full(K.shape[0], global_mu, dtype=float))]
        candidates.extend(fit_kernel_ridge(K, train_env, y_env, global_mu, lambdas))
        candidates.extend(fit_feature_models(X, train_env, y_env, w_env, lambdas, seed))

        score_rows = []
        for name, pred_env in candidates:
            pred_row = pred_env[obs["e_compact"].to_numpy(dtype=int)]
            val_m = metrics(
                obs.iloc[val_idx]["phenotype_value"].to_numpy(float),
                pred_row[val_idx],
                obs.iloc[val_idx]["weight_g_e"].to_numpy(float),
            )
            test_m = metrics(
                obs.iloc[test_idx]["phenotype_value"].to_numpy(float),
                pred_row[test_idx],
                obs.iloc[test_idx]["weight_g_e"].to_numpy(float),
            )
            score_rows.append(
                {
                    "seed": seed,
                    "candidate": name,
                    "val_rmse": val_m["rmse"],
                    "val_mae": val_m["mae"],
                    "val_pearson": val_m["pearson"],
                    "test_rmse": test_m["rmse"],
                    "test_mae": test_m["mae"],
                    "test_pearson": test_m["pearson"],
                    "test_pred_sd": test_m["pred_sd"],
                    "rows_train": len(train_idx),
                    "rows_val": len(val_idx),
                    "rows_test": len(test_idx),
                    "canonical_split_mode": canonical_split,
                    "split_leakage_status": leakage["leakage_status"],
                }
            )
        seed_scores = pd.DataFrame(score_rows).sort_values(["val_rmse", "test_rmse"])
        best = seed_scores.iloc[0]
        rows.extend(score_rows)
        best_pred_env = dict(candidates)[best["candidate"]]
        all_pred_df = obs.copy()
        all_pred_df["y_true"] = all_pred_df["phenotype_value"]
        all_pred_df["env_baseline_pred"] = best_pred_env[all_pred_df["e_compact"].to_numpy(dtype=int)]
        all_pred_df["seed"] = seed
        all_pred_df["selected_candidate"] = best["candidate"]
        all_pred_df["baseline_split"] = "train"
        all_pred_df.loc[val_idx, "baseline_split"] = "val"
        all_pred_df.loc[test_idx, "baseline_split"] = "test"
        all_pred_df.to_csv(args.out_dir / f"dth_env_baseline_v2_seed{seed}_all_predictions.tsv.gz", sep="\t", index=False)

        pred_df = obs.iloc[test_idx].copy()
        pred_df["y_true"] = pred_df["phenotype_value"]
        pred_df["env_baseline_pred"] = best_pred_env[pred_df["e_compact"].to_numpy(dtype=int)]
        pred_df["seed"] = seed
        pred_df["selected_candidate"] = best["candidate"]
        pred_df.to_csv(args.out_dir / f"dth_env_baseline_v2_seed{seed}_test_predictions.tsv.gz", sep="\t", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / "dth_env_baseline_v2_candidate_metrics.tsv", sep="\t", index=False)
    selected = out.sort_values(["seed", "val_rmse"]).groupby("seed", as_index=False).first()
    selected.to_csv(args.out_dir / "dth_env_baseline_v2_selected_by_seed.tsv", sep="\t", index=False)
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
