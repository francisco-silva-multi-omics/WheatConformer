from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_prediction(prefix_dir: Path, prefix: str, split: str) -> pd.DataFrame:
    for suffix in [".tsv.gz", ".parquet"]:
        path = prefix_dir / f"{prefix}_{split}_predictions{suffix}"
        if path.exists():
            if suffix == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path, sep="\t", low_memory=False)
    raise FileNotFoundError(f"Missing {split} predictions for {prefix} in {prefix_dir}")


def validate_prediction_split(df: pd.DataFrame, expected_split: str) -> None:
    if "baseline_split" not in df.columns:
        return
    observed = set(df["baseline_split"].fillna("").astype(str).str.lower().unique())
    if observed != {expected_split}:
        raise SystemExit(
            f"Prediction file for split {expected_split!r} contains baseline_split labels {sorted(observed)}"
        )


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
        "unweighted_rmse": float(np.sqrt(np.mean(err * err))),
        "unweighted_mae": float(np.mean(np.abs(err))),
        "weight_sum": float(np.sum(w)),
    }


def score_grid(df: pd.DataFrame, split: str, grid: list[float], weight_col: str) -> pd.DataFrame:
    y = df["original_phenotype_value"].to_numpy(float)
    base = df["env_baseline_pred"].to_numpy(float)
    residual = df["y_pred"].to_numpy(float)
    weight = df[weight_col].to_numpy(float) if weight_col in df.columns else None
    rows = []
    for lam in grid:
        m = metrics(y, base + lam * residual, weight)
        rows.append({"split": split, "lambda": lam, "weight_col": weight_col if weight is not None else "", **m})
    return pd.DataFrame(rows)


def validate_lambda0_against_baseline(
    val_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    baseline_selected: Path,
    seed: int,
    tolerance: float,
) -> None:
    baseline = pd.read_csv(baseline_selected, sep="\t")
    rows = baseline[baseline["seed"].astype(int).eq(int(seed))]
    if rows.empty:
        raise SystemExit(f"No baseline-selected row found for seed {seed} in {baseline_selected}")
    row = rows.iloc[0]
    checks = [
        ("val", "rmse", float(row["val_rmse"]), float(val_scores[val_scores["lambda"].eq(0)]["rmse"].iloc[0])),
        ("val", "mae", float(row["val_mae"]), float(val_scores[val_scores["lambda"].eq(0)]["mae"].iloc[0])),
        ("test", "rmse", float(row["test_rmse"]), float(test_scores[test_scores["lambda"].eq(0)]["rmse"].iloc[0])),
        ("test", "mae", float(row["test_mae"]), float(test_scores[test_scores["lambda"].eq(0)]["mae"].iloc[0])),
    ]
    mismatches = [
        f"{split}_{metric}: baseline={expected:.12g}; lambda0={observed:.12g}; diff={abs(expected - observed):.6g}"
        for split, metric, expected, observed in checks
        if abs(expected - observed) > tolerance
    ]
    if mismatches:
        raise SystemExit("Lambda-0 residual baseline mismatch:\n" + "\n".join(mismatches))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune DTH residual shrinkage on validation and report test metrics.")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", type=Path, default=Path("trained_models/model_comparisons/dth_residual_shrinkage.tsv"))
    parser.add_argument("--lambda-grid", default="0,0.05,0.1,0.2,0.3,0.4,0.5,0.75,1")
    parser.add_argument("--min-rmse-relative-gain", type=float, default=0.02)
    parser.add_argument("--min-rmse-absolute-gain", type=float, default=0.5)
    parser.add_argument("--max-pearson-drop", type=float, default=0.02)
    parser.add_argument(
        "--weight-col",
        default="original_weight_g_e",
        help="Weight column for decision metrics. Falls back to weight_g_e, then unweighted, if absent.",
    )
    parser.add_argument("--baseline-selected", type=Path, help="Selected baseline TSV used to assert lambda=0 alignment.")
    parser.add_argument("--seed", type=int, help="Seed row to validate in --baseline-selected.")
    parser.add_argument("--baseline-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    grid = [float(x) for x in args.lambda_grid.split(",") if x.strip()]
    val = read_prediction(args.prediction_dir, args.prefix, "val")
    test = read_prediction(args.prediction_dir, args.prefix, "test")
    validate_prediction_split(val, "val")
    validate_prediction_split(test, "test")
    if args.weight_col in val.columns and args.weight_col in test.columns:
        weight_col = args.weight_col
    elif "weight_g_e" in val.columns and "weight_g_e" in test.columns:
        weight_col = "weight_g_e"
    else:
        weight_col = ""
    val_scores = score_grid(val, "val", grid, weight_col)
    test_scores = score_grid(test, "test", grid, weight_col)
    if args.baseline_selected is not None:
        if args.seed is None:
            raise SystemExit("--seed is required with --baseline-selected")
        validate_lambda0_against_baseline(val_scores, test_scores, args.baseline_selected, args.seed, args.baseline_tolerance)
    base_val = val_scores[val_scores["lambda"].eq(0)].iloc[0]
    candidate = val_scores.sort_values(["rmse", "lambda"]).iloc[0]
    rmse_gain = float(base_val["rmse"] - candidate["rmse"])
    rel_gain = rmse_gain / max(float(base_val["rmse"]), 1e-12)
    pearson_drop = float(base_val["pearson"] - candidate["pearson"]) if np.isfinite(base_val["pearson"]) else 0.0
    accepted = (
        (rel_gain >= args.min_rmse_relative_gain or rmse_gain >= args.min_rmse_absolute_gain)
        and pearson_drop <= args.max_pearson_drop
    )
    selected_lambda = float(candidate["lambda"]) if accepted else 0.0
    selected_test = test_scores[test_scores["lambda"].eq(selected_lambda)].iloc[0].to_dict()
    selected_val = val_scores[val_scores["lambda"].eq(selected_lambda)].iloc[0].to_dict()
    out = pd.concat([val_scores, test_scores], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    decision = pd.DataFrame(
        [
            {
                "selected_lambda": selected_lambda,
                "accepted_residual": bool(accepted),
                "best_val_lambda": float(candidate["lambda"]),
                "val_rmse_gain": rmse_gain,
                "val_relative_rmse_gain": rel_gain,
                "val_pearson_drop": pearson_drop,
                "selected_val_rmse": selected_val["rmse"],
                "selected_val_pearson": selected_val["pearson"],
                "selected_test_rmse": selected_test["rmse"],
                "selected_test_pearson": selected_test["pearson"],
                "weight_col_used": weight_col,
            }
        ]
    )
    decision_path = args.out.with_name(args.out.stem + "_decision.tsv")
    decision.to_csv(decision_path, sep="\t", index=False)
    print(decision.to_string(index=False))
    print(f"Wrote {args.out}")
    print(f"Wrote {decision_path}")


if __name__ == "__main__":
    main()
