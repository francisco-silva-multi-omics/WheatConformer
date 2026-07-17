from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


FAMILY_BY_SCENARIO = {
    "unseen_environments": "unseen_environments",
    "unseen_genotypes": "unseen_genotypes",
    "unseen_genotypes_and_environments": "unseen_genotypes_and_environments",
    "temporal_holdout": "temporal_country_holdout",
    "country_holdout": "temporal_country_holdout",
}


def read_table(path: Path) -> pd.DataFrame:
    if "".join(path.suffixes).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - prediction))))


def mae(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(y - prediction)))


def pearson(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) < 2 or np.std(y) <= 0 or np.std(prediction) <= 0:
        return float("nan")
    return float(np.corrcoef(y, prediction)[0, 1])


def calibration_parameters(frame: pd.DataFrame) -> tuple[float, float]:
    x = frame["y_pred"].to_numpy(dtype=float)
    y = frame["phenotype_value"].to_numpy(dtype=float)
    if len(x) < 2 or np.var(x) <= 0:
        return float(np.mean(y)), 0.0
    slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    intercept = float(np.mean(y) - slope * np.mean(x))
    return intercept, slope


def t_critical_95(df: int) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df))
    except Exception:
        return 1.96


def aggregate_metric(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    n = len(numeric)
    mean = float(numeric.mean()) if n else float("nan")
    sd = float(numeric.std(ddof=1)) if n > 1 else float("nan")
    half = t_critical_95(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {
        "fold_count": n,
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half if n > 1 else float("nan"),
        "ci95_high": mean + half if n > 1 else float("nan"),
    }


def run_record(run_dir: Path) -> tuple[pd.DataFrame, dict[str, object]] | None:
    metadata_paths = list(run_dir.glob("*_run_metadata.json"))
    prediction_paths = list(run_dir.glob("*_predictions.parquet"))
    if not prediction_paths:
        prediction_paths = list(run_dir.glob("*_predictions.tsv.gz"))
    if len(metadata_paths) != 1 or len(prediction_paths) != 1:
        return None
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    if metadata.get("evaluation_stage") != "outer_evaluation":
        return None
    external = metadata.get("external_split", {})
    predictions = read_table(prediction_paths[0])
    predictions["run_dir"] = str(run_dir)
    predictions["scenario"] = external.get("scenario")
    predictions["outer_fold"] = external.get("outer_fold")
    predictions["inner_fold"] = external.get("inner_fold")
    predictions["model_label"] = metadata.get("model_label")
    return predictions, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize locked nested outer-fold evaluation with calibration and CIs."
    )
    parser.add_argument("--models-root", type=Path, default=Path("trained_models"))
    parser.add_argument("--run-glob", default="final_nested_*")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames = []
    for run_dir in sorted(args.models_root.glob(args.run_glob)):
        record = run_record(run_dir)
        if record is not None:
            prediction_frames.append(record[0])
    if not prediction_frames:
        raise SystemExit("No completed outer-evaluation prediction files were found")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if "final_holdout" in predictions.get("split", pd.Series(dtype=str)).astype(str).unique():
        raise SystemExit("Locked final-holdout predictions must not enter nested summaries")

    fold_rows = []
    group_columns = ["run_dir", "scenario", "outer_fold", "model_label", "trait_name_canonical"]
    for key, group in predictions.groupby(group_columns, dropna=False, sort=True):
        val = group[group["split"].eq("val")]
        test = group[group["split"].eq("test")]
        if val.empty or test.empty:
            continue
        intercept, slope = calibration_parameters(val)
        y = test["phenotype_value"].to_numpy(dtype=float)
        prediction = test["y_pred"].to_numpy(dtype=float)
        calibrated = intercept + slope * prediction
        baseline = test["y_pred_train_mean"].to_numpy(dtype=float)
        fold_rows.append(
            {
                **dict(zip(group_columns, key)),
                "test_rows": len(test),
                "test_rmse": rmse(y, prediction),
                "test_mae": mae(y, prediction),
                "test_pearson": pearson(y, prediction),
                "test_prediction_sd_ratio": float(np.std(prediction) / np.std(y))
                if np.std(y) > 0
                else float("nan"),
                "train_mean_rmse": rmse(y, baseline),
                "train_mean_improvement": rmse(y, baseline) - rmse(y, prediction),
                "calibration_intercept_from_validation": intercept,
                "calibration_slope_from_validation": slope,
                "calibrated_test_rmse": rmse(y, calibrated),
                "calibrated_test_mae": mae(y, calibrated),
                "calibrated_test_pearson": pearson(y, calibrated),
            }
        )
    folds = pd.DataFrame(fold_rows)
    folds.insert(1, "generalization_family", folds["scenario"].map(FAMILY_BY_SCENARIO))
    folds.to_csv(args.out_dir / "nested_outer_fold_metrics.tsv", sep="\t", index=False)

    metrics = [
        "test_rmse",
        "test_mae",
        "test_pearson",
        "test_prediction_sd_ratio",
        "train_mean_improvement",
        "calibrated_test_rmse",
        "calibrated_test_mae",
        "calibrated_test_pearson",
        "calibration_slope_from_validation",
    ]
    summary_rows = []
    keys = ["generalization_family", "scenario", "model_label", "trait_name_canonical"]
    for key, group in folds.groupby(keys, dropna=False, sort=True):
        for metric in metrics:
            summary_rows.append(
                {**dict(zip(keys, key)), "metric": metric, **aggregate_metric(group[metric])}
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "nested_outer_fold_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
