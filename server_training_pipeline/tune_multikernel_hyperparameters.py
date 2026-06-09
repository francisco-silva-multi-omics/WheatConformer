from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .split_utils import canonical_split_mode
except ImportError:
    from split_utils import canonical_split_mode


SELECTION_ABLATION = "G+RBF+E+GE+RBFE"
HYPERPARAMETER_COLUMNS = ["ridge", "rank_g", "rank_g_rbf", "rank_e"]


def trait_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def select_hyperparameters(
    metrics: pd.DataFrame,
    split_mode: str,
    selection_ablation: str = SELECTION_ABLATION,
) -> tuple[pd.DataFrame, dict[str, object]]:
    split_mode = canonical_split_mode(split_mode, warn=True)
    required = {
        "split",
        "split_mode",
        "ablation",
        "repeat",
        "n",
        "rmse",
        "pearson",
        *HYPERPARAMETER_COLUMNS,
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"Hyperparameter metrics are missing required columns: {missing}")
    validation = metrics[
        metrics["split"].eq("val")
        & metrics["split_mode"].eq(split_mode)
        & metrics["ablation"].eq(selection_ablation)
    ].copy()
    if validation.empty:
        raise ValueError(
            f"No validation rows for split mode {split_mode!r} and ablation {selection_ablation!r}"
        )

    summary = (
        validation.groupby(HYPERPARAMETER_COLUMNS, dropna=False)
        .agg(
            validation_repeats=("repeat", "nunique"),
            validation_n_mean=("n", "mean"),
            validation_rmse_mean=("rmse", "mean"),
            validation_rmse_sd=("rmse", "std"),
            validation_pearson_mean=("pearson", "mean"),
            validation_pearson_sd=("pearson", "std"),
        )
        .reset_index()
        .sort_values(
            ["validation_rmse_mean", "validation_pearson_mean"],
            ascending=[True, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    summary.insert(0, "selection_ablation", selection_ablation)
    summary.insert(0, "split_mode", split_mode)
    summary["selected"] = False
    summary.loc[0, "selected"] = True
    winner = summary.iloc[0]
    selected = {
        "selection_split": "validation",
        "selection_ablation": selection_ablation,
        "split_mode": split_mode,
        "selection_criterion": "lowest validation RMSE; tie breaker highest validation Pearson correlation",
        "test_metrics_used_for_selection": False,
        "ridge": float(winner["ridge"]),
        "rank_g": int(winner["rank_g"]),
        "rank_g_rbf": int(winner["rank_g_rbf"]),
        "rank_e": int(winner["rank_e"]),
        "validation_rmse_mean": float(winner["validation_rmse_mean"]),
        "validation_pearson_mean": (
            float(winner["validation_pearson_mean"])
            if np.isfinite(winner["validation_pearson_mean"])
            else None
        ),
    }
    return summary, selected


def load_sweep_metrics(runs_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for metrics_path in sorted(runs_root.glob("*/hyperparameter_metrics.tsv")):
        config_path = metrics_path.with_name("hyperparameter_config.json")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing hyperparameter run config: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(metrics_path, sep="\t")
        for column in HYPERPARAMETER_COLUMNS:
            frame[column] = config[column]
        frame["run_directory"] = str(metrics_path.parent)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No hyperparameter_metrics.tsv files found directly under {runs_root}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select multikernel ridge and ranks using validation metrics only.")
    parser.add_argument("--trait", required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-mode", default="gho_environment")
    parser.add_argument("--selection-ablation", default=SELECTION_ABLATION)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    metrics = load_sweep_metrics(args.runs_root)
    summary, selected = select_hyperparameters(metrics, args.split_mode, args.selection_ablation)
    selected["trait"] = args.trait
    selected["trait_slug"] = trait_slug(args.trait)
    selected["seed"] = args.seed
    selected["requested_repeats"] = args.repeats
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "hyperparameter_validation_summary.tsv", sep="\t", index=False)
    (args.out_dir / "selected_hyperparameters.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    config = vars(args) | {
        "trait_slug": trait_slug(args.trait),
        "candidate_count": int(len(summary)),
        "test_metrics_used_for_selection": False,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
