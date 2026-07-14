from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated-seed multi-trait quantitative baselines.")
    parser.add_argument("--models-root", type=Path, default=Path("trained_models"))
    parser.add_argument("--run-glob", default="multitrait_quantitative_*_seed*")
    parser.add_argument("--out-dir", type=Path, default=Path("trained_models/model_comparisons"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metric_frames = []
    improvement_frames = []
    for run_dir in sorted(args.models_root.glob(args.run_glob)):
        metadata_paths = list(run_dir.glob("*_run_metadata.json"))
        metric_paths = list(run_dir.glob("*_trait_metrics.tsv"))
        improvement_paths = list(run_dir.glob("*_vs_train_mean.tsv"))
        if len(metadata_paths) != 1 or len(metric_paths) != 1:
            continue
        metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
        metrics = pd.read_csv(metric_paths[0], sep="\t")
        metrics.insert(0, "seed", metadata["seed"])
        metrics.insert(0, "run_dir", str(run_dir))
        metric_frames.append(metrics)
        if len(improvement_paths) == 1:
            improvement = pd.read_csv(improvement_paths[0], sep="\t")
            improvement.insert(0, "model_label", metadata["model_label"])
            improvement.insert(0, "seed", metadata["seed"])
            improvement.insert(0, "run_dir", str(run_dir))
            improvement_frames.append(improvement)

    if not metric_frames:
        raise SystemExit(f"No completed multi-trait runs found under {args.models_root}/{args.run_glob}")
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_metrics.to_csv(args.out_dir / "multitrait_quantitative_all_runs.tsv", sep="\t", index=False)
    summary = (
        all_metrics.groupby(["split", "coverage_group", "model", "trait_name_canonical"])[
            ["weighted_rmse", "unweighted_rmse", "pearson"]
        ]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(value) for value in column if str(value)) if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    summary.to_csv(args.out_dir / "multitrait_quantitative_seed_summary.tsv", sep="\t", index=False)

    if improvement_frames:
        all_improvement = pd.concat(improvement_frames, ignore_index=True)
        all_improvement.to_csv(
            args.out_dir / "multitrait_quantitative_vs_train_mean_all_runs.tsv", sep="\t", index=False
        )
        improvement_summary = (
            all_improvement.groupby(
                ["split", "coverage_group", "model_label", "trait_name_canonical"]
            )[
                ["weighted_rmse_improvement", "unweighted_rmse_improvement", "pearson_model"]
            ]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )
        improvement_summary.columns = [
            "_".join(str(value) for value in column if str(value))
            if isinstance(column, tuple)
            else str(column)
            for column in improvement_summary.columns
        ]
        improvement_summary.to_csv(
            args.out_dir / "multitrait_quantitative_vs_train_mean_seed_summary.tsv",
            sep="\t",
            index=False,
        )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
