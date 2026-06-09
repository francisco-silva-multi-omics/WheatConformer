from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RBF_BASELINES = {
    "RBF": "G",
    "RBF+E": "G+E",
    "RBF+E+RBFE": "G+E+GE",
    "G+RBF+E": "G+E",
    "G+RBF+E+GE+RBFE": "G+E+GE",
}


def combine_reports(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for summary_path in sorted(root.glob("*/validation_ablation_summary.tsv")):
        trait_dir = summary_path.parent
        config_path = trait_dir / "config.json"
        if not config_path.exists():
            raise SystemExit(f"Missing trait-specific ablation config: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        trait = str(config.get("selected_trait", trait_dir.name))
        summary = pd.read_csv(summary_path, sep="\t")
        summary = summary[summary["ablation"].notna() & summary["ablation"].ne("NA")].copy()
        leakage_path = trait_dir / "split_leakage_summary.tsv"
        if not leakage_path.exists():
            raise SystemExit(f"Missing trait-specific leakage summary: {leakage_path}")
        leakage = pd.read_csv(leakage_path, sep="\t").rename(
            columns={
                "repeats_attempted": "folds_attempted",
                "repeats_passed": "folds_passed",
                "repeats_failed": "folds_failed_leakage",
                "repeats_skipped_empty": "folds_skipped_empty",
            }
        )
        summary = summary.merge(
            leakage[["split_mode", "folds_attempted", "folds_passed", "folds_failed_leakage", "folds_skipped_empty"]],
            on="split_mode",
            how="left",
            validate="many_to_one",
        )
        summary.insert(0, "trait", trait)
        rows.append(summary)
    if not rows:
        raise SystemExit(f"No trait-specific validation_ablation_summary.tsv files found under {root}")

    report = pd.concat(rows, ignore_index=True)
    report["best_within_split"] = report.groupby(["trait", "split_mode"])["rmse_mean"].transform("min").eq(report["rmse_mean"])
    lookup = report.set_index(["trait", "split_mode", "ablation"])["rmse_mean"]

    def rbf_improves(row: pd.Series) -> bool:
        baseline = RBF_BASELINES.get(str(row["ablation"]))
        if baseline is None or not np.isfinite(row["rmse_mean"]):
            return False
        baseline_key = (row["trait"], row["split_mode"], baseline)
        return baseline_key in lookup.index and float(row["rmse_mean"]) < float(lookup.loc[baseline_key])

    report["rbf_improves_over_additive"] = report.apply(rbf_improves, axis=1)
    columns = [
        "trait",
        "split_mode",
        "ablation",
        "n_mean",
        "rmse_mean",
        "rmse_sd",
        "pearson_mean",
        "pearson_sd",
        "folds_attempted",
        "folds_passed",
        "folds_failed_leakage",
        "folds_skipped_empty",
        "best_within_split",
        "rbf_improves_over_additive",
    ]
    return report[columns].sort_values(["trait", "split_mode", "rmse_mean"], na_position="last").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine trait-specific validation/ablation summaries.")
    parser.add_argument("--root", type=Path, default=Path("trained_models/validation_ablation"))
    parser.add_argument("--output", type=Path, default=Path("trained_models/validation_ablation_report.tsv"))
    args = parser.parse_args()

    report = combine_reports(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {args.output} with {len(report):,} rows")


if __name__ == "__main__":
    main()
