from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.build_validation_ablation_report import combine_reports


def test_combined_report_excludes_na_and_includes_fold_counts(tmp_path: Path) -> None:
    trait_dir = tmp_path / "grain_yield"
    trait_dir.mkdir()
    (trait_dir / "config.json").write_text(json.dumps({"selected_trait": "Grain Yield"}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "split_mode": "cv1_genotype",
                "ablation": "G",
                "n_mean": 10,
                "rmse_mean": 1.0,
                "rmse_sd": 0.1,
                "pearson_mean": 0.5,
                "pearson_sd": 0.1,
            },
            {
                "split_mode": "cv1_genotype",
                "ablation": "NA",
                "n_mean": 0,
                "rmse_mean": float("nan"),
                "rmse_sd": float("nan"),
                "pearson_mean": float("nan"),
                "pearson_sd": float("nan"),
            },
        ]
    ).to_csv(trait_dir / "validation_ablation_summary.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "split_mode": "cv1_genotype",
                "repeats_attempted": 3,
                "repeats_passed": 1,
                "repeats_failed": 1,
                "repeats_skipped": 1,
                "repeats_skipped_empty": 1,
            }
        ]
    ).to_csv(trait_dir / "split_leakage_summary.tsv", sep="\t", index=False)

    report = combine_reports(tmp_path)

    assert report["ablation"].tolist() == ["G"]
    assert report.iloc[0]["folds_attempted"] == 3
    assert report.iloc[0]["folds_failed_leakage"] == 1
    assert report.iloc[0]["folds_skipped_empty"] == 1
