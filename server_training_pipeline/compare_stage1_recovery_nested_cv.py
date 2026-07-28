from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .final_evaluation_contract import file_sha256


def read_table(path: Path) -> pd.DataFrame:
    if "".join(path.suffixes).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def run_artifacts(run_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    metadata_paths = list(run_dir.glob("*_run_metadata.json"))
    prediction_paths = list(run_dir.glob("*_predictions.parquet"))
    if not prediction_paths:
        prediction_paths = list(run_dir.glob("*_predictions.tsv.gz"))
    if len(metadata_paths) != 1 or len(prediction_paths) != 1:
        raise ValueError(f"Incomplete outer ensemble: {run_dir}")
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    if metadata.get("evaluation_stage") != "outer_evaluation":
        raise ValueError(f"Run is not an outer ensemble: {run_dir}")
    predictions = read_table(prediction_paths[0])
    predictions = predictions[predictions["split"].eq("test")].copy()
    if predictions["canonical_observation_id"].duplicated().any():
        raise ValueError(f"Duplicate outer-test observation IDs: {run_dir}")
    return metadata, predictions


def inventory(root: Path) -> dict[tuple[str, int], Path]:
    result: dict[tuple[str, int], Path] = {}
    for run_dir in sorted(root.glob("final_nested_reaction_norm_*_outer*")):
        metadata, _ = run_artifacts(run_dir)
        external = metadata.get("external_split", {})
        key = (str(external.get("scenario")), int(external.get("outer_fold")))
        if key in result:
            raise ValueError(f"Duplicate outer ensemble key: {key}")
        result[key] = run_dir
    return result


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - prediction))))


def pearson(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) < 2 or np.std(y) <= 0 or np.std(prediction) <= 0:
        return float("nan")
    return float(np.corrcoef(y, prediction)[0, 1])


def metrics(frame: pd.DataFrame, prediction_column: str = "y_pred") -> dict[str, float]:
    y = pd.to_numeric(frame["phenotype_value"], errors="raise").to_numpy(dtype=float)
    prediction = pd.to_numeric(
        frame[prediction_column], errors="raise"
    ).to_numpy(dtype=float)
    true_sd = float(np.std(y))
    prediction_sd_ratio = float(np.std(prediction) / true_sd) if true_sd > 0 else float("nan")
    return {
        "rmse": rmse(y, prediction),
        "normalized_rmse": rmse(y, prediction) / true_sd if true_sd > 0 else float("nan"),
        "pearson": pearson(y, prediction),
        "prediction_sd_ratio": prediction_sd_ratio,
        "calibration_error": abs(1.0 - prediction_sd_ratio),
    }


def aggregate(values: pd.Series) -> tuple[int, float, float, float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    count = len(numeric)
    mean = float(numeric.mean()) if count else float("nan")
    sd = float(numeric.std(ddof=1)) if count > 1 else float("nan")
    half = 1.96 * sd / math.sqrt(count) if count > 1 else float("nan")
    return count, mean, sd, mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report paired common-support nested-CV changes after phenotype-blind "
            "Stage-1 recovery. Outer outcomes are reporting-only."
        )
    )
    parser.add_argument("--baseline-models-dir", type=Path, required=True)
    parser.add_argument("--recovery-models-dir", type=Path, required=True)
    parser.add_argument("--baseline-contract", type=Path, required=True)
    parser.add_argument("--recovery-contract", type=Path, required=True)
    parser.add_argument("--recovery-freeze", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    required_paths = [
        args.baseline_contract,
        args.recovery_contract,
        args.recovery_freeze,
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    freeze = json.loads(args.recovery_freeze.read_text(encoding="utf-8"))
    if freeze.get("status") != "PASS":
        raise ValueError("Stage-1 recovery was not frozen before outer evaluation")
    if freeze.get("outer_test_metrics_read") is not False:
        raise ValueError("Stage-1 recovery freeze already read outer metrics")
    baseline_contract = json.loads(args.baseline_contract.read_text(encoding="utf-8"))
    recovery_contract = json.loads(args.recovery_contract.read_text(encoding="utf-8"))
    for field in ["scenario_assignment_id", "final_holdout_assignment_id"]:
        if baseline_contract.get(field) != recovery_contract.get(field):
            raise ValueError(f"Nested contracts disagree on {field}")

    baseline_runs = inventory(args.baseline_models_dir)
    recovery_runs = inventory(args.recovery_models_dir)
    if set(baseline_runs) != set(recovery_runs):
        raise ValueError(
            "Baseline and recovery outer grids differ: "
            f"missing={sorted(set(baseline_runs)-set(recovery_runs))}; "
            f"extra={sorted(set(recovery_runs)-set(baseline_runs))}"
        )
    rows = []
    coverage_rows = []
    lineage_rows = []
    for scenario, outer_fold in sorted(baseline_runs):
        baseline_dir = baseline_runs[(scenario, outer_fold)]
        recovery_dir = recovery_runs[(scenario, outer_fold)]
        baseline_metadata, baseline = run_artifacts(baseline_dir)
        recovery_metadata, recovery = run_artifacts(recovery_dir)
        baseline_ids = set(baseline["canonical_observation_id"])
        recovery_ids = set(recovery["canonical_observation_id"])
        if not baseline_ids.issubset(recovery_ids):
            raise ValueError(
                "Recovered evaluation lost baseline test observations: "
                f"scenario={scenario} outer={outer_fold} "
                f"missing={len(baseline_ids-recovery_ids)}"
            )
        common = sorted(baseline_ids & recovery_ids)
        baseline_common = baseline.set_index("canonical_observation_id").loc[common]
        recovery_common = recovery.set_index("canonical_observation_id").loc[common]
        identity_columns = [
            "phenotype_value",
            "trait_name_canonical",
            "env_kernel_id",
        ]
        genotype_identity_column = (
            "genotype_id"
            if "genotype_id" in baseline_common
            and "genotype_id" in recovery_common
            else "panel_sample_id"
        )
        identity_columns.append(genotype_identity_column)
        for column in identity_columns:
            left = baseline_common[column].fillna("").astype(str)
            right = recovery_common[column].fillna("").astype(str)
            if not left.equals(right):
                raise ValueError(
                    f"Common-support identity mismatch in {column}: "
                    f"scenario={scenario} outer={outer_fold}"
                )
        for trait, trait_ids in baseline_common.groupby(
            "trait_name_canonical", sort=True
        ).groups.items():
            left = baseline_common.loc[trait_ids]
            right = recovery_common.loc[trait_ids]
            left_metrics = metrics(left)
            right_metrics = metrics(right)
            rows.append(
                {
                    "scenario": scenario,
                    "outer_fold": outer_fold,
                    "trait_name_canonical": trait,
                    "common_test_rows": len(left),
                    **{
                        f"baseline_{name}": value
                        for name, value in left_metrics.items()
                    },
                    **{
                        f"recovery_{name}": value
                        for name, value in right_metrics.items()
                    },
                    "normalized_rmse_gain": left_metrics["normalized_rmse"]
                    - right_metrics["normalized_rmse"],
                    "relative_normalized_rmse_gain": (
                        (left_metrics["normalized_rmse"] - right_metrics["normalized_rmse"])
                        / left_metrics["normalized_rmse"]
                        if left_metrics["normalized_rmse"] > 0
                        else float("nan")
                    ),
                    "pearson_gain": right_metrics["pearson"]
                    - left_metrics["pearson"],
                    "calibration_error_delta": right_metrics["calibration_error"]
                    - left_metrics["calibration_error"],
                }
            )
        recovery_only = recovery[
            ~recovery["canonical_observation_id"].isin(baseline_ids)
        ]
        for trait in sorted(
            set(recovery["trait_name_canonical"]).union(
                set(baseline["trait_name_canonical"])
            )
        ):
            baseline_trait = baseline["trait_name_canonical"].eq(trait)
            recovery_trait = recovery["trait_name_canonical"].eq(trait)
            recovery_only_trait = recovery_only["trait_name_canonical"].eq(trait)
            coverage_rows.append(
                {
                    "scenario": scenario,
                    "outer_fold": outer_fold,
                    "trait_name_canonical": trait,
                    "baseline_test_rows": int(baseline_trait.sum()),
                    "recovery_test_rows": int(recovery_trait.sum()),
                    "common_test_rows": int(
                        baseline_common["trait_name_canonical"].eq(trait).sum()
                    ),
                    "recovery_only_test_rows": int(recovery_only_trait.sum()),
                }
            )
        lineage_rows.append(
            {
                "scenario": scenario,
                "outer_fold": outer_fold,
                "baseline_run_dir": str(baseline_dir.resolve()),
                "recovery_run_dir": str(recovery_dir.resolve()),
                "baseline_model_label": baseline_metadata.get("model_label"),
                "recovery_model_label": recovery_metadata.get("model_label"),
            }
        )

    paired = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage_rows)
    summary_rows = []
    metrics_to_aggregate = [
        "baseline_normalized_rmse",
        "recovery_normalized_rmse",
        "normalized_rmse_gain",
        "relative_normalized_rmse_gain",
        "baseline_pearson",
        "recovery_pearson",
        "pearson_gain",
        "calibration_error_delta",
    ]
    for key, group in paired.groupby(
        ["scenario", "trait_name_canonical"], sort=True
    ):
        row: dict[str, object] = {
            "scenario": key[0],
            "trait_name_canonical": key[1],
            "common_test_rows_sum": int(group["common_test_rows"].sum()),
        }
        for metric_name in metrics_to_aggregate:
            count, mean, sd, low, high = aggregate(group[metric_name])
            row[f"{metric_name}_fold_count"] = count
            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_sd"] = sd
            row[f"{metric_name}_ci95_low"] = low
            row[f"{metric_name}_ci95_high"] = high
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paired_path = args.out_dir / "stage1_recovery_nested_paired_metrics.tsv"
    summary_path = args.out_dir / "stage1_recovery_nested_summary.tsv"
    coverage_path = args.out_dir / "stage1_recovery_nested_coverage.tsv"
    lineage_path = args.out_dir / "stage1_recovery_nested_lineage.tsv"
    paired.to_csv(paired_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    coverage.to_csv(coverage_path, sep="\t", index=False)
    pd.DataFrame(lineage_rows).to_csv(lineage_path, sep="\t", index=False)
    provenance = {
        "status": "PASS",
        "protocol_version": "stage1_recovery_nested_comparison_v1",
        "selection_data": "none_reporting_only",
        "outer_test_metrics_read_for_locked_reporting": True,
        "outer_test_metrics_used_for_selection": False,
        "final_holdout_outcomes_read": False,
        "data_recovery_reversal_or_selection_performed": False,
        "comparison_basis": "paired_common_outer_test_observations",
        "outer_ensemble_count": len(baseline_runs),
        "paired_trait_fold_rows": len(paired),
        "baseline_contract_sha256": file_sha256(args.baseline_contract),
        "recovery_contract_sha256": file_sha256(args.recovery_contract),
        "recovery_freeze_sha256": file_sha256(args.recovery_freeze),
        "artifacts": {
            path.name: file_sha256(path)
            for path in [paired_path, summary_path, coverage_path, lineage_path]
        },
    }
    provenance_path = args.out_dir / "stage1_recovery_nested_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
