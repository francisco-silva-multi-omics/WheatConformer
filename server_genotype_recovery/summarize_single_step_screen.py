from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE = "pedigree_environment_only"


def architecture_name(candidate: str) -> str:
    return re.sub(r"_cfg[0-9a-f]{10}$", "", str(candidate))


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def unique_path(run_dir: Path, pattern: str) -> Path:
    paths = list(run_dir.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern!r} in {run_dir}; found {len(paths)}")
    return paths[0]


def load_runs(models_dir: Path, scenario: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(models_dir.glob(f"genomic_inner_{scenario}_outer*_*_inner*")):
        metadata_path = unique_path(run_dir, "*_run_metadata.json")
        macro_path = unique_path(run_dir, "*_macro_metrics.tsv")
        trait_path = unique_path(run_dir, "*_trait_metrics.tsv")
        prediction_path = unique_path(run_dir, "*_predictions.parquet")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("evaluation_stage") != "inner_selection":
            continue
        external = metadata.get("external_split", {})
        if external.get("scenario") != scenario:
            continue
        macro = pd.read_csv(macro_path, sep="\t")
        traits = pd.read_csv(trait_path, sep="\t")
        if macro["split"].astype(str).eq("test").any() or traits["split"].astype(str).eq("test").any():
            raise ValueError(f"Inner run exposes outer-test metrics: {run_dir}")
        model_label = str(metadata["model_label"])
        selected = macro[
            macro["split"].astype(str).eq("val")
            & macro["model"].astype(str).eq(model_label)
        ]
        selected_traits = traits[
            traits["split"].astype(str).eq("val")
            & traits["coverage_group"].astype(str).eq("all")
            & traits["model"].astype(str).eq(model_label)
        ]
        if len(selected) != 1 or selected_traits.empty:
            raise ValueError(f"Inner run has incomplete validation metrics: {run_dir}")
        candidate = str(metadata.get("hyperparameter_label", ""))
        ratios = pd.to_numeric(
            selected_traits["prediction_sd_ratio"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(ratios).all():
            raise ValueError(f"Inner run contains non-finite calibration values: {run_dir}")
        rows.append(
            {
                "run_dir": str(run_dir),
                "candidate": candidate,
                "architecture": architecture_name(candidate),
                "outer_fold": int(external["outer_fold"]),
                "inner_fold": int(external["inner_fold"]),
                "seed": int(metadata["seed"]),
                "training_configuration": json.dumps(
                    metadata["training_configuration"], sort_keys=True
                ),
                "model_label": model_label,
                "prediction_path": str(prediction_path),
                "val_normalized_rmse": float(
                    selected.iloc[0]["macro_normalized_rmse"]
                ),
                "val_pearson": float(selected.iloc[0]["macro_pearson"]),
                "val_calibration_error": float(
                    np.mean(np.abs(ratios - 1.0))
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_grid(
    runs: pd.DataFrame,
    architectures: set[str],
    expected_outer_folds: int,
    expected_inner_folds: int,
) -> None:
    if runs.empty:
        raise ValueError("No single-step inner-screen runs were found")
    keys = ["architecture", "outer_fold", "inner_fold"]
    if runs.duplicated(keys).any():
        raise ValueError("Single-step inner screen contains duplicate run keys")
    if set(runs["architecture"]) != architectures:
        raise ValueError(
            "Architecture mismatch: "
            f"expected={sorted(architectures)} observed={sorted(set(runs['architecture']))}"
        )
    for architecture, group in runs.groupby("architecture"):
        if set(group["outer_fold"]) != set(range(expected_outer_folds)):
            raise ValueError(f"Incomplete outer folds for {architecture}")
        for outer_fold, local in group.groupby("outer_fold"):
            if set(local["inner_fold"]) != set(range(expected_inner_folds)):
                raise ValueError(
                    f"Incomplete inner folds for {architecture} outer={outer_fold}"
                )
    for key, group in runs.groupby(["outer_fold", "inner_fold"]):
        if group["seed"].nunique() != 1:
            raise ValueError(f"Unmatched seeds at outer/inner={key}")
        if group["training_configuration"].nunique() != 1:
            raise ValueError(f"Unmatched training configuration at outer/inner={key}")


def paired_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    reference = runs[runs["architecture"].eq(REFERENCE)][
        [
            "outer_fold",
            "inner_fold",
            "val_normalized_rmse",
            "val_pearson",
            "val_calibration_error",
        ]
    ].rename(
        columns={
            "val_normalized_rmse": "reference_val_normalized_rmse",
            "val_pearson": "reference_val_pearson",
            "val_calibration_error": "reference_val_calibration_error",
        }
    )
    paired = runs.merge(
        reference, on=["outer_fold", "inner_fold"], validate="many_to_one"
    )
    paired["nrmse_gain_vs_reference"] = (
        paired["reference_val_normalized_rmse"] - paired["val_normalized_rmse"]
    )
    paired["relative_nrmse_gain_vs_reference"] = (
        paired["nrmse_gain_vs_reference"]
        / paired["reference_val_normalized_rmse"]
    )
    paired["pearson_gain_vs_reference"] = (
        paired["val_pearson"] - paired["reference_val_pearson"]
    )
    paired["calibration_error_delta_vs_reference"] = (
        paired["val_calibration_error"]
        - paired["reference_val_calibration_error"]
    )
    return paired


def load_id_set(path: Path) -> set[str]:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    id_col = next(
        (column for column in ["sample_id", "genotype_id", "panel_sample_id"] if column in frame),
        None,
    )
    if id_col is None:
        raise ValueError(f"Direct-genotype order lacks an ID column: {path}")
    ids = frame[id_col].fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError(f"Direct-genotype order has empty or duplicate IDs: {path}")
    return set(ids)


def prediction_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "genotype_id",
        "environment_id",
        "trait_name_canonical",
        "phenotype_value",
        "y_pred",
        "split",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Prediction file is missing columns {missing}: {path}")
    if not frame["split"].astype(str).eq("val").all():
        raise ValueError(f"Inner prediction file contains a non-validation split: {path}")
    return frame.reset_index(drop=True)


def assert_matched_predictions(candidate: pd.DataFrame, reference: pd.DataFrame) -> None:
    keys = ["genotype_id", "environment_id", "trait_name_canonical", "phenotype_value"]
    if len(candidate) != len(reference):
        raise ValueError("Matched runs predict different validation row counts")
    for column in keys:
        left = candidate[column].to_numpy()
        right = reference[column].to_numpy()
        if column == "phenotype_value":
            matched = np.array_equal(
                np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
            )
        else:
            matched = np.array_equal(left.astype(str), right.astype(str))
        if not matched:
            raise ValueError(f"Matched runs disagree on prediction key {column}")


def subset_macro(frame: pd.DataFrame, mask: np.ndarray) -> dict[str, float | int]:
    local = frame.loc[mask].copy()
    trait_rows = []
    for _, group in local.groupby("trait_name_canonical", sort=True):
        true = group["phenotype_value"].to_numpy(dtype=np.float64)
        pred = group["y_pred"].to_numpy(dtype=np.float64)
        if len(true) < 2 or np.std(true, ddof=1) <= 0:
            continue
        rmse = float(np.sqrt(np.mean(np.square(pred - true))))
        true_sd = float(np.std(true, ddof=1))
        pred_sd = float(np.std(pred, ddof=1))
        pearson = (
            float(np.corrcoef(true, pred)[0, 1])
            if pred_sd > 0
            else float("nan")
        )
        trait_rows.append(
            {
                "normalized_rmse": rmse / true_sd,
                "pearson": pearson,
                "calibration_error": abs(pred_sd / true_sd - 1.0),
            }
        )
    metrics = pd.DataFrame(trait_rows)
    if metrics.empty:
        return {
            "rows": len(local),
            "traits": 0,
            "normalized_rmse": float("nan"),
            "pearson": float("nan"),
            "calibration_error": float("nan"),
        }
    return {
        "rows": len(local),
        "traits": len(metrics),
        "normalized_rmse": float(metrics["normalized_rmse"].mean()),
        "pearson": float(metrics["pearson"].mean(skipna=True)),
        "calibration_error": float(metrics["calibration_error"].mean()),
    }


def coverage_metrics(
    runs: pd.DataFrame, plan: pd.DataFrame, root: Path
) -> pd.DataFrame:
    reference_lookup = {
        (int(row.outer_fold), int(row.inner_fold)): row
        for row in runs[runs["architecture"].eq(REFERENCE)].itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    candidates = plan[
        plan["architecture"].ne(REFERENCE)
        & plan["status"].eq("ready")
        & plan["screen_phase"].eq("phase_1_inner_validation")
    ]
    for spec in candidates.itertuples(index=False):
        order_path = resolve(root, Path(str(spec.direct_genotyped_order_path)))
        direct_ids = load_id_set(order_path)
        selected_runs = runs[runs["architecture"].eq(spec.architecture)]
        for run in selected_runs.itertuples(index=False):
            key = (int(run.outer_fold), int(run.inner_fold))
            reference_run = reference_lookup[key]
            candidate_predictions = prediction_frame(Path(run.prediction_path))
            reference_predictions = prediction_frame(Path(reference_run.prediction_path))
            assert_matched_predictions(candidate_predictions, reference_predictions)
            direct = candidate_predictions["genotype_id"].astype(str).isin(direct_ids).to_numpy()
            for group_name, mask in [
                ("directly_genotyped", direct),
                ("pedigree_only_propagated", ~direct),
            ]:
                candidate_metrics = subset_macro(candidate_predictions, mask)
                reference_metrics = subset_macro(reference_predictions, mask)
                rows.append(
                    {
                        "architecture": spec.architecture,
                        "outer_fold": run.outer_fold,
                        "inner_fold": run.inner_fold,
                        "coverage_group": group_name,
                        "rows": candidate_metrics["rows"],
                        "traits": candidate_metrics["traits"],
                        "candidate_normalized_rmse": candidate_metrics[
                            "normalized_rmse"
                        ],
                        "reference_normalized_rmse": reference_metrics[
                            "normalized_rmse"
                        ],
                        "nrmse_gain_vs_reference": reference_metrics[
                            "normalized_rmse"
                        ]
                        - candidate_metrics["normalized_rmse"],
                        "candidate_pearson": candidate_metrics["pearson"],
                        "reference_pearson": reference_metrics["pearson"],
                        "pearson_gain_vs_reference": candidate_metrics["pearson"]
                        - reference_metrics["pearson"],
                        "candidate_calibration_error": candidate_metrics[
                            "calibration_error"
                        ],
                        "reference_calibration_error": reference_metrics[
                            "calibration_error"
                        ],
                        "calibration_error_delta_vs_reference": candidate_metrics[
                            "calibration_error"
                        ]
                        - reference_metrics["calibration_error"],
                    }
                )
    output = pd.DataFrame(rows)
    metric_columns = [
        "candidate_normalized_rmse",
        "reference_normalized_rmse",
        "nrmse_gain_vs_reference",
        "candidate_pearson",
        "reference_pearson",
        "pearson_gain_vs_reference",
        "candidate_calibration_error",
        "reference_calibration_error",
        "calibration_error_delta_vs_reference",
    ]
    if output.empty or not np.isfinite(output[metric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Coverage-specific single-step metrics are empty or non-finite")
    return output


def summarize(
    paired: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    minimum_relative_gain: float,
    minimum_win_rate: float,
    maximum_pearson_drop: float,
) -> pd.DataFrame:
    summary = (
        paired.groupby("architecture")
        .agg(
            paired_inner_folds=("inner_fold", "size"),
            outer_folds=("outer_fold", "nunique"),
            val_normalized_rmse_mean=("val_normalized_rmse", "mean"),
            val_normalized_rmse_sd=("val_normalized_rmse", "std"),
            val_pearson_mean=("val_pearson", "mean"),
            val_pearson_sd=("val_pearson", "std"),
            relative_nrmse_gain_vs_reference_mean=(
                "relative_nrmse_gain_vs_reference",
                "mean",
            ),
            nrmse_win_rate_vs_reference=(
                "nrmse_gain_vs_reference",
                lambda values: float((values > 0).mean()),
            ),
            pearson_gain_vs_reference_mean=("pearson_gain_vs_reference", "mean"),
            calibration_error_delta_vs_reference_mean=(
                "calibration_error_delta_vs_reference",
                "mean",
            ),
        )
        .reset_index()
    )
    pedigree = (
        coverage[coverage["coverage_group"].eq("pedigree_only_propagated")]
        .groupby("architecture")
        .agg(
            pedigree_only_nrmse_gain_vs_reference_mean=(
                "nrmse_gain_vs_reference",
                "mean",
            ),
            pedigree_only_pearson_gain_vs_reference_mean=(
                "pearson_gain_vs_reference",
                "mean",
            ),
            pedigree_only_calibration_error_delta_vs_reference_mean=(
                "calibration_error_delta_vs_reference",
                "mean",
            ),
        )
        .reset_index()
    )
    summary = summary.merge(pedigree, on="architecture", how="left")
    is_reference = summary["architecture"].eq(REFERENCE)
    accepted = (
        summary["relative_nrmse_gain_vs_reference_mean"].ge(minimum_relative_gain)
        & summary["nrmse_win_rate_vs_reference"].ge(minimum_win_rate)
        & summary["pearson_gain_vs_reference_mean"].ge(-maximum_pearson_drop)
        & summary["calibration_error_delta_vs_reference_mean"].le(0.0)
        & summary["pedigree_only_nrmse_gain_vs_reference_mean"].ge(0.0)
        & summary["pedigree_only_pearson_gain_vs_reference_mean"].ge(
            -maximum_pearson_drop
        )
        & summary[
            "pedigree_only_calibration_error_delta_vs_reference_mean"
        ].le(0.0)
    )
    summary["single_step_H_decision"] = np.where(
        is_reference,
        "reference",
        np.where(accepted, "advance_to_frozen_architecture", "do_not_advance"),
    )
    return summary.sort_values(
        ["single_step_H_decision", "val_normalized_rmse_mean", "architecture"],
        kind="stable",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the single-step H screen using inner validation only."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scenario", default="unseen_genotypes")
    parser.add_argument("--expected-outer-folds", type=int, default=5)
    parser.add_argument("--expected-inner-folds", type=int, default=3)
    parser.add_argument("--minimum-relative-gain", type=float, default=0.01)
    parser.add_argument("--minimum-win-rate", type=float, default=2.0 / 3.0)
    parser.add_argument("--maximum-pearson-drop", type=float, default=0.005)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    models_dir = resolve(root, args.models_dir)
    plan_path = resolve(root, args.plan)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv(plan_path, sep="\t", dtype=str).fillna("")
    ready = plan[
        plan["status"].eq("ready")
        & plan["screen_phase"].eq("phase_1_inner_validation")
    ]
    architectures = set(ready["architecture"])
    if REFERENCE not in architectures or len(architectures) < 2:
        raise SystemExit(
            "The single-step screen must contain the pedigree reference and at least one H candidate"
        )
    runs = load_runs(models_dir, args.scenario)
    validate_grid(
        runs, architectures, args.expected_outer_folds, args.expected_inner_folds
    )
    paired = paired_metrics(runs)
    coverage = coverage_metrics(runs, ready, root)
    summary = summarize(
        paired,
        coverage,
        minimum_relative_gain=args.minimum_relative_gain,
        minimum_win_rate=args.minimum_win_rate,
        maximum_pearson_drop=args.maximum_pearson_drop,
    )
    runs.to_csv(out_dir / "single_step_inner_screen_runs.tsv", sep="\t", index=False)
    paired.to_csv(out_dir / "single_step_inner_screen_paired_metrics.tsv", sep="\t", index=False)
    coverage.to_csv(
        out_dir / "single_step_inner_screen_coverage_metrics.tsv", sep="\t", index=False
    )
    summary.to_csv(out_dir / "single_step_inner_screen_summary.tsv", sep="\t", index=False)
    provenance = {
        "status": "PASS",
        "selection_data": "inner_validation_metrics_only",
        "inner_validation_phenotype_values_read": True,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "scenario": args.scenario,
        "run_count": len(runs),
        "architecture_count": len(architectures),
        "outer_fold_count": args.expected_outer_folds,
        "inner_fold_count": args.expected_inner_folds,
        "matched_seed_status": "pass",
        "matched_training_configuration_status": "pass",
        "acceptance_thresholds": {
            "minimum_relative_nrmse_gain_vs_pedigree_reference": args.minimum_relative_gain,
            "minimum_inner_fold_win_rate_vs_pedigree_reference": args.minimum_win_rate,
            "maximum_mean_pearson_drop_vs_pedigree_reference": args.maximum_pearson_drop,
            "maximum_mean_calibration_error_increase": 0.0,
            "minimum_pedigree_only_nrmse_gain": 0.0,
            "maximum_pedigree_only_pearson_drop": args.maximum_pearson_drop,
            "maximum_pedigree_only_calibration_error_increase": 0.0,
        },
        "coverage_groups": ["directly_genotyped", "pedigree_only_propagated"],
        "platform_kernels_combined": False,
    }
    (out_dir / "single_step_inner_screen_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    print("\n=== SINGLE-STEP H DECISION ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
