from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_ARCHITECTURES = {
    "pedigree_environment_only",
    "frozen_existing_HMP_GBS",
}


def architecture_name(candidate: str) -> str:
    return re.sub(r"_cfg[0-9a-f]{10}$", "", str(candidate))


def load_runs(models_dir: Path, scenario: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pattern = f"genomic_inner_{scenario}_outer*_*_inner*"
    for run_dir in sorted(models_dir.glob(pattern)):
        metadata_paths = list(run_dir.glob("*_run_metadata.json"))
        macro_paths = list(run_dir.glob("*_macro_metrics.tsv"))
        if len(metadata_paths) != 1 or len(macro_paths) != 1:
            continue
        metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
        if metadata.get("evaluation_stage") != "inner_selection":
            continue
        external = metadata.get("external_split", {})
        if external.get("scenario") != scenario:
            continue
        macro = pd.read_csv(macro_paths[0], sep="\t")
        if macro["split"].astype(str).eq("test").any():
            raise ValueError(f"Inner run exposes outer-test metrics: {run_dir}")
        selected = macro[
            macro["split"].astype(str).eq("val")
            & macro["model"].astype(str).eq(str(metadata["model_label"]))
        ]
        if len(selected) != 1:
            raise ValueError(f"Inner run has invalid validation macro metrics: {run_dir}")
        candidate = str(metadata.get("hyperparameter_label", ""))
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
                "val_normalized_rmse": float(
                    selected.iloc[0]["macro_normalized_rmse"]
                ),
                "val_pearson": float(selected.iloc[0]["macro_pearson"]),
            }
        )
    return pd.DataFrame(rows)


def validate_grid(
    runs: pd.DataFrame,
    expected_architectures: set[str],
    expected_outer_folds: int,
    expected_inner_folds: int,
) -> None:
    if runs.empty:
        raise ValueError("No inner genomic-screen runs were found")
    key = ["architecture", "outer_fold", "inner_fold"]
    if runs.duplicated(key).any():
        duplicates = runs.loc[runs.duplicated(key, keep=False), key]
        raise ValueError(f"Duplicate inner-screen keys:\n{duplicates.to_string(index=False)}")
    observed_architectures = set(runs["architecture"])
    if observed_architectures != expected_architectures:
        raise ValueError(
            "Architecture grid mismatch: "
            f"missing={sorted(expected_architectures - observed_architectures)} "
            f"extra={sorted(observed_architectures - expected_architectures)}"
        )
    expected_outer = set(range(expected_outer_folds))
    expected_inner = set(range(expected_inner_folds))
    for architecture, group in runs.groupby("architecture"):
        if set(group["outer_fold"]) != expected_outer:
            raise ValueError(f"Incomplete outer folds for {architecture}")
        for outer_fold, local in group.groupby("outer_fold"):
            if set(local["inner_fold"]) != expected_inner:
                raise ValueError(
                    f"Incomplete inner folds for {architecture} outer={outer_fold}"
                )
    for (outer_fold, inner_fold), group in runs.groupby(["outer_fold", "inner_fold"]):
        if group["seed"].nunique() != 1:
            raise ValueError(
                f"Unmatched seeds for outer={outer_fold} inner={inner_fold}"
            )
        if group["training_configuration"].nunique() != 1:
            raise ValueError(
                f"Unmatched training configurations for outer={outer_fold} inner={inner_fold}"
            )


def paired_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    key = ["outer_fold", "inner_fold"]
    references = runs[runs["architecture"].isin(REFERENCE_ARCHITECTURES)].copy()
    if set(references["architecture"]) != REFERENCE_ARCHITECTURES:
        raise ValueError("Both frozen reference architectures are required")
    wide = references.pivot(
        index=key,
        columns="architecture",
        values=["val_normalized_rmse", "val_pearson"],
    )
    wide.columns = [f"{metric}__{architecture}" for metric, architecture in wide.columns]
    wide = wide.reset_index()
    paired = runs.merge(wide, on=key, validate="many_to_one")
    frozen_rmse = paired["val_normalized_rmse__frozen_existing_HMP_GBS"]
    frozen_pearson = paired["val_pearson__frozen_existing_HMP_GBS"]
    pedigree_rmse = paired["val_normalized_rmse__pedigree_environment_only"]
    pedigree_pearson = paired["val_pearson__pedigree_environment_only"]
    pedigree_is_best = pedigree_rmse.le(frozen_rmse)
    paired["best_reference"] = np.where(
        pedigree_is_best, "pedigree_environment_only", "frozen_existing_HMP_GBS"
    )
    paired["best_reference_normalized_rmse"] = np.where(
        pedigree_is_best, pedigree_rmse, frozen_rmse
    )
    paired["best_reference_pearson"] = np.where(
        pedigree_is_best, pedigree_pearson, frozen_pearson
    )
    paired["nrmse_gain_vs_frozen"] = frozen_rmse - paired["val_normalized_rmse"]
    paired["pearson_gain_vs_frozen"] = paired["val_pearson"] - frozen_pearson
    paired["nrmse_gain_vs_best_reference"] = (
        paired["best_reference_normalized_rmse"] - paired["val_normalized_rmse"]
    )
    paired["relative_nrmse_gain_vs_best_reference"] = (
        paired["nrmse_gain_vs_best_reference"]
        / paired["best_reference_normalized_rmse"]
    )
    paired["pearson_gain_vs_best_reference"] = (
        paired["val_pearson"] - paired["best_reference_pearson"]
    )
    return paired


def summarize(
    paired: pd.DataFrame,
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
            nrmse_gain_vs_frozen_mean=("nrmse_gain_vs_frozen", "mean"),
            nrmse_win_rate_vs_frozen=("nrmse_gain_vs_frozen", lambda x: float((x > 0).mean())),
            relative_nrmse_gain_vs_best_reference_mean=(
                "relative_nrmse_gain_vs_best_reference",
                "mean",
            ),
            nrmse_win_rate_vs_best_reference=(
                "nrmse_gain_vs_best_reference",
                lambda x: float((x > 0).mean()),
            ),
            pearson_gain_vs_best_reference_mean=(
                "pearson_gain_vs_best_reference",
                "mean",
            ),
        )
        .reset_index()
    )
    is_reference = summary["architecture"].isin(REFERENCE_ARCHITECTURES)
    accepted = (
        summary["relative_nrmse_gain_vs_best_reference_mean"].ge(
            minimum_relative_gain
        )
        & summary["nrmse_win_rate_vs_best_reference"].ge(minimum_win_rate)
        & summary["pearson_gain_vs_best_reference_mean"].ge(-maximum_pearson_drop)
    )
    summary["quantitative_K_G_decision"] = np.where(
        is_reference,
        "reference",
        np.where(accepted, "advance_pending_coverage_audit", "do_not_advance"),
    )
    summary["regulatory_panel_retention"] = np.where(
        is_reference,
        "not_applicable_reference",
        "retain_for_marker_to_graph_and_K_z",
    )
    return summary.sort_values(
        ["quantitative_K_G_decision", "val_normalized_rmse_mean", "architecture"],
        kind="stable",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate genomic architecture screens using inner validation only."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--models-dir", type=Path, default=Path("trained_models/genomic_expert_inner_screen_v1_runs")
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("model_kernels/genomic_candidate_screen_v1/genomic_candidate_ablation_plan.tsv"),
    )
    parser.add_argument("--expected-outer-folds", type=int, default=5)
    parser.add_argument("--expected-inner-folds", type=int, default=3)
    parser.add_argument("--minimum-relative-gain", type=float, default=0.01)
    parser.add_argument("--minimum-win-rate", type=float, default=2.0 / 3.0)
    parser.add_argument("--maximum-pearson-drop", type=float, default=0.005)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("model_kernels/genomic_expert_inner_screen_v1/summary"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    models_dir = args.models_dir if args.models_dir.is_absolute() else root / args.models_dir
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir = out_dir / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv(plan_path, sep="\t", dtype=str).fillna("")
    ready = plan[
        plan["screen_phase"].eq("phase_1_inner_validation")
        & plan["status"].eq("ready")
    ]
    expected_architectures = set(ready["architecture"])
    runs = load_runs(models_dir, args.scenario)
    validate_grid(
        runs,
        expected_architectures,
        args.expected_outer_folds,
        args.expected_inner_folds,
    )
    paired = paired_metrics(runs)
    summary = summarize(
        paired,
        minimum_relative_gain=args.minimum_relative_gain,
        minimum_win_rate=args.minimum_win_rate,
        maximum_pearson_drop=args.maximum_pearson_drop,
    )
    paired.to_csv(out_dir / "genomic_inner_screen_paired_metrics.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "genomic_inner_screen_summary.tsv", sep="\t", index=False)
    provenance = {
        "status": "PASS",
        "selection_data": "inner_validation_metrics_only",
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "scenario": args.scenario,
        "run_count": len(runs),
        "architecture_count": len(expected_architectures),
        "outer_fold_count": args.expected_outer_folds,
        "inner_fold_count": args.expected_inner_folds,
        "matched_seed_status": "pass",
        "matched_training_configuration_status": "pass",
        "acceptance_thresholds": {
            "minimum_relative_nrmse_gain_vs_best_reference": args.minimum_relative_gain,
            "minimum_inner_fold_win_rate_vs_best_reference": args.minimum_win_rate,
            "maximum_mean_pearson_drop_vs_best_reference": args.maximum_pearson_drop,
            "candidate_coverage_audit_required_before_phase_2": True,
        },
        "regulatory_panel_retention_independent_of_quantitative_K_G_decision": True,
    }
    (out_dir / "genomic_inner_screen_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
