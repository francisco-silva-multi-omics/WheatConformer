from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .final_evaluation_contract import file_sha256
from .summarize_reaction_norm_screen import (
    assert_prediction_match,
    content_hash,
    load_run,
)


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a reaction-norm environment architecture from inner validation only."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--scenario", default="unseen_genotypes")
    parser.add_argument("--expected-outer-folds", type=int, default=5)
    parser.add_argument("--expected-inner-folds", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    models_dir = resolve(root, args.models_dir)
    protocol_path = resolve(root, args.environment_protocol)
    out_dir = resolve(root, args.out_dir)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_inner_validation":
        raise SystemExit("Environment architecture protocol is not frozen")
    candidates = {str(value["name"]): value for value in protocol["candidates"]}
    baseline = "current_corrected_generic_environment"
    challenger = "explicit_E_REACTION_NORM_V1"
    if set(candidates) != {baseline, challenger}:
        raise SystemExit("Environment screen must contain exactly the frozen two-arm comparison")

    rows: list[dict[str, object]] = []
    trait_tables: dict[tuple[str, int, int], pd.DataFrame] = {}
    pattern = f"reaction_environment_inner_{args.scenario}_outer*_*_inner*"
    for run_dir in sorted(models_dir.glob(pattern)):
        metadata_paths = list(run_dir.glob("*_run_metadata.json"))
        if len(metadata_paths) != 1:
            continue
        metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
        architecture = str(metadata.get("environment_architecture", ""))
        if architecture not in candidates:
            continue
        row, traits = load_run(run_dir, architecture=architecture)
        rows.append(row)
        trait_tables[(architecture, int(row["outer_fold"]), int(row["inner_fold"]))] = traits
    runs = pd.DataFrame(rows)
    expected_keys = {
        (architecture, outer, inner)
        for architecture in candidates
        for outer in range(args.expected_outer_folds)
        for inner in range(args.expected_inner_folds)
    }
    observed_keys = set(
        runs[["architecture", "outer_fold", "inner_fold"]].itertuples(
            index=False, name=None
        )
    ) if not runs.empty else set()
    if observed_keys != expected_keys:
        raise SystemExit(
            "Environment screen grid is incomplete: "
            f"missing={sorted(expected_keys-observed_keys)}; extra={sorted(observed_keys-expected_keys)}"
        )
    if runs.duplicated(["architecture", "outer_fold", "inner_fold"]).any():
        raise SystemExit("Environment screen contains duplicate run keys")

    lookup = runs.set_index(["architecture", "outer_fold", "inner_fold"])
    paired_rows = []
    trait_rows = []
    common_kernels = set(protocol["required_common_kernels"])
    for outer in range(args.expected_outer_folds):
        for inner in range(args.expected_inner_folds):
            reference = lookup.loc[(baseline, outer, inner)]
            candidate = lookup.loc[(challenger, outer, inner)]
            reference_metadata = reference["metadata"]
            candidate_metadata = candidate["metadata"]
            for field in ("seed", "manifest_sha256", "protocol_sha256"):
                if reference[field] != candidate[field]:
                    raise SystemExit(f"Matched environment runs disagree on {field}")
            if reference_metadata.get("training_configuration") != candidate_metadata.get(
                "training_configuration"
            ):
                raise SystemExit("Matched environment runs use different training configurations")
            if set(json.loads(reference["active_kernels"])) != common_kernels:
                raise SystemExit("Current corrected comparator violates its kernel contract")
            expected_candidate = set(candidates[challenger]["required_kernels"])
            if set(json.loads(candidate["active_kernels"])) != expected_candidate:
                raise SystemExit("Explicit environment candidate violates its kernel contract")
            for section in ("kernels", "orders"):
                for kernel in common_kernels:
                    left = content_hash(reference_metadata, section, kernel)
                    right = content_hash(candidate_metadata, section, kernel)
                    if not left or left != right:
                        raise SystemExit(f"Common {section} identity mismatch for {kernel}")
            assert_prediction_match(
                str(candidate["prediction_path"]), str(reference["prediction_path"])
            )
            gain = float(reference["val_normalized_rmse"] - candidate["val_normalized_rmse"])
            paired_rows.append(
                {
                    "outer_fold": outer,
                    "inner_fold": inner,
                    "seed": int(candidate["seed"]),
                    "reference_normalized_rmse": float(reference["val_normalized_rmse"]),
                    "candidate_normalized_rmse": float(candidate["val_normalized_rmse"]),
                    "normalized_rmse_gain": gain,
                    "relative_normalized_rmse_gain": gain
                    / float(reference["val_normalized_rmse"]),
                    "reference_pearson": float(reference["val_pearson"]),
                    "candidate_pearson": float(candidate["val_pearson"]),
                    "pearson_gain": float(candidate["val_pearson"] - reference["val_pearson"]),
                    "reference_calibration_error": float(reference["val_calibration_error"]),
                    "candidate_calibration_error": float(candidate["val_calibration_error"]),
                    "calibration_error_delta": float(
                        candidate["val_calibration_error"] - reference["val_calibration_error"]
                    ),
                }
            )
            reference_traits = trait_tables[(baseline, outer, inner)].set_index(
                "trait_name_canonical"
            )
            candidate_traits = trait_tables[(challenger, outer, inner)].set_index(
                "trait_name_canonical"
            )
            if set(reference_traits.index) != set(candidate_traits.index):
                raise SystemExit("Matched environment runs report different traits")
            for trait in sorted(reference_traits.index):
                left = reference_traits.loc[trait]
                right = candidate_traits.loc[trait]
                trait_rows.append(
                    {
                        "outer_fold": outer,
                        "inner_fold": inner,
                        "trait_name_canonical": trait,
                        "reference_normalized_rmse": float(left["normalized_rmse"]),
                        "candidate_normalized_rmse": float(right["normalized_rmse"]),
                        "normalized_rmse_gain": float(
                            left["normalized_rmse"] - right["normalized_rmse"]
                        ),
                        "relative_normalized_rmse_gain": float(
                            (left["normalized_rmse"] - right["normalized_rmse"])
                            / left["normalized_rmse"]
                        ),
                        "reference_pearson": float(left["pearson"]),
                        "candidate_pearson": float(right["pearson"]),
                        "pearson_gain": float(right["pearson"] - left["pearson"]),
                        "reference_prediction_sd_ratio": float(left["prediction_sd_ratio"]),
                        "candidate_prediction_sd_ratio": float(right["prediction_sd_ratio"]),
                    }
                )

    paired = pd.DataFrame(paired_rows)
    trait_paired = pd.DataFrame(trait_rows)
    selection = protocol["selection"]
    relative_gain = float(paired["relative_normalized_rmse_gain"].mean())
    win_rate = float(paired["normalized_rmse_gain"].gt(0).mean())
    pearson_gain = float(paired["pearson_gain"].mean())
    calibration_delta = float(paired["calibration_error_delta"].mean())
    exploratory_traits = set(selection.get("exploratory_traits", []))
    primary_trait_summary = (
        trait_paired[~trait_paired["trait_name_canonical"].isin(exploratory_traits)]
        .groupby("trait_name_canonical", as_index=False)
        .agg(
            relative_normalized_rmse_gain=("relative_normalized_rmse_gain", "mean"),
            pearson_gain=("pearson_gain", "mean"),
        )
    )
    primary_trait_guard = bool(
        primary_trait_summary["relative_normalized_rmse_gain"].ge(
            -float(selection["maximum_primary_trait_relative_nrmse_deterioration"])
        ).all()
        and primary_trait_summary["pearson_gain"].ge(
            -float(selection["maximum_primary_trait_pearson_drop"])
        ).all()
    )
    accepted = (
        len(paired) == args.expected_outer_folds * args.expected_inner_folds
        and relative_gain >= float(selection["minimum_relative_nrmse_gain"])
        and win_rate >= float(selection["minimum_fold_win_rate"])
        and pearson_gain >= -float(selection["maximum_mean_pearson_drop"])
        and calibration_delta <= float(selection["maximum_mean_calibration_error_increase"])
        and primary_trait_guard
    )
    selected = challenger if accepted else baseline
    summary = pd.DataFrame(
        [
            {
                "candidate": challenger,
                "paired_inner_folds": len(paired),
                "validation_normalized_rmse_reference_mean": float(
                    paired["reference_normalized_rmse"].mean()
                ),
                "validation_normalized_rmse_candidate_mean": float(
                    paired["candidate_normalized_rmse"].mean()
                ),
                "relative_normalized_rmse_gain_mean": relative_gain,
                "normalized_rmse_win_rate": win_rate,
                "pearson_gain_mean": pearson_gain,
                "calibration_error_delta_mean": calibration_delta,
                "primary_trait_guard_pass": primary_trait_guard,
                "accepted": accepted,
                "selected_environment_architecture": selected,
            }
        ]
    )
    trait_summary = (
        trait_paired.groupby("trait_name_canonical", as_index=False)
        .agg(
            paired_inner_folds=("inner_fold", "size"),
            normalized_rmse_gain_mean=("normalized_rmse_gain", "mean"),
            relative_normalized_rmse_gain_mean=("relative_normalized_rmse_gain", "mean"),
            normalized_rmse_win_rate=("normalized_rmse_gain", lambda values: float((values > 0).mean())),
            pearson_gain_mean=("pearson_gain", "mean"),
            candidate_prediction_sd_ratio_mean=("candidate_prediction_sd_ratio", "mean"),
            reference_prediction_sd_ratio_mean=("reference_prediction_sd_ratio", "mean"),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    export_runs = runs.drop(columns=["metadata"], errors="ignore")
    export_runs.to_csv(out_dir / "reaction_norm_environment_screen_runs.tsv", sep="\t", index=False)
    paired.to_csv(out_dir / "reaction_norm_environment_screen_paired_metrics.tsv", sep="\t", index=False)
    trait_paired.to_csv(
        out_dir / "reaction_norm_environment_screen_trait_paired_metrics.tsv",
        sep="\t",
        index=False,
    )
    trait_summary.to_csv(
        out_dir / "reaction_norm_environment_screen_trait_summary.tsv",
        sep="\t",
        index=False,
    )
    summary.to_csv(out_dir / "reaction_norm_environment_screen_summary.tsv", sep="\t", index=False)
    freeze = {
        "status": "PASS",
        "selection_data": "inner_validation_only",
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "environment_protocol_sha256": file_sha256(protocol_path),
        "selected_environment_architecture": selected,
        "explicit_environment_architecture_accepted": accepted,
        "paired_inner_fold_count": len(paired),
        "relative_normalized_rmse_gain_mean": relative_gain,
        "normalized_rmse_win_rate": win_rate,
        "pearson_gain_mean": pearson_gain,
        "calibration_error_delta_mean": calibration_delta,
        "primary_trait_guard_pass": primary_trait_guard,
        "outer_evaluation_allowed": False,
        "outer_evaluation_block_reason": (
            "Generate a versioned outer protocol bound to this selection lock before running outer fits"
        ),
    }
    (out_dir / "selected_reaction_norm_environment_architecture.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8"
    )
    print(json.dumps(freeze, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
