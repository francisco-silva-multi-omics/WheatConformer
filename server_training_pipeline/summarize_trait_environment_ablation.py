from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Candidate:
    kernel: str
    trait: str
    variant: str


def parse_candidate(value: str) -> Candidate:
    pieces = [piece.strip() for piece in value.split(":")]
    if len(pieces) != 3 or not all(pieces):
        raise argparse.ArgumentTypeError(
            "Candidate must be KERNEL:TRAIT:VARIANT, for example "
            "K_E_DTM_V2:DAYS_TO_MATURITY:uniform_env_dtm_v2"
        )
    return Candidate(*pieces)


def completed_run(models_root: Path, variant: str, seed: int) -> tuple[pd.DataFrame, dict]:
    run_dir = models_root / f"multitrait_quantitative_{variant}_env_seed{seed}"
    metrics_paths = list(run_dir.glob("*_trait_metrics.tsv"))
    metadata_paths = list(run_dir.glob("*_run_metadata.json"))
    leakage_paths = list(run_dir.glob("*_split_leakage_qc.tsv"))
    if len(metrics_paths) != 1 or len(metadata_paths) != 1 or len(leakage_paths) != 1:
        raise SystemExit(f"Incomplete or ambiguous run artifacts: {run_dir}")
    leakage = pd.read_csv(leakage_paths[0], sep="\t")
    if leakage.empty or not leakage["leakage_status"].astype(str).str.lower().eq("pass").all():
        raise SystemExit(f"Split leakage certification is not PASS: {leakage_paths[0]}")
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    if int(metadata.get("seed", -1)) != seed:
        raise SystemExit(f"Seed metadata mismatch in {metadata_paths[0]}")
    return pd.read_csv(metrics_paths[0], sep="\t"), metadata


def trait_validation_row(metrics: pd.DataFrame, trait: str) -> pd.Series:
    selected = metrics[
        metrics["split"].eq("val")
        & metrics["coverage_group"].eq("all")
        & metrics["trait_name_canonical"].astype(str).str.upper().eq(trait.upper())
        & ~metrics["model"].eq("train_mean")
    ]
    if len(selected) != 1:
        raise SystemExit(
            f"Expected one validation/all/model row for {trait}; found {len(selected)}"
        )
    return selected.iloc[0]


def compare_candidate(
    *,
    models_root: Path,
    baseline_variant: str,
    candidate: Candidate,
    seeds: list[int],
    specific_kernels: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        baseline_metrics, baseline_metadata = completed_run(models_root, baseline_variant, seed)
        candidate_metrics, candidate_metadata = completed_run(models_root, candidate.variant, seed)
        baseline_active = set(baseline_metadata.get("active_kernels", []))
        candidate_active = set(candidate_metadata.get("active_kernels", []))
        if baseline_active & specific_kernels:
            raise SystemExit(
                f"Baseline {baseline_variant} seed {seed} contains trait-specific kernels: "
                f"{sorted(baseline_active & specific_kernels)}"
            )
        unexpected = (candidate_active & specific_kernels) - {candidate.kernel}
        if candidate.kernel not in candidate_active or unexpected:
            raise SystemExit(
                f"Candidate {candidate.variant} seed {seed} has invalid specific experts: "
                f"active={sorted(candidate_active & specific_kernels)}"
            )

        baseline = trait_validation_row(baseline_metrics, candidate.trait)
        proposed = trait_validation_row(candidate_metrics, candidate.trait)
        baseline_nrmse = float(baseline["normalized_rmse"])
        proposed_nrmse = float(proposed["normalized_rmse"])
        baseline_pearson = float(baseline["pearson"])
        proposed_pearson = float(proposed["pearson"])
        baseline_sd_ratio = float(baseline["prediction_sd_ratio"])
        proposed_sd_ratio = float(proposed["prediction_sd_ratio"])
        rows.append(
            {
                "kernel": candidate.kernel,
                "trait_name_canonical": candidate.trait,
                "baseline_variant": baseline_variant,
                "candidate_variant": candidate.variant,
                "seed": seed,
                "baseline_val_normalized_rmse": baseline_nrmse,
                "candidate_val_normalized_rmse": proposed_nrmse,
                "normalized_rmse_improvement": baseline_nrmse - proposed_nrmse,
                "relative_normalized_rmse_improvement": (
                    (baseline_nrmse - proposed_nrmse) / baseline_nrmse
                    if baseline_nrmse > 0
                    else np.nan
                ),
                "candidate_wins": proposed_nrmse < baseline_nrmse,
                "baseline_val_pearson": baseline_pearson,
                "candidate_val_pearson": proposed_pearson,
                "pearson_delta": proposed_pearson - baseline_pearson,
                "baseline_prediction_sd_ratio": baseline_sd_ratio,
                "candidate_prediction_sd_ratio": proposed_sd_ratio,
                "sd_calibration_error_delta": (
                    abs(proposed_sd_ratio - 1.0) - abs(baseline_sd_ratio - 1.0)
                ),
            }
        )
    detail = pd.DataFrame(rows)
    mean_absolute_gain = float(detail["normalized_rmse_improvement"].mean())
    mean_relative_gain = float(detail["relative_normalized_rmse_improvement"].mean())
    wins = int(detail["candidate_wins"].sum())
    mean_pearson_delta = float(detail["pearson_delta"].mean())
    mean_sd_calibration_delta = float(detail["sd_calibration_error_delta"].mean())
    accepted = bool(
        (mean_relative_gain >= 0.02 or mean_absolute_gain >= 0.02)
        and wins >= 3
        and mean_pearson_delta >= -0.02
        and mean_sd_calibration_delta <= 0.0
    )
    decision = {
        "kernel": candidate.kernel,
        "trait_name_canonical": candidate.trait,
        "baseline_variant": baseline_variant,
        "candidate_variant": candidate.variant,
        "seed_count": len(detail),
        "baseline_val_normalized_rmse_mean": float(
            detail["baseline_val_normalized_rmse"].mean()
        ),
        "candidate_val_normalized_rmse_mean": float(
            detail["candidate_val_normalized_rmse"].mean()
        ),
        "normalized_rmse_improvement_mean": mean_absolute_gain,
        "relative_normalized_rmse_improvement_mean": mean_relative_gain,
        "candidate_win_count": wins,
        "pearson_delta_mean": mean_pearson_delta,
        "sd_calibration_error_delta_mean": mean_sd_calibration_delta,
        "accepted": accepted,
        "decision": "accept_for_multitrait_baseline" if accepted else "retain_diagnostic_only",
    }
    return detail, decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Certify repeated-seed incremental trait-environment kernel ablations."
    )
    parser.add_argument("--models-root", type=Path, default=Path("trained_models"))
    parser.add_argument("--baseline-variant", default="uniform_env_generic")
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument(
        "--out-dir", type=Path, default=Path("trained_models/model_comparisons")
    )
    parser.add_argument("--out-prefix", default="trait_environment_kernel_ablation")
    args = parser.parse_args()

    seeds = args.seed or [2026, 2027, 2028, 2029]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("Seeds must be unique")
    if len(seeds) != 4:
        raise SystemExit("Certification requires exactly four repeated seeds")
    specific_kernels = {candidate.kernel for candidate in args.candidate}
    details = []
    decisions = []
    for candidate in args.candidate:
        detail, decision = compare_candidate(
            models_root=args.models_root,
            baseline_variant=args.baseline_variant,
            candidate=candidate,
            seeds=seeds,
            specific_kernels=specific_kernels,
        )
        details.append(detail)
        decisions.append(decision)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_frame = pd.concat(details, ignore_index=True)
    decision_frame = pd.DataFrame(decisions)
    detail_frame.to_csv(
        args.out_dir / f"{args.out_prefix}_seed_detail.tsv", sep="\t", index=False
    )
    decision_frame.to_csv(
        args.out_dir / f"{args.out_prefix}_decision.tsv", sep="\t", index=False
    )
    print(decision_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
