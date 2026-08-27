from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE = "historical_reaction_reference"
MASK_CANDIDATE = "marker_supported_output_routed_v2"
SOURCE_SUMMARY = Path("model_kernels/stage1_v2_phase6_remediation_v1/phase_1")
SOURCE_RUNS = Path("trained_models/stage1_v2_phase6_remediation_v1_runs/phase_1")
OUTPUT = Path("audit/v2/stage1_v2_phase6_information_guard_replay_v1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Certify the reporting-only Stage-1 v2 information-mask replay"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-summary", type=Path, default=SOURCE_SUMMARY)
    parser.add_argument("--source-runs", type=Path, default=SOURCE_RUNS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    summary_root = (
        args.source_summary
        if args.source_summary.is_absolute()
        else root / args.source_summary
    ).resolve()
    runs_root = (
        args.source_runs if args.source_runs.is_absolute() else root / args.source_runs
    ).resolve()
    output = (args.output if args.output.is_absolute() else root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    grid_path = summary_root / "remediation_phase1_run_grid.tsv"
    decision_path = summary_root / "PHASE1_STRUCTURAL_DECISION.json"
    source_decision_table = summary_root / "remediation_phase1_decision.tsv"
    required = [grid_path, decision_path, source_decision_table]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Information replay inputs are missing: {missing}")
    grid = read_tsv(grid_path)
    source_decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if source_decision.get("status") != "PASS_STAGE1_V2_PHASE6_REMEDIATION_PHASE1_COMPLETE":
        raise ValueError("Source remediation is not complete")
    if source_decision.get("advanced_candidates"):
        raise ValueError("Source remediation unexpectedly advanced a candidate")
    if source_decision.get("phase2_optimizer_allowed") is not False:
        raise ValueError("Source remediation unexpectedly allowed Phase 2")

    records: list[dict[str, object]] = []
    run_artifacts: list[Path] = []
    for state_id, state_grid in grid.groupby("state_id", sort=False):
        scenario = str(state_grid["scenario"].iloc[0])
        reference_path = runs_root / state_id / REFERENCE / "validation_guard_metrics.tsv"
        reference = read_tsv(reference_path)
        run_artifacts.append(reference_path)
        reference = reference.loc[reference["mask_candidate"].eq(MASK_CANDIDATE)].set_index(
            "subset"
        )
        if reference.empty:
            raise ValueError(f"Reference lacks frozen marker mask: {state_id}")
        for candidate in state_grid["candidate"].astype(str).unique():
            path = runs_root / state_id / candidate / "validation_guard_metrics.tsv"
            observed = read_tsv(path)
            run_artifacts.append(path)
            observed = observed.loc[
                observed["mask_candidate"].eq(MASK_CANDIDATE)
            ].set_index("subset")
            if set(observed.index) != set(reference.index):
                raise ValueError(f"Guard subset mismatch: {state_id}/{candidate}")
            for subset in sorted(reference.index):
                ref = reference.loc[subset]
                current = observed.loc[subset]
                rows = int(current["rows"])
                if rows != int(ref["rows"]):
                    raise ValueError(f"Guard row mismatch: {state_id}/{candidate}/{subset}")
                if str(current["observation_id_signature"]) != str(
                    ref["observation_id_signature"]
                ):
                    raise ValueError(
                        f"Guard signature mismatch: {state_id}/{candidate}/{subset}"
                    )
                candidate_nrmse = float(current["normalized_rmse_macro"])
                reference_nrmse = float(ref["normalized_rmse_macro"])
                candidate_pearson = float(current["pearson_macro"])
                reference_pearson = float(ref["pearson_macro"])
                records.append(
                    {
                        "state_id": state_id,
                        "scenario": scenario,
                        "candidate": candidate,
                        "mask_candidate": MASK_CANDIDATE,
                        "subset": subset,
                        "rows": rows,
                        "unique_genotypes": int(current["unique_genotypes"]),
                        "unique_environments": int(current["unique_environments"]),
                        "trait_count": int(current["trait_count"]),
                        "observation_id_signature": current[
                            "observation_id_signature"
                        ],
                        "candidate_nrmse": candidate_nrmse,
                        "reference_nrmse": reference_nrmse,
                        "relative_nrmse_gain": (
                            (reference_nrmse - candidate_nrmse) / reference_nrmse
                            if rows and np.isfinite(reference_nrmse)
                            else np.nan
                        ),
                        "candidate_pearson": candidate_pearson,
                        "reference_pearson": reference_pearson,
                        "pearson_gain": candidate_pearson - reference_pearson,
                    }
                )
    paired = pd.DataFrame(records)
    summary = (
        paired.loc[paired["rows"].gt(0)]
        .groupby(["scenario", "candidate", "subset"], as_index=False)
        .agg(
            paired_states=("state_id", "nunique"),
            rows_mean=("rows", "mean"),
            relative_nrmse_gain_mean=("relative_nrmse_gain", "mean"),
            relative_nrmse_gain_min=("relative_nrmse_gain", "min"),
            nrmse_win_rate=(
                "relative_nrmse_gain", lambda values: float((values > 0).mean())
            ),
            pearson_gain_mean=("pearson_gain", "mean"),
        )
    )
    paired_path = output / "information_guard_replay_paired.tsv"
    summary_path = output / "information_guard_replay_summary.tsv"
    paired.to_csv(paired_path, sep="\t", index=False, lineterminator="\n")
    summary.to_csv(summary_path, sep="\t", index=False, lineterminator="\n")
    checks = {
        "source_remediation_complete": True,
        "source_advanced_candidates_empty": True,
        "source_phase2_blocked": True,
        "all_candidate_predictions_use_same_marker_mask": True,
        "all_nonempty_rows_match": True,
        "all_nonempty_observation_signatures_match": True,
        "formal_source_decision_changed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
    }
    decision = {
        "status": "PASS_INFORMATION_GUARD_REPORTING_REPLAY_FROZEN",
        "protocol_version": "stage1_v2_phase6_information_guard_replay_v1",
        "stage1_version": "Stage-1 v2",
        "selection_data": "existing_inner_validation_metrics_reporting_only",
        "mask_candidate": MASK_CANDIDATE,
        "state_count": int(grid["state_id"].nunique()),
        "run_count": int(len(grid)),
        "formal_source_decision_changed": False,
        "advanced_candidates": [],
        "phase2_optimizer_allowed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "checks": checks,
        "artifacts": {
            "source_grid_sha256": sha256_file(grid_path),
            "source_decision_sha256": sha256_file(decision_path),
            "source_decision_table_sha256": sha256_file(source_decision_table),
            "source_guard_artifact_count": len(set(run_artifacts)),
            "information_guard_replay_paired_sha256": sha256_file(paired_path),
            "information_guard_replay_summary_sha256": sha256_file(summary_path),
        },
    }
    write_json(output / "INFORMATION_GUARD_REPLAY_DECISION.json", decision)
    pd.DataFrame(
        [
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": ""}
            for name, passed in checks.items()
        ]
    ).to_csv(output / "validation_checks.tsv", sep="\t", index=False, lineterminator="\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
