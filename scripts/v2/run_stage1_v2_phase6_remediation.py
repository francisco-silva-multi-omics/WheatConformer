from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.v2.run_stage1_v2_phase6_phase1 import validate_runtime
from server_training_pipeline.stage1_v2_trainer_interface import PARITY


PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_remediation_protocol_v1.json"
)
LOCK = Path("audit/v2/stage1_v2_phase6_remediation_v1/PHASE6_REMEDIATION_LOCK.json")
OUTPUT = Path("model_kernels/stage1_v2_phase6_remediation_v1/phase_1")
RUNS = Path("trained_models/stage1_v2_phase6_remediation_v1_runs/phase_1")
TRAINER_MODULE = "server_training_pipeline.train_stage1_v2_phase6_remediation_tf"
REFERENCE = "historical_reaction_reference"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def build_grid(root: Path, protocol: dict[str, Any]) -> pd.DataFrame:
    registry = pd.read_csv(root / PARITY / "splits/state_registry.tsv", sep="\t", dtype=str)
    phase = protocol["phase_1"]
    states = registry.loc[
        registry["state_level"].eq("INNER")
        & registry["scenario"].isin(phase["scenarios"])
    ].copy()
    states["outer_fold"] = states["outer_fold"].astype(int)
    states["inner_fold"] = states["inner_fold"].astype(int)
    states = states.loc[
        states["outer_fold"].isin(phase["outer_folds"])
        & states["inner_fold"].isin(phase["inner_folds"])
    ].copy()
    scenario_order = {value: index for index, value in enumerate(phase["scenarios"])}
    states["scenario_index"] = states["scenario"].map(scenario_order)
    states = states.sort_values(["scenario_index", "inner_fold", "state_id"])
    if len(states) != 25 or not states["state_id"].is_unique:
        raise ValueError(f"Expected 25 Phase-1 states; observed={len(states)}")
    rows = []
    for state in states.itertuples(index=False):
        seed = 74000 + int(state.scenario_index) * 1000 + int(state.inner_fold) * 10 + 1
        for candidate in phase["candidate_order"]:
            if state.scenario not in protocol["candidates"][candidate]["eligible_scenarios"]:
                continue
            rows.append(
                {
                    "state_id": state.state_id,
                    "scenario": state.scenario,
                    "outer_fold": state.outer_fold,
                    "inner_fold": state.inner_fold,
                    "candidate": candidate,
                    "seed": seed,
                }
            )
    grid = pd.DataFrame(rows)
    if len(grid) != int(phase["candidate_state_count"]):
        raise ValueError(f"Remediation grid disagrees with protocol: {len(grid)}")
    return grid


def run_complete(path: Path) -> bool:
    metadata = path / "run_metadata.json"
    required = [
        metadata,
        path / "validation_trait_metrics.tsv",
        path / "validation_guard_metrics.tsv",
        path / "validation_subset_metrics.tsv",
        path / "component_epoch_history.tsv",
        path / "active_component_factors.tsv",
        path / "training_only_calibration.tsv",
    ]
    if not all(item.is_file() for item in required):
        return False
    try:
        value = load_json(metadata)
    except (json.JSONDecodeError, OSError):
        return False
    return value.get("status") == "PASS" and value.get("outer_test_metrics_read") is False


def execute_run(
    root: Path,
    code_root: Path,
    python: Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    out_dir = root / RUNS / row["state_id"] / row["candidate"]
    if run_complete(out_dir):
        return load_json(out_dir / "run_metadata.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    command = [
        str(python),
        "-m",
        TRAINER_MODULE,
        "--root",
        str(root),
        "--state-id",
        str(row["state_id"]),
        "--candidate",
        str(row["candidate"]),
        "--seed",
        str(row["seed"]),
        "--out-dir",
        str(out_dir),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(code_root)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=code_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise RuntimeError(
            f"Remediation run failed: {row['state_id']} {row['candidate']}\n{tail}"
        )
    if not run_complete(out_dir):
        raise RuntimeError(f"Remediation run did not certify: {out_dir}")
    return load_json(out_dir / "run_metadata.json")


def collect_run_tables(root: Path, grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = []
    traits = []
    guards = []
    for row in grid.itertuples(index=False):
        run_dir = root / RUNS / row.state_id / row.candidate
        value = load_json(run_dir / "run_metadata.json")
        metadata.append(value)
        trait = pd.read_csv(run_dir / "validation_trait_metrics.tsv", sep="\t")
        trait.insert(0, "candidate", row.candidate)
        trait.insert(0, "scenario", row.scenario)
        trait.insert(0, "state_id", row.state_id)
        traits.append(trait)
        guard = pd.read_csv(run_dir / "validation_guard_metrics.tsv", sep="\t")
        guard.insert(0, "candidate", row.candidate)
        guard.insert(0, "scenario", row.scenario)
        guard.insert(0, "state_id", row.state_id)
        guards.append(guard)
    return (
        pd.DataFrame(metadata),
        pd.concat(traits, ignore_index=True),
        pd.concat(guards, ignore_index=True),
    )


def pair_runs(runs: pd.DataFrame) -> pd.DataFrame:
    reference_columns = [
        "state_id",
        "validation_observation_signature",
        "validation_macro_normalized_rmse",
        "validation_macro_pearson",
        "validation_macro_calibration_error",
        "within_environment_centered_spearman",
        "within_environment_pairwise_accuracy",
    ]
    reference = runs.loc[runs["candidate"].eq(REFERENCE), reference_columns].copy()
    reference = reference.rename(
        columns={column: f"{column}_reference" for column in reference_columns if column != "state_id"}
    )
    paired = runs.merge(reference, on="state_id", how="left", validate="many_to_one")
    if paired["validation_macro_normalized_rmse_reference"].isna().any():
        raise ValueError("A remediation run lacks its stable reference")
    if not paired["validation_observation_signature"].eq(
        paired["validation_observation_signature_reference"]
    ).all():
        raise ValueError("Remediation validation observations are not paired")
    paired["relative_nrmse_gain"] = (
        paired["validation_macro_normalized_rmse_reference"]
        - paired["validation_macro_normalized_rmse"]
    ) / paired["validation_macro_normalized_rmse_reference"]
    paired["nrmse_win"] = paired["validation_macro_normalized_rmse"].lt(
        paired["validation_macro_normalized_rmse_reference"]
    )
    paired["pearson_gain"] = (
        paired["validation_macro_pearson"] - paired["validation_macro_pearson_reference"]
    )
    paired["centered_spearman_gain"] = (
        paired["within_environment_centered_spearman"]
        - paired["within_environment_centered_spearman_reference"]
    )
    paired["pairwise_accuracy_gain"] = (
        paired["within_environment_pairwise_accuracy"]
        - paired["within_environment_pairwise_accuracy_reference"]
    )
    return paired


def pair_traits(traits: pd.DataFrame) -> pd.DataFrame:
    reference = traits.loc[traits["candidate"].eq(REFERENCE)].copy()
    reference = reference[
        ["state_id", "trait_name_canonical", "rows", "normalized_rmse", "pearson"]
    ].rename(
        columns={
            "rows": "rows_reference",
            "normalized_rmse": "normalized_rmse_reference",
            "pearson": "pearson_reference",
        }
    )
    paired = traits.merge(
        reference,
        on=["state_id", "trait_name_canonical"],
        how="left",
        validate="many_to_one",
    )
    available = paired["normalized_rmse_reference"].notna()
    if not paired.loc[available, "rows"].eq(paired.loc[available, "rows_reference"]).all():
        raise ValueError("Trait comparisons do not use identical rows")
    paired["relative_nrmse_gain"] = (
        paired["normalized_rmse_reference"] - paired["normalized_rmse"]
    ) / paired["normalized_rmse_reference"]
    paired["pearson_gain"] = paired["pearson"] - paired["pearson_reference"]
    return paired


def pair_guards(guards: pd.DataFrame) -> pd.DataFrame:
    candidate_masks = guards.loc[guards["mask_candidate"].eq(guards["candidate"])].copy()
    reference_predictions = guards.loc[guards["candidate"].eq(REFERENCE)].copy()
    reference_predictions = reference_predictions[
        [
            "state_id",
            "mask_candidate",
            "subset",
            "rows",
            "observation_id_signature",
            "normalized_rmse_macro",
            "pearson_macro",
        ]
    ].rename(
        columns={
            "rows": "rows_reference",
            "observation_id_signature": "observation_id_signature_reference",
            "normalized_rmse_macro": "normalized_rmse_macro_reference",
            "pearson_macro": "pearson_macro_reference",
        }
    )
    paired = candidate_masks.merge(
        reference_predictions,
        on=["state_id", "mask_candidate", "subset"],
        how="left",
        validate="many_to_one",
    )
    comparable = paired["rows"].gt(0)
    if not paired.loc[comparable, "rows"].eq(paired.loc[comparable, "rows_reference"]).all():
        raise ValueError("Guard comparisons have unequal row counts")
    if not paired.loc[comparable, "observation_id_signature"].eq(
        paired.loc[comparable, "observation_id_signature_reference"]
    ).all():
        raise ValueError("Guard comparisons have unequal observation identifiers")
    paired["relative_nrmse_gain"] = (
        paired["normalized_rmse_macro_reference"] - paired["normalized_rmse_macro"]
    ) / paired["normalized_rmse_macro_reference"]
    paired["pearson_gain"] = paired["pearson_macro"] - paired["pearson_macro_reference"]
    return paired


def summarize(
    protocol: dict[str, Any],
    paired: pd.DataFrame,
    paired_traits: pd.DataFrame,
    paired_guards: pd.DataFrame,
) -> pd.DataFrame:
    acceptance = protocol["phase_1_acceptance"]
    primary = set(protocol["primary_traits"])
    information_subsets = {
        "PEDIGREE_ONLY",
        "MARKER_SUPPORTED",
        "PEDIGREE_AND_MARKER",
        "NEITHER_PRODUCTION_PEDIGREE_NOR_DENSE_MARKERS",
        "RECOVERED_IDENTITY_OR_COMPONENT",
    }
    rows = []
    for (scenario, candidate), local in paired.groupby(["scenario", "candidate"], sort=False):
        local_traits = paired_traits.loc[
            paired_traits["scenario"].eq(scenario)
            & paired_traits["candidate"].eq(candidate)
        ]
        local_guards = paired_guards.loc[
            paired_guards["scenario"].eq(scenario)
            & paired_guards["candidate"].eq(candidate)
            & paired_guards["rows"].ge(int(acceptance["minimum_rows_for_guard"]))
        ]
        primary_gain = local_traits.loc[
            local_traits["trait_name_canonical"].isin(primary), "relative_nrmse_gain"
        ].min()
        information_gain = local_guards.loc[
            local_guards["subset"].isin(information_subsets), "relative_nrmse_gain"
        ].min()
        inactive_gain = mean(
            local_guards.loc[
                local_guards["subset"].eq(
                    "PROJECTION_CORE_INACTIVE_814_ENVIRONMENTS"
                ),
                "relative_nrmse_gain",
            ]
        )
        gain = mean(local["relative_nrmse_gain"])
        win_rate = mean(local["nrmse_win"])
        pearson_gain = mean(local["pearson_gain"])
        spearman_gain = mean(local["centered_spearman_gain"])
        pairwise_gain = mean(local["pairwise_accuracy_gain"])
        macro_calibration = float(local_traits["calibration_error"].max())
        primary_calibration = float(
            local_traits.loc[
                local_traits["trait_name_canonical"].isin(primary), "calibration_error"
            ].max()
        )
        negative_slopes = int((local_traits["calibration_slope"] < 0).sum())
        guards_ok = {
            "overall_gain": gain >= float(acceptance["minimum_relative_nrmse_gain"]),
            "fold_win_rate": win_rate
            >= float(acceptance["minimum_paired_inner_fold_win_rate"]),
            "pearson": pearson_gain >= -float(acceptance["maximum_mean_pearson_drop"]),
            "macro_calibration": macro_calibration
            <= float(acceptance["maximum_absolute_macro_calibration_error"]),
            "primary_calibration": primary_calibration
            <= float(acceptance["maximum_primary_trait_absolute_calibration_error"]),
            "negative_slopes": negative_slopes == 0,
            "centered_spearman": spearman_gain
            >= -float(acceptance["within_environment_centered_spearman_maximum_drop"]),
            "pairwise_accuracy": pairwise_gain
            >= -float(acceptance["within_environment_pairwise_accuracy_maximum_drop"]),
            "primary_traits": np.isnan(primary_gain)
            or primary_gain >= -float(acceptance["primary_trait_maximum_relative_nrmse_loss"]),
            "information_subsets": np.isnan(information_gain)
            or information_gain
            >= -float(acceptance["information_class_maximum_relative_nrmse_loss"]),
            "projection_inactive": np.isnan(inactive_gain)
            or inactive_gain
            >= -float(
                acceptance["projection_inactive_environment_maximum_relative_nrmse_loss"]
            ),
        }
        eligible = candidate == REFERENCE or all(guards_ok.values())
        rows.append(
            {
                "scenario": scenario,
                "candidate": candidate,
                "paired_inner_folds": int(local["state_id"].nunique()),
                "validation_normalized_rmse_mean": mean(
                    local["validation_macro_normalized_rmse"]
                ),
                "validation_pearson_mean": mean(local["validation_macro_pearson"]),
                "relative_normalized_rmse_gain_mean": gain,
                "normalized_rmse_win_rate": win_rate,
                "pearson_gain_mean": pearson_gain,
                "centered_spearman_gain_mean": spearman_gain,
                "pairwise_accuracy_gain_mean": pairwise_gain,
                "absolute_macro_calibration_error_max": macro_calibration,
                "primary_trait_calibration_error_max": primary_calibration,
                "negative_trait_calibration_slopes": negative_slopes,
                "primary_trait_relative_nrmse_gain_min": primary_gain,
                "information_subset_relative_nrmse_gain_min": information_gain,
                "projection_inactive_relative_nrmse_gain_mean": inactive_gain,
                **{f"guard_{name}": value for name, value in guards_ok.items()},
                "eligible_for_full_confirmation": eligible,
                "decision": (
                    "stable_reference"
                    if candidate == REFERENCE
                    else (
                        "advance_to_full_125_state_confirmation"
                        if eligible
                        else "do_not_advance"
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage-1 v2 structural remediation on inner validation only"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-mode", choices=["server_cpu", "wsl_gpu"], default="server_cpu")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=5)
    parser.add_argument("--inter-op-threads", type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    python = Path(os.environ.get("PYTHON", sys.executable)).resolve()
    runtime = validate_runtime(code_root, args.runtime_mode)
    lock = load_json(root / LOCK)
    if lock.get("status") != "PASS_FROZEN_BEFORE_REMEDIATION_INNER_VALIDATION":
        raise ValueError("Remediation screen is not frozen")
    protocol = load_json(code_root / PROTOCOL)
    grid = build_grid(root, protocol)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output / "remediation_phase1_run_grid.tsv", sep="\t", index=False)
    started = {
        "status": "RUNNING",
        "protocol_version": protocol["protocol_version"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(grid),
        "parallel_workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "runtime": runtime,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
    }
    write_json(output / "remediation_phase1_status.json", started)
    environment = os.environ
    environment["STAGE1_V2_EXECUTION_BACKEND"] = args.runtime_mode
    environment["STAGE1_V2_INTRA_OP_THREADS"] = str(args.threads_per_worker)
    environment["STAGE1_V2_INTER_OP_THREADS"] = str(args.inter_op_threads)
    pending = [row._asdict() for row in grid.itertuples(index=False) if not run_complete(
        root / RUNS / row.state_id / row.candidate
    )]
    print(
        f"RUN Stage-1 v2 remediation; total={len(grid)} pending={len(pending)} "
        f"workers={args.workers} threads_per_worker={args.threads_per_worker}",
        flush=True,
    )
    completed = len(grid) - len(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(execute_run, root, code_root, python, row): row
            for row in pending
        }
        for future in as_completed(futures):
            row = futures[future]
            future.result()
            completed += 1
            print(
                f"[{completed}/{len(grid)}] DONE {row['state_id']} {row['candidate']}",
                flush=True,
            )
            gc.collect()
    runs, traits, guards = collect_run_tables(root, grid)
    paired = pair_runs(runs)
    paired_traits = pair_traits(traits)
    paired_guards = pair_guards(guards)
    decision = summarize(protocol, paired, paired_traits, paired_guards)
    runs.to_csv(output / "remediation_phase1_runs.tsv", sep="\t", index=False)
    paired.to_csv(output / "remediation_phase1_paired_metrics.tsv", sep="\t", index=False)
    paired_traits.to_csv(
        output / "remediation_phase1_paired_trait_metrics.tsv", sep="\t", index=False
    )
    paired_guards.to_csv(
        output / "remediation_phase1_paired_guard_metrics.tsv", sep="\t", index=False
    )
    decision.to_csv(output / "remediation_phase1_decision.tsv", sep="\t", index=False)
    advanced = decision.loc[
        decision["decision"].eq("advance_to_full_125_state_confirmation"),
        ["scenario", "candidate"],
    ].to_dict("records")
    final = {
        "status": "PASS_STAGE1_V2_PHASE6_REMEDIATION_PHASE1_COMPLETE",
        "protocol_version": protocol["protocol_version"],
        "selection_data": "nested_inner_validation_only",
        "run_count": len(runs),
        "state_count": int(runs["state_id"].nunique()),
        "advanced_candidates": advanced,
        "phase2_optimizer_allowed": bool(advanced),
        "outer_evaluation_allowed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "PHASE1_STRUCTURAL_DECISION.json", final)
    write_json(output / "remediation_phase1_status.json", final)
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
