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

import pandas as pd

from scripts.v2.run_stage1_v2_phase6_phase1 import validate_runtime
from scripts.v2.run_stage1_v2_phase6_remediation import (
    pair_runs,
    pair_traits,
    summarize,
)
from server_training_pipeline.stage1_v2_trainer_interface import PARITY


PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_hierarchy_calibration_protocol_v1.json"
)
LOCK = Path(
    "audit/v2/stage1_v2_phase6_hierarchy_calibration_v1/"
    "PHASE6_HIERARCHY_CALIBRATION_LOCK.json"
)
SOURCE_RUNS = Path("trained_models/stage1_v2_phase6_remediation_v1_runs/phase_1")
OUTPUT = Path("model_kernels/stage1_v2_phase6_hierarchy_calibration_v1/phase_1")
RUNS = Path("trained_models/stage1_v2_phase6_hierarchy_calibration_v1_runs/phase_1")
TRAINER_MODULE = (
    "server_training_pipeline.train_stage1_v2_phase6_hierarchy_calibration_tf"
)
REFERENCE = "historical_reaction_reference"
SOURCE_HIERARCHY = "known_environment_hierarchical_v2"
MASK_CANDIDATE = "marker_supported_output_routed_v2"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_grid(root: Path, protocol: dict[str, Any]) -> pd.DataFrame:
    registry = pd.read_csv(root / PARITY / "splits/state_registry.tsv", sep="\t", dtype=str)
    phase = protocol["phase_1"]
    states = registry.loc[
        registry["state_level"].eq("INNER")
        & registry["scenario"].eq("GNEW_EOBS")
    ].copy()
    states["outer_fold"] = states["outer_fold"].astype(int)
    states["inner_fold"] = states["inner_fold"].astype(int)
    states = states.loc[
        states["outer_fold"].isin(phase["outer_folds"])
        & states["inner_fold"].isin(phase["inner_folds"])
    ].sort_values(["inner_fold", "state_id"])
    if len(states) != 5 or not states["state_id"].is_unique:
        raise ValueError(f"Expected five hierarchy calibration states; observed={len(states)}")
    rows: list[dict[str, object]] = []
    for state in states.itertuples(index=False):
        seed = 74000 + int(state.inner_fold) * 10 + 1
        for candidate in phase["candidate_order"]:
            rows.append(
                {
                    "state_id": state.state_id,
                    "scenario": state.scenario,
                    "outer_fold": int(state.outer_fold),
                    "inner_fold": int(state.inner_fold),
                    "candidate": candidate,
                    "seed": seed,
                }
            )
    grid = pd.DataFrame(rows)
    if len(grid) != int(phase["new_training_run_count"]):
        raise ValueError("Hierarchy calibration grid disagrees with the protocol")
    return grid


def run_complete(path: Path) -> bool:
    required = [
        path / "run_metadata.json",
        path / "validation_trait_metrics.tsv",
        path / "validation_guard_metrics.tsv",
        path / "validation_subset_metrics.tsv",
        path / "component_epoch_history.tsv",
        path / "active_component_factors.tsv",
        path / "training_only_calibration.tsv",
        path / "training_only_calibration_crossfit.tsv",
    ]
    if not all(item.is_file() for item in required):
        return False
    try:
        metadata = read_json(path / "run_metadata.json")
    except (json.JSONDecodeError, OSError):
        return False
    return (
        metadata.get("status") == "PASS"
        and metadata.get("outer_test_metrics_read") is False
        and metadata.get("final_holdout_outcomes_read") is False
    )


def execute_run(
    root: Path,
    code_root: Path,
    python: Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    out_dir = root / RUNS / row["state_id"] / row["candidate"]
    if run_complete(out_dir):
        return read_json(out_dir / "run_metadata.json")
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
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        )
        raise RuntimeError(
            f"Hierarchy calibration run failed: {row['state_id']} {row['candidate']}\n{tail}"
        )
    if not run_complete(out_dir):
        raise RuntimeError(f"Hierarchy calibration run did not certify: {out_dir}")
    return read_json(out_dir / "run_metadata.json")


def load_run_tables(
    run_root: Path,
    state_id: str,
    candidate: str,
    scenario: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    path = run_root / state_id / candidate
    metadata = read_json(path / "run_metadata.json")
    trait = pd.read_csv(path / "validation_trait_metrics.tsv", sep="\t")
    guard = pd.read_csv(path / "validation_guard_metrics.tsv", sep="\t")
    for frame in (trait, guard):
        frame.insert(0, "candidate", candidate)
        frame.insert(0, "scenario", scenario)
        frame.insert(0, "state_id", state_id)
    return metadata, trait, guard


def collect_tables(
    root: Path, grid: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata: list[dict[str, Any]] = []
    traits: list[pd.DataFrame] = []
    guards: list[pd.DataFrame] = []
    for state_id, state_grid in grid.groupby("state_id", sort=False):
        scenario = str(state_grid["scenario"].iloc[0])
        for candidate in (REFERENCE, SOURCE_HIERARCHY):
            values = load_run_tables(root / SOURCE_RUNS, state_id, candidate, scenario)
            metadata.append(values[0])
            traits.append(values[1])
            guards.append(values[2])
        for candidate in state_grid["candidate"].astype(str):
            values = load_run_tables(root / RUNS, state_id, candidate, scenario)
            metadata.append(values[0])
            traits.append(values[1])
            guards.append(values[2])
    return (
        pd.DataFrame(metadata),
        pd.concat(traits, ignore_index=True),
        pd.concat(guards, ignore_index=True),
    )


def pair_corrected_guards(guards: pd.DataFrame) -> pd.DataFrame:
    selected = guards.loc[guards["mask_candidate"].eq(MASK_CANDIDATE)].copy()
    reference = selected.loc[selected["candidate"].eq(REFERENCE)].copy()
    reference = reference[
        [
            "state_id",
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
    paired = selected.merge(
        reference,
        on=["state_id", "subset"],
        how="left",
        validate="many_to_one",
    )
    comparable = paired["rows"].gt(0)
    if not paired.loc[comparable, "rows"].eq(
        paired.loc[comparable, "rows_reference"]
    ).all():
        raise ValueError("Corrected information guards have unequal rows")
    if not paired.loc[comparable, "observation_id_signature"].eq(
        paired.loc[comparable, "observation_id_signature_reference"]
    ).all():
        raise ValueError("Corrected information guards have unequal identifiers")
    paired["relative_nrmse_gain"] = (
        paired["normalized_rmse_macro_reference"] - paired["normalized_rmse_macro"]
    ) / paired["normalized_rmse_macro_reference"]
    paired["pearson_gain"] = paired["pearson_macro"] - paired["pearson_macro_reference"]
    return paired


def select_candidate(
    decision: pd.DataFrame, protocol: dict[str, Any]
) -> str | None:
    eligible = decision.loc[
        decision["candidate"].isin(protocol["phase_1"]["candidate_order"])
        & decision["eligible_for_full_confirmation"]
    ].copy()
    if eligible.empty:
        return None
    order = {
        candidate: index
        for index, candidate in enumerate(protocol["phase_1"]["candidate_order"])
    }
    eligible["candidate_order"] = eligible["candidate"].map(order)
    eligible = eligible.sort_values(
        [
            "validation_normalized_rmse_mean",
            "validation_pearson_mean",
            "candidate_order",
        ],
        ascending=[True, False, True],
    )
    return str(eligible.iloc[0]["candidate"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the bounded Stage-1 v2 hierarchy calibration screen"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--runtime-mode", choices=["server_cpu", "wsl_gpu"], default="server_cpu"
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=5)
    parser.add_argument("--inter-op-threads", type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    python = Path(os.environ.get("PYTHON", sys.executable)).resolve()
    runtime = validate_runtime(code_root, args.runtime_mode)
    lock = read_json(root / LOCK)
    if lock.get("status") != "PASS_FROZEN_BEFORE_HIERARCHY_CALIBRATION_INNER_SCREEN":
        raise ValueError("Hierarchy calibration screen is not frozen")
    protocol = read_json(code_root / PROTOCOL)
    grid = build_grid(root, protocol)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(
        output / "hierarchy_calibration_run_grid.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    write_json(
        output / "hierarchy_calibration_status.json",
        {
            "status": "RUNNING",
            "protocol_version": protocol["protocol_version"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "new_training_run_count": len(grid),
            "parallel_workers": args.workers,
            "threads_per_worker": args.threads_per_worker,
            "runtime": runtime,
            "outer_test_metrics_read": False,
            "final_holdout_outcomes_read": False,
        },
    )
    os.environ["STAGE1_V2_EXECUTION_BACKEND"] = args.runtime_mode
    os.environ["STAGE1_V2_INTRA_OP_THREADS"] = str(args.threads_per_worker)
    os.environ["STAGE1_V2_INTER_OP_THREADS"] = str(args.inter_op_threads)
    pending = [
        row._asdict()
        for row in grid.itertuples(index=False)
        if not run_complete(root / RUNS / row.state_id / row.candidate)
    ]
    print(
        f"RUN Stage-1 v2 hierarchy calibration; total={len(grid)} pending={len(pending)} "
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
    runs, traits, guards = collect_tables(root, grid)
    paired = pair_runs(runs)
    paired_traits = pair_traits(traits)
    paired_guards = pair_corrected_guards(guards)
    decision = summarize(protocol, paired, paired_traits, paired_guards)
    decision["candidate_role"] = decision["candidate"].map(
        {
            REFERENCE: "stable_reference",
            SOURCE_HIERARCHY: "source_failed_calibration_comparator",
        }
    ).fillna("new_calibration_candidate")
    decision.loc[decision["candidate"].eq(SOURCE_HIERARCHY), "decision"] = (
        "source_failed_calibration_comparator"
    )
    decision.loc[decision["candidate"].eq(SOURCE_HIERARCHY), "eligible_for_full_confirmation"] = False
    selected = select_candidate(decision, protocol)
    decision.loc[
        decision["candidate"].isin(protocol["phase_1"]["candidate_order"]),
        "decision",
    ] = "do_not_advance"
    if selected is not None:
        decision.loc[decision["candidate"].eq(selected), "decision"] = (
            "selected_for_full_125_state_confirmation"
        )
    runs.to_csv(output / "hierarchy_calibration_runs.tsv", sep="\t", index=False)
    paired.to_csv(
        output / "hierarchy_calibration_paired_metrics.tsv", sep="\t", index=False
    )
    paired_traits.to_csv(
        output / "hierarchy_calibration_paired_trait_metrics.tsv",
        sep="\t",
        index=False,
    )
    paired_guards.to_csv(
        output / "hierarchy_calibration_paired_guard_metrics.tsv",
        sep="\t",
        index=False,
    )
    decision.to_csv(
        output / "hierarchy_calibration_decision.tsv", sep="\t", index=False
    )
    final = {
        "status": "PASS_STAGE1_V2_PHASE6_HIERARCHY_CALIBRATION_PHASE1_COMPLETE",
        "protocol_version": protocol["protocol_version"],
        "selection_data": "nested_inner_validation_only",
        "new_training_run_count": len(grid),
        "evaluated_run_count_including_source_comparators": len(runs),
        "state_count": int(grid["state_id"].nunique()),
        "selected_candidate": selected,
        "full_125_state_confirmation_allowed": selected is not None,
        "full_confirmation_automatic_launch": False,
        "batch_size_screen_performed": False,
        "fixed_batch_size": int(protocol["fixed_configuration"]["batch_size"]),
        "information_guard_reporting_corrected": True,
        "outer_evaluation_allowed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "PHASE1_HIERARCHY_CALIBRATION_DECISION.json", final)
    write_json(output / "hierarchy_calibration_status.json", final)
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
