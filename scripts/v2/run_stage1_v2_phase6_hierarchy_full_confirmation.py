from __future__ import annotations

import argparse
import gc
import hashlib
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
from scripts.v2.run_stage1_v2_phase6_remediation import (
    pair_runs,
    pair_traits,
    summarize,
)
from server_training_pipeline.stage1_v2_trainer_interface import PARITY


PROTOCOL = Path(
    "server_training_pipeline/"
    "stage1_v2_phase6_hierarchy_full_confirmation_protocol_v1.json"
)
LOCK = Path(
    "audit/v2/stage1_v2_phase6_hierarchy_full_confirmation_v1/"
    "PHASE6_HIERARCHY_FULL_CONFIRMATION_LOCK.json"
)
RUNS = Path("trained_models/stage1_v2_phase6_hierarchy_full_confirmation_v1_runs")
OUTPUT = Path("model_kernels/stage1_v2_phase6_hierarchy_full_confirmation_v1")
SOURCE_RUNS = Path("trained_models/stage1_v2_phase6_confirmation_v1_runs")
TRAINER_MODULE = (
    "server_training_pipeline."
    "train_stage1_v2_phase6_hierarchy_full_confirmation_tf"
)
RUN_PROTOCOL = "stage1_v2_phase6_hierarchy_full_confirmation_tf_v1"
REFERENCE = "historical_reaction_reference"
SELECTED = "hierarchy_test_weight_identity_calibration_v1"
MASK_CANDIDATE = "marker_supported_output_routed_v2"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def build_grid(root: Path, protocol: dict[str, Any]) -> pd.DataFrame:
    registry = pd.read_csv(root / PARITY / "splits/state_registry.tsv", sep="\t", dtype=str)
    states = registry.loc[
        registry["state_level"].eq("INNER")
        & registry["scenario"].eq("GNEW_EOBS")
    ].copy()
    states["outer_fold"] = states["outer_fold"].astype(int)
    states["inner_fold"] = states["inner_fold"].astype(int)
    states = states.sort_values(["outer_fold", "inner_fold", "state_id"])
    if len(states) != 25 or not states["state_id"].is_unique:
        raise ValueError(f"Expected 25 GNEW_EOBS inner states; observed={len(states)}")
    rows: list[dict[str, object]] = []
    for state in states.itertuples(index=False):
        seed = 63000 + int(state.outer_fold) * 100 + int(state.inner_fold) * 10 + 1
        for candidate in (REFERENCE, SELECTED):
            rows.append(
                {
                    "state_id": str(state.state_id),
                    "scenario": "GNEW_EOBS",
                    "outer_fold": int(state.outer_fold),
                    "inner_fold": int(state.inner_fold),
                    "candidate": candidate,
                    "seed": seed,
                }
            )
    grid = pd.DataFrame(rows)
    expected = int(protocol["confirmation_grid"]["matched_training_run_count"])
    if len(grid) != expected or grid.groupby("state_id")["seed"].nunique().max() != 1:
        raise ValueError("Full-confirmation matched run grid is invalid")
    return grid


def run_dir(root: Path, row: dict[str, Any]) -> Path:
    return root / RUNS / str(row["state_id"]) / str(row["candidate"])


def run_complete(path: Path) -> bool:
    required = [
        path / "run_metadata.json",
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
        metadata = read_json(path / "run_metadata.json")
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("status") == "PASS"
        and metadata.get("protocol_version") == RUN_PROTOCOL
        and metadata.get("outer_test_metrics_read") is False
        and metadata.get("outer_test_outcomes_read") is False
        and metadata.get("final_holdout_outcomes_read") is False
    )


def execute_run(
    root: Path, code_root: Path, python: Path, row: dict[str, Any]
) -> dict[str, Any]:
    out_dir = run_dir(root, row)
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
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]
        )
        raise RuntimeError(
            f"Hierarchy confirmation run failed: {row['state_id']} "
            f"{row['candidate']}\n{tail}"
        )
    if not run_complete(out_dir):
        raise RuntimeError(f"Hierarchy confirmation run did not certify: {out_dir}")
    return read_json(out_dir / "run_metadata.json")


def collect_tables(
    root: Path, grid: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs: list[dict[str, Any]] = []
    traits: list[pd.DataFrame] = []
    guards: list[pd.DataFrame] = []
    for row in grid.itertuples(index=False):
        path = root / RUNS / row.state_id / row.candidate
        runs.append(read_json(path / "run_metadata.json"))
        trait = pd.read_csv(path / "validation_trait_metrics.tsv", sep="\t")
        guard = pd.read_csv(path / "validation_guard_metrics.tsv", sep="\t")
        for frame in (trait, guard):
            frame.insert(0, "candidate", row.candidate)
            frame.insert(0, "scenario", row.scenario)
            frame.insert(0, "state_id", row.state_id)
        traits.append(trait)
        guards.append(guard)
    return (
        pd.DataFrame(runs),
        pd.concat(traits, ignore_index=True),
        pd.concat(guards, ignore_index=True),
    )


def pair_guards(guards: pd.DataFrame) -> pd.DataFrame:
    shared = guards.loc[guards["mask_candidate"].eq(MASK_CANDIDATE)].copy()
    reference = shared.loc[shared["candidate"].eq(REFERENCE)].copy()
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
    paired = shared.merge(
        reference, on=["state_id", "subset"], how="left", validate="many_to_one"
    )
    comparable = paired["rows"].gt(0)
    if not paired.loc[comparable, "rows"].eq(
        paired.loc[comparable, "rows_reference"]
    ).all():
        raise ValueError("Full-confirmation guards have unequal row counts")
    if not paired.loc[comparable, "observation_id_signature"].eq(
        paired.loc[comparable, "observation_id_signature_reference"]
    ).all():
        raise ValueError("Full-confirmation guards have unequal observation identifiers")
    paired["relative_nrmse_gain"] = (
        paired["normalized_rmse_macro_reference"] - paired["normalized_rmse_macro"]
    ) / paired["normalized_rmse_macro_reference"]
    paired["pearson_gain"] = (
        paired["pearson_macro"] - paired["pearson_macro_reference"]
    )
    return paired


def build_route_manifest(
    root: Path, protocol: dict[str, Any], candidate_runs: pd.DataFrame
) -> pd.DataFrame:
    registry = pd.read_csv(root / PARITY / "splits/state_registry.tsv", sep="\t", dtype=str)
    states = registry.loc[
        registry["state_level"].eq("INNER")
        & registry["scenario"].isin(protocol["confirmation_grid"]["scenarios"])
    ].copy()
    candidate_lookup = candidate_runs.set_index(["state_id", "candidate"])
    rows: list[dict[str, object]] = []
    for state in states.itertuples(index=False):
        scenario = str(state.scenario)
        candidate = str(protocol["routing_policy"][scenario]["candidate"])
        if scenario == "GNEW_EOBS":
            metadata_path = root / RUNS / state.state_id / candidate / "run_metadata.json"
            source_class = "NEW_MATCHED_HIERARCHY_CONFIRMATION"
            metadata = candidate_lookup.loc[(state.state_id, candidate)].to_dict()
        else:
            metadata_path = (
                root / SOURCE_RUNS / state.state_id / REFERENCE / "run_metadata.json"
            )
            source_class = "EXACT_CERTIFIED_HISTORICAL_REFERENCE_REUSE"
            metadata = read_json(metadata_path)
        if metadata.get("status") != "PASS":
            raise ValueError(f"Routed state is not certified: {state.state_id}")
        if metadata.get("outer_test_metrics_read") is not False:
            raise ValueError(f"Routed state read outer metrics: {state.state_id}")
        rows.append(
            {
                "state_id": state.state_id,
                "scenario": scenario,
                "outer_fold": int(state.outer_fold),
                "inner_fold": int(state.inner_fold),
                "routed_candidate": candidate,
                "route_source_class": source_class,
                "run_metadata_path": metadata_path.relative_to(root).as_posix(),
                "run_metadata_sha256": sha256_file(metadata_path),
                "seed": int(metadata["seed"]),
                "validation_observation_signature": metadata[
                    "validation_observation_signature"
                ],
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["scenario", "outer_fold", "inner_fold", "state_id"]
    )
    if len(result) != 125 or not result["state_id"].is_unique:
        raise ValueError("Routed full-confirmation manifest is incomplete")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run routed Stage-1 v2 hierarchy full inner confirmation"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    parser.add_argument(
        "--runtime-mode", choices=["server_cpu", "wsl_gpu"], default="server_cpu"
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=5)
    parser.add_argument("--inter-op-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = (
        args.code_root
        or Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root))
    ).resolve()
    python = Path(os.environ.get("PYTHON", sys.executable)).resolve()
    runtime = validate_runtime(code_root, args.runtime_mode)
    protocol = read_json(code_root / PROTOCOL)
    lock = read_json(root / LOCK)
    if lock.get("status") != "PASS_FROZEN_BEFORE_HIERARCHY_FULL_INNER_CONFIRMATION":
        raise ValueError("Hierarchy full confirmation is not frozen")
    grid = build_grid(root, protocol)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output / "full_confirmation_run_grid.tsv", sep="\t", index=False)
    write_json(
        output / "full_confirmation_status.json",
        {
            "status": "RUNNING",
            "protocol_version": protocol["protocol_version"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "matched_training_run_count": len(grid),
            "routed_state_count": 125,
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
        f"RUN hierarchy full confirmation; total={len(grid)} pending={len(pending)} "
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
    paired_guards = pair_guards(guards)
    decision = summarize(protocol, paired, paired_traits, paired_guards)
    selected_row = decision.loc[decision["candidate"].eq(SELECTED)]
    eligible_guard_rows = paired_guards["rows"].ge(
        int(protocol["phase_1_acceptance"]["minimum_rows_for_guard"])
    )
    all_guards = (
        len(selected_row) == 1
        and bool(selected_row.filter(regex=r"^guard_").iloc[0].astype(bool).all())
        and bool(selected_row["eligible_for_full_confirmation"].iloc[0])
        and int(selected_row["paired_inner_folds"].iloc[0]) == 25
    )
    decision["decision"] = np.where(
        decision["candidate"].eq(REFERENCE),
        "stable_reference",
        np.where(all_guards, "freeze_routed_inner_architecture", "do_not_advance"),
    )
    routes = build_route_manifest(root, protocol, runs)
    route_counts = routes.groupby(
        ["routed_candidate", "route_source_class"], sort=False
    ).size().reset_index(name="states")
    checks = {
        "matched_run_count": len(runs) == 50,
        "matched_seed_status": runs.groupby("state_id")["seed"].nunique().max() == 1,
        "matched_validation_observations": paired[
            "validation_observation_signature"
        ].eq(paired["validation_observation_signature_reference"]).all(),
        "corrected_guard_rows_exact": paired_guards.loc[
            paired_guards["rows"].gt(0), "rows"
        ].eq(paired_guards.loc[paired_guards["rows"].gt(0), "rows_reference"]).all(),
        "corrected_guard_identifiers_exact": paired_guards.loc[
            paired_guards["rows"].gt(0), "observation_id_signature"
        ].eq(
            paired_guards.loc[
                paired_guards["rows"].gt(0),
                "observation_id_signature_reference",
            ]
        ).all(),
        "guard_subset_schema_complete": len(paired_guards) == 25 * 2 * 7
        and set(paired_guards["subset"])
        == set(protocol["mandatory_reporting_subsets"]),
        "eligible_guard_metrics_finite": np.isfinite(
            paired_guards.loc[
                eligible_guard_rows,
                ["relative_nrmse_gain", "pearson_gain"],
            ].to_numpy(dtype=float)
        ).all(),
        "selected_candidate_all_guards_pass": all_guards,
        "route_manifest_125": len(routes) == 125 and routes["state_id"].is_unique,
        "active_hierarchy_routes_25": int(routes["routed_candidate"].eq(SELECTED).sum())
        == 25,
        "exact_reference_reuse_routes_100": int(
            routes["route_source_class"].eq(
                "EXACT_CERTIFIED_HISTORICAL_REFERENCE_REUSE"
            ).sum()
        )
        == 100,
        "outer_unread": runs["outer_test_metrics_read"].eq(False).all()
        and routes["outer_test_metrics_read"].eq(False).all(),
        "final_holdout_unread": runs["final_holdout_outcomes_read"].eq(False).all()
        and routes["final_holdout_outcomes_read"].eq(False).all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]

    artifacts = {
        "full_confirmation_runs.tsv": runs,
        "full_confirmation_trait_metrics.tsv": traits,
        "full_confirmation_guard_metrics.tsv": guards,
        "full_confirmation_paired_metrics.tsv": paired,
        "full_confirmation_paired_trait_metrics.tsv": paired_traits,
        "full_confirmation_paired_guard_metrics.tsv": paired_guards,
        "full_confirmation_decision.tsv": decision,
        "full_confirmation_route_manifest.tsv": routes,
        "full_confirmation_route_summary.tsv": route_counts,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output / name, sep="\t", index=False, lineterminator="\n")
    final = {
        "status": (
            "PASS_STAGE1_V2_PHASE6_HIERARCHY_FULL_INNER_CONFIRMATION"
            if not failed
            else "FAIL_STAGE1_V2_PHASE6_HIERARCHY_FULL_INNER_CONFIRMATION"
        ),
        "protocol_version": protocol["protocol_version"],
        "selection_data": "nested_inner_validation_only",
        "selected_candidate": SELECTED if not failed else None,
        "routed_state_count": 125,
        "active_hierarchy_state_count": 25,
        "exact_reference_reuse_state_count": 100,
        "matched_training_run_count": 50,
        "test_weight_calibration": "identity",
        "outer_evaluation_performed": False,
        "outer_evaluation_allowed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "checks": checks,
        "failed_checks": failed,
        "artifact_sha256": {
            name: sha256_file(output / name) for name in artifacts
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "FULL_CONFIRMATION_DECISION.json", final)
    write_json(output / "full_confirmation_status.json", final)
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    if failed:
        raise SystemExit(f"Hierarchy full inner confirmation failed: {failed}")


if __name__ == "__main__":
    main()
