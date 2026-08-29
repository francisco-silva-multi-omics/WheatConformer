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
    "stage1_v2_phase6_hierarchy_calibration_amendment_protocol_v2.json"
)
LOCK = Path(
    "audit/v2/stage1_v2_phase6_hierarchy_calibration_amendment_v2/"
    "PHASE6_HIERARCHY_CALIBRATION_AMENDMENT_LOCK.json"
)
RUNS = Path(
    "trained_models/stage1_v2_phase6_hierarchy_calibration_amendment_v2_runs"
)
OUTPUT = Path(
    "model_kernels/stage1_v2_phase6_hierarchy_calibration_amendment_v2"
)
SOURCE_RUNS = Path(
    "trained_models/stage1_v2_phase6_hierarchy_full_confirmation_v1_runs"
)
TRAINER_MODULE = (
    "server_training_pipeline."
    "train_stage1_v2_phase6_hierarchy_calibration_amendment_tf"
)
RUN_PROTOCOL = "stage1_v2_phase6_hierarchy_calibration_amendment_tf_v2"
REFERENCE = "historical_reaction_reference"
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
    states["seed"] = (
        63000 + states["outer_fold"] * 100 + states["inner_fold"] * 10 + 1
    )
    expected = int(protocol["confirmation_scope"]["state_count"])
    if len(states) != expected:
        raise ValueError("Calibration amendment state grid does not match protocol")
    return states[["state_id", "scenario", "outer_fold", "inner_fold", "seed"]]


def state_dir(root: Path, state_id: str) -> Path:
    return root / RUNS / state_id


def state_complete(path: Path, protocol: dict[str, Any]) -> bool:
    shared_path = path / "shared_fit_metadata.json"
    if not shared_path.is_file():
        return False
    try:
        shared = read_json(shared_path)
    except (OSError, json.JSONDecodeError):
        return False
    if (
        shared.get("status") != "PASS"
        or shared.get("protocol_version") != RUN_PROTOCOL
        or shared.get("identity_replay_pass") is not True
    ):
        return False
    required = (
        "run_metadata.json",
        "training_only_calibration.tsv",
        "training_only_calibration_crossfit.tsv",
        "validation_trait_metrics.tsv",
        "validation_subset_metrics.tsv",
        "validation_guard_metrics.tsv",
    )
    for candidate in protocol["confirmation_scope"]["candidate_order"]:
        candidate_dir = path / candidate
        if not all((candidate_dir / name).is_file() for name in required):
            return False
        try:
            metadata = read_json(candidate_dir / "run_metadata.json")
        except (OSError, json.JSONDecodeError):
            return False
        if (
            metadata.get("status") != "PASS"
            or metadata.get("protocol_version") != RUN_PROTOCOL
            or metadata.get("outer_test_metrics_read") is not False
            or metadata.get("final_holdout_outcomes_read") is not False
        ):
            return False
    return True


def execute_state(
    root: Path,
    code_root: Path,
    python: Path,
    row: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    out_dir = state_dir(root, str(row["state_id"]))
    if state_complete(out_dir, protocol):
        return read_json(out_dir / "shared_fit_metadata.json")
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
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        )
        raise RuntimeError(
            f"Hierarchy calibration amendment failed: {row['state_id']}\n{tail}"
        )
    if not state_complete(out_dir, protocol):
        raise RuntimeError(f"Hierarchy calibration state did not certify: {out_dir}")
    return read_json(out_dir / "shared_fit_metadata.json")


def _load_tables(
    path: Path, state_id: str, scenario: str, candidate: str
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    metadata = read_json(path / "run_metadata.json")
    trait = pd.read_csv(path / "validation_trait_metrics.tsv", sep="\t")
    guard = pd.read_csv(path / "validation_guard_metrics.tsv", sep="\t")
    for frame in (trait, guard):
        frame.insert(0, "candidate", candidate)
        frame.insert(0, "scenario", scenario)
        frame.insert(0, "state_id", state_id)
    return metadata, trait, guard


def collect_tables(
    root: Path, grid: pd.DataFrame, protocol: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs: list[dict[str, Any]] = []
    traits: list[pd.DataFrame] = []
    guards: list[pd.DataFrame] = []
    shared: list[dict[str, Any]] = []
    for row in grid.itertuples(index=False):
        state_id = str(row.state_id)
        scenario = str(row.scenario)
        source = root / SOURCE_RUNS / state_id / REFERENCE
        values = _load_tables(source, state_id, scenario, REFERENCE)
        runs.append(values[0])
        traits.append(values[1])
        guards.append(values[2])
        state_root = state_dir(root, state_id)
        shared.append(read_json(state_root / "shared_fit_metadata.json"))
        for candidate in protocol["confirmation_scope"]["candidate_order"]:
            values = _load_tables(state_root / candidate, state_id, scenario, candidate)
            runs.append(values[0])
            traits.append(values[1])
            guards.append(values[2])
    return (
        pd.DataFrame(runs),
        pd.concat(traits, ignore_index=True),
        pd.concat(guards, ignore_index=True),
        pd.DataFrame(shared),
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
        raise ValueError("Calibration guard comparison has unequal row counts")
    if not paired.loc[comparable, "observation_id_signature"].eq(
        paired.loc[comparable, "observation_id_signature_reference"]
    ).all():
        raise ValueError("Calibration guard comparison has unequal observation IDs")
    paired["relative_nrmse_gain"] = (
        paired["normalized_rmse_macro_reference"] - paired["normalized_rmse_macro"]
    ) / paired["normalized_rmse_macro_reference"]
    paired["pearson_gain"] = (
        paired["pearson_macro"] - paired["pearson_macro_reference"]
    )
    return paired


def select_candidate(decision: pd.DataFrame, protocol: dict[str, Any]) -> str | None:
    order = {
        candidate: index
        for index, candidate in enumerate(
            protocol["confirmation_scope"]["candidate_order"]
        )
    }
    eligible = decision.loc[
        decision["candidate"].isin(order)
        & decision["eligible_for_full_confirmation"]
        & decision["paired_inner_folds"].eq(25)
    ].copy()
    if eligible.empty:
        return None
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
        description="Run the Stage-1 v2 hierarchy calibration-only amendment"
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
        args.code_root or Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root))
    ).resolve()
    python = Path(os.environ.get("PYTHON", sys.executable)).resolve()
    runtime = validate_runtime(code_root, args.runtime_mode)
    protocol = read_json(code_root / PROTOCOL)
    lock = read_json(root / LOCK)
    if lock.get("status") != "PASS_FROZEN_CALIBRATION_ONLY_AMENDMENT_V2":
        raise ValueError("Hierarchy calibration-only amendment is not frozen")
    grid = build_grid(root, protocol)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output / "calibration_amendment_state_grid.tsv", sep="\t", index=False)
    write_json(
        output / "calibration_amendment_status.json",
        {
            "status": "RUNNING",
            "protocol_version": protocol["protocol_version"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "state_count": len(grid),
            "new_model_fit_count": len(grid),
            "derived_calibration_result_count": len(grid)
            * len(protocol["confirmation_scope"]["candidate_order"]),
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
        if not state_complete(state_dir(root, str(row.state_id)), protocol)
    ]
    print(
        f"RUN hierarchy calibration-only amendment; states={len(grid)} "
        f"pending={len(pending)} shared_model_fits={len(grid)} "
        f"derived_results={len(grid) * 3} workers={args.workers}",
        flush=True,
    )
    completed = len(grid) - len(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                execute_state, root, code_root, python, row, protocol
            ): row
            for row in pending
        }
        for future in as_completed(futures):
            row = futures[future]
            future.result()
            completed += 1
            print(f"[{completed}/{len(grid)}] DONE {row['state_id']}", flush=True)
            gc.collect()

    runs, traits, guards, shared = collect_tables(root, grid, protocol)
    paired = pair_runs(runs)
    paired_traits = pair_traits(traits)
    paired_guards = pair_guards(guards)
    decision = summarize(protocol, paired, paired_traits, paired_guards)
    selected = select_candidate(decision, protocol)
    decision["decision"] = np.where(
        decision["candidate"].eq(REFERENCE),
        "stable_reference",
        np.where(
            decision["candidate"].eq(selected),
            "freeze_for_new_125_route_lock",
            "do_not_advance",
        ),
    )

    candidates = set(protocol["confirmation_scope"]["candidate_order"])
    candidate_runs = runs.loc[runs["candidate"].isin(candidates)].copy()
    test_weight = traits.loc[
        traits["candidate"].isin(candidates)
        & traits["trait_name_canonical"].eq("TEST_WEIGHT")
    ]
    candidate_guards = paired_guards.loc[paired_guards["candidate"].isin(candidates)]
    eligible_guard_rows = candidate_guards["rows"].ge(
        int(protocol["phase_1_acceptance"]["minimum_rows_for_guard"])
    )
    checks = {
        "source_failed_confirmation_terminal": lock.get(
            "source_failed_confirmation_terminal"
        )
        is True,
        "state_count_25": len(shared) == 25 and shared["state_id"].nunique() == 25,
        "one_shared_model_fit_per_state": shared["one_shared_model_fit"].eq(True).all(),
        "derived_candidate_runs_75": len(candidate_runs) == 75,
        "reference_runs_25": int(runs["candidate"].eq(REFERENCE).sum()) == 25,
        "identity_replay_every_state": shared["identity_replay_pass"].eq(True).all(),
        "non_test_weight_calibration_fixed": candidate_runs.groupby("state_id")[
            "non_test_weight_calibration_sha256"
        ].nunique().max()
        == 1,
        "shared_model_hash_fixed": candidate_runs.groupby("state_id")[
            "shared_model_fit_sha256"
        ].nunique().max()
        == 1,
        "matched_seeds": runs.groupby("state_id")["seed"].nunique().max() == 1,
        "matched_validation_observations": paired[
            "validation_observation_signature"
        ].eq(paired["validation_observation_signature_reference"]).all(),
        "positive_test_weight_slopes": test_weight["calibration_slope"].gt(0).all(),
        "corrected_guard_rows_exact": candidate_guards.loc[
            candidate_guards["rows"].gt(0), "rows"
        ].eq(candidate_guards.loc[candidate_guards["rows"].gt(0), "rows_reference"]).all(),
        "corrected_guard_identifiers_exact": candidate_guards.loc[
            candidate_guards["rows"].gt(0), "observation_id_signature"
        ].eq(
            candidate_guards.loc[
                candidate_guards["rows"].gt(0),
                "observation_id_signature_reference",
            ]
        ).all(),
        "guard_subset_schema_complete": len(candidate_guards) == 25 * 3 * 7
        and set(candidate_guards["subset"])
        == set(protocol["mandatory_reporting_subsets"]),
        "eligible_guard_metrics_finite": np.isfinite(
            candidate_guards.loc[
                eligible_guard_rows, ["relative_nrmse_gain", "pearson_gain"]
            ].to_numpy(dtype=float)
        ).all(),
        "outer_unread": runs["outer_test_metrics_read"].eq(False).all()
        and runs["outer_test_outcomes_read"].eq(False).all(),
        "final_holdout_unread": runs["final_holdout_outcomes_read"].eq(False).all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]

    artifacts = {
        "calibration_amendment_runs.tsv": runs,
        "calibration_amendment_shared_fits.tsv": shared,
        "calibration_amendment_trait_metrics.tsv": traits,
        "calibration_amendment_guard_metrics.tsv": guards,
        "calibration_amendment_paired_metrics.tsv": paired,
        "calibration_amendment_paired_trait_metrics.tsv": paired_traits,
        "calibration_amendment_paired_guard_metrics.tsv": paired_guards,
        "calibration_amendment_decision.tsv": decision,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output / name, sep="\t", index=False, lineterminator="\n")
    status = (
        "PASS_STAGE1_V2_PHASE6_CALIBRATION_AMENDMENT_CANDIDATE_SELECTED"
        if selected is not None
        else "PASS_STAGE1_V2_PHASE6_CALIBRATION_AMENDMENT_COMPLETE_NO_ADVANCE"
    )
    if failed:
        status = "FAIL_STAGE1_V2_PHASE6_CALIBRATION_AMENDMENT_INTEGRITY"
    final = {
        "status": status,
        "protocol_version": protocol["protocol_version"],
        "selection_data": "nested_inner_validation_only",
        "source_failed_confirmation_preserved": True,
        "selected_candidate": selected,
        "state_count": 25,
        "new_model_fit_count": 25,
        "derived_calibration_result_count": 75,
        "test_weight_calibration_threshold": float(
            protocol["phase_1_acceptance"]["maximum_absolute_macro_calibration_error"]
        ),
        "retrospective_threshold_change_performed": False,
        "route_freeze_allowed": selected is not None and not failed,
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
    write_json(output / "CALIBRATION_AMENDMENT_DECISION.json", final)
    write_json(output / "calibration_amendment_status.json", final)
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    if failed:
        raise SystemExit(f"Calibration amendment integrity failed: {failed}")


if __name__ == "__main__":
    main()
