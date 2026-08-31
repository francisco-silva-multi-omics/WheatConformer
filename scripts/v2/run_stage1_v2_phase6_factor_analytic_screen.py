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

from scripts.v2 import run_stage1_v2_phase6_trait_balance_screen as reporting
from scripts.v2.run_stage1_v2_phase6_phase1 import validate_runtime
from server_training_pipeline.stage1_v2_trainer_interface import PARITY


PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_factor_analytic_screen_protocol_v1.json"
)
LOCK = Path(
    "audit/v2/stage1_v2_phase6_factor_analytic_screen_v1/"
    "FACTOR_ANALYTIC_SCREEN_LOCK.json"
)
SOURCE_RUNS = Path(
    "trained_models/stage1_v2_phase6_hierarchy_calibration_amendment_v2_runs"
)
RUNS = Path("trained_models/stage1_v2_phase6_factor_analytic_screen_v1_runs")
REPLAY_RUNS = Path(
    "trained_models/stage1_v2_phase6_factor_analytic_screen_v1_same_seed_replay_runs"
)
OUTPUT = Path("model_kernels/stage1_v2_phase6_factor_analytic_screen_v1/phase_1")
TRAINER_MODULE = "server_training_pipeline.train_stage1_v2_phase6_factor_analytic_tf"
RUN_PROTOCOL = "stage1_v2_phase6_covariate_linked_factor_analytic_tf_v1"
REFERENCE = "current_huber_authoritative_row_mass"
SOURCE_REFERENCE = "hierarchy_test_weight_environment_oof_huber_v2"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_grid(root: Path, protocol: dict[str, Any]) -> pd.DataFrame:
    registry = pd.read_csv(
        root / PARITY / "splits/state_registry.tsv", sep="\t", dtype=str
    )
    scope = protocol["phase_1_scope"]
    outer_fold = pd.to_numeric(registry["outer_fold"], errors="coerce")
    inner_fold = pd.to_numeric(registry["inner_fold"], errors="coerce")
    states = registry.loc[
        registry["state_level"].eq("INNER")
        & registry["scenario"].eq("GNEW_EOBS")
        & inner_fold.eq(int(scope["inner_fold"]))
        & outer_fold.isin(scope["outer_folds"])
    ].copy()
    states["outer_fold"] = outer_fold.loc[states.index].astype(int)
    states["inner_fold"] = inner_fold.loc[states.index].astype(int)
    states = states.sort_values(["outer_fold", "inner_fold", "state_id"])
    if len(states) != int(scope["state_count"]) or not states["state_id"].is_unique:
        raise ValueError(f"Expected five factor-analytic states; observed={len(states)}")
    seeds = []
    for state_id in states["state_id"]:
        source = root / SOURCE_RUNS / state_id / SOURCE_REFERENCE / "run_metadata.json"
        if not source.is_file():
            raise FileNotFoundError(f"FA reference is missing: {source}")
        metadata = read_json(source)
        if metadata.get("status") != "PASS":
            raise ValueError(f"FA reference is not certified: {state_id}")
        seeds.append(int(metadata["seed"]))
    states["seed"] = seeds
    return states[["state_id", "scenario", "outer_fold", "inner_fold", "seed"]]


def run_dir(root: Path, state_id: str, candidate: str) -> Path:
    return root / RUNS / state_id / candidate


def run_complete(
    path: Path,
    state_id: str,
    candidate: str,
    seed: int,
    lock: dict[str, Any],
) -> bool:
    metadata_path = path / "run_metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = read_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return False
    required = {
        "status": "PASS",
        "protocol_version": RUN_PROTOCOL,
        "state_id": state_id,
        "candidate": candidate,
        "seed": seed,
        "only_mutable_component": "covariate_linked_factor_analytic_residual",
        "authoritative_row_mass_changed": False,
        "historical_backbone_changed": False,
        "hierarchy_changed": False,
        "free_environment_loadings": False,
        "projection_feature_count": 153,
        "tensorflow_op_determinism_enabled": True,
        "deterministic_tf_data_order": True,
        "per_batch_finite_assertions_enabled": True,
        "factor_cache_expected_state_validation": True,
        "checkpoint_and_prediction_replay_artifacts_persisted": True,
        "file_access_attestation_scope": "controlled_process_only",
        "os_level_complete_file_open_audit_performed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "future_SSP_values_read": False,
        "future_covariate_matrices_used": 0,
        "future_predictions_generated": 0,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        return False
    implementation_fields = {
        "factor_analytic_protocol": "protocol_sha256",
        "trainer": "trainer_sha256",
        "calibration_helper": "calibration_helper_sha256",
        "calibration_trainer": "calibration_trainer_sha256",
        "remediation_helper": "remediation_helper_sha256",
        "model_builder": "model_builder_sha256",
        "factor_builder": "factor_builder_sha256",
        "trainer_interface": "trainer_interface_sha256",
        "post_hierarchy_plan": "post_hierarchy_plan_sha256",
        "projection_protocol": "projection_protocol_sha256",
        "parent_private_head_decision": "private_head_decision_sha256",
    }
    if any(
        metadata.get(metadata_key) != lock["artifacts"].get(lock_key)
        for lock_key, metadata_key in implementation_fields.items()
    ):
        return False
    if int(metadata.get("factor_analytic_rank", 0)) not in {2, 4}:
        return False
    if int(metadata.get("factor_analytic_variable_count", 0)) != 3:
        return False
    if int(metadata.get("factor_analytic_parameter_count", 0)) <= 0:
        return False
    if int(metadata.get("factor_analytic_active_training_rows", 0)) <= 0:
        return False
    if int(metadata.get("factor_analytic_active_validation_rows", 0)) <= 0:
        return False
    if not metadata.get("runtime_thread_configuration_sha256"):
        return False
    artifacts = metadata.get("artifacts", {})
    return bool(artifacts) and all(
        (path / name).is_file() and sha256_file(path / name) == expected
        for name, expected in artifacts.items()
    )


def execute_run(
    root: Path,
    code_root: Path,
    python: Path,
    row: dict[str, object],
    candidate: str,
    lock: dict[str, Any],
    output_override: Path | None = None,
) -> None:
    state_id = str(row["state_id"])
    seed = int(row["seed"])
    output = output_override or run_dir(root, state_id, candidate)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    command = [
        str(python),
        "-m",
        TRAINER_MODULE,
        "--root",
        str(root),
        "--state-id",
        state_id,
        "--candidate",
        candidate,
        "--seed",
        str(seed),
        "--out-dir",
        str(output),
    ]
    environment = os.environ.copy()
    environment["WHEATCONFORMER_CODE_ROOT"] = str(code_root)
    environment["PYTHONHASHSEED"] = str(seed)
    environment["TF_DETERMINISTIC_OPS"] = "1"
    environment["TF_CUDNN_DETERMINISTIC"] = "1"
    environment["PYTHONPATH"] = str(code_root) + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=code_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0 or not run_complete(
        output, state_id, candidate, seed, lock
    ):
        raise RuntimeError(
            f"Factor-analytic run failed: {state_id} {candidate}; log={log_path}"
        )


def prewarm_factor_caches(
    root: Path, grid: pd.DataFrame, protocol: dict[str, Any]
) -> pd.DataFrame:
    from server_training_pipeline.train_stage1_v2_phase6_confirmation_tf import (
        build_confirmation_factors,
    )
    from server_training_pipeline.train_stage1_v2_phase6_tf import (
        build_projection_environment,
    )

    configuration = dict(protocol["fixed_configuration"])
    configuration.pop("label", None)
    rows: list[dict[str, object]] = []
    for state_id in grid["state_id"].astype(str):
        genotype, environment, _, _, historical_ids, _ = build_confirmation_factors(
            root, state_id, "historical_reaction_reference", configuration
        )
        projection, features, active, projection_ids = build_projection_environment(
            root, state_id
        )
        if features.shape[1] != 153:
            raise ValueError(f"Projection schema differs from 153: {state_id}")
        if not np.array_equal(historical_ids.astype(str), projection_ids.astype(str)):
            raise ValueError(f"Historical/projection axes disagree: {state_id}")
        for block in (*genotype, *environment, *projection):
            rows.append(
                {
                    "state_id": state_id,
                    "component": block.name,
                    "axis": block.axis,
                    "rank": block.values.shape[1],
                    "state_hash": block.state_hash,
                    "status": "PASS",
                }
            )
        rows.append(
            {
                "state_id": state_id,
                "component": "E_PROJECTION_CORE_V1_STANDARDIZED_FEATURES",
                "axis": "environment",
                "rank": features.shape[1],
                "state_hash": hashlib.sha256(
                    np.asarray(features, dtype="<f4").tobytes(order="C")
                ).hexdigest(),
                "active_entities": int(active.sum()),
                "status": "PASS",
            }
        )
    return pd.DataFrame(rows)


def run_same_seed_replays(
    root: Path,
    code_root: Path,
    python: Path,
    grid: pd.DataFrame,
    protocol: dict[str, Any],
    lock: dict[str, Any],
) -> pd.DataFrame:
    contract = protocol["integrity_hardening"]["same_seed_replay"]
    state_id = str(contract["state_id"])
    matches = grid.loc[grid["state_id"].eq(state_id)]
    if len(matches) != 1:
        raise ValueError(f"Same-seed replay state is absent or ambiguous: {state_id}")
    row = matches.iloc[0].to_dict()
    seed = int(row["seed"])
    rows = []
    for candidate in contract["candidates"]:
        source = run_dir(root, state_id, candidate)
        replay = root / REPLAY_RUNS / state_id / candidate
        if not run_complete(replay, state_id, candidate, seed, lock):
            execute_run(
                root,
                code_root,
                python,
                row,
                candidate,
                lock,
                output_override=replay,
            )
        source_metadata = read_json(source / "run_metadata.json")
        replay_metadata = read_json(replay / "run_metadata.json")
        required = list(contract["required_exact_artifacts"])
        exact = all(
            source_metadata["artifacts"].get(name)
            == replay_metadata["artifacts"].get(name)
            for name in required
        )
        metric_keys = [
            "validation_macro_normalized_rmse",
            "validation_macro_pearson",
            "validation_macro_calibration_error",
        ]
        metric_delta = max(
            abs(float(source_metadata[key]) - float(replay_metadata[key]))
            for key in metric_keys
        )
        same_runtime = source_metadata.get(
            "runtime_thread_configuration_sha256"
        ) == replay_metadata.get("runtime_thread_configuration_sha256")
        rows.append(
            {
                "state_id": state_id,
                "candidate": candidate,
                "seed": seed,
                "required_artifact_count": len(required),
                "required_artifacts_exact": exact,
                "maximum_metric_absolute_delta": metric_delta,
                "same_runtime_and_thread_configuration": same_runtime,
                "status": (
                    "PASS" if exact and metric_delta == 0.0 and same_runtime else "FAIL"
                ),
            }
        )
    return pd.DataFrame(rows)


def collect_tables(
    root: Path, grid: pd.DataFrame, protocol: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, object]] = []
    trait_frames = []
    guard_frames = []
    candidates = [
        name
        for name, value in protocol["candidates"].items()
        if value.get("source_reuse") is False
    ]
    for row in grid.itertuples(index=False):
        state_id = str(row.state_id)
        scenario = str(row.scenario)
        source = root / SOURCE_RUNS / state_id / SOURCE_REFERENCE
        metadata, traits, guards = reporting._load_metrics(
            source, state_id, scenario, REFERENCE
        )
        metadata["source_reuse"] = True
        metadata["source_candidate"] = SOURCE_REFERENCE
        run_rows.append(metadata)
        trait_frames.append(traits)
        guard_frames.append(guards)
        for candidate in candidates:
            metadata, traits, guards = reporting._load_metrics(
                run_dir(root, state_id, candidate), state_id, scenario, candidate
            )
            metadata["source_reuse"] = False
            run_rows.append(metadata)
            trait_frames.append(traits)
            guard_frames.append(guards)
    return (
        pd.DataFrame(run_rows),
        pd.concat(trait_frames, ignore_index=True),
        pd.concat(guard_frames, ignore_index=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Stage-1 v2 covariate-linked FA Phase-1 screen"
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
    if lock.get("status") != (
        "PASS_FROZEN_STAGE1_V2_PHASE6_FACTOR_ANALYTIC_SCREEN_V1"
    ):
        raise ValueError("Factor-analytic screen is not frozen")
    implementation = {
        "factor_analytic_protocol": PROTOCOL,
        "post_hierarchy_plan": Path(
            "server_training_pipeline/stage1_v2_phase6_post_hierarchy_screen_plan_v2.json"
        ),
        "projection_protocol": Path(
            "server_training_pipeline/phase6a_split_bound_projection_inputs_protocol_v1.json"
        ),
        "trainer": Path(
            "server_training_pipeline/train_stage1_v2_phase6_factor_analytic_tf.py"
        ),
        "calibration_helper": Path(
            "server_training_pipeline/stage1_v2_phase6_hierarchy_calibration_amendment_v2.py"
        ),
        "calibration_trainer": Path(
            "server_training_pipeline/"
            "train_stage1_v2_phase6_hierarchy_calibration_amendment_tf.py"
        ),
        "remediation_helper": Path(
            "server_training_pipeline/stage1_v2_phase6_remediation.py"
        ),
        "model_builder": Path(
            "server_training_pipeline/train_stage1_v2_phase6_remediation_tf.py"
        ),
        "factor_builder": Path("server_training_pipeline/train_stage1_v2_phase6_tf.py"),
        "trainer_interface": Path("server_training_pipeline/stage1_v2_trainer_interface.py"),
        "runner": Path("scripts/v2/run_stage1_v2_phase6_factor_analytic_screen.py"),
    }
    for name, relative in implementation.items():
        if sha256_file(code_root / relative) != lock["artifacts"].get(name):
            raise ValueError(f"Factor-analytic frozen implementation drift: {name}")
    parent = (
        root
        / "model_kernels/stage1_v2_phase6_private_head_screen_v1/phase_1/"
        "PRIVATE_HEAD_PHASE1_DECISION.json"
    )
    if sha256_file(parent) != lock["artifacts"]["parent_private_head_decision"]:
        raise ValueError("Factor-analytic parent decision drift")

    grid = build_grid(root, protocol)
    candidates = [
        name
        for name, value in protocol["candidates"].items()
        if value.get("source_reuse") is False
    ]
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(
        output / "factor_analytic_state_grid.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    write_json(
        output / "factor_analytic_status.json",
        {
            "status": "RUNNING",
            "protocol_version": protocol["protocol_version"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "state_count": len(grid),
            "reference_reuse_count": len(grid),
            "new_model_fit_count": len(grid) * len(candidates),
            "same_seed_replay_fit_count": 2,
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
    print("PREWARM historical and split-bound projection factors", flush=True)
    prewarm = prewarm_factor_caches(root, grid, protocol)
    prewarm.to_csv(
        output / "factor_analytic_factor_cache_prewarm.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    pending = []
    for row in grid.to_dict(orient="records"):
        for candidate in candidates:
            if not run_complete(
                run_dir(root, str(row["state_id"]), candidate),
                str(row["state_id"]),
                candidate,
                int(row["seed"]),
                lock,
            ):
                pending.append((row, candidate))
    total = len(grid) * len(candidates)
    print(
        f"RUN Stage-1 v2 covariate-linked FA phase_1; states={len(grid)} "
        f"new_fits={total} pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    complete = total - len(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                execute_run, root, code_root, python, row, candidate, lock
            ): (row, candidate)
            for row, candidate in pending
        }
        for future in as_completed(futures):
            row, candidate = futures[future]
            future.result()
            complete += 1
            print(f"[{complete}/{total}] DONE {row['state_id']} {candidate}", flush=True)
            gc.collect()

    replay = run_same_seed_replays(root, code_root, python, grid, protocol, lock)
    replay.to_csv(
        output / "factor_analytic_same_seed_replay.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    runs, traits, guards = collect_tables(root, grid, protocol)
    paired = reporting.pair_runs(runs)
    paired_traits = reporting.pair_traits(traits)
    paired_guards = reporting.pair_guards(guards)
    decision = reporting.summarize(protocol, paired, paired_traits, paired_guards)
    selected = reporting.select_candidate(decision, protocol)
    decision["decision"] = np.where(
        decision["candidate"].eq(REFERENCE),
        "stable_corrected_huber_reference",
        np.where(
            decision["candidate"].eq(selected),
            "advance_to_25_state_confirmation",
            "do_not_advance",
        ),
    )
    candidate_runs = runs.loc[~runs["candidate"].eq(REFERENCE)]
    candidate_guards = paired_guards.loc[~paired_guards["candidate"].eq(REFERENCE)]
    checks = {
        "state_count_5": grid["state_id"].nunique() == 5,
        "result_count_15": len(runs) == 15,
        "reference_reuse_count_5": int(runs["candidate"].eq(REFERENCE).sum()) == 5,
        "new_fit_count_10": len(candidate_runs) == 10
        and candidate_runs["model_training_performed"].eq(True).all(),
        "FA_ranks_exact": set(candidate_runs["factor_analytic_rank"].dropna().astype(int))
        == {2, 4},
        "FA_parameters_active": candidate_runs[
            "factor_analytic_parameter_count"
        ].gt(0).all()
        and candidate_runs["factor_analytic_variable_count"].eq(3).all(),
        "projection_features_active": candidate_runs[
            "factor_analytic_active_training_rows"
        ].gt(0).all()
        and candidate_runs["factor_analytic_active_validation_rows"].gt(0).all(),
        "free_environment_loadings_absent": candidate_runs[
            "free_environment_loadings"
        ].eq(False).all(),
        "fixed_architecture_unchanged": candidate_runs[
            "historical_backbone_changed"
        ].eq(False).all()
        and candidate_runs["hierarchy_changed"].eq(False).all()
        and candidate_runs["authoritative_row_mass_changed"].eq(False).all(),
        "matched_seeds": runs.groupby("state_id")["seed"].nunique().max() == 1,
        "matched_validation_observations": paired[
            "validation_observation_signature"
        ].eq(paired["validation_observation_signature_reference"]).all(),
        "guard_rows_exact": candidate_guards.loc[
            candidate_guards["rows"].gt(0), "rows"
        ].eq(
            candidate_guards.loc[
                candidate_guards["rows"].gt(0), "rows_reference"
            ]
        ).all(),
        "guard_identifiers_exact": candidate_guards.loc[
            candidate_guards["rows"].gt(0), "observation_id_signature"
        ].eq(
            candidate_guards.loc[
                candidate_guards["rows"].gt(0),
                "observation_id_signature_reference",
            ]
        ).all(),
        "outer_unread": runs["outer_test_metrics_read"].eq(False).all()
        and runs["outer_test_outcomes_read"].eq(False).all(),
        "final_holdout_unread": runs["final_holdout_outcomes_read"].eq(False).all(),
        "future_values_unused": candidate_runs["future_SSP_values_read"].eq(False).all()
        and candidate_runs["future_covariate_matrices_used"].eq(0).all()
        and candidate_runs["future_predictions_generated"].eq(0).all(),
        "tensorflow_determinism_enabled": candidate_runs[
            "tensorflow_op_determinism_enabled"
        ].eq(True).all(),
        "finite_assertions_enabled": candidate_runs[
            "per_batch_finite_assertions_enabled"
        ].eq(True).all(),
        "factor_cache_prewarm_complete": prewarm["status"].eq("PASS").all()
        and prewarm["state_id"].nunique() == 5,
        "same_seed_replay_count_2": len(replay) == 2,
        "same_seed_replay_exact": replay["status"].eq("PASS").all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    artifacts = {
        "factor_analytic_runs.tsv": runs,
        "factor_analytic_trait_metrics.tsv": traits,
        "factor_analytic_guard_metrics.tsv": guards,
        "factor_analytic_paired_metrics.tsv": paired,
        "factor_analytic_paired_trait_metrics.tsv": paired_traits,
        "factor_analytic_paired_guard_metrics.tsv": paired_guards,
        "factor_analytic_decision.tsv": decision,
        "factor_analytic_same_seed_replay.tsv": replay,
        "factor_analytic_factor_cache_prewarm.tsv": prewarm,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output / name, sep="\t", index=False, lineterminator="\n")
    status = (
        "PASS_STAGE1_V2_PHASE6_FACTOR_ANALYTIC_PHASE1_CANDIDATE_SELECTED"
        if selected is not None
        else "PASS_STAGE1_V2_PHASE6_FACTOR_ANALYTIC_PHASE1_COMPLETE_NO_ADVANCE"
    )
    if failed:
        status = "FAIL_STAGE1_V2_PHASE6_FACTOR_ANALYTIC_PHASE1_INTEGRITY"
    result = {
        "status": status,
        "protocol_version": protocol["protocol_version"],
        "selection_data": "nested_inner_validation_only",
        "selected_candidate": selected,
        "reference_candidate": REFERENCE,
        "state_count": 5,
        "reference_reuse_count": 5,
        "new_model_fit_count": 10,
        "same_seed_replay_fit_count": 2,
        "same_seed_replay_exact": bool(replay["status"].eq("PASS").all()),
        "full_confirmation_allowed": selected is not None and not failed,
        "outer_evaluation_allowed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "future_SSP_values_read": False,
        "future_predictions_generated": 0,
        "checks": checks,
        "failed_checks": failed,
        "artifacts": {name: sha256_file(output / name) for name in artifacts},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "FACTOR_ANALYTIC_PHASE1_DECISION.json", result)
    write_json(output / "factor_analytic_status.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if failed:
        raise SystemExit(f"Factor-analytic phase_1 integrity failed: {failed}")


if __name__ == "__main__":
    main()
