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
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.v2 import run_stage1_v2_phase6_factor_analytic_screen as legacy_runner
from scripts.v2 import run_stage1_v2_phase6_trait_balance_screen as reporting
from scripts.v2.run_stage1_v2_phase6_phase1 import validate_runtime


PROTOCOL = Path(
    "server_training_pipeline/"
    "stage1_v2_phase6_factor_analytic_optimization_amendment_protocol_v1.json"
)
LOCK = Path(
    "audit/v2/stage1_v2_phase6_factor_analytic_optimization_amendment_v1/"
    "FA_OPTIMIZATION_AMENDMENT_LOCK.json"
)
SOURCE_RUNS = Path(
    "trained_models/stage1_v2_phase6_hierarchy_calibration_amendment_v2_runs"
)
RUNS = Path(
    "trained_models/"
    "stage1_v2_phase6_factor_analytic_optimization_amendment_v1_runs"
)
REPLAY_RUNS = Path(
    "trained_models/"
    "stage1_v2_phase6_factor_analytic_optimization_amendment_v1_same_seed_replay_runs"
)
OUTPUT = Path(
    "model_kernels/"
    "stage1_v2_phase6_factor_analytic_optimization_amendment_v1/phase_1"
)
TRAINER_MODULE = (
    "server_training_pipeline."
    "train_stage1_v2_phase6_factor_analytic_optimization_amendment_tf"
)
RUN_PROTOCOL = (
    "stage1_v2_phase6_normalized_direction_factor_analytic_optimization_tf_v1"
)
REFERENCE = "current_huber_authoritative_row_mass"
SOURCE_REFERENCE = "hierarchy_test_weight_environment_oof_huber_v2"
LOCK_STATUS = "PASS_FROZEN_STAGE1_V2_PHASE6_FA_OPTIMIZATION_AMENDMENT_V1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "only_mutable_component": "covariate_linked_factor_analytic_optimization",
        "normalized_direction_parameterization": True,
        "genotype_direction_L2_penalty": False,
        "environment_direction_L2_penalty": False,
        "trait_amplitude_loading_L2_penalty": True,
        "primary_macro_trait_count": 6,
        "primary_macro_excludes_TEST_WEIGHT": True,
        "training_likelihood_trait_count": 7,
        "TEST_WEIGHT_training_rows_retained": True,
        "TEST_WEIGHT_reporting_retained": True,
        "TEST_WEIGHT_calibration_retained": True,
        "FA_optimization_path_certified": True,
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
        "protocol": "protocol_sha256",
        "trainer": "trainer_sha256",
        "calibration_helper": "calibration_helper_sha256",
        "calibration_trainer": "calibration_trainer_sha256",
        "remediation_helper": "remediation_helper_sha256",
        "model_builder": "model_builder_sha256",
        "factor_builder": "factor_builder_sha256",
        "trainer_interface": "trainer_interface_sha256",
        "post_hierarchy_plan": "post_hierarchy_plan_sha256",
        "projection_protocol": "projection_protocol_sha256",
        "private_head_decision": "private_head_decision_sha256",
        "parent_factor_analytic_decision": (
            "parent_factor_analytic_decision_sha256"
        ),
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
    if int(metadata.get("TEST_WEIGHT_validation_rows", 0)) <= 0:
        return False
    if not metadata.get("runtime_thread_configuration_sha256"):
        return False
    artifacts = metadata.get("artifacts", {})
    if "component_activity_history.tsv" not in artifacts:
        return False
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
            f"FA optimization-amendment run failed: {state_id} {candidate}; "
            f"log={log_path}"
        )


def _primary_metric_summary(
    metadata: dict[str, object],
    traits: pd.DataFrame,
    primary_traits: Sequence[str],
) -> None:
    local = traits.loc[
        traits["trait_name_canonical"].isin(primary_traits)
    ].copy()
    if set(local["trait_name_canonical"].astype(str)) != set(primary_traits):
        raise ValueError("A six-trait primary macro is incomplete")
    all_traits = set(traits["trait_name_canonical"].astype(str))
    if all_traits != {*primary_traits, "TEST_WEIGHT"}:
        raise ValueError(f"Seven-trait reporting is incomplete: {sorted(all_traits)}")
    metadata.setdefault(
        "validation_all_seven_macro_normalized_rmse",
        metadata["validation_macro_normalized_rmse"],
    )
    metadata.setdefault(
        "validation_all_seven_macro_pearson",
        metadata["validation_macro_pearson"],
    )
    metadata.setdefault(
        "validation_all_seven_macro_calibration_error",
        metadata["validation_macro_calibration_error"],
    )
    metadata["validation_macro_normalized_rmse"] = float(
        local["normalized_rmse"].mean()
    )
    metadata["validation_macro_pearson"] = float(local["pearson"].mean())
    metadata["validation_macro_calibration_error"] = float(
        local["calibration_error"].mean()
    )
    metadata["primary_macro_trait_count"] = 6
    metadata["primary_macro_excludes_TEST_WEIGHT"] = True
    metadata["training_likelihood_trait_count"] = 7
    metadata["TEST_WEIGHT_reporting_retained"] = True


def collect_tables(
    root: Path, grid: pd.DataFrame, protocol: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, object]] = []
    trait_frames: list[pd.DataFrame] = []
    guard_frames: list[pd.DataFrame] = []
    activity_rows: list[dict[str, object]] = []
    candidates = [
        name
        for name, value in protocol["candidates"].items()
        if value.get("source_reuse") is False
    ]
    primary_traits = protocol["objective_policy"]["primary_macro_traits"]
    for row in grid.itertuples(index=False):
        state_id = str(row.state_id)
        scenario = str(row.scenario)
        source = root / SOURCE_RUNS / state_id / SOURCE_REFERENCE
        metadata, traits, guards = reporting._load_metrics(
            source, state_id, scenario, REFERENCE
        )
        _primary_metric_summary(metadata, traits, primary_traits)
        metadata["source_reuse"] = True
        metadata["source_candidate"] = SOURCE_REFERENCE
        run_rows.append(metadata)
        trait_frames.append(traits)
        guard_frames.append(guards)
        activity_rows.append(
            {
                "state_id": state_id,
                "candidate": REFERENCE,
                "source_reuse": True,
                "FA_optimization_path_certified": True,
                "FA_final_component_active": True,
            }
        )
        for candidate in candidates:
            directory = run_dir(root, state_id, candidate)
            metadata, traits, guards = reporting._load_metrics(
                directory, state_id, scenario, candidate
            )
            _primary_metric_summary(metadata, traits, primary_traits)
            metadata["source_reuse"] = False
            run_rows.append(metadata)
            trait_frames.append(traits)
            guard_frames.append(guards)
            activity_rows.append(
                {
                    "state_id": state_id,
                    "candidate": candidate,
                    "source_reuse": False,
                    "FA_optimization_path_certified": bool(
                        metadata["FA_optimization_path_certified"]
                    ),
                    "FA_final_component_active": bool(
                        metadata["FA_final_component_active"]
                    ),
                    "FA_selected_training_residual_rms": metadata[
                        "FA_selected_training_residual_rms"
                    ],
                    "FA_selected_validation_residual_rms": metadata[
                        "FA_selected_validation_residual_rms"
                    ],
                    "FA_minimum_raw_genotype_direction_norm": metadata[
                        "FA_minimum_raw_genotype_direction_norm"
                    ],
                    "FA_minimum_raw_environment_direction_norm": metadata[
                        "FA_minimum_raw_environment_direction_norm"
                    ],
                    "FA_minimum_trait_amplitude_norm": metadata[
                        "FA_minimum_trait_amplitude_norm"
                    ],
                    "FA_maximum_observed_genotype_gradient_norm": metadata[
                        "FA_maximum_observed_genotype_gradient_norm"
                    ],
                    "FA_maximum_observed_environment_gradient_norm": metadata[
                        "FA_maximum_observed_environment_gradient_norm"
                    ],
                    "FA_maximum_observed_trait_amplitude_gradient_norm": metadata[
                        "FA_maximum_observed_trait_amplitude_gradient_norm"
                    ],
                }
            )
    return (
        pd.DataFrame(run_rows),
        pd.concat(trait_frames, ignore_index=True),
        pd.concat(guard_frames, ignore_index=True),
        pd.DataFrame(activity_rows),
    )


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
        raise ValueError(f"Replay state is absent or ambiguous: {state_id}")
    row = matches.iloc[0].to_dict()
    seed = int(row["seed"])
    rows: list[dict[str, object]] = []
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
            "validation_all_seven_macro_normalized_rmse",
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
                    "PASS"
                    if exact and metric_delta == 0.0 and same_runtime
                    else "FAIL"
                ),
            }
        )
    return pd.DataFrame(rows)


def _activity_histories_pass(root: Path, runs: pd.DataFrame) -> bool:
    candidates = runs.loc[~runs["candidate"].eq(REFERENCE)]
    required_columns = {
        "epoch",
        "record_type",
        "training_FA_residual_rms",
        "validation_FA_residual_rms",
        "genotype_gradient_norm",
        "environment_gradient_norm",
        "trait_amplitude_gradient_norm",
        "genotype_raw_direction_norm_min",
        "environment_raw_direction_norm_min",
        "trait_amplitude_norm_min",
    }
    for row in candidates.itertuples(index=False):
        path = (
            run_dir(root, str(row.state_id), str(row.candidate))
            / "component_activity_history.tsv"
        )
        history = pd.read_csv(path, sep="\t")
        if not required_columns.issubset(history.columns):
            return False
        counts = history["record_type"].value_counts()
        if counts.get("initialization", 0) != 1:
            return False
        if counts.get("selected_checkpoint", 0) != 1:
            return False
        if counts.get("training_epoch", 0) < 1:
            return False
        numeric = history.loc[
            history["record_type"].eq("training_epoch"),
            [
                "training_FA_residual_rms",
                "genotype_gradient_norm",
                "environment_gradient_norm",
                "trait_amplitude_gradient_norm",
                "genotype_raw_direction_norm_min",
                "environment_raw_direction_norm_min",
                "trait_amplitude_norm_min",
            ],
        ].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Stage-1 v2 normalized-direction FA amendment"
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
    if lock.get("status") != LOCK_STATUS:
        raise ValueError("FA optimization amendment is not frozen")
    implementation = {
        "protocol": PROTOCOL,
        "trainer": Path(
            "server_training_pipeline/"
            "train_stage1_v2_phase6_factor_analytic_optimization_amendment_tf.py"
        ),
        "runner": Path(
            "scripts/v2/"
            "run_stage1_v2_phase6_factor_analytic_optimization_amendment.py"
        ),
        "parent_factor_analytic_protocol": Path(
            "server_training_pipeline/"
            "stage1_v2_phase6_factor_analytic_screen_protocol_v1.json"
        ),
        "post_hierarchy_plan": Path(
            "server_training_pipeline/stage1_v2_phase6_post_hierarchy_screen_plan_v2.json"
        ),
        "projection_protocol": Path(
            "server_training_pipeline/phase6a_split_bound_projection_inputs_protocol_v1.json"
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
        "factor_builder": Path(
            "server_training_pipeline/train_stage1_v2_phase6_tf.py"
        ),
        "trainer_interface": Path(
            "server_training_pipeline/stage1_v2_trainer_interface.py"
        ),
    }
    for name, relative in implementation.items():
        if sha256_file(code_root / relative) != lock["artifacts"].get(name):
            raise ValueError(f"Frozen FA amendment implementation drift: {name}")
    parent = root / (
        "model_kernels/stage1_v2_phase6_factor_analytic_screen_v1/phase_1/"
        "FACTOR_ANALYTIC_PHASE1_DECISION.json"
    )
    if sha256_file(parent) != lock["artifacts"]["parent_factor_analytic_decision"]:
        raise ValueError("Parent FA decision drift")

    grid = legacy_runner.build_grid(root, protocol)
    candidates = [
        name
        for name, value in protocol["candidates"].items()
        if value.get("source_reuse") is False
    ]
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(
        output / "fa_optimization_amendment_state_grid.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    write_json(
        output / "fa_optimization_amendment_status.json",
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
            "primary_macro_trait_count": 6,
            "TEST_WEIGHT_retained_outside_primary_macro": True,
            "runtime": runtime,
            "outer_test_metrics_read": False,
            "final_holdout_outcomes_read": False,
        },
    )
    os.environ["STAGE1_V2_EXECUTION_BACKEND"] = args.runtime_mode
    os.environ["STAGE1_V2_INTRA_OP_THREADS"] = str(args.threads_per_worker)
    os.environ["STAGE1_V2_INTER_OP_THREADS"] = str(args.inter_op_threads)
    print("PREWARM expected-state-validated historical/projection factors", flush=True)
    prewarm = legacy_runner.prewarm_factor_caches(root, grid, protocol)
    prewarm.to_csv(
        output / "fa_optimization_amendment_factor_cache_prewarm.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    pending: list[tuple[dict[str, object], str]] = []
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
        f"RUN Stage-1 v2 FA optimization amendment; states={len(grid)} "
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
    runs, traits, guards, activity = collect_tables(root, grid, protocol)
    paired = reporting.pair_runs(runs)
    paired_traits = reporting.pair_traits(traits)
    paired_guards = reporting.pair_guards(guards)
    decision = reporting.summarize(protocol, paired, paired_traits, paired_guards)
    activity_decision = activity.groupby("candidate", as_index=False).agg(
        guard_FA_optimization_path=("FA_optimization_path_certified", "all"),
        guard_FA_final_component_active=("FA_final_component_active", "all"),
    )
    decision = decision.merge(
        activity_decision, on="candidate", how="left", validate="one_to_one"
    )
    decision["eligible_for_confirmation"] = (
        decision["eligible_for_confirmation"]
        & decision["guard_FA_optimization_path"]
        & decision["guard_FA_final_component_active"]
    )
    selected = reporting.select_candidate(decision, protocol)
    decision["decision"] = np.where(
        decision["candidate"].eq(REFERENCE),
        "stable_corrected_huber_reference",
        np.where(
            ~decision["guard_FA_optimization_path"],
            "invalid_optimization_path",
            np.where(
                ~decision["guard_FA_final_component_active"],
                "valid_inactive_component_do_not_advance",
                np.where(
                    decision["candidate"].eq(selected),
                    "advance_to_25_state_confirmation",
                    "active_component_do_not_advance",
                ),
            ),
        ),
    )

    candidate_runs = runs.loc[~runs["candidate"].eq(REFERENCE)]
    candidate_traits = traits.loc[~traits["candidate"].eq(REFERENCE)]
    candidate_guards = paired_guards.loc[~paired_guards["candidate"].eq(REFERENCE)]
    expected_traits = set(protocol["objective_policy"]["training_likelihood_traits"])
    trait_sets = candidate_traits.groupby(["state_id", "candidate"])[
        "trait_name_canonical"
    ].agg(lambda values: set(values.astype(str)))
    checks = {
        "state_count_5": grid["state_id"].nunique() == 5,
        "result_count_15": len(runs) == 15,
        "reference_reuse_count_5": int(runs["candidate"].eq(REFERENCE).sum()) == 5,
        "new_fit_count_10": len(candidate_runs) == 10
        and candidate_runs["model_training_performed"].eq(True).all(),
        "FA_ranks_exact": set(
            candidate_runs["factor_analytic_rank"].dropna().astype(int)
        )
        == {2, 4},
        "normalized_direction_parameterization_active": candidate_runs[
            "normalized_direction_parameterization"
        ].eq(True).all(),
        "direction_L2_disabled": candidate_runs[
            "genotype_direction_L2_penalty"
        ].eq(False).all()
        and candidate_runs["environment_direction_L2_penalty"].eq(False).all(),
        "trait_amplitude_L2_active": candidate_runs[
            "trait_amplitude_loading_L2_penalty"
        ].eq(True).all(),
        "FA_optimization_paths_certified": candidate_runs[
            "FA_optimization_path_certified"
        ].eq(True).all(),
        "activity_history_complete": _activity_histories_pass(root, runs),
        "all_seven_traits_reported": trait_sets.map(
            lambda values: values == expected_traits
        ).all(),
        "six_trait_primary_macro_exact": runs[
            "primary_macro_trait_count"
        ].eq(6).all()
        and runs["primary_macro_excludes_TEST_WEIGHT"].eq(True).all(),
        "TEST_WEIGHT_retained": candidate_runs[
            "TEST_WEIGHT_training_rows_retained"
        ].eq(True).all()
        and candidate_runs["TEST_WEIGHT_validation_rows"].gt(0).all()
        and candidate_runs["TEST_WEIGHT_reporting_retained"].eq(True).all()
        and candidate_runs["TEST_WEIGHT_calibration_retained"].eq(True).all(),
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
        "fa_optimization_amendment_runs.tsv": runs,
        "fa_optimization_amendment_trait_metrics.tsv": traits,
        "fa_optimization_amendment_guard_metrics.tsv": guards,
        "fa_optimization_amendment_activity.tsv": activity,
        "fa_optimization_amendment_paired_metrics.tsv": paired,
        "fa_optimization_amendment_paired_trait_metrics.tsv": paired_traits,
        "fa_optimization_amendment_paired_guard_metrics.tsv": paired_guards,
        "fa_optimization_amendment_decision.tsv": decision,
        "fa_optimization_amendment_same_seed_replay.tsv": replay,
        "fa_optimization_amendment_factor_cache_prewarm.tsv": prewarm,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output / name, sep="\t", index=False, lineterminator="\n")
    status = (
        "PASS_STAGE1_V2_PHASE6_FA_OPTIMIZATION_AMENDMENT_CANDIDATE_SELECTED"
        if selected is not None
        else "PASS_STAGE1_V2_PHASE6_FA_OPTIMIZATION_AMENDMENT_COMPLETE_NO_ADVANCE"
    )
    if failed:
        status = "FAIL_STAGE1_V2_PHASE6_FA_OPTIMIZATION_AMENDMENT_INTEGRITY"
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
        "primary_macro_traits": protocol["objective_policy"][
            "primary_macro_traits"
        ],
        "TEST_WEIGHT_retained_outside_primary_macro": True,
        "full_confirmation_allowed": selected is not None and not failed,
        "outer_evaluation_allowed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "future_SSP_values_read": False,
        "future_predictions_generated": 0,
        "checks": checks,
        "failed_checks": failed,
        "artifacts": {
            name: sha256_file(output / name) for name in artifacts
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "FA_OPTIMIZATION_AMENDMENT_DECISION.json", result)
    write_json(output / "fa_optimization_amendment_status.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if failed:
        raise SystemExit(f"FA optimization-amendment integrity failed: {failed}")


if __name__ == "__main__":
    main()
