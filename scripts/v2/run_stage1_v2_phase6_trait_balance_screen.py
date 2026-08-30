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
from server_training_pipeline.stage1_v2_trainer_interface import PARITY


PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_trait_balance_screen_protocol_v1.json"
)
LOCK = Path(
    "audit/v2/stage1_v2_phase6_trait_balance_screen_v1/TRAIT_BALANCE_SCREEN_LOCK.json"
)
SOURCE_RUNS = Path(
    "trained_models/stage1_v2_phase6_hierarchy_calibration_amendment_v2_runs"
)
RUNS = Path("trained_models/stage1_v2_phase6_trait_balance_screen_v1_runs")
OUTPUT = Path("model_kernels/stage1_v2_phase6_trait_balance_screen_v1/phase_1")
TRAINER_MODULE = "server_training_pipeline.train_stage1_v2_phase6_trait_balance_tf"
RUN_PROTOCOL = "stage1_v2_phase6_trait_balance_tf_v1"
REFERENCE = "current_huber_authoritative_row_mass"
SOURCE_REFERENCE = "hierarchy_test_weight_environment_oof_huber_v2"
MASK_CANDIDATE = "marker_supported_output_routed_v2"
INFORMATION_SUBSETS = {
    "PEDIGREE_ONLY",
    "MARKER_SUPPORTED",
    "PEDIGREE_AND_MARKER",
    "NEITHER_PRODUCTION_PEDIGREE_NOR_DENSE_MARKERS",
    "RECOVERED_IDENTITY_OR_COMPONENT",
}


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
    registry = pd.read_csv(root / PARITY / "splits/state_registry.tsv", sep="\t", dtype=str)
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
        raise ValueError(f"Expected five trait-balance states; observed={len(states)}")
    seeds = []
    for state_id in states["state_id"]:
        source = root / SOURCE_RUNS / state_id / SOURCE_REFERENCE / "run_metadata.json"
        if not source.is_file():
            raise FileNotFoundError(f"Trait-balance reference is missing: {source}")
        metadata = read_json(source)
        if metadata.get("status") != "PASS":
            raise ValueError(f"Trait-balance reference is not certified: {state_id}")
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
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        return False
    implementation_fields = {
        "trait_balance_protocol": "protocol_sha256",
        "trainer": "trainer_sha256",
        "loss_helper": "loss_helper_sha256",
        "calibration_helper": "calibration_helper_sha256",
        "calibration_trainer": "calibration_trainer_sha256",
        "remediation_helper": "remediation_helper_sha256",
        "remediation_trainer": "remediation_trainer_sha256",
        "factor_builder": "factor_builder_sha256",
        "trainer_interface": "trainer_interface_sha256",
    }
    if any(
        metadata.get(metadata_key) != lock["artifacts"].get(lock_key)
        for lock_key, metadata_key in implementation_fields.items()
    ):
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
) -> None:
    state_id = str(row["state_id"])
    seed = int(row["seed"])
    output = run_dir(root, state_id, candidate)
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
            f"Trait-balance run failed: {state_id} {candidate}; log={log_path}"
        )


def _load_metrics(
    directory: Path,
    state_id: str,
    scenario: str,
    candidate: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    metadata = read_json(directory / "run_metadata.json")
    metadata = dict(metadata)
    metadata["candidate"] = candidate
    traits = pd.read_csv(directory / "validation_trait_metrics.tsv", sep="\t")
    traits.insert(0, "candidate", candidate)
    traits.insert(0, "scenario", scenario)
    traits.insert(0, "state_id", state_id)
    guards = pd.read_csv(directory / "validation_guard_metrics.tsv", sep="\t")
    guards = guards.loc[guards["mask_candidate"].eq(MASK_CANDIDATE)].copy()
    guards.insert(0, "candidate", candidate)
    guards.insert(0, "scenario", scenario)
    guards.insert(0, "state_id", state_id)
    return metadata, traits, guards


def collect_tables(
    root: Path, grid: pd.DataFrame, protocol: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, object]] = []
    trait_frames: list[pd.DataFrame] = []
    guard_frames: list[pd.DataFrame] = []
    new_candidates = [
        name
        for name, value in protocol["candidates"].items()
        if value.get("source_reuse") is False
    ]
    for row in grid.itertuples(index=False):
        state_id = str(row.state_id)
        scenario = str(row.scenario)
        source_dir = root / SOURCE_RUNS / state_id / SOURCE_REFERENCE
        metadata, traits, guards = _load_metrics(
            source_dir, state_id, scenario, REFERENCE
        )
        metadata["source_reuse"] = True
        metadata["source_candidate"] = SOURCE_REFERENCE
        run_rows.append(metadata)
        trait_frames.append(traits)
        guard_frames.append(guards)
        for candidate in new_candidates:
            directory = run_dir(root, state_id, candidate)
            metadata, traits, guards = _load_metrics(
                directory, state_id, scenario, candidate
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


def pair_runs(runs: pd.DataFrame) -> pd.DataFrame:
    reference = runs.loc[runs["candidate"].eq(REFERENCE)].copy()
    reference = reference[
        [
            "state_id",
            "validation_observation_signature",
            "validation_macro_normalized_rmse",
            "validation_macro_pearson",
            "validation_macro_calibration_error",
            "within_environment_centered_spearman",
            "within_environment_pairwise_accuracy",
        ]
    ].rename(columns=lambda value: value if value == "state_id" else f"{value}_reference")
    paired = runs.merge(reference, on="state_id", how="left", validate="many_to_one")
    paired["relative_nrmse_gain"] = (
        paired["validation_macro_normalized_rmse_reference"]
        - paired["validation_macro_normalized_rmse"]
    ) / paired["validation_macro_normalized_rmse_reference"]
    paired["nrmse_win"] = paired["validation_macro_normalized_rmse"].lt(
        paired["validation_macro_normalized_rmse_reference"]
    )
    paired["pearson_gain"] = (
        paired["validation_macro_pearson"]
        - paired["validation_macro_pearson_reference"]
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
    keys = ["state_id", "trait_name_canonical"]
    reference = traits.loc[traits["candidate"].eq(REFERENCE), keys + ["rows", "normalized_rmse", "pearson"]].rename(
        columns={
            "rows": "rows_reference",
            "normalized_rmse": "normalized_rmse_reference",
            "pearson": "pearson_reference",
        }
    )
    paired = traits.merge(reference, on=keys, how="left", validate="many_to_one")
    paired["relative_nrmse_gain"] = (
        paired["normalized_rmse_reference"] - paired["normalized_rmse"]
    ) / paired["normalized_rmse_reference"]
    paired["pearson_gain"] = paired["pearson"] - paired["pearson_reference"]
    return paired


def pair_guards(guards: pd.DataFrame) -> pd.DataFrame:
    keys = ["state_id", "subset"]
    reference = guards.loc[
        guards["candidate"].eq(REFERENCE),
        keys + ["rows", "observation_id_signature", "normalized_rmse_macro", "pearson_macro"],
    ].rename(
        columns={
            "rows": "rows_reference",
            "observation_id_signature": "observation_id_signature_reference",
            "normalized_rmse_macro": "normalized_rmse_macro_reference",
            "pearson_macro": "pearson_macro_reference",
        }
    )
    paired = guards.merge(reference, on=keys, how="left", validate="many_to_one")
    paired["relative_nrmse_gain"] = (
        paired["normalized_rmse_macro_reference"] - paired["normalized_rmse_macro"]
    ) / paired["normalized_rmse_macro_reference"]
    paired["pearson_gain"] = paired["pearson_macro"] - paired["pearson_macro_reference"]
    return paired


def _mean(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").mean())


def summarize(
    protocol: dict[str, Any],
    paired: pd.DataFrame,
    paired_traits: pd.DataFrame,
    paired_guards: pd.DataFrame,
) -> pd.DataFrame:
    acceptance = protocol["phase_1_acceptance"]
    primary = set(protocol["primary_traits"])
    exploratory = set(protocol["exploratory_traits"])
    rows: list[dict[str, object]] = []
    for candidate, local in paired.groupby("candidate", sort=False):
        local_traits = paired_traits.loc[paired_traits["candidate"].eq(candidate)]
        local_guards = paired_guards.loc[
            paired_guards["candidate"].eq(candidate)
            & paired_guards["rows"].ge(int(acceptance["minimum_rows_for_guard"]))
        ]
        trait_means = local_traits.groupby("trait_name_canonical")[
            "relative_nrmse_gain"
        ].mean()
        primary_gain = float(trait_means.loc[list(primary)].min())
        exploratory_gain = float(trait_means.loc[list(exploratory)].min())
        information_gain = float(
            local_guards.loc[
                local_guards["subset"].isin(INFORMATION_SUBSETS),
                "relative_nrmse_gain",
            ].min()
        )
        inactive_gain = _mean(
            local_guards.loc[
                local_guards["subset"].eq(
                    "PROJECTION_CORE_INACTIVE_814_ENVIRONMENTS"
                ),
                "relative_nrmse_gain",
            ]
        )
        macro_calibration = float(local["validation_macro_calibration_error"].max())
        primary_calibration = float(
            local_traits.loc[
                local_traits["trait_name_canonical"].isin(primary), "calibration_error"
            ].max()
        )
        negative_slopes = int(local_traits["calibration_slope"].lt(0).sum())
        gain = _mean(local["relative_nrmse_gain"])
        win_rate = _mean(local["nrmse_win"])
        pearson_gain = _mean(local["pearson_gain"])
        spearman_gain = _mean(local["centered_spearman_gain"])
        pairwise_gain = _mean(local["pairwise_accuracy_gain"])
        guards = {
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
            >= -float(
                acceptance["within_environment_centered_spearman_maximum_drop"]
            ),
            "pairwise_accuracy": pairwise_gain
            >= -float(
                acceptance["within_environment_pairwise_accuracy_maximum_drop"]
            ),
            "primary_traits": primary_gain
            >= -float(acceptance["primary_trait_maximum_mean_relative_nrmse_loss"]),
            "exploratory_traits": exploratory_gain
            >= -float(
                acceptance["exploratory_trait_maximum_mean_relative_nrmse_loss"]
            ),
            "information_subsets": information_gain
            >= -float(acceptance["information_class_maximum_relative_nrmse_loss"]),
            "projection_inactive": inactive_gain
            >= -float(
                acceptance[
                    "projection_inactive_environment_maximum_relative_nrmse_loss"
                ]
            ),
        }
        eligible = candidate == REFERENCE or all(guards.values())
        rows.append(
            {
                "scenario": "GNEW_EOBS",
                "candidate": candidate,
                "paired_inner_folds": int(local["state_id"].nunique()),
                "validation_normalized_rmse_mean": _mean(
                    local["validation_macro_normalized_rmse"]
                ),
                "validation_pearson_mean": _mean(local["validation_macro_pearson"]),
                "relative_normalized_rmse_gain_mean": gain,
                "normalized_rmse_win_rate": win_rate,
                "pearson_gain_mean": pearson_gain,
                "centered_spearman_gain_mean": spearman_gain,
                "pairwise_accuracy_gain_mean": pairwise_gain,
                "absolute_macro_calibration_error_max": macro_calibration,
                "primary_trait_calibration_error_max": primary_calibration,
                "negative_trait_calibration_slopes": negative_slopes,
                "primary_trait_relative_nrmse_gain_min": primary_gain,
                "exploratory_trait_relative_nrmse_gain_min": exploratory_gain,
                "information_subset_relative_nrmse_gain_min": information_gain,
                "projection_inactive_relative_nrmse_gain_mean": inactive_gain,
                **{f"guard_{name}": bool(value) for name, value in guards.items()},
                "eligible_for_confirmation": bool(eligible),
            }
        )
    return pd.DataFrame(rows)


def select_candidate(decision: pd.DataFrame, protocol: dict[str, Any]) -> str | None:
    candidates = decision.loc[
        ~decision["candidate"].eq(REFERENCE)
        & decision["eligible_for_confirmation"].eq(True)
    ].copy()
    if candidates.empty:
        return None
    order = {
        name: index
        for index, name in enumerate(protocol["selection"]["final_tiebreaker"])
    }
    candidates["order"] = candidates["candidate"].map(order)
    candidates = candidates.sort_values(
        ["validation_normalized_rmse_mean", "validation_pearson_mean", "order"],
        ascending=[True, False, True],
    )
    return str(candidates.iloc[0]["candidate"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Stage-1 v2 fixed-architecture trait-balance phase-1 screen"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--runtime-mode", choices=["server_cpu", "wsl_gpu"], default="server_cpu")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=5)
    parser.add_argument("--inter-op-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = (args.code_root or Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root))).resolve()
    python = Path(os.environ.get("PYTHON", sys.executable)).resolve()
    runtime = validate_runtime(code_root, args.runtime_mode)
    protocol = read_json(code_root / PROTOCOL)
    lock = read_json(root / LOCK)
    if lock.get("status") != "PASS_FROZEN_STAGE1_V2_PHASE6_TRAIT_BALANCE_SCREEN_V1":
        raise ValueError("Trait-balance screen is not frozen")
    for artifact, relative_path in {
        "trait_balance_protocol": PROTOCOL,
        "trainer": Path(
            "server_training_pipeline/train_stage1_v2_phase6_trait_balance_tf.py"
        ),
        "loss_helper": Path(
            "server_training_pipeline/stage1_v2_phase6_trait_balance_v1.py"
        ),
        "calibration_helper": Path(
            "server_training_pipeline/"
            "stage1_v2_phase6_hierarchy_calibration_amendment_v2.py"
        ),
        "calibration_trainer": Path(
            "server_training_pipeline/"
            "train_stage1_v2_phase6_hierarchy_calibration_amendment_tf.py"
        ),
        "remediation_helper": Path(
            "server_training_pipeline/stage1_v2_phase6_remediation.py"
        ),
        "remediation_trainer": Path(
            "server_training_pipeline/train_stage1_v2_phase6_remediation_tf.py"
        ),
        "factor_builder": Path(
            "server_training_pipeline/train_stage1_v2_phase6_tf.py"
        ),
        "trainer_interface": Path(
            "server_training_pipeline/stage1_v2_trainer_interface.py"
        ),
        "runner": Path("scripts/v2/run_stage1_v2_phase6_trait_balance_screen.py"),
    }.items():
        if sha256_file(code_root / relative_path) != lock["artifacts"].get(artifact):
            raise ValueError(f"Trait-balance frozen implementation drift: {artifact}")
    grid = build_grid(root, protocol)
    new_candidates = [
        name
        for name, value in protocol["candidates"].items()
        if value.get("source_reuse") is False
    ]
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output / "trait_balance_state_grid.tsv", sep="\t", index=False)
    write_json(
        output / "trait_balance_status.json",
        {
            "status": "RUNNING",
            "protocol_version": protocol["protocol_version"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "state_count": len(grid),
            "reference_reuse_count": len(grid),
            "new_model_fit_count": len(grid) * len(new_candidates),
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
    pending: list[tuple[dict[str, object], str]] = []
    for row in grid.to_dict(orient="records"):
        for candidate in new_candidates:
            if not run_complete(
                run_dir(root, str(row["state_id"]), candidate),
                str(row["state_id"]),
                candidate,
                int(row["seed"]),
                lock,
            ):
                pending.append((row, candidate))
    print(
        f"RUN Stage-1 v2 trait-balance phase_1; states={len(grid)} "
        f"new_fits={len(grid) * len(new_candidates)} pending={len(pending)} "
        f"workers={args.workers}",
        flush=True,
    )
    complete = len(grid) * len(new_candidates) - len(pending)
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
            print(
                f"[{complete}/{len(grid) * len(new_candidates)}] DONE "
                f"{row['state_id']} {candidate}",
                flush=True,
            )
            gc.collect()

    runs, traits, guards = collect_tables(root, grid, protocol)
    paired = pair_runs(runs)
    paired_traits = pair_traits(traits)
    paired_guards = pair_guards(guards)
    decision = summarize(protocol, paired, paired_traits, paired_guards)
    selected = select_candidate(decision, protocol)
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
        "matched_seeds": runs.groupby("state_id")["seed"].nunique().max() == 1,
        "matched_validation_observations": paired[
            "validation_observation_signature"
        ].eq(paired["validation_observation_signature_reference"]).all(),
        "guard_rows_exact": candidate_guards.loc[
            candidate_guards["rows"].gt(0), "rows"
        ].eq(candidate_guards.loc[candidate_guards["rows"].gt(0), "rows_reference"]).all(),
        "guard_identifiers_exact": candidate_guards.loc[
            candidate_guards["rows"].gt(0), "observation_id_signature"
        ].eq(
            candidate_guards.loc[
                candidate_guards["rows"].gt(0), "observation_id_signature_reference"
            ]
        ).all(),
        "outer_unread": runs["outer_test_metrics_read"].eq(False).all()
        and runs["outer_test_outcomes_read"].eq(False).all(),
        "final_holdout_unread": runs["final_holdout_outcomes_read"].eq(False).all(),
        "projection_product_unchanged": protocol["product_policy"][
            "projection_compatible_product"
        ]
        == "unchanged_and_separate",
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    artifacts = {
        "trait_balance_runs.tsv": runs,
        "trait_balance_trait_metrics.tsv": traits,
        "trait_balance_guard_metrics.tsv": guards,
        "trait_balance_paired_metrics.tsv": paired,
        "trait_balance_paired_trait_metrics.tsv": paired_traits,
        "trait_balance_paired_guard_metrics.tsv": paired_guards,
        "trait_balance_decision.tsv": decision,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output / name, sep="\t", index=False, lineterminator="\n")
    status = (
        "PASS_STAGE1_V2_PHASE6_TRAIT_BALANCE_PHASE1_CANDIDATE_SELECTED"
        if selected is not None
        else "PASS_STAGE1_V2_PHASE6_TRAIT_BALANCE_PHASE1_COMPLETE_NO_ADVANCE"
    )
    if failed:
        status = "FAIL_STAGE1_V2_PHASE6_TRAIT_BALANCE_PHASE1_INTEGRITY"
    result = {
        "status": status,
        "protocol_version": protocol["protocol_version"],
        "selection_data": "nested_inner_validation_only",
        "selected_candidate": selected,
        "reference_candidate": REFERENCE,
        "state_count": 5,
        "reference_reuse_count": 5,
        "new_model_fit_count": 10,
        "full_confirmation_allowed": selected is not None and not failed,
        "outer_evaluation_allowed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "checks": checks,
        "failed_checks": failed,
        "artifacts": {name: sha256_file(output / name) for name in artifacts},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "TRAIT_BALANCE_PHASE1_DECISION.json", result)
    write_json(output / "trait_balance_status.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if failed:
        raise SystemExit(f"Trait-balance phase_1 integrity failed: {failed}")


if __name__ == "__main__":
    main()
