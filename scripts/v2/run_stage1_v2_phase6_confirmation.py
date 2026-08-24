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
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.v2.run_stage1_v2_phase6_phase1 import validate_runtime
from server_training_pipeline.stage1_v2_trainer_interface import PARITY


LOCK = Path(
    "audit/v2/stage1_v2_phase6_confirmation_v1/PHASE6_CONFIRMATION_LOCK.json"
)
OUTPUT = Path("model_kernels/stage1_v2_phase6_confirmation_v1")
RUNS = Path("trained_models/stage1_v2_phase6_confirmation_v1_runs")
TRAINER = Path(
    "server_training_pipeline/train_stage1_v2_phase6_confirmation_tf.py"
)
ORCHESTRATOR = Path("scripts/v2/run_stage1_v2_phase6_confirmation.py")
EXECUTION_PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_execution_protocol_v2.json"
)
PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_confirmation_protocol_v1.json"
)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "Unable to identify code commit")
    return process.stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_confirmation_protocol(root: Path) -> dict[str, Any]:
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    protocol = json.loads((code_root / PROTOCOL).read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "stage1_v2_phase6_confirmation_v1":
        raise ValueError("Unexpected Stage-1 v2 confirmation protocol")
    return protocol


def _mean(values: Iterable[object]) -> float:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def confirmation_grid(root: Path, protocol: dict[str, Any]) -> pd.DataFrame:
    registry = pd.read_csv(root / PARITY / "splits/state_registry.tsv", sep="\t", dtype=str)
    states = registry.loc[registry["state_level"].eq("INNER")].copy()
    expected = protocol["confirmation_grid"]
    expected_scenarios = list(expected["scenarios"])
    states = states.loc[states["scenario"].isin(expected_scenarios)]
    states["outer_fold"] = states["outer_fold"].astype(int)
    states["inner_fold"] = states["inner_fold"].astype(int)
    states["scenario_index"] = states["scenario"].map(
        {value: index for index, value in enumerate(expected_scenarios)}
    )
    states = states.sort_values(
        ["scenario_index", "outer_fold", "inner_fold", "state_id"]
    )
    if len(states) != int(expected["state_count"]) or not states["state_id"].is_unique:
        raise ValueError(f"Confirmation state grid is incomplete: states={len(states)}")
    rows = []
    for state in states.itertuples(index=False):
        seed = (
            63000
            + int(state.scenario_index) * 10000
            + int(state.outer_fold) * 100
            + int(state.inner_fold) * 10
            + 1
        )
        for candidate in protocol["candidate_order"]:
            rows.append(
                {
                    "state_id": str(state.state_id),
                    "scenario": str(state.scenario),
                    "outer_fold": int(state.outer_fold),
                    "inner_fold": int(state.inner_fold),
                    "candidate": candidate,
                    "configuration_label": protocol["candidates"][candidate][
                        "configuration"
                    ],
                    "seed": seed,
                }
            )
    grid = pd.DataFrame(rows)
    if len(grid) != int(expected["run_count"]):
        raise ValueError(f"Confirmation run grid is incomplete: runs={len(grid)}")
    if grid.groupby("state_id")["seed"].nunique().max() != 1:
        raise ValueError("Confirmation candidates do not use matched within-state seeds")
    return grid


def run_dir(root: Path, row: pd.Series) -> Path:
    return root / RUNS / str(row["state_id"]) / str(row["candidate"])


def metadata_matches(
    path: Path,
    row: pd.Series,
    *,
    commit: str,
    protocol_sha: str,
    trainer_sha: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("status") == "PASS"
        and value.get("protocol_version") == "stage1_v2_phase6_confirmation_tf_v1"
        and value.get("state_id") == row["state_id"]
        and value.get("candidate") == row["candidate"]
        and value.get("configuration_label") == row["configuration_label"]
        and int(value.get("seed", -1)) == int(row["seed"])
        and value.get("code_commit") == commit
        and value.get("selection_protocol_sha256") == protocol_sha
        and value.get("trainer_sha256") == trainer_sha
        and value.get("guard_mask_observation_signatures_written") is True
        and value.get("outer_test_outcomes_read") is False
        and value.get("outer_test_metrics_read") is False
        and value.get("final_holdout_outcomes_read") is False
    )


def execute_run(
    data_root: Path,
    code_root: Path,
    row: pd.Series,
    *,
    runtime_mode: str,
    intra_op_threads: int,
    inter_op_threads: int,
) -> None:
    destination = run_dir(data_root, row)
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "server_training_pipeline.train_stage1_v2_phase6_confirmation_tf",
        "--root",
        str(data_root),
        "--state-id",
        str(row["state_id"]),
        "--candidate",
        str(row["candidate"]),
        "--seed",
        str(int(row["seed"])),
        "--out-dir",
        str(destination),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "WHEATCONFORMER_CODE_ROOT": str(code_root),
            "STAGE1_V2_EXECUTION_BACKEND": runtime_mode,
            "STAGE1_V2_INTRA_OP_THREADS": str(intra_op_threads),
            "STAGE1_V2_INTER_OP_THREADS": str(inter_op_threads),
            "OMP_NUM_THREADS": str(intra_op_threads),
            "MKL_NUM_THREADS": str(intra_op_threads),
            "OPENBLAS_NUM_THREADS": str(intra_op_threads),
            "NUMEXPR_NUM_THREADS": str(intra_op_threads),
            "TF_NUM_INTRAOP_THREADS": str(intra_op_threads),
            "TF_NUM_INTEROP_THREADS": str(inter_op_threads),
        }
    )
    if runtime_mode == "server_cpu":
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    with (destination / "run.log").open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=code_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Confirmation run failed with exit code {return_code}: "
            f"{row['state_id']} {row['candidate']}"
        )


def warm_factor_caches(
    root: Path, grid: pd.DataFrame, protocol: dict[str, Any]
) -> None:
    from server_training_pipeline.train_stage1_v2_phase6_confirmation_tf import (
        build_confirmation_factors,
    )

    unique = grid.drop_duplicates(["state_id", "candidate"])
    print(
        f"PREWARM phenotype-blind confirmation factors; bindings={len(unique)}",
        flush=True,
    )
    for number, row in enumerate(unique.itertuples(index=False), start=1):
        configuration = protocol["hyperparameter_configurations"][
            row.configuration_label
        ]
        factors = build_confirmation_factors(
            root, row.state_id, row.candidate, configuration
        )
        del factors
        gc.collect()
        if number == 1 or number % 10 == 0 or number == len(unique):
            print(f"PREWARM {number}/{len(unique)}", flush=True)


def pair_guard_metrics(
    guards: pd.DataFrame,
    candidate: str,
    reference: str,
) -> pd.DataFrame:
    candidate_rows = guards.loc[
        guards["candidate"].eq(candidate) & guards["mask_candidate"].eq(candidate)
    ].copy()
    reference_rows = guards.loc[
        guards["candidate"].eq(reference) & guards["mask_candidate"].eq(candidate),
        [
            "state_id",
            "subset",
            "rows",
            "observation_id_signature",
            "normalized_rmse_macro",
            "pearson_macro",
        ],
    ]
    paired = candidate_rows.merge(
        reference_rows,
        on=["state_id", "subset"],
        suffixes=("", "_reference"),
        validate="one_to_one",
    )
    exact = paired["rows"].eq(paired["rows_reference"]) & paired[
        "observation_id_signature"
    ].eq(paired["observation_id_signature_reference"])
    if not bool(exact.all()):
        raise ValueError(
            "Confirmation candidate/reference guard masks are not exactly paired:\n"
            + paired.loc[~exact].head(20).to_string(index=False)
        )
    paired["candidate"] = candidate
    paired["relative_nrmse_gain"] = (
        paired["normalized_rmse_macro_reference"]
        - paired["normalized_rmse_macro"]
    ) / paired["normalized_rmse_macro_reference"]
    return paired


def summarize(
    root: Path,
    code_root: Path,
    grid: pd.DataFrame,
    protocol: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    metadata_rows = []
    trait_frames = []
    guard_frames = []
    factor_frames = []
    for row in grid.itertuples(index=False):
        destination = root / RUNS / row.state_id / row.candidate
        metadata_rows.append(
            json.loads((destination / "run_metadata.json").read_text(encoding="utf-8"))
        )
        traits = pd.read_csv(destination / "validation_trait_metrics.tsv", sep="\t")
        traits["state_id"] = row.state_id
        traits["scenario"] = row.scenario
        traits["candidate"] = row.candidate
        trait_frames.append(traits)
        guards = pd.read_csv(destination / "validation_guard_metrics.tsv", sep="\t")
        guards["state_id"] = row.state_id
        guards["scenario"] = row.scenario
        guards["candidate"] = row.candidate
        guard_frames.append(guards)
        factors = pd.read_csv(destination / "active_component_factors.tsv", sep="\t")
        factors["state_id"] = row.state_id
        factors["scenario"] = row.scenario
        factors["candidate"] = row.candidate
        factor_frames.append(factors)
    runs = pd.DataFrame(metadata_rows)
    traits = pd.concat(trait_frames, ignore_index=True)
    guards = pd.concat(guard_frames, ignore_index=True)
    factors = pd.concat(factor_frames, ignore_index=True)
    reference = protocol["stable_reference_candidate"]
    metrics = [
        "validation_macro_normalized_rmse",
        "validation_macro_pearson",
        "validation_macro_calibration_error",
        "within_environment_centered_spearman",
        "within_environment_pairwise_accuracy",
    ]
    reference_runs = runs.loc[
        runs["candidate"].eq(reference),
        ["state_id", "validation_observation_signature", *metrics],
    ]
    paired = runs.merge(
        reference_runs,
        on="state_id",
        suffixes=("", "_reference"),
        validate="many_to_one",
    )
    if not paired["validation_observation_signature"].eq(
        paired["validation_observation_signature_reference"]
    ).all():
        raise ValueError("Confirmation validation observation signatures are not matched")
    paired["relative_nrmse_gain"] = (
        paired["validation_macro_normalized_rmse_reference"]
        - paired["validation_macro_normalized_rmse"]
    ) / paired["validation_macro_normalized_rmse_reference"]
    paired["nrmse_win"] = paired["relative_nrmse_gain"] > 0
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

    reference_traits = traits.loc[
        traits["candidate"].eq(reference),
        ["state_id", "trait_name_canonical", "normalized_rmse"],
    ]
    trait_paired = traits.merge(
        reference_traits,
        on=["state_id", "trait_name_canonical"],
        suffixes=("", "_reference"),
        validate="many_to_one",
    )
    trait_paired["relative_nrmse_gain"] = (
        trait_paired["normalized_rmse_reference"]
        - trait_paired["normalized_rmse"]
    ) / trait_paired["normalized_rmse_reference"]
    guard_paired = pd.concat(
        [
            pair_guard_metrics(guards, candidate, reference)
            for candidate in protocol["candidate_order"]
        ],
        ignore_index=True,
    )
    state_scenarios = grid.drop_duplicates("state_id").set_index("state_id")["scenario"]
    guard_paired["scenario"] = guard_paired["state_id"].map(state_scenarios)

    selection = protocol["scenario_route_selection"]
    guard_rules = protocol["guards"]
    primary = set(protocol["primary_traits"])
    information_subsets = {
        "PEDIGREE_ONLY",
        "MARKER_SUPPORTED",
        "PEDIGREE_AND_MARKER",
        "NEITHER_PRODUCTION_PEDIGREE_NOR_DENSE_MARKERS",
        "RECOVERED_IDENTITY_OR_COMPONENT",
    }
    rows = []
    for (scenario, candidate), group in paired.groupby(
        ["scenario", "candidate"], sort=False
    ):
        local_traits = trait_paired.loc[
            trait_paired["scenario"].eq(scenario)
            & trait_paired["candidate"].eq(candidate)
        ]
        primary_traits = local_traits.loc[
            local_traits["trait_name_canonical"].isin(primary)
        ]
        local_guards = guard_paired.loc[
            guard_paired["scenario"].eq(scenario)
            & guard_paired["candidate"].eq(candidate)
            & guard_paired["rows"].ge(int(guard_rules["minimum_rows_for_guard"]))
        ]
        information = local_guards.loc[
            local_guards["subset"].isin(information_subsets)
        ]
        projection_inactive = local_guards.loc[
            local_guards["subset"].eq(
                "PROJECTION_CORE_INACTIVE_814_ENVIRONMENTS"
            )
        ]
        nrmse_gain = _mean(group["relative_nrmse_gain"])
        win_rate = _mean(group["nrmse_win"])
        pearson_gain = _mean(group["pearson_gain"])
        spearman_gain = _mean(group["centered_spearman_gain"])
        pairwise_gain = _mean(group["pairwise_accuracy_gain"])
        macro_calibration = _mean(group["validation_macro_calibration_error"])
        macro_calibration_max = float(
            pd.to_numeric(
                group["validation_macro_calibration_error"], errors="coerce"
            ).max()
        )
        primary_gain_min = float(
            primary_traits.groupby("trait_name_canonical")["relative_nrmse_gain"]
            .mean()
            .min()
        )
        primary_calibration_max = float(
            pd.to_numeric(primary_traits["calibration_error"], errors="coerce").max()
        )
        negative_slopes = int(local_traits["calibration_slope"].lt(0).sum())
        information_gain_min = (
            float(
                information.groupby("subset")["relative_nrmse_gain"].mean().min()
            )
            if not information.empty
            else float("nan")
        )
        projection_inactive_gain = (
            _mean(projection_inactive["relative_nrmse_gain"])
            if not projection_inactive.empty
            else float("nan")
        )
        is_reference = candidate == reference
        eligible = is_reference or (
            nrmse_gain >= float(selection["minimum_relative_nrmse_gain"])
            and win_rate >= float(selection["minimum_paired_inner_fold_win_rate"])
            and pearson_gain >= -float(selection["maximum_mean_pearson_drop"])
            and macro_calibration_max
            <= float(selection["maximum_absolute_macro_calibration_error"])
            and primary_calibration_max
            <= float(selection["maximum_primary_trait_absolute_calibration_error"])
            and negative_slopes == 0
            and spearman_gain
            >= -float(
                guard_rules["within_environment_centered_spearman_maximum_drop"]
            )
            and pairwise_gain
            >= -float(
                guard_rules["within_environment_pairwise_accuracy_maximum_drop"]
            )
            and primary_gain_min
            >= -float(guard_rules["primary_trait_maximum_relative_nrmse_loss"])
            and (
                np.isnan(information_gain_min)
                or information_gain_min
                >= -float(
                    guard_rules["information_class_maximum_relative_nrmse_loss"]
                )
            )
            and (
                np.isnan(projection_inactive_gain)
                or projection_inactive_gain
                >= -float(
                    guard_rules[
                        "projection_inactive_environment_maximum_relative_nrmse_loss"
                    ]
                )
            )
        )
        rows.append(
            {
                "scenario": scenario,
                "candidate": candidate,
                "paired_inner_folds": len(group),
                "validation_normalized_rmse_mean": _mean(
                    group["validation_macro_normalized_rmse"]
                ),
                "validation_pearson_mean": _mean(
                    group["validation_macro_pearson"]
                ),
                "relative_normalized_rmse_gain_mean": nrmse_gain,
                "normalized_rmse_win_rate": win_rate,
                "pearson_gain_mean": pearson_gain,
                "absolute_macro_calibration_error_mean": macro_calibration,
                "absolute_macro_calibration_error_max": macro_calibration_max,
                "primary_trait_calibration_error_max": primary_calibration_max,
                "negative_trait_calibration_slopes": negative_slopes,
                "centered_spearman_gain_mean": spearman_gain,
                "pairwise_accuracy_gain_mean": pairwise_gain,
                "primary_trait_relative_nrmse_gain_min": primary_gain_min,
                "information_subset_relative_nrmse_gain_min": information_gain_min,
                "projection_inactive_relative_nrmse_gain_mean": (
                    projection_inactive_gain
                ),
                "eligible_for_scenario_route": bool(eligible),
                "decision": "eligible" if eligible else "do_not_route",
            }
        )
    scenario_summary = pd.DataFrame(rows)
    route_rows = []
    for scenario in protocol["confirmation_grid"]["scenarios"]:
        eligible = scenario_summary.loc[
            scenario_summary["scenario"].eq(scenario)
            & scenario_summary["eligible_for_scenario_route"]
        ].sort_values(
            ["validation_normalized_rmse_mean", "validation_pearson_mean"],
            ascending=[True, False],
        )
        if eligible.empty:
            raise ValueError(f"No eligible confirmation route for {scenario}")
        selected = eligible.iloc[0]
        route_rows.append(
            {
                "scenario": scenario,
                "selected_candidate": selected["candidate"],
                "validation_normalized_rmse_mean": selected[
                    "validation_normalized_rmse_mean"
                ],
                "validation_pearson_mean": selected["validation_pearson_mean"],
                "selection_data": "inner_validation_only",
                "outer_test_metrics_read": False,
            }
        )
    routes = pd.DataFrame(route_rows)

    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output / "confirmation_run_grid.tsv", sep="\t", index=False)
    runs.to_csv(output / "confirmation_runs.tsv", sep="\t", index=False)
    paired.to_csv(output / "confirmation_paired_metrics.tsv", sep="\t", index=False)
    traits.to_csv(output / "confirmation_trait_metrics.tsv", sep="\t", index=False)
    trait_paired.to_csv(
        output / "confirmation_paired_trait_metrics.tsv", sep="\t", index=False
    )
    guards.to_csv(output / "confirmation_guard_metrics.tsv", sep="\t", index=False)
    guard_paired.to_csv(
        output / "confirmation_paired_guard_metrics.tsv", sep="\t", index=False
    )
    factors.to_csv(output / "confirmation_factor_inventory.tsv", sep="\t", index=False)
    scenario_summary.to_csv(
        output / "confirmation_scenario_summary.tsv", sep="\t", index=False
    )
    routes.to_csv(output / "confirmation_scenario_routes.tsv", sep="\t", index=False)
    result_artifacts = [
        output / "confirmation_run_grid.tsv",
        output / "confirmation_runs.tsv",
        output / "confirmation_paired_metrics.tsv",
        output / "confirmation_trait_metrics.tsv",
        output / "confirmation_paired_trait_metrics.tsv",
        output / "confirmation_guard_metrics.tsv",
        output / "confirmation_paired_guard_metrics.tsv",
        output / "confirmation_factor_inventory.tsv",
        output / "confirmation_scenario_summary.tsv",
        output / "confirmation_scenario_routes.tsv",
    ]
    artifact_hashes = {
        path.name: sha256_file(path) for path in result_artifacts
    }
    route_lock = {
        "status": "PASS_STAGE1_V2_PHASE6_SCENARIO_ROUTES_FROZEN",
        "protocol_version": "stage1_v2_phase6_confirmation_route_lock_v1",
        "selection_data": "inner_validation_only",
        "stable_reference_candidate": reference,
        "selected_scenario_routes": dict(
            zip(routes["scenario"], routes["selected_candidate"])
        ),
        "result_artifact_sha256": artifact_hashes,
        "code_commit": git_commit(code_root),
        "selection_protocol_sha256": sha256_file(code_root / PROTOCOL),
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "outer_evaluation_allowed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "CONFIRMATION_SCENARIO_ROUTE_LOCK.json", route_lock)
    artifact_hashes["CONFIRMATION_SCENARIO_ROUTE_LOCK.json"] = sha256_file(
        output / "CONFIRMATION_SCENARIO_ROUTE_LOCK.json"
    )
    provenance = {
        "status": "PASS_STAGE1_V2_PHASE6_CONFIRMATION_COMPLETE",
        "protocol_version": "stage1_v2_phase6_confirmation_summary_v1",
        "selection_data": "inner_validation_only",
        "stage1_version": "Stage-1 v2",
        "state_count": int(grid["state_id"].nunique()),
        "run_count": len(runs),
        "candidate_count": int(runs["candidate"].nunique()),
        "scenario_count": int(runs["scenario"].nunique()),
        "matched_seed_status": "pass",
        "matched_validation_observation_status": "pass",
        "matched_component_mask_status": "pass",
        "stable_reference_candidate": reference,
        "selected_scenario_routes": dict(
            zip(routes["scenario"], routes["selected_candidate"])
        ),
        "result_artifact_sha256": artifact_hashes,
        "runtime": runtime,
        "code_commit": git_commit(code_root),
        "selection_protocol_sha256": sha256_file(code_root / PROTOCOL),
        "trainer_sha256": sha256_file(code_root / TRAINER),
        "orchestrator_sha256": sha256_file(code_root / ORCHESTRATOR),
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "outer_evaluation_allowed_after_route_lock": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "confirmation_provenance.json", provenance)
    write_json(output / "confirmation_status.json", provenance)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all 125 Stage-1 v2 Phase-6 adjudicated inner states"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--runtime-mode", choices=["wsl_gpu", "server_cpu"], default="wsl_gpu"
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--threads-per-worker", type=int)
    parser.add_argument("--inter-op-threads", type=int, default=1)
    parser.add_argument("--warm-factor-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    code_root = args.code_root.resolve()
    os.environ["WHEATCONFORMER_CODE_ROOT"] = str(code_root)
    protocol = load_confirmation_protocol(root)
    runtime = validate_runtime(code_root, args.runtime_mode)
    lock = json.loads((root / LOCK).read_text(encoding="utf-8"))
    commit = git_commit(code_root)
    if lock.get("status") != "PASS_READY_FOR_STAGE1_V2_PHASE6_CONFIRMATION":
        raise ValueError("Stage-1 v2 confirmation lock is not PASS")
    if lock.get("code_commit") != commit:
        raise ValueError("Stage-1 v2 confirmation lock targets a different commit")
    if lock.get("selection_protocol_sha256") != sha256_file(code_root / PROTOCOL):
        raise ValueError("Stage-1 v2 confirmation protocol changed after freeze")

    grid = confirmation_grid(root, protocol)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output / "confirmation_run_grid.tsv", sep="\t", index=False)
    protocol_sha = sha256_file(code_root / PROTOCOL)
    trainer_sha = sha256_file(code_root / TRAINER)
    workers = args.workers or min(4, int(runtime["physical_cpu_count"]))
    threads = args.threads_per_worker or max(
        1, int(runtime["physical_cpu_count"]) // workers
    )
    if workers < 1 or threads < 1 or args.inter_op_threads < 1:
        raise ValueError("Confirmation worker and thread counts must be positive")
    if workers * threads > int(runtime["physical_cpu_count"]):
        raise ValueError("Confirmation CPU allocation exceeds physical core count")
    if args.warm_factor_cache:
        warm_factor_caches(root, grid, protocol)

    pending = []
    for _, row in grid.iterrows():
        path = run_dir(root, row) / "run_metadata.json"
        if args.resume and metadata_matches(
            path,
            row,
            commit=commit,
            protocol_sha=protocol_sha,
            trainer_sha=trainer_sha,
        ):
            continue
        pending.append(row)
    print(
        f"RUN Stage-1 v2 confirmation; total={len(grid)} pending={len(pending)} "
        f"workers={workers} threads_per_worker={threads}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                execute_run,
                root,
                code_root,
                row,
                runtime_mode=args.runtime_mode,
                intra_op_threads=threads,
                inter_op_threads=args.inter_op_threads,
            ): row
            for row in pending
        }
        completed = len(grid) - len(pending)
        for future in as_completed(futures):
            row = futures[future]
            future.result()
            completed += 1
            print(
                f"[{completed}/{len(grid)}] DONE {row['state_id']} {row['candidate']}",
                flush=True,
            )
    for _, row in grid.iterrows():
        if not metadata_matches(
            run_dir(root, row) / "run_metadata.json",
            row,
            commit=commit,
            protocol_sha=protocol_sha,
            trainer_sha=trainer_sha,
        ):
            raise ValueError(
                f"Incomplete confirmation run: {row['state_id']} {row['candidate']}"
            )
    result = summarize(root, code_root, grid, protocol, runtime)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
