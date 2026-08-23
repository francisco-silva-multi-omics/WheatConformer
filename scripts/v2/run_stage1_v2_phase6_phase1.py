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

from server_training_pipeline.stage1_v2_trainer_interface import load_selection_protocol


HANDOFF = Path("audit/v2/phase6_model_selection_handoff_v1/PHASE6_MODEL_SELECTION_HANDOFF.json")
TRAINER = Path("server_training_pipeline/train_stage1_v2_phase6_tf.py")
ORCHESTRATOR = Path("scripts/v2/run_stage1_v2_phase6_phase1.py")
LAUNCHER = Path("scripts/v2/run_stage1_v2_phase6_phase1.sh")
SERVER_LAUNCHER = Path("scripts/v2/run_stage1_v2_phase6_phase1_server_cpu.sh")
DATA_PACKAGER = Path("scripts/v2/package_stage1_v2_phase6_phase1_server_data.py")
GPU_RUNTIME = Path("server_training_pipeline/stage1_v2_training_runtime_v1.json")
CPU_RUNTIME = Path("server_training_pipeline/stage1_v2_phase6_server_cpu_runtime_v1.json")
EXECUTION_PROTOCOL = Path("server_training_pipeline/stage1_v2_phase6_execution_protocol_v2.json")
GUARD_REPLAY = os.environ.get("STAGE1_V2_PHASE1_GUARD_REPLAY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
GUARD_REPLAY_LOCK = Path(
    "audit/v2/stage1_v2_phase6_phase1_guard_replay_v1/"
    "PHASE1_GUARD_REPLAY_LOCK.json"
)
PARENT_OUTPUT = Path("model_kernels/stage1_v2_phase6_phase1_v2")
PARENT_RUNS = Path("trained_models/stage1_v2_phase6_phase1_v2_runs")
OUTPUT = (
    Path("model_kernels/stage1_v2_phase6_phase1_guard_replay_v1")
    if GUARD_REPLAY
    else PARENT_OUTPUT
)
RUNS = (
    Path("trained_models/stage1_v2_phase6_phase1_guard_replay_v1_runs")
    if GUARD_REPLAY
    else PARENT_RUNS
)
FULL_VALIDATION_METRICS = (
    "validation_macro_normalized_rmse",
    "validation_macro_pearson",
    "validation_macro_calibration_error",
    "within_environment_centered_spearman",
    "within_environment_pairwise_accuracy",
)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "Unable to identify code commit")
    return process.stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def phase1_grid(protocol: dict[str, Any]) -> pd.DataFrame:
    schedule = protocol["screen_schedule"]
    scenario = str(schedule["phase_1_scenario"])
    outer_fold = int(schedule["phase_1_outer_fold"])
    inner_folds = [int(value) for value in schedule["phase_1_inner_folds"]]
    candidates = list(protocol["candidate_stages"]["phase_1_individual"])
    configurations = list(protocol["hyperparameter_configurations"])
    if scenario != "GNEW_EOBS" or outer_fold != 1 or inner_folds != [1, 2, 3, 4, 5]:
        raise ValueError("Phase-1 must use frozen GNEW_EOBS outer-1 five-inner-fold routing")
    rows = []
    for inner_fold in inner_folds:
        seed = 62000 + outer_fold * 100 + inner_fold * 10 + 1
        state_id = f"{scenario}__OUTER{outer_fold}__INNER{inner_fold}"
        for candidate in candidates:
            for configuration in configurations:
                rows.append(
                    {
                        "scenario": scenario,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "state_id": state_id,
                        "candidate": candidate,
                        "configuration_label": configuration,
                        "seed": seed,
                    }
                )
    grid = pd.DataFrame(rows)
    if len(grid) != 120 or grid.duplicated(
        ["state_id", "candidate", "configuration_label"]
    ).any():
        raise ValueError("Frozen Phase-1 grid must contain exactly 120 unique runs")
    if not grid.groupby("inner_fold")["seed"].nunique().eq(1).all():
        raise ValueError("All candidates and configurations must use matched fold seeds")
    return grid


def resolve_code_root(data_root: Path, requested: Path | None = None) -> Path:
    configured = requested or Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", data_root))
    code_root = configured.resolve()
    if not (code_root / TRAINER).is_file():
        raise FileNotFoundError(f"Stage-1 v2 code root is incomplete: {code_root}")
    return code_root


def recommended_cpu_parallelism(physical_cores: int) -> tuple[int, int]:
    if physical_cores < 1:
        raise ValueError("Physical CPU core count must be positive")
    if physical_cores >= 8:
        workers = min(6, max(4, physical_cores // 5))
    else:
        workers = max(1, physical_cores // 2)
    threads = max(1, physical_cores // workers)
    return workers, threads


def _memory_gib() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size / 1024**3)
    except (AttributeError, OSError, ValueError):
        return float("nan")


def _cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[-1].strip()
    return ""


def _physical_cpu_count() -> int:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return os.cpu_count() or 1
    processors: set[tuple[str, str]] = set()
    physical_id = "0"
    core_id = ""
    for line in (*path.read_text(encoding="utf-8", errors="replace").splitlines(), ""):
        if not line.strip():
            if core_id:
                processors.add((physical_id, core_id))
            physical_id = "0"
            core_id = ""
        elif line.lower().startswith("physical id"):
            physical_id = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("core id"):
            core_id = line.split(":", 1)[-1].strip()
    return len(processors) or (os.cpu_count() or 1)


def validate_runtime(code_root: Path, runtime_mode: str) -> dict[str, Any]:
    import duckdb
    import tensorflow as tf

    runtime_path = GPU_RUNTIME if runtime_mode == "wsl_gpu" else CPU_RUNTIME
    runtime = json.loads((code_root / runtime_path).read_text(encoding="utf-8"))
    observed = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "tensorflow": tf.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "duckdb": duckdb.__version__,
        "gpu_count": len(tf.config.list_physical_devices("GPU")),
        "logical_cpu_count": os.cpu_count() or 1,
        "physical_cpu_count": _physical_cpu_count(),
        "memory_gib": _memory_gib(),
        "cpu_model": _cpu_model(),
        "runtime_mode": runtime_mode,
    }
    if runtime_mode == "wsl_gpu":
        expected_python = runtime["python"]
    else:
        expected_python = str(runtime["python_major_minor"])
        if not observed["python"].startswith(expected_python + "."):
            raise ValueError(
                f"Certified server runtime mismatch for python: observed={observed['python']} "
                f"expected_major_minor={expected_python}"
            )
        expected_python = observed["python"]
    expected = {
        "python": expected_python,
        "tensorflow": runtime["tensorflow"],
        "pandas": runtime["pandas"],
        "numpy": runtime["numpy"],
        "duckdb": runtime["duckdb"],
    }
    for key in ("python", "tensorflow", "pandas", "numpy", "duckdb"):
        if observed[key] != expected[key]:
            raise ValueError(
                f"Certified runtime mismatch for {key}: observed={observed[key]} expected={expected[key]}"
            )
    if runtime.get("tensorflow_gpu_required_for_training") and observed["gpu_count"] < 1:
        raise ValueError("Certified Phase-1 training requires a visible TensorFlow GPU")
    if runtime_mode == "server_cpu":
        required_cpu = str(runtime["cpu_model_required_substring"])
        if required_cpu not in observed["cpu_model"]:
            raise ValueError(
                f"Server CPU identity mismatch: observed={observed['cpu_model']!r}; "
                f"required_substring={required_cpu!r}"
            )
        if observed["memory_gib"] < float(runtime["minimum_memory_gib"]):
            raise ValueError(
                f"Server memory is below the frozen minimum: {observed['memory_gib']:.1f} GiB"
            )
        if observed["gpu_count"] != 0:
            raise ValueError("Server CPU execution must hide all GPUs")
    observed["runtime_protocol"] = runtime_path.as_posix()
    observed["runtime_protocol_sha256"] = sha256_file(code_root / runtime_path)
    return observed


def validate_handoff(data_root: Path, code_root: Path) -> dict[str, Any]:
    if GUARD_REPLAY:
        lock_path = data_root / GUARD_REPLAY_LOCK
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("status") != "PASS_READY_FOR_PHASE1_MATCHED_GUARD_REPLAY":
            raise ValueError("Phase-1 guard replay lock is not ready")
        if lock.get("code_commit") != git_commit(code_root):
            raise ValueError("Phase-1 guard replay lock is bound to another commit")
        if lock.get("outer_test_outcomes_read") is not False:
            raise ValueError("Guard replay lock unexpectedly read outer outcomes")
        if lock.get("outer_test_metrics_read") is not False:
            raise ValueError("Guard replay lock unexpectedly read outer metrics")
        if lock.get("final_holdout_outcomes_read") is not False:
            raise ValueError("Guard replay lock unexpectedly read final holdout outcomes")
        for relative, expected_sha in lock["implementation_sha256"].items():
            path = code_root / relative
            if not path.is_file() or sha256_file(path) != expected_sha:
                raise ValueError(f"Frozen guard replay implementation mismatch: {relative}")
        protocol_path = (
            code_root
            / "server_training_pipeline/stage1_v2_phase6_selection_protocol_v1.json"
        )
        if lock.get("selection_protocol_sha256") != sha256_file(protocol_path):
            raise ValueError("Guard replay selection protocol checksum mismatch")
        return lock
    handoff = json.loads((data_root / HANDOFF).read_text(encoding="utf-8"))
    if handoff.get("status") != "PASS_READY_FOR_STAGE1_V2_PHASE6_INNER_MODEL_SELECTION":
        raise ValueError("Aggregate Phase-6 handoff is not ready")
    if handoff.get("code_commit") != git_commit(code_root):
        raise ValueError("Aggregate Phase-6 handoff is not bound to the active commit")
    if handoff.get("outer_evaluation_allowed") is not False:
        raise ValueError("Phase-1 handoff unexpectedly permits outer evaluation")
    expected = handoff.get("phase1_implementation_sha256", {})
    for relative in (
        TRAINER,
        ORCHESTRATOR,
        LAUNCHER,
        SERVER_LAUNCHER,
        DATA_PACKAGER,
        CPU_RUNTIME,
        EXECUTION_PROTOCOL,
    ):
        observed = sha256_file(code_root / relative)
        if expected.get(relative.as_posix()) != observed:
            raise ValueError(f"Frozen Phase-1 implementation mismatch: {relative}")
    protocol_path = code_root / "server_training_pipeline/stage1_v2_phase6_selection_protocol_v1.json"
    if handoff.get("selection_protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("Frozen Phase-1 selection protocol checksum mismatch")
    if handoff.get("phase1_run_count") != 120:
        raise ValueError("Aggregate handoff does not freeze the 120-run Phase-1 grid")
    return handoff


def run_dir(root: Path, row: pd.Series) -> Path:
    return (
        root
        / RUNS
        / str(row["state_id"])
        / str(row["candidate"])
        / str(row["configuration_label"])
    )


def parent_run_dir(root: Path, row: pd.Series) -> Path:
    return (
        root
        / PARENT_RUNS
        / str(row["state_id"])
        / str(row["candidate"])
        / str(row["configuration_label"])
    )


def validate_guard_replay_run(root: Path, row: pd.Series) -> None:
    current_path = run_dir(root, row) / "run_metadata.json"
    parent_path = parent_run_dir(root, row) / "run_metadata.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    deltas = []
    for metric in FULL_VALIDATION_METRICS:
        observed = float(current[metric])
        expected = float(parent[metric])
        if np.isnan(observed) and np.isnan(expected):
            delta = 0.0
        elif not np.isfinite(observed) or not np.isfinite(expected):
            delta = float("inf")
        else:
            delta = abs(observed - expected)
        deltas.append(delta)
    maximum_delta = max(deltas)
    if maximum_delta > 1e-5:
        raise ValueError(
            "Guard replay changed frozen full-validation metrics for "
            f"{row['state_id']} {row['candidate']} {row['configuration_label']}: "
            f"maximum_absolute_delta={maximum_delta:.9g}"
        )


def metadata_matches(
    path: Path,
    row: pd.Series,
    *,
    commit: str,
    protocol_sha: str,
    trainer_sha: str,
    execution_protocol_sha: str,
    runtime_mode: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    matches = (
        value.get("status") == "PASS"
        and value.get("state_id") == row["state_id"]
        and value.get("candidate") == row["candidate"]
        and value.get("configuration_label") == row["configuration_label"]
        and int(value.get("seed", -1)) == int(row["seed"])
        and value.get("code_commit") == commit
        and value.get("selection_protocol_sha256") == protocol_sha
        and value.get("trainer_sha256") == trainer_sha
        and value.get("execution_protocol_sha256") == execution_protocol_sha
        and value.get("execution_backend") == runtime_mode
        and value.get("outer_test_outcomes_read") is False
        and value.get("outer_test_metrics_read") is False
        and value.get("final_holdout_outcomes_read") is False
    )
    if GUARD_REPLAY:
        matches = (
            matches
            and value.get("protocol_version")
            == "stage1_v2_phase6_phase1_guard_replay_v1"
            and value.get("guard_mask_observation_signatures_written") is True
            and (path.parent / "validation_guard_metrics.tsv").is_file()
        )
    return matches


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
        "server_training_pipeline.train_stage1_v2_phase6_tf",
        "--root",
        str(data_root),
        "--state-id",
        str(row["state_id"]),
        "--candidate",
        str(row["candidate"]),
        "--configuration",
        str(row["configuration_label"]),
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
            f"Phase-1 run failed with exit code {return_code}: "
            f"{row['state_id']} {row['candidate']} {row['configuration_label']}"
        )
    if GUARD_REPLAY:
        validate_guard_replay_run(data_root, row)


def warm_factor_caches(data_root: Path, grid: pd.DataFrame, protocol: dict[str, Any]) -> None:
    from server_training_pipeline.train_stage1_v2_phase6_tf import build_candidate_factors

    unique = grid.drop_duplicates(["state_id", "candidate", "configuration_label"])
    print(
        f"PREWARM phenotype-blind factor caches; requested_bindings={len(unique)}",
        flush=True,
    )
    for number, (_, row) in enumerate(unique.iterrows(), start=1):
        configuration = protocol["hyperparameter_configurations"][row["configuration_label"]]
        factors = build_candidate_factors(
            data_root,
            str(row["state_id"]),
            str(row["candidate"]),
            configuration,
        )
        del factors
        gc.collect()
        if number == 1 or number % 10 == 0 or number == len(unique):
            print(f"PREWARM {number}/{len(unique)}", flush=True)


def _mean(values: Iterable[object]) -> float:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def pair_guard_metrics(guard_metrics: pd.DataFrame, baseline: str) -> pd.DataFrame:
    candidate_guards = guard_metrics.loc[
        guard_metrics["candidate"].eq(guard_metrics["mask_candidate"])
    ].copy()
    reference_guards = guard_metrics.loc[
        guard_metrics["candidate"].eq(baseline),
        [
            "state_id",
            "configuration_label",
            "mask_candidate",
            "subset",
            "rows",
            "observation_id_signature",
            "normalized_rmse_macro",
            "pearson_macro",
        ],
    ]
    paired = candidate_guards.merge(
        reference_guards,
        on=["state_id", "configuration_label", "mask_candidate", "subset"],
        suffixes=("", "_reference"),
        validate="one_to_one",
    )
    row_match = paired["rows"].eq(paired["rows_reference"])
    signature_match = paired["observation_id_signature"].eq(
        paired["observation_id_signature_reference"]
    )
    if not bool((row_match & signature_match).all()):
        failed = paired.loc[
            ~(row_match & signature_match),
            [
                "state_id",
                "candidate",
                "configuration_label",
                "subset",
                "rows",
                "rows_reference",
                "observation_id_signature",
                "observation_id_signature_reference",
            ],
        ]
        raise ValueError(
            "Candidate/reference guard masks are not exactly paired:\n"
            + failed.head(20).to_string(index=False)
        )
    return paired


def summarize(
    data_root: Path,
    code_root: Path,
    grid: pd.DataFrame,
    protocol: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    metadata_rows = []
    trait_frames = []
    subset_frames = []
    guard_frames = []
    for _, row in grid.iterrows():
        destination = run_dir(data_root, row)
        metadata = json.loads((destination / "run_metadata.json").read_text(encoding="utf-8"))
        metadata_rows.append(metadata)
        traits = pd.read_csv(destination / "validation_trait_metrics.tsv", sep="\t")
        traits["state_id"] = row["state_id"]
        traits["candidate"] = row["candidate"]
        traits["configuration_label"] = row["configuration_label"]
        trait_frames.append(traits)
        subsets = pd.read_csv(destination / "validation_subset_metrics.tsv", sep="\t")
        subsets["state_id"] = row["state_id"]
        subsets["candidate"] = row["candidate"]
        subsets["configuration_label"] = row["configuration_label"]
        subset_frames.append(subsets)
        if GUARD_REPLAY:
            guards = pd.read_csv(destination / "validation_guard_metrics.tsv", sep="\t")
            guards["state_id"] = row["state_id"]
            guards["candidate"] = row["candidate"]
            guards["configuration_label"] = row["configuration_label"]
            guard_frames.append(guards)
    runs = pd.DataFrame(metadata_rows)
    traits = pd.concat(trait_frames, ignore_index=True)
    subsets = pd.concat(subset_frames, ignore_index=True)
    guard_metrics = (
        pd.concat(guard_frames, ignore_index=True) if guard_frames else pd.DataFrame()
    )
    replay_runs = runs.copy()
    replay_traits = traits.copy()
    if GUARD_REPLAY:
        parent_metadata_rows = []
        parent_trait_frames = []
        for _, row in grid.iterrows():
            destination = parent_run_dir(data_root, row)
            parent_metadata_rows.append(
                json.loads((destination / "run_metadata.json").read_text(encoding="utf-8"))
            )
            parent_traits = pd.read_csv(
                destination / "validation_trait_metrics.tsv", sep="\t"
            )
            parent_traits["state_id"] = row["state_id"]
            parent_traits["candidate"] = row["candidate"]
            parent_traits["configuration_label"] = row["configuration_label"]
            parent_trait_frames.append(parent_traits)
        runs = pd.DataFrame(parent_metadata_rows)
        traits = pd.concat(parent_trait_frames, ignore_index=True)
    baseline = "ka_identity_location_baseline"
    metrics = list(FULL_VALIDATION_METRICS)
    replay_audit = pd.DataFrame()
    if GUARD_REPLAY:
        replay_rows = []
        for _, row in grid.iterrows():
            parent = runs.loc[
                runs["state_id"].eq(row["state_id"])
                & runs["candidate"].eq(row["candidate"])
                & runs["configuration_label"].eq(row["configuration_label"])
            ].iloc[0]
            current = replay_runs.loc[
                replay_runs["state_id"].eq(row["state_id"])
                & replay_runs["candidate"].eq(row["candidate"])
                & replay_runs["configuration_label"].eq(row["configuration_label"])
            ].iloc[0]
            for metric in metrics:
                observed = float(current[metric])
                expected = float(parent[metric])
                if np.isnan(observed) and np.isnan(expected):
                    absolute_delta = 0.0
                elif not np.isfinite(observed) or not np.isfinite(expected):
                    absolute_delta = float("inf")
                else:
                    absolute_delta = abs(observed - expected)
                replay_rows.append(
                    {
                        "state_id": row["state_id"],
                        "candidate": row["candidate"],
                        "configuration_label": row["configuration_label"],
                        "metric": metric,
                        "parent_value": expected,
                        "replay_value": observed,
                        "absolute_delta": absolute_delta,
                    }
                )
        replay_audit = pd.DataFrame(replay_rows)
        maximum_delta = float(replay_audit["absolute_delta"].max())
        if maximum_delta > 1e-5:
            raise ValueError(
                "Guard replay changed frozen full-validation metrics: "
                f"maximum_absolute_delta={maximum_delta:.9g}"
            )
    paired = runs.merge(
        runs.loc[runs["candidate"].eq(baseline), ["state_id", "configuration_label", *metrics]],
        on=["state_id", "configuration_label"],
        suffixes=("", "_reference"),
        validate="many_to_one",
    )
    paired["relative_nrmse_gain"] = (
        paired["validation_macro_normalized_rmse_reference"]
        - paired["validation_macro_normalized_rmse"]
    ) / paired["validation_macro_normalized_rmse_reference"]
    paired["nrmse_win"] = paired["relative_nrmse_gain"] > 0
    paired["pearson_gain"] = (
        paired["validation_macro_pearson"]
        - paired["validation_macro_pearson_reference"]
    )
    paired["calibration_error_delta"] = (
        paired["validation_macro_calibration_error"]
        - paired["validation_macro_calibration_error_reference"]
    )
    paired["centered_spearman_gain"] = (
        paired["within_environment_centered_spearman"]
        - paired["within_environment_centered_spearman_reference"]
    )
    paired["pairwise_accuracy_gain"] = (
        paired["within_environment_pairwise_accuracy"]
        - paired["within_environment_pairwise_accuracy_reference"]
    )

    primary = set(protocol["primary_traits"])
    trait_reference = traits.loc[traits["candidate"].eq(baseline), [
        "state_id", "configuration_label", "trait_name_canonical", "normalized_rmse"
    ]]
    trait_paired = traits.merge(
        trait_reference,
        on=["state_id", "configuration_label", "trait_name_canonical"],
        suffixes=("", "_reference"),
        validate="many_to_one",
    )
    trait_paired["relative_nrmse_gain"] = (
        trait_paired["normalized_rmse_reference"] - trait_paired["normalized_rmse"]
    ) / trait_paired["normalized_rmse_reference"]

    if GUARD_REPLAY:
        subset_paired = pair_guard_metrics(guard_metrics, baseline)
    else:
        subset_reference = subsets.loc[subsets["candidate"].eq(baseline), [
            "state_id", "configuration_label", "subset", "normalized_rmse_macro"
        ]]
        subset_paired = subsets.merge(
            subset_reference,
            on=["state_id", "configuration_label", "subset"],
            suffixes=("", "_reference"),
            validate="many_to_one",
        )
    subset_paired["relative_nrmse_gain"] = (
        subset_paired["normalized_rmse_macro_reference"]
        - subset_paired["normalized_rmse_macro"]
    ) / subset_paired["normalized_rmse_macro_reference"]

    selection = protocol["selection_metrics"]
    guards = protocol["guards"]
    information_subsets = {
        "PEDIGREE_ONLY",
        "MARKER_SUPPORTED",
        "PEDIGREE_AND_MARKER",
        "NEITHER_PRODUCTION_PEDIGREE_NOR_DENSE_MARKERS",
        "RECOVERED_IDENTITY_OR_COMPONENT",
    }
    projection_inactive_subset = "PROJECTION_CORE_INACTIVE_814_ENVIRONMENTS"
    rows = []
    for (candidate, configuration), group in paired.groupby(
        ["candidate", "configuration_label"], sort=False
    ):
        local_traits = trait_paired.loc[
            trait_paired["candidate"].eq(candidate)
            & trait_paired["configuration_label"].eq(configuration)
            & trait_paired["trait_name_canonical"].isin(primary)
        ]
        local_subsets = subset_paired.loc[
            subset_paired["candidate"].eq(candidate)
            & subset_paired["configuration_label"].eq(configuration)
            & subset_paired["rows"].ge(int(guards["minimum_rows_for_guard"]))
        ]
        local_information_subsets = local_subsets.loc[
            local_subsets["subset"].isin(information_subsets)
        ]
        local_projection_inactive = local_subsets.loc[
            local_subsets["subset"].eq(projection_inactive_subset)
        ]
        nrmse_gain = _mean(group["relative_nrmse_gain"])
        win_rate = _mean(group["nrmse_win"])
        pearson_gain = _mean(group["pearson_gain"])
        calibration_delta = _mean(group["calibration_error_delta"])
        spearman_gain = _mean(group["centered_spearman_gain"])
        pairwise_gain = _mean(group["pairwise_accuracy_gain"])
        primary_min = (
            float(local_traits.groupby("trait_name_canonical")["relative_nrmse_gain"].mean().min())
            if not local_traits.empty else float("nan")
        )
        subset_min = (
            float(
                local_information_subsets.groupby("subset")["relative_nrmse_gain"]
                .mean()
                .min()
            )
            if not local_information_subsets.empty
            else float("nan")
        )
        projection_inactive_gain = (
            _mean(local_projection_inactive["relative_nrmse_gain"])
            if not local_projection_inactive.empty
            else float("nan")
        )
        is_reference = candidate == baseline
        accepted = is_reference or (
            nrmse_gain >= float(selection["minimum_relative_nrmse_gain"])
            and win_rate >= float(selection["minimum_paired_inner_fold_win_rate"])
            and pearson_gain >= -float(selection["maximum_mean_pearson_drop"])
            and calibration_delta <= float(selection["maximum_calibration_error_increase"])
            and spearman_gain >= -float(guards["within_environment_centered_spearman_maximum_drop"])
            and pairwise_gain >= -float(guards["within_environment_pairwise_accuracy_maximum_drop"])
            and primary_min >= -float(guards["primary_trait_maximum_relative_nrmse_loss"])
            and (np.isnan(subset_min) or subset_min >= -float(guards["information_class_maximum_relative_nrmse_loss"]))
            and (
                np.isnan(projection_inactive_gain)
                or projection_inactive_gain
                >= -float(
                    guards[
                        "projection_inactive_environment_maximum_relative_nrmse_loss"
                    ]
                )
            )
        )
        rows.append(
            {
                "candidate": candidate,
                "configuration_label": configuration,
                "paired_inner_folds": len(group),
                "validation_normalized_rmse_mean": _mean(group["validation_macro_normalized_rmse"]),
                "validation_pearson_mean": _mean(group["validation_macro_pearson"]),
                "relative_normalized_rmse_gain_mean": nrmse_gain,
                "normalized_rmse_win_rate": win_rate,
                "pearson_gain_mean": pearson_gain,
                "calibration_error_delta_mean": calibration_delta,
                "centered_spearman_gain_mean": spearman_gain,
                "pairwise_accuracy_gain_mean": pairwise_gain,
                "primary_trait_relative_nrmse_gain_min": primary_min,
                "information_subset_relative_nrmse_gain_min": subset_min,
                "projection_inactive_relative_nrmse_gain_mean": projection_inactive_gain,
                "accepted": bool(accepted),
                "decision": "reference" if is_reference else (
                    "advance_to_confirmation" if accepted else "do_not_advance"
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["validation_normalized_rmse_mean", "candidate", "configuration_label"]
    )
    output = data_root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output / "phase1_run_grid.tsv", sep="\t", index=False)
    runs.to_csv(output / "phase1_runs.tsv", sep="\t", index=False)
    paired.to_csv(output / "phase1_paired_metrics.tsv", sep="\t", index=False)
    traits.to_csv(output / "phase1_trait_metrics.tsv", sep="\t", index=False)
    subsets.to_csv(output / "phase1_subset_metrics.tsv", sep="\t", index=False)
    if GUARD_REPLAY:
        replay_runs.to_csv(
            output / "phase1_replay_runs.tsv", sep="\t", index=False
        )
        replay_traits.to_csv(
            output / "phase1_replay_trait_metrics.tsv", sep="\t", index=False
        )
        guard_metrics.to_csv(
            output / "phase1_guard_metrics.tsv", sep="\t", index=False
        )
        subset_paired.to_csv(
            output / "phase1_paired_guard_metrics.tsv", sep="\t", index=False
        )
        replay_audit.to_csv(
            output / "phase1_parent_metric_replay_audit.tsv", sep="\t", index=False
        )
    summary.to_csv(output / "phase1_decision.tsv", sep="\t", index=False)
    provenance = {
        "status": "PASS",
        "protocol_version": (
            "stage1_v2_phase6_phase1_matched_guard_replay_v1"
            if GUARD_REPLAY
            else "stage1_v2_phase6_phase1_screen_v2_cpu_parallel"
        ),
        "selection_data": "five_nested_inner_validation_folds_only",
        "scenario": "GNEW_EOBS",
        "outer_fold": 1,
        "inner_fold_count": 5,
        "candidate_count": int(runs["candidate"].nunique()),
        "configuration_count": int(runs["configuration_label"].nunique()),
        "run_count": len(runs),
        "matched_seed_status": "pass",
        "matched_validation_observation_status": (
            "pass" if runs.groupby("state_id")["validation_observation_signature"].nunique().eq(1).all() else "fail"
        ),
        "matched_component_mask_status": (
            "pass"
            if GUARD_REPLAY
            and subset_paired["rows"].eq(subset_paired["rows_reference"]).all()
            and subset_paired["observation_id_signature"].eq(
                subset_paired["observation_id_signature_reference"]
            ).all()
            else ("not_replayed" if not GUARD_REPLAY else "fail")
        ),
        "parent_full_metric_replay_status": (
            "pass"
            if GUARD_REPLAY and replay_audit["absolute_delta"].max() <= 1e-5
            else ("not_replayed" if not GUARD_REPLAY else "fail")
        ),
        "parent_full_metric_maximum_absolute_delta": (
            float(replay_audit["absolute_delta"].max())
            if GUARD_REPLAY
            else None
        ),
        "global_and_trait_selection_metrics_source": (
            "immutable_parent_phase1" if GUARD_REPLAY else "current_runs"
        ),
        "component_guard_metrics_source": (
            "matched_guard_replay" if GUARD_REPLAY else "architecture_local_unpaired"
        ),
        "h_seeds_direct_marker_support_included": bool(GUARD_REPLAY),
        "projection_core_mask_candidate_independent": bool(GUARD_REPLAY),
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "outer_evaluation_allowed": False,
        "execution_backend": runtime["runtime_mode"],
        "runtime_protocol_sha256": runtime["runtime_protocol_sha256"],
        "execution_protocol_sha256": sha256_file(code_root / EXECUTION_PROTOCOL),
        "selection_protocol_sha256": sha256_file(
            code_root / "server_training_pipeline/stage1_v2_phase6_selection_protocol_v1.json"
        ),
        "trainer_sha256": sha256_file(code_root / TRAINER),
        "code_commit": git_commit(code_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if provenance["matched_validation_observation_status"] != "pass":
        raise ValueError("Candidates did not use identical validation observations")
    if GUARD_REPLAY and provenance["matched_component_mask_status"] != "pass":
        raise ValueError("Candidates did not use exactly matched component masks")
    write_json(output / "phase1_provenance.json", provenance)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Stage-1 v2 Phase-1 inner screen")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    parser.add_argument(
        "--runtime-mode", choices=("wsl_gpu", "server_cpu"), default="wsl_gpu"
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--threads-per-worker", type=int)
    parser.add_argument("--inter-op-threads", type=int)
    parser.add_argument(
        "--warm-factor-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Build shared phenotype-blind factors before starting parallel workers",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.root.resolve()
    code_root = resolve_code_root(data_root, args.code_root)
    os.environ["WHEATCONFORMER_CODE_ROOT"] = str(code_root)
    if args.runtime_mode == "server_cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    protocol = load_selection_protocol(data_root)
    grid = phase1_grid(protocol)
    handoff = validate_handoff(data_root, code_root)
    runtime = validate_runtime(code_root, args.runtime_mode)
    if args.runtime_mode == "server_cpu":
        recommended_workers, recommended_threads = recommended_cpu_parallelism(
            int(runtime["physical_cpu_count"])
        )
    else:
        recommended_workers, recommended_threads = 1, 16
    workers = args.workers if args.workers is not None else recommended_workers
    threads_per_worker = (
        args.threads_per_worker
        if args.threads_per_worker is not None
        else recommended_threads
    )
    inter_op_threads = (
        args.inter_op_threads
        if args.inter_op_threads is not None
        else (1 if args.runtime_mode == "server_cpu" else 2)
    )
    if workers < 1 or threads_per_worker < 1 or inter_op_threads < 1:
        raise ValueError("Worker and TensorFlow thread counts must be positive")
    if args.runtime_mode == "wsl_gpu" and workers != 1:
        raise ValueError("The frozen WSL GPU runtime permits exactly one training worker")
    if (
        args.runtime_mode == "server_cpu"
        and workers * threads_per_worker > int(runtime["logical_cpu_count"])
    ):
        raise ValueError("Requested CPU workers oversubscribe the server's logical CPUs")
    warm_factor_cache = (
        args.warm_factor_cache if args.warm_factor_cache is not None else workers > 1
    )
    output = data_root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output / "phase1_run_grid.tsv", sep="\t", index=False)
    execution_protocol_sha = sha256_file(code_root / EXECUTION_PROTOCOL)
    startup = {
        "status": "RUNNING",
        "protocol_version": (
            "stage1_v2_phase6_phase1_matched_guard_replay_v1"
            if GUARD_REPLAY
            else "stage1_v2_phase6_phase1_screen_v2_cpu_parallel"
        ),
        "run_count": len(grid),
        "data_root": str(data_root),
        "code_root": str(code_root),
        "code_commit": git_commit(code_root),
        "runtime": runtime,
        "parallel_workers": workers,
        "threads_per_worker": threads_per_worker,
        "inter_op_threads_per_worker": inter_op_threads,
        "factor_cache_prewarm": warm_factor_cache,
        "execution_protocol_sha256": execution_protocol_sha,
        "handoff_sha256": sha256_file(
            data_root / (GUARD_REPLAY_LOCK if GUARD_REPLAY else HANDOFF)
        ),
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "phase1_status.json", startup)
    print(json.dumps(startup, indent=2, sort_keys=True), flush=True)
    commit = str(handoff["code_commit"])
    protocol_sha = str(handoff["selection_protocol_sha256"])
    trainer_sha = sha256_file(code_root / TRAINER)
    pending: list[tuple[int, pd.Series]] = []
    for number, (_, row) in enumerate(grid.iterrows(), start=1):
        metadata_path = run_dir(data_root, row) / "run_metadata.json"
        if args.resume and metadata_matches(
            metadata_path,
            row,
            commit=commit,
            protocol_sha=protocol_sha,
            trainer_sha=trainer_sha,
            execution_protocol_sha=execution_protocol_sha,
            runtime_mode=args.runtime_mode,
        ):
            print(f"[{number}/{len(grid)}] SKIP certified {row['state_id']} {row['candidate']} {row['configuration_label']}", flush=True)
            continue
        pending.append((number, row.copy()))
    if warm_factor_cache and pending:
        warm_factor_caches(data_root, grid, protocol)
    if pending:
        print(
            f"EXECUTE pending={len(pending)} workers={workers} "
            f"threads_per_worker={threads_per_worker}",
            flush=True,
        )
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="phase1")
        futures = {}
        try:
            for number, row in pending:
                print(
                    f"[{number}/{len(grid)}] QUEUE {row['state_id']} "
                    f"{row['candidate']} {row['configuration_label']}",
                    flush=True,
                )
                future = executor.submit(
                    execute_run,
                    data_root,
                    code_root,
                    row,
                    runtime_mode=args.runtime_mode,
                    intra_op_threads=threads_per_worker,
                    inter_op_threads=inter_op_threads,
                )
                futures[future] = (number, row)
            for future in as_completed(futures):
                number, row = futures[future]
                future.result()
                print(
                    f"[{number}/{len(grid)}] DONE {row['state_id']} "
                    f"{row['candidate']} {row['configuration_label']}",
                    flush=True,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    provenance = summarize(data_root, code_root, grid, protocol, runtime)
    write_json(output / "phase1_status.json", {**provenance, "status": "COMPLETE"})
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
