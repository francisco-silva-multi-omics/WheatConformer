from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
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
RUNTIME = Path("server_training_pipeline/stage1_v2_training_runtime_v1.json")
OUTPUT = Path("model_kernels/stage1_v2_phase6_phase1_v1")
RUNS = Path("trained_models/stage1_v2_phase6_phase1_v1_runs")


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


def validate_runtime(root: Path) -> dict[str, Any]:
    import tensorflow as tf

    runtime = json.loads((root / RUNTIME).read_text(encoding="utf-8"))
    observed = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "tensorflow": tf.__version__,
        "pandas": pd.__version__,
        "gpu_count": len(tf.config.list_physical_devices("GPU")),
    }
    for key in ("python", "tensorflow", "pandas"):
        if observed[key] != runtime[key]:
            raise ValueError(
                f"Certified WSL runtime mismatch for {key}: observed={observed[key]} expected={runtime[key]}"
            )
    if runtime.get("tensorflow_gpu_required_for_training") and observed["gpu_count"] < 1:
        raise ValueError("Certified Phase-1 training requires a visible TensorFlow GPU")
    return observed


def validate_handoff(root: Path) -> dict[str, Any]:
    handoff = json.loads((root / HANDOFF).read_text(encoding="utf-8"))
    if handoff.get("status") != "PASS_READY_FOR_STAGE1_V2_PHASE6_INNER_MODEL_SELECTION":
        raise ValueError("Aggregate Phase-6 handoff is not ready")
    if handoff.get("code_commit") != git_commit(root):
        raise ValueError("Aggregate Phase-6 handoff is not bound to the active commit")
    if handoff.get("outer_evaluation_allowed") is not False:
        raise ValueError("Phase-1 handoff unexpectedly permits outer evaluation")
    expected = handoff.get("phase1_implementation_sha256", {})
    for relative in (TRAINER, ORCHESTRATOR, LAUNCHER):
        observed = sha256_file(root / relative)
        if expected.get(relative.as_posix()) != observed:
            raise ValueError(f"Frozen Phase-1 implementation mismatch: {relative}")
    protocol_path = root / "server_training_pipeline/stage1_v2_phase6_selection_protocol_v1.json"
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


def metadata_matches(path: Path, row: pd.Series, *, commit: str, protocol_sha: str, trainer_sha: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("status") == "PASS"
        and value.get("state_id") == row["state_id"]
        and value.get("candidate") == row["candidate"]
        and value.get("configuration_label") == row["configuration_label"]
        and int(value.get("seed", -1)) == int(row["seed"])
        and value.get("code_commit") == commit
        and value.get("selection_protocol_sha256") == protocol_sha
        and value.get("trainer_sha256") == trainer_sha
        and value.get("outer_test_outcomes_read") is False
        and value.get("outer_test_metrics_read") is False
        and value.get("final_holdout_outcomes_read") is False
    )


def execute_run(root: Path, row: pd.Series) -> None:
    destination = run_dir(root, row)
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "server_training_pipeline.train_stage1_v2_phase6_tf",
        "--root",
        str(root),
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
    with (destination / "run.log").open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=root,
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


def _mean(values: Iterable[object]) -> float:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def summarize(root: Path, grid: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, Any]:
    metadata_rows = []
    trait_frames = []
    subset_frames = []
    for _, row in grid.iterrows():
        destination = run_dir(root, row)
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
    runs = pd.DataFrame(metadata_rows)
    traits = pd.concat(trait_frames, ignore_index=True)
    subsets = pd.concat(subset_frames, ignore_index=True)
    baseline = "ka_identity_location_baseline"
    metrics = [
        "validation_macro_normalized_rmse",
        "validation_macro_pearson",
        "validation_macro_calibration_error",
        "within_environment_centered_spearman",
        "within_environment_pairwise_accuracy",
    ]
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
            float(local_subsets.groupby("subset")["relative_nrmse_gain"].mean().min())
            if not local_subsets.empty else float("nan")
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
                "accepted": bool(accepted),
                "decision": "reference" if is_reference else (
                    "advance_to_confirmation" if accepted else "do_not_advance"
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["validation_normalized_rmse_mean", "candidate", "configuration_label"]
    )
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output / "phase1_run_grid.tsv", sep="\t", index=False)
    runs.to_csv(output / "phase1_runs.tsv", sep="\t", index=False)
    paired.to_csv(output / "phase1_paired_metrics.tsv", sep="\t", index=False)
    traits.to_csv(output / "phase1_trait_metrics.tsv", sep="\t", index=False)
    subsets.to_csv(output / "phase1_subset_metrics.tsv", sep="\t", index=False)
    summary.to_csv(output / "phase1_decision.tsv", sep="\t", index=False)
    provenance = {
        "status": "PASS",
        "protocol_version": "stage1_v2_phase6_phase1_screen_v1",
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
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "outer_evaluation_allowed": False,
        "selection_protocol_sha256": sha256_file(
            root / "server_training_pipeline/stage1_v2_phase6_selection_protocol_v1.json"
        ),
        "trainer_sha256": sha256_file(root / TRAINER),
        "code_commit": git_commit(root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if provenance["matched_validation_observation_status"] != "pass":
        raise ValueError("Candidates did not use identical validation observations")
    write_json(output / "phase1_provenance.json", provenance)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Stage-1 v2 Phase-1 inner screen")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol = load_selection_protocol(root)
    grid = phase1_grid(protocol)
    handoff = validate_handoff(root)
    runtime = validate_runtime(root)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output / "phase1_run_grid.tsv", sep="\t", index=False)
    startup = {
        "status": "RUNNING",
        "protocol_version": "stage1_v2_phase6_phase1_screen_v1",
        "run_count": len(grid),
        "code_commit": git_commit(root),
        "runtime": runtime,
        "handoff_sha256": sha256_file(root / HANDOFF),
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "phase1_status.json", startup)
    print(json.dumps(startup, indent=2, sort_keys=True), flush=True)
    commit = str(handoff["code_commit"])
    protocol_sha = str(handoff["selection_protocol_sha256"])
    trainer_sha = sha256_file(root / TRAINER)
    for number, (_, row) in enumerate(grid.iterrows(), start=1):
        metadata_path = run_dir(root, row) / "run_metadata.json"
        if args.resume and metadata_matches(
            metadata_path,
            row,
            commit=commit,
            protocol_sha=protocol_sha,
            trainer_sha=trainer_sha,
        ):
            print(f"[{number}/{len(grid)}] SKIP certified {row['state_id']} {row['candidate']} {row['configuration_label']}", flush=True)
            continue
        print(f"[{number}/{len(grid)}] TRAIN {row['state_id']} {row['candidate']} {row['configuration_label']}", flush=True)
        execute_run(root, row)
    provenance = summarize(root, grid, protocol)
    write_json(output / "phase1_status.json", {**provenance, "status": "COMPLETE"})
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
