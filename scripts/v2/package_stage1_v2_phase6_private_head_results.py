from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY = Path("model_kernels/stage1_v2_phase6_private_head_screen_v1/phase_1")
RUNS = Path("trained_models/stage1_v2_phase6_private_head_screen_v1_runs")
REPLAY_RUNS = Path(
    "trained_models/stage1_v2_phase6_private_head_screen_v1_same_seed_replay_runs"
)
SOURCE_RUNS = Path(
    "trained_models/stage1_v2_phase6_hierarchy_calibration_amendment_v2_runs"
)
AUDIT = Path("audit/v2/stage1_v2_phase6_private_head_screen_v1")
DEFAULT_OUTPUT = Path("exports/stage1_v2_phase6_private_head_phase1_results_v1.tar.gz")
EXPECTED_STATUSES = {
    "PASS_STAGE1_V2_PHASE6_PRIVATE_HEAD_PHASE1_CANDIDATE_SELECTED",
    "PASS_STAGE1_V2_PHASE6_PRIVATE_HEAD_PHASE1_COMPLETE_NO_ADVANCE",
}
REFERENCE = "current_huber_authoritative_row_mass"
SOURCE_REFERENCE = "hierarchy_test_weight_environment_oof_huber_v2"
NEW_CANDIDATES = {
    "trait_private_residual_heads",
    "family_shared_trait_private_residual_heads",
}
SUMMARY_FILES = (
    "private_head_state_grid.tsv",
    "private_head_runs.tsv",
    "private_head_trait_metrics.tsv",
    "private_head_guard_metrics.tsv",
    "private_head_paired_metrics.tsv",
    "private_head_paired_trait_metrics.tsv",
    "private_head_paired_guard_metrics.tsv",
    "private_head_decision.tsv",
    "private_head_same_seed_replay.tsv",
    "private_head_factor_cache_prewarm.tsv",
    "PRIVATE_HEAD_PHASE1_DECISION.json",
    "private_head_status.json",
)
RUN_FILES = (
    "run_metadata.json",
    "trait_scaling.tsv",
    "authoritative_row_mass_diagnostics.tsv",
    "decoder_parameter_inventory.tsv",
    "training_only_calibration.tsv",
    "training_only_calibration_crossfit.tsv",
    "validation_trait_metrics.tsv",
    "validation_subset_metrics.tsv",
    "validation_guard_metrics.tsv",
    "component_epoch_history.tsv",
    "active_component_factors.tsv",
    "trial_environment_hierarchy_support.tsv",
    "best_model_weight_manifest.tsv",
    "component_replay_manifest.tsv",
    "training_observation_ids.npy",
    "validation_observation_ids.npy",
    "training_predictions_raw.npy",
    "validation_predictions_raw.npy",
    "validation_predictions_calibrated.npy",
)
SOURCE_RUN_FILES = (
    "run_metadata.json",
    "validation_trait_metrics.tsv",
    "validation_guard_metrics.tsv",
)
CODE_FILES = (
    "server_training_pipeline/stage1_v2_phase6_private_head_screen_protocol_v1.json",
    "server_training_pipeline/train_stage1_v2_phase6_private_heads_tf.py",
    "server_training_pipeline/train_stage1_v2_phase6_remediation_tf.py",
    "server_training_pipeline/train_stage1_v2_phase6_tf.py",
    "server_training_pipeline/stage1_v2_phase6_post_hierarchy_screen_plan_v2.json",
    "scripts/v2/freeze_stage1_v2_phase6_private_head_screen.py",
    "scripts/v2/run_stage1_v2_phase6_private_head_screen.py",
    "scripts/v2/run_stage1_v2_phase6_private_head_screen_server_cpu.sh",
    "scripts/v2/show_stage1_v2_phase6_private_head_screen_server_cpu_status.sh",
    "scripts/v2/package_stage1_v2_phase6_private_head_results.py",
    "scripts/v2/package_stage1_v2_phase6_private_head_results.sh",
    "tests/test_stage1_v2_phase6_private_head_screen.py",
    "tests/test_stage1_v2_phase6_private_head_tf.py",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_files(root: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Private-head export inputs are missing: {missing}")


def certified_run_files(directory: Path) -> list[Path]:
    require_files(directory, RUN_FILES)
    metadata = read_json(directory / "run_metadata.json")
    names = {Path(name) for name in RUN_FILES}
    names.update(Path(name) for name in metadata.get("artifacts", {}))
    for name, expected in metadata.get("artifacts", {}).items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Run artifact checksum mismatch: {path}")
    return sorted(names, key=lambda value: value.as_posix())


def git_commit(code_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Unable to resolve Git commit")
    return result.stdout.strip()


def validate(root: Path, code_root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    require_files(root / SUMMARY, SUMMARY_FILES)
    require_files(root / AUDIT, ("PRIVATE_HEAD_SCREEN_LOCK.json", "validation_checks.tsv"))
    require_files(code_root, CODE_FILES)
    decision = read_json(root / SUMMARY / "PRIVATE_HEAD_PHASE1_DECISION.json")
    status = read_json(root / SUMMARY / "private_head_status.json")
    lock = read_json(root / AUDIT / "PRIVATE_HEAD_SCREEN_LOCK.json")
    for value in (decision, status):
        if value.get("status") not in EXPECTED_STATUSES:
            raise ValueError("Private-head screen is incomplete")
        if value.get("outer_test_metrics_read") is not False:
            raise ValueError("Private-head screen read outer metrics")
        if value.get("outer_test_outcomes_read") is not False:
            raise ValueError("Private-head screen read outer outcomes")
        if value.get("final_holdout_outcomes_read") is not False:
            raise ValueError("Private-head screen read final-holdout outcomes")
    if decision != status:
        raise ValueError("Private-head decision and status differ")
    if lock.get("status") != "PASS_FROZEN_STAGE1_V2_PHASE6_PRIVATE_HEAD_SCREEN_V1":
        raise ValueError("Private-head lock is not PASS")
    if lock.get("outer_test_metrics_read") is not False:
        raise ValueError("Private-head lock read outer metrics")
    for name, expected in decision["artifacts"].items():
        if sha256_file(root / SUMMARY / name) != expected:
            raise ValueError(f"Private-head summary checksum mismatch: {name}")

    runs = pd.read_csv(root / SUMMARY / "private_head_runs.tsv", sep="\t")
    candidate_runs = runs.loc[runs["candidate"].isin(NEW_CANDIDATES)].copy()
    if len(runs) != 15 or len(candidate_runs) != 10:
        raise ValueError("Private-head run inventory is incomplete")
    if runs["state_id"].nunique() != 5:
        raise ValueError("Private-head run inventory lacks a state")
    if set(runs["candidate"].astype(str)) != {REFERENCE, *NEW_CANDIDATES}:
        raise ValueError("Private-head run inventory has an unexpected candidate")
    if not runs.groupby("state_id")["seed"].nunique().eq(1).all():
        raise ValueError("Private-head seeds are not matched")
    if not candidate_runs["decoder_variable_count"].gt(0).all():
        raise ValueError("A private-head run has no decoder variables")
    if not candidate_runs["authoritative_row_mass_changed"].eq(False).all():
        raise ValueError("A private-head run changed authoritative row mass")

    for row in candidate_runs.itertuples(index=False):
        directory = root / RUNS / str(row.state_id) / str(row.candidate)
        run_files = certified_run_files(directory)
        metadata = read_json(directory / "run_metadata.json")
        if metadata.get("protocol_version") != (
            "stage1_v2_phase6_private_heads_tf_v2_integrity_hardened"
        ):
            raise ValueError(f"Unexpected run protocol: {directory}")
        if not run_files:
            raise ValueError(f"Private-head run has no certified artifacts: {directory}")
    replay = pd.read_csv(root / SUMMARY / "private_head_same_seed_replay.tsv", sep="\t")
    if len(replay) != 2 or not replay["status"].eq("PASS").all():
        raise ValueError("Private-head same-seed replay is incomplete")
    for row in replay.itertuples(index=False):
        certified_run_files(root / REPLAY_RUNS / str(row.state_id) / str(row.candidate))
    for state_id in sorted(runs["state_id"].astype(str).unique()):
        require_files(root / SOURCE_RUNS / state_id / SOURCE_REFERENCE, SOURCE_RUN_FILES)

    overview = {
        "status": "PASS_READY_TO_EXPORT",
        "protocol_version": "stage1_v2_phase6_private_head_results_export_v1",
        "stage1_version": "Stage-1 v2",
        "screen_status": decision["status"],
        "state_count": 5,
        "new_model_fit_count": 10,
        "reference_reuse_count": 5,
        "selected_candidate": decision.get("selected_candidate"),
        "full_confirmation_allowed": decision.get("full_confirmation_allowed"),
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "code_commit": git_commit(code_root),
    }
    return overview, runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and package Stage-1 v2 private-head Phase-1 results"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = (args.code_root or root).resolve()
    output = (args.output or (root / DEFAULT_OUTPUT)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    overview, runs = validate(root, code_root)

    files: list[tuple[Path, Path]] = []
    files.extend((root / SUMMARY / name, SUMMARY / name) for name in SUMMARY_FILES)
    files.extend(
        (root / AUDIT / name, AUDIT / name)
        for name in ("PRIVATE_HEAD_SCREEN_LOCK.json", "validation_checks.tsv")
    )
    candidate_runs = runs.loc[runs["candidate"].isin(NEW_CANDIDATES)]
    for row in candidate_runs.itertuples(index=False):
        relative = RUNS / str(row.state_id) / str(row.candidate)
        files.extend(
            (root / relative / name, relative / name)
            for name in certified_run_files(root / relative)
        )
    replay = pd.read_csv(root / SUMMARY / "private_head_same_seed_replay.tsv", sep="\t")
    for row in replay.itertuples(index=False):
        relative = REPLAY_RUNS / str(row.state_id) / str(row.candidate)
        files.extend(
            (root / relative / name, relative / name)
            for name in certified_run_files(root / relative)
        )
    for state_id in sorted(runs["state_id"].astype(str).unique()):
        relative = SOURCE_RUNS / state_id / SOURCE_REFERENCE
        files.extend(
            (root / relative / name, relative / name) for name in SOURCE_RUN_FILES
        )
    files.extend((code_root / name, Path(name)) for name in CODE_FILES)

    overview_path = output.parent / "private_head_export_overview.json"
    overview_path.write_text(
        json.dumps(overview, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files.append((overview_path, Path("private_head_export_overview.json")))
    with tarfile.open(output, "w:gz") as archive:
        for source, relative in files:
            archive.add(source, arcname=relative.as_posix(), recursive=False)

    digest = sha256_file(output)
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    contents = output.with_suffix(output.suffix + ".contents.tsv")
    pd.DataFrame(
        [
            {
                "path": relative.as_posix(),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
            for source, relative in files
        ]
    ).to_csv(contents, sep="\t", index=False, lineterminator="\n")
    commit_path = output.with_suffix(output.suffix + ".code_commit.txt")
    commit_path.write_text(overview["code_commit"] + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                **overview,
                "archive": str(output),
                "archive_sha256": digest,
                "file_count": len(files),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
