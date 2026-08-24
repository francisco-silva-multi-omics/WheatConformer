"""Validate and package the completed Stage-1 v2 Phase-1 guard replay.

The export contains reporting artifacts only. It deliberately excludes model
weights, factor caches, row-level predictions, outer outcomes, and holdout data.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PARENT_SUMMARY = Path("model_kernels/stage1_v2_phase6_phase1_v2")
REPLAY_SUMMARY = Path(
    "model_kernels/stage1_v2_phase6_phase1_guard_replay_v1"
)
PARENT_RUNS = Path("trained_models/stage1_v2_phase6_phase1_v2_runs")
REPLAY_RUNS = Path(
    "trained_models/stage1_v2_phase6_phase1_guard_replay_v1_runs"
)
REPLAY_AUDIT = Path("audit/v2/stage1_v2_phase6_phase1_guard_replay_v1")
DEFAULT_OUTPUT = Path(
    "audit/v2/stage1_v2_phase6_phase1_guard_replay_export_v1"
)

REPLAY_PROTOCOL = "stage1_v2_phase6_phase1_matched_guard_replay_v1"
RUN_PROTOCOL = "stage1_v2_phase6_phase1_guard_replay_v1"
BASELINE = "ka_identity_location_baseline"
EXPECTED_RUNS = 120
EXPECTED_STATES = 5
EXPECTED_CANDIDATES = 8
EXPECTED_CONFIGURATIONS = 3
FULL_VALIDATION_METRICS = (
    "validation_macro_normalized_rmse",
    "validation_macro_pearson",
    "validation_macro_calibration_error",
    "within_environment_centered_spearman",
    "within_environment_pairwise_accuracy",
)

REPLAY_SUMMARY_FILES = (
    "phase1_status.json",
    "phase1_provenance.json",
    "phase1_run_grid.tsv",
    "phase1_runs.tsv",
    "phase1_replay_runs.tsv",
    "phase1_paired_metrics.tsv",
    "phase1_trait_metrics.tsv",
    "phase1_replay_trait_metrics.tsv",
    "phase1_subset_metrics.tsv",
    "phase1_guard_metrics.tsv",
    "phase1_paired_guard_metrics.tsv",
    "phase1_parent_metric_replay_audit.tsv",
    "phase1_decision.tsv",
)
PARENT_SUMMARY_FILES = (
    "phase1_status.json",
    "phase1_provenance.json",
    "phase1_run_grid.tsv",
    "phase1_runs.tsv",
    "phase1_paired_metrics.tsv",
    "phase1_trait_metrics.tsv",
    "phase1_subset_metrics.tsv",
    "phase1_decision.tsv",
)
PARENT_RUN_FILES = (
    "run_metadata.json",
    "epoch_history.tsv",
    "trait_scaling.tsv",
    "validation_trait_metrics.tsv",
    "validation_subset_metrics.tsv",
    "active_component_factors.tsv",
)
REPLAY_RUN_FILES = (*PARENT_RUN_FILES, "validation_guard_metrics.tsv")
AUDIT_FILES = (
    "PHASE1_GUARD_REPLAY_LOCK.json",
    "validation_checks.tsv",
)
CODE_FILES = (
    "server_training_pipeline/stage1_v2_phase6_selection_protocol_v1.json",
    "server_training_pipeline/stage1_v2_phase6_execution_protocol_v2.json",
    "server_training_pipeline/stage1_v2_phase6_server_cpu_runtime_v1.json",
    "server_training_pipeline/stage1_v2_trainer_interface.py",
    "server_training_pipeline/train_stage1_v2_phase6_tf.py",
    "scripts/v2/freeze_stage1_v2_phase6_phase1_guard_replay.py",
    "scripts/v2/run_stage1_v2_phase6_phase1.py",
    "scripts/v2/run_stage1_v2_phase6_phase1_guard_replay_server_cpu.sh",
)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def require_files(root: Path, names: Iterable[str]) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts under {root}: {missing}")


def require_false(value: dict[str, Any], key: str, source: Path) -> None:
    if value.get(key) is not False:
        raise ValueError(f"{source} does not certify {key}=false")


def validate_summary(root: Path) -> dict[str, Any]:
    replay_root = root / REPLAY_SUMMARY
    parent_root = root / PARENT_SUMMARY
    require_files(replay_root, REPLAY_SUMMARY_FILES)
    require_files(parent_root, PARENT_SUMMARY_FILES)
    require_files(root / REPLAY_AUDIT, AUDIT_FILES)

    status = read_json(replay_root / "phase1_status.json")
    provenance = read_json(replay_root / "phase1_provenance.json")
    if status.get("status") != "COMPLETE":
        raise ValueError(f"Guard replay is not complete: {status.get('status')}")
    if status.get("protocol_version") != REPLAY_PROTOCOL:
        raise ValueError("Guard replay status uses an unexpected protocol")
    if provenance.get("status") != "PASS":
        raise ValueError("Guard replay provenance is not PASS")
    if provenance.get("protocol_version") != REPLAY_PROTOCOL:
        raise ValueError("Guard replay provenance uses an unexpected protocol")
    for value, source in (
        (status, replay_root / "phase1_status.json"),
        (provenance, replay_root / "phase1_provenance.json"),
    ):
        for key in (
            "outer_test_outcomes_read",
            "outer_test_metrics_read",
            "final_holdout_outcomes_read",
        ):
            require_false(value, key, source)
    expected_statuses = {
        "matched_seed_status": "pass",
        "matched_validation_observation_status": "pass",
        "matched_component_mask_status": "pass",
        "parent_full_metric_replay_status": "pass",
    }
    for key, expected in expected_statuses.items():
        if provenance.get(key) != expected:
            raise ValueError(f"Replay provenance failed {key}: {provenance.get(key)}")
    if provenance.get("h_seeds_direct_marker_support_included") is not True:
        raise ValueError("H_SEEDS direct marker support was not certified")
    if provenance.get("projection_core_mask_candidate_independent") is not True:
        raise ValueError("Projection-core masks were not candidate-independent")
    if int(provenance.get("run_count", -1)) != EXPECTED_RUNS:
        raise ValueError("Replay provenance does not contain 120 runs")

    grid = read_tsv(replay_root / "phase1_run_grid.tsv")
    runs = read_tsv(replay_root / "phase1_runs.tsv")
    replay_runs = read_tsv(replay_root / "phase1_replay_runs.tsv")
    decision = read_tsv(replay_root / "phase1_decision.tsv")
    paired_guards = read_tsv(replay_root / "phase1_paired_guard_metrics.tsv")
    replay_audit = read_tsv(
        replay_root / "phase1_parent_metric_replay_audit.tsv"
    )

    run_key = ["state_id", "candidate", "configuration_label"]
    if len(grid) != EXPECTED_RUNS or grid.duplicated(run_key).any():
        raise ValueError("Replay run grid is incomplete or duplicated")
    observed_shape = (
        grid["state_id"].nunique(),
        grid["candidate"].nunique(),
        grid["configuration_label"].nunique(),
    )
    if observed_shape != (
        EXPECTED_STATES,
        EXPECTED_CANDIDATES,
        EXPECTED_CONFIGURATIONS,
    ):
        raise ValueError(f"Unexpected replay grid shape: {observed_shape}")
    for label, frame in (("parent runs", runs), ("replay runs", replay_runs)):
        if len(frame) != EXPECTED_RUNS or frame.duplicated(run_key).any():
            raise ValueError(f"{label} are incomplete or duplicated")
    if len(decision) != EXPECTED_CANDIDATES * EXPECTED_CONFIGURATIONS:
        raise ValueError("Decision table does not cover all candidate configurations")

    expected_audit_rows = EXPECTED_RUNS * len(FULL_VALIDATION_METRICS)
    if len(replay_audit) != expected_audit_rows:
        raise ValueError(
            f"Parent metric replay audit has {len(replay_audit)} rows; "
            f"expected {expected_audit_rows}"
        )
    deltas = pd.to_numeric(replay_audit["absolute_delta"], errors="coerce")
    if not np.isfinite(deltas).all() or float(deltas.max()) > 1e-5:
        raise ValueError("Parent full-validation metrics were not exactly replayed")

    rows = pd.to_numeric(paired_guards["rows"], errors="coerce")
    reference_rows = pd.to_numeric(
        paired_guards["rows_reference"], errors="coerce"
    )
    paired = (
        rows.eq(reference_rows)
        & paired_guards["observation_id_signature"].eq(
            paired_guards["observation_id_signature_reference"]
        )
    )
    if paired_guards.empty or not bool(paired.all()):
        raise ValueError("Candidate/reference guard rows are not exactly paired")

    h_seeds_marker = paired_guards.loc[
        paired_guards["candidate"].astype(str).str.startswith("h_seeds_")
        & paired_guards["subset"].eq("MARKER_SUPPORTED")
    ].copy()
    expected_h_seeds_rows = 2 * EXPECTED_CONFIGURATIONS * EXPECTED_STATES
    if len(h_seeds_marker) != expected_h_seeds_rows:
        raise ValueError(
            "H_SEEDS marker-supported guard coverage is incomplete: "
            f"observed={len(h_seeds_marker)} expected={expected_h_seeds_rows}"
        )
    h_seeds_counts = pd.to_numeric(h_seeds_marker["rows"], errors="coerce")
    if h_seeds_counts.isna().any() or not h_seeds_counts.gt(0).all():
        raise ValueError("H_SEEDS still reports empty marker-supported subsets")

    advancing = decision.loc[
        decision["decision"].eq("advance_to_confirmation"),
        ["candidate", "configuration_label"],
    ]
    return {
        "status": "PASS_READY_TO_EXPORT",
        "protocol_version": "stage1_v2_phase6_phase1_guard_replay_export_v1",
        "code_commit": str(provenance.get("code_commit", "")),
        "run_count": int(len(grid)),
        "state_count": int(grid["state_id"].nunique()),
        "candidate_count": int(grid["candidate"].nunique()),
        "configuration_count": int(grid["configuration_label"].nunique()),
        "decision_row_count": int(len(decision)),
        "advancing_candidate_configuration_count": int(len(advancing)),
        "advancing_candidate_configurations": [
            {
                "candidate": str(row.candidate),
                "configuration_label": str(row.configuration_label),
            }
            for row in advancing.itertuples(index=False)
        ],
        "paired_guard_row_count": int(len(paired_guards)),
        "h_seeds_marker_supported_guard_row_count": int(len(h_seeds_marker)),
        "parent_full_metric_maximum_absolute_delta": float(deltas.max()),
        "matched_component_mask_status": "pass",
        "parent_full_metric_replay_status": "pass",
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
    }


def git_commit(code_root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "Unable to identify code commit")
    return process.stdout.strip()


def copy_artifact(
    source: Path,
    destination: Path,
    *,
    source_root: Path,
    package_root: Path,
    records: list[dict[str, Any]],
    category: str,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    records.append(
        {
            "category": category,
            "source_path": source.relative_to(source_root).as_posix(),
            "package_path": destination.relative_to(package_root).as_posix(),
            "bytes": int(source.stat().st_size),
            "sha256": sha256_file(source),
        }
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def add_tree_to_archive(source: Path, archive: Path) -> None:
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w") as tar:
                for path in sorted(source.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = Path(source.name) / path.relative_to(source)
                    info = tar.gettarinfo(str(path), arcname=relative.as_posix())
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
    os.replace(temporary, archive)


def build_export(
    root: Path, code_root: Path, output_dir: Path
) -> tuple[Path, Path, dict[str, Any]]:
    summary = validate_summary(root)
    active_commit = git_commit(code_root)
    lock = read_json(root / REPLAY_AUDIT / "PHASE1_GUARD_REPLAY_LOCK.json")
    if lock.get("status") != "PASS_READY_FOR_PHASE1_MATCHED_GUARD_REPLAY":
        raise ValueError("The frozen guard-replay lock is not ready")
    for relative, expected in lock.get("implementation_sha256", {}).items():
        path = code_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Active implementation differs from replay lock: {relative}")
    selection_protocol = (
        code_root
        / "server_training_pipeline/stage1_v2_phase6_selection_protocol_v1.json"
    )
    if sha256_file(selection_protocol) != lock.get("selection_protocol_sha256"):
        raise ValueError("Active selection protocol differs from replay lock")
    require_files(code_root, CODE_FILES)

    replay_grid = read_tsv(root / REPLAY_SUMMARY / "phase1_run_grid.tsv")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "stage1_v2_phase6_phase1_guard_replay_results.tar.gz"
    checksum = archive.with_suffix(archive.suffix + ".sha256")

    with tempfile.TemporaryDirectory(
        prefix="guard_replay_export_", dir=output_dir
    ) as temporary_name:
        staging = Path(temporary_name) / "stage1_v2_phase6_phase1_guard_replay_results"
        staging.mkdir(parents=True)
        records: list[dict[str, Any]] = []

        for name in PARENT_SUMMARY_FILES:
            copy_artifact(
                root / PARENT_SUMMARY / name,
                staging / "parent_summary" / name,
                source_root=root,
                package_root=staging,
                records=records,
                category="parent_summary",
            )
        for name in REPLAY_SUMMARY_FILES:
            copy_artifact(
                root / REPLAY_SUMMARY / name,
                staging / "guard_replay_summary" / name,
                source_root=root,
                package_root=staging,
                records=records,
                category="guard_replay_summary",
            )
        for name in AUDIT_FILES:
            copy_artifact(
                root / REPLAY_AUDIT / name,
                staging / "guard_replay_lock" / name,
                source_root=root,
                package_root=staging,
                records=records,
                category="guard_replay_lock",
            )
        for relative in CODE_FILES:
            copy_artifact(
                code_root / relative,
                staging / "code_snapshot" / relative,
                source_root=code_root,
                package_root=staging,
                records=records,
                category="code_snapshot",
            )

        for row in replay_grid.itertuples(index=False):
            relative_run = Path(str(row.state_id)) / str(row.candidate) / str(
                row.configuration_label
            )
            for name in PARENT_RUN_FILES:
                copy_artifact(
                    root / PARENT_RUNS / relative_run / name,
                    staging / "runs" / "parent" / relative_run / name,
                    source_root=root,
                    package_root=staging,
                    records=records,
                    category="parent_run",
                )
            for name in REPLAY_RUN_FILES:
                copy_artifact(
                    root / REPLAY_RUNS / relative_run / name,
                    staging / "runs" / "guard_replay" / relative_run / name,
                    source_root=root,
                    package_root=staging,
                    records=records,
                    category="guard_replay_run",
                )
            replay_metadata = read_json(
                root / REPLAY_RUNS / relative_run / "run_metadata.json"
            )
            if replay_metadata.get("status") != "PASS":
                raise ValueError(f"Replay run is not PASS: {relative_run}")
            if replay_metadata.get("protocol_version") != RUN_PROTOCOL:
                raise ValueError(f"Replay run protocol mismatch: {relative_run}")
            if replay_metadata.get("guard_mask_observation_signatures_written") is not True:
                raise ValueError(f"Replay run omitted guard signatures: {relative_run}")
            for key in (
                "outer_test_outcomes_read",
                "outer_test_metrics_read",
                "final_holdout_outcomes_read",
            ):
                require_false(
                    replay_metadata,
                    key,
                    root / REPLAY_RUNS / relative_run / "run_metadata.json",
                )

        summary["payload_artifact_count"] = int(len(records))
        summary["payload_bytes"] = int(sum(row["bytes"] for row in records))
        summary["active_code_commit"] = active_commit
        (staging / "EXPORT_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_tsv(staging / "payload_manifest.tsv", records)
        add_tree_to_archive(staging, archive)

    archive_sha = sha256_file(archive)
    checksum.write_text(
        f"{archive_sha}  {archive.name}\n", encoding="ascii"
    )
    summary["archive"] = str(archive)
    summary["archive_bytes"] = int(archive.stat().st_size)
    summary["archive_sha256"] = archive_sha
    summary["checksum_file"] = str(checksum)
    return archive, checksum, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the completed Stage-1 v2 Phase-1 matched-guard replay"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    code_root = (
        args.code_root
        or Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root))
    ).resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root / DEFAULT_OUTPUT
    )
    archive, checksum, summary = build_export(root, code_root, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Archive: {archive}")
    print(f"Checksum: {checksum}")


if __name__ == "__main__":
    main()
