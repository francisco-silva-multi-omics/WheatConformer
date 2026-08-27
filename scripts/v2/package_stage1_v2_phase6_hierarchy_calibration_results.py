"""Validate and package Stage-1 v2 hierarchy-calibration reporting artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.v2.package_stage1_v2_phase6_remediation_results import (
    add_tree_to_archive,
    bool_values,
    copy_artifact,
    git_commit,
    read_json,
    read_tsv,
    require_columns,
    require_files,
    require_sealed,
    sha256_file,
    validate_frozen_artifacts,
    write_tsv,
)


SUMMARY = Path("model_kernels/stage1_v2_phase6_hierarchy_calibration_v1/phase_1")
RUNS = Path("trained_models/stage1_v2_phase6_hierarchy_calibration_v1_runs/phase_1")
SOURCE_RUNS = Path("trained_models/stage1_v2_phase6_remediation_v1_runs/phase_1")
SOURCE_SUMMARY = Path("model_kernels/stage1_v2_phase6_remediation_v1/phase_1")
AUDIT = Path("audit/v2/stage1_v2_phase6_hierarchy_calibration_v1")
REPLAY = Path("audit/v2/stage1_v2_phase6_information_guard_replay_v1")
DEFAULT_OUTPUT = Path("audit/v2/stage1_v2_phase6_hierarchy_calibration_export_v1")

EXPECTED_STATUS = "PASS_STAGE1_V2_PHASE6_HIERARCHY_CALIBRATION_PHASE1_COMPLETE"
EXPECTED_LOCK = "PASS_FROZEN_BEFORE_HIERARCHY_CALIBRATION_INNER_SCREEN"
EXPECTED_PROTOCOL = "stage1_v2_phase6_hierarchy_calibration_v1"
EXPECTED_RUN_PROTOCOL = "stage1_v2_phase6_hierarchy_calibration_tf_v1"
EXPECTED_NEW_RUNS = 15
EXPECTED_SOURCE_RUNS = 10
EXPECTED_STATES = 5
REFERENCE = "historical_reaction_reference"
SOURCE_HIERARCHY = "known_environment_hierarchical_v2"
NEW_CANDIDATES = (
    "hierarchy_test_weight_identity_calibration_v1",
    "hierarchy_test_weight_group_crossfit_calibration_v1",
    "hierarchy_test_weight_group_crossfit_strong_head_v1",
)
EXPECTED_TRAITS = (
    "1000_GRAIN_WEIGHT",
    "ABOVE_GROUND_BIOMASS",
    "DAYS_TO_HEADING",
    "DAYS_TO_MATURITY",
    "GRAIN_YIELD",
    "PLANT_HEIGHT",
    "TEST_WEIGHT",
)
SUMMARY_FILES = (
    "hierarchy_calibration_run_grid.tsv",
    "hierarchy_calibration_runs.tsv",
    "hierarchy_calibration_paired_metrics.tsv",
    "hierarchy_calibration_paired_trait_metrics.tsv",
    "hierarchy_calibration_paired_guard_metrics.tsv",
    "hierarchy_calibration_decision.tsv",
    "PHASE1_HIERARCHY_CALIBRATION_DECISION.json",
    "hierarchy_calibration_status.json",
)
NEW_RUN_FILES = (
    "run_metadata.json",
    "trait_scaling.tsv",
    "component_epoch_history.tsv",
    "active_component_factors.tsv",
    "trial_environment_hierarchy_support.tsv",
    "training_only_calibration.tsv",
    "training_only_calibration_crossfit.tsv",
    "validation_trait_metrics.tsv",
    "validation_subset_metrics.tsv",
    "validation_guard_metrics.tsv",
)
SOURCE_RUN_FILES = (
    "run_metadata.json",
    "validation_trait_metrics.tsv",
    "validation_guard_metrics.tsv",
)
AUDIT_FILES = (
    "PHASE6_HIERARCHY_CALIBRATION_LOCK.json",
    "validation_checks.tsv",
)
REPLAY_FILES = (
    "INFORMATION_GUARD_REPLAY_DECISION.json",
    "information_guard_replay_paired.tsv",
    "information_guard_replay_summary.tsv",
    "validation_checks.tsv",
)
SOURCE_SUMMARY_FILES = (
    "PHASE1_STRUCTURAL_DECISION.json",
    "remediation_phase1_decision.tsv",
)
CODE_FILES = (
    "server_training_pipeline/stage1_v2_phase6_hierarchy_calibration_protocol_v1.json",
    "server_training_pipeline/stage1_v2_phase6_remediation_protocol_v1.json",
    "server_training_pipeline/stage1_v2_phase6_remediation.py",
    "server_training_pipeline/train_stage1_v2_phase6_hierarchy_calibration_tf.py",
    "server_training_pipeline/train_stage1_v2_phase6_tf.py",
    "server_training_pipeline/stage1_v2_trainer_interface.py",
    "scripts/v2/certify_stage1_v2_phase6_information_guard_replay.py",
    "scripts/v2/freeze_stage1_v2_phase6_hierarchy_calibration.py",
    "scripts/v2/run_stage1_v2_phase6_hierarchy_calibration.py",
    "scripts/v2/run_stage1_v2_phase6_hierarchy_calibration_server_cpu.sh",
    "scripts/v2/package_stage1_v2_phase6_hierarchy_calibration_results.py",
    "scripts/v2/package_stage1_v2_phase6_hierarchy_calibration_results.sh",
    "scripts/v2/package_stage1_v2_phase6_remediation_results.py",
    "requirements/stage1_v2_server_cpu_test_addons.txt",
    "tests/test_stage1_v2_phase6_hierarchy_calibration.py",
    "tests/test_stage1_v2_phase6_hierarchy_calibration_tf.py",
    "tests/test_stage1_v2_phase6_hierarchy_calibration_export.py",
)


def validate_summary(root: Path, code_root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    summary_root = root / SUMMARY
    audit_root = root / AUDIT
    require_files(summary_root, SUMMARY_FILES)
    require_files(audit_root, AUDIT_FILES)
    require_files(root / REPLAY, REPLAY_FILES)
    require_files(root / SOURCE_SUMMARY, SOURCE_SUMMARY_FILES)
    require_files(code_root, CODE_FILES)

    decision_path = summary_root / "PHASE1_HIERARCHY_CALIBRATION_DECISION.json"
    status_path = summary_root / "hierarchy_calibration_status.json"
    lock_path = audit_root / "PHASE6_HIERARCHY_CALIBRATION_LOCK.json"
    decision = read_json(decision_path)
    status = read_json(status_path)
    lock = read_json(lock_path)
    protocol = read_json(
        code_root
        / "server_training_pipeline/stage1_v2_phase6_hierarchy_calibration_protocol_v1.json"
    )
    for value, path in ((decision, decision_path), (status, status_path)):
        if value.get("status") != EXPECTED_STATUS:
            raise ValueError(f"Hierarchy calibration screen is incomplete: {path}")
        require_sealed(value, path)
        if value.get("outer_evaluation_allowed") is not False:
            raise ValueError(f"Outer evaluation was unexpectedly authorized: {path}")
    if lock.get("status") != EXPECTED_LOCK:
        raise ValueError("Hierarchy calibration lock is not PASS")
    require_sealed(lock, lock_path)
    validate_frozen_artifacts(lock)
    if protocol.get("protocol_version") != EXPECTED_PROTOCOL:
        raise ValueError("Unexpected hierarchy calibration protocol")

    grid = read_tsv(summary_root / "hierarchy_calibration_run_grid.tsv")
    runs = read_tsv(summary_root / "hierarchy_calibration_runs.tsv")
    paired = read_tsv(summary_root / "hierarchy_calibration_paired_metrics.tsv")
    traits = read_tsv(summary_root / "hierarchy_calibration_paired_trait_metrics.tsv")
    guards = read_tsv(summary_root / "hierarchy_calibration_paired_guard_metrics.tsv")
    decisions = read_tsv(summary_root / "hierarchy_calibration_decision.tsv")
    key = ["state_id", "candidate"]
    require_columns(grid, [*key, "scenario", "outer_fold", "inner_fold", "seed"], "grid")
    if len(grid) != EXPECTED_NEW_RUNS or grid.duplicated(key).any():
        raise ValueError("Hierarchy calibration grid is incomplete or duplicated")
    if grid["state_id"].nunique() != EXPECTED_STATES:
        raise ValueError("Hierarchy calibration grid does not contain five states")
    if set(grid["candidate"].astype(str)) != set(NEW_CANDIDATES):
        raise ValueError("Hierarchy calibration candidate grid changed")
    if set(grid["scenario"].astype(str)) != {"GNEW_EOBS"}:
        raise ValueError("Hierarchy calibration escaped GNEW_EOBS")
    if set(pd.to_numeric(grid["outer_fold"], errors="raise")) != {1}:
        raise ValueError("Hierarchy calibration escaped outer fold 1")
    if set(pd.to_numeric(grid["inner_fold"], errors="raise")) != {1, 2, 3, 4, 5}:
        raise ValueError("Hierarchy calibration lacks an inner fold")
    if not grid.groupby("state_id")["seed"].nunique().eq(1).all():
        raise ValueError("Hierarchy calibration seeds are not matched within state")

    expected_summary_runs = EXPECTED_NEW_RUNS + EXPECTED_SOURCE_RUNS
    require_columns(
        runs,
        [*key, "protocol_version", "validation_macro_normalized_rmse", "validation_macro_pearson"],
        "run summary",
    )
    if len(runs) != expected_summary_runs or runs.duplicated(key).any():
        raise ValueError("Hierarchy calibration run summary is incomplete or duplicated")
    if set(runs["candidate"].astype(str)) != {
        REFERENCE,
        SOURCE_HIERARCHY,
        *NEW_CANDIDATES,
    }:
        raise ValueError("Hierarchy calibration run summary has unexpected candidates")
    for metric in ("validation_macro_normalized_rmse", "validation_macro_pearson"):
        if not np.isfinite(pd.to_numeric(runs[metric], errors="coerce")).all():
            raise ValueError(f"Non-finite hierarchy calibration metric: {metric}")
    require_columns(
        paired,
        [*key, "validation_observation_signature", "validation_observation_signature_reference"],
        "paired metrics",
    )
    if len(paired) != expected_summary_runs or paired.duplicated(key).any():
        raise ValueError("Paired hierarchy calibration metrics are incomplete")
    if not paired["validation_observation_signature"].eq(
        paired["validation_observation_signature_reference"]
    ).all():
        raise ValueError("Paired hierarchy calibration observations differ")
    if traits.empty or guards.empty or decisions.empty:
        raise ValueError("Hierarchy calibration detail reporting is empty")
    if not set(traits["trait_name_canonical"].astype(str)).issubset(EXPECTED_TRAITS):
        raise ValueError("Unexpected hierarchy calibration trait")
    comparable_traits = traits["rows_reference"].notna()
    if not traits.loc[comparable_traits, "rows"].eq(
        traits.loc[comparable_traits, "rows_reference"]
    ).all():
        raise ValueError("Paired trait rows differ")
    comparable_guards = guards["rows"].gt(0)
    if not guards.loc[comparable_guards, "rows"].eq(
        guards.loc[comparable_guards, "rows_reference"]
    ).all():
        raise ValueError("Paired guard rows differ")
    if not guards.loc[comparable_guards, "observation_id_signature"].eq(
        guards.loc[comparable_guards, "observation_id_signature_reference"]
    ).all():
        raise ValueError("Paired guard observation identifiers differ")

    selected = decision.get("selected_candidate")
    allowed = bool(decision.get("full_125_state_confirmation_allowed"))
    selected_rows = decisions["decision"].eq("selected_for_full_125_state_confirmation")
    if allowed != (selected is not None) or int(selected_rows.sum()) != int(allowed):
        raise ValueError("Hierarchy calibration selection authorization is inconsistent")
    if selected is not None:
        if selected not in NEW_CANDIDATES:
            raise ValueError("An unregistered hierarchy calibration candidate was selected")
        if decisions.loc[selected_rows, "candidate"].tolist() != [selected]:
            raise ValueError("Decision table and selection JSON disagree")

    overview = {
        "status": "PASS_READY_TO_EXPORT",
        "protocol_version": "stage1_v2_phase6_hierarchy_calibration_export_v1",
        "stage1_version": "Stage-1 v2",
        "screen_status": EXPECTED_STATUS,
        "new_training_run_count": EXPECTED_NEW_RUNS,
        "source_comparator_run_count": EXPECTED_SOURCE_RUNS,
        "evaluated_run_count": expected_summary_runs,
        "state_count": EXPECTED_STATES,
        "paired_trait_rows": int(len(traits)),
        "paired_guard_rows": int(len(guards)),
        "selected_candidate": selected,
        "full_125_state_confirmation_allowed": allowed,
        "frozen_training_commit": str(lock.get("code_commit")),
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
    }
    return overview, grid


def validate_new_run(root: Path, row: Any, frozen_commit: str) -> dict[str, Any]:
    relative = Path(str(row.state_id)) / str(row.candidate)
    run_root = root / RUNS / relative
    require_files(run_root, NEW_RUN_FILES)
    metadata_path = run_root / "run_metadata.json"
    metadata = read_json(metadata_path)
    expected = {
        "status": "PASS",
        "protocol_version": EXPECTED_RUN_PROTOCOL,
        "state_id": str(row.state_id),
        "scenario": "GNEW_EOBS",
        "candidate": str(row.candidate),
        "code_commit": frozen_commit,
    }
    for key, value in expected.items():
        if str(metadata.get(key)) != value:
            raise ValueError(f"Run {relative} has unexpected {key}")
    if int(metadata.get("seed", -1)) != int(row.seed):
        raise ValueError(f"Run {relative} has an unexpected seed")
    require_sealed(metadata, metadata_path)
    if metadata.get("calibration_validation_values_used") is not False:
        raise ValueError(f"Run {relative} used validation values for calibration")
    if metadata.get("information_guard_reporting_corrected") is not True:
        raise ValueError(f"Run {relative} lacks corrected information guards")

    traits = read_tsv(run_root / "validation_trait_metrics.tsv")
    guards = read_tsv(run_root / "validation_guard_metrics.tsv")
    epochs = read_tsv(run_root / "component_epoch_history.tsv")
    calibration = read_tsv(run_root / "training_only_calibration.tsv")
    crossfit = read_tsv(run_root / "training_only_calibration_crossfit.tsv")
    if len(traits) != len(EXPECTED_TRAITS) or guards.empty or epochs.empty:
        raise ValueError(f"Run {relative} has incomplete reporting")
    if guards["observation_id_signature"].fillna("").astype(str).eq("").any():
        raise ValueError(f"Run {relative} has empty guard signatures")
    if calibration.empty or bool(bool_values(calibration["validation_values_used"]).any()):
        raise ValueError(f"Run {relative} lacks training-only calibration evidence")
    target = calibration.loc[calibration["trait_name_canonical"].eq("TEST_WEIGHT")]
    if len(target) != 1 or float(target.iloc[0]["slope"]) <= 0:
        raise ValueError(f"Run {relative} has invalid TEST_WEIGHT calibration")
    if "group_crossfit" in str(row.candidate):
        if len(crossfit) != 5 or bool(bool_values(crossfit["validation_values_used"]).any()):
            raise ValueError(f"Run {relative} has invalid cross-fit evidence")
    elif not crossfit.empty:
        raise ValueError(f"Identity-calibration run {relative} unexpectedly has cross-fit rows")
    return metadata


def copy_file(
    source: Path,
    destination: Path,
    *,
    source_root: Path,
    staging: Path,
    records: list[dict[str, Any]],
    category: str,
) -> None:
    copy_artifact(
        source,
        destination,
        source_root=source_root,
        package_root=staging,
        records=records,
        category=category,
    )


def build_export(root: Path, code_root: Path, output_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    overview, grid = validate_summary(root, code_root)
    lock = read_json(root / AUDIT / "PHASE6_HIERARCHY_CALIBRATION_LOCK.json")
    frozen_commit = str(lock["code_commit"])
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "stage1_v2_phase6_hierarchy_calibration_results.tar.gz"
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    with tempfile.TemporaryDirectory(prefix="hierarchy_calibration_export_", dir=output_dir) as temporary:
        staging = Path(temporary) / "stage1_v2_phase6_hierarchy_calibration_results"
        staging.mkdir(parents=True)
        records: list[dict[str, Any]] = []
        for name in SUMMARY_FILES:
            copy_file(root / SUMMARY / name, staging / "summary" / name, source_root=root, staging=staging, records=records, category="summary")
        for name in AUDIT_FILES:
            copy_file(root / AUDIT / name, staging / "freeze" / name, source_root=root, staging=staging, records=records, category="freeze")
        for name in REPLAY_FILES:
            copy_file(root / REPLAY / name, staging / "information_guard_replay" / name, source_root=root, staging=staging, records=records, category="information_guard_replay")
        for name in SOURCE_SUMMARY_FILES:
            copy_file(root / SOURCE_SUMMARY / name, staging / "source_remediation" / name, source_root=root, staging=staging, records=records, category="source_remediation")
        for relative in CODE_FILES:
            copy_file(code_root / relative, staging / "code_snapshot" / relative, source_root=code_root, staging=staging, records=records, category="code_snapshot")

        support_rows: list[dict[str, Any]] = []
        for row in grid.itertuples(index=False):
            relative = Path(str(row.state_id)) / str(row.candidate)
            metadata = validate_new_run(root, row, frozen_commit)
            support_rows.append(
                {
                    "state_id": row.state_id,
                    "candidate": row.candidate,
                    "test_weight_calibration_mode": metadata.get("test_weight_calibration_mode"),
                    "test_weight_calibration_status": metadata.get("test_weight_calibration_status"),
                    "test_weight_calibration_slope": metadata.get("test_weight_calibration_slope"),
                    "test_weight_crossfit_valid_folds": metadata.get("test_weight_crossfit_valid_folds"),
                    "test_weight_residual_scale_floor": metadata.get("test_weight_residual_scale_floor"),
                    "test_weight_trait_loading_penalty_multiplier": metadata.get("test_weight_trait_loading_penalty_multiplier"),
                    "validation_rows": metadata.get("validation_rows"),
                }
            )
            for name in NEW_RUN_FILES:
                copy_file(root / RUNS / relative / name, staging / "new_runs" / relative / name, source_root=root, staging=staging, records=records, category="new_run_reporting")

        states = sorted(grid["state_id"].astype(str).unique())
        for state_id in states:
            for candidate in (REFERENCE, SOURCE_HIERARCHY):
                relative = Path(state_id) / candidate
                for name in SOURCE_RUN_FILES:
                    copy_file(root / SOURCE_RUNS / relative / name, staging / "source_comparator_runs" / relative / name, source_root=root, staging=staging, records=records, category="source_comparator_reporting")

        support_path = staging / "summary" / "hierarchy_calibration_support.tsv"
        pd.DataFrame(support_rows).to_csv(support_path, sep="\t", index=False, lineterminator="\n")
        records.append(
            {
                "category": "derived_reporting",
                "source_path": "generated/hierarchy_calibration_support.tsv",
                "package_path": support_path.relative_to(staging).as_posix(),
                "bytes": int(support_path.stat().st_size),
                "sha256": sha256_file(support_path),
            }
        )
        overview["payload_artifact_count"] = int(len(records))
        overview["payload_bytes"] = int(sum(record["bytes"] for record in records))
        overview["active_exporter_code_commit"] = git_commit(code_root)
        (staging / "EXPORT_SUMMARY.json").write_text(
            json.dumps(overview, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_tsv(staging / "payload_manifest.tsv", records)
        add_tree_to_archive(staging, archive)
    archive_sha = sha256_file(archive)
    checksum.write_text(f"{archive_sha}  {archive.name}\n", encoding="ascii")
    overview.update(
        {
            "archive": str(archive),
            "archive_bytes": int(archive.stat().st_size),
            "archive_sha256": archive_sha,
            "checksum_file": str(checksum),
        }
    )
    return archive, checksum, overview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package hierarchy-calibration reporting results")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    code_root = (args.code_root or Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root))).resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else root / DEFAULT_OUTPUT
    archive, checksum, overview = build_export(root, code_root, output_dir)
    print(json.dumps(overview, indent=2, sort_keys=True))
    print(f"Archive: {archive}")
    print(f"Checksum: {checksum}")


if __name__ == "__main__":
    main()
