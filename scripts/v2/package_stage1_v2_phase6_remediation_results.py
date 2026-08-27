"""Validate and package Stage-1 v2 Phase-6 remediation reporting artifacts.

The export is inner-validation reporting only. It excludes phenotype tables,
predictions, checkpoints, factor caches, outer outcomes, and final-holdout data.
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


SUMMARY = Path("model_kernels/stage1_v2_phase6_remediation_v1/phase_1")
RUNS = Path("trained_models/stage1_v2_phase6_remediation_v1_runs/phase_1")
AUDIT = Path("audit/v2/stage1_v2_phase6_remediation_v1")
DEFAULT_OUTPUT = Path("audit/v2/stage1_v2_phase6_remediation_export_v1")

EXPECTED_STATUS = "PASS_STAGE1_V2_PHASE6_REMEDIATION_PHASE1_COMPLETE"
EXPECTED_LOCK = "PASS_FROZEN_BEFORE_REMEDIATION_INNER_VALIDATION"
EXPECTED_PROTOCOL = "stage1_v2_phase6_structural_remediation_v1"
EXPECTED_RUN_PROTOCOL = "stage1_v2_phase6_structural_remediation_tf_v1"
EXPECTED_RUNS = 70
EXPECTED_STATES = 25
EXPECTED_SCENARIOS = (
    "GNEW_EOBS",
    "GOBS_ENEW",
    "GNEW_ENEW",
    "TEMPORAL_YEAR",
    "COUNTRY_HOLDOUT",
)
REFERENCE = "historical_reaction_reference"
HIERARCHY = "known_environment_hierarchical_v2"
PROJECTION = "projection_output_routed_calibrated_v2"
MARKER = "marker_supported_output_routed_v2"
EXPECTED_CANDIDATES = (REFERENCE, HIERARCHY, PROJECTION, MARKER)
EXPECTED_TRAITS = (
    "1000_GRAIN_WEIGHT",
    "ABOVE_GROUND_BIOMASS",
    "DAYS_TO_HEADING",
    "DAYS_TO_MATURITY",
    "GRAIN_YIELD",
    "PLANT_HEIGHT",
    "TEST_WEIGHT",
)
SEALED_KEYS = (
    "outer_test_outcomes_read",
    "outer_test_metrics_read",
    "final_holdout_outcomes_read",
)
CORE_METRICS = (
    "validation_macro_normalized_rmse",
    "validation_macro_pearson",
    "validation_macro_calibration_error",
    "within_environment_centered_spearman",
    "within_environment_pairwise_accuracy",
)

SUMMARY_FILES = (
    "remediation_phase1_run_grid.tsv",
    "remediation_phase1_runs.tsv",
    "remediation_phase1_paired_metrics.tsv",
    "remediation_phase1_paired_trait_metrics.tsv",
    "remediation_phase1_paired_guard_metrics.tsv",
    "remediation_phase1_decision.tsv",
    "PHASE1_STRUCTURAL_DECISION.json",
    "remediation_phase1_status.json",
)
RUN_FILES = (
    "run_metadata.json",
    "trait_scaling.tsv",
    "component_epoch_history.tsv",
    "active_component_factors.tsv",
    "trial_environment_hierarchy_support.tsv",
    "training_only_calibration.tsv",
    "validation_trait_metrics.tsv",
    "validation_subset_metrics.tsv",
    "validation_guard_metrics.tsv",
)
AUDIT_FILES = (
    "PHASE6_REMEDIATION_LOCK.json",
    "validation_checks.tsv",
)
CODE_FILES = (
    "server_training_pipeline/stage1_v2_phase6_remediation_protocol_v1.json",
    "server_training_pipeline/stage1_v2_phase6_remediation.py",
    "server_training_pipeline/train_stage1_v2_phase6_remediation_tf.py",
    "scripts/v2/freeze_stage1_v2_phase6_remediation.py",
    "scripts/v2/run_stage1_v2_phase6_remediation.py",
    "scripts/v2/run_stage1_v2_phase6_remediation_server_cpu.sh",
    "scripts/v2/launch_stage1_v2_phase6_remediation_server_cpu.sh",
    "scripts/v2/show_stage1_v2_phase6_remediation_server_cpu_status.sh",
    "scripts/v2/package_stage1_v2_phase6_remediation_results.py",
    "scripts/v2/package_stage1_v2_phase6_remediation_results.sh",
    "tests/test_stage1_v2_phase6_remediation.py",
    "tests/test_stage1_v2_phase6_remediation_tf.py",
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


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def require_sealed(value: dict[str, Any], source: Path) -> None:
    for key in SEALED_KEYS:
        if value.get(key) is not False:
            raise ValueError(f"{source} does not certify {key}=false")


def bool_values(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def git_commit(code_root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "Unable to resolve Git commit")
    return process.stdout.strip()


def validate_frozen_artifacts(lock: dict[str, Any]) -> None:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Remediation lock lacks frozen artifact hashes")
    failures = []
    for label, record in artifacts.items():
        if not isinstance(record, dict):
            failures.append(f"invalid_record:{label}")
            continue
        path = Path(str(record.get("path", "")))
        expected = str(record.get("sha256", ""))
        if not path.is_file():
            failures.append(f"missing:{label}")
        elif sha256_file(path) != expected:
            failures.append(f"sha256:{label}")
    if failures:
        raise ValueError(f"Frozen remediation inputs changed: {failures[:10]}")


def expected_candidate_scenarios() -> set[tuple[str, str]]:
    pairs = {(scenario, REFERENCE) for scenario in EXPECTED_SCENARIOS}
    pairs.add(("GNEW_EOBS", HIERARCHY))
    pairs.update((scenario, PROJECTION) for scenario in EXPECTED_SCENARIOS[1:4])
    pairs.update((scenario, MARKER) for scenario in EXPECTED_SCENARIOS)
    return pairs


def validate_summary(
    root: Path, code_root: Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    summary_root = root / SUMMARY
    audit_root = root / AUDIT
    require_files(summary_root, SUMMARY_FILES)
    require_files(audit_root, AUDIT_FILES)
    require_files(code_root, CODE_FILES)

    status_path = summary_root / "remediation_phase1_status.json"
    decision_path = summary_root / "PHASE1_STRUCTURAL_DECISION.json"
    lock_path = audit_root / "PHASE6_REMEDIATION_LOCK.json"
    status = read_json(status_path)
    decision = read_json(decision_path)
    lock = read_json(lock_path)
    protocol = read_json(
        code_root
        / "server_training_pipeline/stage1_v2_phase6_remediation_protocol_v1.json"
    )
    for value, path in ((status, status_path), (decision, decision_path)):
        if value.get("status") != EXPECTED_STATUS:
            raise ValueError(f"Remediation screen is not complete: {path}")
        require_sealed(value, path)
        if value.get("outer_evaluation_allowed") is not False:
            raise ValueError(f"Outer evaluation was unexpectedly authorized: {path}")
    if lock.get("status") != EXPECTED_LOCK:
        raise ValueError("Remediation freeze lock is not PASS")
    require_sealed(lock, lock_path)
    if protocol.get("protocol_version") != EXPECTED_PROTOCOL:
        raise ValueError("Unexpected remediation protocol")
    if protocol.get("outer_test_metrics_read") is not False:
        raise ValueError("Protocol reports outer-test metric access")
    validate_frozen_artifacts(lock)

    grid = read_tsv(summary_root / "remediation_phase1_run_grid.tsv")
    runs = read_tsv(summary_root / "remediation_phase1_runs.tsv")
    paired = read_tsv(summary_root / "remediation_phase1_paired_metrics.tsv")
    traits = read_tsv(summary_root / "remediation_phase1_paired_trait_metrics.tsv")
    guards = read_tsv(summary_root / "remediation_phase1_paired_guard_metrics.tsv")
    decisions = read_tsv(summary_root / "remediation_phase1_decision.tsv")
    key = ["state_id", "candidate"]
    require_columns(
        grid,
        [*key, "scenario", "outer_fold", "inner_fold", "seed"],
        "remediation grid",
    )
    if len(grid) != EXPECTED_RUNS or grid.duplicated(key).any():
        raise ValueError("Remediation grid is incomplete or duplicated")
    if grid["state_id"].nunique() != EXPECTED_STATES:
        raise ValueError("Remediation grid does not contain 25 states")
    if set(map(tuple, grid[["scenario", "candidate"]].drop_duplicates().to_numpy())) != (
        expected_candidate_scenarios()
    ):
        raise ValueError("Remediation scenario/candidate grid differs from protocol")
    if not grid.groupby("state_id")["seed"].nunique().eq(1).all():
        raise ValueError("Candidate seeds are not matched within state")
    if set(pd.to_numeric(grid["outer_fold"], errors="raise")) != {1}:
        raise ValueError("Remediation screen escaped outer fold 1")
    if set(pd.to_numeric(grid["inner_fold"], errors="raise")) != {1, 2, 3, 4, 5}:
        raise ValueError("Remediation screen lacks one or more inner folds")

    require_columns(runs, [*key, "scenario", "protocol_version", *CORE_METRICS], "runs")
    if len(runs) != EXPECTED_RUNS or runs.duplicated(key).any():
        raise ValueError("Remediation run summary is incomplete or duplicated")
    if not runs["protocol_version"].eq(EXPECTED_RUN_PROTOCOL).all():
        raise ValueError("Unexpected remediation run implementation")
    if not grid[key].merge(runs[key], on=key, how="left", indicator=True)[
        "_merge"
    ].eq("both").all():
        raise ValueError("Remediation grid and run summary disagree")
    for metric in CORE_METRICS:
        values = pd.to_numeric(runs[metric], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite remediation metric: {metric}")
    if bool(bool_values(runs["outer_test_metrics_read"]).any()):
        raise ValueError("Run summary reports outer-test metric access")
    if bool(bool_values(runs["final_holdout_outcomes_read"]).any()):
        raise ValueError("Run summary reports final-holdout access")
    if not runs["code_commit"].astype(str).eq(str(lock.get("code_commit"))).all():
        raise ValueError("Remediation runs do not use the frozen code commit")

    require_columns(
        paired,
        [*key, "validation_observation_signature", "validation_observation_signature_reference"],
        "paired metrics",
    )
    if len(paired) != EXPECTED_RUNS or paired.duplicated(key).any():
        raise ValueError("Paired remediation metrics are incomplete or duplicated")
    if not paired["validation_observation_signature"].eq(
        paired["validation_observation_signature_reference"]
    ).all():
        raise ValueError("Paired remediation observations differ")
    if traits.empty or guards.empty or decisions.empty:
        raise ValueError("Remediation trait, guard, or decision reporting is empty")
    require_columns(
        traits,
        [*key, "trait_name_canonical", "rows", "rows_reference"],
        "paired trait metrics",
    )
    if not set(traits["trait_name_canonical"].astype(str)).issubset(EXPECTED_TRAITS):
        raise ValueError("Unexpected trait in remediation reporting")
    available_traits = traits["rows_reference"].notna()
    if not traits.loc[available_traits, "rows"].eq(
        traits.loc[available_traits, "rows_reference"]
    ).all():
        raise ValueError("Paired trait metrics use unequal row counts")
    comparable_guards = guards["rows"].gt(0)
    if not guards.loc[comparable_guards, "rows"].eq(
        guards.loc[comparable_guards, "rows_reference"]
    ).all():
        raise ValueError("Paired guard metrics use unequal row counts")
    if not guards.loc[comparable_guards, "observation_id_signature"].eq(
        guards.loc[comparable_guards, "observation_id_signature_reference"]
    ).all():
        raise ValueError("Paired guard metrics use unequal observation IDs")

    advanced = decisions["decision"].eq("advance_to_full_125_state_confirmation")
    expected_advanced = decisions.loc[advanced, ["scenario", "candidate"]].to_dict(
        "records"
    )
    if status.get("advanced_candidates") != expected_advanced:
        raise ValueError("Status and decision table disagree on advancing candidates")
    if bool(status.get("phase2_optimizer_allowed")) != bool(advanced.any()):
        raise ValueError("Optimizer-screen authorization disagrees with decisions")

    overview = {
        "status": "PASS_READY_TO_EXPORT",
        "protocol_version": "stage1_v2_phase6_remediation_export_v1",
        "stage1_version": "Stage-1 v2",
        "screen_status": EXPECTED_STATUS,
        "run_count": EXPECTED_RUNS,
        "state_count": EXPECTED_STATES,
        "scenario_count": len(EXPECTED_SCENARIOS),
        "candidate_count": len(EXPECTED_CANDIDATES),
        "paired_trait_rows": int(len(traits)),
        "paired_guard_rows": int(len(guards)),
        "decision_rows": int(len(decisions)),
        "advanced_candidates": expected_advanced,
        "phase2_optimizer_allowed": bool(advanced.any()),
        "frozen_training_commit": str(lock.get("code_commit")),
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
    }
    return overview, grid


def validate_run(root: Path, row: Any, frozen_commit: str) -> dict[str, Any]:
    relative = Path(str(row.state_id)) / str(row.candidate)
    run_root = root / RUNS / relative
    require_files(run_root, RUN_FILES)
    metadata_path = run_root / "run_metadata.json"
    metadata = read_json(metadata_path)
    expected = {
        "status": "PASS",
        "protocol_version": EXPECTED_RUN_PROTOCOL,
        "state_id": str(row.state_id),
        "scenario": str(row.scenario),
        "candidate": str(row.candidate),
        "code_commit": frozen_commit,
    }
    for key, value in expected.items():
        if str(metadata.get(key)) != value:
            raise ValueError(f"Run {relative} has unexpected {key}")
    if int(metadata.get("seed", -1)) != int(row.seed):
        raise ValueError(f"Run {relative} has an unexpected seed")
    require_sealed(metadata, metadata_path)
    if metadata.get("phenotype_values_read") is not True:
        raise ValueError(f"Run {relative} did not certify inner phenotype use")
    if metadata.get("inner_validation_metrics_read") is not True:
        raise ValueError(f"Run {relative} did not certify inner metric use")
    if metadata.get("calibration_validation_values_used") is not False:
        raise ValueError(f"Run {relative} used validation values for calibration")
    if metadata.get("guard_mask_observation_signatures_written") is not True:
        raise ValueError(f"Run {relative} omitted guard signatures")

    traits = read_tsv(run_root / "validation_trait_metrics.tsv")
    guards = read_tsv(run_root / "validation_guard_metrics.tsv")
    epochs = read_tsv(run_root / "component_epoch_history.tsv")
    calibration = read_tsv(run_root / "training_only_calibration.tsv")
    require_columns(traits, ["trait_name_canonical"], f"traits for {relative}")
    if traits.empty or len(traits) > len(EXPECTED_TRAITS):
        raise ValueError(f"Run {relative} has invalid trait availability")
    if guards.empty or epochs.empty:
        raise ValueError(f"Run {relative} has empty guard or convergence reporting")
    if guards["observation_id_signature"].fillna("").astype(str).eq("").any():
        raise ValueError(f"Run {relative} has empty guard signatures")
    if str(row.candidate) == REFERENCE:
        if not calibration.empty:
            raise ValueError(f"Stable reference unexpectedly contains calibration: {relative}")
    else:
        if calibration.empty or not bool_values(
            calibration["validation_values_used"]
        ).eq(False).all():
            raise ValueError(f"Run {relative} lacks training-only calibration evidence")
    if str(row.candidate) in {PROJECTION, MARKER}:
        if metadata.get("fallback_predictions_preserved_exactly") is not True:
            raise ValueError(f"Routed run lacks exact fallback certification: {relative}")
        active = int(metadata.get("active_route_validation_rows", -1))
        fallback = int(metadata.get("fallback_validation_rows", -1))
        validation = int(metadata.get("validation_rows", -1))
        if active < 1 or fallback < 0 or active + fallback != validation:
            raise ValueError(f"Routed support accounting is invalid: {relative}")
    return metadata


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
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
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
    overview, grid = validate_summary(root, code_root)
    lock = read_json(root / AUDIT / "PHASE6_REMEDIATION_LOCK.json")
    frozen_commit = str(lock["code_commit"])
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "stage1_v2_phase6_remediation_results.tar.gz"
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    with tempfile.TemporaryDirectory(
        prefix="remediation_export_", dir=output_dir
    ) as temporary_name:
        staging = Path(temporary_name) / "stage1_v2_phase6_remediation_results"
        staging.mkdir(parents=True)
        records: list[dict[str, Any]] = []
        for name in SUMMARY_FILES:
            copy_artifact(
                root / SUMMARY / name,
                staging / "summary" / name,
                source_root=root,
                package_root=staging,
                records=records,
                category="summary",
            )
        for name in AUDIT_FILES:
            copy_artifact(
                root / AUDIT / name,
                staging / "freeze" / name,
                source_root=root,
                package_root=staging,
                records=records,
                category="freeze",
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
        support_rows = []
        for row in grid.itertuples(index=False):
            relative = Path(str(row.state_id)) / str(row.candidate)
            metadata = validate_run(root, row, frozen_commit)
            support_rows.append(
                {
                    "state_id": row.state_id,
                    "scenario": row.scenario,
                    "candidate": row.candidate,
                    "active_route_training_rows": metadata.get(
                        "active_route_training_rows"
                    ),
                    "active_route_validation_rows": metadata.get(
                        "active_route_validation_rows"
                    ),
                    "fallback_validation_rows": metadata.get(
                        "fallback_validation_rows"
                    ),
                    "fallback_predictions_preserved_exactly": metadata.get(
                        "fallback_predictions_preserved_exactly"
                    ),
                    "positive_training_calibration_fitted": metadata.get(
                        "positive_training_calibration_fitted"
                    ),
                    "trait_specific_regularization": metadata.get(
                        "trait_specific_regularization"
                    ),
                }
            )
            for name in RUN_FILES:
                copy_artifact(
                    root / RUNS / relative / name,
                    staging / "runs" / relative / name,
                    source_root=root,
                    package_root=staging,
                    records=records,
                    category="run_reporting",
                )
        support_path = staging / "summary" / "remediation_route_support.tsv"
        pd.DataFrame(support_rows).to_csv(support_path, sep="\t", index=False)
        records.append(
            {
                "category": "derived_reporting",
                "source_path": "generated/remediation_route_support.tsv",
                "package_path": support_path.relative_to(staging).as_posix(),
                "bytes": int(support_path.stat().st_size),
                "sha256": sha256_file(support_path),
            }
        )
        overview["payload_artifact_count"] = int(len(records))
        overview["payload_bytes"] = int(sum(row["bytes"] for row in records))
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
    parser = argparse.ArgumentParser(
        description="Package Stage-1 v2 Phase-6 remediation reporting results"
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
        args.output_dir.resolve() if args.output_dir else root / DEFAULT_OUTPUT
    )
    archive, checksum, overview = build_export(root, code_root, output_dir)
    print(json.dumps(overview, indent=2, sort_keys=True))
    print(f"Archive: {archive}")
    print(f"Checksum: {checksum}")


if __name__ == "__main__":
    main()
