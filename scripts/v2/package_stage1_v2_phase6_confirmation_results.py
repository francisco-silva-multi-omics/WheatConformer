"""Validate and package completed Stage-1 v2 Phase-6 confirmation results.

The export is reporting-only. It contains inner-validation summaries and
per-run reporting metadata, while excluding checkpoints, factor caches,
row-level predictions, outer outcomes, and final-holdout material.
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


SUMMARY = Path("model_kernels/stage1_v2_phase6_confirmation_v1")
RUNS = Path("trained_models/stage1_v2_phase6_confirmation_v1_runs")
AUDIT = Path("audit/v2/stage1_v2_phase6_confirmation_v1")
DEFAULT_OUTPUT = Path("audit/v2/stage1_v2_phase6_confirmation_export_v1")

EXPECTED_STATUS = "PASS_STAGE1_V2_PHASE6_CONFIRMATION_COMPLETE"
EXPECTED_SUMMARY_PROTOCOL = (
    "stage1_v2_phase6_confirmation_summary_v2_split_corrected"
)
EXPECTED_ROUTE_STATUS = "PASS_STAGE1_V2_PHASE6_SCENARIO_ROUTES_FROZEN"
EXPECTED_ROUTE_PROTOCOL = (
    "stage1_v2_phase6_confirmation_route_lock_v2_split_corrected"
)
EXPECTED_LOCK_STATUS = "PASS_READY_FOR_STAGE1_V2_PHASE6_CONFIRMATION"
CURRENT_RUN_PROTOCOL = (
    "stage1_v2_phase6_confirmation_tf_v4_masked_reaction_corrected"
)
LEGACY_RUN_PROTOCOL = "stage1_v2_phase6_confirmation_tf_v2_split_corrected"

EXPECTED_STATES = 125
EXPECTED_RUNS = 375
EXPECTED_CANDIDATES = (
    "historical_reaction_reference",
    "historical_v2_native_multikernel",
    "projection_reaction_routed_fallback",
)
EXPECTED_SCENARIOS = (
    "GNEW_EOBS",
    "GOBS_ENEW",
    "GNEW_ENEW",
    "TEMPORAL_YEAR",
    "COUNTRY_HOLDOUT",
)
LEGACY_SCENARIOS = frozenset(EXPECTED_SCENARIOS[:3])
TRANSFER_SCENARIOS = frozenset(EXPECTED_SCENARIOS[3:])
TRAITS_PER_RUN = 7
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
    "confirmation_run_grid.tsv",
    "confirmation_state_role_preflight.tsv",
    "confirmation_runs.tsv",
    "confirmation_paired_metrics.tsv",
    "confirmation_trait_metrics.tsv",
    "confirmation_paired_trait_metrics.tsv",
    "confirmation_guard_metrics.tsv",
    "confirmation_paired_guard_metrics.tsv",
    "confirmation_factor_inventory.tsv",
    "confirmation_run_implementation_inventory.tsv",
    "confirmation_scenario_summary.tsv",
    "confirmation_scenario_routes.tsv",
    "CONFIRMATION_SCENARIO_ROUTE_LOCK.json",
    "confirmation_provenance.json",
    "confirmation_status.json",
)
RUN_FILES = (
    "run_metadata.json",
    "epoch_history.tsv",
    "trait_scaling.tsv",
    "validation_trait_metrics.tsv",
    "validation_subset_metrics.tsv",
    "validation_guard_metrics.tsv",
    "active_component_factors.tsv",
)
AUDIT_FILES = (
    "PHASE6_CONFIRMATION_LOCK.json",
    "validation_checks.tsv",
)
CODE_FILES = (
    "server_training_pipeline/stage1_v2_phase6_confirmation_protocol_v1.json",
    "server_training_pipeline/stage1_v2_phase6_selection_protocol_v1.json",
    "server_training_pipeline/stage1_v2_phase6_execution_protocol_v2.json",
    "server_training_pipeline/stage1_v2_phase6_server_cpu_runtime_v1.json",
    "server_training_pipeline/stage1_v2_phase6_confirmation_execution_correction_v2.json",
    "server_training_pipeline/stage1_v2_phase6_confirmation_execution_correction_v3.json",
    "server_training_pipeline/stage1_v2_phase6_confirmation_execution_correction_v4.json",
    "server_training_pipeline/stage1_v2_trainer_interface.py",
    "server_training_pipeline/train_stage1_v2_phase6_tf.py",
    "server_training_pipeline/train_stage1_v2_phase6_confirmation_tf.py",
    "scripts/v2/freeze_stage1_v2_phase6_confirmation.py",
    "scripts/v2/run_stage1_v2_phase6_confirmation.py",
    "scripts/v2/run_stage1_v2_phase6_confirmation_server_cpu.sh",
    "scripts/v2/show_stage1_v2_phase6_confirmation_server_cpu_status.sh",
    "scripts/v2/package_stage1_v2_phase6_confirmation_results.py",
    "scripts/v2/package_stage1_v2_phase6_confirmation_results.sh",
)
CORE_METRICS = (
    "validation_macro_normalized_rmse",
    "validation_macro_pearson",
    "validation_macro_calibration_error",
    "within_environment_centered_spearman",
    "within_environment_pairwise_accuracy",
)
SEALED_KEYS = (
    "outer_test_outcomes_read",
    "outer_test_metrics_read",
    "final_holdout_outcomes_read",
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
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


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


def validate_result_hashes(
    summary_root: Path, source: dict[str, Any], source_path: Path
) -> None:
    hashes = source.get("result_artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"{source_path} has no result artifact hash inventory")
    for name, expected in hashes.items():
        path = summary_root / str(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"Result artifact differs from {source_path.name}: {name}"
            )


def validate_run_protocol(
    metadata: dict[str, Any], scenario: str, correction: dict[str, Any]
) -> str:
    protocol = str(metadata.get("protocol_version", ""))
    legacy = correction.get("legacy_run_compatibility", {})
    if protocol == CURRENT_RUN_PROTOCOL:
        return "current_v4"
    if (
        protocol == LEGACY_RUN_PROTOCOL
        and legacy.get("allowed") is True
        and scenario in set(legacy.get("allowed_scenarios", []))
        and scenario in LEGACY_SCENARIOS
    ):
        expected = {
            "code_commit": legacy.get("legacy_code_commit"),
            "trainer_sha256": legacy.get("legacy_confirmation_trainer_sha256"),
            "execution_correction_sha256": legacy.get(
                "legacy_execution_correction_sha256"
            ),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Legacy run does not match frozen {key}")
        return "legacy_v2"
    raise ValueError(
        f"Run protocol is not eligible for scenario {scenario}: {protocol}"
    )


def build_trait_availability(
    grid: pd.DataFrame, traits: pd.DataFrame
) -> pd.DataFrame:
    key = ["state_id", "candidate", "trait_name_canonical"]
    require_columns(traits, key, "trait metrics")
    if traits.duplicated(key).any():
        raise ValueError("Trait metrics contain duplicate state/candidate/trait rows")
    observed_traits = set(traits["trait_name_canonical"].astype(str))
    if not observed_traits.issubset(set(EXPECTED_TRAITS)):
        raise ValueError(
            f"Trait metrics contain unexpected traits: "
            f"{sorted(observed_traits - set(EXPECTED_TRAITS))}"
        )
    if observed_traits != set(EXPECTED_TRAITS):
        raise ValueError("At least one frozen trait is absent from every state")

    states = grid[
        ["state_id", "scenario", "outer_fold", "inner_fold"]
    ].drop_duplicates("state_id")
    expected = states.assign(_join=1).merge(
        pd.DataFrame({"trait_name_canonical": EXPECTED_TRAITS, "_join": 1}),
        on="_join",
    ).drop(columns="_join")
    counts = (
        traits.groupby(["state_id", "trait_name_canonical"])["candidate"]
        .nunique()
        .rename("available_candidate_count")
        .reset_index()
    )
    availability = expected.merge(
        counts,
        on=["state_id", "trait_name_canonical"],
        how="left",
        validate="one_to_one",
    )
    availability["available_candidate_count"] = (
        availability["available_candidate_count"].fillna(0).astype(int)
    )
    invalid = ~availability["available_candidate_count"].isin(
        {0, len(EXPECTED_CANDIDATES)}
    )
    if bool(invalid.any()):
        raise ValueError(
            "Trait availability differs among matched candidates:\n"
            + availability.loc[invalid].head(20).to_string(index=False)
        )
    availability["availability_status"] = np.where(
        availability["available_candidate_count"].eq(len(EXPECTED_CANDIDATES)),
        "AVAILABLE_ALL_CANDIDATES",
        "UNAVAILABLE_IN_STATE",
    )
    return availability


def validate_summary(
    root: Path, code_root: Path
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    summary_root = root / SUMMARY
    audit_root = root / AUDIT
    require_files(summary_root, SUMMARY_FILES)
    require_files(audit_root, AUDIT_FILES)
    require_files(code_root, CODE_FILES)

    status_path = summary_root / "confirmation_status.json"
    provenance_path = summary_root / "confirmation_provenance.json"
    route_lock_path = summary_root / "CONFIRMATION_SCENARIO_ROUTE_LOCK.json"
    freeze_lock_path = audit_root / "PHASE6_CONFIRMATION_LOCK.json"
    status = read_json(status_path)
    provenance = read_json(provenance_path)
    route_lock = read_json(route_lock_path)
    freeze_lock = read_json(freeze_lock_path)
    correction = read_json(
        code_root
        / "server_training_pipeline/"
        "stage1_v2_phase6_confirmation_execution_correction_v4.json"
    )
    protocol = read_json(
        code_root
        / "server_training_pipeline/stage1_v2_phase6_confirmation_protocol_v1.json"
    )

    for value, path in ((status, status_path), (provenance, provenance_path)):
        if value.get("status") != EXPECTED_STATUS:
            raise ValueError(f"Confirmation is not complete: {path}")
        if value.get("protocol_version") != EXPECTED_SUMMARY_PROTOCOL:
            raise ValueError(f"Unexpected confirmation summary protocol: {path}")
        require_sealed(value, path)
    if route_lock.get("status") != EXPECTED_ROUTE_STATUS:
        raise ValueError("Scenario route lock is not PASS")
    if route_lock.get("protocol_version") != EXPECTED_ROUTE_PROTOCOL:
        raise ValueError("Unexpected scenario route-lock protocol")
    require_sealed(route_lock, route_lock_path)
    if freeze_lock.get("status") != EXPECTED_LOCK_STATUS:
        raise ValueError("Confirmation freeze lock is not ready")
    require_sealed(freeze_lock, freeze_lock_path)

    expected_counts = {
        "state_count": EXPECTED_STATES,
        "run_count": EXPECTED_RUNS,
        "candidate_count": len(EXPECTED_CANDIDATES),
        "scenario_count": len(EXPECTED_SCENARIOS),
    }
    for key, expected in expected_counts.items():
        if int(provenance.get(key, -1)) != expected:
            raise ValueError(f"Confirmation {key} is not {expected}")
    for key in (
        "matched_seed_status",
        "matched_validation_observation_status",
        "matched_component_mask_status",
    ):
        if provenance.get(key) != "pass":
            raise ValueError(f"Confirmation failed {key}")

    if freeze_lock.get("code_commit") != provenance.get("code_commit"):
        raise ValueError("Confirmation summary does not use the frozen code commit")
    implementation_hashes = freeze_lock.get("implementation_sha256", {})
    if not isinstance(implementation_hashes, dict) or not implementation_hashes:
        raise ValueError("Confirmation freeze lacks implementation hashes")
    for relative, expected in implementation_hashes.items():
        path = code_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Active implementation differs from freeze: {relative}")
    selection_protocol = (
        code_root
        / "server_training_pipeline/stage1_v2_phase6_confirmation_protocol_v1.json"
    )
    if sha256_file(selection_protocol) != freeze_lock.get(
        "selection_protocol_sha256"
    ):
        raise ValueError("Active confirmation protocol differs from freeze")
    if correction.get("protocol_version") != (
        "stage1_v2_phase6_confirmation_execution_correction_v4"
    ):
        raise ValueError("Unexpected confirmation correction protocol")

    validate_result_hashes(summary_root, route_lock, route_lock_path)
    validate_result_hashes(summary_root, provenance, provenance_path)

    grid = read_tsv(summary_root / "confirmation_run_grid.tsv")
    runs = read_tsv(summary_root / "confirmation_runs.tsv")
    paired = read_tsv(summary_root / "confirmation_paired_metrics.tsv")
    traits = read_tsv(summary_root / "confirmation_trait_metrics.tsv")
    paired_traits = read_tsv(
        summary_root / "confirmation_paired_trait_metrics.tsv"
    )
    guards = read_tsv(summary_root / "confirmation_guard_metrics.tsv")
    paired_guards = read_tsv(
        summary_root / "confirmation_paired_guard_metrics.tsv"
    )
    factors = read_tsv(summary_root / "confirmation_factor_inventory.tsv")
    implementations = read_tsv(
        summary_root / "confirmation_run_implementation_inventory.tsv"
    )
    scenario_summary = read_tsv(
        summary_root / "confirmation_scenario_summary.tsv"
    )
    routes = read_tsv(summary_root / "confirmation_scenario_routes.tsv")
    preflight = read_tsv(summary_root / "confirmation_state_role_preflight.tsv")

    key = ["state_id", "candidate"]
    require_columns(
        grid,
        [*key, "scenario", "outer_fold", "inner_fold", "configuration_label", "seed"],
        "confirmation grid",
    )
    if len(grid) != EXPECTED_RUNS or grid.duplicated(key).any():
        raise ValueError("Confirmation grid is incomplete or duplicated")
    if grid["state_id"].nunique() != EXPECTED_STATES:
        raise ValueError("Confirmation grid does not contain 125 states")
    if set(grid["scenario"].astype(str)) != set(EXPECTED_SCENARIOS):
        raise ValueError("Confirmation grid scenario set is incorrect")
    if set(grid["candidate"].astype(str)) != set(EXPECTED_CANDIDATES):
        raise ValueError("Confirmation grid candidate set is incorrect")
    if not grid.groupby("state_id")["candidate"].nunique().eq(3).all():
        raise ValueError("Every state must contain all three candidates")
    if not grid.groupby("state_id")["seed"].nunique().eq(1).all():
        raise ValueError("Candidate seeds are not matched within state")

    require_columns(runs, [*key, "scenario", "protocol_version", *CORE_METRICS], "runs")
    if len(runs) != EXPECTED_RUNS or runs.duplicated(key).any():
        raise ValueError("Confirmation run summary is incomplete or duplicated")
    merged = grid.merge(runs[key], on=key, how="left", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Confirmation grid and run summary disagree")
    for metric in CORE_METRICS:
        values = pd.to_numeric(runs[metric], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite confirmation metric: {metric}")

    if len(paired) != EXPECTED_RUNS or paired.duplicated(key).any():
        raise ValueError("Paired metric table is incomplete or duplicated")
    require_columns(
        paired,
        ["validation_observation_signature", "validation_observation_signature_reference"],
        "paired metrics",
    )
    if not paired["validation_observation_signature"].eq(
        paired["validation_observation_signature_reference"]
    ).all():
        raise ValueError("Paired validation observations do not match")

    trait_availability = build_trait_availability(grid, traits)
    trait_key = ["state_id", "candidate", "trait_name_canonical"]
    require_columns(paired_traits, trait_key, "paired trait metrics")
    if paired_traits.duplicated(trait_key).any():
        raise ValueError("Paired trait metrics contain duplicate keys")
    observed_trait_keys = set(map(tuple, traits[trait_key].astype(str).to_numpy()))
    paired_trait_keys = set(
        map(tuple, paired_traits[trait_key].astype(str).to_numpy())
    )
    if observed_trait_keys != paired_trait_keys:
        raise ValueError("Trait and paired-trait availability disagree")
    if guards.empty or paired_guards.empty or factors.empty:
        raise ValueError("Guard or factor reporting artifacts are empty")
    if len(scenario_summary) != len(EXPECTED_SCENARIOS) * len(EXPECTED_CANDIDATES):
        raise ValueError("Scenario summary does not cover every candidate route")
    if int(pd.to_numeric(implementations["run_count"], errors="coerce").sum()) != EXPECTED_RUNS:
        raise ValueError("Implementation inventory does not account for 375 runs")

    require_columns(routes, ["scenario", "selected_candidate"], "scenario routes")
    if len(routes) != len(EXPECTED_SCENARIOS) or not routes["scenario"].is_unique:
        raise ValueError("Scenario route table is incomplete or duplicated")
    route_map = dict(zip(routes["scenario"], routes["selected_candidate"]))
    if set(route_map) != set(EXPECTED_SCENARIOS):
        raise ValueError("Scenario route table has an unexpected scenario set")
    if not set(route_map.values()).issubset(set(EXPECTED_CANDIDATES)):
        raise ValueError("Scenario route table selected an unknown candidate")
    if route_map != route_lock.get("selected_scenario_routes"):
        raise ValueError("Scenario routes differ from the frozen route lock")
    if route_map != provenance.get("selected_scenario_routes"):
        raise ValueError("Scenario routes differ from provenance")

    require_columns(preflight, ["state_id", "status"], "state preflight")
    if (
        len(preflight) != EXPECTED_STATES
        or not preflight["state_id"].is_unique
        or not preflight["status"].eq("PASS").all()
    ):
        raise ValueError("State role preflight did not pass for all 125 states")

    if "reaction_disabled_by_component_mask" not in runs:
        raise ValueError("Run summary lacks masked-reaction reporting")
    masked = bool_values(runs["reaction_disabled_by_component_mask"])
    historical_temporal = (
        runs["candidate"].eq("historical_reaction_reference")
        & runs["scenario"].eq("TEMPORAL_YEAR")
        & masked
    )
    expected_masked = int(
        correction["masked_reaction_policy"]["states_without_stage_components"]
    )
    if int(historical_temporal.sum()) != expected_masked:
        raise ValueError(
            "Masked temporal reaction count differs from the frozen correction: "
            f"observed={int(historical_temporal.sum())} expected={expected_masked}"
        )
    historical_outside_temporal = (
        runs["candidate"].eq("historical_reaction_reference")
        & ~runs["scenario"].eq("TEMPORAL_YEAR")
        & masked
    )
    if bool(historical_outside_temporal.any()):
        raise ValueError("Historical reaction was masked outside temporal states")

    implementation_counts = (
        runs.groupby("protocol_version").size().astype(int).to_dict()
    )
    if set(implementation_counts) - {CURRENT_RUN_PROTOCOL, LEGACY_RUN_PROTOCOL}:
        raise ValueError("Unexpected run implementation in confirmation summary")
    legacy_rows = runs["protocol_version"].eq(LEGACY_RUN_PROTOCOL)
    if not set(runs.loc[legacy_rows, "scenario"]).issubset(LEGACY_SCENARIOS):
        raise ValueError("Legacy confirmation runs leaked into transfer scenarios")
    if not runs.loc[runs["scenario"].isin(TRANSFER_SCENARIOS), "protocol_version"].eq(
        CURRENT_RUN_PROTOCOL
    ).all():
        raise ValueError("Temporal/country confirmation runs are not v4")
    if int(provenance.get("legacy_compatible_run_count", -1)) != int(
        legacy_rows.sum()
    ):
        raise ValueError("Legacy run count differs from confirmation provenance")

    overview = {
        "status": "PASS_READY_TO_EXPORT",
        "protocol_version": "stage1_v2_phase6_confirmation_export_v1",
        "stage1_version": "Stage-1 v2",
        "confirmation_code_commit": str(provenance.get("code_commit", "")),
        "run_count": EXPECTED_RUNS,
        "state_count": EXPECTED_STATES,
        "candidate_count": len(EXPECTED_CANDIDATES),
        "scenario_count": len(EXPECTED_SCENARIOS),
        "trait_metric_rows": int(len(traits)),
        "maximum_possible_trait_metric_rows": EXPECTED_RUNS * TRAITS_PER_RUN,
        "unavailable_candidate_trait_rows": int(
            EXPECTED_RUNS * TRAITS_PER_RUN - len(traits)
        ),
        "unavailable_state_trait_pairs": int(
            trait_availability["availability_status"]
            .eq("UNAVAILABLE_IN_STATE")
            .sum()
        ),
        "guard_metric_rows": int(len(guards)),
        "paired_guard_metric_rows": int(len(paired_guards)),
        "factor_inventory_rows": int(len(factors)),
        "masked_historical_temporal_state_count": int(historical_temporal.sum()),
        "run_counts_by_implementation": {
            str(key): int(value) for key, value in implementation_counts.items()
        },
        "selected_scenario_routes": route_map,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
    }
    return overview, grid, trait_availability


def validate_run(
    root: Path,
    row: Any,
    correction: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    relative = Path(str(row.state_id)) / str(row.candidate)
    run_root = root / RUNS / relative
    require_files(run_root, RUN_FILES)
    metadata_path = run_root / "run_metadata.json"
    metadata = read_json(metadata_path)
    if metadata.get("status") != "PASS":
        raise ValueError(f"Run is not PASS: {relative}")
    expected = {
        "state_id": str(row.state_id),
        "scenario": str(row.scenario),
        "candidate": str(row.candidate),
        "configuration_label": str(row.configuration_label),
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
    if metadata.get("guard_mask_observation_signatures_written") is not True:
        raise ValueError(f"Run {relative} omitted guard signatures")
    implementation_class = validate_run_protocol(
        metadata, str(row.scenario), correction
    )

    traits = read_tsv(run_root / "validation_trait_metrics.tsv")
    guards = read_tsv(run_root / "validation_guard_metrics.tsv")
    epochs = read_tsv(run_root / "epoch_history.tsv")
    require_columns(traits, ["trait_name_canonical"], f"traits for {relative}")
    if (
        traits.empty
        or len(traits) > TRAITS_PER_RUN
        or not traits["trait_name_canonical"].is_unique
        or not set(traits["trait_name_canonical"].astype(str)).issubset(
            set(EXPECTED_TRAITS)
        )
    ):
        raise ValueError(f"Run {relative} has invalid trait availability")
    if guards.empty or epochs.empty:
        raise ValueError(f"Run {relative} has empty guard or epoch reporting")
    require_columns(guards, ["observation_id_signature"], f"guards for {relative}")
    if guards["observation_id_signature"].fillna("").astype(str).eq("").any():
        raise ValueError(f"Run {relative} has empty guard signatures")
    return metadata, implementation_class


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
    overview, grid, trait_availability = validate_summary(root, code_root)
    active_commit = git_commit(code_root)
    correction = read_json(
        code_root
        / "server_training_pipeline/"
        "stage1_v2_phase6_confirmation_execution_correction_v4.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "stage1_v2_phase6_confirmation_results.tar.gz"
    checksum = archive.with_suffix(archive.suffix + ".sha256")

    with tempfile.TemporaryDirectory(
        prefix="confirmation_export_", dir=output_dir
    ) as temporary_name:
        staging = Path(temporary_name) / "stage1_v2_phase6_confirmation_results"
        staging.mkdir(parents=True)
        records: list[dict[str, Any]] = []
        implementation_counts: dict[str, int] = {}

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

        availability_path = staging / "summary" / "confirmation_trait_availability.tsv"
        trait_availability.to_csv(availability_path, sep="\t", index=False)
        records.append(
            {
                "category": "derived_reporting",
                "source_path": "generated/confirmation_trait_availability.tsv",
                "package_path": availability_path.relative_to(staging).as_posix(),
                "bytes": int(availability_path.stat().st_size),
                "sha256": sha256_file(availability_path),
            }
        )

        for row in grid.itertuples(index=False):
            relative = Path(str(row.state_id)) / str(row.candidate)
            _, implementation_class = validate_run(root, row, correction)
            implementation_counts[implementation_class] = (
                implementation_counts.get(implementation_class, 0) + 1
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

        if sum(implementation_counts.values()) != EXPECTED_RUNS:
            raise ValueError("Per-run validation did not account for all 375 runs")
        overview["validated_run_counts_by_implementation_class"] = (
            implementation_counts
        )
        overview["payload_artifact_count"] = int(len(records))
        overview["payload_bytes"] = int(sum(row["bytes"] for row in records))
        overview["active_exporter_code_commit"] = active_commit
        (staging / "EXPORT_SUMMARY.json").write_text(
            json.dumps(overview, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_tsv(staging / "payload_manifest.tsv", records)
        add_tree_to_archive(staging, archive)

    archive_sha = sha256_file(archive)
    checksum.write_text(f"{archive_sha}  {archive.name}\n", encoding="ascii")
    overview["archive"] = str(archive)
    overview["archive_bytes"] = int(archive.stat().st_size)
    overview["archive_sha256"] = archive_sha
    overview["checksum_file"] = str(checksum)
    return archive, checksum, overview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package completed Stage-1 v2 Phase-6 confirmation results"
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
