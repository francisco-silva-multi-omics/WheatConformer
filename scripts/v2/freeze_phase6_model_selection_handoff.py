from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT = Path("audit/v2/phase6_model_selection_handoff_v1")
SELECTION_PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_selection_protocol_v1.json"
)
GPU_RUNTIME_PROTOCOL = Path("server_training_pipeline/stage1_v2_training_runtime_v1.json")
CPU_RUNTIME_PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_server_cpu_runtime_v1.json"
)
EXECUTION_PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_execution_protocol_v2.json"
)
TRAINER_INTERFACE = Path("server_training_pipeline/stage1_v2_trainer_interface.py")
TRAINER = Path("server_training_pipeline/train_stage1_v2_phase6_tf.py")
PHASE1_ORCHESTRATOR = Path("scripts/v2/run_stage1_v2_phase6_phase1.py")
PHASE1_LAUNCHER = Path("scripts/v2/run_stage1_v2_phase6_phase1.sh")
PHASE1_SERVER_LAUNCHER = Path(
    "scripts/v2/run_stage1_v2_phase6_phase1_server_cpu.sh"
)
PHASE1_DATA_PACKAGER = Path(
    "scripts/v2/package_stage1_v2_phase6_phase1_server_data.py"
)

PARENT_RELEASES = (
    (
        "phase5_split_bound",
        Path("audit/v2/phase5_split_bound_kernel_validation_v2/PHASE5_RELEASE_DECISION.json"),
        "PASS_PHASE5_KERNEL_VALIDATION",
    ),
    (
        "phase5_parity",
        Path(
            "audit/v2/phase5_panel_environment_scenario_parity_extension_v2/PHASE5_PARITY_EXTENSION_DECISION.json"
        ),
        "PASS_PHASE5_PARITY_EXTENSION_WITH_EXPLICIT_COMPONENT_BLOCKERS",
    ),
    (
        "ka_150_state",
        Path(
            "audit/v2/phase5_ka_temporal_country_extension_v1/PHASE5_KA_TEMPORAL_COUNTRY_EXTENSION_DECISION.json"
        ),
        "PASS_KA_TEMPORAL_COUNTRY_EXTENSION",
    ),
    (
        "regulatory_eligibility_v2",
        Path(
            "audit/v2/phase5_regulatory_eligibility_v2/REGULATORY_ELIGIBILITY_V2_DECISION.json"
        ),
        "PASS_REGULATORY_ELIGIBILITY_V2_WITH_KZ_DEFERRED",
    ),
    (
        "projection_core_readiness",
        Path("audit/v2/e_projection_core_v1_readiness/E_PROJECTION_CORE_V1_READINESS.json"),
        "PASS_READY_TO_GENERATE_MEMBER_RESOLVED_FUTURE_COVARIATES",
    ),
    (
        "projection_core_split_bound_historical",
        Path(
            "audit/v2/e_projection_core_v1_split_bound_historical_v1_release/SPLIT_BOUND_PROJECTION_INPUT_RELEASE_DECISION.json"
        ),
        "PASS_SPLIT_BOUND_HISTORICAL_PROJECTION_INPUTS_CERTIFIED",
    ),
    (
        "projection_core_future_covariates",
        Path(
            "audit/v2/e_projection_core_v1_future_covariates_v1_release/FUTURE_COVARIATE_RELEASE_DECISION.json"
        ),
        "PASS_MEMBER_RESOLVED_FUTURE_COVARIATES_CERTIFIED",
    ),
    (
        "panel_prerequisite_recovery",
        Path(
            "audit/v2/phase5_panel_prerequisite_recovery_v1/PHASE5_PANEL_PREREQUISITE_RECOVERY_DECISION.json"
        ),
        "PASS_BOUNDED_PANEL_PREREQUISITE_RECOVERY_WITH_EXPLICIT_REMAINING_BLOCKERS",
    ),
    (
        "cimmyt_pre_qc_split_local",
        Path(
            "audit/v2/phase5_cimmyt_pre_qc_split_local_v1/CIMMYT_PRE_QC_SPLIT_LOCAL_DECISION.json"
        ),
        "PASS_CIMMYT_PRE_QC_SPLIT_LOCAL_150_STATE_CERTIFIED",
    ),
    (
        "h_seeds_operator",
        Path("audit/v2/phase6_h_seeds_operator_v1/H_SEEDS_OPERATOR_DECISION.json"),
        "PASS_H_SEEDS_150_STATE_OPERATOR_CERTIFIED",
    ),
)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "git command failed")
    return process.stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the aggregate Stage-1 v2 Phase-6 handoff")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing handoff deterministically after a code-only release update",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.root.resolve()
    code_root = (
        args.code_root
        or Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", data_root))
    ).resolve()
    output = (
        (data_root / args.output).resolve()
        if not args.output.is_absolute()
        else args.output
    )
    if output.exists() and any(output.iterdir()) and not args.replace:
        raise SystemExit(f"Refusing to overwrite aggregate handoff: {output}")
    output.mkdir(parents=True, exist_ok=True)

    release_rows = []
    decisions: dict[str, dict[str, Any]] = {}
    for label, relative, expected_status in PARENT_RELEASES:
        path = data_root / relative
        decision = json.loads(path.read_text(encoding="utf-8"))
        observed = str(decision.get("status", ""))
        release_rows.append(
            {
                "release": label,
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "expected_status": expected_status,
                "observed_status": observed,
                "release_id": decision.get("release_id", ""),
                "status": "PASS" if observed == expected_status else "FAIL",
            }
        )
        decisions[label] = decision
    release_inventory = pd.DataFrame(release_rows)

    selection_path = code_root / SELECTION_PROTOCOL
    gpu_runtime_path = code_root / GPU_RUNTIME_PROTOCOL
    cpu_runtime_path = code_root / CPU_RUNTIME_PROTOCOL
    execution_protocol_path = code_root / EXECUTION_PROTOCOL
    trainer_path = code_root / TRAINER_INTERFACE
    implementation_relatives = (
        TRAINER,
        PHASE1_ORCHESTRATOR,
        PHASE1_LAUNCHER,
        PHASE1_SERVER_LAUNCHER,
        PHASE1_DATA_PACKAGER,
        CPU_RUNTIME_PROTOCOL,
        EXECUTION_PROTOCOL,
    )
    implementation_paths = tuple(code_root / path for path in implementation_relatives)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    gpu_runtime = json.loads(gpu_runtime_path.read_text(encoding="utf-8"))
    cpu_runtime = json.loads(cpu_runtime_path.read_text(encoding="utf-8"))
    execution_protocol = json.loads(execution_protocol_path.read_text(encoding="utf-8"))
    git_status = git(code_root, "status", "--short").splitlines()
    allowed_unrelated = {
        "D audit/new_genotypic_matches_impact.md",
        " D audit/new_genotypic_matches_impact.md",
    }
    release_dirty = sorted(line for line in git_status if line not in allowed_unrelated)
    commit = git(code_root, "rev-parse", "HEAD")
    branch = git(code_root, "branch", "--show-current")

    checks = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append(
            {"check": check, "status": "PASS" if passed else "FAIL", "detail": detail}
        )

    add(
        "authoritative_parent_release_statuses",
        release_inventory["status"].eq("PASS").all(),
        f"passed={int(release_inventory['status'].eq('PASS').sum())}/{len(release_inventory)}",
    )
    add(
        "code_release_committed",
        not release_dirty and commit != "274e41df1abbae54785f86eec709f2012efcab7b",
        f"commit={commit}; disallowed_dirty={release_dirty}",
    )
    add(
        "stage1_v2_runtime_frozen",
        gpu_runtime.get("python") == "3.11.15"
        and gpu_runtime.get("tensorflow") == "2.15.1"
        and gpu_runtime.get("pandas") == "2.2.3"
        and cpu_runtime.get("python_major_minor") == "3.11"
        and cpu_runtime.get("tensorflow") == "2.15.1"
        and cpu_runtime.get("pandas") == "2.2.3",
        f"gpu={gpu_runtime.get('runtime_version', '')}; "
        f"cpu={cpu_runtime.get('runtime_version', '')}",
    )
    add(
        "execution_only_amendment_frozen",
        execution_protocol.get("scientific_selection_protocol_unchanged") is True
        and execution_protocol.get("all_120_runs_must_be_recomputed") is True
        and execution_protocol.get("old_run_reuse_allowed") is False,
        execution_protocol.get("protocol_version", ""),
    )
    add(
        "exact_scenario_grid",
        selection.get("scenario_order")
        == ["GNEW_EOBS", "GOBS_ENEW", "GNEW_ENEW", "TEMPORAL_YEAR", "COUNTRY_HOLDOUT"],
        ",".join(selection.get("scenario_order", [])),
    )
    schedule = selection.get("screen_schedule", {})
    add(
        "phase1_grid_frozen",
        schedule.get("phase_1_scenario") == "GNEW_EOBS"
        and schedule.get("phase_1_outer_fold") == 1
        and schedule.get("phase_1_inner_folds") == [1, 2, 3, 4, 5]
        and len(selection.get("candidate_stages", {}).get("phase_1_individual", [])) == 8
        and len(selection.get("hyperparameter_configurations", {})) == 3,
        "scenario=GNEW_EOBS; outer=1; inner=1..5; candidates=8; configurations=3",
    )
    add(
        "phase1_implementation_frozen",
        all(path.is_file() for path in implementation_paths),
        ";".join(path.as_posix() for path in implementation_relatives),
    )
    add(
        "selection_metrics_frozen",
        selection.get("selection_metrics", {}).get("primary")
        == "macro_trait_scenario_normalized_rmse",
        selection.get("selection_metrics", {}).get("primary", ""),
    )
    mandatory = set(selection.get("mandatory_reporting_subsets", []))
    add(
        "mandatory_information_class_reporting",
        {
            "PEDIGREE_ONLY",
            "MARKER_SUPPORTED",
            "NEITHER_PRODUCTION_PEDIGREE_NOR_DENSE_MARKERS",
            "RECOVERED_IDENTITY_OR_COMPONENT",
        }.issubset(mandatory),
        f"subsets={len(mandatory)}",
    )
    add(
        "projection_inactive_reporting",
        "PROJECTION_CORE_INACTIVE_814_ENVIRONMENTS" in mandatory
        and decisions["projection_core_split_bound_historical"].get(
            "inactive_historical_environment_count"
        )
        == 814,
        "required_inactive_environments=814",
    )
    add(
        "h_seeds_decided_before_metrics",
        decisions["h_seeds_operator"].get("phase6_candidate_preregistered") is True
        and decisions["h_seeds_operator"].get("inner_validation_metrics_read") is False,
        decisions["h_seeds_operator"].get("status", ""),
    )
    add(
        "kz_remains_deferred",
        decisions["regulatory_eligibility_v2"].get("phase6_K_z_candidate_allowed") is False
        or "DEFERRED" in decisions["regulatory_eligibility_v2"].get("status", ""),
        decisions["regulatory_eligibility_v2"].get("status", ""),
    )
    add(
        "protected_outcomes_unread",
        all(
            decision.get("outer_test_outcomes_read", False) is False
            and decision.get("final_holdout_outcomes_read", False) is False
            for decision in decisions.values()
        ),
        "all bound decisions report protected outcomes unread",
    )
    validation = pd.DataFrame(checks)
    write_tsv(output / "authoritative_release_inventory.tsv", release_inventory)
    write_tsv(output / "validation_checks.tsv", validation)
    if not validation["status"].eq("PASS").all():
        raise ValueError("Aggregate Phase-6 handoff validation failed")

    handoff = {
        "status": "PASS_READY_FOR_STAGE1_V2_PHASE6_INNER_MODEL_SELECTION",
        "release_id": "P6MSH_20260822_V2_CPU_SERVER",
        "protocol_version": "stage1_v2_phase6_aggregate_handoff_v2_cpu_server",
        "stage1_version": "Stage-1 v2",
        "branch": branch,
        "code_commit": commit,
        "bound_release_count": len(release_inventory),
        "selection_protocol": SELECTION_PROTOCOL.as_posix(),
        "selection_protocol_sha256": sha256_file(selection_path),
        "gpu_runtime_protocol": GPU_RUNTIME_PROTOCOL.as_posix(),
        "gpu_runtime_protocol_sha256": sha256_file(gpu_runtime_path),
        "server_cpu_runtime_protocol": CPU_RUNTIME_PROTOCOL.as_posix(),
        "server_cpu_runtime_protocol_sha256": sha256_file(cpu_runtime_path),
        "execution_protocol": EXECUTION_PROTOCOL.as_posix(),
        "execution_protocol_sha256": sha256_file(execution_protocol_path),
        "trainer_interface": TRAINER_INTERFACE.as_posix(),
        "trainer_interface_sha256": sha256_file(trainer_path),
        "phase1_implementation_sha256": {
            relative.as_posix(): sha256_file(code_root / relative)
            for relative in implementation_relatives
        },
        "authoritative_release_inventory_sha256": sha256_file(
            output / "authoritative_release_inventory.tsv"
        ),
        "validation_sha256": sha256_file(output / "validation_checks.tsv"),
        "scenario_count": 5,
        "state_count": 150,
        "inner_state_count": 125,
        "outer_state_count": 25,
        "projection_inactive_environment_count": 814,
        "H_SEEDS_decision": "PREREGISTERED_OPERATOR_CANDIDATE",
        "K_z_decision": "DEFERRED_BEFORE_METRICS",
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "model_training_performed": False,
        "phase1_scenario": "GNEW_EOBS",
        "phase1_outer_fold": 1,
        "phase1_inner_folds": [1, 2, 3, 4, 5],
        "phase1_run_count": 120,
        "outer_evaluation_allowed": False,
        "next_action": "run_preregistered_phase1_inner_screen_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "PHASE6_MODEL_SELECTION_HANDOFF.json", handoff)
    files = [
        output / "authoritative_release_inventory.tsv",
        output / "validation_checks.tsv",
        output / "PHASE6_MODEL_SELECTION_HANDOFF.json",
    ]
    manifest = pd.DataFrame(
        [
            {
                "path": path.relative_to(data_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ]
    )
    write_tsv(output / "artifact_manifest.tsv", manifest)
    print(json.dumps(handoff, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
