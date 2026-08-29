from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


PROTOCOL = Path(
    "server_training_pipeline/"
    "stage1_v2_phase6_hierarchy_calibration_amendment_protocol_v2.json"
)
SOURCE_PROTOCOL = Path(
    "server_training_pipeline/"
    "stage1_v2_phase6_hierarchy_full_confirmation_protocol_v1.json"
)
CALIBRATION_HELPER = Path(
    "server_training_pipeline/"
    "stage1_v2_phase6_hierarchy_calibration_amendment_v2.py"
)
TRAINER = Path(
    "server_training_pipeline/"
    "train_stage1_v2_phase6_hierarchy_calibration_amendment_tf.py"
)
RUNNER = Path(
    "scripts/v2/run_stage1_v2_phase6_hierarchy_calibration_amendment.py"
)
SERVER_RUNNER = Path(
    "scripts/v2/run_stage1_v2_phase6_hierarchy_calibration_amendment_server_cpu.sh"
)
STATUS_SCRIPT = Path(
    "scripts/v2/show_stage1_v2_phase6_hierarchy_calibration_amendment_server_cpu_status.sh"
)
ROUTE_FREEZE = Path(
    "scripts/v2/freeze_stage1_v2_phase6_hierarchy_calibration_route.py"
)
SOURCE_LOCK = Path(
    "audit/v2/stage1_v2_phase6_hierarchy_full_confirmation_v1/"
    "PHASE6_HIERARCHY_FULL_CONFIRMATION_LOCK.json"
)
SOURCE_OUTPUT = Path("model_kernels/stage1_v2_phase6_hierarchy_full_confirmation_v1")
FAILED_FREEZE = Path(
    "audit/v2/stage1_v2_phase6_hierarchy_failed_confirmation_freeze_v1"
)
OUTPUT = Path(
    "audit/v2/stage1_v2_phase6_hierarchy_calibration_amendment_v2"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the failed hierarchy confirmation and calibration-only amendment"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = (
        args.code_root or Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root))
    ).resolve()
    source_output = root / SOURCE_OUTPUT
    source_decision_path = source_output / "FULL_CONFIRMATION_DECISION.json"
    source_lock_path = root / SOURCE_LOCK
    code_paths = [
        code_root / PROTOCOL,
        code_root / SOURCE_PROTOCOL,
        code_root / CALIBRATION_HELPER,
        code_root / TRAINER,
        code_root / RUNNER,
        code_root / SERVER_RUNNER,
        code_root / STATUS_SCRIPT,
        code_root / ROUTE_FREEZE,
    ]
    required = [source_decision_path, source_lock_path, *code_paths]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Calibration amendment freeze inputs are missing: {missing}")

    source_decision = read_json(source_decision_path)
    source_lock = read_json(source_lock_path)
    protocol = read_json(code_root / PROTOCOL)
    source_protocol = read_json(code_root / SOURCE_PROTOCOL)
    source_artifacts: list[Path] = []
    artifact_hash_checks: dict[str, bool] = {}
    for name, expected in source_decision.get("artifact_sha256", {}).items():
        path = source_output / name
        source_artifacts.append(path)
        artifact_hash_checks[f"source_artifact_{name}"] = (
            path.is_file() and sha256_file(path) == expected
        )

    source_checks = {
        "source_status_is_terminal_failure": source_decision.get("status")
        == "FAIL_STAGE1_V2_PHASE6_HIERARCHY_FULL_INNER_CONFIRMATION",
        "source_selected_candidate_is_null": source_decision.get("selected_candidate")
        is None,
        "source_failed_only_original_guard": source_decision.get("failed_checks")
        == ["selected_candidate_all_guards_pass"],
        "source_matched_training_runs_50": int(
            source_decision.get("matched_training_run_count", -1)
        )
        == 50,
        "source_routed_states_125": int(source_decision.get("routed_state_count", -1))
        == 125,
        "source_active_hierarchy_states_25": int(
            source_decision.get("active_hierarchy_state_count", -1)
        )
        == 25,
        "source_reference_reuse_states_100": int(
            source_decision.get("exact_reference_reuse_state_count", -1)
        )
        == 100,
        "source_outer_not_performed": source_decision.get("outer_evaluation_performed")
        is False,
        "source_outer_unread": source_decision.get("outer_test_metrics_read") is False
        and source_decision.get("outer_test_outcomes_read") is False,
        "source_final_holdout_unread": source_decision.get("final_holdout_outcomes_read")
        is False,
        "source_lock_pass": source_lock.get("status")
        == "PASS_FROZEN_BEFORE_HIERARCHY_FULL_INNER_CONFIRMATION",
        **artifact_hash_checks,
    }
    source_failed = [name for name, passed in source_checks.items() if not passed]
    failure_freeze = {
        "status": (
            "PASS_TERMINAL_FAILED_HIERARCHY_CONFIRMATION_FROZEN"
            if not source_failed
            else "FAIL_TERMINAL_FAILED_HIERARCHY_CONFIRMATION_FREEZE"
        ),
        "protocol_version": "stage1_v2_phase6_hierarchy_failed_confirmation_freeze_v1",
        "stage1_version": "Stage-1 v2",
        "scientific_result": "valid_failed_inner_confirmation",
        "threshold_relaxed": False,
        "retrospective_tolerance_added": False,
        "source_decision_path": source_decision_path.relative_to(root).as_posix(),
        "source_decision_sha256": sha256_file(source_decision_path),
        "source_lock_sha256": sha256_file(source_lock_path),
        "source_artifact_sha256": {
            path.name: sha256_file(path) for path in source_artifacts if path.is_file()
        },
        "checks": source_checks,
        "failed_checks": source_failed,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
    }
    failed_dir = root / FAILED_FREEZE
    write_json(failed_dir / "FAILED_CONFIRMATION_FREEZE.json", failure_freeze)
    pd.DataFrame(
        [
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": ""}
            for name, passed in source_checks.items()
        ]
    ).to_csv(
        failed_dir / "validation_checks.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    if source_failed:
        print(json.dumps(failure_freeze, indent=2, sort_keys=True))
        raise SystemExit("Terminal failed hierarchy confirmation could not be frozen")

    checks = {
        "protocol_identity": protocol.get("protocol_version")
        == "stage1_v2_phase6_hierarchy_calibration_amendment_v2",
        "stage1_v2": protocol.get("stage1_version") == "Stage-1 v2",
        "failed_confirmation_terminal": protocol.get(
            "source_failed_confirmation_terminal"
        )
        is True,
        "retrospective_threshold_change_forbidden": protocol.get(
            "retrospective_threshold_change_allowed"
        )
        is False,
        "fixed_configuration_exact": protocol["fixed_configuration"]
        == source_protocol["fixed_configuration"],
        "fixed_hierarchy_exact": protocol["trial_environment_hierarchy"]
        == source_protocol["trial_environment_hierarchy"],
        "fixed_trait_regularization_exact": protocol["trait_specific_regularization"]
        == source_protocol["trait_specific_regularization"],
        "fixed_non_target_calibration_exact": protocol["positive_training_calibration"]
        == source_protocol["positive_training_calibration"],
        "acceptance_thresholds_exact": protocol["phase_1_acceptance"]
        == source_protocol["phase_1_acceptance"],
        "macro_calibration_threshold_unchanged_0_20": float(
            protocol["phase_1_acceptance"]["maximum_absolute_macro_calibration_error"]
        )
        == 0.2,
        "one_shared_model_fit_per_state": protocol["architecture_policy"][
            "one_shared_model_fit_per_state"
        ]
        is True,
        "calibration_only_mutation": protocol["architecture_policy"][
            "calibration_is_only_mutable_component"
        ]
        is True,
        "state_count_25": int(protocol["confirmation_scope"]["state_count"]) == 25,
        "new_model_fit_count_25": int(
            protocol["confirmation_scope"]["new_model_fit_count"]
        )
        == 25,
        "calibration_result_count_75": int(
            protocol["confirmation_scope"]["derived_calibration_result_count"]
        )
        == 75,
        "three_preregistered_calibrators": set(
            value["method"] for value in protocol["calibration_candidates"].values()
        )
        == {"identity", "environment_oof_affine_ridge", "environment_oof_huber"},
        "positive_slope_constraint": float(
            protocol["test_weight_environment_oof_calibration"]["minimum_slope"]
        )
        > 0,
        "training_only_calibration": protocol[
            "test_weight_environment_oof_calibration"
        ]["validation_values_used"]
        is False,
        "outer_evaluation_blocked": protocol["outer_test_policy"][
            "outer_evaluation_allowed"
        ]
        is False,
        "outer_unread": protocol.get("outer_test_metrics_read") is False
        and protocol.get("outer_test_outcomes_read") is False,
        "final_holdout_sealed": protocol["final_holdout_policy"]["sealed"] is True
        and protocol.get("final_holdout_outcomes_read") is False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    artifacts = {
        str(path.relative_to(code_root)): {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in code_paths
    }
    lock = {
        "status": (
            "PASS_FROZEN_CALIBRATION_ONLY_AMENDMENT_V2"
            if not failed
            else "FAIL_CALIBRATION_ONLY_AMENDMENT_V2_FREEZE"
        ),
        "protocol_version": "stage1_v2_phase6_hierarchy_calibration_amendment_freeze_v2",
        "stage1_version": "Stage-1 v2",
        "selection_data": "terminal_inner_confirmation_and_frozen_protocols_only",
        "source_failed_confirmation_terminal": True,
        "source_failed_confirmation_freeze_sha256": sha256_file(
            failed_dir / "FAILED_CONFIRMATION_FREEZE.json"
        ),
        "new_model_fit_count": 25,
        "derived_calibration_result_count": 75,
        "route_freeze_allowed_before_results": False,
        "outer_evaluation_allowed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "checks": checks,
        "failed_checks": failed,
        "artifacts": artifacts,
        "code_commit": git_commit(code_root),
    }
    output = root / OUTPUT
    write_json(output / "PHASE6_HIERARCHY_CALIBRATION_AMENDMENT_LOCK.json", lock)
    pd.DataFrame(
        [
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": ""}
            for name, passed in checks.items()
        ]
    ).to_csv(
        output / "validation_checks.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    print(json.dumps(lock, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"Calibration-only amendment freeze failed: {failed}")


if __name__ == "__main__":
    main()
