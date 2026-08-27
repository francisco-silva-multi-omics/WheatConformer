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
    "server_training_pipeline/stage1_v2_phase6_hierarchy_calibration_protocol_v1.json"
)
SOURCE_PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_remediation_protocol_v1.json"
)
TRAINER = Path(
    "server_training_pipeline/train_stage1_v2_phase6_hierarchy_calibration_tf.py"
)
HELPER = Path("server_training_pipeline/stage1_v2_phase6_remediation.py")
RUNNER = Path("scripts/v2/run_stage1_v2_phase6_hierarchy_calibration.py")
REPLAY = Path("scripts/v2/certify_stage1_v2_phase6_information_guard_replay.py")
SOURCE_SUMMARY = Path("model_kernels/stage1_v2_phase6_remediation_v1/phase_1")
SOURCE_RUNS = Path("trained_models/stage1_v2_phase6_remediation_v1_runs/phase_1")
REPLAY_OUTPUT = Path("audit/v2/stage1_v2_phase6_information_guard_replay_v1")
OUTPUT = Path("audit/v2/stage1_v2_phase6_hierarchy_calibration_v1")
TESTS = (
    Path("tests/test_stage1_v2_phase6_hierarchy_calibration.py"),
    Path("tests/test_stage1_v2_phase6_hierarchy_calibration_tf.py"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the Stage-1 v2 hierarchy calibration screen"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-summary", type=Path, default=SOURCE_SUMMARY)
    parser.add_argument("--source-runs", type=Path, default=SOURCE_RUNS)
    parser.add_argument(
        "--replay-decision",
        type=Path,
        default=REPLAY_OUTPUT / "INFORMATION_GUARD_REPLAY_DECISION.json",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    source_summary = (
        args.source_summary
        if args.source_summary.is_absolute()
        else root / args.source_summary
    ).resolve()
    source_runs = (
        args.source_runs if args.source_runs.is_absolute() else root / args.source_runs
    ).resolve()
    replay_path = (
        args.replay_decision
        if args.replay_decision.is_absolute()
        else root / args.replay_decision
    ).resolve()
    source_decision_path = source_summary / "PHASE1_STRUCTURAL_DECISION.json"
    source_grid_path = source_summary / "remediation_phase1_run_grid.tsv"
    code_files = [
        code_root / PROTOCOL,
        code_root / SOURCE_PROTOCOL,
        code_root / TRAINER,
        code_root / HELPER,
        code_root / RUNNER,
        code_root / REPLAY,
        *(code_root / path for path in TESTS),
    ]
    required = [*code_files, source_decision_path, source_grid_path, replay_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Hierarchy calibration freeze inputs are missing: {missing}")
    protocol = read_json(code_root / PROTOCOL)
    source_protocol = read_json(code_root / SOURCE_PROTOCOL)
    source_decision = read_json(source_decision_path)
    replay = read_json(replay_path)
    source_grid = pd.read_csv(source_grid_path, sep="\t")
    gnew = source_grid.loc[source_grid["scenario"].eq("GNEW_EOBS")]
    source_run_files: list[Path] = []
    for state_id in sorted(gnew["state_id"].unique()):
        for candidate in (
            "historical_reaction_reference",
            "known_environment_hierarchical_v2",
        ):
            run_root = source_runs / state_id / candidate
            source_run_files.extend(
                [
                    run_root / "run_metadata.json",
                    run_root / "validation_trait_metrics.tsv",
                    run_root / "validation_guard_metrics.tsv",
                ]
            )
    missing_runs = [str(path) for path in source_run_files if not path.is_file()]
    if missing_runs:
        raise FileNotFoundError(f"Source comparator runs are missing: {missing_runs}")
    fixed = protocol["fixed_configuration"]
    source_fixed = source_protocol["hyperparameter_configurations"][
        "historical_capacity_16_batch8192"
    ]
    fixed_without_label = {key: value for key, value in fixed.items() if key != "label"}
    checks = {
        "protocol_identity": protocol.get("protocol_version")
        == "stage1_v2_phase6_hierarchy_calibration_v1",
        "stage1_v2": protocol.get("stage1_version") == "Stage-1 v2",
        "source_remediation_complete": source_decision.get("status")
        == "PASS_STAGE1_V2_PHASE6_REMEDIATION_PHASE1_COMPLETE",
        "source_candidates_not_advanced": source_decision.get("advanced_candidates") == [],
        "source_phase2_blocked": source_decision.get("phase2_optimizer_allowed") is False,
        "source_outer_unread": source_decision.get("outer_test_metrics_read") is False
        and source_decision.get("outer_test_outcomes_read") is False,
        "source_final_unread": source_decision.get("final_holdout_outcomes_read") is False,
        "information_replay_pass": replay.get("status")
        == "PASS_INFORMATION_GUARD_REPORTING_REPLAY_FROZEN",
        "information_replay_decision_unchanged": replay.get(
            "formal_source_decision_changed"
        )
        is False,
        "information_replay_outer_unread": replay.get("outer_test_metrics_read") is False
        and replay.get("outer_test_outcomes_read") is False,
        "information_replay_final_unread": replay.get("final_holdout_outcomes_read")
        is False,
        "acceptance_thresholds_unchanged": protocol["phase_1_acceptance"]
        == source_protocol["phase_1_acceptance"],
        "optimizer_configuration_unchanged": fixed_without_label == source_fixed,
        "batch_size_fixed_8192": int(fixed["batch_size"]) == 8192,
        "architecture_fixed": protocol["architecture_policy"]["fixed_candidate"]
        == "known_environment_hierarchical_v2",
        "bounded_new_run_count": int(protocol["phase_1"]["new_training_run_count"])
        == 15,
        "source_comparator_state_count": int(gnew["state_id"].nunique()) == 5,
        "outer_evaluation_blocked": protocol["outer_test_policy"][
            "outer_evaluation_allowed"
        ]
        is False,
        "final_holdout_sealed": protocol["final_holdout_policy"]["sealed"] is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    artifacts = {
        str(path.relative_to(code_root) if path.is_relative_to(code_root) else path): {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in required
    }
    source_signature = hashlib.sha256()
    for path in sorted(set(source_run_files), key=str):
        source_signature.update(str(path.relative_to(root)).encode("utf-8"))
        source_signature.update(sha256_file(path).encode("ascii"))
    freeze = {
        "status": (
            "PASS_FROZEN_BEFORE_HIERARCHY_CALIBRATION_INNER_SCREEN"
            if not failed
            else "FAIL_HIERARCHY_CALIBRATION_FREEZE"
        ),
        "protocol_version": "stage1_v2_phase6_hierarchy_calibration_freeze_v1",
        "stage1_version": "Stage-1 v2",
        "selection_data": "completed_inner_metrics_and_frozen_identifiers_only",
        "new_training_run_count": int(protocol["phase_1"]["new_training_run_count"]),
        "source_comparator_run_count": 10,
        "batch_size_screen_performed": False,
        "fixed_batch_size": 8192,
        "full_125_state_confirmation_allowed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "checks": checks,
        "failed_checks": failed,
        "artifacts": artifacts,
        "source_comparator_artifact_sha256": source_signature.hexdigest(),
        "code_commit": git_commit(code_root),
    }
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    (output / "PHASE6_HIERARCHY_CALIBRATION_LOCK.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": ""}
            for name, passed in checks.items()
        ]
    ).to_csv(output / "validation_checks.tsv", sep="\t", index=False, lineterminator="\n")
    print(json.dumps(freeze, indent=2, sort_keys=True))
    if failed:
        raise SystemExit("Stage-1 v2 hierarchy calibration freeze failed")


if __name__ == "__main__":
    main()
