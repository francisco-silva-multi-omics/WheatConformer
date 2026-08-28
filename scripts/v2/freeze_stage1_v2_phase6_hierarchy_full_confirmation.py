from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from server_training_pipeline.stage1_v2_trainer_interface import PARITY


PROTOCOL = Path(
    "server_training_pipeline/"
    "stage1_v2_phase6_hierarchy_full_confirmation_protocol_v1.json"
)
TRAINER = Path(
    "server_training_pipeline/"
    "train_stage1_v2_phase6_hierarchy_full_confirmation_tf.py"
)
AMENDMENT_CERTIFIER = Path(
    "scripts/v2/certify_stage1_v2_phase6_hierarchy_guard_amendment.py"
)
FREEZER = Path(
    "scripts/v2/freeze_stage1_v2_phase6_hierarchy_full_confirmation.py"
)
RUNNER = Path("scripts/v2/run_stage1_v2_phase6_hierarchy_full_confirmation.py")
SERVER_LAUNCHER = Path(
    "scripts/v2/run_stage1_v2_phase6_hierarchy_full_confirmation_server_cpu.sh"
)
SERVER_STATUS = Path(
    "scripts/v2/show_stage1_v2_phase6_hierarchy_full_confirmation_server_cpu_status.sh"
)
TESTS = (
    Path("tests/test_stage1_v2_phase6_hierarchy_guard_amendment.py"),
    Path("tests/test_stage1_v2_phase6_hierarchy_full_confirmation.py"),
)
AMENDMENT = Path(
    "audit/v2/stage1_v2_phase6_hierarchy_calibration_guard_amendment_v1/"
    "HIERARCHY_CALIBRATION_GUARD_AMENDMENT.json"
)
SOURCE_STATUS = Path(
    "model_kernels/stage1_v2_phase6_confirmation_v1/confirmation_status.json"
)
SOURCE_RUNS = Path("trained_models/stage1_v2_phase6_confirmation_v1_runs")
OUTPUT = Path("audit/v2/stage1_v2_phase6_hierarchy_full_confirmation_v1")
LOCK_NAME = "PHASE6_HIERARCHY_FULL_CONFIRMATION_LOCK.json"
REFERENCE = "historical_reaction_reference"
SELECTED = "hierarchy_test_weight_identity_calibration_v1"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
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
        description="Freeze the routed Stage-1 v2 hierarchy full confirmation"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = (
        args.code_root
        or Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root))
    ).resolve()
    protocol_path = code_root / PROTOCOL
    trainer_path = code_root / TRAINER
    amendment_path = root / AMENDMENT
    source_status_path = root / SOURCE_STATUS
    registry_path = root / PARITY / "splits/state_registry.tsv"
    code_files = [
        protocol_path,
        trainer_path,
        code_root / AMENDMENT_CERTIFIER,
        code_root / FREEZER,
        code_root / RUNNER,
        code_root / SERVER_LAUNCHER,
        code_root / SERVER_STATUS,
        *(code_root / path for path in TESTS),
    ]
    required = [
        *code_files,
        amendment_path,
        source_status_path,
        registry_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Full-confirmation freeze inputs are missing: {missing}")

    protocol = read_json(protocol_path)
    amendment = read_json(amendment_path)
    source_status = read_json(source_status_path)
    registry = pd.read_csv(registry_path, sep="\t", dtype=str)
    states = registry.loc[
        registry["state_level"].eq("INNER")
        & registry["scenario"].isin(protocol["confirmation_grid"]["scenarios"])
    ].copy()
    states["outer_fold"] = states["outer_fold"].astype(int)
    states["inner_fold"] = states["inner_fold"].astype(int)
    gnew = states.loc[states["scenario"].eq("GNEW_EOBS")]

    source_metadata: list[Path] = []
    source_valid = True
    for state_id in sorted(states["state_id"].astype(str)):
        path = root / SOURCE_RUNS / state_id / REFERENCE / "run_metadata.json"
        source_metadata.append(path)
        if not path.is_file():
            source_valid = False
            continue
        value = read_json(path)
        source_valid &= (
            value.get("status") == "PASS"
            and value.get("candidate") == REFERENCE
            and value.get("outer_test_metrics_read") is False
            and value.get("outer_test_outcomes_read") is False
            and value.get("final_holdout_outcomes_read") is False
        )

    routes = protocol["routing_policy"]
    checks = {
        "protocol_identity": protocol.get("protocol_version")
        == "stage1_v2_phase6_hierarchy_full_confirmation_v1",
        "stage1_v2": protocol.get("stage1_version") == "Stage-1 v2",
        "guard_amendment_pass": amendment.get("status")
        == "PASS_HIERARCHY_CALIBRATION_GUARD_AMENDMENT",
        "guard_amendment_selected_candidate": amendment.get(
            "selected_candidate_after_amendment"
        )
        == SELECTED,
        "guard_amendment_selection_unchanged": amendment.get(
            "scientific_selection_changed"
        )
        is False,
        "source_confirmation_pass": source_status.get("status")
        == "PASS_STAGE1_V2_PHASE6_CONFIRMATION_COMPLETE",
        "source_confirmation_inner_only": source_status.get("selection_data")
        == "inner_validation_only",
        "source_outer_unread": source_status.get("outer_test_metrics_read") is False
        and source_status.get("outer_test_outcomes_read") is False,
        "source_final_unread": source_status.get("final_holdout_outcomes_read")
        is False,
        "state_grid_125": len(states) == 125 and states["state_id"].is_unique,
        "gnew_grid_25": len(gnew) == 25 and gnew["state_id"].is_unique,
        "all_source_reference_runs_certified": source_valid
        and len(source_metadata) == 125,
        "selected_route_only_gnew": routes["GNEW_EOBS"]["candidate"] == SELECTED
        and all(
            routes[scenario]["candidate"] == REFERENCE
            for scenario in routes
            if scenario != "GNEW_EOBS"
        ),
        "matched_run_count_50": int(
            protocol["confirmation_grid"]["matched_training_run_count"]
        )
        == 50,
        "reference_reuse_count_100": int(
            protocol["confirmation_grid"]["exact_reference_reuse_state_count"]
        )
        == 100,
        "fixed_test_weight_identity": protocol["selected_candidate_contract"][
            "test_weight_calibration"
        ]
        == "identity",
        "outer_unread": protocol.get("outer_test_metrics_read") is False
        and protocol.get("outer_test_outcomes_read") is False,
        "final_holdout_sealed": protocol["final_holdout_policy"]["sealed"] is True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    source_digest = hashlib.sha256()
    for path in source_metadata:
        if path.is_file():
            source_digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            source_digest.update(sha256_file(path).encode("ascii"))
    lock = {
        "status": (
            "PASS_FROZEN_BEFORE_HIERARCHY_FULL_INNER_CONFIRMATION"
            if not failed
            else "FAIL_HIERARCHY_FULL_CONFIRMATION_FREEZE"
        ),
        "protocol_version": "stage1_v2_phase6_hierarchy_full_confirmation_freeze_v1",
        "stage1_version": "Stage-1 v2",
        "selection_data": "completed_inner_metrics_and_frozen_identifiers_only",
        "selected_candidate": SELECTED,
        "active_hierarchy_state_count": 25,
        "matched_training_run_count": 50,
        "exact_reference_reuse_state_count": 100,
        "routed_state_count": 125,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "checks": checks,
        "failed_checks": failed,
        "artifacts": {
            path.relative_to(root if path.is_relative_to(root) else code_root).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in required
        },
        "source_reference_metadata_sha256": source_digest.hexdigest(),
        "code_commit": git_commit(code_root),
    }
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / LOCK_NAME
    if lock_path.exists() and not args.replace:
        existing = read_json(lock_path)
        comparable = {key: value for key, value in existing.items() if key != "code_commit"}
        current = {key: value for key, value in lock.items() if key != "code_commit"}
        if comparable != current:
            raise FileExistsError(f"Existing full-confirmation lock differs: {lock_path}")
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
        raise SystemExit(f"Hierarchy full-confirmation freeze failed: {failed}")


if __name__ == "__main__":
    main()
