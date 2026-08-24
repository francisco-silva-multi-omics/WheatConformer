from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PARENT_HANDOFF = Path(
    "audit/v2/phase6_model_selection_handoff_v1/PHASE6_MODEL_SELECTION_HANDOFF.json"
)
PARENT_OUTPUT = Path("model_kernels/stage1_v2_phase6_phase1_v2")
PARENT_RUNS = Path("trained_models/stage1_v2_phase6_phase1_v2_runs")
OUTPUT = Path("audit/v2/stage1_v2_phase6_phase1_guard_replay_v1")
LOCK = OUTPUT / "PHASE1_GUARD_REPLAY_LOCK.json"
SELECTION_PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_selection_protocol_v1.json"
)
IMPLEMENTATION = (
    Path("server_training_pipeline/train_stage1_v2_phase6_tf.py"),
    Path("scripts/v2/run_stage1_v2_phase6_phase1.py"),
    Path("scripts/v2/freeze_stage1_v2_phase6_phase1_guard_replay.py"),
    Path("scripts/v2/run_stage1_v2_phase6_phase1_guard_replay_server_cpu.sh"),
    Path("scripts/v2/show_stage1_v2_phase6_phase1_guard_replay_server_cpu_status.sh"),
    Path("server_training_pipeline/stage1_v2_phase6_execution_protocol_v2.json"),
    Path("server_training_pipeline/stage1_v2_phase6_server_cpu_runtime_v1.json"),
)
PARENT_ARTIFACTS = (
    PARENT_HANDOFF,
    PARENT_OUTPUT / "phase1_provenance.json",
    PARENT_OUTPUT / "phase1_run_grid.tsv",
    PARENT_OUTPUT / "phase1_runs.tsv",
    PARENT_OUTPUT / "phase1_paired_metrics.tsv",
    PARENT_OUTPUT / "phase1_trait_metrics.tsv",
    PARENT_OUTPUT / "phase1_subset_metrics.tsv",
    PARENT_OUTPUT / "phase1_decision.tsv",
)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "Unable to resolve Git commit")
    return process.stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record(checks: list[dict[str, object]], name: str, passed: bool, detail: str) -> None:
    checks.append(
        {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the reporting-only Stage-1 v2 Phase-1 matched-guard replay"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    code_root = args.code_root.resolve()
    output = root / OUTPUT
    if (root / LOCK).exists() and not args.replace:
        raise FileExistsError(f"Guard replay lock already exists: {root / LOCK}")

    handoff = json.loads((root / PARENT_HANDOFF).read_text(encoding="utf-8"))
    provenance = json.loads(
        (root / PARENT_OUTPUT / "phase1_provenance.json").read_text(encoding="utf-8")
    )
    run_metadata_paths = sorted((root / PARENT_RUNS).rglob("run_metadata.json"))
    run_metadata = [json.loads(path.read_text(encoding="utf-8")) for path in run_metadata_paths]
    checks: list[dict[str, object]] = []
    record(
        checks,
        "parent_handoff_ready",
        handoff.get("status") == "PASS_READY_FOR_STAGE1_V2_PHASE6_INNER_MODEL_SELECTION",
        str(handoff.get("status")),
    )
    record(checks, "parent_run_count", len(run_metadata) == 120, str(len(run_metadata)))
    record(
        checks,
        "parent_runs_pass",
        len(run_metadata) == 120 and all(row.get("status") == "PASS" for row in run_metadata),
        f"pass={sum(row.get('status') == 'PASS' for row in run_metadata)}/120",
    )
    record(
        checks,
        "parent_runs_bound_to_parent_handoff_commit",
        len(run_metadata) == 120
        and all(row.get("code_commit") == handoff.get("code_commit") for row in run_metadata),
        f"handoff_commit={handoff.get('code_commit')}",
    )
    run_keys = {
        (
            row.get("state_id"),
            row.get("candidate"),
            row.get("configuration_label"),
        )
        for row in run_metadata
    }
    record(
        checks,
        "parent_run_keys_unique",
        len(run_metadata) == 120 and len(run_keys) == 120,
        f"unique={len(run_keys)}; rows={len(run_metadata)}",
    )
    record(
        checks,
        "parent_validation_observations_matched",
        provenance.get("matched_validation_observation_status") == "pass",
        str(provenance.get("matched_validation_observation_status")),
    )
    record(
        checks,
        "parent_outer_outcomes_sealed",
        provenance.get("outer_test_outcomes_read") is False
        and provenance.get("outer_test_metrics_read") is False,
        "outer outcomes and metrics were not read by parent selection",
    )
    record(
        checks,
        "final_holdout_sealed",
        provenance.get("final_holdout_outcomes_read") is False,
        str(provenance.get("final_holdout_outcomes_read")),
    )
    for path in (*PARENT_ARTIFACTS, SELECTION_PROTOCOL, *IMPLEMENTATION):
        base = root if path in PARENT_ARTIFACTS else code_root
        record(checks, f"file_present::{path.as_posix()}", (base / path).is_file(), str(base / path))
    failed = [row for row in checks if row["status"] != "PASS"]
    if failed:
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(checks).to_csv(output / "validation_checks.tsv", sep="\t", index=False)
        raise ValueError("Phase-1 guard replay freeze failed")

    parent_hashes = {
        path.as_posix(): sha256_file(root / path) for path in PARENT_ARTIFACTS
    }
    implementation_hashes = {
        path.as_posix(): sha256_file(code_root / path) for path in IMPLEMENTATION
    }
    lock = {
        "status": "PASS_READY_FOR_PHASE1_MATCHED_GUARD_REPLAY",
        "protocol_version": "stage1_v2_phase6_phase1_guard_replay_lock_v1",
        "stage1_version": "Stage-1 v2",
        "selection_data": "previously_opened_inner_validation_metrics_only",
        "purpose": "regenerate exact candidate/reference component-mask guards without changing candidates, configurations, seeds, or outer policy",
        "parent_inner_metrics_known": True,
        "new_candidate_or_hyperparameter_selection_performed": False,
        "required_run_count": 120,
        "code_commit": git_commit(code_root),
        "selection_protocol_sha256": sha256_file(code_root / SELECTION_PROTOCOL),
        "implementation_sha256": implementation_hashes,
        "parent_artifact_sha256": parent_hashes,
        "required_corrections": {
            "baseline_evaluated_on_every_candidate_mask": True,
            "subset_observation_id_signatures": True,
            "h_seeds_uses_direct_seeds_marker_support": True,
            "projection_core_mask_candidate_independent": True,
            "strict_candidate_reference_mask_pairing": True,
            "parent_full_metrics_replayed_with_maximum_absolute_delta": 1e-5,
        },
        "original_phase1_results_modified": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "outer_evaluation_allowed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(output / "validation_checks.tsv", sep="\t", index=False)
    write_json(root / LOCK, lock)
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
