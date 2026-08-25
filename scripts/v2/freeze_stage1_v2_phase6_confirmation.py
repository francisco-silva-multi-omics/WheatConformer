from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPLAY = Path("model_kernels/stage1_v2_phase6_phase1_guard_replay_v1")
OUTPUT = Path("audit/v2/stage1_v2_phase6_confirmation_v1")
LOCK = OUTPUT / "PHASE6_CONFIRMATION_LOCK.json"
PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_confirmation_protocol_v1.json"
)
IMPLEMENTATION = (
    Path(
        "server_training_pipeline/"
        "stage1_v2_phase6_confirmation_execution_correction_v2.json"
    ),
    Path("server_training_pipeline/stage1_v2_trainer_interface.py"),
    Path("server_training_pipeline/train_stage1_v2_phase6_tf.py"),
    Path("server_training_pipeline/train_stage1_v2_phase6_confirmation_tf.py"),
    Path("scripts/v2/freeze_stage1_v2_phase6_confirmation.py"),
    Path("scripts/v2/run_stage1_v2_phase6_confirmation.py"),
    Path("scripts/v2/run_stage1_v2_phase6_confirmation_server_cpu.sh"),
    Path("scripts/v2/show_stage1_v2_phase6_confirmation_server_cpu_status.sh"),
)
REPLAY_ARTIFACTS = (
    REPLAY / "phase1_provenance.json",
    REPLAY / "phase1_runs.tsv",
    REPLAY / "phase1_paired_metrics.tsv",
    REPLAY / "phase1_trait_metrics.tsv",
    REPLAY / "phase1_guard_metrics.tsv",
    REPLAY / "phase1_paired_guard_metrics.tsv",
    REPLAY / "phase1_decision.tsv",
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


def record(
    checks: list[dict[str, object]], name: str, passed: bool, detail: str
) -> None:
    checks.append(
        {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the adjudicated Stage-1 v2 Phase-6 confirmation"
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
    lock_path = root / LOCK
    if lock_path.exists() and not args.replace:
        raise FileExistsError(f"Confirmation lock already exists: {lock_path}")

    protocol = json.loads((code_root / PROTOCOL).read_text(encoding="utf-8"))
    provenance = json.loads(
        (root / REPLAY / "phase1_provenance.json").read_text(encoding="utf-8")
    )
    runs = pd.read_csv(root / REPLAY / "phase1_runs.tsv", sep="\t")
    decision = pd.read_csv(root / REPLAY / "phase1_decision.tsv", sep="\t")
    checks: list[dict[str, object]] = []
    record(
        checks,
        "protocol_version",
        protocol.get("protocol_version") == "stage1_v2_phase6_confirmation_v1",
        str(protocol.get("protocol_version")),
    )
    record(
        checks,
        "replay_pass",
        provenance.get("status") == "PASS"
        and provenance.get("protocol_version")
        == "stage1_v2_phase6_phase1_matched_guard_replay_v1",
        str(provenance.get("status")),
    )
    record(
        checks,
        "replay_complete",
        int(provenance.get("run_count", -1)) == 120
        and provenance.get("matched_seed_status") == "pass"
        and provenance.get("matched_validation_observation_status") == "pass"
        and provenance.get("matched_component_mask_status") == "pass",
        f"runs={provenance.get('run_count')}",
    )
    record(
        checks,
        "replay_exact_parent_metrics",
        float(provenance.get("parent_full_metric_maximum_absolute_delta", 1.0))
        == 0.0,
        str(provenance.get("parent_full_metric_maximum_absolute_delta")),
    )
    record(
        checks,
        "protected_outcomes_sealed",
        provenance.get("outer_test_outcomes_read") is False
        and provenance.get("outer_test_metrics_read") is False
        and provenance.get("final_holdout_outcomes_read") is False,
        "outer and final outcomes remained sealed",
    )
    record(
        checks,
        "formal_rank64_result_preserved",
        set(
            decision.loc[
                decision["decision"].astype(str).eq("advance_to_confirmation"),
                "configuration_label",
            ].astype(str)
        )
        == {"reaction_rank_64_regularized"},
        "original formal decision is bound but not rewritten",
    )
    rank64_reference = runs.loc[
        runs["candidate"].eq("ka_identity_location_baseline")
        & runs["configuration_label"].eq("reaction_rank_64_regularized")
    ]
    stable_reference = runs.loc[
        runs["candidate"].eq("ka_historical_environment")
        & runs["configuration_label"].eq("frozen_capacity_16")
    ]
    projection32 = runs.loc[
        runs["candidate"].eq("ka_projection_core")
        & runs["configuration_label"].eq("capacity_32")
    ]
    record(
        checks,
        "rank64_reference_pathological_calibration",
        len(rank64_reference) == 5
        and float(rank64_reference["validation_macro_calibration_error"].mean()) > 1.0,
        "rank64 identity/location calibration error exceeds one",
    )
    record(
        checks,
        "stable_capacity16_reference_supported",
        len(stable_reference) == 5
        and float(stable_reference["validation_macro_calibration_error"].mean()) < 0.2,
        "historical capacity-16 has acceptable absolute macro calibration",
    )
    record(
        checks,
        "projection_rank32_prior_supported",
        len(projection32) == 5,
        f"rows={len(projection32)}",
    )
    record(
        checks,
        "exact_candidate_scope",
        protocol.get("candidate_order")
        == [
            "historical_reaction_reference",
            "historical_v2_native_multikernel",
            "projection_reaction_routed_fallback",
        ],
        ",".join(protocol.get("candidate_order", [])),
    )
    record(
        checks,
        "exact_confirmation_grid",
        protocol.get("confirmation_grid", {}).get("state_count") == 125
        and protocol.get("confirmation_grid", {}).get("run_count") == 375,
        json.dumps(protocol.get("confirmation_grid", {}), sort_keys=True),
    )
    record(
        checks,
        "absolute_calibration_rule",
        protocol.get("scenario_route_selection", {}).get(
            "maximum_absolute_macro_calibration_error"
        )
        == 0.2,
        str(
            protocol.get("scenario_route_selection", {}).get(
                "maximum_absolute_macro_calibration_error"
            )
        ),
    )
    record(
        checks,
        "projection_fallback_frozen",
        "IDENTIFIER_ROUTED_HISTORICAL_FALLBACK_MAIN_EFFECTS"
        in protocol["candidates"]["projection_reaction_routed_fallback"][
            "environment_components"
        ],
        protocol["candidates"]["projection_reaction_routed_fallback"][
            "fallback_rule"
        ],
    )
    record(
        checks,
        "multikernel_comparator_frozen",
        protocol["candidates"]["historical_v2_native_multikernel"]["model_class"]
        == "multitrait_multikernel_main_effects",
        protocol["candidates"]["historical_v2_native_multikernel"]["model_class"],
    )
    for path in REPLAY_ARTIFACTS:
        record(
            checks,
            f"replay_artifact::{path.as_posix()}",
            (root / path).is_file(),
            str(root / path),
        )
    for path in (PROTOCOL, *IMPLEMENTATION):
        record(
            checks,
            f"implementation::{path.as_posix()}",
            (code_root / path).is_file(),
            str(code_root / path),
        )
    failed = [row for row in checks if row["status"] != "PASS"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(
        output / "validation_checks.tsv", sep="\t", index=False
    )
    if failed:
        write_json(
            lock_path,
            {
                "status": "FAIL_STAGE1_V2_PHASE6_CONFIRMATION_FREEZE",
                "failed_checks": [row["check"] for row in failed],
            },
        )
        raise ValueError("Stage-1 v2 Phase-6 confirmation freeze failed")

    replay_hashes = {
        path.as_posix(): sha256_file(root / path) for path in REPLAY_ARTIFACTS
    }
    implementation_hashes = {
        path.as_posix(): sha256_file(code_root / path)
        for path in IMPLEMENTATION
    }
    lock = {
        "status": "PASS_READY_FOR_STAGE1_V2_PHASE6_CONFIRMATION",
        "protocol_version": "stage1_v2_phase6_confirmation_lock_v1",
        "stage1_version": "Stage-1 v2",
        "selection_data": "previously_opened_phase1_inner_metrics_only",
        "adjudication_type": "new_versioned_inner_development_release_original_formal_decision_preserved",
        "stable_reference_candidate": protocol["stable_reference_candidate"],
        "frozen_candidates": protocol["candidate_order"],
        "required_state_count": 125,
        "required_run_count": 375,
        "code_commit": git_commit(code_root),
        "selection_protocol_sha256": sha256_file(code_root / PROTOCOL),
        "implementation_sha256": implementation_hashes,
        "source_replay_sha256": replay_hashes,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "outer_evaluation_allowed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(lock_path, lock)
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
