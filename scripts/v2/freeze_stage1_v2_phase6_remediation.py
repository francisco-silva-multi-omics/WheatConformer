from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_remediation_protocol_v1.json"
)
TRAINER = Path(
    "server_training_pipeline/train_stage1_v2_phase6_remediation_tf.py"
)
REMEDIATION_HELPER = Path(
    "server_training_pipeline/stage1_v2_phase6_remediation.py"
)
ORCHESTRATOR = Path("scripts/v2/run_stage1_v2_phase6_remediation.py")
CONFIRMATION_ANALYZER = Path(
    "scripts/v2/analyze_stage1_v2_phase6_confirmation_results.py"
)
FACTOR_BUILDER = Path("server_training_pipeline/train_stage1_v2_phase6_tf.py")
CONFIRMATION_TRAINER = Path(
    "server_training_pipeline/train_stage1_v2_phase6_confirmation_tf.py"
)
TESTS = (
    Path("tests/test_stage1_v2_phase6_remediation.py"),
    Path("tests/test_stage1_v2_phase6_remediation_tf.py"),
)
OUTPUT = Path("audit/v2/stage1_v2_phase6_remediation_v1")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the Stage-1 v2 structural-remediation screen"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--confirmation-summary-root",
        type=Path,
        default=Path(
            "retrieved_phase6_confirmation/extracted_dc522b71b/"
            "stage1_v2_phase6_confirmation_results/summary"
        ),
    )
    parser.add_argument(
        "--analysis-provenance",
        type=Path,
        default=Path(
            "audit/v2/stage1_v2_phase6_confirmation_analysis_v1/"
            "analysis_provenance.json"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    summary_root = (
        args.confirmation_summary_root
        if args.confirmation_summary_root.is_absolute()
        else root / args.confirmation_summary_root
    ).resolve()
    analysis_path = (
        args.analysis_provenance
        if args.analysis_provenance.is_absolute()
        else root / args.analysis_provenance
    ).resolve()
    status_path = summary_root / "confirmation_status.json"
    route_path = summary_root / "CONFIRMATION_SCENARIO_ROUTE_LOCK.json"
    required = [
        code_root / PROTOCOL,
        code_root / TRAINER,
        code_root / REMEDIATION_HELPER,
        code_root / ORCHESTRATOR,
        code_root / CONFIRMATION_ANALYZER,
        code_root / FACTOR_BUILDER,
        code_root / CONFIRMATION_TRAINER,
        *(code_root / path for path in TESTS),
        status_path,
        route_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Remediation freeze inputs are missing: {missing}")
    protocol = read_json(code_root / PROTOCOL)
    status = read_json(status_path)
    route = read_json(route_path)
    analysis = read_json(analysis_path) if analysis_path.is_file() else None
    selected_routes = route.get("selected_scenario_routes", {})
    expected_scenarios = set(protocol["phase_1"]["scenarios"])
    checks = {
        "protocol_identity": protocol.get("protocol_version")
        == "stage1_v2_phase6_structural_remediation_v1",
        "confirmation_complete": status.get("status")
        == "PASS_STAGE1_V2_PHASE6_CONFIRMATION_COMPLETE",
        "confirmation_run_count": int(status.get("run_count", -1)) == 375,
        "confirmation_state_count": int(status.get("state_count", -1)) == 125,
        "confirmation_outer_unread": status.get("outer_test_metrics_read") is False
        and status.get("outer_test_outcomes_read") is False,
        "confirmation_final_unread": status.get("final_holdout_outcomes_read") is False,
        "route_lock_pass": route.get("status")
        == "PASS_STAGE1_V2_PHASE6_SCENARIO_ROUTES_FROZEN",
        "route_scenario_grid": set(selected_routes) == expected_scenarios,
        "stable_reference_all_routes": set(selected_routes.values())
        == {"historical_reaction_reference"},
        "remediation_outer_unread": protocol.get("outer_test_metrics_read") is False,
        "remediation_final_unread": protocol.get("final_holdout_outcomes_read") is False,
        "phase1_grid_frozen": int(protocol["phase_1"]["candidate_state_count"]) == 70,
        "phase2_optimizer_blocked": protocol["phase_2_optimizer_screen"]["status"]
        == "blocked_until_phase_1_candidate_acceptance",
        "analysis_consistent": analysis is None
        or analysis.get("status")
        == "PASS_CONFIRMATION_EVIDENCE_WITH_REMEDIATION_REQUIRED_BEFORE_OUTER_EVALUATION",
        "analysis_outer_unread": analysis is None
        or analysis.get("outer_test_metrics_read") is False,
        "analysis_final_unread": analysis is None
        or analysis.get("final_holdout_outcomes_read") is False,
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
    if analysis_path.is_file():
        artifacts[str(analysis_path)] = {
            "path": str(analysis_path),
            "sha256": sha256_file(analysis_path),
            "bytes": analysis_path.stat().st_size,
        }
    freeze = {
        "status": (
            "PASS_FROZEN_BEFORE_REMEDIATION_INNER_VALIDATION"
            if not failed
            else "FAIL_REMEDIATION_FREEZE"
        ),
        "protocol_version": "stage1_v2_phase6_remediation_freeze_v1",
        "selection_data": "completed_confirmation_inner_metrics_and_frozen_identifiers_only",
        "source_confirmation_routes": selected_routes,
        "phase1_run_count": int(protocol["phase_1"]["candidate_state_count"]),
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "phase2_optimizer_allowed": False,
        "checks": checks,
        "failed_checks": failed,
        "artifacts": artifacts,
        "code_commit": git_commit(code_root),
    }
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "PHASE6_REMEDIATION_LOCK.json"
    lock_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validation = [
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "detail": "",
        }
        for name, passed in checks.items()
    ]
    import pandas as pd

    pd.DataFrame(validation).to_csv(
        output / "validation_checks.tsv", sep="\t", index=False
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))
    if failed:
        raise SystemExit("Stage-1 v2 remediation freeze failed")


if __name__ == "__main__":
    main()
