from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_trait_balance_screen_protocol_v1.json"
)
SOURCE_PROTOCOL = Path(
    "server_training_pipeline/"
    "stage1_v2_phase6_hierarchy_calibration_amendment_protocol_v2.json"
)
CORRECTION = Path(
    "audit/v2/"
    "stage1_v2_phase6_hierarchy_calibration_adjudication_correction_v1/"
    "CALIBRATION_ADJUDICATION_CORRECTION.json"
)
ROUTE_LOCK = Path(
    "audit/v2/stage1_v2_phase6_hierarchy_calibration_corrected_route_lock_v1/"
    "CORRECTED_HIERARCHY_ROUTE_LOCK.json"
)
OUTPUT = Path("audit/v2/stage1_v2_phase6_trait_balance_screen_v1")
TRAINER = Path("server_training_pipeline/train_stage1_v2_phase6_trait_balance_tf.py")
LOSS_HELPER = Path("server_training_pipeline/stage1_v2_phase6_trait_balance_v1.py")
CALIBRATION_HELPER = Path(
    "server_training_pipeline/stage1_v2_phase6_hierarchy_calibration_amendment_v2.py"
)
CALIBRATION_TRAINER = Path(
    "server_training_pipeline/"
    "train_stage1_v2_phase6_hierarchy_calibration_amendment_tf.py"
)
REMEDIATION_HELPER = Path("server_training_pipeline/stage1_v2_phase6_remediation.py")
REMEDIATION_TRAINER = Path(
    "server_training_pipeline/train_stage1_v2_phase6_remediation_tf.py"
)
FACTOR_BUILDER = Path("server_training_pipeline/train_stage1_v2_phase6_tf.py")
TRAINER_INTERFACE = Path("server_training_pipeline/stage1_v2_trainer_interface.py")
RUNNER = Path("scripts/v2/run_stage1_v2_phase6_trait_balance_screen.py")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the fixed-architecture Stage-1 v2 trait-balance screen"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = (args.code_root or root).resolve()
    paths = {
        "protocol": code_root / PROTOCOL,
        "source_protocol": code_root / SOURCE_PROTOCOL,
        "correction": root / CORRECTION,
        "route_lock": root / ROUTE_LOCK,
        "trainer": code_root / TRAINER,
        "loss_helper": code_root / LOSS_HELPER,
        "calibration_helper": code_root / CALIBRATION_HELPER,
        "calibration_trainer": code_root / CALIBRATION_TRAINER,
        "remediation_helper": code_root / REMEDIATION_HELPER,
        "remediation_trainer": code_root / REMEDIATION_TRAINER,
        "factor_builder": code_root / FACTOR_BUILDER,
        "trainer_interface": code_root / TRAINER_INTERFACE,
        "runner": code_root / RUNNER,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Trait-balance freeze inputs are missing: {missing}")
    protocol = read_json(paths["protocol"])
    source = read_json(paths["source_protocol"])
    correction = read_json(paths["correction"])
    route = read_json(paths["route_lock"])
    candidates = protocol["candidates"]
    new_candidates = [
        name for name, policy in candidates.items() if policy.get("source_reuse") is False
    ]
    checks = {
        "protocol_identity": protocol.get("protocol_version")
        == "stage1_v2_phase6_trait_balance_screen_v1",
        "stage1_v2": protocol.get("stage1_version") == "Stage-1 v2",
        "corrected_adjudication_pass": correction.get("status")
        == protocol["source_corrected_adjudication_status"],
        "corrected_huber_selected": correction.get("selected_candidate")
        == protocol["reference_source_candidate"],
        "corrected_route_pass": route.get("status")
        == protocol["source_corrected_route_status"],
        "route_trait_screen_allowed": route.get("trait_balance_screen_allowed") is True,
        "only_loss_mutable": protocol["architecture_policy"]["only_mutable_component"]
        == "training_loss_trait_mass",
        "fixed_configuration_exact": protocol["fixed_configuration"]
        == source["fixed_configuration"],
        "fixed_hierarchy_exact": protocol["trial_environment_hierarchy"]
        == source["trial_environment_hierarchy"],
        "fixed_trait_regularization_exact": protocol["trait_specific_regularization"]
        == source["trait_specific_regularization"],
        "fixed_non_target_calibration_exact": protocol["positive_training_calibration"]
        == source["positive_training_calibration"],
        "fixed_huber_calibration_exact": protocol["test_weight_environment_oof_calibration"]
        == source["test_weight_environment_oof_calibration"],
        "fixed_huber_candidate_exact": protocol.get("calibration_candidates", {})
        == {
            "hierarchy_test_weight_environment_oof_huber_v2": {
                "method": "environment_oof_huber",
                "purpose": "Fixed training-only TEST_WEIGHT calibration inherited from the corrected hierarchy reference",
            }
        },
        "batch_size_8192": int(protocol["fixed_configuration"]["batch_size"]) == 8192,
        "phase1_grid_5": int(protocol["phase_1_scope"]["state_count"]) == 5
        and protocol["phase_1_scope"]["outer_folds"] == [1, 2, 3, 4, 5]
        and int(protocol["phase_1_scope"]["inner_fold"]) == 1,
        "reference_reuse_5": int(protocol["phase_1_scope"]["reference_reuse_count"])
        == 5,
        "new_fit_count_10": int(protocol["phase_1_scope"]["new_candidate_fit_count"])
        == 10
        and len(new_candidates) == 2,
        "candidate_result_count_15": int(
            protocol["phase_1_scope"]["candidate_result_count"]
        )
        == 15,
        "projection_product_separate": protocol["product_policy"][
            "projection_compatible_product"
        ]
        == "unchanged_and_separate",
        "outer_and_final_unread": protocol["outer_test_metrics_read"] is False
        and protocol["outer_test_outcomes_read"] is False
        and protocol["final_holdout_outcomes_read"] is False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"check": name, "status": "PASS" if passed else "FAIL"} for name, passed in checks.items()]
    ).to_csv(
        output / "validation_checks.tsv", sep="\t", index=False, lineterminator="\n"
    )
    result = {
        "status": (
            "PASS_FROZEN_STAGE1_V2_PHASE6_TRAIT_BALANCE_SCREEN_V1"
            if not failed
            else "FAIL_STAGE1_V2_PHASE6_TRAIT_BALANCE_SCREEN_FREEZE"
        ),
        "protocol_version": "stage1_v2_phase6_trait_balance_screen_freeze_v1",
        "selection_data": "future_nested_inner_validation_only",
        "reference_candidate": protocol["reference_candidate"],
        "new_candidates": new_candidates,
        "state_count": 5,
        "new_model_fit_count": 10,
        "outer_evaluation_allowed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "checks": checks,
        "failed_checks": failed,
        "artifacts": {
            "trait_balance_protocol": sha256_file(paths["protocol"]),
            "source_calibration_protocol": sha256_file(paths["source_protocol"]),
            "corrected_adjudication": sha256_file(paths["correction"]),
            "corrected_route_lock": sha256_file(paths["route_lock"]),
            "trainer": sha256_file(paths["trainer"]),
            "loss_helper": sha256_file(paths["loss_helper"]),
            "calibration_helper": sha256_file(paths["calibration_helper"]),
            "calibration_trainer": sha256_file(paths["calibration_trainer"]),
            "remediation_helper": sha256_file(paths["remediation_helper"]),
            "remediation_trainer": sha256_file(paths["remediation_trainer"]),
            "factor_builder": sha256_file(paths["factor_builder"]),
            "trainer_interface": sha256_file(paths["trainer_interface"]),
            "runner": sha256_file(paths["runner"]),
            "validation_checks.tsv": sha256_file(output / "validation_checks.tsv"),
        },
    }
    write_json(output / "TRAIT_BALANCE_SCREEN_LOCK.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"Trait-balance screen freeze failed: {failed}")


if __name__ == "__main__":
    main()
