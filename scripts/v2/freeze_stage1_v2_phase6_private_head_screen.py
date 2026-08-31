from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_private_head_screen_protocol_v1.json"
)
SOURCE_PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_trait_balance_screen_protocol_v1.json"
)
SOURCE_DECISION = Path(
    "model_kernels/stage1_v2_phase6_trait_balance_screen_v1/phase_1/"
    "TRAIT_BALANCE_PHASE1_DECISION.json"
)
POST_HIERARCHY_PLAN = Path(
    "server_training_pipeline/stage1_v2_phase6_post_hierarchy_screen_plan_v2.json"
)
OUTPUT = Path("audit/v2/stage1_v2_phase6_private_head_screen_v1")
TRAINER = Path("server_training_pipeline/train_stage1_v2_phase6_private_heads_tf.py")
CALIBRATION_HELPER = Path(
    "server_training_pipeline/stage1_v2_phase6_hierarchy_calibration_amendment_v2.py"
)
CALIBRATION_TRAINER = Path(
    "server_training_pipeline/"
    "train_stage1_v2_phase6_hierarchy_calibration_amendment_tf.py"
)
REMEDIATION_HELPER = Path("server_training_pipeline/stage1_v2_phase6_remediation.py")
MODEL_BUILDER = Path("server_training_pipeline/train_stage1_v2_phase6_remediation_tf.py")
FACTOR_BUILDER = Path("server_training_pipeline/train_stage1_v2_phase6_tf.py")
TRAINER_INTERFACE = Path("server_training_pipeline/stage1_v2_trainer_interface.py")
RUNNER = Path("scripts/v2/run_stage1_v2_phase6_private_head_screen.py")


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
        description="Freeze the Stage-1 v2 trait-family private-head screen"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = (args.code_root or root).resolve()
    paths = {
        "protocol": code_root / PROTOCOL,
        "source_protocol": code_root / SOURCE_PROTOCOL,
        "source_decision": root / SOURCE_DECISION,
        "post_hierarchy_plan": code_root / POST_HIERARCHY_PLAN,
        "trainer": code_root / TRAINER,
        "calibration_helper": code_root / CALIBRATION_HELPER,
        "calibration_trainer": code_root / CALIBRATION_TRAINER,
        "remediation_helper": code_root / REMEDIATION_HELPER,
        "model_builder": code_root / MODEL_BUILDER,
        "factor_builder": code_root / FACTOR_BUILDER,
        "trainer_interface": code_root / TRAINER_INTERFACE,
        "runner": code_root / RUNNER,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Private-head freeze inputs are missing: {missing}")

    protocol = read_json(paths["protocol"])
    source_protocol = read_json(paths["source_protocol"])
    source_decision = read_json(paths["source_decision"])
    plan = read_json(paths["post_hierarchy_plan"])
    new_candidates = [
        name
        for name, value in protocol["candidates"].items()
        if value.get("source_reuse") is False
    ]
    decoder_modes = {
        protocol["candidates"][name]["decoder_policy"]["mode"]
        for name in new_candidates
    }
    gate = next(
        entry
        for entry in plan["ordered_gates"]
        if entry["name"] == "trait_family_private_heads"
    )
    expected_traits = {
        *protocol["primary_traits"],
        *protocol["exploratory_traits"],
    }
    checks = {
        "protocol_identity": protocol.get("protocol_version")
        == "stage1_v2_phase6_private_head_screen_v1",
        "stage1_v2": protocol.get("stage1_version") == "Stage-1 v2",
        "parent_trait_balance_terminal": source_decision.get("status")
        == protocol["parent_terminal_status"],
        "parent_selected_no_candidate": source_decision.get("selected_candidate") is None,
        "parent_full_confirmation_blocked": source_decision.get(
            "full_confirmation_allowed"
        )
        is False,
        "parent_outer_and_final_unread": source_decision.get(
            "outer_test_metrics_read"
        )
        is False
        and source_decision.get("outer_test_outcomes_read") is False
        and source_decision.get("final_holdout_outcomes_read") is False,
        "ordered_gate_exact": gate.get("only_mutable_component")
        == "trait_decoder_sharing",
        "only_decoder_sharing_mutable": protocol["architecture_policy"][
            "only_mutable_component"
        ]
        == "trait_decoder_sharing",
        "authoritative_row_mass_fixed": protocol["architecture_policy"][
            "fixed_authoritative_row_mass"
        ]
        is True,
        "fixed_configuration_exact": protocol["fixed_configuration"]
        == source_protocol["fixed_configuration"],
        "fixed_hierarchy_exact": protocol["trial_environment_hierarchy"]
        == source_protocol["trial_environment_hierarchy"],
        "fixed_trait_regularization_exact": protocol["trait_specific_regularization"]
        == source_protocol["trait_specific_regularization"],
        "fixed_non_target_calibration_exact": protocol["positive_training_calibration"]
        == source_protocol["positive_training_calibration"],
        "fixed_huber_calibration_exact": protocol[
            "test_weight_environment_oof_calibration"
        ]
        == source_protocol["test_weight_environment_oof_calibration"],
        "batch_size_8192": int(protocol["fixed_configuration"]["batch_size"])
        == 8192,
        "trait_family_axis_complete": set(protocol["trait_families"])
        == expected_traits,
        "two_decoder_candidates": len(new_candidates) == 2
        and decoder_modes
        == {
            "trait_private_residual",
            "family_shared_trait_private_residual",
        },
        "zero_residual_initialization": protocol["architecture_policy"][
            "private_decoder_initialization"
        ]
        == "zero_residual_from_shared_reference_parameterization",
        "phase1_grid_5": int(protocol["phase_1_scope"]["state_count"]) == 5
        and protocol["phase_1_scope"]["outer_folds"] == [1, 2, 3, 4, 5]
        and int(protocol["phase_1_scope"]["inner_fold"]) == 1,
        "reference_reuse_5": int(
            protocol["phase_1_scope"]["reference_reuse_count"]
        )
        == 5,
        "new_fit_count_10": int(
            protocol["phase_1_scope"]["new_candidate_fit_count"]
        )
        == 10,
        "same_seed_replay_count_2": int(
            protocol["phase_1_scope"]["same_seed_replay_fit_count"]
        )
        == 2,
        "tensorflow_determinism_required": protocol["integrity_hardening"][
            "tensorflow_op_determinism_required"
        ]
        is True,
        "finite_assertions_required": protocol["integrity_hardening"][
            "per_batch_finite_prediction_loss_gradient_assertions_required"
        ]
        is True,
        "cache_expected_state_validation_required": protocol[
            "integrity_hardening"
        ]["factor_cache_expected_state_hash_validation_required"]
        is True,
        "checkpoint_and_predictions_required": protocol["integrity_hardening"][
            "best_model_weights_persisted_and_checksummed"
        ]
        is True
        and protocol["integrity_hardening"][
            "raw_training_and_validation_predictions_persisted_and_checksummed"
        ]
        is True
        and protocol["integrity_hardening"][
            "calibrated_validation_predictions_persisted_and_checksummed"
        ]
        is True,
        "outer_os_file_audit_preregistered": protocol["integrity_hardening"][
            "file_access_attestation"
        ]["outer_evaluation_requires_syscall_or_os_audit_manifest"]
        is True,
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
        [
            {"check": name, "status": "PASS" if passed else "FAIL"}
            for name, passed in checks.items()
        ]
    ).to_csv(
        output / "validation_checks.tsv", sep="\t", index=False, lineterminator="\n"
    )
    result = {
        "status": (
            "PASS_FROZEN_STAGE1_V2_PHASE6_PRIVATE_HEAD_SCREEN_V1"
            if not failed
            else "FAIL_STAGE1_V2_PHASE6_PRIVATE_HEAD_SCREEN_FREEZE"
        ),
        "protocol_version": "stage1_v2_phase6_private_head_screen_freeze_v1",
        "selection_data": "completed_parent_inner_decision_and_future_inner_validation_only",
        "reference_candidate": protocol["reference_candidate"],
        "new_candidates": new_candidates,
        "state_count": 5,
        "new_model_fit_count": 10,
        "same_seed_replay_fit_count": 2,
        "outer_evaluation_allowed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "checks": checks,
        "failed_checks": failed,
        "artifacts": {
            "private_head_protocol": sha256_file(paths["protocol"]),
            "source_trait_balance_protocol": sha256_file(paths["source_protocol"]),
            "source_trait_balance_decision": sha256_file(paths["source_decision"]),
            "post_hierarchy_plan": sha256_file(paths["post_hierarchy_plan"]),
            "trainer": sha256_file(paths["trainer"]),
            "calibration_helper": sha256_file(paths["calibration_helper"]),
            "calibration_trainer": sha256_file(paths["calibration_trainer"]),
            "remediation_helper": sha256_file(paths["remediation_helper"]),
            "model_builder": sha256_file(paths["model_builder"]),
            "factor_builder": sha256_file(paths["factor_builder"]),
            "trainer_interface": sha256_file(paths["trainer_interface"]),
            "runner": sha256_file(paths["runner"]),
            "validation_checks.tsv": sha256_file(output / "validation_checks.tsv"),
        },
    }
    write_json(output / "PRIVATE_HEAD_SCREEN_LOCK.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"Private-head screen freeze failed: {failed}")


if __name__ == "__main__":
    main()
