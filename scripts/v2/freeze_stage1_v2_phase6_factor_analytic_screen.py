from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_factor_analytic_screen_protocol_v1.json"
)
PARENT_PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_private_head_screen_protocol_v1.json"
)
PARENT_DECISION = Path(
    "model_kernels/stage1_v2_phase6_private_head_screen_v1/phase_1/"
    "PRIVATE_HEAD_PHASE1_DECISION.json"
)
POST_HIERARCHY_PLAN = Path(
    "server_training_pipeline/stage1_v2_phase6_post_hierarchy_screen_plan_v2.json"
)
PROJECTION_PROTOCOL = Path(
    "server_training_pipeline/phase6a_split_bound_projection_inputs_protocol_v1.json"
)
TRAINER = Path(
    "server_training_pipeline/train_stage1_v2_phase6_factor_analytic_tf.py"
)
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
RUNNER = Path("scripts/v2/run_stage1_v2_phase6_factor_analytic_screen.py")
OUTPUT = Path("audit/v2/stage1_v2_phase6_factor_analytic_screen_v1")


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
        description="Freeze the Stage-1 v2 covariate-linked FA screen"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = (args.code_root or root).resolve()
    paths = {
        "protocol": code_root / PROTOCOL,
        "parent_protocol": code_root / PARENT_PROTOCOL,
        "parent_decision": root / PARENT_DECISION,
        "post_hierarchy_plan": code_root / POST_HIERARCHY_PLAN,
        "projection_protocol": code_root / PROJECTION_PROTOCOL,
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
        raise FileNotFoundError(f"Factor-analytic freeze inputs are missing: {missing}")

    protocol = read_json(paths["protocol"])
    parent_protocol = read_json(paths["parent_protocol"])
    parent_decision = read_json(paths["parent_decision"])
    plan = read_json(paths["post_hierarchy_plan"])
    projection_protocol = read_json(paths["projection_protocol"])
    gate = next(
        entry
        for entry in plan["ordered_gates"]
        if entry["name"] == "covariate_linked_factor_analytic_decomposition"
    )
    new_candidates = {
        name: value
        for name, value in protocol["candidates"].items()
        if value.get("source_reuse") is False
    }
    ranks = {int(value["factor_analytic_rank"]) for value in new_candidates.values()}
    architecture = protocol["architecture_policy"]
    feature_contract = protocol["projection_feature_contract"]
    checks = {
        "protocol_identity": protocol.get("protocol_version")
        == "stage1_v2_phase6_factor_analytic_screen_v1",
        "stage1_v2": protocol.get("stage1_version") == "Stage-1 v2",
        "parent_private_head_terminal": parent_decision.get("status")
        == protocol["parent_terminal_status"],
        "parent_selected_no_candidate": parent_decision.get("selected_candidate")
        is None,
        "parent_confirmation_blocked": parent_decision.get(
            "full_confirmation_allowed"
        )
        is False,
        "parent_outer_and_final_unread": parent_decision.get(
            "outer_test_metrics_read"
        )
        is False
        and parent_decision.get("outer_test_outcomes_read") is False
        and parent_decision.get("final_holdout_outcomes_read") is False,
        "ordered_gate_exact": int(gate.get("order", -1)) == 3,
        "prospective_covariate_rule": "certified prospective covariates"
        in str(gate.get("projection_rule", "")),
        "only_FA_residual_mutable": architecture["only_mutable_component"]
        == "covariate_linked_factor_analytic_residual",
        "historical_backbone_fixed": architecture["historical_backbone_changed"]
        is False,
        "hierarchy_fixed": architecture["hierarchy_changed"] is False,
        "calibration_fixed": architecture["calibration_changed"] is False,
        "authoritative_row_mass_fixed": architecture[
            "authoritative_row_mass_changed"
        ]
        is False,
        "free_environment_loadings_forbidden": architecture[
            "free_environment_loadings_allowed"
        ]
        is False,
        "projection_inactive_zero": architecture["projection_inactive_policy"]
        == "exact_zero_FA_residual",
        "fixed_configuration_exact": protocol["fixed_configuration"]
        == parent_protocol["fixed_configuration"],
        "fixed_hierarchy_exact": protocol["trial_environment_hierarchy"]
        == parent_protocol["trial_environment_hierarchy"],
        "fixed_trait_regularization_exact": protocol[
            "trait_specific_regularization"
        ]
        == parent_protocol["trait_specific_regularization"],
        "fixed_non_target_calibration_exact": protocol[
            "positive_training_calibration"
        ]
        == parent_protocol["positive_training_calibration"],
        "fixed_huber_calibration_exact": protocol[
            "test_weight_environment_oof_calibration"
        ]
        == parent_protocol["test_weight_environment_oof_calibration"],
        "projection_protocol_identity": projection_protocol.get(
            "protocol_version"
        )
        == "phase6a_split_bound_historical_projection_inputs_v1",
        "projection_schema_153": int(feature_contract["expected_feature_count"])
        == 153
        and int(projection_protocol["expected_feature_count"]) == 153,
        "projection_split_bound": feature_contract[
            "training_only_imputation_scaling_and_factorization_required"
        ]
        is True
        and feature_contract["held_out_environments_use_frozen_training_parameters"]
        is True,
        "future_values_forbidden": feature_contract["future_SSP_values_read"]
        is False
        and feature_contract["future_covariate_matrices_used_for_training"] is False,
        "two_bounded_candidates": set(new_candidates)
        == {"covariate_linked_FA_rank2", "covariate_linked_FA_rank4"}
        and ranks == {2, 4},
        "phase1_grid_5": int(protocol["phase_1_scope"]["state_count"]) == 5
        and protocol["phase_1_scope"]["outer_folds"] == [1, 2, 3, 4, 5]
        and int(protocol["phase_1_scope"]["inner_fold"]) == 1,
        "new_fit_count_10": int(
            protocol["phase_1_scope"]["new_candidate_fit_count"]
        )
        == 10,
        "same_seed_replay_count_2": int(
            protocol["phase_1_scope"]["same_seed_replay_fit_count"]
        )
        == 2,
        "acceptance_rules_unchanged": protocol["phase_1_acceptance"]
        == parent_protocol["phase_1_acceptance"],
        "tensorflow_determinism_required": protocol["integrity_hardening"][
            "tensorflow_op_determinism_required"
        ]
        is True,
        "finite_assertions_required": protocol["integrity_hardening"][
            "per_batch_finite_prediction_loss_gradient_assertions_required"
        ]
        is True,
        "outer_os_file_audit_preregistered": protocol["integrity_hardening"][
            "file_access_attestation"
        ]["outer_evaluation_requires_syscall_or_os_audit_manifest"]
        is True,
        "outer_and_final_unread": protocol["outer_test_metrics_read"] is False
        and protocol["outer_test_outcomes_read"] is False
        and protocol["final_holdout_outcomes_read"] is False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    validation = pd.DataFrame(
        [
            {"check": name, "status": "PASS" if passed else "FAIL"}
            for name, passed in checks.items()
        ]
    )
    validation.to_csv(
        output / "validation_checks.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    result = {
        "status": (
            "PASS_FROZEN_STAGE1_V2_PHASE6_FACTOR_ANALYTIC_SCREEN_V1"
            if not failed
            else "FAIL_STAGE1_V2_PHASE6_FACTOR_ANALYTIC_SCREEN_FREEZE"
        ),
        "protocol_version": "stage1_v2_phase6_factor_analytic_screen_freeze_v1",
        "selection_data": "completed_parent_inner_decision_and_future_inner_validation_only",
        "reference_candidate": protocol["reference_candidate"],
        "new_candidates": list(new_candidates),
        "factor_analytic_ranks": sorted(ranks),
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
            "factor_analytic_protocol": sha256_file(paths["protocol"]),
            "parent_private_head_protocol": sha256_file(paths["parent_protocol"]),
            "parent_private_head_decision": sha256_file(paths["parent_decision"]),
            "post_hierarchy_plan": sha256_file(paths["post_hierarchy_plan"]),
            "projection_protocol": sha256_file(paths["projection_protocol"]),
            "trainer": sha256_file(paths["trainer"]),
            "calibration_helper": sha256_file(paths["calibration_helper"]),
            "calibration_trainer": sha256_file(paths["calibration_trainer"]),
            "remediation_helper": sha256_file(paths["remediation_helper"]),
            "model_builder": sha256_file(paths["model_builder"]),
            "factor_builder": sha256_file(paths["factor_builder"]),
            "trainer_interface": sha256_file(paths["trainer_interface"]),
            "runner": sha256_file(paths["runner"]),
            "validation_checks.tsv": sha256_file(
                output / "validation_checks.tsv"
            ),
        },
    }
    write_json(output / "FACTOR_ANALYTIC_SCREEN_LOCK.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"Factor-analytic screen freeze failed: {failed}")


if __name__ == "__main__":
    main()
