from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROTOCOL = Path(
    "server_training_pipeline/"
    "stage1_v2_phase6_factor_analytic_optimization_amendment_protocol_v1.json"
)
PARENT_PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_factor_analytic_screen_protocol_v1.json"
)
PARENT_DECISION = Path(
    "model_kernels/stage1_v2_phase6_factor_analytic_screen_v1/phase_1/"
    "FACTOR_ANALYTIC_PHASE1_DECISION.json"
)
POST_HIERARCHY_PLAN = Path(
    "server_training_pipeline/stage1_v2_phase6_post_hierarchy_screen_plan_v2.json"
)
PROJECTION_PROTOCOL = Path(
    "server_training_pipeline/phase6a_split_bound_projection_inputs_protocol_v1.json"
)
TRAINER = Path(
    "server_training_pipeline/"
    "train_stage1_v2_phase6_factor_analytic_optimization_amendment_tf.py"
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
PRIVATE_HEAD_DECISION = Path(
    "model_kernels/stage1_v2_phase6_private_head_screen_v1/phase_1/"
    "PRIVATE_HEAD_PHASE1_DECISION.json"
)
RUNNER = Path(
    "scripts/v2/"
    "run_stage1_v2_phase6_factor_analytic_optimization_amendment.py"
)
OUTPUT = Path(
    "audit/v2/stage1_v2_phase6_factor_analytic_optimization_amendment_v1"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the Stage-1 v2 FA optimization amendment"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = (args.code_root or root).resolve()
    paths = {
        "protocol": code_root / PROTOCOL,
        "parent_factor_analytic_protocol": code_root / PARENT_PROTOCOL,
        "parent_factor_analytic_decision": root / PARENT_DECISION,
        "post_hierarchy_plan": code_root / POST_HIERARCHY_PLAN,
        "projection_protocol": code_root / PROJECTION_PROTOCOL,
        "trainer": code_root / TRAINER,
        "calibration_helper": code_root / CALIBRATION_HELPER,
        "calibration_trainer": code_root / CALIBRATION_TRAINER,
        "remediation_helper": code_root / REMEDIATION_HELPER,
        "model_builder": code_root / MODEL_BUILDER,
        "factor_builder": code_root / FACTOR_BUILDER,
        "trainer_interface": code_root / TRAINER_INTERFACE,
        "private_head_decision": root / PRIVATE_HEAD_DECISION,
        "runner": code_root / RUNNER,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"FA amendment freeze inputs are missing: {missing}")

    protocol = read_json(paths["protocol"])
    parent_protocol = read_json(paths["parent_factor_analytic_protocol"])
    parent_decision = read_json(paths["parent_factor_analytic_decision"])
    projection_protocol = read_json(paths["projection_protocol"])
    objective = protocol["objective_policy"]
    architecture = protocol["architecture_policy"]
    activity = protocol["activity_certification"]
    primary_macro = set(objective["primary_macro_traits"])
    training_traits = set(objective["training_likelihood_traits"])
    demoted = set(objective["demoted_from_primary_macro"])
    candidates = {
        name: value
        for name, value in protocol["candidates"].items()
        if value.get("source_reuse") is False
    }
    ranks = {int(value["factor_analytic_rank"]) for value in candidates.values()}
    retention_flags = [
        "seven_trait_metrics_preserved",
        "all_seven_traits_retained_in_training_rows",
        "test_weight_predictions_retained",
        "test_weight_trait_reporting_retained",
        "test_weight_subset_reporting_retained",
        "test_weight_training_only_huber_calibration_retained",
        "test_weight_negative_slope_guard_retained",
        "test_weight_exploratory_non_deterioration_guard_retained",
    ]
    checks = {
        "protocol_identity": protocol.get("protocol_version")
        == "stage1_v2_phase6_factor_analytic_optimization_amendment_v1",
        "stage1_v2": protocol.get("stage1_version") == "Stage-1 v2",
        "parent_V1_terminal_no_advance": parent_decision.get("status")
        == "PASS_STAGE1_V2_PHASE6_FACTOR_ANALYTIC_PHASE1_COMPLETE_NO_ADVANCE"
        and parent_decision.get("selected_candidate") is None
        and parent_decision.get("full_confirmation_allowed") is False,
        "parent_V1_preserved_as_implementation_result": protocol.get(
            "parent_interpretation"
        )
        == "valid_no_advance_for_implementation_v1_not_a_biological_rejection_of_FA",
        "parent_outer_and_final_unread": parent_decision.get(
            "outer_test_metrics_read"
        )
        is False
        and parent_decision.get("outer_test_outcomes_read") is False
        and parent_decision.get("final_holdout_outcomes_read") is False,
        "training_likelihood_retains_seven_traits": len(training_traits) == 7
        and training_traits == {
            *parent_protocol["primary_traits"],
            *parent_protocol["exploratory_traits"],
        },
        "primary_macro_has_exactly_six_traits": len(primary_macro) == 6,
        "only_TEST_WEIGHT_demoted": demoted == {"TEST_WEIGHT"}
        and training_traits - primary_macro == demoted,
        "TEST_WEIGHT_retention_complete": all(
            objective.get(name) is True for name in retention_flags
        ),
        "six_trait_selection_and_early_stopping": objective.get(
            "early_stopping_metric"
        )
        == "six_trait_validation_macro_normalized_rmse"
        and objective.get("selection_primary_metric")
        == "six_trait_validation_macro_normalized_rmse",
        "only_FA_optimization_mutable": architecture.get(
            "only_mutable_component"
        )
        == "covariate_linked_factor_analytic_optimization",
        "directions_normalized": architecture.get(
            "genotype_and_environment_direction_columns_unit_normalized"
        )
        is True,
        "amplitude_parameterization": architecture.get(
            "trait_loadings_carry_factor_amplitude"
        )
        is True,
        "direction_L2_disabled": architecture.get(
            "genotype_direction_L2_penalty"
        )
        is False
        and architecture.get("environment_direction_L2_penalty") is False,
        "trait_amplitude_L2_active": architecture.get(
            "trait_amplitude_loading_L2_penalty"
        )
        is True,
        "historical_hierarchy_calibration_fixed": architecture.get(
            "historical_backbone_changed"
        )
        is False
        and architecture.get("hierarchy_changed") is False
        and architecture.get("calibration_changed") is False
        and architecture.get("authoritative_row_mass_changed") is False,
        "free_environment_loadings_forbidden": architecture.get(
            "free_environment_loadings_allowed"
        )
        is False,
        "projection_inactive_zero": architecture.get(
            "projection_inactive_policy"
        )
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
        "acceptance_thresholds_exact": protocol["phase_1_acceptance"]
        == parent_protocol["phase_1_acceptance"],
        "trait_guards_exact": protocol["primary_traits"]
        == parent_protocol["primary_traits"]
        and protocol["exploratory_traits"] == parent_protocol["exploratory_traits"],
        "reporting_subsets_exact": protocol["mandatory_reporting_subsets"]
        == parent_protocol["mandatory_reporting_subsets"],
        "projection_protocol_identity": projection_protocol.get(
            "protocol_version"
        )
        == "phase6a_split_bound_historical_projection_inputs_v1",
        "projection_schema_153": int(
            protocol["projection_feature_contract"]["expected_feature_count"]
        )
        == 153
        and int(projection_protocol["expected_feature_count"]) == 153,
        "future_values_forbidden": protocol["projection_feature_contract"][
            "future_SSP_values_read"
        ]
        is False
        and protocol["projection_feature_contract"][
            "future_covariate_matrices_used_for_training"
        ]
        is False,
        "two_bounded_candidates": set(candidates)
        == {"normalized_direction_FA_rank2", "normalized_direction_FA_rank4"}
        and ranks == {2, 4},
        "activity_diagnostics_every_epoch": activity.get(
            "diagnostics_written_every_epoch"
        )
        is True,
        "activity_thresholds_positive": all(
            float(activity[name]) > 0
            for name in (
                "minimum_raw_direction_column_norm",
                "maximum_normalized_direction_norm_error",
                "minimum_observed_gradient_norm_per_FA_tensor",
                "minimum_initial_training_residual_rms",
                "minimum_final_validation_residual_rms_for_performance_eligibility",
            )
        ),
        "inactive_component_is_not_integrity_failure": activity.get(
            "inactive_final_component_policy"
        )
        == "valid_null_component_no_advance",
        "failed_optimization_is_integrity_failure": activity.get(
            "optimization_path_failure_policy"
        )
        == "fail_screen_integrity",
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
        "determinism_and_finite_checks_required": protocol[
            "integrity_hardening"
        ]["tensorflow_op_determinism_required"]
        is True
        and protocol["integrity_hardening"][
            "per_batch_finite_prediction_loss_gradient_assertions_required"
        ]
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
    validation_path = output / "validation_checks.tsv"
    validation.to_csv(
        validation_path, sep="\t", index=False, lineterminator="\n"
    )
    result = {
        "status": (
            "PASS_FROZEN_STAGE1_V2_PHASE6_FA_OPTIMIZATION_AMENDMENT_V1"
            if not failed
            else "FAIL_STAGE1_V2_PHASE6_FA_OPTIMIZATION_AMENDMENT_FREEZE"
        ),
        "protocol_version": (
            "stage1_v2_phase6_factor_analytic_optimization_amendment_freeze_v1"
        ),
        "selection_data": (
            "terminal_parent_inner_decision_and_future_inner_validation_only"
        ),
        "parent_interpretation": protocol["parent_interpretation"],
        "reference_candidate": protocol["reference_candidate"],
        "new_candidates": list(candidates),
        "factor_analytic_ranks": sorted(ranks),
        "training_likelihood_trait_count": 7,
        "primary_macro_trait_count": 6,
        "TEST_WEIGHT_retained_outside_primary_macro": True,
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
            name: sha256_file(path) for name, path in paths.items()
        }
        | {"validation_checks.tsv": sha256_file(validation_path)},
    }
    write_json(output / "FA_OPTIMIZATION_AMENDMENT_LOCK.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"FA optimization-amendment freeze failed: {failed}")


if __name__ == "__main__":
    main()
