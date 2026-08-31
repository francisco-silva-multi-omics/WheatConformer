from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_factor_analytic_screen_protocol_v1.json"
)
PARENT = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_private_head_screen_protocol_v1.json"
)
PLAN = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_post_hierarchy_screen_plan_v2.json"
)
TRAINER = (
    ROOT
    / "server_training_pipeline"
    / "train_stage1_v2_phase6_factor_analytic_tf.py"
)
RUNNER = ROOT / "scripts" / "v2" / "run_stage1_v2_phase6_factor_analytic_screen.py"
FREEZE = ROOT / "scripts" / "v2" / "freeze_stage1_v2_phase6_factor_analytic_screen.py"
PACKAGER = (
    ROOT
    / "scripts"
    / "v2"
    / "package_stage1_v2_phase6_factor_analytic_results.py"
)


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_FA_gate_is_third_and_uses_only_prospective_environment_loadings() -> None:
    value = protocol()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    gate = plan["ordered_gates"][2]
    policy = value["architecture_policy"]
    assert gate["name"] == "covariate_linked_factor_analytic_decomposition"
    assert "certified prospective covariates" in gate["projection_rule"]
    assert policy["only_mutable_component"] == (
        "covariate_linked_factor_analytic_residual"
    )
    assert policy["environment_loading_source"] == (
        "split_bound_E_PROJECTION_CORE_V1_standardized_features"
    )
    assert policy["free_environment_loadings_allowed"] is False
    assert policy["free_environment_identifier_residuals_in_FA_component"] is False
    assert policy["projection_inactive_policy"] == "exact_zero_FA_residual"


def test_FA_gate_preserves_the_private_head_reference_contract() -> None:
    value = protocol()
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    assert value["parent_terminal_status"] == (
        "PASS_STAGE1_V2_PHASE6_PRIVATE_HEAD_PHASE1_COMPLETE_NO_ADVANCE"
    )
    assert value["fixed_configuration"] == parent["fixed_configuration"]
    assert value["trial_environment_hierarchy"] == parent[
        "trial_environment_hierarchy"
    ]
    assert value["trait_specific_regularization"] == parent[
        "trait_specific_regularization"
    ]
    assert value["positive_training_calibration"] == parent[
        "positive_training_calibration"
    ]
    assert value["test_weight_environment_oof_calibration"] == parent[
        "test_weight_environment_oof_calibration"
    ]
    assert value["phase_1_acceptance"] == parent["phase_1_acceptance"]
    assert value["fixed_configuration"]["batch_size"] == 8192


def test_FA_candidates_are_bounded_to_ranks_two_and_four() -> None:
    candidates = {
        name: candidate
        for name, candidate in protocol()["candidates"].items()
        if candidate.get("source_reuse") is False
    }
    assert set(candidates) == {
        "covariate_linked_FA_rank2",
        "covariate_linked_FA_rank4",
    }
    assert {candidate["factor_analytic_rank"] for candidate in candidates.values()} == {
        2,
        4,
    }
    assert all(
        candidate["factor_penalty_multiplier"] == 1.0
        for candidate in candidates.values()
    )


def test_projection_feature_contract_is_split_bound_and_future_blind() -> None:
    contract = protocol()["projection_feature_contract"]
    assert contract["expected_feature_count"] == 153
    assert contract["split_bound_state_binding"] == "ka_projection_core"
    assert contract[
        "training_only_imputation_scaling_and_factorization_required"
    ] is True
    assert contract["held_out_environments_use_frozen_training_parameters"] is True
    assert contract["explicit_missing_input_masks_required"] is True
    assert contract["future_SSP_values_read"] is False
    assert contract["future_covariate_matrices_used_for_training"] is False


def test_implementation_freezes_and_reports_integrity_evidence() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    freeze = FREEZE.read_text(encoding="utf-8")
    packager = PACKAGER.read_text(encoding="utf-8")
    assert "CovariateLinkedFactorAnalyticHierarchicalReactionNorm" in trainer
    assert "projection_design" in trainer
    assert "fa_environment_index" in trainer
    assert "enable_op_determinism" in trainer
    assert "assert_all_finite" in trainer
    assert "future_SSP_values_read" in trainer
    assert "factor_analytic_same_seed_replay.tsv" in runner
    assert "parent_private_head_terminal" in freeze
    assert "free_environment_loadings_forbidden" in freeze
    assert "EXPECTED_STATUSES" in packager
    assert "validation_predictions_calibrated.npy" in packager


def test_outer_and_final_outcomes_remain_sealed() -> None:
    value = protocol()
    assert value["outer_test_metrics_read"] is False
    assert value["outer_test_outcomes_read"] is False
    assert value["final_holdout_outcomes_read"] is False
    assert value["confirmation_policy"]["outer_evaluation_allowed"] is False
    assert value["product_policy"]["future_predictions_allowed"] is False
