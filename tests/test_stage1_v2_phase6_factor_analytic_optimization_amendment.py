from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.v2.run_stage1_v2_phase6_factor_analytic_optimization_amendment import (
    _primary_metric_summary,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_factor_analytic_optimization_amendment_protocol_v1.json"
)
PARENT = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_factor_analytic_screen_protocol_v1.json"
)
TRAINER = (
    ROOT
    / "server_training_pipeline"
    / "train_stage1_v2_phase6_factor_analytic_optimization_amendment_tf.py"
)
RUNNER = (
    ROOT
    / "scripts"
    / "v2"
    / "run_stage1_v2_phase6_factor_analytic_optimization_amendment.py"
)
FREEZE = (
    ROOT
    / "scripts"
    / "v2"
    / "freeze_stage1_v2_phase6_factor_analytic_optimization_amendment.py"
)
PACKAGER = (
    ROOT
    / "scripts"
    / "v2"
    / "package_stage1_v2_phase6_factor_analytic_optimization_amendment_results.py"
)


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_test_weight_is_demoted_only_from_the_primary_macro() -> None:
    objective = protocol()["objective_policy"]
    training = set(objective["training_likelihood_traits"])
    primary = set(objective["primary_macro_traits"])
    assert len(training) == 7
    assert len(primary) == 6
    assert training - primary == {"TEST_WEIGHT"}
    assert objective["demoted_from_primary_macro"] == ["TEST_WEIGHT"]
    for key in (
        "seven_trait_metrics_preserved",
        "all_seven_traits_retained_in_training_rows",
        "test_weight_predictions_retained",
        "test_weight_trait_reporting_retained",
        "test_weight_subset_reporting_retained",
        "test_weight_training_only_huber_calibration_retained",
        "test_weight_negative_slope_guard_retained",
        "test_weight_exploratory_non_deterioration_guard_retained",
    ):
        assert objective[key] is True


def test_amendment_preserves_all_non_objective_parent_contracts() -> None:
    value = protocol()
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    assert value["fixed_configuration"] == parent["fixed_configuration"]
    assert value["trait_specific_regularization"] == parent[
        "trait_specific_regularization"
    ]
    assert value["trial_environment_hierarchy"] == parent[
        "trial_environment_hierarchy"
    ]
    assert value["positive_training_calibration"] == parent[
        "positive_training_calibration"
    ]
    assert value["test_weight_environment_oof_calibration"] == parent[
        "test_weight_environment_oof_calibration"
    ]
    assert value["phase_1_acceptance"] == parent["phase_1_acceptance"]
    assert value["primary_traits"] == parent["primary_traits"]
    assert value["exploratory_traits"] == parent["exploratory_traits"]
    assert value["mandatory_reporting_subsets"] == parent[
        "mandatory_reporting_subsets"
    ]


def test_normalized_direction_amplitude_parameterization_is_exact() -> None:
    architecture = protocol()["architecture_policy"]
    assert architecture["only_mutable_component"] == (
        "covariate_linked_factor_analytic_optimization"
    )
    assert architecture[
        "genotype_and_environment_direction_columns_unit_normalized"
    ] is True
    assert architecture["trait_loadings_carry_factor_amplitude"] is True
    assert architecture["genotype_direction_L2_penalty"] is False
    assert architecture["environment_direction_L2_penalty"] is False
    assert architecture["trait_amplitude_loading_L2_penalty"] is True
    assert architecture["free_environment_loadings_allowed"] is False
    assert architecture["projection_inactive_policy"] == "exact_zero_FA_residual"
    candidates = {
        name: value
        for name, value in protocol()["candidates"].items()
        if value.get("source_reuse") is False
    }
    assert set(candidates) == {
        "normalized_direction_FA_rank2",
        "normalized_direction_FA_rank4",
    }
    assert {value["factor_analytic_rank"] for value in candidates.values()} == {
        2,
        4,
    }


def test_activity_certification_distinguishes_failure_from_valid_null() -> None:
    activity = protocol()["activity_certification"]
    assert activity["diagnostics_written_every_epoch"] is True
    assert activity["inactive_final_component_policy"] == (
        "valid_null_component_no_advance"
    )
    assert activity["optimization_path_failure_policy"] == (
        "fail_screen_integrity"
    )
    assert all(
        activity[key] > 0
        for key in (
            "minimum_raw_direction_column_norm",
            "maximum_normalized_direction_norm_error",
            "minimum_observed_gradient_norm_per_FA_tensor",
            "minimum_initial_training_residual_rms",
            "minimum_final_validation_residual_rms_for_performance_eligibility",
        )
    )


def test_implementation_persists_activity_and_seals_outer_data() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    freeze = FREEZE.read_text(encoding="utf-8")
    packager = PACKAGER.read_text(encoding="utf-8")
    assert "component_activity_history.tsv" in trainer
    assert 'record_type="initialization"' in trainer
    assert "primary_macro_nrmse" in trainer
    assert "Not every amended FA tensor received a gradient" in trainer
    assert "guard_FA_final_component_active" in runner
    assert "valid_inactive_component_do_not_advance" in runner
    assert "only_TEST_WEIGHT_demoted" in freeze
    assert "component_activity_history.tsv" in packager
    value = protocol()
    assert value["outer_test_metrics_read"] is False
    assert value["outer_test_outcomes_read"] is False
    assert value["final_holdout_outcomes_read"] is False
    assert value["future_predictions_allowed"] is False


def test_reference_and_candidates_use_the_same_six_trait_macro() -> None:
    primary = protocol()["objective_policy"]["primary_macro_traits"]
    traits = pd.DataFrame(
        {
            "trait_name_canonical": [*primary, "TEST_WEIGHT"],
            "normalized_rmse": [0.5] * 6 + [9.0],
            "pearson": [0.7] * 6 + [-0.5],
            "calibration_error": [0.1] * 6 + [4.0],
        }
    )
    metadata = {
        "validation_macro_normalized_rmse": float(
            traits["normalized_rmse"].mean()
        ),
        "validation_macro_pearson": float(traits["pearson"].mean()),
        "validation_macro_calibration_error": float(
            traits["calibration_error"].mean()
        ),
    }
    _primary_metric_summary(metadata, traits, primary)
    assert metadata["validation_macro_normalized_rmse"] == 0.5
    assert metadata["validation_macro_pearson"] == pytest.approx(0.7)
    assert metadata["validation_macro_calibration_error"] == pytest.approx(0.1)
    assert metadata["validation_all_seven_macro_normalized_rmse"] > 0.5
    assert metadata["TEST_WEIGHT_reporting_retained"] is True
