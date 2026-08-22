from __future__ import annotations

import json
from pathlib import Path

from server_training_pipeline.stage1_v2_trainer_interface import (
    load_selection_protocol,
    load_state_spec,
)


ROOT = Path(__file__).resolve().parents[1]


def test_phase6_selection_protocol_freezes_metrics_guards_and_subsets() -> None:
    protocol = load_selection_protocol(ROOT)
    assert protocol["selection_metrics"]["primary"] == "macro_trait_scenario_normalized_rmse"
    assert protocol["selection_metrics"]["minimum_relative_nrmse_gain"] == 0.01
    assert protocol["guards"]["inactive_component_rows_must_be_reported"] is True
    subsets = set(protocol["mandatory_reporting_subsets"])
    assert "PEDIGREE_ONLY" in subsets
    assert "MARKER_SUPPORTED" in subsets
    assert "NEITHER_PRODUCTION_PEDIGREE_NOR_DENSE_MARKERS" in subsets
    assert "RECOVERED_IDENTITY_OR_COMPONENT" in subsets
    assert "PROJECTION_CORE_INACTIVE_814_ENVIRONMENTS" in subsets
    assert protocol["addition_of_unregistered_candidates_after_metric_access_allowed"] is False
    schedule = protocol["screen_schedule"]
    assert schedule["phase_1_scenario"] == "GNEW_EOBS"
    assert schedule["phase_1_outer_fold"] == 1
    assert schedule["phase_1_inner_folds"] == [1, 2, 3, 4, 5]


def test_v2_interface_preflights_one_inner_projection_state_without_outcomes() -> None:
    spec = load_state_spec(
        ROOT,
        "GNEW_EOBS__OUTER1__INNER1",
        "ka_projection_core",
    )
    assert spec.state_level == "INNER"
    assert spec.training_observation_count > 0
    assert spec.validation_observation_count > 0
    assert spec.pedigree.entity_count == 8762
    assert spec.pedigree.factor_nonzero_count == 423695
    assert spec.projection_environment is not None
    assert spec.projection_environment.feature_count == 153
    assert spec.projection_environment.factor_rank == 64
    assert spec.projection_environment.inactive_environment_count == 814
    assert spec.phenotype_values_read is False
    assert spec.outer_test_outcomes_read is False
    assert spec.final_holdout_outcomes_read is False


def test_h_seeds_interface_preserves_ka_when_temporal_support_is_masked() -> None:
    spec = load_state_spec(
        ROOT,
        "TEMPORAL_YEAR__OUTER1__INNER1",
        "h_seeds_projection_core",
    )
    assert spec.h_seeds is not None
    assert spec.h_seeds.component_available is False
    assert spec.h_seeds.training_overlap_count == 4
    assert spec.h_seeds.absence_mask == (
        "SEEDS_TRAINING_PEDIGREE_OVERLAP_LT20_KA_BACKBONE_RETAINED"
    )
    assert spec.pedigree.entity_count == 8762


def test_training_runtime_is_the_certified_wsl_environment() -> None:
    runtime = json.loads(
        (ROOT / "server_training_pipeline/stage1_v2_training_runtime_v1.json").read_text()
    )
    assert runtime["python"] == "3.11.15"
    assert runtime["tensorflow"] == "2.15.1"
    assert runtime["pandas"] == "2.2.3"
    assert runtime["windows_audit_venv_is_training_runtime"] is False
