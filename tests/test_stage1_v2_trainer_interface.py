from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from server_training_pipeline.stage1_v2_trainer_interface import (
    load_selection_protocol,
    load_environment_identity_axis,
    load_state_spec,
    normalize_cycle_year,
    state_role_masks,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("STAGE1_V2_DATA_ROOT", ROOT)).resolve()


def test_cycle_year_normalization_matches_phase5_contract() -> None:
    assert normalize_cycle_year("79-80") == "1980"
    assert normalize_cycle_year("00-01") == "2001"
    assert normalize_cycle_year("2022") == "2022"


def test_temporal_state_roles_match_normalized_year_assignments() -> None:
    observations = pd.DataFrame(
        {
            "canonical_gid": ["g1", "g2", "g3"],
            "environment_id": ["e1", "e2", "e3"],
            "year": ["79-80", "80-81", "81-82"],
            "temporal_year_outer1_role": ["TRAIN"] * 3,
        }
    )
    assignments = pd.DataFrame(
        {
            "scenario": ["TEMPORAL_YEAR"] * 3,
            "outer_fold": ["1"] * 3,
            "inner_fold": ["1"] * 3,
            "entity_type": ["NORMALIZED_YEAR"] * 3,
            "entity_id": ["1980", "1981", "1982"],
            "assignment": [
                "TRAIN",
                "EMBARGO_ONE_YEAR",
                "INNER_VALIDATION_ID_ONLY",
            ],
        }
    )
    training, validation, embargo = state_role_masks(
        observations,
        scenario="TEMPORAL_YEAR",
        outer_fold=1,
        inner_fold=1,
        training_gids={"g1"},
        training_environments={"e1"},
        assignments=assignments,
    )
    assert training.tolist() == [True, False, False]
    assert validation.tolist() == [False, False, True]
    assert embargo.tolist() == [False, True, False]


def test_gobs_enew_validation_excludes_unseen_genotypes() -> None:
    observations = pd.DataFrame(
        {
            "canonical_gid": ["g1", "g1", "g2"],
            "environment_id": ["e1", "e2", "e2"],
            "gobs_enew_outer1_role": ["TRAIN"] * 3,
        }
    )
    training, validation, embargo = state_role_masks(
        observations,
        scenario="GOBS_ENEW",
        outer_fold=1,
        inner_fold=1,
        training_gids={"g1"},
        training_environments={"e1"},
    )
    assert training.tolist() == [True, False, False]
    assert validation.tolist() == [False, True, False]
    assert embargo.tolist() == [False, False, True]


def test_confirmation_execution_correction_freezes_parity_axis_and_legacy_scope() -> None:
    correction = json.loads(
        (
            ROOT
            / "server_training_pipeline/"
            "stage1_v2_phase6_confirmation_execution_correction_v4.json"
        ).read_text(encoding="utf-8")
    )
    assert correction["scientific_candidate_protocol_unchanged"] is True
    assert correction["frozen_split_artifacts_unchanged"] is True
    assert correction["execution_requirements"][
        "prewarm_all_375_candidate_factor_bindings_before_tensorflow"
    ] is True
    assert set(correction["legacy_run_compatibility"]["allowed_scenarios"]) == {
        "GNEW_EOBS",
        "GOBS_ENEW",
        "GNEW_ENEW",
    }
    assert correction["legacy_run_compatibility"][
        "temporal_or_country_legacy_reuse_allowed"
    ] is False
    assert correction["outer_test_outcomes_read"] is False


def test_temporal_identity_axis_uses_parity_training_partition() -> None:
    axis = load_environment_identity_axis(
        DATA_ROOT, "TEMPORAL_YEAR__OUTER1__INNER1"
    )
    assert len(axis) == 11161
    assert axis["environment_id"].is_unique
    assert int(axis["partition"].eq("TRAINING").sum()) == 338
    assert int(axis["geo_level_index"].ge(0).sum()) >= 338


def test_original_identity_axis_remains_exactly_bound_to_phase5() -> None:
    state_id = "GNEW_EOBS__OUTER1__INNER1"
    registry = pd.read_csv(
        DATA_ROOT
        / "audit/v2/phase5_split_bound_kernel_validation_v2/"
        "environment/ke_registry.tsv",
        sep="\t",
        dtype=str,
    )
    row = registry.loc[
        registry["state_id"].eq(state_id)
        & registry["component"].eq("K_E_identity")
    ].iloc[0]
    expected = pd.read_csv(
        DATA_ROOT
        / "audit/v2/phase5_split_bound_kernel_validation_v2"
        / str(row["entity_order_path"]),
        sep="\t",
        dtype=str,
    )
    observed = load_environment_identity_axis(DATA_ROOT, state_id)
    pd.testing.assert_frame_equal(observed, expected)


def test_temporal_stage_absence_is_an_explicit_22_state_mask() -> None:
    parity = DATA_ROOT / (
        "audit/v2/phase5_panel_environment_scenario_parity_extension_v2"
    )
    states = pd.read_csv(parity / "splits/state_registry.tsv", sep="\t", dtype=str)
    registry = pd.read_csv(
        parity / "environment/environment_component_registry.tsv",
        sep="\t",
        dtype=str,
    )
    temporal_inner = set(
        states.loc[
            states["scenario"].eq("TEMPORAL_YEAR")
            & states["state_level"].eq("INNER"),
            "state_id",
        ]
    )
    stage_available = set(
        registry.loc[
            registry["state_id"].isin(temporal_inner)
            & registry["component"].str.startswith("K_E_STAGE_")
            & registry["component_available"].str.lower().eq("true"),
            "state_id",
        ]
    )
    zero_stage = temporal_inner - stage_available
    assert len(zero_stage) == 22
    main = registry.loc[
        registry["state_id"].isin(zero_stage)
        & registry["component"].isin(
            {"K_E_MANAGEMENT", "K_E_STRESS", "K_E_WEATHER"}
        )
        & registry["component_available"].str.lower().eq("true")
    ]
    assert main.groupby("state_id")["component"].nunique().eq(3).all()


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
        DATA_ROOT,
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
        DATA_ROOT,
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


def test_server_cpu_runtime_is_frozen_without_gpu_requirement() -> None:
    runtime = json.loads(
        (
            ROOT
            / "server_training_pipeline/stage1_v2_phase6_server_cpu_runtime_v1.json"
        ).read_text()
    )
    assert runtime["python_major_minor"] == "3.11"
    assert runtime["tensorflow"] == "2.15.1"
    assert runtime["pandas"] == "2.2.3"
    assert runtime["tensorflow_gpu_required_for_training"] is False
    assert runtime["default_parallel_workers_for_20_physical_cores"] == 4


def test_execution_amendment_recomputes_every_run_without_scientific_change() -> None:
    protocol = json.loads(
        (
            ROOT
            / "server_training_pipeline/stage1_v2_phase6_execution_protocol_v2.json"
        ).read_text()
    )
    assert protocol["scientific_selection_protocol_unchanged"] is True
    assert protocol["all_120_runs_must_be_recomputed"] is True
    assert protocol["old_run_reuse_allowed"] is False
    assert protocol["prior_partial_inner_metrics_used_for_execution_change"] is False
    assert protocol["outer_test_outcomes_read"] is False
