from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DISPOSITION = (
    ROOT
    / "server_training_pipeline"
    / "cimmyt_pre_qc_model_disposition_v1.json"
)


def test_cimmyt_quantitative_kg_is_not_advanced() -> None:
    protocol = json.loads(DISPOSITION.read_text(encoding="utf-8"))
    assert protocol["qc_certification_preserved"] is True
    assert protocol["prior_frozen_inner_experiments_preserved"] is True
    assert protocol["quantitative_K_G"]["advance_allowed"] is False
    assert (
        protocol["quantitative_K_G"]["may_enter_new_outer_or_final_holdout_evaluation"]
        is False
    )
    assert protocol["quantitative_K_G"]["may_be_merged_with_other_marker_panels"] is False


def test_cimmyt_kz_is_observed_window_only() -> None:
    protocol = json.loads(DISPOSITION.read_text(encoding="utf-8"))
    kz = protocol["regulatory_K_z"]
    assert kz["dense_imputed_calls_allowed"] is False
    assert kz["training_mean_imputed_calls_allowed"] is False
    assert kz["observed_call_only_window_coverage_audit_allowed"] is True
    assert kz["current_production_ready"] is False
    assert "regulatory_window_overlap_certified" in kz["required_before_any_panel_specific_K_z"]
    assert "no_imputation_across_unsupported_regulatory_loci" in kz[
        "required_before_any_panel_specific_K_z"
    ]


def test_multi_panel_kz_preserves_panel_specific_evidence() -> None:
    protocol = json.loads(DISPOSITION.read_text(encoding="utf-8"))
    policy = protocol["multi_panel_K_z_policy"]
    assert policy["combine_at_biological_feature_or_kernel_level_only"] is True
    assert policy["concatenate_marker_matrices_across_platforms"] is False
    assert policy["retain_panel_specific_confidence_and_missingness_masks"] is True
    assert policy["K_A_is_fallback_not_K_z_imputation"] is True
    assert policy["priority_order"][0] == "seeds_of_discovery_dartseq_coordinate_recovery"
