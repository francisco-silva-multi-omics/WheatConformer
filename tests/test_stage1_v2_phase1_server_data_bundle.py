from __future__ import annotations

from scripts.v2.package_stage1_v2_phase6_phase1_server_data import PAYLOAD_PATHS


def test_phase1_server_bundle_includes_exact_projection_screen_states() -> None:
    values = {path.as_posix() for path in PAYLOAD_PATHS}
    projection = {
        value
        for value in values
        if value.startswith(
            "environment/v2/e_projection_core_v1_split_bound_historical_v1/states/"
        )
    }
    assert projection == {
        "environment/v2/e_projection_core_v1_split_bound_historical_v1/"
        f"states/GNEW_EOBS__OUTER1__INNER{fold}"
        for fold in range(1, 6)
    }


def test_phase1_server_bundle_contains_training_inputs_and_no_future_matrix() -> None:
    values = {path.as_posix() for path in PAYLOAD_PATHS}
    assert any("corrected_promoted_phenotypes.parquet" in value for value in values)
    assert any("phase5_split_bound_kernel_validation_v2/splits" in value for value in values)
    assert any("phase5_cimmyt_pre_qc_split_local_v1/genomic" in value for value in values)
    assert any("phase6_h_seeds_operator_v1" in value for value in values)
    assert all("future_covariates_v1/states" not in value for value in values)
    assert all("final_holdout" not in value.lower() for value in values)
