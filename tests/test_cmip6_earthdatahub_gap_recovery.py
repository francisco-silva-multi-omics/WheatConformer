from __future__ import annotations

import numpy as np
import xarray as xr

from server_training_pipeline.recover_cmip6_earthdatahub_gap import (
    concordance_checks,
    concordance_metrics,
    select_experiment,
)


def test_select_experiment_uses_unique_scenario_coordinate() -> None:
    dataset = xr.Dataset(
        {"rsds": (("experiment_id", "time"), np.arange(6).reshape(2, 3))},
        coords={"experiment_id": ["ssp126", "ssp585"], "time": [0, 1, 2]},
    )
    selected = select_experiment(dataset, "ssp126")
    assert selected["rsds"].values.tolist() == [0, 1, 2]


def test_concordance_metrics_and_checks_accept_small_quantization() -> None:
    reference = np.linspace(0.0, 300.0, 10000).reshape(100, 100)
    candidate = reference + 0.001
    metrics = concordance_metrics(reference, candidate)
    protocol = {
        "exact_overlap": {
            "expected_sites": 100,
            "expected_days": 100,
            "minimum_compared_values": 10000,
        },
        "acceptance_thresholds": {
            "maximum_source_latitude_delta_degrees": 1e-6,
            "maximum_source_longitude_delta_degrees": 1e-6,
            "maximum_missing_mask_disagreement_fraction": 0.0,
            "maximum_absolute_bias_w_m2": 0.01,
            "maximum_rmse_w_m2": 0.05,
            "maximum_p99_absolute_delta_w_m2": 0.1,
            "maximum_absolute_delta_w_m2": 0.5,
            "minimum_pearson_correlation": 0.999999,
        },
    }
    checks = concordance_checks(metrics, protocol, 100, 100, 0.0, 0.0)
    assert all(checks.values())


def test_concordance_checks_reject_material_provider_change() -> None:
    reference = np.linspace(0.0, 300.0, 10000).reshape(100, 100)
    metrics = concordance_metrics(reference, reference + 1.0)
    protocol = {
        "exact_overlap": {
            "expected_sites": 100,
            "expected_days": 100,
            "minimum_compared_values": 10000,
        },
        "acceptance_thresholds": {
            "maximum_source_latitude_delta_degrees": 1e-6,
            "maximum_source_longitude_delta_degrees": 1e-6,
            "maximum_missing_mask_disagreement_fraction": 0.0,
            "maximum_absolute_bias_w_m2": 0.01,
            "maximum_rmse_w_m2": 0.05,
            "maximum_p99_absolute_delta_w_m2": 0.1,
            "maximum_absolute_delta_w_m2": 0.5,
            "minimum_pearson_correlation": 0.999999,
        },
    }
    checks = concordance_checks(metrics, protocol, 100, 100, 0.0, 0.0)
    assert not checks["absolute_bias"]
    assert not checks["rmse"]
    assert not checks["maximum_absolute_delta"]
