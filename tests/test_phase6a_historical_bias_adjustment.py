from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from server_training_pipeline.fit_phase6a_historical_bias_adjustment import fit_pair, transform


def protocol() -> dict:
    return json.loads(
        Path("server_training_pipeline/phase6a_bias_adjustment_contract_v2.json").read_text(
            encoding="utf-8"
        )
    )


def test_temperature_pair_stores_historical_quantiles_only() -> None:
    values = np.linspace(0, 30, 900)
    result = fit_pair(values, values + 2, "tasmean_c", np.asarray([0.1, 0.5, 0.9]), protocol())
    assert result["eligible"]
    assert np.allclose(result["reference_quantiles"] - result["model_quantiles"], 2)


def test_precipitation_pair_freezes_wet_day_threshold() -> None:
    model = np.r_[np.zeros(400), np.linspace(0.2, 20, 500)]
    reference = np.r_[np.zeros(450), np.linspace(0.2, 25, 450)]
    result = fit_pair(
        model,
        reference,
        "precipitation_mm_day",
        np.asarray([0.1, 0.5, 0.9]),
        protocol(),
    )
    assert result["eligible"]
    assert np.isfinite(result["model_wet_threshold"])
    assert result["reference_wet_fraction"] == 0.5


def test_humidity_transform_is_bounded_and_finite() -> None:
    values = transform(np.asarray([0.0, 50.0, 100.0]), "relative_humidity_percent", protocol())
    assert np.isfinite(values).all()
    assert values[0] < values[1] < values[2]


def test_dry_month_uses_explicit_intensity_fallback_without_lowering_quantile_support() -> None:
    model = np.r_[np.zeros(850), np.linspace(0.2, 8.0, 50)]
    reference = np.r_[np.zeros(870), np.linspace(0.2, 6.0, 30)]
    result = fit_pair(
        model,
        reference,
        "precipitation_mm_day",
        np.asarray([0.1, 0.5, 0.9]),
        protocol(),
    )
    assert result["eligible"]
    assert result["method_code"] == 2
    assert np.isfinite(result["precipitation_fallback_multiplier"])
    assert np.isnan(result["model_quantiles"]).all()
