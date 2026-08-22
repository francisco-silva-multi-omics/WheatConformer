from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from server_training_pipeline.build_phase6a_bias_adjusted_cmip6_backcast import apply_frozen_bias


def test_additive_temperature_bias_uses_frozen_historical_quantiles() -> None:
    protocol = json.loads(
        Path("server_training_pipeline/phase6a_bias_adjustment_contract_v2.json").read_text(
            encoding="utf-8"
        )
    )
    quantiles = np.asarray(protocol["quantile_probabilities"])
    shape = (1, 12, 1, len(quantiles))
    model = np.broadcast_to(np.linspace(0, 20, len(quantiles)), shape).copy()
    reference = model + 2
    parameters = xr.Dataset(
        {
            "model_historical_quantile": (("site", "month", "variable", "quantile"), model),
            "reference_historical_quantile": (("site", "month", "variable", "quantile"), reference),
            "parameter_eligible": (("site", "month", "variable"), np.ones(shape[:-1], dtype=np.int8)),
            "parameter_method_code": (
                ("site", "month", "variable"),
                np.ones(shape[:-1], dtype=np.int8),
            ),
            "model_wet_threshold_mm": (("site", "month"), np.zeros((1, 12))),
        },
        coords={
            "site": [0],
            "month": np.arange(1, 13),
            "variable": ["tasmean_c"],
            "quantile": quantiles,
        },
    )
    adjusted, eligible = apply_frozen_bias(
        np.asarray([5.0, 10.0]),
        np.asarray([1, 1]),
        "tasmean_c",
        0,
        parameters,
        protocol,
    )
    assert eligible.all()
    assert np.allclose(adjusted, [7.0, 12.0])


def test_dry_month_fallback_uses_frozen_threshold_and_multiplier() -> None:
    protocol = json.loads(
        Path("server_training_pipeline/phase6a_bias_adjustment_contract_v2.json").read_text(
            encoding="utf-8"
        )
    )
    quantiles = np.asarray(protocol["quantile_probabilities"])
    shape = (1, 12, 1, len(quantiles))
    parameters = xr.Dataset(
        {
            "model_historical_quantile": (
                ("site", "month", "variable", "quantile"),
                np.full(shape, np.nan),
            ),
            "reference_historical_quantile": (
                ("site", "month", "variable", "quantile"),
                np.full(shape, np.nan),
            ),
            "parameter_eligible": (
                ("site", "month", "variable"),
                np.ones(shape[:-1], dtype=np.int8),
            ),
            "parameter_method_code": (
                ("site", "month", "variable"),
                np.full(shape[:-1], 2, dtype=np.int8),
            ),
            "model_wet_threshold_mm": (("site", "month"), np.full((1, 12), 1.0)),
            "precipitation_fallback_multiplier": (
                ("site", "month"),
                np.full((1, 12), 0.5),
            ),
        },
        coords={
            "site": [0],
            "month": np.arange(1, 13),
            "variable": ["precipitation_mm_day"],
            "quantile": quantiles,
        },
    )
    adjusted, eligible = apply_frozen_bias(
        np.asarray([0.0, 0.5, 2.0, 4.0]),
        np.asarray([1, 1, 1, 1]),
        "precipitation_mm_day",
        0,
        parameters,
        protocol,
    )
    assert eligible.all()
    assert np.allclose(adjusted, [0.0, 0.0, 1.0, 2.0])
