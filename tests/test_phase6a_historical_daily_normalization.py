from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from server_training_pipeline.normalize_phase6a_historical_daily import (
    normalize_cds_bytes,
    saturation_vapor_pressure_kpa,
)


def test_saturation_vapor_pressure_increases_with_temperature() -> None:
    values = saturation_vapor_pressure_kpa(np.asarray([0.0, 10.0, 20.0]))
    assert np.all(np.diff(values) > 0)


def test_cds_daily_normalization_uses_hourly_sums_and_physical_units() -> None:
    times = pd.date_range("2000-01-01", periods=24, freq="h")
    common = {"valid_time": times, "latitude": 10.0, "longitude": 20.0}
    members = {
        "wind.csv": pd.DataFrame({**common, "u10": 3.0, "v10": 4.0}),
        "temperature.csv": pd.DataFrame({**common, "d2m": 278.15, "t2m": 283.15}),
        "radiation.csv": pd.DataFrame({**common, "ssrd": 1_000_000.0}),
        "precipitation.csv": pd.DataFrame({**common, "sp": 100_000.0, "tp": 0.001}),
    }
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name, frame in members.items():
            archive.writestr(name, frame.to_csv(index=False))
    protocol = {
        "tiny_negative_tolerances": {
            "CDS_tp_m_per_hour": -0.000001,
            "CDS_ssrd_J_m2_per_hour": -10.0,
        },
        "physical_domains": {
            "tasmin_c": [-90, 60],
            "tasmean_c": [-90, 60],
            "tasmax_c": [-90, 70],
            "precipitation_mm_day": [0, 1000],
            "solar_radiation_mj_m2_day": [0, 60],
            "relative_humidity_percent": [0, 100],
            "wind_speed_m_s": [0, 100],
            "surface_pressure_pa": [30000, 110000],
        },
    }
    result = normalize_cds_bytes(payload.getvalue(), "request", protocol)
    assert len(result) == 1
    row = result.iloc[0]
    assert np.isclose(row.tasmean_c, 10.0)
    assert np.isclose(row.precipitation_mm_day, 24.0)
    assert np.isclose(row.solar_radiation_mj_m2_day, 24.0)
    assert np.isclose(row.wind_speed_m_s, 5.0)
    assert 0 < row.relative_humidity_percent < 100
    assert bool(row.required_climate_complete)


def test_cds_daily_normalization_preserves_missing_inputs_with_masks() -> None:
    times = pd.date_range("2000-01-01", periods=24, freq="h")
    common = {"valid_time": times, "latitude": 10.0, "longitude": 20.0}
    members = {
        "wind.csv": pd.DataFrame({**common, "u10": np.nan, "v10": np.nan}),
        "temperature.csv": pd.DataFrame({**common, "d2m": np.nan, "t2m": np.nan}),
        "radiation.csv": pd.DataFrame({**common, "ssrd": 1_000_000.0}),
        "precipitation.csv": pd.DataFrame({**common, "sp": np.nan, "tp": 0.001}),
    }
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name, frame in members.items():
            archive.writestr(name, frame.to_csv(index=False))
    protocol = {
        "tiny_negative_tolerances": {
            "CDS_tp_m_per_hour": -0.000001,
            "CDS_ssrd_J_m2_per_hour": -10.0,
        },
        "physical_domains": {
            "tasmin_c": [-90, 60],
            "tasmean_c": [-90, 60],
            "tasmax_c": [-90, 70],
            "precipitation_mm_day": [0, 1000],
            "solar_radiation_mj_m2_day": [0, 60],
            "relative_humidity_percent": [0, 100],
            "wind_speed_m_s": [0, 100],
            "surface_pressure_pa": [30000, 110000],
        },
    }
    row = normalize_cds_bytes(payload.getvalue(), "request", protocol).iloc[0]
    assert np.isnan(row.tasmean_c)
    assert not bool(row.tasmean_c_available)
    assert not bool(row.required_climate_complete)
    assert np.isclose(row.precipitation_mm_day, 24.0)


def test_cmip6_radiation_tolerance_covers_only_quantization_noise() -> None:
    protocol = json.loads(
        Path("server_training_pipeline/phase6a_daily_normalization_contract_v1.json").read_text(
            encoding="utf-8"
        )
    )
    tolerance = protocol["tiny_negative_tolerances"]["CMIP6_rsds_W_m2"]
    assert tolerance <= -0.00390625
    assert tolerance > -0.1
    assert "preserve_unmodified_raw_values" in protocol["tiny_negative_policy"]
