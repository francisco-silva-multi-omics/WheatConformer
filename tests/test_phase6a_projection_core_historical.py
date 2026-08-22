from __future__ import annotations

import numpy as np
import pandas as pd

from server_training_pipeline.build_phase6a_projection_core_historical import (
    available_water_capacity_mm,
    derive_feature_row,
    fao56_et0,
)


def test_available_water_capacity_is_depth_weighted_and_coarse_corrected() -> None:
    rows = []
    for depth in ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"):
        rows.extend(
            [
                {"depth": depth, "property": "wv0033", "canonical_value": 30.0},
                {"depth": depth, "property": "wv1500", "canonical_value": 10.0},
                {"depth": depth, "property": "cfvo", "canonical_value": 0.0},
            ]
        )
    assert np.isclose(available_water_capacity_mm(pd.DataFrame(rows)), 200.0)


def test_fao56_et0_is_positive_for_complete_warm_weather() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=3),
            "tasmin_c": 10.0,
            "tasmean_c": 20.0,
            "tasmax_c": 30.0,
            "solar_radiation_mj_m2_day": 18.0,
            "relative_humidity_percent": 55.0,
            "wind_speed_m_s": 2.0,
            "surface_pressure_pa": 100000.0,
        }
    )
    result = fao56_et0(frame, 30.0)
    assert np.isfinite(result).all()
    assert (result > 0).all()


def test_feature_builder_uses_fixed_sowing_windows_only() -> None:
    dates = pd.date_range("1999-12-02", periods=210)
    daily = pd.DataFrame(
        {
            "date": dates,
            "tasmin_c": 10.0,
            "tasmean_c": 20.0,
            "tasmax_c": 30.0,
            "precipitation_mm_day": 1.0,
            "solar_radiation_mj_m2_day": 18.0,
            "relative_humidity_percent": 55.0,
            "wind_speed_m_s": 2.0,
            "surface_pressure_pa": 100000.0,
            "required_climate_complete": True,
        }
    )
    windows = {"antecedent_30": [-30, -1], "w00_29": [0, 29]}
    row = derive_feature_row(
        daily,
        {
            "environment_id": "E1",
            "cds_request_id": "R1",
            "latitude": 30.0,
            "longitude": -110.0,
            "elevation_m": 500.0,
            "sowing_date": "2000-01-01",
        },
        {
            "soil_status": "EXACT_SOIL_CELL",
            "soil_source_class": "exact",
            "soil_feature_eligible": True,
            "soil_missing_mask": False,
            "available_water_capacity_mm": 200.0,
        },
        windows,
    )
    assert row["all_fixed_windows_complete"]
    assert row["w00_29__day_count"] == 30
    assert row["w00_29__precipitation_sum_mm"] == 30
    assert not row["water_balance_enabled"]
