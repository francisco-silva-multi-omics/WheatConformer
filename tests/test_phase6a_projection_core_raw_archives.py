from __future__ import annotations

import cftime
import io
import numpy as np
import pandas as pd
import zipfile

from server_training_pipeline.certify_phase6a_projection_core_raw_archives import (
    cds_component_frames,
    native_daily_axis,
    numeric_daily_axis,
)


def test_native_daily_axis_accepts_365_day_calendar_without_february_29() -> None:
    values = np.asarray(
        [
            cftime.DatetimeNoLeap(2000, 2, 28),
            cftime.DatetimeNoLeap(2000, 3, 1),
            cftime.DatetimeNoLeap(2000, 3, 2),
        ],
        dtype=object,
    )
    unique, daily, expected = native_daily_axis(values, "365_day")
    assert unique
    assert daily
    assert expected == 3


def test_native_daily_axis_rejects_duplicate() -> None:
    values = np.asarray(
        [cftime.DatetimeGregorian(2000, 1, 1), cftime.DatetimeGregorian(2000, 1, 1)],
        dtype=object,
    )
    unique, daily, expected = native_daily_axis(values, "gregorian")
    assert not unique
    assert not daily
    assert expected == 1


def test_native_daily_axis_rejects_gap() -> None:
    values = np.asarray(
        [cftime.DatetimeGregorian(2000, 1, 1), cftime.DatetimeGregorian(2000, 1, 3)],
        dtype=object,
    )
    unique, daily, expected = native_daily_axis(values, "gregorian")
    assert unique
    assert not daily
    assert expected == 3


def test_numeric_daily_axis_preserves_native_noleap_calendar() -> None:
    first, last, unique, daily, expected = numeric_daily_axis(
        np.asarray([58.0, 59.0, 60.0]),
        "days since 2001-01-01 00:00:00",
        "365_day",
    )
    assert (first, last) == ("2001-02-28", "2001-03-02")
    assert unique
    assert daily
    assert expected == 3


def _cds_part(start: str) -> bytes:
    timestamp = pd.Timestamp(start)
    common = {
        "valid_time": [timestamp],
        "latitude": [10.0],
        "longitude": [20.0],
    }
    members = {
        "wind.csv": pd.DataFrame({**common, "u10": [1.0], "v10": [2.0]}),
        "temperature.csv": pd.DataFrame({**common, "d2m": [280.0], "t2m": [285.0]}),
        "radiation.csv": pd.DataFrame({**common, "ssrd": [100.0]}),
        "precipitation.csv": pd.DataFrame({**common, "sp": [100000.0], "tp": [0.001]}),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, frame in members.items():
            archive.writestr(name, frame.to_csv(index=False))
    return output.getvalue()


def test_partitioned_cds_bundle_is_recursed_without_losing_components() -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr("parts/000_a.zip", _cds_part("2000-01-01"))
        archive.writestr("parts/001_b.zip", _cds_part("2000-01-02"))
    components = cds_component_frames(output.getvalue())
    assert set(components) == {"wind", "temperature", "radiation", "precipitation"}
    assert all(len(frames) == 2 for frames in components.values())
