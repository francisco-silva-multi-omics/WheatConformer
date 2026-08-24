from __future__ import annotations

import numpy as np
import pytest

from server_training_pipeline.repair_cmip6_mri_historical_pr import assert_daily_axis


def test_daily_axis_accepts_complete_gregorian_period() -> None:
    values = np.asarray(["2000-01-01", "2000-01-02", "2000-01-03"])
    assert_daily_axis(values, "2000-01-01", "2000-01-03", 3)


def test_daily_axis_rejects_gap() -> None:
    values = np.asarray(["2000-01-01", "2000-01-03"])
    with pytest.raises(ValueError, match="gaps"):
        assert_daily_axis(values, "2000-01-01", "2000-01-03", 2)


def test_daily_axis_rejects_duplicates() -> None:
    values = np.asarray(["2000-01-01", "2000-01-01"])
    with pytest.raises(ValueError, match="duplicate"):
        assert_daily_axis(values, "2000-01-01", "2000-01-01", 2)
