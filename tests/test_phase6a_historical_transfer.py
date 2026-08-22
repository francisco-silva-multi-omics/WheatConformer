from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from server_training_pipeline.certify_phase6a_historical_transfer import logical_dtype


def test_historical_transfer_contract_does_not_require_synchronized_weather() -> None:
    protocol = json.loads(
        Path("server_training_pipeline/phase6a_historical_transfer_contract_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["comparison_basis"].startswith("distributional_transfer")
    assert not protocol["synchronized_correlation_required"]
    assert not protocol["future_SSP_values_read"]
    assert not protocol["performance_thresholds_changed_from_v1"]


def test_nullable_and_dense_integer_counts_have_same_logical_dtype() -> None:
    assert logical_dtype(pd.Series([1, 2], dtype="int64"), "wet_day_count") == (
        "nullable_integer_count"
    )
    assert logical_dtype(pd.Series([1.0, None], dtype="float64"), "wet_day_count") == (
        "nullable_integer_count"
    )
