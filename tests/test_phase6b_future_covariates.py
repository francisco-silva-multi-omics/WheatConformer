from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from server_training_pipeline.build_phase6b_member_resolved_future_covariates import (
    build_season_indices,
    canonicalize,
    resolve_anchor_key,
)
from server_training_pipeline.certify_phase6b_member_resolved_future_covariates import (
    build_gcm_agreement,
    classify,
)
from server_training_pipeline.freeze_phase6b_future_covariates import select_sowing_medoid


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_keeps_predictions_blocked_and_exact_grid() -> None:
    protocol = json.loads(
        (ROOT / "server_training_pipeline/phase6b_future_covariate_protocol_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["expected_matrix_count"] == 13 * 4 * 2
    assert protocol["expected_location_count"] == 907
    assert protocol["member_dimension_must_remain_resolved"] is True
    assert protocol["future_prediction_allowed"] is False
    assert protocol["future_predictions_generated"] == 0


def test_sowing_medoid_is_circular_and_supports_february_29() -> None:
    selected, support, dispersion = select_sowing_medoid(
        pd.Series(["2000-12-30", "2001-01-02", "2002-01-02"])
    )
    assert selected == "01-02"
    assert support == 3
    assert dispersion >= 0
    leap_selected, leap_support, _ = select_sowing_medoid(pd.Series(["2000-02-29"]))
    assert leap_selected == "02-29"
    assert leap_support == 1


def test_native_calendar_february_29_fallback() -> None:
    lookup = {"2032-02-29": 10, "2033-02-28": 20}
    assert resolve_anchor_key(2032, "02-29", lookup) == "2032-02-29"
    assert resolve_anchor_key(2033, "02-29", lookup) == "2033-02-28"


def test_season_index_requires_complete_native_window() -> None:
    dates = pd.date_range("2030-01-01", "2032-12-31", freq="D")
    keys = dates.strftime("%Y-%m-%d").to_numpy()
    locations = pd.DataFrame(
        {
            "prospective_sowing_month_day": ["05-01", ""],
        }
    )
    indices, valid = build_season_indices(keys, locations, [2031])
    assert indices.shape == (1, 210, 2)
    assert valid.tolist() == [[True, False]]
    assert np.all(np.diff(indices[0, :, 0]) == 1)


def test_canonical_units_and_terminal_ood_classes() -> None:
    np.testing.assert_allclose(canonicalize(np.array([273.15]), "tas"), [0.0])
    np.testing.assert_allclose(canonicalize(np.array([1.0]), "pr"), [86400.0])
    np.testing.assert_allclose(canonicalize(np.array([1.0]), "rsds"), [0.0864])
    classes = classify(
        required_missing=np.array([False, False, True]),
        range_fraction=np.array([0.01, 0.1, 0.0]),
        robust_rms=np.array([1.0, 6.0, 0.0]),
        mahalanobis=np.array([2.0, 10.0, 0.0]),
        thresholds={"mahalanobis_99": 5.0, "mahalanobis_999": 10.0},
    )
    assert classes.tolist() == ["SUPPORTED", "EXTRAPOLATIVE", "UNSUPPORTED"]


def test_gcm_agreement_accepts_boolean_projection_features(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "SSP": ["SSP1-2.6"] * 13,
            "period": ["2031_2060"] * 13,
            "location_id": ["site"] * 13,
            "source_id": [f"source_{value:02d}" for value in range(13)],
            "soil_feature_eligible": [True] * 7 + [False] * 6,
        }
    )
    reference = pd.DataFrame(
        {
            "feature": ["soil_feature_eligible"],
            "median": [1.0],
        }
    )
    path, rows = build_gcm_agreement(frame, reference, tmp_path)
    result = pd.read_parquet(path)
    assert rows == 1
    assert result.source_count.iloc[0] == 13
    assert result.q25.iloc[0] == 0.0
