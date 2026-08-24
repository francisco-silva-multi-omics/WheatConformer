from __future__ import annotations

import numpy as np

from server_training_pipeline.audit_phase6a_cross_provider import (
    paired_sufficient_statistics,
    statistics_to_metrics,
)


def test_sufficient_statistics_reconstruct_bias_rmse_and_correlation() -> None:
    left = np.asarray([1.0, 2.0, 3.0])
    right = np.asarray([2.0, 3.0, 4.0])
    metrics = statistics_to_metrics(paired_sufficient_statistics(left, right))
    assert metrics["n"] == 3
    assert np.isclose(metrics["bias"], 1.0)
    assert np.isclose(metrics["rmse"], 1.0)
    assert np.isclose(metrics["mae"], 1.0)
    assert np.isclose(metrics["pearson"], 1.0)


def test_sufficient_statistics_record_missing_mask_disagreement() -> None:
    stats = paired_sufficient_statistics(
        np.asarray([1.0, np.nan, 3.0]), np.asarray([1.0, 2.0, np.nan])
    )
    metrics = statistics_to_metrics(stats)
    assert metrics["n"] == 1
    assert np.isclose(metrics["missing_disagreement_fraction"], 2 / 3)
