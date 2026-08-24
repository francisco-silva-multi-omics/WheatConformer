from __future__ import annotations

import numpy as np
import pandas as pd

from build_baseline import compute_hmp_qc


def test_missing_calls_are_excluded_from_qc_and_imputed_after_filtering() -> None:
    matrix = pd.DataFrame(
        {
            "m1": [0, 0, -9],
            "m2": [0, 2, -9],
            "m3": [-9, 2, -9],
            "m4": [2, 2, -9],
            "m5": [-9, -9, 0],
        }
    )
    thresholds = {
        "maf_min": 0.0,
        "marker_het_max": 1.0,
        "sample_het_max": 1.0,
        "marker_missing_max": 0.8,
        "sample_missing_max": 0.5,
    }
    result = compute_hmp_qc(matrix, pd.Series(["s1", "s2", "s3"]), thresholds)

    imputed = result["matrix_imputed"]
    assert result["sample_ids"].tolist() == ["s1", "s2"]
    assert "m5" not in imputed.columns
    assert result["marker_qc"].set_index("marker").loc["m5", "removal_reason"] == "all_nan_after_sample_qc"
    assert result["sample_qc"].set_index("sample_id").loc["s3", "removal_reason"] == "high_missingness"
    assert result["missing_before_imputation"] > 0
    assert result["missing_after_imputation"] == 0
    assert np.isfinite(imputed.to_numpy()).all()
    assert not np.any(imputed.to_numpy() == -9)
    assert np.isfinite(result["kernel"]).all()


def test_missing_code_does_not_bias_maf() -> None:
    matrix = pd.DataFrame({"marker": [0, 2, -9, -9], "variable": [0, 0, 2, 2]})
    thresholds = {
        "maf_min": 0.0,
        "marker_het_max": 1.0,
        "sample_het_max": 1.0,
        "marker_missing_max": 1.0,
        "sample_missing_max": 1.0,
    }
    result = compute_hmp_qc(matrix, pd.Series(["s1", "s2", "s3", "s4"]), thresholds)
    marker_qc = result["marker_qc"].set_index("marker")
    assert marker_qc.loc["marker", "maf"] == 0.5
