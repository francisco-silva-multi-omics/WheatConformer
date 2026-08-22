from __future__ import annotations

import pandas as pd

from server_training_pipeline.fit_phase6a_applicability_domain_reference import (
    feature_block,
    robust_scale_parameters,
)


def test_feature_blocks_are_deterministic() -> None:
    assert feature_block("w00_29__tasmax_max_c") == "heat"
    assert feature_block("w00_29__precipitation_sum_mm") == "water"
    assert feature_block("w00_29__radiation_sum_mj_m2") == "radiation"
    assert feature_block("latitude") == "geography"


def test_robust_scale_has_nonzero_fallback() -> None:
    frame = pd.DataFrame({"feature": [1.0, 1.0, 1.0]})
    result = robust_scale_parameters(frame, ["feature"]).iloc[0]
    assert result.robust_scale == 1.0
    assert result.scale_source == "UNIT_FALLBACK"
