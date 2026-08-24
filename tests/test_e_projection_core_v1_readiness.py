from __future__ import annotations

import ast
from pathlib import Path


def test_readiness_gate_is_metadata_only_and_fail_closed() -> None:
    source = Path("server_training_pipeline/certify_e_projection_core_v1_readiness.py").read_text(
        encoding="utf-8"
    )
    ast.parse(source)
    assert "pd.read_parquet" not in source
    assert "future_covariate_generation_allowed\": ready" in source
    assert "future_prediction_allowed\": False" in source
    assert "phase6a_bias_adjustment_v2" in source
    assert "e_projection_core_v1_release_v2" in source
