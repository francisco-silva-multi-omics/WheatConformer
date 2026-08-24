from __future__ import annotations

from pathlib import Path


def test_release_freezer_keeps_prediction_blocked() -> None:
    source = Path("server_training_pipeline/freeze_e_projection_core_v1.py").read_text(
        encoding="utf-8"
    )
    assert '"future_prediction_allowed": False' in source
    assert '"future_covariate_matrices_generated": 0' in source
    assert '"future_predictions_generated": 0' in source
    assert "phase6a_bias_adjustment_contract_v2.json" in source
    assert "phase6a_historical_transfer_contract_v2.json" in source
    assert "e_projection_core_v1_release_v2" in source
