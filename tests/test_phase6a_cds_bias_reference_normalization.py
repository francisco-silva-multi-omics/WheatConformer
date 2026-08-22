from __future__ import annotations

from pathlib import Path

from server_training_pipeline.normalize_phase6a_cds_bias_reference import load_json


def test_bias_reference_contract_is_historical_only() -> None:
    contract = load_json(Path("server_training_pipeline/phase6a_cds_bias_reference_protocol_v1.json"))
    assert contract["reference_start"] == "1981-01-01"
    assert contract["reference_end"] == "2010-12-31"
    assert contract["request_concurrency"] == 1
    assert not contract["future_covariate_matrices_generated"]
    assert not contract["future_predictions_generated"]
