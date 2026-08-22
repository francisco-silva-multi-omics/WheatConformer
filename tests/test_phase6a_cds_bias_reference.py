from __future__ import annotations

import pytest

from server_training_pipeline.fetch_phase6a_cds_bias_reference import (
    effective_worker_count,
    request_payload,
)


def test_reference_request_is_full_frozen_period_and_location_specific() -> None:
    protocol = {
        "variables": ["2m_temperature", "total_precipitation"],
        "reference_start": "1981-01-01",
        "reference_end": "2010-12-31",
        "data_format": "csv",
    }
    payload = request_payload(protocol, 12.345678, -98.765432)
    assert payload["date"] == "1981-01-01/2010-12-31"
    assert payload["location"] == {"latitude": 12.34568, "longitude": -98.76543}
    assert payload["variable"] == ["2m_temperature", "total_precipitation"]
    assert payload["data_format"] == "csv"


def test_transport_workers_are_bounded_by_account_and_selected_requests() -> None:
    amendment = {"default_worker_count": 8, "maximum_pending_requests": 20}
    assert effective_worker_count(None, 100, amendment) == 8
    assert effective_worker_count(20, 4, amendment) == 4
    assert effective_worker_count(8, 0, amendment) == 0
    with pytest.raises(ValueError, match="between 1 and 20"):
        effective_worker_count(21, 100, amendment)
