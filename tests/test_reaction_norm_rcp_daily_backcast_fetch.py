from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from server_training_pipeline.fetch_reaction_norm_rcp_historical_daily import (
    main as fetch_daily,
    parse_daily_payload,
    request_url,
)
from server_training_pipeline.final_evaluation_contract import file_sha256


def protocol() -> dict[str, object]:
    return json.loads(
        Path(
            "server_training_pipeline/reaction_norm_rcp_daily_backcast_protocol_v1.json"
        ).read_text()
    )


def request_row() -> pd.Series:
    return pd.Series(
        {
            "request_id": "abc123",
            "request_kind": "pre_sowing_antecedent_water_balance",
            "request_start_date": "2020-01-01",
            "request_end_date": "2020-01-02",
            "latitude": 20.0,
            "longitude": -100.0,
            "required_daily_variables": "pr;et0;soil_moisture_0_7;soil_moisture_7_28",
            "request_status": "READY_TO_FETCH",
        }
    )


def payload() -> dict[str, object]:
    return {
        "daily": {
            "time": ["2020-01-01", "2020-01-02"],
            "precipitation_sum": [1.0, 2.0],
            "et0_fao_evapotranspiration": [0.5, 0.75],
        },
        "hourly": {
            "time": [
                "2020-01-01T00:00",
                "2020-01-01T12:00",
                "2020-01-02T00:00",
                "2020-01-02T12:00",
            ],
            "soil_moisture_0_to_7cm": [0.1, 0.2, 0.3, 0.4],
            "soil_moisture_7_to_28cm": [0.2, 0.3, 0.4, 0.5],
        },
    }


def test_daily_backcast_url_and_payload_preserve_dates_and_variables() -> None:
    row = request_row()
    url = request_url(row, protocol())
    query = parse_qs(urlparse(url).query)
    assert query["models"] == ["era5"]
    assert query["timezone"] == ["GMT"]
    assert query["daily"] == ["precipitation_sum,et0_fao_evapotranspiration"]
    assert query["hourly"] == [
        "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm"
    ]
    frame = parse_daily_payload(payload(), row, protocol())
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-01",
        "2020-01-02",
    ]
    assert frame["precipitation_sum_mm"].tolist() == [1.0, 2.0]
    assert frame["et0_fao_evapotranspiration_mm"].tolist() == [0.5, 0.75]
    np.testing.assert_allclose(
        frame["soil_moisture_0_7_mean_m3m3"].to_numpy(), [0.15, 0.35]
    )


def test_bounded_fetch_writes_resumable_cache_without_future_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    inventory = tmp_path / "requests.tsv"
    pd.DataFrame([request_row().to_dict()]).to_csv(inventory, sep="\t", index=False)
    certification = tmp_path / "reconstruction.json"
    certification.write_text(
        json.dumps(
            {
                "status": "PASS",
                "output_artifacts": {inventory.name: file_sha256(inventory)},
            }
        ),
        encoding="utf-8",
    )
    protocol_path = Path(
        "server_training_pipeline/reaction_norm_rcp_daily_backcast_protocol_v1.json"
    ).resolve()
    out_dir = tmp_path / "daily"

    monkeypatch.setattr(
        "server_training_pipeline.fetch_reaction_norm_rcp_historical_daily.fetch_json",
        lambda *args, **kwargs: payload(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch",
            "--request-inventory",
            str(inventory),
            "--reconstruction-certification",
            str(certification),
            "--protocol",
            str(protocol_path),
            "--out-dir",
            str(out_dir),
            "--workers",
            "1",
            "--request-sleep",
            "0",
        ],
    )
    fetch_daily()
    provenance = json.loads((out_dir / "RCP_daily_backcast_provenance.json").read_text())
    assert provenance["status"] == "PASS"
    assert provenance["run_status"] == "COMPLETE"
    assert provenance["archive_complete"] is True
    assert provenance["future_covariate_population_allowed"] is False
    assert provenance["rcp_predictions_allowed"] is False
    index = pd.read_csv(out_dir / "RCP_daily_backcast_request_index.tsv", sep="\t")
    assert index["status"].tolist() == ["FETCHED"]
    cache = Path(index.loc[0, "data_path"])
    assert cache.is_file()
    assert not list(out_dir.glob("*prediction*"))

    monkeypatch.setattr(sys, "argv", sys.argv)
    fetch_daily()
    second = json.loads((out_dir / "RCP_daily_backcast_provenance.json").read_text())
    assert second["status_counts"] == {"CACHED": 1}
    assert second["fetched_this_run"] == 0


def test_pre_1940_request_is_reported_without_fetch_or_clipping(
    tmp_path: Path, monkeypatch
) -> None:
    row = request_row().copy()
    row["request_start_date"] = "1939-01-01"
    row["request_end_date"] = "1939-01-02"
    inventory = tmp_path / "requests.tsv"
    pd.DataFrame([row.to_dict()]).to_csv(inventory, sep="\t", index=False)
    certification = tmp_path / "reconstruction.json"
    certification.write_text(
        json.dumps(
            {
                "status": "PASS",
                "output_artifacts": {inventory.name: file_sha256(inventory)},
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "daily"

    def forbidden_fetch(*args, **kwargs):
        raise AssertionError("Out-of-coverage request must not be fetched")

    monkeypatch.setattr(
        "server_training_pipeline.fetch_reaction_norm_rcp_historical_daily.fetch_json",
        forbidden_fetch,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch",
            "--request-inventory",
            str(inventory),
            "--reconstruction-certification",
            str(certification),
            "--protocol",
            str(
                Path(
                    "server_training_pipeline/reaction_norm_rcp_daily_backcast_protocol_v1.json"
                ).resolve()
            ),
            "--out-dir",
            str(out_dir),
            "--workers",
            "1",
            "--request-sleep",
            "0",
        ],
    )
    fetch_daily()
    index = pd.read_csv(out_dir / "RCP_daily_backcast_request_index.tsv", sep="\t")
    assert index["status"].tolist() == ["OUT_OF_COVERAGE"]
    assert "not_clamped" in index.loc[0, "detail"]
