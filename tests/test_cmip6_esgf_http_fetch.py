from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest

from server_training_pipeline.fetch_cmip6_esgf_http import (
    assemble_asset,
    asset_transport_priority,
    candidate_urls,
    checksum_file,
)


def test_candidate_urls_prefers_https_upgrade_and_retains_original() -> None:
    observed = candidate_urls('["http://node.example/a.nc","https://b.example/a.nc"]')
    assert observed == [
        "https://node.example/a.nc",
        "http://node.example/a.nc",
        "https://b.example/a.nc",
    ]


def test_candidate_urls_applies_frozen_replica_host_priority() -> None:
    observed = candidate_urls(
        '["http://slow.example/a.nc","https://fast.example/a.nc"]',
        ["fast.example", "slow.example"],
    )
    assert observed == [
        "https://fast.example/a.nc",
        "https://slow.example/a.nc",
        "http://slow.example/a.nc",
    ]


def test_asset_transport_priority_uses_slowest_required_file() -> None:
    files = pd.DataFrame(
        {
            "http_urls_json": [
                json.dumps(["https://fast.example/a.nc"]),
                json.dumps(
                    [
                        "https://fast.example/b.nc",
                        "https://fallback.example/b.nc",
                    ]
                ),
                json.dumps(["https://slow.example/c.nc"]),
            ]
        }
    )
    assert asset_transport_priority(
        files, ["fast.example", "fallback.example", "slow.example"]
    ) == 2


def test_checksum_file_supports_esgf_algorithms(tmp_path: Path) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"abc")
    assert checksum_file(path, "MD5") == "900150983cd24fb0d6963f7d28e17f72"
    assert (
        checksum_file(path, "SHA256")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_esgf_http_fallback_downloads_subsets_and_certifies_local_archive(
    tmp_path: Path,
) -> None:
    xr = pytest.importorskip("xarray")
    source = tmp_path / "tas_day_TEST_ssp245_r1i1p1f1_gn_20150101-20150103.nc"
    values = np.arange(3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4)
    dataset = xr.Dataset(
        {"tas": (("time", "lat", "lon"), values)},
        coords={
            "time": pd.date_range("2015-01-01", periods=3),
            "lat": [-10.0, 0.0, 10.0],
            "lon": [0.0, 90.0, 180.0, 270.0],
        },
        attrs={
            "source_id": "TEST",
            "experiment_id": "ssp245",
            "variant_label": "r1i1p1f1",
            "grid_label": "gn",
            "variable_id": "tas",
        },
    )
    dataset.to_netcdf(source, engine="h5netcdf")
    checksum = checksum_file(source, "SHA256")
    request_id = "a" * 64
    row = pd.Series(
        {
            "request_id": request_id,
            "source_id": "TEST",
            "experiment_id": "ssp245",
            "member_id": "r1i1p1f1",
            "variable_id": "tas",
            "grid_label": "gn",
            "version": "20250101",
            "calendar": "proleptic_gregorian",
            "fetch_start": "2015-01-01",
            "fetch_end": "2015-01-03",
        }
    )
    files = pd.DataFrame(
        [
            {
                "title": source.name,
                "size_bytes": source.stat().st_size,
                "checksum_type": "SHA256",
                "checksum": checksum,
                "http_urls_json": json.dumps([source.as_uri()]),
            }
        ]
    )
    sites = pd.DataFrame(
        {
            "site_id": ["a", "b"],
            "latitude": [1.0, 9.0],
            "longitude": [-89.0, 179.0],
        }
    )
    cache = tmp_path / "cache"
    protocol = {
        "protocol_version": "test",
        "_protocol_sha256": "p",
        "selected_asset_manifest_sha256": "a",
        "site_inventory_sha256": "s",
        "minimum_free_space_gib": 0,
    }
    result = assemble_asset(row, files, sites, cache, protocol, retries=0, timeout=10)
    assert result["status"] == "FETCHED"
    assert result["time_count"] == 3
    output = Path(result["output_path"])
    assert output.is_file()
    assert checksum_file(output, "SHA256") == result["output_sha256"]
    with xr.open_dataset(output, engine="h5netcdf") as observed:
        assert observed.sizes == {"time": 3, "site": 2}
        assert observed.attrs["transport"] == "ESGF_HTTP_EXACT"
