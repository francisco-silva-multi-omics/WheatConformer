from __future__ import annotations

import pandas as pd

from server_training_pipeline.resolve_cmip6_esgf_transport import (
    file_period,
    overlaps_interval,
    resolve_asset_files,
    service_urls,
)


def document(title: str, node: str, checksum: str = "abc") -> dict:
    return {
        "dataset_id": f"INSTANCE|{node}",
        "title": title,
        "size": 123,
        "checksum_type": "SHA256",
        "checksum": checksum,
        "url": [
            f"https://{node}/file/{title}|application/netcdf|HTTPServer",
            f"https://{node}/dods/{title}.html|application/opendap-html|OPENDAP",
        ],
    }


def row() -> pd.Series:
    return pd.Series(
        {
            "request_id": "r",
            "source_id": "M",
            "experiment_id": "ssp245",
            "member_id": "r1i1p1f1",
            "variable_id": "tas",
            "grid_label": "gn",
            "version": "20250101",
            "catalog_record_id": "INSTANCE|selected-node",
            "fetch_start": "2015-01-01",
            "fetch_end": "2100-12-31",
        }
    )


def test_file_period_and_overlap_are_calendar_agnostic() -> None:
    assert file_period("tas_day_M_ssp245_r1i1p1f1_gn_20150101-20391230.nc") == (
        "20150101",
        "20391230",
    )
    assert overlaps_interval("20150101", "20391230", "2015-01-01", "2100-12-31")
    assert not overlaps_interval("19000101", "19191231", "1981-01-01", "2014-12-31")


def test_service_urls_remove_only_opendap_html_suffix() -> None:
    item = document("a_20150101-21001231.nc", "node.example")
    assert service_urls(item, "HTTPServer") == [
        "https://node.example/file/a_20150101-21001231.nc"
    ]
    assert service_urls(item, "OPENDAP") == [
        "https://node.example/dods/a_20150101-21001231.nc"
    ]


def test_exact_replicas_with_matching_checksums_are_http_ready() -> None:
    title = "tas_day_M_ssp245_r1i1p1f1_gn_20150101-21001231.nc"
    response = {"response": {"docs": [document(title, "a"), document(title, "b")]}}
    files, summary = resolve_asset_files(row(), response)
    assert len(files) == 1
    assert files[0]["replica_count"] == 2
    assert summary["transport_status"] == "READY_ESGF_HTTP_EXACT"
    assert summary["required_opendap_file_count"] == 1


def test_checksum_conflict_blocks_automatic_transport() -> None:
    title = "tas_day_M_ssp245_r1i1p1f1_gn_20150101-21001231.nc"
    response = {
        "response": {"docs": [document(title, "a", "abc"), document(title, "b", "def")]}
    }
    _, summary = resolve_asset_files(row(), response)
    assert summary["transport_status"] == "BLOCKED_REPLICA_CHECKSUM_CONFLICT"
