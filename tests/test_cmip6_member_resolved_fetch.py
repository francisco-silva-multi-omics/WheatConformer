from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from server_training_pipeline.fetch_cmip6_member_resolved import (
    asset_key_columns,
    build_transport_inventory,
    haversine_km,
    normalize_version,
    site_subset,
)


PROTOCOL = json.loads(
    Path(
        "server_training_pipeline/phase6a_cmip6_member_fetch_protocol_v1.json"
    ).read_text(encoding="utf-8")
)


def selected_assets() -> pd.DataFrame:
    rows = []
    for experiment in ("historical", "ssp126", "ssp245", "ssp370", "ssp585"):
        for variable in ("tasmin", "tasmax", "tas", "pr", "rsds", "sfcWind", "hurs"):
            rows.append(
                {
                    "institution_id": "TEST",
                    "source_id": "MODEL-A",
                    "experiment_id": experiment,
                    "member_id": "r1i1p1f1",
                    "variable": variable,
                    "grid_label": "gn",
                    "version": "v20250101",
                    "catalog_record_id": f"MODEL-A.{experiment}.{variable}|node",
                    "calendar": "365_day",
                }
            )
    result = pd.DataFrame(rows)
    return pd.concat([result] * 13, ignore_index=True).assign(
        source_id=lambda frame: [f"MODEL-{i // 35:02d}" for i in range(len(frame))],
        catalog_record_id=lambda frame: [f"asset-{i}" for i in range(len(frame))],
    )


def cloud_catalog(assets: pd.DataFrame, drop_last: bool = False) -> pd.DataFrame:
    cloud = assets.rename(columns={"variable": "variable_id"}).copy()
    cloud["table_id"] = "day"
    cloud["zstore"] = [f"gs://cmip6/{value}" for value in cloud.catalog_record_id]
    columns = asset_key_columns() + ["table_id", "zstore"]
    return cloud.iloc[:-1][columns] if drop_last else cloud[columns]


def test_transport_inventory_requires_exact_version_and_has_455_unique_requests() -> None:
    assets = selected_assets()
    result = build_transport_inventory(assets, cloud_catalog(assets), PROTOCOL)
    assert len(result) == 455
    assert result.request_id.nunique() == 455
    assert result.transport_status.eq("READY_TO_FETCH").all()
    assert set(result.member_id) == {"r1i1p1f1"}


def test_missing_exact_cloud_replica_remains_explicitly_pending() -> None:
    assets = selected_assets()
    result = build_transport_inventory(assets, cloud_catalog(assets, drop_last=True), PROTOCOL)
    assert result.transport_status.value_counts().to_dict() == {
        "READY_TO_FETCH": 454,
        "PENDING_EXACT_REPLICA_RESOLUTION": 1,
    }


def test_duplicate_exact_cloud_assets_fail_closed() -> None:
    assets = selected_assets()
    cloud = cloud_catalog(assets)
    cloud = pd.concat([cloud, cloud.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate exact scientific assets"):
        build_transport_inventory(assets, cloud, PROTOCOL)


def test_version_normalization_only_removes_transport_prefix_and_float_suffix() -> None:
    observed = normalize_version(pd.Series(["v20250101", "20250101.0", "20250102"]))
    assert observed.tolist() == ["20250101", "20250101", "20250102"]


def test_haversine_handles_zero_and_dateline_distance() -> None:
    zero = haversine_km(
        np.array([10.0]), np.array([179.5]), np.array([10.0]), np.array([179.5])
    )
    crossing = haversine_km(
        np.array([0.0]), np.array([179.5]), np.array([0.0]), np.array([-179.5])
    )
    assert zero[0] == pytest.approx(0.0)
    assert crossing[0] == pytest.approx(111.195, rel=1e-3)


def test_site_subset_uses_vectorized_nearest_cells_and_preserves_sites() -> None:
    xr = pytest.importorskip("xarray")
    time = pd.date_range("1981-01-01", periods=3)
    values = np.arange(3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4)
    dataset = xr.Dataset(
        {"tas": (("time", "lat", "lon"), values)},
        coords={"time": time, "lat": [-10.0, 0.0, 10.0], "lon": [0.0, 90.0, 180.0, 270.0]},
    )
    row = pd.Series(
        {
            "variable_id": "tas",
            "fetch_start": "1981-01-01",
            "fetch_end": "1981-01-03",
        }
    )
    sites = pd.DataFrame(
        {"site_id": ["a", "b"], "latitude": [1.0, 9.0], "longitude": [-89.0, 179.0]}
    )
    result = site_subset(dataset, row, sites)
    assert result.sizes == {"time": 3, "site": 2}
    assert result.site_id.values.tolist() == ["a", "b"]
    assert result.source_longitude.values.tolist() == [270.0, 180.0]
    assert np.isfinite(result.source_grid_distance_km.values).all()


def test_protocol_keeps_predictions_and_feature_generation_disabled() -> None:
    assert PROTOCOL["maximum_fetch_workers"] == 1
    assert PROTOCOL["member_dimension_must_be_retained"]
    assert not PROTOCOL["future_covariate_matrices_generated"]
    assert not PROTOCOL["future_predictions_generated"]
    assert not PROTOCOL["phenotype_values_read"]
