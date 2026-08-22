from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from server_training_pipeline.fetch_cmip6_member_resolved import (
    atomic_json,
    resolve,
    sha256_file,
    site_subset,
)


DEFAULT_PROTOCOL = Path(
    "server_training_pipeline/phase6a_cmip6_mri_historical_pr_repair_protocol_v1.json"
)
DEFAULT_CACHE = Path("environment/v2/phase6a_cmip6_member_resolved_daily_v1")
DEFAULT_AUDIT = Path("audit/v2/phase6a_cmip6_mri_historical_pr_repair_v1")


def date_keys(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{value.year:04d}-{value.month:02d}-{value.day:02d}" for value in values],
        dtype=str,
    )


def sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_protocol(root: Path, path: Path) -> dict[str, Any]:
    resolved = resolve(root, path)
    protocol = json.loads(resolved.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "phase6a_cmip6_mri_historical_pr_repair_v1":
        raise ValueError("MRI historical precipitation repair protocol mismatch")
    for relative, expected in protocol["parent_artifacts"].items():
        observed = sha256_file(resolve(root, Path(relative)))
        if observed != expected:
            raise ValueError(f"Frozen parent artifact changed: {relative}")
    protocol["_path"] = str(resolved)
    protocol["_sha256"] = sha256_file(resolved)
    return protocol


def download_exact_file(
    urls: list[str], target: Path, expected_bytes: int, expected_sha256: str
) -> tuple[str, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == expected_bytes:
        if sha256_file(target) == expected_sha256:
            return "CACHED", "published_exact_esgf_sha256"
    part = target.with_suffix(target.suffix + ".part")
    for url in urls:
        for attempt in range(4):
            try:
                offset = part.stat().st_size if part.is_file() else 0
                headers = {"User-Agent": "WheatConformer-Phase6A/1.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=120) as response:
                    status = getattr(response, "status", response.getcode())
                    mode = "ab" if offset and status == 206 else "wb"
                    if mode == "wb":
                        offset = 0
                    with part.open(mode) as handle:
                        while True:
                            block = response.read(8 * 1024 * 1024)
                            if not block:
                                break
                            handle.write(block)
                if part.stat().st_size != expected_bytes:
                    raise ValueError(
                        f"Exact ESGF byte count mismatch: {part.stat().st_size} != {expected_bytes}"
                    )
                observed = sha256_file(part)
                if observed != expected_sha256:
                    raise ValueError(f"Exact ESGF checksum mismatch: {observed}")
                os.replace(part, target)
                return "FETCHED", sha256_bytes(url)
            except (OSError, ValueError, urllib.error.URLError):
                if attempt == 3:
                    break
                time.sleep(2 ** attempt)
    raise ValueError("No frozen exact ESGF replica produced the published file")


def assert_daily_axis(keys: np.ndarray, start: str, end: str, expected: int) -> None:
    if len(keys) != expected:
        raise ValueError(f"Daily time count mismatch: {len(keys)} != {expected}")
    if keys[0] != start or keys[-1] != end:
        raise ValueError(f"Daily extent mismatch: {keys[0]}..{keys[-1]}")
    if len(set(keys.tolist())) != len(keys):
        raise ValueError("Daily time axis contains duplicate dates")
    parsed = pd.to_datetime(keys)
    if not np.all(np.diff(parsed.values).astype("timedelta64[D]") == np.timedelta64(1, "D")):
        raise ValueError("Daily time axis is nonmonotonic or contains gaps")


def run(root: Path, protocol_path: Path, cache_path: Path, audit_path: Path) -> dict[str, Any]:
    protocol = load_protocol(root, protocol_path)
    target = protocol["target"]
    request_id = target["request_id"]
    cache = resolve(root, cache_path)
    audit = resolve(root, audit_path)
    audit.mkdir(parents=True, exist_ok=True)
    asset = cache / "assets" / request_id[:2] / f"{request_id}.nc"
    receipt = cache / "requests" / request_id[:2] / f"{request_id}.json"
    parent = protocol["incomplete_parent"]
    if sha256_file(asset) != parent["asset_sha256"]:
        existing_receipt = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            existing_receipt.get("repair_protocol_sha256") == protocol["_sha256"]
            and existing_receipt.get("status") == "FETCHED"
            and existing_receipt.get("output_sha256") == sha256_file(asset)
        ):
            return existing_receipt
        raise ValueError("Incomplete parent asset identity changed before repair")
    if sha256_file(receipt) != parent["receipt_sha256"]:
        raise ValueError("Incomplete parent receipt identity changed before repair")

    exact = protocol["exact_esgf_file"]
    raw = audit / "raw" / exact["title"]
    transport_status, transport_url_sha256 = download_exact_file(
        exact["urls"], raw, int(exact["bytes"]), exact["sha256"]
    )
    decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(asset, engine="h5netcdf", decode_times=decoder) as prior:
        prior_loaded = prior.load()
        site_ids = np.asarray(prior_loaded.site_id.values).astype(str)
        sites = pd.DataFrame(
            {
                "site_id": site_ids,
                "latitude": prior_loaded.target_latitude.values,
                "longitude": prior_loaded.target_longitude.values,
            }
        )
        prior_keys = date_keys(prior_loaded.time.values)
        variable_attrs = dict(prior_loaded[target["variable_id"]].attrs)
        inherited_attrs = dict(prior_loaded.attrs)
    assert_daily_axis(
        prior_keys,
        parent["start"],
        parent["end"],
        int(parent["time_count"]),
    )

    row = pd.Series(
        {
            "source_id": target["source_id"],
            "experiment_id": target["experiment_id"],
            "member_id": target["member_id"],
            "grid_label": target["grid_label"],
            "variable_id": target["variable_id"],
            "fetch_start": exact["start"],
            "fetch_end": exact["end"],
        }
    )
    with xr.open_dataset(raw, engine="h5netcdf", decode_times=decoder) as source:
        selected = site_subset(source, row, sites).load()
    selected_keys = date_keys(selected.time.values)
    assert_daily_axis(
        selected_keys,
        exact["start"],
        exact["end"],
        len(pd.date_range(exact["start"], exact["end"], freq="D")),
    )
    for name in ("site_id", "source_latitude", "source_longitude"):
        left = np.asarray(prior_loaded[name].values)
        right = np.asarray(selected[name].values)
        if name == "site_id":
            equal = np.array_equal(left.astype(str), right.astype(str))
        else:
            equal = np.allclose(left.astype(float), right.astype(float), atol=1e-10, rtol=0)
        if not equal:
            raise ValueError(f"Exact ESGF repair changed the frozen {name} axis")

    repaired = xr.concat(
        [prior_loaded[[target["variable_id"]]], selected[[target["variable_id"]]]],
        dim="time",
    )
    for name in (
        "site",
        "site_id",
        "target_latitude",
        "target_longitude",
        "source_latitude",
        "source_longitude",
        "source_grid_distance_km",
    ):
        repaired = repaired.assign_coords({name: prior_loaded[name]})
    repaired[target["variable_id"]].attrs.update(variable_attrs)
    repaired.attrs.update(inherited_attrs)
    repaired.attrs.update(
        {
            "phase6a_repair_protocol_version": protocol["protocol_version"],
            "phase6a_repair_protocol_sha256": protocol["_sha256"],
            "transport": "PANGEO_ZARR_PLUS_EXACT_ESGF_HTTP_REPAIR",
            "exact_esgf_file_sha256": exact["sha256"],
            "incomplete_parent_asset_sha256": parent["asset_sha256"],
            "future_covariate_matrix": "not_generated",
            "future_prediction": "not_generated",
        }
    )
    repaired_keys = date_keys(repaired.time.values)
    assert_daily_axis(
        repaired_keys,
        target["required_start"],
        target["required_end"],
        int(target["expected_time_count"]),
    )
    if int(repaired.sizes["site"]) != int(target["site_count"]):
        raise ValueError("Repaired asset changed the frozen site count")
    values = np.asarray(repaired[target["variable_id"]].values)
    if not np.isfinite(values).all() or float(np.nanmin(values)) < -1e-7:
        raise ValueError("Repaired precipitation contains nonfinite or materially negative values")

    temporary = asset.with_name(f".{asset.name}.{os.getpid()}.repair")
    repaired = repaired.chunk({"time": 366, "site": 128})
    repaired.to_netcdf(
        temporary,
        engine="h5netcdf",
        encoding={
            target["variable_id"]: {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "chunksizes": (366, 128),
            }
        },
    )
    with xr.open_dataset(temporary, engine="h5netcdf", decode_times=decoder) as check:
        assert_daily_axis(
            date_keys(check.time.values),
            target["required_start"],
            target["required_end"],
            int(target["expected_time_count"]),
        )
        if int(check.sizes["site"]) != int(target["site_count"]):
            raise ValueError("Serialized repair changed the site axis")
    output_sha256 = sha256_file(temporary)
    output_bytes = temporary.stat().st_size
    os.replace(temporary, asset)
    repaired_receipt = {
        "status": "FETCHED",
        "request_id": request_id,
        "source_id": target["source_id"],
        "experiment_id": target["experiment_id"],
        "member_id": target["member_id"],
        "variable": target["variable_id"],
        "grid_label": target["grid_label"],
        "version": target["version"],
        "calendar": target["calendar"],
        "fetch_start": target["required_start"],
        "fetch_end": target["required_end"],
        "site_count": int(target["site_count"]),
        "time_count": int(target["expected_time_count"]),
        "transport": "PANGEO_ZARR_PLUS_EXACT_ESGF_HTTP_REPAIR",
        "transport_source_sha256": transport_url_sha256,
        "repair_protocol_sha256": protocol["_sha256"],
        "exact_esgf_file_sha256": exact["sha256"],
        "incomplete_parent_asset_sha256": parent["asset_sha256"],
        "output_path": str(asset),
        "output_bytes": output_bytes,
        "output_sha256": output_sha256,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(receipt, repaired_receipt)
    provenance = {
        "status": "PASS",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["_sha256"],
        "transport_status": transport_status,
        "request_id": request_id,
        "exact_esgf_file_sha256": exact["sha256"],
        "repaired_asset_sha256": output_sha256,
        "repaired_time_count": int(target["expected_time_count"]),
        "repaired_site_count": int(target["site_count"]),
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(audit / "mri_historical_pr_repair_provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    result = run(args.root.resolve(), args.protocol, args.cache, args.audit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
