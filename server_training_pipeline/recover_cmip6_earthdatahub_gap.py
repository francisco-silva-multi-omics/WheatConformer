from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

from server_training_pipeline.fetch_cmip6_member_resolved import (
    atomic_json,
    resolve,
    sha256_file,
)


DEFAULT_PROTOCOL = Path(
    "server_training_pipeline/phase6a_cmip6_earthdatahub_gap_recovery_protocol_v1.json"
)
DEFAULT_CACHE = Path("environment/v2/phase6a_cmip6_member_resolved_daily_v1")
DEFAULT_AUDIT = Path("audit/v2/phase6a_cmip6_earthdatahub_gap_recovery_v1")
TOKEN_ENVIRONMENT_VARIABLE = "EARTHDATAHUB_API_KEY"


def normalize_label(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return "".join(character for character in str(value).lower() if character.isalnum())


def time_keys(values: np.ndarray) -> np.ndarray:
    result = []
    for value in values:
        if hasattr(value, "year"):
            result.append(f"{value.year:04d}-{value.month:02d}-{value.day:02d}")
        else:
            result.append(np.datetime_as_string(value, unit="D"))
    return np.asarray(result, dtype=str)


def select_experiment(dataset: xr.Dataset, target: str) -> xr.Dataset:
    normalized_target = normalize_label(target)
    preferred = ("experiment_id", "experiment", "scenario", "ssp")
    candidates = list(preferred) + [
        name for name in dataset.coords if name not in preferred
    ]
    for name in candidates:
        if name not in dataset.variables:
            continue
        coordinate = dataset[name]
        if coordinate.ndim != 1 or coordinate.size > 100:
            continue
        values = np.asarray(coordinate.values)
        matches = [
            index
            for index, value in enumerate(values)
            if normalize_label(value) == normalized_target
        ]
        if len(matches) != 1:
            continue
        selected_value = values[matches[0]].item() if hasattr(values[matches[0]], "item") else values[matches[0]]
        dimension = coordinate.dims[0]
        if coordinate.name == dimension:
            return dataset.sel({dimension: selected_value}, drop=True)
        return dataset.isel({dimension: matches[0]}, drop=True)
    raise ValueError(f"Earth Data Hub dataset has no unique {target!r} experiment coordinate")


def coordinate_name(dataset: xr.Dataset, standard_name: str, fallbacks: tuple[str, ...]) -> str:
    for name in dataset.coords:
        if str(dataset[name].attrs.get("standard_name", "")).lower() == standard_name:
            return name
    for name in fallbacks:
        if name in dataset.coords:
            return name
    raise ValueError(f"Earth Data Hub dataset lacks a {standard_name} coordinate")


def normalize_longitudes(values: np.ndarray, provider_longitudes: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    provider_longitudes = np.asarray(provider_longitudes, dtype=np.float64)
    if np.nanmin(provider_longitudes) >= 0.0:
        return np.mod(values, 360.0)
    return (values + 180.0) % 360.0 - 180.0


def longitude_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs((np.asarray(left) - np.asarray(right) + 180.0) % 360.0 - 180.0)


def select_sites(
    dataset: xr.Dataset,
    variable: str,
    source_latitudes: np.ndarray,
    source_longitudes: np.ndarray,
) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    latitude_name = coordinate_name(dataset, "latitude", ("lat", "latitude"))
    longitude_name = coordinate_name(dataset, "longitude", ("lon", "longitude"))
    latitude = dataset[latitude_name]
    longitude = dataset[longitude_name]
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("Earth Data Hub fallback requires one-dimensional native lat/lon axes")
    target_longitudes = normalize_longitudes(source_longitudes, longitude.values)
    selected = dataset[variable].sel(
        {
            latitude_name: xr.DataArray(source_latitudes, dims="site"),
            longitude_name: xr.DataArray(target_longitudes, dims="site"),
        },
        method="nearest",
    )
    for dimension in list(selected.dims):
        if dimension not in {"time", "site"}:
            if selected.sizes[dimension] != 1:
                raise ValueError(
                    f"Earth Data Hub variable retains unsupported dimension {dimension!r}"
                )
            selected = selected.isel({dimension: 0}, drop=True)
    if set(selected.dims) != {"time", "site"}:
        raise ValueError(f"Unexpected selected dimensions: {selected.dims}")
    selected = selected.transpose("time", "site")
    selected_latitudes = np.asarray(selected[latitude_name].values, dtype=np.float64)
    selected_longitudes = np.asarray(selected[longitude_name].values, dtype=np.float64)
    return selected, selected_latitudes, selected_longitudes


def concordance_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(f"Concordance shape mismatch: {reference.shape} != {candidate.shape}")
    reference_finite = np.isfinite(reference)
    candidate_finite = np.isfinite(candidate)
    mask_disagreement = np.logical_xor(reference_finite, candidate_finite)
    shared = reference_finite & candidate_finite
    if not shared.any():
        raise ValueError("No finite values are available for provider concordance")
    left = reference[shared]
    right = candidate[shared]
    delta = right - left
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.sqrt(
        np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered)
    )
    correlation = float(np.dot(left_centered, right_centered) / denominator)
    return {
        "compared_values": int(shared.sum()),
        "missing_mask_disagreement_fraction": float(mask_disagreement.mean()),
        "absolute_bias_w_m2": float(abs(delta.mean())),
        "rmse_w_m2": float(np.sqrt(np.mean(np.square(delta)))),
        "p99_absolute_delta_w_m2": float(np.quantile(np.abs(delta), 0.99)),
        "maximum_absolute_delta_w_m2": float(np.max(np.abs(delta))),
        "pearson_correlation": correlation,
    }


def concordance_checks(
    metrics: dict[str, float | int],
    protocol: dict[str, Any],
    site_count: int,
    day_count: int,
    latitude_delta: float,
    longitude_delta_value: float,
) -> dict[str, bool]:
    overlap = protocol["exact_overlap"]
    thresholds = protocol["acceptance_thresholds"]
    return {
        "overlap_site_count": site_count == int(overlap["expected_sites"]),
        "overlap_day_count": day_count == int(overlap["expected_days"]),
        "minimum_compared_values": int(metrics["compared_values"])
        >= int(overlap["minimum_compared_values"]),
        "source_latitude_alignment": latitude_delta
        <= float(thresholds["maximum_source_latitude_delta_degrees"]),
        "source_longitude_alignment": longitude_delta_value
        <= float(thresholds["maximum_source_longitude_delta_degrees"]),
        "missing_mask_concordance": float(metrics["missing_mask_disagreement_fraction"])
        <= float(thresholds["maximum_missing_mask_disagreement_fraction"]),
        "absolute_bias": float(metrics["absolute_bias_w_m2"])
        <= float(thresholds["maximum_absolute_bias_w_m2"]),
        "rmse": float(metrics["rmse_w_m2"])
        <= float(thresholds["maximum_rmse_w_m2"]),
        "p99_absolute_delta": float(metrics["p99_absolute_delta_w_m2"])
        <= float(thresholds["maximum_p99_absolute_delta_w_m2"]),
        "maximum_absolute_delta": float(metrics["maximum_absolute_delta_w_m2"])
        <= float(thresholds["maximum_absolute_delta_w_m2"]),
        "pearson_correlation": float(metrics["pearson_correlation"])
        >= float(thresholds["minimum_pearson_correlation"]),
    }


def load_protocol(root: Path, path: Path) -> dict[str, Any]:
    resolved = resolve(root, path)
    protocol = json.loads(resolved.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "phase6a_cmip6_earthdatahub_gap_recovery_v1":
        raise ValueError("Earth Data Hub gap-recovery protocol identity mismatch")
    if protocol.get("asset_replacement_policy") != (
        "replace_entire_target_site_time_asset_only_after_exact_overlap_concordance_passes"
    ):
        raise ValueError("Earth Data Hub asset replacement policy is not frozen")
    for relative, expected in protocol["parent_artifacts"].items():
        artifact = resolve(root, Path(relative))
        observed = sha256_file(artifact)
        if observed != expected:
            raise ValueError(f"Frozen parent artifact changed: {relative}")
    protocol["_path"] = str(resolved)
    protocol["_sha256"] = sha256_file(resolved)
    return protocol


def authenticated_mapper(url: str, token: str):
    authorization = base64.b64encode(f"edh:{token}".encode("utf-8")).decode("ascii")
    filesystem = fsspec.filesystem(
        "https", headers={"Authorization": f"Basic {authorization}"}
    )
    return filesystem.get_mapper(url)


def sanitized_error(exc: Exception, token: str) -> str:
    return str(exc).replace(token, "[REDACTED]") if token else str(exc)


def run(root: Path, protocol_path: Path, cache_path: Path, audit_path: Path) -> dict[str, Any]:
    started = time.time()
    protocol = load_protocol(root, protocol_path)
    target = protocol["target"]
    token = os.environ.get(TOKEN_ENVIRONMENT_VARIABLE, "").strip()
    if not token:
        raise ValueError(f"{TOKEN_ENVIRONMENT_VARIABLE} is not set")
    cache = resolve(root, cache_path)
    audit = resolve(root, audit_path)
    audit.mkdir(parents=True, exist_ok=True)
    reference_path = resolve(
        root,
        Path(
            "environment/v2/phase6a_cmip6_member_resolved_daily_v1/"
            "tmp_http/cc53a57abadeaaf5/parts/8fa9c4c2c7eb1751.nc"
        ),
    )
    decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(reference_path, engine="h5netcdf", decode_times=decoder) as exact:
        site_ids = np.asarray(exact["site_id"].values).astype(str)
        source_latitudes = np.asarray(exact["source_latitude"].values, dtype=np.float64)
        source_longitudes = np.asarray(exact["source_longitude"].values, dtype=np.float64)
        exact_times = time_keys(exact["time"].values)
        exact_values = np.asarray(exact[target["variable_id"]].values, dtype=np.float64)
        site_coordinates = {
            name: np.asarray(exact[name].values)
            for name in (
                "site",
                "site_id",
                "target_latitude",
                "target_longitude",
                "source_latitude",
                "source_longitude",
                "source_grid_distance_km",
            )
        }
        variable_attrs = dict(exact[target["variable_id"]].attrs)

    mapper = authenticated_mapper(protocol["earth_data_hub"]["dataset_url"], token)
    remote = xr.open_dataset(
        mapper, engine="zarr", chunks={}, zarr_format=3, decode_times=decoder
    )
    try:
        selected_experiment = select_experiment(remote, target["experiment_id"])
        if target["variable_id"] not in selected_experiment:
            raise ValueError("Earth Data Hub dataset does not contain rsds")
        selected, selected_latitudes, selected_longitudes = select_sites(
            selected_experiment,
            target["variable_id"],
            source_latitudes,
            source_longitudes,
        )
        selected = selected.sel(
            time=slice(target["fetch_start"], target["fetch_end"])
        ).load()
    finally:
        remote.close()

    provider_times = time_keys(selected["time"].values)
    if len(provider_times) != int(target["expected_time_count"]):
        raise ValueError(
            f"Earth Data Hub time count mismatch: {len(provider_times)} != "
            f"{target['expected_time_count']}"
        )
    if len(set(provider_times.tolist())) != len(provider_times):
        raise ValueError("Earth Data Hub time axis contains duplicates")
    if provider_times[0] != target["fetch_start"] or provider_times[-1] != target["fetch_end"]:
        raise ValueError("Earth Data Hub time extent does not match the frozen target")
    provider_lookup = {value: index for index, value in enumerate(provider_times)}
    if any(value not in provider_lookup for value in exact_times):
        raise ValueError("Earth Data Hub does not cover every exact-overlap day")
    overlap_indices = np.asarray([provider_lookup[value] for value in exact_times], dtype=int)
    candidate_overlap = np.asarray(selected.values[overlap_indices, :], dtype=np.float64)
    metrics = concordance_metrics(exact_values, candidate_overlap)
    latitude_delta = float(np.max(np.abs(selected_latitudes - source_latitudes)))
    longitude_delta_value = float(
        np.max(longitude_delta(selected_longitudes, source_longitudes))
    )
    checks = concordance_checks(
        metrics,
        protocol,
        site_count=len(site_ids),
        day_count=len(exact_times),
        latitude_delta=latitude_delta,
        longitude_delta_value=longitude_delta_value,
    )
    certification = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["_sha256"],
        "request_id": target["request_id"],
        "provider": "Earth Data Hub",
        "provider_dataset_url_sha256": __import__("hashlib").sha256(
            protocol["earth_data_hub"]["dataset_url"].encode("utf-8")
        ).hexdigest(),
        "source_latitude_max_abs_delta_degrees": latitude_delta,
        "source_longitude_max_abs_delta_degrees": longitude_delta_value,
        "metrics": metrics,
        "checks": checks,
        "credentials_present": True,
        "credentials_or_token_logged": False,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_climate_values_read_for_archive_recovery": True,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    certification_path = audit / "earthdatahub_esgf_overlap_certification.json"
    atomic_json(certification_path, certification)
    if certification["status"] != "PASS":
        raise ValueError("Earth Data Hub exact-overlap concordance certification failed")

    values = np.asarray(selected.values, dtype=np.float32)
    time_attrs = dict(selected["time"].attrs)
    time_encoding = {
        key: value
        for key, value in selected["time"].encoding.items()
        if key in {"units", "calendar"}
    }
    output_dataset = xr.Dataset(
        {target["variable_id"]: (("time", "site"), values)},
        coords={
            "time": ("time", selected["time"].values, time_attrs),
            **{
                name: ("site", value)
                for name, value in site_coordinates.items()
                if name != "site"
            },
            "site": ("site", site_coordinates["site"]),
        },
        attrs={
            "phase6a_protocol_version": protocol["protocol_version"],
            "phase6a_protocol_sha256": protocol["_sha256"],
            "request_id": target["request_id"],
            "transport": protocol["accepted_output_transport"],
            "source_id": target["source_id"],
            "experiment_id": target["experiment_id"],
            "variant_label": target["member_id"],
            "grid_label": target["grid_label"],
            "target_esgf_version": target["esgf_version"],
            "provider_version": protocol["earth_data_hub"]["provider_version"],
            "provider_concordance_certification_sha256": sha256_file(certification_path),
            "future_covariate_matrix": "not_generated",
            "future_prediction": "not_generated",
        },
    )
    output_dataset[target["variable_id"]].attrs.update(variable_attrs)
    output_dataset["time"].encoding.update(time_encoding)
    output_dataset = output_dataset.chunk({"time": 366, "site": 128})
    output = cache / "assets" / target["request_id"][:2] / f"{target['request_id']}.nc"
    receipt_path = cache / "requests" / target["request_id"][:2] / f"{target['request_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.part")
    temporary.unlink(missing_ok=True)
    output_dataset.to_netcdf(
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
    output_dataset.close()
    output_sha256 = sha256_file(temporary)
    output_bytes = temporary.stat().st_size
    os.replace(temporary, output)
    receipt = {
        "status": "FETCHED",
        "request_id": target["request_id"],
        "source_id": target["source_id"],
        "experiment_id": target["experiment_id"],
        "member_id": target["member_id"],
        "variable": target["variable_id"],
        "grid_label": target["grid_label"],
        "version": target["esgf_version"],
        "calendar": target["calendar"],
        "fetch_start": target["fetch_start"],
        "fetch_end": target["fetch_end"],
        "site_count": len(site_ids),
        "time_count": len(provider_times),
        "transport": protocol["accepted_output_transport"],
        "provider_version": protocol["earth_data_hub"]["provider_version"],
        "provider_concordance_certification_sha256": sha256_file(certification_path),
        "output_path": str(output),
        "output_bytes": output_bytes,
        "output_sha256": output_sha256,
        "protocol_sha256": protocol["_sha256"],
        "fetcher_sha256": sha256_file(Path(__file__)),
        "elapsed_seconds": round(time.time() - started, 3),
        "credentials_or_token_logged": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(receipt_path, receipt)
    provenance = {
        **certification,
        "status": "PASS",
        "archive_asset_recovered": True,
        "receipt_sha256": sha256_file(receipt_path),
        "output_sha256": output_sha256,
        "fetcher_sha256": sha256_file(Path(__file__)),
        "elapsed_seconds": receipt["elapsed_seconds"],
    }
    atomic_json(audit / "earthdatahub_gap_recovery_provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    token = os.environ.get(TOKEN_ENVIRONMENT_VARIABLE, "")
    try:
        result = run(
            args.root.resolve(), args.protocol, args.cache, args.audit
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    except Exception as exc:
        print(
            f"Earth Data Hub gap recovery failed: {type(exc).__name__}: "
            f"{sanitized_error(exc, token)}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
