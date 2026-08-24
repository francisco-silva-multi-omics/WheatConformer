from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from server_training_pipeline.certify_phase6a_projection_core_raw_archives import (
    cds_component_frames,
)
from server_training_pipeline.fetch_cmip6_member_resolved import (
    atomic_json,
    atomic_tsv,
    resolve,
    sha256_file,
)


DEFAULT_PROTOCOL = Path(
    "server_training_pipeline/phase6a_daily_normalization_contract_v1.json"
)
DEFAULT_OUTPUT = Path("environment/v2/e_projection_core_v1_historical_daily")
DEFAULT_AUDIT = Path("audit/v2/phase6a_daily_normalization_v1")
CDS_ROOT = Path("environment/v2/phase6a_cds_era5_land_daily_full_v1")
CMIP_ROOT = Path("environment/v2/phase6a_cmip6_member_resolved_daily_v1")
CMIP_MANIFEST = Path(
    "audit/v2/phase6a_projection_core_raw_archive_v1/cmip6_raw_archive_manifest.tsv"
)


def load_protocol(root: Path, path: Path) -> dict[str, Any]:
    resolved = resolve(root, path)
    protocol = json.loads(resolved.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "phase6a_daily_normalization_v1":
        raise ValueError("Daily normalization protocol identity mismatch")
    for relative, expected in protocol["parent_artifacts"].items():
        if sha256_file(resolve(root, Path(relative))) != expected:
            raise ValueError(f"Frozen normalization parent changed: {relative}")
    raw = json.loads(
        (root / "audit/v2/phase6a_projection_core_raw_archive_v1/raw_archive_certification.json").read_text(
            encoding="utf-8"
        )
    )
    allowed_failure = ["continuous_bias_reference_complete"]
    if raw.get("failed_checks") != allowed_failure:
        raise ValueError("Historical normalization requires all currently available raw archives to pass")
    protocol["_path"] = str(resolved)
    protocol["_sha256"] = sha256_file(resolved)
    protocol["_normalizer_sha256"] = sha256_file(Path(__file__))
    return protocol


def saturation_vapor_pressure_kpa(temperature_c: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp(17.27 * temperature_c / (temperature_c + 237.3))


def physical_checks(frame: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, bool]:
    domains = protocol["physical_domains"]
    checks = {}
    for column, bounds in domains.items():
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        checks[f"{column}_has_observed_values"] = bool(len(finite))
        checks[f"{column}_observed_domain"] = bool(
            not len(finite)
            or (finite.min() >= float(bounds[0]) and finite.max() <= float(bounds[1]))
        )
    complete_temperature = frame[["tasmin_c", "tasmean_c", "tasmax_c"]].notna().all(axis=1)
    checks["temperature_order"] = bool(
        (
            frame.loc[complete_temperature, "tasmin_c"]
            <= frame.loc[complete_temperature, "tasmean_c"]
        ).all()
        and (
            frame.loc[complete_temperature, "tasmean_c"]
            <= frame.loc[complete_temperature, "tasmax_c"]
        ).all()
    )
    return checks


def normalize_cds_bytes(payload: bytes, request_id: str, protocol: dict[str, Any]) -> pd.DataFrame:
    components = {
        key: pd.concat(frames, ignore_index=True).sort_values("valid_time").reset_index(drop=True)
        for key, frames in cds_component_frames(payload).items()
    }
    times = pd.to_datetime(components["temperature"].valid_time, errors="raise")
    if any(
        not times.equals(pd.to_datetime(frame.valid_time, errors="raise"))
        for frame in components.values()
    ):
        raise ValueError("CDS component axes disagree during normalization")
    if len(times) % 24 or times.duplicated().any():
        raise ValueError("CDS hourly axis is not a complete set of days")
    dates = times.dt.floor("D")
    day_counts = dates.value_counts(sort=False)
    if not day_counts.eq(24).all():
        raise ValueError("CDS daily normalization requires exactly 24 hourly rows per UTC day")

    temperature = components["temperature"]
    wind = components["wind"]
    radiation = components["radiation"]
    precipitation = components["precipitation"]
    hourly = pd.DataFrame(
        {
            "date": dates,
            "t2m_k": pd.to_numeric(temperature.t2m),
            "d2m_k": pd.to_numeric(temperature.d2m),
            "u10_m_s": pd.to_numeric(wind.u10),
            "v10_m_s": pd.to_numeric(wind.v10),
            "ssrd_j_m2": pd.to_numeric(radiation.ssrd),
            "sp_pa": pd.to_numeric(precipitation.sp),
            "tp_m": pd.to_numeric(precipitation.tp),
        }
    )
    tolerances = protocol["tiny_negative_tolerances"]
    if hourly.tp_m.min() < float(tolerances["CDS_tp_m_per_hour"]):
        raise ValueError("CDS precipitation contains a material negative hourly value")
    if hourly.ssrd_j_m2.min() < float(tolerances["CDS_ssrd_J_m2_per_hour"]):
        raise ValueError("CDS radiation contains a material negative hourly value")
    hourly["tp_nonnegative_m"] = hourly.tp_m.clip(lower=0.0)
    hourly["ssrd_nonnegative_j_m2"] = hourly.ssrd_j_m2.clip(lower=0.0)
    hourly["wind_speed_m_s"] = np.hypot(hourly.u10_m_s, hourly.v10_m_s)
    temperature_c = hourly.t2m_k.to_numpy(dtype=float) - 273.15
    dewpoint_c = hourly.d2m_k.to_numpy(dtype=float) - 273.15
    hourly["relative_humidity_percent"] = np.clip(
        100.0
        * saturation_vapor_pressure_kpa(dewpoint_c)
        / saturation_vapor_pressure_kpa(temperature_c),
        0.0,
        100.0,
    )
    grouped = hourly.groupby("date", sort=True)
    result = pd.DataFrame(
        {
            "request_id": request_id,
            "date": grouped.size().index,
            "raw_t2m_min_k": grouped.t2m_k.min().values,
            "raw_t2m_mean_k": grouped.t2m_k.mean().values,
            "raw_t2m_max_k": grouped.t2m_k.max().values,
            "raw_d2m_mean_k": grouped.d2m_k.mean().values,
            "raw_tp_sum_m": grouped.tp_m.sum().values,
            "raw_ssrd_sum_j_m2": grouped.ssrd_j_m2.sum().values,
            "raw_sp_mean_pa": grouped.sp_pa.mean().values,
            "tasmin_c": grouped.t2m_k.min().values - 273.15,
            "tasmean_c": grouped.t2m_k.mean().values - 273.15,
            "tasmax_c": grouped.t2m_k.max().values - 273.15,
            "precipitation_mm_day": grouped.tp_nonnegative_m.sum().values * 1000.0,
            "solar_radiation_mj_m2_day": grouped.ssrd_nonnegative_j_m2.sum().values * 1e-6,
            "relative_humidity_percent": grouped.relative_humidity_percent.mean().values,
            "wind_speed_m_s": grouped.wind_speed_m_s.mean().values,
            "surface_pressure_pa": grouped.sp_pa.mean().values,
            "tp_tiny_negative_hour_count": grouped.tp_m.apply(lambda value: int((value < 0).sum())).values,
            "ssrd_tiny_negative_hour_count": grouped.ssrd_j_m2.apply(lambda value: int((value < 0).sum())).values,
        }
    )
    required = [
        "tasmin_c",
        "tasmean_c",
        "tasmax_c",
        "precipitation_mm_day",
        "solar_radiation_mj_m2_day",
        "relative_humidity_percent",
        "wind_speed_m_s",
    ]
    for column in required + ["surface_pressure_pa"]:
        result[f"{column}_available"] = result[column].notna()
    result["required_climate_complete"] = result[required].notna().all(axis=1)
    checks = physical_checks(result, protocol)
    required_checks = {
        key: value
        for key, value in checks.items()
        if key.endswith("_observed_domain") or key == "temperature_order"
    }
    if not all(required_checks.values()):
        raise ValueError(
            f"CDS normalized observed-value physical checks failed for {request_id}: {checks}"
        )
    return result


def normalize_one_cds(
    root: Path, output: Path, row: Any, protocol: dict[str, Any]
) -> dict[str, Any]:
    source = root / CDS_ROOT / row.raw_path
    target = output / "cds_trial_windows" / row.request_id[:2] / f"{row.request_id}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = target.with_suffix(".json")
    if target.is_file() and receipt.is_file():
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            metadata.get("protocol_sha256") == protocol["_sha256"]
            and metadata.get("normalizer_sha256") == protocol["_normalizer_sha256"]
            and metadata.get("output_sha256") == sha256_file(target)
        ):
            return metadata
    result = normalize_cds_bytes(source.read_bytes(), row.request_id, protocol)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    result.to_parquet(temporary, index=False, compression="zstd")
    output_sha256 = sha256_file(temporary)
    os.replace(temporary, target)
    metadata = {
        "status": "PASS",
        "request_id": row.request_id,
        "source_sha256": row.raw_sha256,
        "output_path": target.relative_to(root).as_posix(),
        "output_sha256": output_sha256,
        "output_bytes": target.stat().st_size,
        "daily_rows": len(result),
        "first_date": result.date.min().strftime("%Y-%m-%d"),
        "last_date": result.date.max().strftime("%Y-%m-%d"),
        "required_climate_complete_days": int(result.required_climate_complete.sum()),
        "required_climate_incomplete_days": int((~result.required_climate_complete).sum()),
        "protocol_sha256": protocol["_sha256"],
        "normalizer_sha256": protocol["_normalizer_sha256"],
    }
    atomic_json(receipt, metadata)
    return metadata


def normalize_cds_partition(
    root_value: str,
    output_value: str,
    records: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    root = Path(root_value)
    output = Path(output_value)
    return [
        normalize_one_cds(root, output, SimpleNamespace(**record), protocol)
        for record in records
    ]


def normalize_cds(root: Path, output: Path, audit: Path, protocol: dict[str, Any], workers: int) -> dict[str, Any]:
    index = pd.read_csv(root / CDS_ROOT / "cds_era5_land_fetch_index.tsv", sep="\t", dtype=str)
    if workers < 1 or workers > 8:
        raise ValueError("CDS normalization workers must be between 1 and 8")
    records = index.to_dict("records")
    partitions = [records[index::workers] for index in range(workers)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                normalize_cds_partition,
                str(root),
                str(output),
                partition,
                protocol,
            ): len(partition)
            for partition in partitions
            if partition
        }
        completed = 0
        for future in as_completed(futures):
            partition_rows = future.result()
            rows.extend(partition_rows)
            completed += len(partition_rows)
            print(f"CDS daily normalization {completed}/{len(records)}", flush=True)
    frame = pd.DataFrame(rows).sort_values("request_id").reset_index(drop=True)
    audit.mkdir(parents=True, exist_ok=True)
    atomic_tsv(audit / "cds_daily_normalization_index.tsv", frame)
    return {
        "status": "PASS",
        "request_count": len(frame),
        "daily_row_count": int(frame.daily_rows.sum()),
        "index_sha256": sha256_file(audit / "cds_daily_normalization_index.tsv"),
    }


def cmip_date_keys(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{value.year:04d}-{value.month:02d}-{value.day:02d}" for value in values], dtype=str
    )


def normalize_cmip_model(
    root: Path, output: Path, group: pd.DataFrame, protocol: dict[str, Any]
) -> dict[str, Any]:
    group = group.set_index("variable")
    source_id = group.source_id.iloc[0]
    member_id = group.member_id.iloc[0]
    decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    loaded: dict[str, xr.DataArray] = {}
    dates: np.ndarray | None = None
    coordinates: dict[str, xr.DataArray] = {}
    source_hashes = {}
    for variable in ("tasmin", "tas", "tasmax", "pr", "rsds", "hurs", "sfcWind"):
        row = group.loc[variable]
        path = root / row.asset_path
        with xr.open_dataset(path, engine="h5netcdf", decode_times=decoder) as dataset:
            current_dates = cmip_date_keys(dataset.time.values)
            if dates is None:
                dates = current_dates
                coordinates = {
                    name: dataset[name].load()
                    for name in (
                        "time",
                        "site",
                        "site_id",
                        "target_latitude",
                        "target_longitude",
                        "source_latitude",
                        "source_longitude",
                        "source_grid_distance_km",
                    )
                }
            elif not np.array_equal(dates, current_dates):
                raise ValueError(f"CMIP6 variables have different native time axes for {source_id}")
            loaded[variable] = dataset[variable].load()
        source_hashes[variable] = row.asset_sha256
    tolerances = protocol["tiny_negative_tolerances"]
    if float(loaded["pr"].min()) < float(tolerances["CMIP6_pr_kg_m2_s"]):
        raise ValueError(f"CMIP6 {source_id} precipitation has material negative values")
    if float(loaded["rsds"].min()) < float(tolerances["CMIP6_rsds_W_m2"]):
        raise ValueError(f"CMIP6 {source_id} radiation has material negative values")
    dataset = xr.Dataset(
        {
            "tasmin_c": loaded["tasmin"] - 273.15,
            "tasmean_c": loaded["tas"] - 273.15,
            "tasmax_c": loaded["tasmax"] - 273.15,
            "precipitation_mm_day": loaded["pr"].clip(min=0.0) * 86400.0,
            "solar_radiation_mj_m2_day": loaded["rsds"].clip(min=0.0) * 0.0864,
            "relative_humidity_percent": loaded["hurs"].clip(min=0.0, max=100.0),
            "wind_speed_m_s": loaded["sfcWind"].clip(min=0.0),
        },
        coords=coordinates,
        attrs={
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": protocol["_sha256"],
            "source_id": source_id,
            "member_id": member_id,
            "experiment_id": "historical",
            "calendar": group.calendar.iloc[0],
            "source_asset_sha256_json": json.dumps(source_hashes, sort_keys=True),
            "surface_pressure_status": "ABSENT_OPTIONAL_DIAGNOSTIC",
            "raw_values_preserved_in_checksum_bound_parent_assets": "true",
            "future_covariate_matrix": "not_generated",
            "future_prediction": "not_generated",
        },
    )
    for name, unit in protocol["canonical_schema"].items():
        if name in dataset:
            dataset[name].attrs["units"] = unit
    target = output / "cmip6_historical" / f"{source_id}__{member_id}.nc"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    chunks = (min(366, dataset.sizes["time"]), min(128, dataset.sizes["site"]))
    encoding = {
        name: {"zlib": True, "complevel": 4, "shuffle": True, "chunksizes": chunks}
        for name in dataset.data_vars
    }
    dataset.chunk({"time": chunks[0], "site": chunks[1]}).to_netcdf(
        temporary, engine="h5netcdf", encoding=encoding
    )
    output_sha256 = sha256_file(temporary)
    os.replace(temporary, target)
    return {
        "status": "PASS",
        "source_id": source_id,
        "member_id": member_id,
        "calendar": group.calendar.iloc[0],
        "daily_rows": dataset.sizes["time"],
        "site_count": dataset.sizes["site"],
        "first_date": dates[0],
        "last_date": dates[-1],
        "output_path": target.relative_to(root).as_posix(),
        "output_sha256": output_sha256,
        "output_bytes": target.stat().st_size,
    }


def normalize_cmip6(root: Path, output: Path, audit: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    manifest = pd.read_csv(root / CMIP_MANIFEST, sep="\t", dtype=str)
    historical = manifest[manifest.experiment_id.eq("historical")]
    rows = []
    for number, (_, group) in enumerate(historical.groupby(["source_id", "member_id"]), start=1):
        rows.append(normalize_cmip_model(root, output, group, protocol))
        print(f"CMIP6 historical normalization {number}/13", flush=True)
    frame = pd.DataFrame(rows).sort_values("source_id").reset_index(drop=True)
    audit.mkdir(parents=True, exist_ok=True)
    atomic_tsv(audit / "cmip6_historical_normalization_index.tsv", frame)
    return {
        "status": "PASS",
        "model_count": len(frame),
        "asset_count": 91,
        "index_sha256": sha256_file(audit / "cmip6_historical_normalization_index.tsv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("cds", "cmip6", "all"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = load_protocol(root, args.protocol)
    output = resolve(root, args.output)
    audit = resolve(root, args.audit)
    result: dict[str, Any] = {
        "status": "PASS",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["_sha256"],
        "historical_staging_only": True,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    if args.command in {"cds", "all"}:
        result["cds"] = normalize_cds(root, output, audit, protocol, args.workers)
    if args.command in {"cmip6", "all"}:
        result["cmip6"] = normalize_cmip6(root, output, audit, protocol)
    atomic_json(audit / "daily_normalization_provenance.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
