from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from server_training_pipeline.build_phase6a_bias_adjusted_cmip6_backcast import (
    apply_frozen_bias,
)
from server_training_pipeline.build_phase6a_projection_core_historical import (
    saturation_vapor_pressure_kpa,
)
from server_training_pipeline.fetch_cmip6_member_resolved import (
    atomic_json,
    atomic_tsv,
    resolve,
    sha256_file,
)
from server_training_pipeline.fit_phase6a_historical_bias_adjustment import VARIABLES


DEFAULT_PROTOCOL = Path("server_training_pipeline/phase6b_future_covariate_protocol_v1.json")
DEFAULT_FEATURE_PROTOCOL = Path("server_training_pipeline/phase6a_projection_core_feature_contract_v1.json")
DEFAULT_LOCK = Path(
    "audit/v2/e_projection_core_v1_future_covariates_v1_freeze/"
    "future_covariate_generation_lock.json"
)
DEFAULT_LOCATIONS = Path(
    "audit/v2/e_projection_core_v1_future_covariates_v1_freeze/"
    "future_projection_location_manifest.tsv"
)
DEFAULT_PLAN = Path(
    "audit/v2/e_projection_core_v1_future_covariates_v1_freeze/"
    "future_covariate_generation_plan.tsv"
)
DEFAULT_EXTREMES = Path(
    "environment/v2/e_projection_core_v1_future_covariates_v1_reference/"
    "historical_daily_extreme_reference.nc"
)
DEFAULT_OUTPUT = Path("environment/v2/e_projection_core_v1_future_covariates_v1")
DEFAULT_AUDIT = Path("audit/v2/e_projection_core_v1_future_covariates_v1")


RAW_VARIABLES = {
    "tasmin_c": "tasmin",
    "tasmean_c": "tas",
    "tasmax_c": "tasmax",
    "precipitation_mm_day": "pr",
    "solar_radiation_mj_m2_day": "rsds",
    "relative_humidity_percent": "hurs",
    "wind_speed_m_s": "sfcWind",
}


def finite_mean(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis, dtype=np.float64)
    return np.divide(total, count, out=np.full(np.shape(total), np.nan), where=count > 0)


def finite_extreme(values: np.ndarray, axis: int, operation: str) -> np.ndarray:
    finite = np.isfinite(values)
    fill = np.inf if operation == "min" else -np.inf
    supplied = np.where(finite, values, fill)
    result = supplied.min(axis=axis) if operation == "min" else supplied.max(axis=axis)
    result[~finite.any(axis=axis)] = np.nan
    return result


def canonicalize(values: np.ndarray, variable: str) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if variable in {"tasmin", "tas", "tasmax"}:
        return values - 273.15
    if variable == "pr":
        return np.maximum(values, 0.0) * 86400.0
    if variable == "rsds":
        return np.maximum(values, 0.0) * 0.0864
    if variable == "hurs":
        return np.clip(values, 0.0, 100.0)
    if variable == "sfcWind":
        return np.maximum(values, 0.0)
    raise KeyError(variable)


def date_keys(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{value.year:04d}-{value.month:02d}-{value.day:02d}" for value in values],
        dtype=str,
    )


def resolve_anchor_key(year: int, month_day: str, lookup: dict[str, int]) -> str | None:
    candidate = f"{year:04d}-{month_day}"
    if candidate in lookup:
        return candidate
    if month_day == "02-29":
        fallback = f"{year:04d}-02-28"
        if fallback in lookup:
            return fallback
    return None


def build_season_indices(
    keys: np.ndarray,
    locations: pd.DataFrame,
    years: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {value: index for index, value in enumerate(keys)}
    indices = np.zeros((len(years), 210, len(locations)), dtype=np.int32)
    valid = np.zeros((len(years), len(locations)), dtype=bool)
    offsets = np.arange(-30, 180, dtype=np.int32)
    for site_index, location in enumerate(locations.itertuples(index=False)):
        anchor = str(location.prospective_sowing_month_day)
        if not anchor or anchor == "nan":
            continue
        for year_index, year in enumerate(years):
            key = resolve_anchor_key(year, anchor, lookup)
            if key is None:
                continue
            supplied = lookup[key] + offsets
            if supplied[0] < 0 or supplied[-1] >= len(keys):
                continue
            indices[year_index, :, site_index] = supplied
            valid[year_index, site_index] = True
    return indices, valid


def extraterrestrial_radiation(latitude_deg: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    latitude = np.radians(latitude_deg)[None, None, :]
    day = day_of_year.astype(np.float64)
    inverse_distance = 1.0 + 0.033 * np.cos(2.0 * np.pi * day / 365.0)
    declination = 0.409 * np.sin(2.0 * np.pi * day / 365.0 - 1.39)
    argument = np.clip(-np.tan(latitude) * np.tan(declination), -1.0, 1.0)
    sunset = np.arccos(argument)
    return (
        24.0
        * 60.0
        / np.pi
        * 0.0820
        * inverse_distance
        * (
            sunset * np.sin(latitude) * np.sin(declination)
            + np.cos(latitude) * np.cos(declination) * np.sin(sunset)
        )
    )


def vectorized_pet(
    climate: dict[str, np.ndarray],
    day_of_year: np.ndarray,
    latitude: np.ndarray,
    elevation: np.ndarray,
) -> np.ndarray:
    tmin = climate["tasmin_c"].astype(np.float64)
    tmean = climate["tasmean_c"].astype(np.float64)
    tmax = climate["tasmax_c"].astype(np.float64)
    radiation = climate["solar_radiation_mj_m2_day"].astype(np.float64)
    humidity = climate["relative_humidity_percent"].astype(np.float64)
    wind = climate["wind_speed_m_s"].astype(np.float64)
    wind_2m = wind * 4.87 / np.log(67.8 * 10.0 - 5.42)
    pressure = 101.325 * np.power(
        np.maximum((293.0 - 0.0065 * elevation) / 293.0, 0.1), 5.26
    )
    pressure = pressure[None, None, :]
    es = (saturation_vapor_pressure_kpa(tmin) + saturation_vapor_pressure_kpa(tmax)) / 2.0
    ea = saturation_vapor_pressure_kpa(tmean) * np.clip(humidity, 0.0, 100.0) / 100.0
    delta = 4098.0 * saturation_vapor_pressure_kpa(tmean) / np.square(tmean + 237.3)
    gamma = 0.000665 * pressure
    ra = extraterrestrial_radiation(latitude, day_of_year)
    clear_sky = np.maximum((0.75 + 2e-5 * elevation[None, None, :]) * ra, 1e-6)
    net_short = 0.77 * radiation
    cloud = np.clip(1.35 * np.clip(radiation / clear_sky, 0.0, 1.0) - 0.35, 0.05, 1.0)
    net_long = (
        4.903e-9
        * (np.power(tmax + 273.16, 4) + np.power(tmin + 273.16, 4))
        / 2.0
        * np.maximum(0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0)), 0.05)
        * cloud
    )
    net = net_short - net_long
    numerator = 0.408 * delta * net + gamma * 900.0 / (tmean + 273.0) * wind_2m * (es - ea)
    denominator = delta + gamma * (1.0 + 0.34 * wind_2m)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.maximum(numerator / denominator, 0.0)
    complete = np.isfinite(tmin) & np.isfinite(tmean) & np.isfinite(tmax)
    complete &= np.isfinite(radiation) & np.isfinite(humidity) & np.isfinite(wind)
    complete &= np.isfinite(pressure)
    result[~complete] = np.nan
    return result.astype(np.float32)


def annual_stat(values: np.ndarray, day_slice: slice, operation: str) -> np.ndarray:
    supplied = values[:, day_slice, :]
    if operation == "mean":
        return finite_mean(supplied, axis=1)
    if operation == "sum":
        finite = np.isfinite(supplied)
        result = np.where(finite, supplied, 0.0).sum(axis=1, dtype=np.float64)
        result[~finite.any(axis=1)] = np.nan
        return result
    if operation in {"min", "max"}:
        return finite_extreme(supplied, axis=1, operation=operation)
    raise KeyError(operation)


def annual_count(values: np.ndarray, day_slice: slice, predicate: Any) -> np.ndarray:
    supplied = values[:, day_slice, :]
    finite = np.isfinite(supplied)
    result = (predicate(supplied) & finite).sum(axis=1).astype(np.float64)
    result[~finite.any(axis=1)] = np.nan
    return result


def aggregate_annual(values: np.ndarray, selected: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    supplied = values[selected].copy()
    supplied[~eligible[selected]] = np.nan
    return finite_mean(supplied, axis=0)


def derive_period_frame(
    locations: pd.DataFrame,
    climate: dict[str, np.ndarray],
    pet: np.ndarray,
    day_extreme: np.ndarray,
    day_extreme_observed: np.ndarray,
    calendar_valid: np.ndarray,
    bias_eligible: np.ndarray,
    period_selector: np.ndarray,
    windows: dict[str, list[int]],
    identity: dict[str, Any],
    minimum_seasons: int,
) -> pd.DataFrame:
    required_complete = np.logical_and.reduce([np.isfinite(climate[name]) for name in VARIABLES])
    season_climate_complete = required_complete.all(axis=1)
    season_pet_complete = np.isfinite(pet).all(axis=1)
    season_bias_complete = bias_eligible.all(axis=1)
    season_eligible = calendar_valid & season_climate_complete & season_pet_complete & season_bias_complete
    eligible_count = season_eligible[period_selector].sum(axis=0)
    calendar_count = calendar_valid[period_selector].sum(axis=0)
    projection_eligible = eligible_count >= minimum_seasons
    rows: dict[str, Any] = {
        "environment_id": [
            f"FUTURE::{identity['source_id']}::{identity['SSP']}::{identity['period']}::{value}"
            for value in locations.location_id
        ],
        "daily_request_id": [f"CMIP6::{identity['matrix_id']}::{value}" for value in locations.location_id],
        "latitude": locations.latitude.to_numpy(dtype=float),
        "longitude": locations.longitude.to_numpy(dtype=float),
        "elevation_m": locations.elevation_m.to_numpy(dtype=float),
        "sowing_date": "PROSPECTIVE::" + locations.prospective_sowing_month_day.fillna(""),
        "sowing_date_observed": False,
        "soil_status": locations.soil_status,
        "soil_source_class": locations.soil_source_class,
        "soil_feature_eligible": locations.soil_feature_eligible.astype(bool),
        "soil_missing_mask": locations.soil_missing_mask.astype(bool),
        "available_water_capacity_mm": locations.available_water_capacity_mm.to_numpy(dtype=float),
        "management_observed": False,
        "water_balance_enabled": False,
        "water_balance_disabled_reason": np.where(
            locations.soil_feature_eligible.astype(bool),
            "explicit_management_scenario_not_supplied",
            "soil_missing_or_not_automatically_eligible",
        ),
    }
    vpd = saturation_vapor_pressure_kpa(climate["tasmean_c"].astype(np.float64)) * (
        1.0 - np.clip(climate["relative_humidity_percent"], 0.0, 100.0) / 100.0
    )
    for prefix, (start, end) in windows.items():
        day_slice = slice(start + 30, end + 31)
        expected = end - start + 1
        rows[f"{prefix}__window_complete"] = calendar_count >= minimum_seasons
        rows[f"{prefix}__day_count"] = float(expected)
        annual_complete = required_complete[:, day_slice, :].mean(axis=1)
        annual_pet_complete = np.isfinite(pet[:, day_slice, :]).mean(axis=1)
        rows[f"{prefix}__complete_climate_fraction"] = aggregate_annual(
            annual_complete, period_selector, calendar_valid
        )
        rows[f"{prefix}__pet_complete_fraction"] = aggregate_annual(
            annual_pet_complete, period_selector, calendar_valid
        )
        annual_features = {
            f"{prefix}__tasmin_min_c": annual_stat(climate["tasmin_c"], day_slice, "min"),
            f"{prefix}__tasmean_mean_c": annual_stat(climate["tasmean_c"], day_slice, "mean"),
            f"{prefix}__tasmax_max_c": annual_stat(climate["tasmax_c"], day_slice, "max"),
            f"{prefix}__precipitation_sum_mm": annual_stat(
                climate["precipitation_mm_day"], day_slice, "sum"
            ),
            f"{prefix}__wet_day_count": annual_count(
                climate["precipitation_mm_day"], day_slice, lambda value: value >= 1.0
            ),
            f"{prefix}__dry_day_count": annual_count(
                climate["precipitation_mm_day"], day_slice, lambda value: value < 1.0
            ),
            f"{prefix}__radiation_sum_mj_m2": annual_stat(
                climate["solar_radiation_mj_m2_day"], day_slice, "sum"
            ),
            f"{prefix}__radiation_mean_mj_m2_day": annual_stat(
                climate["solar_radiation_mj_m2_day"], day_slice, "mean"
            ),
            f"{prefix}__relative_humidity_mean_percent": annual_stat(
                climate["relative_humidity_percent"], day_slice, "mean"
            ),
            f"{prefix}__wind_speed_mean_m_s": annual_stat(
                climate["wind_speed_m_s"], day_slice, "mean"
            ),
            f"{prefix}__gdd_base0_sum": annual_stat(
                np.maximum(climate["tasmean_c"], 0.0), day_slice, "sum"
            ),
            f"{prefix}__gdd_base5_sum": annual_stat(
                np.maximum(climate["tasmean_c"] - 5.0, 0.0), day_slice, "sum"
            ),
            f"{prefix}__gdd_base10_sum": annual_stat(
                np.maximum(climate["tasmean_c"] - 10.0, 0.0), day_slice, "sum"
            ),
            f"{prefix}__frost_day_count": annual_count(
                climate["tasmin_c"], day_slice, lambda value: value < 0.0
            ),
            f"{prefix}__heat_day_30_count": annual_count(
                climate["tasmax_c"], day_slice, lambda value: value >= 30.0
            ),
            f"{prefix}__extreme_heat_day_35_count": annual_count(
                climate["tasmax_c"], day_slice, lambda value: value >= 35.0
            ),
            f"{prefix}__vpd_mean_kpa": annual_stat(vpd, day_slice, "mean"),
            f"{prefix}__vpd_max_kpa": annual_stat(vpd, day_slice, "max"),
            f"{prefix}__high_vpd_day_2kpa_count": annual_count(
                vpd, day_slice, lambda value: value >= 2.0
            ),
            f"{prefix}__pet_sum_mm": annual_stat(pet, day_slice, "sum"),
            f"{prefix}__climatic_water_balance_mm": annual_stat(
                climate["precipitation_mm_day"] - pet, day_slice, "sum"
            ),
        }
        for name, annual in annual_features.items():
            rows[name] = aggregate_annual(annual, period_selector, season_eligible)
    rows["all_fixed_windows_complete"] = calendar_count >= minimum_seasons
    rows["projection_core_climate_eligible"] = projection_eligible
    rows["location_id"] = locations.location_id
    rows["source_id"] = identity["source_id"]
    rows["member_id"] = identity["member_id"]
    rows["experiment_id"] = identity["experiment_id"]
    rows["SSP"] = identity["SSP"]
    rows["period"] = identity["period"]
    rows["period_start_year"] = identity["start_year"]
    rows["period_end_year"] = identity["end_year"]
    rows["calendar"] = identity["calendar"]
    rows["prospective_sowing_month_day"] = locations.prospective_sowing_month_day
    rows["sowing_anchor_status"] = locations.sowing_anchor_status
    rows["calendar_complete_annual_season_count"] = calendar_count
    rows["climate_complete_annual_season_count"] = eligible_count
    rows["bias_parameter_complete_fraction"] = finite_mean(
        bias_eligible[period_selector].astype(float), axis=(0, 1)
    )
    extreme_count = day_extreme[period_selector].sum(axis=(0, 1), dtype=np.int64)
    extreme_observed = day_extreme_observed[period_selector].sum(axis=(0, 1), dtype=np.int64)
    rows["daily_extreme_fraction"] = np.divide(
        extreme_count,
        extreme_observed,
        out=np.full(len(locations), np.nan),
        where=extreme_observed > 0,
    )
    rows["daily_extreme_observed_value_count"] = extreme_observed
    frame = pd.DataFrame(rows)
    climate_columns = [
        column
        for column in frame.columns
        if "__" in column
        and not column.endswith("__window_complete")
        and not column.endswith("__complete_climate_fraction")
        and not column.endswith("__pet_complete_fraction")
        and not column.endswith("__day_count")
    ]
    frame.loc[~projection_eligible, climate_columns] = np.nan
    return frame.sort_values("location_id").reset_index(drop=True)


def load_group_axis(
    root: Path, asset_paths: dict[str, str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first_path = root / asset_paths["tasmin"]
    decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(first_path, engine="h5netcdf", decode_times=decoder) as dataset:
        times = dataset.time.values
        return (
            date_keys(times),
            dataset.time.dt.dayofyear.values.astype(np.int16),
            dataset.time.dt.month.values.astype(np.int8),
            dataset.site_id.values.astype(str),
        )


def build_group(
    root: Path,
    group: pd.DataFrame,
    locations: pd.DataFrame,
    windows: dict[str, list[int]],
    protocol: dict[str, Any],
    extremes_path: Path,
    output: Path,
) -> list[dict[str, Any]]:
    first = group.iloc[0]
    source_id = first.source_id
    member_id = first.member_id
    experiment_id = first.experiment_id
    asset_paths = json.loads(first.raw_asset_paths_json)
    asset_hashes = json.loads(first.raw_asset_sha256_json)
    for variable, relative in asset_paths.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != asset_hashes[variable]:
            raise ValueError(f"Future raw asset changed: {relative}")
    parameter_path = root / first.bias_parameter_path
    if sha256_file(parameter_path) != first.bias_parameter_sha256:
        raise ValueError(f"Bias parameters changed for {source_id}/{member_id}")
    receipt_path = output / "receipts" / f"{source_id}__{member_id}__{experiment_id}.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        cached = True
        for row in receipt.get("matrices", []):
            path = root / row["output_path"]
            cached &= path.is_file() and sha256_file(path) == row["output_sha256"]
        if cached and len(receipt.get("matrices", [])) == 2:
            return receipt["matrices"]
    keys, day_of_year_axis, month_axis, site_ids = load_group_axis(root, asset_paths)
    expected_sites = locations.location_id.to_numpy(dtype=str)
    if not np.array_equal(site_ids, expected_sites):
        raise ValueError(f"Future raw site order differs from the frozen location axis: {source_id}")
    years = []
    period_selectors: dict[str, np.ndarray] = {}
    for period, (start_year, end_year) in protocol["future_periods"].items():
        start = len(years)
        years.extend(range(int(start_year), int(end_year) + 1))
        period_selectors[period] = np.arange(start, len(years))
    indices, calendar_valid = build_season_indices(keys, locations, years)
    site_selector = np.arange(len(locations), dtype=np.int32)[None, None, :]
    valid_days = calendar_valid[:, None, :]
    selected_months = month_axis[indices]
    selected_day_of_year = day_of_year_axis[indices]
    selected_day_of_year[~valid_days.repeat(210, axis=1)] = 1
    decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    climate: dict[str, np.ndarray] = {}
    bias_eligible = np.broadcast_to(valid_days, indices.shape).copy()
    extreme_count = np.zeros(indices.shape, dtype=np.uint8)
    extreme_observed = np.zeros(indices.shape, dtype=np.uint8)
    with xr.open_dataset(parameter_path, engine="h5netcdf") as parameters, xr.open_dataset(
        extremes_path, engine="h5netcdf"
    ) as extremes:
        if not np.array_equal(parameters.site_id.values.astype(str), expected_sites):
            raise ValueError(f"Bias parameter site order differs for {source_id}")
        if not np.array_equal(extremes.site_id.values.astype(str), expected_sites):
            raise ValueError("Historical daily extreme site order differs from future locations")
        extreme_variables = extremes.variable.values.astype(str)
        for canonical, raw_variable in RAW_VARIABLES.items():
            raw_path = root / asset_paths[raw_variable]
            with xr.open_dataset(raw_path, engine="h5netcdf", decode_times=decoder) as dataset:
                current_keys = date_keys(dataset.time.values)
                if not np.array_equal(keys, current_keys):
                    raise ValueError(f"Future variable time axes differ: {source_id}/{experiment_id}")
                raw_values = dataset[raw_variable].values
            gathered = canonicalize(raw_values[indices, site_selector], raw_variable)
            gathered[~np.broadcast_to(valid_days, gathered.shape)] = np.nan
            corrected = np.full(gathered.shape, np.nan, dtype=np.float32)
            current_eligible = np.zeros(gathered.shape, dtype=bool)
            for site_index in range(len(locations)):
                values, eligible = apply_frozen_bias(
                    gathered[:, :, site_index].reshape(-1),
                    selected_months[:, :, site_index].reshape(-1),
                    canonical,
                    site_index,
                    parameters,
                    protocol=protocol["_bias_protocol"],
                )
                corrected[:, :, site_index] = values.reshape(len(years), 210).astype(np.float32)
                current_eligible[:, :, site_index] = eligible.reshape(len(years), 210)
            climate[canonical] = corrected
            bias_eligible &= current_eligible
            extreme_variable_index = int(np.flatnonzero(extreme_variables == canonical)[0])
            lower = extremes.lower_quantile.values[:, extreme_variable_index][None, None, :]
            upper = extremes.upper_quantile.values[:, extreme_variable_index][None, None, :]
            finite = np.isfinite(corrected) & np.isfinite(lower) & np.isfinite(upper)
            extreme_observed += finite.astype(np.uint8)
            extreme_count += (finite & ((corrected < lower) | (corrected > upper))).astype(np.uint8)
            del raw_values, gathered, corrected
    pet = vectorized_pet(
        climate,
        selected_day_of_year,
        locations.latitude.to_numpy(dtype=float),
        locations.elevation_m.to_numpy(dtype=float),
    )
    matrix_rows = []
    for row in group.itertuples(index=False):
        identity = row._asdict()
        frame = derive_period_frame(
            locations,
            climate,
            pet,
            extreme_count,
            extreme_observed,
            calendar_valid,
            bias_eligible,
            period_selectors[row.period],
            windows,
            identity,
            int(protocol["period_aggregation"]["minimum_complete_annual_seasons"]),
        )
        target = (
            output
            / "matrices"
            / source_id
            / member_id
            / experiment_id
            / f"{row.period}.parquet"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, target)
        matrix_rows.append(
            {
                "status": "PASS",
                "matrix_id": row.matrix_id,
                "source_id": source_id,
                "member_id": member_id,
                "experiment_id": experiment_id,
                "SSP": row.SSP,
                "period": row.period,
                "calendar": row.calendar,
                "location_count": len(frame),
                "climate_eligible_location_count": int(
                    frame.projection_core_climate_eligible.astype(bool).sum()
                ),
                "unsupported_anchor_location_count": int(
                    frame.sowing_anchor_status.ne("OBSERVED_SITE_MEDOID").sum()
                ),
                "output_path": target.relative_to(root).as_posix(),
                "output_sha256": sha256_file(target),
                "output_bytes": target.stat().st_size,
            }
        )
    receipt = {
        "status": "PASS",
        "source_id": source_id,
        "member_id": member_id,
        "experiment_id": experiment_id,
        "matrices": matrix_rows,
        "future_predictions_generated": 0,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt_path, receipt)
    return matrix_rows


def build_group_worker(payload: tuple[Any, ...]) -> tuple[str, str, list[dict[str, Any]]]:
    (
        root_value,
        group,
        locations,
        windows,
        protocol,
        extremes_value,
        output_value,
    ) = payload
    rows = build_group(
        Path(root_value),
        group,
        locations,
        windows,
        protocol,
        Path(extremes_value),
        Path(output_value),
    )
    return group.source_id.iloc[0], group.experiment_id.iloc[0], rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--feature-protocol", type=Path, default=DEFAULT_FEATURE_PROTOCOL)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--locations", type=Path, default=DEFAULT_LOCATIONS)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--extremes", type=Path, default=DEFAULT_EXTREMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--only-source", default="")
    parser.add_argument("--only-experiment", default="")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = resolve(root, args.protocol)
    feature_path = resolve(root, args.feature_protocol)
    lock_path = resolve(root, args.lock)
    location_path = resolve(root, args.locations)
    plan_path = resolve(root, args.plan)
    extremes_path = resolve(root, args.extremes)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    feature_protocol = json.loads(feature_path.read_text(encoding="utf-8"))
    bias_protocol_path = root / "server_training_pipeline/phase6a_bias_adjustment_contract_v2.json"
    protocol["_bias_protocol"] = json.loads(bias_protocol_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checksums = {
        "protocol_sha256": (protocol_path, lock["protocol_sha256"]),
        "location_manifest_sha256": (location_path, lock["location_manifest_sha256"]),
        "generation_plan_sha256": (plan_path, lock["generation_plan_sha256"]),
    }
    for name, (path, expected) in checksums.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Phase 6B generation-lock artifact changed: {name}")
    locations = pd.read_csv(location_path, sep="\t", dtype=str)
    for column in ("latitude", "longitude", "elevation_m", "available_water_capacity_mm"):
        locations[column] = pd.to_numeric(locations[column], errors="coerce")
    for column in ("soil_feature_eligible", "soil_missing_mask"):
        locations[column] = locations[column].map({"True": True, "False": False})
    plan = pd.read_csv(plan_path, sep="\t", dtype=str)
    groups = list(plan.groupby(["source_id", "member_id", "experiment_id"], sort=True))
    if args.only_source:
        groups = [item for item in groups if item[0][0] == args.only_source]
    if args.only_experiment:
        groups = [item for item in groups if item[0][2] == args.only_experiment]
    if args.limit_groups > 0:
        groups = groups[: args.limit_groups]
    if args.workers < 1 or args.workers > 3:
        raise ValueError("workers must be between 1 and 3")
    output = resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    completed = []
    payloads = [
        (
            str(root),
            group,
            locations,
            feature_protocol["fixed_windows_days_relative_to_sowing"],
            protocol,
            str(extremes_path),
            str(output),
        )
        for _, group in groups
    ]
    if args.workers == 1:
        for number, payload in enumerate(payloads, start=1):
            source_id, experiment_id, rows = build_group_worker(payload)
            completed.extend(rows)
            print(
                f"Future projection-core groups {number}/{len(groups)} "
                f"source={source_id} experiment={experiment_id}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(build_group_worker, payload) for payload in payloads]
            for number, future in enumerate(as_completed(futures), start=1):
                source_id, experiment_id, rows = future.result()
                completed.extend(rows)
                print(
                    f"Future projection-core groups {number}/{len(groups)} "
                    f"source={source_id} experiment={experiment_id}",
                    flush=True,
                )
    receipt_rows = []
    for receipt_path in sorted((output / "receipts").glob("*.json")):
        receipt_rows.extend(json.loads(receipt_path.read_text(encoding="utf-8"))["matrices"])
    index = pd.DataFrame(receipt_rows).sort_values(
        ["source_id", "experiment_id", "period"]
    ).reset_index(drop=True)
    audit = resolve(root, args.audit)
    audit.mkdir(parents=True, exist_ok=True)
    index_path = audit / "future_covariate_matrix_index.tsv"
    atomic_tsv(index_path, index)
    complete = len(index) == int(protocol["expected_matrix_count"])
    result = {
        "status": "PASS" if complete else "PASS_PARTIAL",
        "run_status": "COMPLETE" if complete else "PARTIAL",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(protocol_path),
        "feature_protocol_sha256": sha256_file(feature_path),
        "bias_protocol_sha256": sha256_file(bias_protocol_path),
        "generation_lock_sha256": sha256_file(lock_path),
        "daily_extreme_reference_sha256": sha256_file(extremes_path),
        "matrix_count": len(index),
        "expected_matrix_count": int(protocol["expected_matrix_count"]),
        "location_count_per_matrix_min": int(index.location_count.astype(int).min()),
        "location_count_per_matrix_max": int(index.location_count.astype(int).max()),
        "matrix_index_sha256": sha256_file(index_path),
        "future_climate_values_read": True,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": len(index),
        "future_predictions_generated": 0,
        "future_prediction_allowed": False,
    }
    atomic_json(audit / "future_covariate_generation_provenance.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
