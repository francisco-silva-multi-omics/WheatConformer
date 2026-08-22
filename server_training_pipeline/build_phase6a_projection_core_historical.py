from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from server_training_pipeline.fetch_cmip6_member_resolved import (
    atomic_json,
    atomic_tsv,
    resolve,
    sha256_file,
)


DEFAULT_PROTOCOL = Path("server_training_pipeline/phase6a_projection_core_feature_contract_v1.json")
DEFAULT_OUTPUT = Path("environment/v2/e_projection_core_v1_historical_backcast")
DEFAULT_AUDIT = Path("audit/v2/phase6a_projection_core_historical_backcast_v1")
ENVIRONMENT_MAP = Path("audit/v2/phase6a_environment_source_contract_v10/environment_daily_request_map.tsv")
CDS_REQUESTS = Path("audit/v2/phase6a_environment_source_contract_v10/cds_era5_land_request_inventory.tsv")
CDS_NORMALIZED = Path("environment/v2/e_projection_core_v1_historical_daily/cds_trial_windows")
SOIL_SITES = Path("audit/v2/phase6a_environment_source_contract_v10/soilgrids_request_inventory.tsv")
SOIL_BASE = Path("environment/v2/phase6a_soilgrids_water_full_v1")
SOIL_RECOVERY = Path("environment/v2/phase6a_soilgrids_missing_resolution_v1")
STATIC_SITES = Path(
    "audit/v2/phase6a_environmental_projection_readiness_v1/backcast/historical_static_site_backcast.parquet"
)

DEPTH_THICKNESS_MM = {
    "0-5cm": 50.0,
    "5-15cm": 100.0,
    "15-30cm": 150.0,
    "30-60cm": 300.0,
    "60-100cm": 400.0,
}


def saturation_vapor_pressure_kpa(temperature_c: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp(17.27 * temperature_c / (temperature_c + 237.3))


def extraterrestrial_radiation_mj_m2_day(latitude_deg: float, dates: Any) -> np.ndarray:
    latitude = math.radians(latitude_deg)
    supplied = np.asarray(dates)
    if np.issubdtype(supplied.dtype, np.number):
        day = supplied.astype(float)
    else:
        day = pd.to_datetime(dates).dt.dayofyear.to_numpy(dtype=float)
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


def fao56_et0(frame: pd.DataFrame, latitude: float) -> np.ndarray:
    required = [
        "tasmin_c",
        "tasmean_c",
        "tasmax_c",
        "solar_radiation_mj_m2_day",
        "relative_humidity_percent",
        "wind_speed_m_s",
        "surface_pressure_pa",
    ]
    values = frame[required].apply(pd.to_numeric, errors="coerce")
    tmin = values.tasmin_c.to_numpy(dtype=float)
    tmean = values.tasmean_c.to_numpy(dtype=float)
    tmax = values.tasmax_c.to_numpy(dtype=float)
    radiation = values.solar_radiation_mj_m2_day.to_numpy(dtype=float)
    humidity = values.relative_humidity_percent.to_numpy(dtype=float)
    wind_2m = values.wind_speed_m_s.to_numpy(dtype=float) * 4.87 / np.log(67.8 * 10.0 - 5.42)
    pressure_kpa = values.surface_pressure_pa.to_numpy(dtype=float) / 1000.0
    if "elevation_m" in frame:
        elevation_supplied = pd.to_numeric(frame.elevation_m, errors="coerce").to_numpy(dtype=float)
        pressure_from_elevation = 101.325 * np.power(
            np.maximum((293.0 - 0.0065 * elevation_supplied) / 293.0, 0.1), 5.26
        )
        pressure_kpa = np.where(np.isfinite(pressure_kpa), pressure_kpa, pressure_from_elevation)
    es = (saturation_vapor_pressure_kpa(tmin) + saturation_vapor_pressure_kpa(tmax)) / 2.0
    ea = saturation_vapor_pressure_kpa(tmean) * np.clip(humidity, 0.0, 100.0) / 100.0
    delta = 4098.0 * saturation_vapor_pressure_kpa(tmean) / np.square(tmean + 237.3)
    gamma = 0.000665 * pressure_kpa
    elevation = 44330.0 * (1.0 - np.power(np.clip(pressure_kpa / 101.325, 0.1, 2.0), 0.1903))
    radiation_axis = frame.day_of_year if "day_of_year" in frame else frame.date
    ra = extraterrestrial_radiation_mj_m2_day(latitude, radiation_axis)
    clear_sky = np.maximum((0.75 + 2e-5 * elevation) * ra, 1e-6)
    net_short = 0.77 * radiation
    sigma = 4.903e-9
    cloud = np.clip(1.35 * np.clip(radiation / clear_sky, 0.0, 1.0) - 0.35, 0.05, 1.0)
    net_long = (
        sigma
        * (np.power(tmax + 273.16, 4) + np.power(tmin + 273.16, 4))
        / 2.0
        * np.maximum(0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0)), 0.05)
        * cloud
    )
    net = net_short - net_long
    numerator = 0.408 * delta * net + gamma * 900.0 / (tmean + 273.0) * wind_2m * (es - ea)
    denominator = delta + gamma * (1.0 + 0.34 * wind_2m)
    result = np.maximum(numerator / denominator, 0.0)
    complete = values.drop(columns="surface_pressure_pa").notna().all(axis=1).to_numpy(copy=True)
    complete &= np.isfinite(pressure_kpa)
    result[~complete] = np.nan
    return result


def available_water_capacity_mm(values: pd.DataFrame) -> float:
    pivot = values.pivot(index="depth", columns="property", values="canonical_value")
    total = 0.0
    for depth, thickness in DEPTH_THICKNESS_MM.items():
        if depth not in pivot.index or not {"wv0033", "wv1500", "cfvo"}.issubset(pivot.columns):
            return np.nan
        field = float(pivot.loc[depth, "wv0033"]) / 100.0
        wilt = float(pivot.loc[depth, "wv1500"]) / 100.0
        coarse = float(pivot.loc[depth, "cfvo"]) / 100.0
        total += max(field - wilt, 0.0) * thickness * np.clip(1.0 - coarse, 0.0, 1.0)
    return float(total)


def load_soil_lookup(root: Path) -> dict[str, dict[str, Any]]:
    sites = pd.read_csv(root / SOIL_SITES, sep="\t", dtype=str)
    base = pd.read_csv(root / SOIL_BASE / "soilgrids_fetch_index.tsv", sep="\t", dtype=str).set_index(
        "site_id"
    )
    recovery = pd.read_csv(
        root / SOIL_RECOVERY / "soilgrids_missing_resolution_index.tsv", sep="\t", dtype=str
    ).set_index("site_id")
    lookup: dict[str, dict[str, Any]] = {}
    for row in sites.itertuples(index=False):
        status = "MASKED_NO_VALID_SOIL_CELL_WITHIN_RADIUS"
        path: Path | None = None
        source = "missing"
        if row.site_id in base.index and str(base.loc[row.site_id].get("values_path", "")) not in {"", "nan"}:
            status = "EXACT_SOIL_CELL"
            path = root / SOIL_BASE / str(base.loc[row.site_id, "values_path"])
            source = "exact"
        elif row.site_id in recovery.index:
            recovered = recovery.loc[row.site_id]
            status = str(recovered.status)
            if status == "RECOVERED_NEAREST_VALID_SOIL_CELL":
                path = root / SOIL_RECOVERY / str(recovered.values_path)
                source = "nearest_within_2km"
        awc = np.nan
        if path is not None and path.is_file():
            awc = available_water_capacity_mm(pd.read_parquet(path))
        lookup[row.site_id] = {
            "soil_status": status,
            "soil_source_class": source,
            "soil_feature_eligible": bool(np.isfinite(awc)),
            "soil_missing_mask": not bool(np.isfinite(awc)),
            "available_water_capacity_mm": awc,
        }
    return lookup


def site_key(latitude: float, longitude: float) -> str:
    return f"{round(float(latitude), 5):.5f}|{round(float(longitude), 5):.5f}"


def summarize_window(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    tmin = frame.tasmin_c.to_numpy(dtype=float)
    tmean = frame.tasmean_c.to_numpy(dtype=float)
    tmax = frame.tasmax_c.to_numpy(dtype=float)
    precipitation = frame.precipitation_mm_day.to_numpy(dtype=float)
    radiation = frame.solar_radiation_mj_m2_day.to_numpy(dtype=float)
    humidity = frame.relative_humidity_percent.to_numpy(dtype=float)
    wind = frame.wind_speed_m_s.to_numpy(dtype=float)
    vpd = frame.vpd_kpa.to_numpy(dtype=float)
    pet = frame.pet_fao56_mm_day.to_numpy(dtype=float)

    def observed_stat(values: np.ndarray, operation: str) -> float:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return np.nan
        return float(getattr(np, operation)(finite))

    def observed_count(values: np.ndarray, predicate: Any) -> float | int:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return np.nan
        return int(np.sum(predicate(finite)))

    return {
        f"{prefix}__day_count": len(frame),
        f"{prefix}__complete_climate_fraction": float(frame.required_climate_complete.mean()),
        f"{prefix}__pet_complete_fraction": float(np.isfinite(pet).mean()),
        f"{prefix}__tasmin_min_c": observed_stat(tmin, "min"),
        f"{prefix}__tasmean_mean_c": observed_stat(tmean, "mean"),
        f"{prefix}__tasmax_max_c": observed_stat(tmax, "max"),
        f"{prefix}__precipitation_sum_mm": observed_stat(precipitation, "sum"),
        f"{prefix}__wet_day_count": observed_count(precipitation, lambda value: value >= 1.0),
        f"{prefix}__dry_day_count": observed_count(precipitation, lambda value: value < 1.0),
        f"{prefix}__radiation_sum_mj_m2": observed_stat(radiation, "sum"),
        f"{prefix}__radiation_mean_mj_m2_day": observed_stat(radiation, "mean"),
        f"{prefix}__relative_humidity_mean_percent": observed_stat(humidity, "mean"),
        f"{prefix}__wind_speed_mean_m_s": observed_stat(wind, "mean"),
        f"{prefix}__gdd_base0_sum": observed_stat(np.maximum(tmean, 0.0), "sum"),
        f"{prefix}__gdd_base5_sum": observed_stat(np.maximum(tmean - 5.0, 0.0), "sum"),
        f"{prefix}__gdd_base10_sum": observed_stat(np.maximum(tmean - 10.0, 0.0), "sum"),
        f"{prefix}__frost_day_count": observed_count(tmin, lambda value: value < 0.0),
        f"{prefix}__heat_day_30_count": observed_count(tmax, lambda value: value >= 30.0),
        f"{prefix}__extreme_heat_day_35_count": observed_count(tmax, lambda value: value >= 35.0),
        f"{prefix}__vpd_mean_kpa": observed_stat(vpd, "mean"),
        f"{prefix}__vpd_max_kpa": observed_stat(vpd, "max"),
        f"{prefix}__high_vpd_day_2kpa_count": observed_count(vpd, lambda value: value >= 2.0),
        f"{prefix}__pet_sum_mm": observed_stat(pet, "sum"),
        f"{prefix}__climatic_water_balance_mm": observed_stat(precipitation - pet, "sum"),
    }


def derive_feature_row(
    daily: pd.DataFrame,
    environment: dict[str, Any],
    soil: dict[str, Any],
    windows: dict[str, list[int]],
) -> dict[str, Any]:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily.date, errors="raise")
    sowing = pd.Timestamp(environment["sowing_date"])
    if "relative_day" not in daily:
        daily["relative_day"] = (daily.date - sowing).dt.days
    if "elevation_m" not in daily:
        daily["elevation_m"] = float(environment["elevation_m"])
    daily["vpd_kpa"] = saturation_vapor_pressure_kpa(daily.tasmean_c.to_numpy(dtype=float)) * (
        1.0 - np.clip(daily.relative_humidity_percent.to_numpy(dtype=float), 0.0, 100.0) / 100.0
    )
    daily["pet_fao56_mm_day"] = fao56_et0(daily, float(environment["latitude"]))
    row: dict[str, Any] = {
        "environment_id": environment["environment_id"],
        "daily_request_id": environment["cds_request_id"],
        "latitude": float(environment["latitude"]),
        "longitude": float(environment["longitude"]),
        "elevation_m": float(environment["elevation_m"]),
        "sowing_date": sowing.strftime("%Y-%m-%d"),
        "sowing_date_observed": True,
        **soil,
        "management_observed": False,
        "water_balance_enabled": False,
        "water_balance_disabled_reason": "explicit_management_scenario_not_supplied"
        if soil["soil_feature_eligible"]
        else "soil_missing_or_not_automatically_eligible",
    }
    all_complete = True
    for prefix, (start, end) in windows.items():
        selected = daily[daily.relative_day.between(start, end)].sort_values("relative_day")
        expected = end - start + 1
        complete = (
            len(selected) == expected
            and selected.relative_day.nunique() == expected
            and int(selected.relative_day.min()) == start
            and int(selected.relative_day.max()) == end
        )
        row[f"{prefix}__window_complete"] = complete
        all_complete &= complete
        if complete:
            row.update(summarize_window(selected, prefix))
    row["all_fixed_windows_complete"] = all_complete
    row["projection_core_climate_eligible"] = bool(
        all_complete
        and all(
            row.get(f"{prefix}__complete_climate_fraction", 0.0) == 1.0
            and row.get(f"{prefix}__pet_complete_fraction", 0.0) == 1.0
            for prefix in windows
        )
    )
    return row


def build_partition(
    root_value: str,
    records: list[dict[str, Any]],
    soil_lookup: dict[str, dict[str, Any]],
    coordinate_to_site: dict[str, str],
    windows: dict[str, list[int]],
) -> list[dict[str, Any]]:
    root = Path(root_value)
    rows = []
    for record in records:
        request_id = record["cds_request_id"]
        daily = pd.read_parquet(root / CDS_NORMALIZED / request_id[:2] / f"{request_id}.parquet")
        site_id = coordinate_to_site[site_key(record["latitude"], record["longitude"])]
        rows.append(derive_feature_row(daily, record, soil_lookup[site_id], windows))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = resolve(root, args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "phase6a_projection_core_feature_v1":
        raise ValueError("Projection-core feature protocol identity mismatch")
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be between 1 and 8")
    environments = pd.read_csv(root / ENVIRONMENT_MAP, sep="\t", dtype=str)
    cds = pd.read_csv(root / CDS_REQUESTS, sep="\t", dtype=str)[
        ["request_id", "source_request_id"]
    ].rename(columns={"request_id": "cds_request_id"})
    environments = environments.merge(
        cds,
        left_on="daily_request_id",
        right_on="source_request_id",
        how="left",
        validate="many_to_one",
    )
    static = pd.read_parquet(root / STATIC_SITES)[["environment_id", "elevation_m"]]
    environments = environments.merge(static, on="environment_id", how="left", validate="one_to_one")
    environments = environments[
        environments.status.eq("READY_TO_FETCH")
        & environments.sowing_date.notna()
        & environments.cds_request_id.notna()
    ].copy()
    if environments.environment_id.duplicated().any():
        raise ValueError("Projection-core environment identities are not unique")
    sites = pd.read_csv(root / SOIL_SITES, sep="\t", dtype=str)
    coordinate_to_site = {
        site_key(row.latitude, row.longitude): row.site_id for row in sites.itertuples(index=False)
    }
    soil_lookup = load_soil_lookup(root)
    records = environments[
        [
            "environment_id",
            "latitude",
            "longitude",
            "elevation_m",
            "sowing_date",
            "cds_request_id",
        ]
    ].to_dict("records")
    partitions = [records[index::args.workers] for index in range(args.workers)]
    rows = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                build_partition,
                str(root),
                partition,
                soil_lookup,
                coordinate_to_site,
                protocol["fixed_windows_days_relative_to_sowing"],
            )
            for partition in partitions
            if partition
        ]
        for future in as_completed(futures):
            result = future.result()
            rows.extend(result)
            completed += len(result)
            print(f"Projection-core ERA5 environments {completed}/{len(records)}", flush=True)
    frame = pd.DataFrame(rows).sort_values("environment_id").reset_index(drop=True)
    output = resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "era5_land_historical_projection_core_features.parquet"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, target)
    schema = pd.DataFrame(
        {
            "feature": frame.columns,
            "dtype": [str(frame[column].dtype) for column in frame.columns],
            "missing_count": [int(frame[column].isna().sum()) for column in frame.columns],
        }
    )
    audit = resolve(root, args.audit)
    audit.mkdir(parents=True, exist_ok=True)
    atomic_tsv(audit / "historical_projection_core_feature_schema.tsv", schema)
    result = {
        "status": "PASS",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(protocol_path),
        "builder_sha256": sha256_file(Path(__file__)),
        "environment_count": len(frame),
        "feature_count": len(frame.columns),
        "all_windows_complete_count": int(frame.all_fixed_windows_complete.sum()),
        "climate_eligible_count": int(frame.projection_core_climate_eligible.sum()),
        "soil_feature_eligible_count": int(frame.soil_feature_eligible.sum()),
        "soil_missing_mask_count": int(frame.soil_missing_mask.sum()),
        "water_balance_enabled_count": int(frame.water_balance_enabled.sum()),
        "output_sha256": sha256_file(target),
        "schema_sha256": sha256_file(audit / "historical_projection_core_feature_schema.tsv"),
        "management_scenario_required_for_soil_water_balance": True,
        "phenotype_values_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(audit / "historical_projection_core_backcast_provenance.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
