from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from server_training_pipeline.build_phase6a_projection_core_historical import (
    CDS_REQUESTS,
    ENVIRONMENT_MAP,
    SOIL_SITES,
    STATIC_SITES,
    derive_feature_row,
    load_soil_lookup,
    site_key,
)
from server_training_pipeline.fetch_cmip6_member_resolved import atomic_json, atomic_tsv, resolve, sha256_file
from server_training_pipeline.fit_phase6a_historical_bias_adjustment import VARIABLES, transform


DEFAULT_BIAS_PROTOCOL = Path("server_training_pipeline/phase6a_bias_adjustment_contract_v2.json")
DEFAULT_FEATURE_PROTOCOL = Path("server_training_pipeline/phase6a_projection_core_feature_contract_v1.json")
DEFAULT_CMIP_INDEX = Path("audit/v2/phase6a_daily_normalization_v1/cmip6_historical_normalization_index.tsv")
DEFAULT_PARAMETER_INDEX = Path("audit/v2/phase6a_bias_adjustment_v2/bias_adjustment_parameter_index.tsv")
DEFAULT_OUTPUT = Path("environment/v2/e_projection_core_v1_historical_backcast/cmip6_v2")
DEFAULT_AUDIT = Path("audit/v2/phase6a_projection_core_historical_backcast_v2")


def inverse_transform(values: np.ndarray, variable: str) -> np.ndarray:
    if variable in {"tasmin_c", "tasmean_c", "tasmax_c"}:
        return values
    if variable in {"solar_radiation_mj_m2_day", "wind_speed_m_s"}:
        return np.maximum(np.expm1(values), 0.0)
    if variable == "relative_humidity_percent":
        return np.clip(100.0 / (1.0 + np.exp(-values)), 0.0, 100.0)
    if variable == "precipitation_mm_day":
        return np.maximum(values, 0.0)
    raise KeyError(variable)


def apply_frozen_bias(
    values: np.ndarray,
    months: np.ndarray,
    variable: str,
    site_index: int,
    parameters: xr.Dataset,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    months = np.asarray(months, dtype=int)
    corrected = np.full(values.shape, np.nan, dtype=float)
    eligible_rows = np.zeros(values.shape, dtype=bool)
    probabilities = parameters.coords["quantile"].values.astype(float)
    variable_index = int(
        np.flatnonzero(parameters.coords["variable"].values.astype(str) == variable)[0]
    )
    for month in np.unique(months):
        selected = (months == month) & np.isfinite(values)
        parameter_eligible = bool(
            parameters.parameter_eligible.values[site_index, month - 1, variable_index]
        )
        if not parameter_eligible or not selected.any():
            continue
        current = values[selected]
        method_code = int(
            parameters.parameter_method_code.values[
                site_index, month - 1, variable_index
            ]
        )
        if variable == "precipitation_mm_day" and method_code == 2:
            threshold = float(
                parameters.model_wet_threshold_mm.values[site_index, month - 1]
            )
            multiplier = float(
                parameters.precipitation_fallback_multiplier.values[
                    site_index, month - 1
                ]
            )
            if not np.isfinite(threshold) or not np.isfinite(multiplier):
                continue
            adjusted = np.zeros(current.shape, dtype=float)
            wet = current > threshold
            adjusted[wet] = current[wet] * multiplier
            corrected[selected] = adjusted
            eligible_rows[selected] = True
            continue
        model_q = parameters.model_historical_quantile.values[
            site_index, month - 1, variable_index
        ].astype(float)
        reference_q = parameters.reference_historical_quantile.values[
            site_index, month - 1, variable_index
        ].astype(float)
        finite = np.isfinite(model_q) & np.isfinite(reference_q)
        model_q = model_q[finite]
        reference_q = reference_q[finite]
        p_grid = probabilities[finite]
        if len(model_q) < 3:
            continue
        if variable == "precipitation_mm_day":
            threshold = float(parameters.model_wet_threshold_mm.values[site_index, month - 1])
            wet = current > threshold
            adjusted = np.zeros(current.shape, dtype=float)
            if wet.any():
                p = np.interp(current[wet], model_q, p_grid, left=p_grid[0], right=p_grid[-1])
                model_at_p = np.interp(p, p_grid, model_q)
                reference_at_p = np.interp(p, p_grid, reference_q)
                floor = float(protocol["precipitation"]["ratio_floor_mm"])
                cap = float(protocol["precipitation"]["maximum_frozen_multiplier"])
                ratio = np.clip(reference_at_p / np.maximum(model_at_p, floor), 0.0, cap)
                adjusted[wet] = current[wet] * ratio
        else:
            transformed = transform(current, variable, protocol)
            p = np.interp(transformed, model_q, p_grid, left=p_grid[0], right=p_grid[-1])
            model_at_p = np.interp(p, p_grid, model_q)
            reference_at_p = np.interp(p, p_grid, reference_q)
            adjusted = inverse_transform(transformed + reference_at_p - model_at_p, variable)
        corrected[selected] = adjusted
        eligible_rows[selected] = True
    return corrected, eligible_rows


def prepare_environments(root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    environments = pd.read_csv(root / ENVIRONMENT_MAP, sep="\t", dtype=str)
    static = pd.read_parquet(root / STATIC_SITES)[["environment_id", "elevation_m"]]
    environments = environments.merge(static, on="environment_id", how="left", validate="one_to_one")
    environments = environments[
        environments.status.eq("READY_TO_FETCH") & environments.sowing_date.notna()
    ].copy()
    environments["sowing_year"] = pd.to_datetime(environments.sowing_date).dt.year
    environments = environments[environments.sowing_year.between(1981, 2010)].copy()
    sites = pd.read_csv(root / SOIL_SITES, sep="\t", dtype=str)
    coordinate_to_site = {
        site_key(row.latitude, row.longitude): row.site_id for row in sites.itertuples(index=False)
    }
    environments["site_id"] = [
        coordinate_to_site[site_key(row.latitude, row.longitude)]
        for row in environments.itertuples(index=False)
    ]
    if environments.environment_id.duplicated().any():
        raise ValueError("CMIP6 backcast environments are not unique")
    return environments, coordinate_to_site


def build_model(
    root: Path,
    model: Any,
    parameter_row: Any,
    environments: pd.DataFrame,
    soil_lookup: dict[str, dict[str, Any]],
    windows: dict[str, list[int]],
    bias_protocol: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(root / model.output_path, engine="h5netcdf", decode_times=decoder) as climate, xr.open_dataset(
        root / parameter_row.parameter_path, engine="h5netcdf"
    ) as parameters:
        times = climate.time.values
        date_keys = np.asarray(
            [f"{value.year:04d}-{value.month:02d}-{value.day:02d}" for value in times], dtype=str
        )
        day_of_year = climate.time.dt.dayofyear.values.astype(int)
        months_all = climate.time.dt.month.values.astype(int)
        time_lookup = {value: index for index, value in enumerate(date_keys)}
        climate_sites = climate.site_id.values.astype(str)
        parameter_sites = parameters.site_id.values.astype(str)
        if not np.array_equal(climate_sites, parameter_sites):
            raise ValueError(f"Climate/parameter site order differs for {model.source_id}")
        site_lookup = {value: index for index, value in enumerate(climate_sites)}
        rows = []
        unalignable = 0
        for environment in environments.itertuples(index=False):
            sowing_key = pd.Timestamp(environment.sowing_date).strftime("%Y-%m-%d")
            if sowing_key not in time_lookup:
                unalignable += 1
                continue
            sowing_index = time_lookup[sowing_key]
            indices = np.arange(sowing_index - 30, sowing_index + 180)
            if indices.min() < 0 or indices.max() >= len(times):
                unalignable += 1
                continue
            site_index = site_lookup[environment.site_id]
            months = months_all[indices]
            daily = pd.DataFrame(
                {
                    "date": date_keys[indices],
                    "relative_day": np.arange(-30, 180),
                    "day_of_year": day_of_year[indices],
                    "elevation_m": float(environment.elevation_m),
                    "surface_pressure_pa": np.nan,
                }
            )
            all_eligible = np.ones(len(indices), dtype=bool)
            for variable in VARIABLES:
                corrected, eligible = apply_frozen_bias(
                    climate[variable].isel(time=indices, site=site_index).values,
                    months,
                    variable,
                    site_index,
                    parameters,
                    bias_protocol,
                )
                daily[variable] = corrected
                all_eligible &= eligible
            daily["required_climate_complete"] = daily[list(VARIABLES)].notna().all(axis=1)
            environment_record = {
                "environment_id": environment.environment_id,
                "cds_request_id": f"CMIP6::{model.source_id}::{model.member_id}",
                "latitude": environment.latitude,
                "longitude": environment.longitude,
                "elevation_m": environment.elevation_m,
                "sowing_date": environment.sowing_date,
            }
            result = derive_feature_row(
                daily,
                environment_record,
                soil_lookup[environment.site_id],
                windows,
            )
            result["source_id"] = model.source_id
            result["member_id"] = model.member_id
            result["calendar"] = model.calendar
            result["bias_parameter_complete_fraction"] = float(all_eligible.mean())
            rows.append(result)
    frame = pd.DataFrame(rows).sort_values("environment_id").reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{model.source_id}__{model.member_id}.parquet"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, target)
    return {
        "status": "PASS",
        "source_id": model.source_id,
        "member_id": model.member_id,
        "calendar": model.calendar,
        "environment_count": len(frame),
        "unalignable_native_calendar_environment_count": unalignable,
        "climate_eligible_count": int(frame.projection_core_climate_eligible.sum()),
        "output_path": target.relative_to(root).as_posix(),
        "output_sha256": sha256_file(target),
        "output_bytes": target.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--bias-protocol", type=Path, default=DEFAULT_BIAS_PROTOCOL)
    parser.add_argument("--feature-protocol", type=Path, default=DEFAULT_FEATURE_PROTOCOL)
    parser.add_argument("--cmip-index", type=Path, default=DEFAULT_CMIP_INDEX)
    parser.add_argument("--parameter-index", type=Path, default=DEFAULT_PARAMETER_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    root = args.root.resolve()
    bias_path = resolve(root, args.bias_protocol)
    feature_path = resolve(root, args.feature_protocol)
    bias_protocol = json.loads(bias_path.read_text(encoding="utf-8"))
    feature_protocol = json.loads(feature_path.read_text(encoding="utf-8"))
    if bias_protocol.get("protocol_version") != "phase6a_bias_adjustment_v2":
        raise ValueError("Bias protocol mismatch")
    if feature_protocol.get("protocol_version") != "phase6a_projection_core_feature_v1":
        raise ValueError("Feature protocol mismatch")
    parameter_index_path = resolve(root, args.parameter_index)
    if not parameter_index_path.is_file():
        raise ValueError("Historical-only bias parameters must be complete before CMIP6 backcast")
    cmip = pd.read_csv(resolve(root, args.cmip_index), sep="\t", dtype=str)
    parameters = pd.read_csv(parameter_index_path, sep="\t", dtype=str)
    merged = cmip.merge(parameters, on=["source_id", "member_id"], validate="one_to_one")
    if len(merged) != 13:
        raise ValueError("CMIP6 backcast requires all 13 selected historical members")
    environments, _ = prepare_environments(root)
    soil_lookup = load_soil_lookup(root)
    output = resolve(root, args.output)
    rows = []
    for number, row in enumerate(merged.itertuples(index=False), start=1):
        rows.append(
            build_model(
                root,
                row,
                row,
                environments,
                soil_lookup,
                feature_protocol["fixed_windows_days_relative_to_sowing"],
                bias_protocol,
                output,
            )
        )
        print(f"Bias-adjusted historical CMIP6 features {number}/13 source={row.source_id}", flush=True)
    frame = pd.DataFrame(rows)
    audit = resolve(root, args.audit)
    audit.mkdir(parents=True, exist_ok=True)
    index_path = audit / "cmip6_historical_backcast_index.tsv"
    atomic_tsv(index_path, frame)
    result = {
        "status": "PASS",
        "protocol_version": "phase6a_bias_adjusted_cmip6_historical_backcast_v2",
        "source_count": len(frame),
        "environment_union_count": len(environments),
        "feature_protocol_sha256": sha256_file(feature_path),
        "bias_protocol_sha256": sha256_file(bias_path),
        "parameter_index_sha256": sha256_file(parameter_index_path),
        "backcast_index_sha256": sha256_file(index_path),
        "future_SSP_values_read": False,
        "phenotype_values_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(audit / "cmip6_historical_backcast_provenance.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
