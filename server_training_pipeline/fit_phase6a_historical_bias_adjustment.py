from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from server_training_pipeline.fetch_cmip6_member_resolved import (
    atomic_json,
    atomic_tsv,
    resolve,
    sha256_file,
)


DEFAULT_PROTOCOL = Path("server_training_pipeline/phase6a_bias_adjustment_contract_v2.json")
DEFAULT_CMIP_INDEX = Path("audit/v2/phase6a_daily_normalization_v1/cmip6_historical_normalization_index.tsv")
DEFAULT_REFERENCE_INDEX = Path(
    "audit/v2/phase6a_daily_normalization_v1/cds_bias_reference/cds_bias_reference_daily_normalization_index.tsv"
)
DEFAULT_REFERENCE_CUBE = Path(
    "environment/v2/e_projection_core_v1_historical_daily/cds_bias_reference/cds_era5_land_1981_2010_daily_reference.nc"
)
DEFAULT_REFERENCE_PROVENANCE = Path(
    "audit/v2/phase6a_daily_normalization_v1/cds_bias_reference/cds_bias_reference_daily_normalization_provenance.json"
)
DEFAULT_OUTPUT = Path("environment/v2/e_projection_core_v1_bias_adjustment_parameters_v2")
DEFAULT_AUDIT = Path("audit/v2/phase6a_bias_adjustment_v2")

VARIABLES = (
    "tasmin_c",
    "tasmean_c",
    "tasmax_c",
    "precipitation_mm_day",
    "solar_radiation_mj_m2_day",
    "relative_humidity_percent",
    "wind_speed_m_s",
)


def load_protocol(root: Path, path: Path) -> dict[str, Any]:
    resolved = resolve(root, path)
    protocol = json.loads(resolved.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "phase6a_bias_adjustment_v2":
        raise ValueError("Bias-adjustment protocol identity mismatch")
    for relative, expected in protocol["parent_artifacts"].items():
        if sha256_file(resolve(root, Path(relative))) != expected:
            raise ValueError(f"Frozen bias-adjustment parent changed: {relative}")
    if protocol.get("future_SSP_values_read_during_parameter_fit") is not False:
        raise ValueError("Bias-adjustment protocol must prohibit future fitting")
    protocol["_sha256"] = sha256_file(resolved)
    return protocol


def transform(values: np.ndarray, variable: str, protocol: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if variable in {"tasmin_c", "tasmean_c", "tasmax_c"}:
        return values
    if variable == "precipitation_mm_day":
        return values
    if variable in {"solar_radiation_mj_m2_day", "wind_speed_m_s"}:
        return np.log1p(np.maximum(values, 0.0))
    if variable == "relative_humidity_percent":
        bounds = protocol["bounded_humidity"]
        fraction = np.clip(
            values,
            float(bounds["lower_percent"]),
            float(bounds["upper_percent"]),
        ) / 100.0
        return np.log(fraction / (1.0 - fraction))
    raise KeyError(variable)


def fit_pair(
    model_values: np.ndarray,
    reference_values: np.ndarray,
    variable: str,
    probabilities: np.ndarray,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    model_values = np.asarray(model_values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    model_values = model_values[np.isfinite(model_values)]
    reference_values = reference_values[np.isfinite(reference_values)]
    minimum = int(protocol["minimum_support"]["observed_days_per_site_month_variable"])
    result: dict[str, Any] = {
        "eligible": False,
        "model_count": len(model_values),
        "reference_count": len(reference_values),
        "model_wet_fraction": np.nan,
        "reference_wet_fraction": np.nan,
        "model_wet_threshold": np.nan,
        "precipitation_fallback_multiplier": np.nan,
        "method_code": 0,
        "model_quantiles": np.full(len(probabilities), np.nan, dtype=np.float32),
        "reference_quantiles": np.full(len(probabilities), np.nan, dtype=np.float32),
    }
    if len(model_values) < minimum or len(reference_values) < minimum:
        return result
    if variable == "precipitation_mm_day":
        wet_threshold = float(protocol["precipitation"]["wet_day_threshold_mm"])
        reference_wet = reference_values[reference_values >= wet_threshold]
        reference_wet_fraction = len(reference_wet) / len(reference_values)
        model_threshold = float(np.quantile(np.maximum(model_values, 0.0), 1.0 - reference_wet_fraction))
        model_wet = model_values[model_values > model_threshold]
        wet_minimum = int(
            protocol["minimum_support"]["wet_days_per_site_month_for_precipitation_quantiles"]
        )
        result.update(
            {
                "model_wet_fraction": len(model_wet) / len(model_values),
                "reference_wet_fraction": reference_wet_fraction,
                "model_wet_threshold": model_threshold,
            }
        )
        if len(model_wet) < wet_minimum or len(reference_wet) < wet_minimum:
            floor = float(protocol["precipitation"]["ratio_floor_mm"])
            cap = float(protocol["precipitation"]["maximum_frozen_multiplier"])
            model_positive = model_values[model_values > 0.0]
            model_mean = float(model_positive.mean()) if len(model_positive) else floor
            reference_mean = float(reference_wet.mean()) if len(reference_wet) else 0.0
            result["precipitation_fallback_multiplier"] = float(
                np.clip(reference_mean / max(model_mean, floor), 0.0, cap)
            )
            result["method_code"] = 2
            result["eligible"] = True
            return result
        model_fit = model_wet
        reference_fit = reference_wet
    else:
        model_fit = transform(model_values, variable, protocol)
        reference_fit = transform(reference_values, variable, protocol)
    result["model_quantiles"] = np.quantile(model_fit, probabilities).astype(np.float32)
    result["reference_quantiles"] = np.quantile(reference_fit, probabilities).astype(np.float32)
    result["method_code"] = 1
    result["eligible"] = True
    return result


def fit_model(
    root: Path,
    row: Any,
    reference_cube: Path,
    output: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    source_path = root / row.output_path
    probabilities = np.asarray(protocol["quantile_probabilities"], dtype=float)
    decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(source_path, engine="h5netcdf", decode_times=decoder) as dataset, xr.open_dataset(
        reference_cube, engine="h5netcdf"
    ) as reference:
        years = dataset.time.dt.year.values
        reference_period = (years >= 1981) & (years <= 2010)
        months = dataset.time.dt.month.values[reference_period]
        model_site_ids = dataset.site_id.values.astype(str)
        reference_site_ids = reference.site_id.values.astype(str)
        if not np.array_equal(model_site_ids, reference_site_ids):
            raise ValueError(f"Reference site alignment failed for {row.source_id}")
        shape = (len(model_site_ids), 12, len(VARIABLES), len(probabilities))
        model_quantiles = np.full(shape, np.nan, dtype=np.float32)
        reference_quantiles = np.full(shape, np.nan, dtype=np.float32)
        eligible = np.zeros(shape[:-1], dtype=np.int8)
        method_code = np.zeros(shape[:-1], dtype=np.int8)
        model_count = np.zeros(shape[:-1], dtype=np.int16)
        reference_count = np.zeros(shape[:-1], dtype=np.int16)
        model_wet_fraction = np.full(shape[:2], np.nan, dtype=np.float32)
        reference_wet_fraction = np.full(shape[:2], np.nan, dtype=np.float32)
        model_wet_threshold = np.full(shape[:2], np.nan, dtype=np.float32)
        precipitation_fallback_multiplier = np.full(shape[:2], np.nan, dtype=np.float32)
        reference_months = reference.time.dt.month.values.astype(int)
        minimum = int(protocol["minimum_support"]["observed_days_per_site_month_variable"])
        wet_minimum = int(
            protocol["minimum_support"]["wet_days_per_site_month_for_precipitation_quantiles"]
        )
        for variable_index, variable in enumerate(VARIABLES):
            model_all = dataset[variable].isel(time=reference_period).values.astype(float)
            reference_all = reference[variable].values.astype(float)
            for month in range(1, 13):
                model_values = model_all[months == month]
                reference_values = reference_all[reference_months == month]
                model_observed = np.isfinite(model_values).sum(axis=0)
                reference_observed = np.isfinite(reference_values).sum(axis=0)
                model_count[:, month - 1, variable_index] = model_observed
                reference_count[:, month - 1, variable_index] = reference_observed
                base_supported = (model_observed >= minimum) & (reference_observed >= minimum)
                supported = base_supported.copy()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if variable == "precipitation_mm_day":
                        wet_threshold = float(protocol["precipitation"]["wet_day_threshold_mm"])
                        reference_wet = np.isfinite(reference_values) & (
                            reference_values >= wet_threshold
                        )
                        reference_fraction = reference_wet.sum(axis=0) / np.maximum(
                            reference_observed, 1
                        )
                        sorted_model = np.sort(
                            np.where(np.isfinite(model_values), np.maximum(model_values, 0.0), np.nan),
                            axis=0,
                        )
                        position = np.clip(1.0 - reference_fraction, 0.0, 1.0) * np.maximum(
                            model_observed - 1, 0
                        )
                        lower = np.floor(position).astype(int)
                        upper = np.ceil(position).astype(int)
                        columns = np.arange(len(model_site_ids))
                        fraction = position - lower
                        threshold = (
                            sorted_model[lower, columns] * (1.0 - fraction)
                            + sorted_model[upper, columns] * fraction
                        )
                        model_wet = np.isfinite(model_values) & (model_values > threshold[None, :])
                        model_wet_count = model_wet.sum(axis=0)
                        reference_wet_count = reference_wet.sum(axis=0)
                        quantile_supported = base_supported & (model_wet_count >= wet_minimum) & (
                            reference_wet_count >= wet_minimum
                        )
                        fallback_supported = base_supported & ~quantile_supported
                        model_fit = np.where(model_wet, model_values, np.nan)
                        reference_fit = np.where(reference_wet, reference_values, np.nan)
                        model_wet_fraction[:, month - 1] = model_wet_count / np.maximum(
                            model_observed, 1
                        )
                        reference_wet_fraction[:, month - 1] = reference_fraction
                        model_wet_threshold[:, month - 1] = threshold
                        model_positive = np.where(
                            np.isfinite(model_values) & (model_values > 0.0),
                            model_values,
                            np.nan,
                        )
                        model_positive_mean = np.nanmean(model_positive, axis=0)
                        reference_wet_mean = np.nanmean(
                            np.where(reference_wet, reference_values, np.nan), axis=0
                        )
                        floor = float(protocol["precipitation"]["ratio_floor_mm"])
                        cap = float(protocol["precipitation"]["maximum_frozen_multiplier"])
                        multiplier = np.clip(
                            np.where(
                                reference_wet_count == 0,
                                0.0,
                                reference_wet_mean
                                / np.maximum(
                                    np.where(
                                        np.isfinite(model_positive_mean),
                                        model_positive_mean,
                                        floor,
                                    ),
                                    floor,
                                ),
                            ),
                            0.0,
                            cap,
                        )
                        precipitation_fallback_multiplier[:, month - 1] = np.where(
                            fallback_supported, multiplier, np.nan
                        )
                        supported = base_supported
                    else:
                        model_fit = transform(model_values, variable, protocol)
                        reference_fit = transform(reference_values, variable, protocol)
                    model_q = np.nanquantile(model_fit, probabilities, axis=0).T
                    reference_q = np.nanquantile(reference_fit, probabilities, axis=0).T
                if variable == "precipitation_mm_day":
                    model_q[~quantile_supported] = np.nan
                    reference_q[~quantile_supported] = np.nan
                    method_code[:, month - 1, variable_index] = np.where(
                        quantile_supported, 1, np.where(fallback_supported, 2, 0)
                    )
                else:
                    model_q[~supported] = np.nan
                    reference_q[~supported] = np.nan
                    method_code[:, month - 1, variable_index] = supported.astype(np.int8)
                model_quantiles[:, month - 1, variable_index] = model_q.astype(np.float32)
                reference_quantiles[:, month - 1, variable_index] = reference_q.astype(np.float32)
                eligible[:, month - 1, variable_index] = supported.astype(np.int8)
        parameters = xr.Dataset(
            {
                "model_historical_quantile": (("site", "month", "variable", "quantile"), model_quantiles),
                "reference_historical_quantile": (("site", "month", "variable", "quantile"), reference_quantiles),
                "parameter_eligible": (("site", "month", "variable"), eligible),
                "parameter_method_code": (("site", "month", "variable"), method_code),
                "model_day_count": (("site", "month", "variable"), model_count),
                "reference_day_count": (("site", "month", "variable"), reference_count),
                "model_wet_fraction": (("site", "month"), model_wet_fraction),
                "reference_wet_fraction": (("site", "month"), reference_wet_fraction),
                "model_wet_threshold_mm": (("site", "month"), model_wet_threshold),
                "precipitation_fallback_multiplier": (
                    ("site", "month"), precipitation_fallback_multiplier
                ),
            },
            coords={
                "site": np.arange(len(model_site_ids), dtype=np.int32),
                "site_id": ("site", model_site_ids),
                "month": np.arange(1, 13, dtype=np.int8),
                "variable": np.asarray(VARIABLES, dtype=str),
                "quantile": probabilities,
            },
            attrs={
                "protocol_version": protocol["protocol_version"],
                "protocol_sha256": protocol["_sha256"],
                "source_id": row.source_id,
                "member_id": row.member_id,
                "reference_period": "1981-01-01/2010-12-31",
                "reference_cube_sha256": sha256_file(reference_cube),
                "future_SSP_values_read": "false",
            },
        )
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{row.source_id}__{row.member_id}.nc"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    parameters.to_netcdf(temporary, engine="h5netcdf")
    os.replace(temporary, target)
    return {
        "status": "PASS",
        "source_id": row.source_id,
        "member_id": row.member_id,
        "site_count": len(model_site_ids),
        "eligible_parameter_cells": int(eligible.sum()),
        "ineligible_parameter_cells": int((eligible == 0).sum()),
        "quantile_parameter_cells": int((method_code == 1).sum()),
        "dry_month_fallback_parameter_cells": int((method_code == 2).sum()),
        "parameter_path": target.relative_to(root).as_posix(),
        "parameter_sha256": sha256_file(target),
        "parameter_bytes": target.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cmip-index", type=Path, default=DEFAULT_CMIP_INDEX)
    parser.add_argument("--reference-index", type=Path, default=DEFAULT_REFERENCE_INDEX)
    parser.add_argument("--reference-cube", type=Path, default=DEFAULT_REFERENCE_CUBE)
    parser.add_argument(
        "--reference-provenance", type=Path, default=DEFAULT_REFERENCE_PROVENANCE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = load_protocol(root, args.protocol)
    cmip = pd.read_csv(resolve(root, args.cmip_index), sep="\t", dtype=str)
    reference_path = resolve(root, args.reference_index)
    if not reference_path.is_file():
        raise ValueError("Complete normalized CDS 1981-2010 reference is required")
    reference = pd.read_csv(reference_path, sep="\t", dtype=str)
    reference_cube = resolve(root, args.reference_cube)
    if not reference_cube.is_file():
        raise ValueError("Complete consolidated CDS 1981-2010 reference cube is required")
    reference_provenance_path = resolve(root, args.reference_provenance)
    if not reference_provenance_path.is_file():
        raise ValueError("CDS 1981-2010 reference normalization provenance is required")
    reference_provenance = json.loads(reference_provenance_path.read_text(encoding="utf-8"))
    if (
        reference_provenance.get("status") != "PASS"
        or reference_provenance.get("reference_cube_sha256") != sha256_file(reference_cube)
        or reference_provenance.get("site_count") != 907
    ):
        raise ValueError("CDS 1981-2010 reference normalization certification failed")
    if len(cmip) != 13 or len(reference) != 907 or reference.site_id.duplicated().any():
        raise ValueError("Bias-adjustment model/site grid is incomplete")
    output = resolve(root, args.output)
    rows = []
    for number, row in enumerate(cmip.sort_values("source_id").itertuples(index=False), start=1):
        rows.append(fit_model(root, row, reference_cube, output, protocol))
        print(f"Bias parameters {number}/13 source={row.source_id}", flush=True)
    frame = pd.DataFrame(rows)
    audit = resolve(root, args.audit)
    audit.mkdir(parents=True, exist_ok=True)
    index_path = audit / "bias_adjustment_parameter_index.tsv"
    atomic_tsv(index_path, frame)
    provenance = {
        "status": "PASS",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["_sha256"],
        "source_count": len(frame),
        "site_count": 907,
        "reference_period": ["1981-01-01", "2010-12-31"],
        "reference_cube_sha256": sha256_file(reference_cube),
        "reference_normalization_provenance_sha256": sha256_file(
            reference_provenance_path
        ),
        "parameter_index_sha256": sha256_file(index_path),
        "future_SSP_values_read": False,
        "phenotype_values_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(audit / "bias_adjustment_provenance.json", provenance)
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
