from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from server_training_pipeline.fetch_cmip6_member_resolved import (
    atomic_json,
    atomic_tsv,
    resolve,
    sha256_file,
)
from server_training_pipeline.fit_phase6a_historical_bias_adjustment import VARIABLES


DEFAULT_PROTOCOL = Path("server_training_pipeline/phase6b_future_covariate_protocol_v1.json")
DEFAULT_LOCK = Path(
    "audit/v2/e_projection_core_v1_future_covariates_v1_freeze/"
    "future_covariate_generation_lock.json"
)
DEFAULT_REFERENCE = Path(
    "environment/v2/e_projection_core_v1_historical_daily/cds_bias_reference/"
    "cds_era5_land_1981_2010_daily_reference.nc"
)
DEFAULT_OUTPUT = Path(
    "environment/v2/e_projection_core_v1_future_covariates_v1_reference"
)
DEFAULT_AUDIT = Path(
    "audit/v2/e_projection_core_v1_future_covariates_v1_reference"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = resolve(root, args.protocol)
    lock_path = resolve(root, args.lock)
    reference_path = resolve(root, args.reference)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "PASS_FUTURE_COVARIATE_GENERATION_LOCKED":
        raise ValueError("Phase 6B generation lock has not passed")
    if sha256_file(protocol_path) != lock.get("protocol_sha256"):
        raise ValueError("Phase 6B protocol changed after the generation lock")
    lower_probability = float(protocol["daily_extreme_reference"]["lower_quantile"])
    upper_probability = float(protocol["daily_extreme_reference"]["upper_quantile"])
    decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(reference_path, engine="h5netcdf", decode_times=decoder) as source:
        site_ids = source.site_id.values.astype(str)
        lower = np.full((len(site_ids), len(VARIABLES)), np.nan, dtype=np.float32)
        upper = np.full_like(lower, np.nan)
        observed = np.zeros((len(site_ids), len(VARIABLES)), dtype=np.int32)
        for variable_index, variable in enumerate(VARIABLES):
            values = source[variable].values.astype(np.float64, copy=False)
            observed[:, variable_index] = np.isfinite(values).sum(axis=0)
            with np.errstate(all="ignore"):
                lower[:, variable_index] = np.nanquantile(
                    values, lower_probability, axis=0
                ).astype(np.float32)
                upper[:, variable_index] = np.nanquantile(
                    values, upper_probability, axis=0
                ).astype(np.float32)
            print(f"Historical daily extreme reference {variable_index + 1}/{len(VARIABLES)}", flush=True)
    output = resolve(root, args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Historical daily extreme reference already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset = xr.Dataset(
        {
            "lower_quantile": (("site", "variable"), lower),
            "upper_quantile": (("site", "variable"), upper),
            "observed_day_count": (("site", "variable"), observed),
        },
        coords={
            "site": np.arange(len(site_ids), dtype=np.int32),
            "site_id": ("site", site_ids),
            "variable": list(VARIABLES),
        },
        attrs={
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": sha256_file(protocol_path),
            "generation_lock_sha256": sha256_file(lock_path),
            "reference_source": "CDS_ERA5_Land_1981_2010",
            "reference_sha256": sha256_file(reference_path),
            "lower_probability": lower_probability,
            "upper_probability": upper_probability,
            "future_climate_values_read": "false",
        },
    )
    target = output / "historical_daily_extreme_reference.nc"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    dataset.to_netcdf(temporary, engine="h5netcdf")
    os.replace(temporary, target)
    summary_rows = []
    for variable_index, variable in enumerate(VARIABLES):
        eligible = observed[:, variable_index] > 0
        summary_rows.append(
            {
                "variable": variable,
                "site_count": len(site_ids),
                "eligible_site_count": int(eligible.sum()),
                "missing_site_count": int((~eligible).sum()),
                "minimum_observed_days": int(observed[eligible, variable_index].min()),
                "maximum_observed_days": int(observed[eligible, variable_index].max()),
            }
        )
    audit = resolve(root, args.audit)
    audit.mkdir(parents=True, exist_ok=True)
    summary_path = audit / "historical_daily_extreme_reference_summary.tsv"
    atomic_tsv(summary_path, pd.DataFrame(summary_rows))
    result = {
        "status": "PASS",
        "protocol_version": "phase6b_historical_daily_extreme_reference_v1",
        "protocol_sha256": sha256_file(protocol_path),
        "generation_lock_sha256": sha256_file(lock_path),
        "reference_source_sha256": sha256_file(reference_path),
        "site_count": len(site_ids),
        "variable_count": len(VARIABLES),
        "lower_quantile": lower_probability,
        "upper_quantile": upper_probability,
        "output_path": target.relative_to(root).as_posix(),
        "output_sha256": sha256_file(target),
        "summary_sha256": sha256_file(summary_path),
        "future_climate_values_read": False,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(audit / "historical_daily_extreme_reference_provenance.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
