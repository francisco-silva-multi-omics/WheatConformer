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

from server_training_pipeline.fetch_cmip6_member_resolved import (
    atomic_json,
    atomic_tsv,
    resolve,
    sha256_file,
)
from server_training_pipeline.normalize_phase6a_historical_daily import (
    normalize_cds_bytes,
)


DEFAULT_CONTRACT = Path("server_training_pipeline/phase6a_daily_normalization_contract_v1.json")
DEFAULT_FETCH_PROTOCOL = Path("server_training_pipeline/phase6a_cds_bias_reference_protocol_v1.json")
DEFAULT_INVENTORY = Path("audit/v2/phase6a_cds_bias_reference_v1/cds_bias_reference_request_inventory.tsv")
DEFAULT_CACHE = Path("environment/v2/phase6a_cds_era5_land_bias_reference_v1")
DEFAULT_OUTPUT = Path("environment/v2/e_projection_core_v1_historical_daily/cds_bias_reference")
DEFAULT_AUDIT = Path("audit/v2/phase6a_daily_normalization_v1/cds_bias_reference")
REFERENCE_VARIABLES = (
    "tasmin_c",
    "tasmean_c",
    "tasmax_c",
    "precipitation_mm_day",
    "solar_radiation_mj_m2_day",
    "relative_humidity_percent",
    "wind_speed_m_s",
    "surface_pressure_pa",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def normalize_one(
    root_value: str,
    cache_value: str,
    output_value: str,
    record: dict[str, Any],
    contract: dict[str, Any],
    contract_sha256: str,
    normalizer_sha256: str,
) -> dict[str, Any]:
    root = Path(root_value)
    cache = Path(cache_value)
    output = Path(output_value)
    request_id = record["request_id"]
    receipt_path = cache / "requests" / request_id[:2] / f"{request_id}.json"
    if not receipt_path.is_file():
        raise ValueError(f"Missing CDS bias-reference receipt: {request_id}")
    receipt = load_json(receipt_path)
    raw = cache / receipt["raw_path"]
    if not raw.is_file() or sha256_file(raw) != receipt["raw_sha256"]:
        raise ValueError(f"CDS bias-reference raw checksum failed: {request_id}")
    target = output / request_id[:2] / f"{request_id}.parquet"
    target_receipt = target.with_suffix(".json")
    if target.is_file() and target_receipt.is_file():
        prior = load_json(target_receipt)
        if (
            prior.get("source_sha256") == receipt["raw_sha256"]
            and prior.get("contract_sha256") == contract_sha256
            and prior.get("normalizer_sha256") == normalizer_sha256
            and prior.get("output_sha256") == sha256_file(target)
        ):
            return prior
    frame = normalize_cds_bytes(raw.read_bytes(), request_id, contract)
    frame.insert(1, "site_id", record["site_id"])
    frame.insert(2, "latitude", float(record["latitude"]))
    frame.insert(3, "longitude", float(record["longitude"]))
    if len(frame) != 10957 or frame.date.min().strftime("%Y-%m-%d") != "1981-01-01" or frame.date.max().strftime("%Y-%m-%d") != "2010-12-31":
        raise ValueError(f"CDS bias-reference daily axis failed: {request_id}")
    atomic_parquet(target, frame)
    result = {
        "status": "PASS",
        "provider": "CDS_ERA5_LAND",
        "dataset": receipt["dataset"],
        "request_id": request_id,
        "site_id": record["site_id"],
        "latitude": record["latitude"],
        "longitude": record["longitude"],
        "source_sha256": receipt["raw_sha256"],
        "raw_path": raw.relative_to(root).as_posix(),
        "raw_bytes": raw.stat().st_size,
        "raw_receipt_path": receipt_path.relative_to(root).as_posix(),
        "raw_receipt_sha256": sha256_file(receipt_path),
        "calendar": "proleptic_gregorian",
        "nominal_start": "1981-01-01",
        "nominal_end": "2010-12-31",
        "internal_start": frame.date.min().strftime("%Y-%m-%d"),
        "internal_end": frame.date.max().strftime("%Y-%m-%d"),
        "daily_rows": len(frame),
        "required_climate_complete_days": int(frame.required_climate_complete.sum()),
        "required_climate_incomplete_days": int((~frame.required_climate_complete).sum()),
        "output_path": target.relative_to(root).as_posix(),
        "output_sha256": sha256_file(target),
        "output_bytes": target.stat().st_size,
        "contract_sha256": contract_sha256,
        "normalizer_sha256": normalizer_sha256,
    }
    atomic_json(target_receipt, result)
    return result


def normalize_partition(*args: Any) -> list[dict[str, Any]]:
    root_value, cache_value, output_value, records, contract, contract_sha256, normalizer_sha256 = args
    return [
        normalize_one(
            root_value,
            cache_value,
            output_value,
            record,
            contract,
            contract_sha256,
            normalizer_sha256,
        )
        for record in records
    ]


def build_reference_cube(root: Path, output: Path, index: pd.DataFrame, contract_sha256: str) -> dict[str, Any]:
    dates = pd.date_range("1981-01-01", "2010-12-31", freq="D")
    shape = (len(dates), len(index))
    arrays = {
        variable: np.full(shape, np.nan, dtype=np.float32) for variable in REFERENCE_VARIABLES
    }
    complete = np.zeros(shape, dtype=np.int8)
    for site_index, row in enumerate(index.itertuples(index=False)):
        frame = pd.read_parquet(
            root / row.output_path,
            columns=["date", *REFERENCE_VARIABLES, "required_climate_complete"],
        )
        observed_dates = pd.to_datetime(frame.date, errors="raise")
        if not observed_dates.equals(pd.Series(dates)):
            raise ValueError(f"Bias-reference normalized site axis changed: {row.site_id}")
        for variable in REFERENCE_VARIABLES:
            arrays[variable][:, site_index] = pd.to_numeric(
                frame[variable], errors="coerce"
            ).to_numpy(dtype=np.float32)
        complete[:, site_index] = frame.required_climate_complete.astype(np.int8)
    dataset = xr.Dataset(
        {
            **{
                variable: (("time", "site"), values)
                for variable, values in arrays.items()
            },
            "required_climate_complete": (("time", "site"), complete),
        },
        coords={
            "time": dates,
            "site": np.arange(len(index), dtype=np.int32),
            "site_id": ("site", index.site_id.astype(str).to_numpy()),
            "latitude": ("site", pd.to_numeric(index.latitude).to_numpy(dtype=float)),
            "longitude": ("site", pd.to_numeric(index.longitude).to_numpy(dtype=float)),
        },
        attrs={
            "protocol_version": "phase6a_cds_bias_reference_daily_normalization_v1",
            "daily_normalization_contract_sha256": contract_sha256,
            "reference_period": "1981-01-01/2010-12-31",
            "future_covariate_matrix": "not_generated",
            "future_prediction": "not_generated",
        },
    )
    target = output / "cds_era5_land_1981_2010_daily_reference.nc"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    encoding = {
        variable: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": (366, 64),
        }
        for variable in dataset.data_vars
    }
    dataset.to_netcdf(temporary, engine="h5netcdf", encoding=encoding)
    os.replace(temporary, target)
    return {
        "reference_cube_path": target.relative_to(root).as_posix(),
        "reference_cube_sha256": sha256_file(target),
        "reference_cube_bytes": target.stat().st_size,
        "reference_cube_time_count": len(dates),
        "reference_cube_site_count": len(index),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--fetch-protocol", type=Path, default=DEFAULT_FETCH_PROTOCOL)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    contract_path = resolve(root, args.contract)
    fetch_protocol_path = resolve(root, args.fetch_protocol)
    inventory_path = resolve(root, args.inventory)
    cache = resolve(root, args.cache)
    output = resolve(root, args.output)
    audit = resolve(root, args.audit)
    contract = load_json(contract_path)
    fetch_protocol = load_json(fetch_protocol_path)
    if contract.get("protocol_version") != "phase6a_daily_normalization_v1":
        raise ValueError("Daily normalization contract identity mismatch")
    if fetch_protocol.get("protocol_version") != "phase6a_cds_bias_reference_v1":
        raise ValueError("CDS bias-reference protocol identity mismatch")
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be between 1 and 8")
    inventory = pd.read_csv(inventory_path, sep="\t", dtype=str).sort_values("request_id")
    if len(inventory) != 907 or inventory.request_id.duplicated().any():
        raise ValueError("CDS bias-reference inventory is not the frozen 907-site grid")
    receipts = list((cache / "requests").glob("*/*.json"))
    if len(receipts) != len(inventory):
        raise ValueError(
            f"CDS bias-reference archive incomplete: receipts={len(receipts)} expected={len(inventory)}"
        )
    contract_sha256 = sha256_file(contract_path)
    normalizer_sha256 = sha256_file(Path(__file__))
    records = inventory.to_dict("records")
    partitions = [records[index::args.workers] for index in range(args.workers)]
    rows: list[dict[str, Any]] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                normalize_partition,
                str(root),
                str(cache),
                str(output),
                partition,
                contract,
                contract_sha256,
                normalizer_sha256,
            )
            for partition in partitions
            if partition
        ]
        for future in as_completed(futures):
            result = future.result()
            rows.extend(result)
            completed += len(result)
            print(f"CDS bias-reference normalization {completed}/{len(records)}", flush=True)
    frame = pd.DataFrame(rows).sort_values("site_id").reset_index(drop=True)
    if int(frame.daily_rows.sum()) != 907 * 10957:
        raise ValueError("CDS bias-reference normalized day count mismatch")
    referenced_raw = {str((root / value).resolve()) for value in frame.raw_path}
    actual_raw = {str(path.resolve()) for path in (cache / "raw").glob("*/*.zip")}
    checks = {
        "one_manifest_row_per_request": len(frame) == 907 and not frame.request_id.duplicated().any(),
        "one_manifest_row_per_site": not frame.site_id.duplicated().any(),
        "one_to_one_raw_asset_paths": not frame.raw_path.duplicated().any(),
        "no_unmanifested_or_missing_raw_assets": referenced_raw == actual_raw,
        "all_internal_daily_axes_complete": bool(frame.daily_rows.astype(int).eq(10957).all()),
        "all_raw_and_output_hashes_present": bool(
            frame.source_sha256.astype(str).str.len().eq(64).all()
            and frame.output_sha256.astype(str).str.len().eq(64).all()
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"CDS bias-reference raw archive certification failed: {checks}")
    audit.mkdir(parents=True, exist_ok=True)
    index_path = audit / "cds_bias_reference_daily_normalization_index.tsv"
    atomic_tsv(index_path, frame)
    cube = build_reference_cube(root, output, frame, contract_sha256)
    provenance = {
        "status": "PASS",
        "protocol_version": "phase6a_cds_bias_reference_daily_normalization_v1",
        "site_count": len(frame),
        "daily_row_count": int(frame.daily_rows.sum()),
        "complete_required_climate_day_count": int(frame.required_climate_complete_days.sum()),
        "incomplete_required_climate_day_count": int(frame.required_climate_incomplete_days.sum()),
        "reference_start": "1981-01-01",
        "reference_end": "2010-12-31",
        "contract_sha256": contract_sha256,
        "fetch_protocol_sha256": sha256_file(fetch_protocol_path),
        "request_inventory_sha256": sha256_file(inventory_path),
        "index_sha256": sha256_file(index_path),
        "raw_archive_checks": checks,
        **cube,
        "phenotype_values_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(audit / "cds_bias_reference_daily_normalization_provenance.json", provenance)
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
