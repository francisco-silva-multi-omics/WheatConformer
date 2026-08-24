from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from .final_evaluation_contract import file_sha256


REQUIRED_INVENTORY_COLUMNS = {
    "request_id",
    "request_kind",
    "request_start_date",
    "request_end_date",
    "latitude",
    "longitude",
    "required_daily_variables",
    "request_status",
}
FORBIDDEN_OUTPUT_TOKENS = ("E_REACTION_NORM_RCP", "prediction", "future_matrix")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(root: Path, value: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()


def requested_api_variables(
    row: pd.Series, protocol: dict[str, object]
) -> tuple[list[str], list[str]]:
    api = dict(protocol["api"])
    daily_map = dict(api["daily_variables"])
    hourly_map = dict(api["hourly_variables"])
    requested = {
        token.strip()
        for token in str(row["required_daily_variables"]).split(";")
        if token.strip()
    }
    daily = [value for key, value in daily_map.items() if key in requested]
    hourly = [value for key, value in hourly_map.items() if key in requested]
    if "pr" in requested and daily_map.get("pr") not in daily:
        raise ValueError(f"Request {row['request_id']} cannot map precipitation")
    return daily, hourly


def request_url(row: pd.Series, protocol: dict[str, object]) -> str:
    api = dict(protocol["api"])
    daily, hourly = requested_api_variables(row, protocol)
    params: dict[str, str] = {
        "latitude": f"{float(row['latitude']):.5f}",
        "longitude": f"{float(row['longitude']):.5f}",
        "start_date": str(row["request_start_date"]),
        "end_date": str(row["request_end_date"]),
        "timezone": str(api["timezone"]),
        "models": str(api["model"]),
        "precipitation_unit": str(api["precipitation_unit"]),
    }
    if daily:
        params["daily"] = ",".join(daily)
    if hourly:
        params["hourly"] = ",".join(hourly)
    return str(api["endpoint"]) + "?" + urllib.parse.urlencode(params)


def fetch_json(
    url: str,
    *,
    timeout: int,
    retries: int,
    retry_sleep_seconds: float,
) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "WheatConformer-historical-backcast/1.0"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
        except Exception as exc:
            last_error = exc
        if attempt + 1 < retries:
            time.sleep(retry_sleep_seconds * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}")


def parse_daily_payload(
    payload: dict[str, object],
    row: pd.Series,
    protocol: dict[str, object],
) -> pd.DataFrame:
    daily_variables, hourly_variables = requested_api_variables(row, protocol)
    daily_payload = payload.get("daily")
    if not isinstance(daily_payload, dict) or "time" not in daily_payload:
        raise ValueError("Open-Meteo response has no daily time axis")
    daily = pd.DataFrame(daily_payload)
    daily["date"] = pd.to_datetime(daily["time"], errors="coerce")
    daily = daily.drop(columns=["time"])
    rename = {
        "precipitation_sum": "precipitation_sum_mm",
        "et0_fao_evapotranspiration": "et0_fao_evapotranspiration_mm",
    }
    daily = daily.rename(columns=rename)
    for variable in daily_variables:
        exported = rename.get(variable, variable)
        if exported not in daily:
            raise ValueError(f"Open-Meteo response is missing daily variable {variable}")
        daily[exported] = pd.to_numeric(daily[exported], errors="coerce")

    if hourly_variables:
        hourly_payload = payload.get("hourly")
        if isinstance(hourly_payload, dict) and "time" in hourly_payload:
            hourly = pd.DataFrame(hourly_payload)
            hourly["date"] = pd.to_datetime(hourly["time"], errors="coerce").dt.normalize()
            hourly = hourly.drop(columns=["time"])
            for variable in hourly_variables:
                if variable in hourly:
                    hourly[variable] = pd.to_numeric(hourly[variable], errors="coerce")
            available = [variable for variable in hourly_variables if variable in hourly]
            if available:
                means = hourly.groupby("date", as_index=False)[available].mean()
                means = means.rename(
                    columns={
                        "soil_moisture_0_to_7cm": "soil_moisture_0_7_mean_m3m3",
                        "soil_moisture_7_to_28cm": "soil_moisture_7_28_mean_m3m3",
                    }
                )
                daily = daily.merge(means, on="date", how="left", validate="one_to_one")

    start = pd.Timestamp(str(row["request_start_date"]))
    end = pd.Timestamp(str(row["request_end_date"]))
    expected_days = int((end - start).days) + 1
    response_contract = dict(protocol["response_contract"])
    daily = daily[daily["date"].between(start, end, inclusive="both")].copy()
    if daily["date"].isna().any() or daily["date"].duplicated().any():
        raise ValueError("Daily response contains invalid or duplicate dates")
    coverage = len(daily) / expected_days if expected_days > 0 else 0.0
    if coverage < float(response_contract["minimum_expected_date_coverage_fraction"]):
        raise ValueError(
            f"Daily date coverage is too low: observed={len(daily)} expected={expected_days}"
        )
    if "precipitation_sum_mm" in daily and not np.isfinite(
        daily["precipitation_sum_mm"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Daily precipitation contains nonfinite values")
    if "et0_fao_evapotranspiration_mm" in daily and not np.isfinite(
        daily["et0_fao_evapotranspiration_mm"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Daily ET0 contains nonfinite values")
    daily.insert(0, "request_id", str(row["request_id"]))
    daily.insert(1, "request_kind", str(row["request_kind"]))
    daily["latitude"] = float(row["latitude"])
    daily["longitude"] = float(row["longitude"])
    daily["weather_source"] = "openmeteo_era5"
    return daily.sort_values("date", kind="stable").reset_index(drop=True)


def cache_path(cache_dir: Path, request_id: str) -> Path:
    return cache_dir / request_id[:2] / f"{request_id}.parquet"


def validate_cache(path: Path, row: pd.Series, protocol: dict[str, object]) -> dict[str, object]:
    daily_variables, _ = requested_api_variables(row, protocol)
    required = {"request_id", "date", "precipitation_sum_mm"}
    if "et0_fao_evapotranspiration" in daily_variables:
        required.add("et0_fao_evapotranspiration_mm")
    frame = pd.read_parquet(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cached request lacks columns {sorted(missing)}")
    if frame.empty or frame["request_id"].astype(str).ne(str(row["request_id"])).any():
        raise ValueError("Cached request identity mismatch")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Cached request has invalid or duplicate dates")
    return {
        "data_path": str(path.resolve()),
        "data_sha256": file_sha256(path),
        "daily_rows": len(frame),
        "first_date": dates.min().strftime("%Y-%m-%d"),
        "last_date": dates.max().strftime("%Y-%m-%d"),
        "soil_moisture_columns": int(
            sum(column.startswith("soil_moisture_") for column in frame.columns)
        ),
    }


def write_cache_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a resumable, phenotype-blind daily ERA5 historical backcast archive."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--request-inventory", type=Path, required=True)
    parser.add_argument("--reconstruction-certification", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--retries", type=int)
    parser.add_argument("--retry-sleep", type=float)
    parser.add_argument("--request-sleep", type=float, default=0.1)
    args = parser.parse_args()

    root = args.root.resolve()
    inventory_path = resolve(root, args.request_inventory)
    certification_path = resolve(root, args.reconstruction_certification)
    protocol_path = resolve(root, args.protocol)
    out_dir = resolve(root, args.out_dir)
    cache_dir = out_dir / "requests"
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = read_json(protocol_path)
    certification = read_json(certification_path)
    if protocol.get("status") != "frozen_before_daily_backcast_fetch":
        raise SystemExit("Daily historical backcast protocol is not frozen")
    if dict(protocol.get("api", {})).get("date_clipping_allowed") is not False:
        raise SystemExit("Daily historical backcast protocol permits date clipping")
    if any(
        protocol.get(key) is not False
        for key in (
            "phenotype_values_allowed",
            "outer_test_environment_identifiers_allowed",
            "outer_test_outcomes_allowed",
            "outer_test_metrics_allowed",
            "final_holdout_outcomes_allowed",
            "model_selection_allowed",
            "future_covariate_matrices_allowed",
            "rcp_predictions_allowed",
        )
    ):
        raise SystemExit("Daily historical backcast protocol is not audit-only")
    expected_inventory_sha = dict(certification.get("output_artifacts", {})).get(
        inventory_path.name
    )
    if certification.get("status") != "PASS" or not expected_inventory_sha:
        raise SystemExit("Historical reconstruction certification does not bind the request inventory")
    if file_sha256(inventory_path) != expected_inventory_sha:
        raise SystemExit("Daily request inventory checksum does not match reconstruction certification")

    inventory = pd.read_csv(inventory_path, sep="\t", dtype=str, low_memory=False)
    missing_columns = sorted(REQUIRED_INVENTORY_COLUMNS - set(inventory.columns))
    if missing_columns:
        raise SystemExit(f"Daily request inventory lacks {missing_columns}")
    if inventory["request_id"].isna().any() or inventory["request_id"].duplicated().any():
        raise SystemExit("Daily request inventory IDs are empty or duplicated")
    for column in ("latitude", "longitude"):
        inventory[column] = pd.to_numeric(inventory[column], errors="coerce")
    for column in ("request_start_date", "request_end_date"):
        inventory[f"_{column}"] = pd.to_datetime(inventory[column], errors="coerce")

    execution = dict(protocol["execution"])
    workers = args.workers or int(execution["default_workers"])
    timeout = args.timeout or int(execution["default_timeout_seconds"])
    retries = args.retries or int(execution["default_retries"])
    retry_sleep = (
        args.retry_sleep
        if args.retry_sleep is not None
        else float(execution["default_retry_sleep_seconds"])
    )
    if workers < 1 or timeout < 1 or retries < 1:
        raise SystemExit("Workers, timeout, and retries must be positive")

    ready = inventory["request_status"].eq("READY_TO_FETCH")
    valid_identity = (
        inventory["latitude"].notna()
        & inventory["longitude"].notna()
        & inventory["_request_start_date"].notna()
        & inventory["_request_end_date"].notna()
        & inventory["_request_end_date"].ge(inventory["_request_start_date"])
    )
    coverage_start = pd.Timestamp(str(dict(protocol["api"])["coverage_start_date"]))
    in_coverage = inventory["_request_start_date"].ge(coverage_start)

    cache_records: dict[str, dict[str, object]] = {}
    cache_errors: dict[str, str] = {}
    for _, row in inventory[ready & valid_identity & in_coverage].iterrows():
        path = cache_path(cache_dir, str(row["request_id"]))
        if not path.is_file():
            continue
        try:
            cache_records[str(row["request_id"])] = validate_cache(path, row, protocol)
        except Exception as exc:
            cache_errors[str(row["request_id"])] = f"invalid_existing_cache:{type(exc).__name__}:{exc}"

    pending = inventory[
        ready
        & valid_identity
        & in_coverage
        & ~inventory["request_id"].isin(cache_records)
    ].sort_values("request_id", kind="stable")
    if args.limit is not None:
        if args.limit < 0:
            raise SystemExit("--limit must be nonnegative")
        pending = pending.head(args.limit)

    fetched_records: dict[str, dict[str, object]] = {}
    failures: dict[str, str] = dict(cache_errors)

    def fetch_one(row: pd.Series) -> tuple[str, dict[str, object] | None, str | None]:
        request_id = str(row["request_id"])
        path = cache_path(cache_dir, request_id)
        try:
            url = request_url(row, protocol)
            payload = fetch_json(
                url,
                timeout=timeout,
                retries=retries,
                retry_sleep_seconds=retry_sleep,
            )
            frame = parse_daily_payload(payload, row, protocol)
            write_cache_atomic(frame, path)
            metadata = validate_cache(path, row, protocol)
            metadata["request_url_sha256"] = hashlib.sha256(url.encode("utf-8")).hexdigest()
            return request_id, metadata, None
        except Exception as exc:
            return request_id, None, f"{type(exc).__name__}:{exc}"

    selected_rows = [row for _, row in pending.iterrows()]
    if workers == 1:
        iterator = (fetch_one(row) for row in selected_rows)
        for request_id, metadata, error in iterator:
            if metadata is not None:
                fetched_records[request_id] = metadata
            if error is not None:
                failures[request_id] = error
            if args.request_sleep:
                time.sleep(args.request_sleep)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_one, row) for row in selected_rows]
            for future in as_completed(futures):
                request_id, metadata, error = future.result()
                if metadata is not None:
                    fetched_records[request_id] = metadata
                if error is not None:
                    failures[request_id] = error
                if args.request_sleep:
                    time.sleep(args.request_sleep)

    completed = {**cache_records, **fetched_records}
    unrepaired_cache_errors = set(cache_errors) - set(fetched_records)
    requested_this_run = set(pending["request_id"].astype(str))
    index_rows = []
    for _, row in inventory.iterrows():
        request_id = str(row["request_id"])
        if row["request_status"] != "READY_TO_FETCH":
            status = "BLOCKED_INPUT"
            detail = str(row["request_status"])
        elif not bool(valid_identity.loc[row.name]):
            status = "BLOCKED_INVALID_IDENTITY_OR_DATES"
            detail = "coordinates_or_dates_invalid"
        elif not bool(in_coverage.loc[row.name]):
            status = "OUT_OF_COVERAGE"
            detail = f"start_before_{coverage_start.strftime('%Y-%m-%d')}_not_clamped"
        elif request_id in fetched_records:
            status = "FETCHED"
            detail = ""
        elif request_id in cache_records:
            status = "CACHED"
            detail = ""
        elif request_id in failures:
            status = "FAILED"
            detail = failures[request_id]
        elif request_id not in requested_this_run:
            status = "PENDING_LIMIT"
            detail = "not_selected_in_bounded_run"
        else:
            status = "FAILED"
            detail = "request_completed_without_cache_or_failure_record"
        record = {
            "request_id": request_id,
            "request_kind": row["request_kind"],
            "request_start_date": row["request_start_date"],
            "request_end_date": row["request_end_date"],
            "required_daily_variables": row["required_daily_variables"],
            "status": status,
            "detail": detail,
        }
        record.update(completed.get(request_id, {}))
        index_rows.append(record)
    request_index = pd.DataFrame(index_rows)
    request_index_path = out_dir / "RCP_daily_backcast_request_index.tsv"
    request_index.to_csv(request_index_path, sep="\t", index=False, lineterminator="\n")

    counts = request_index["status"].value_counts().to_dict()
    ready_count = int(ready.sum())
    complete_count = int(request_index["status"].isin(["FETCHED", "CACHED"]).sum())
    archive_complete = complete_count == ready_count
    forbidden_outputs = [
        path.name
        for path in out_dir.iterdir()
        if path.is_file()
        and any(token.lower() in path.name.lower() for token in FORBIDDEN_OUTPUT_TOKENS)
    ]
    checks = {
        "protocol_frozen": protocol.get("status") == "frozen_before_daily_backcast_fetch",
        "reconstruction_certification_pass": certification.get("status") == "PASS",
        "request_inventory_checksum": file_sha256(inventory_path) == expected_inventory_sha,
        "request_ids_unique": inventory["request_id"].nunique() == len(inventory),
        "date_clipping_not_performed": dict(protocol["api"])[
            "date_clipping_allowed"
        ]
        is False,
        "cached_files_validate": not unrepaired_cache_errors,
        "no_future_matrix_or_prediction_generated": not forbidden_outputs,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed_checks = [name for name, value in checks.items() if not value]
    status = "PASS" if not failed_checks else "FAIL"
    provenance = {
        "status": status,
        "run_status": "COMPLETE" if archive_complete else "PARTIAL",
        "protocol_version": protocol["protocol_version"],
        "selection_data": protocol["selection_data"],
        "phenotype_values_read": False,
        "outer_test_environment_identifiers_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_matrix_count_generated": 0,
        "rcp_prediction_count_generated": 0,
        "request_inventory_rows": len(inventory),
        "ready_request_count": ready_count,
        "selected_pending_request_count": len(pending),
        "fetched_this_run": len(fetched_records),
        "cached_before_run": len(cache_records),
        "completed_ready_request_count": complete_count,
        "archive_complete": archive_complete,
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "request_inventory_sha256": file_sha256(inventory_path),
        "reconstruction_certification_sha256": file_sha256(certification_path),
        "protocol_sha256": file_sha256(protocol_path),
        "request_index_sha256": file_sha256(request_index_path),
        "fetcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": checks,
        "failed_checks": failed_checks,
        "future_covariate_population_allowed": False,
        "rcp_predictions_allowed": False,
    }
    provenance_path = out_dir / "RCP_daily_backcast_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2), flush=True)
    if failed_checks:
        raise SystemExit("Daily historical backcast fetch failed provenance checks")


if __name__ == "__main__":
    main()
