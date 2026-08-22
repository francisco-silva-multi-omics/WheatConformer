from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import cftime
import h5py
import numpy as np
import pandas as pd

from server_training_pipeline.fetch_cmip6_member_resolved import (
    atomic_json,
    atomic_tsv,
    resolve,
    sha256_file,
)


DEFAULT_PROTOCOL = Path(
    "server_training_pipeline/phase6a_projection_core_raw_archive_protocol_v1.json"
)
DEFAULT_OUTPUT = Path("audit/v2/phase6a_projection_core_raw_archive_v1")
CDS_MEMBERS = {
    "wind": {"valid_time", "u10", "v10", "latitude", "longitude"},
    "temperature": {"valid_time", "d2m", "t2m", "latitude", "longitude"},
    "radiation": {"valid_time", "ssrd", "latitude", "longitude"},
    "precipitation": {"valid_time", "sp", "tp", "latitude", "longitude"},
}


def load_protocol(root: Path, path: Path) -> dict[str, Any]:
    resolved = resolve(root, path)
    protocol = json.loads(resolved.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "phase6a_projection_core_raw_archive_v1":
        raise ValueError("Raw archive protocol identity mismatch")
    for relative, expected in protocol["parent_artifacts"].items():
        artifact = resolve(root, Path(relative))
        observed = sha256_file(artifact)
        if observed != expected:
            raise ValueError(f"Frozen parent artifact changed: {relative}")
    protocol["_path"] = str(resolved)
    protocol["_sha256"] = sha256_file(resolved)
    return protocol


def native_date_keys(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{value.year:04d}-{value.month:02d}-{value.day:02d}" for value in values],
        dtype=str,
    )


def native_daily_axis(values: np.ndarray, calendar: str) -> tuple[bool, bool, int]:
    if len(values) == 0:
        return False, False, 0
    numbers = np.asarray(
        cftime.date2num(
            list(values),
            units="days since 0001-01-01 00:00:00",
            calendar=calendar,
        ),
        dtype=float,
    )
    differences = np.diff(numbers)
    unique = len(np.unique(numbers)) == len(numbers)
    daily = bool(unique and np.allclose(differences, 1.0, atol=1e-8, rtol=0))
    expected = int(round(numbers[-1] - numbers[0])) + 1
    return unique, daily, expected


def scalar_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def numeric_daily_axis(
    values: np.ndarray, units: str, calendar: str
) -> tuple[str, str, bool, bool, int]:
    numeric = np.asarray(values, dtype=float)
    if len(numeric) == 0:
        return "", "", False, False, 0
    differences = np.diff(numeric)
    unique = len(np.unique(numeric)) == len(numeric)
    daily = bool(unique and np.allclose(differences, 1.0, atol=1e-8, rtol=0))
    expected = int(round(numeric[-1] - numeric[0])) + 1
    first, last = cftime.num2date(
        [numeric[0], numeric[-1]],
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=True,
    )
    return (
        f"{first.year:04d}-{first.month:02d}-{first.day:02d}",
        f"{last.year:04d}-{last.month:02d}-{last.day:02d}",
        unique,
        daily,
        expected,
    )


def request_id_set(path: Path, suffix: str) -> set[str]:
    return {item.stem for item in path.rglob(f"*.{suffix}")}


def certify_cmip6(root: Path, protocol: dict[str, Any], output: Path) -> pd.DataFrame:
    manifest = pd.read_csv(
        root / "audit/v2/phase6a_cmip6_metadata_inventory_v1/cmip6_selected_asset_manifest.tsv",
        sep="\t",
        dtype=str,
    )
    transport = pd.read_csv(
        root / "audit/v2/phase6a_cmip6_member_resolved_fetch_v1/cmip6_member_resolved_transport_inventory.tsv",
        sep="\t",
        dtype=str,
    )
    cache = resolve(root, Path(protocol["archives"]["cmip6"]))
    expected_ids = set(transport.request_id)
    asset_ids = request_id_set(cache / "assets", "nc")
    receipt_ids = request_id_set(cache / "requests", "json")
    if expected_ids != asset_ids or expected_ids != receipt_ids:
        raise ValueError(
            "CMIP6 request, asset and receipt identifiers are not one-to-one: "
            f"expected={len(expected_ids)} assets={len(asset_ids)} receipts={len(receipt_ids)}"
        )
    manifest_keys = manifest[
        ["source_id", "experiment_id", "member_id", "grid_label", "version", "variable"]
    ].rename(columns={"variable": "variable_id"})
    transport_keys = transport[
        ["source_id", "experiment_id", "member_id", "grid_label", "version", "variable_id"]
    ]
    if len(manifest_keys.merge(transport_keys, how="inner")) != len(manifest):
        raise ValueError("CMIP6 selected manifest and request inventory identities disagree")

    rows: list[dict[str, Any]] = []
    for number, row in enumerate(transport.itertuples(index=False), start=1):
        asset = cache / "assets" / row.request_id[:2] / f"{row.request_id}.nc"
        receipt_path = cache / "requests" / row.request_id[:2] / f"{row.request_id}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        asset_hash = sha256_file(asset)
        receipt_hash = sha256_file(receipt_path)
        with h5py.File(asset, "r") as dataset:
            times = np.asarray(dataset["time"][:])
            time_units = scalar_text(dataset["time"].attrs["units"])
            time_calendar = scalar_text(dataset["time"].attrs.get("calendar", row.calendar))
            internal_start, internal_end, unique, daily, expected_days = numeric_daily_axis(
                times, time_units, time_calendar
            )
            site_ids = np.asarray(
                [scalar_text(value) for value in dataset["site_id"][:]], dtype=str
            )
            target_lat = np.asarray(dataset["target_latitude"][:], dtype=float)
            target_lon = np.asarray(dataset["target_longitude"][:], dtype=float)
            source_lat = np.asarray(dataset["source_latitude"][:], dtype=float)
            source_lon = np.asarray(dataset["source_longitude"][:], dtype=float)
            distance = np.asarray(dataset["source_grid_distance_km"][:], dtype=float)
            identity_pass = all(
                scalar_text(dataset.attrs.get(key, "")) == str(value)
                for key, value in (
                    ("source_id", row.source_id),
                    ("experiment_id", row.experiment_id),
                    ("variant_label", row.member_id),
                    ("grid_label", row.grid_label),
                )
            )
            variable_units = scalar_text(dataset[row.variable_id].attrs.get("units", ""))
            finite_coordinate_pass = bool(
                np.isfinite(target_lat).all()
                and np.isfinite(target_lon).all()
                and np.isfinite(source_lat).all()
                and np.isfinite(source_lon).all()
                and np.isfinite(distance).all()
                and np.all(distance >= 0)
            )
        historical_reference_covered = (
            row.experiment_id != "historical"
            or (
                internal_start <= protocol["reference_period"]["start"]
                and internal_end >= protocol["reference_period"]["end"]
            )
        )
        checks = {
            "receipt_status": receipt.get("status") == "FETCHED",
            "receipt_hash_and_size": (
                receipt.get("output_sha256") == asset_hash
                and int(receipt.get("output_bytes", -1)) == asset.stat().st_size
            ),
            "identity": identity_pass,
            "calendar": row.calendar in protocol["accepted_calendars"],
            "time_unique": unique,
            "time_daily_gap_free": daily,
            "time_count_native_expected": len(times) == expected_days,
            "declared_extent": internal_start == row.fetch_start and internal_end == row.fetch_end,
            "historical_reference_covered": historical_reference_covered,
            "site_count": len(site_ids) == int(protocol["expected"]["sites"]),
            "site_ids_unique": len(np.unique(site_ids)) == len(site_ids),
            "geographic_coverage": finite_coordinate_pass,
            "variable_present": bool(variable_units),
        }
        rows.append(
            {
                "request_id": row.request_id,
                "source_id": row.source_id,
                "institution_id": row.institution_id,
                "experiment_id": row.experiment_id,
                "member_id": row.member_id,
                "grid_label": row.grid_label,
                "version": row.version,
                "variable": row.variable_id,
                "calendar": row.calendar,
                "nominal_start": row.fetch_start,
                "nominal_end": row.fetch_end,
                "internal_start": internal_start,
                "internal_end": internal_end,
                "internal_day_count": len(times),
                "native_expected_day_count": expected_days,
                "site_count": len(site_ids),
                "maximum_grid_distance_km": float(np.max(distance)),
                "variable_units": variable_units,
                "transport": receipt.get("transport", ""),
                "download_status": receipt.get("status", ""),
                "asset_path": asset.relative_to(root).as_posix(),
                "asset_bytes": asset.stat().st_size,
                "asset_sha256": asset_hash,
                "receipt_path": receipt_path.relative_to(root).as_posix(),
                "receipt_sha256": receipt_hash,
                "check_count": len(checks),
                "failed_check_count": sum(not value for value in checks.values()),
                "failed_checks": ";".join(key for key, value in checks.items() if not value),
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )
        if number % 10 == 0 or number == len(transport):
            print(f"CMIP6 raw certification {number}/{len(transport)}", flush=True)
    frame = pd.DataFrame(rows).sort_values("request_id").reset_index(drop=True)
    atomic_tsv(output / "cmip6_raw_archive_manifest.tsv", frame)
    return frame


def cds_component_frames(payload: bytes) -> dict[str, list[pd.DataFrame]]:
    result: dict[str, list[pd.DataFrame]] = {name: [] for name in CDS_MEMBERS}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        files = [member for member in archive.infolist() if not member.is_dir()]
        names = {member.filename for member in files}
        if "manifest.json" in names:
            part_names = sorted(
                name
                for name in names
                if name.startswith("parts/") and name.endswith(".zip")
            )
            if not part_names:
                raise ValueError("Partitioned CDS bundle contains no ZIP parts")
            for part_name in part_names:
                nested = cds_component_frames(archive.read(part_name))
                for key, frames in nested.items():
                    result[key].extend(frames)
            return result
        for member in files:
            frame = pd.read_csv(io.BytesIO(archive.read(member.filename)))
            columns = set(frame.columns)
            matched = [name for name, required in CDS_MEMBERS.items() if required == columns]
            if len(matched) != 1:
                raise ValueError(f"Unexpected CDS member schema: {sorted(columns)}")
            result[matched[0]].append(frame)
    if any(not frames for frames in result.values()):
        raise ValueError("CDS component set is incomplete")
    return result


def inspect_cds_payload(path: Path) -> dict[str, Any]:
    axes = []
    latitudes = []
    longitudes = []
    components = cds_component_frames(path.read_bytes())
    for frames in components.values():
        frame = pd.concat(frames, ignore_index=True)
        times = pd.to_datetime(frame.valid_time, errors="raise")
        if times.duplicated().any() or not times.is_monotonic_increasing:
            raise ValueError(f"CDS hourly axis is duplicate or nonmonotonic: {path}")
        if len(times) > 1 and not (
            times.diff().dropna() == pd.Timedelta(hours=1)
        ).all():
            raise ValueError(f"CDS hourly axis contains gaps: {path}")
        axes.append(times)
        latitudes.append(float(frame.latitude.iloc[0]))
        longitudes.append(float(frame.longitude.iloc[0]))
    first = axes[0]
    if any(not first.equals(axis) for axis in axes[1:]):
        raise ValueError(f"CDS component time axes disagree: {path}")
    return {
        "hour_rows": len(first),
        "first_time": first.iloc[0].isoformat(),
        "last_time": first.iloc[-1].isoformat(),
        "latitude": float(np.mean(latitudes)),
        "longitude": float(np.mean(longitudes)),
    }


def certify_historical_providers(
    root: Path, protocol: dict[str, Any], output: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cds_root = resolve(root, Path(protocol["archives"]["cds_trial_windows"]))
    open_root = resolve(root, Path(protocol["archives"]["openmeteo_trial_windows"]))
    cds = pd.read_csv(cds_root / "cds_era5_land_fetch_index.tsv", sep="\t", dtype=str)
    opened = pd.read_csv(open_root / "daily_request_fetch_index.tsv", sep="\t", dtype=str)
    rows: list[dict[str, Any]] = []
    for number, row in enumerate(cds.itertuples(index=False), start=1):
        raw = cds_root / row.raw_path
        observed_hash = sha256_file(raw)
        payload = inspect_cds_payload(raw)
        requested_days = len(pd.date_range(row.request_start_date, row.request_end_date, freq="D"))
        checks = {
            "index_status": row.status in {"FETCHED_RAW", "CACHED"},
            "hash": observed_hash == row.raw_sha256,
            "bytes": raw.stat().st_size == int(row.raw_bytes),
            "hour_count": payload["hour_rows"] == requested_days * 24,
            "extent": (
                payload["first_time"][:10] == row.request_start_date
                and payload["last_time"][:10] == row.request_end_date
            ),
            "grid_location": (
                abs(payload["latitude"] - float(row.latitude)) <= 0.051
                and abs(((payload["longitude"] - float(row.longitude) + 180) % 360) - 180) <= 0.051
            ),
        }
        rows.append(
            {
                "provider": "CDS_ERA5_LAND",
                "request_id": row.request_id,
                "request_start": row.request_start_date,
                "request_end": row.request_end_date,
                "internal_rows": payload["hour_rows"],
                "internal_start": payload["first_time"],
                "internal_end": payload["last_time"],
                "path": raw.relative_to(root).as_posix(),
                "bytes": raw.stat().st_size,
                "sha256": observed_hash,
                "failed_checks": ";".join(key for key, value in checks.items() if not value),
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )
        if number % 250 == 0 or number == len(cds):
            print(f"CDS trial-window certification {number}/{len(cds)}", flush=True)
    for number, row in enumerate(opened.itertuples(index=False), start=1):
        daily = open_root / row.daily_path
        observed_hash = sha256_file(daily)
        frame = pd.read_parquet(daily, columns=["request_id", "date", "latitude", "longitude"])
        dates = pd.to_datetime(frame.date, errors="raise")
        requested_days = len(pd.date_range(row.request_start_date, row.request_end_date, freq="D"))
        checks = {
            "index_status": row.status in {"FETCHED", "CACHED"},
            "hash": observed_hash == row.daily_sha256,
            "row_count": len(frame) == int(row.daily_rows) == requested_days,
            "time_unique": not dates.duplicated().any(),
            "time_daily_gap_free": (
                dates.is_monotonic_increasing
                and (len(dates) < 2 or (dates.diff().dropna() == pd.Timedelta(days=1)).all())
            ),
            "extent": (
                dates.iloc[0].strftime("%Y-%m-%d") == row.request_start_date
                and dates.iloc[-1].strftime("%Y-%m-%d") == row.request_end_date
            ),
            "identity": frame.request_id.astype(str).eq(row.request_id).all(),
        }
        rows.append(
            {
                "provider": "OPENMETEO_ERA5_DIAGNOSTIC",
                "request_id": row.request_id,
                "request_start": row.request_start_date,
                "request_end": row.request_end_date,
                "internal_rows": len(frame),
                "internal_start": dates.iloc[0].isoformat(),
                "internal_end": dates.iloc[-1].isoformat(),
                "path": daily.relative_to(root).as_posix(),
                "bytes": daily.stat().st_size,
                "sha256": observed_hash,
                "failed_checks": ";".join(key for key, value in checks.items() if not value),
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )
        if number % 500 == 0 or number == len(opened):
            print(f"Open-Meteo trial-window certification {number}/{len(opened)}", flush=True)

    coverage = []
    reference_start = pd.Timestamp(protocol["reference_period"]["start"])
    reference_end = pd.Timestamp(protocol["reference_period"]["end"])
    expected_reference_days = (reference_end - reference_start).days + 1
    cds_numeric = cds.assign(
        start=pd.to_datetime(cds.request_start_date),
        end=pd.to_datetime(cds.request_end_date),
        site=cds.latitude.astype(float).round(5).astype(str)
        + "|"
        + cds.longitude.astype(float).round(5).astype(str),
    )
    for site, group in cds_numeric.groupby("site"):
        intervals = sorted(
            (max(value.start, reference_start), min(value.end, reference_end))
            for value in group.itertuples(index=False)
            if value.end >= reference_start and value.start <= reference_end
        )
        merged: list[list[pd.Timestamp]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1] + pd.Timedelta(days=1):
                merged.append([start, end])
            elif end > merged[-1][1]:
                merged[-1][1] = end
        covered = sum((end - start).days + 1 for start, end in merged)
        coverage.append(covered)
    continuous_reference = (
        len(coverage) == int(protocol["expected"]["sites"])
        and min(coverage, default=0) == expected_reference_days
    )
    frame = pd.DataFrame(rows).sort_values(["provider", "request_id"]).reset_index(drop=True)
    atomic_tsv(output / "historical_provider_raw_archive_manifest.tsv", frame)
    summary = {
        "cds_request_count": len(cds),
        "openmeteo_request_count": len(opened),
        "cds_trial_window_checks_pass": bool(
            frame.loc[frame.provider.eq("CDS_ERA5_LAND"), "status"].eq("PASS").all()
        ),
        "openmeteo_trial_window_checks_pass": bool(
            frame.loc[frame.provider.eq("OPENMETEO_ERA5_DIAGNOSTIC"), "status"].eq("PASS").all()
        ),
        "bias_reference_site_count": len(coverage),
        "bias_reference_expected_days_per_site": expected_reference_days,
        "bias_reference_minimum_days_per_site": min(coverage, default=0),
        "bias_reference_median_days_per_site": float(np.median(coverage)) if coverage else 0,
        "bias_reference_maximum_days_per_site": max(coverage, default=0),
        "bias_reference_complete_sites": sum(value == expected_reference_days for value in coverage),
        "continuous_1981_2010_bias_reference_complete": continuous_reference,
    }
    return frame, summary


def soil_policy(root: Path, protocol: dict[str, Any], output: Path) -> pd.DataFrame:
    index = pd.read_csv(
        root / "environment/v2/phase6a_soilgrids_missing_resolution_v1/soilgrids_missing_resolution_index.tsv",
        sep="\t",
        dtype=str,
    )
    policy = protocol["soil_policy"]
    rows = []
    for status, group in index.groupby("status"):
        action = policy[status]
        accepted = status == "RECOVERED_NEAREST_VALID_SOIL_CELL"
        rows.append(
            {
                "source_status": status,
                "site_count": len(group),
                "environment_count": pd.to_numeric(group.mapped_environment_count).sum(),
                "projection_core_action": action,
                "soil_values_eligible": accepted,
                "explicit_missing_soil_mask": not accepted,
                "soil_dependent_water_balance_enabled": accepted,
                "observations_excluded": 0,
            }
        )
    frame = pd.DataFrame(rows).sort_values("source_status").reset_index(drop=True)
    atomic_tsv(output / "soil_and_management_policy.tsv", frame)
    return frame


def run(root: Path, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    protocol = load_protocol(root, protocol_path)
    output = resolve(root, output_path)
    output.mkdir(parents=True, exist_ok=True)
    cmip = certify_cmip6(root, protocol, output)
    providers, provider_summary = certify_historical_providers(root, protocol, output)
    soil = soil_policy(root, protocol, output)
    expected = protocol["expected"]
    checks = {
        "cmip6_asset_count": len(cmip) == int(expected["cmip6_assets"]),
        "cmip6_all_assets_pass": cmip.status.eq("PASS").all(),
        "cmip6_historical_asset_count": cmip.experiment_id.eq("historical").sum()
        == int(expected["cmip6_historical_assets"]),
        "cmip6_future_asset_count": cmip.experiment_id.ne("historical").sum()
        == int(expected["cmip6_future_assets"]),
        "cmip6_historical_reference_coverage": cmip.loc[
            cmip.experiment_id.eq("historical"), "failed_checks"
        ].map(lambda value: "historical_reference_covered" not in str(value)).all(),
        "cds_trial_windows_complete": provider_summary["cds_trial_window_checks_pass"],
        "openmeteo_trial_windows_complete": provider_summary[
            "openmeteo_trial_window_checks_pass"
        ],
        "soil_policy_terminal": int(soil.site_count.sum()) == 212,
        "soil_policy_no_observation_exclusion": int(soil.observations_excluded.sum()) == 0,
        "continuous_bias_reference_complete": provider_summary[
            "continuous_1981_2010_bias_reference_complete"
        ],
        "no_future_matrices_or_predictions": True,
    }
    raw_archives_pass = all(
        value for key, value in checks.items() if key != "continuous_bias_reference_complete"
    )
    status = (
        "PASS_READY_FOR_DAILY_NORMALIZATION"
        if all(checks.values())
        else (
            "BLOCKED_CONTINUOUS_1981_2010_CDS_BIAS_REFERENCE_MISSING"
            if raw_archives_pass
            else "FAIL_RAW_ARCHIVE_CERTIFICATION"
        )
    )
    artifacts = {}
    for name in (
        "cmip6_raw_archive_manifest.tsv",
        "historical_provider_raw_archive_manifest.tsv",
        "soil_and_management_policy.tsv",
    ):
        artifacts[name] = sha256_file(output / name)
    result = {
        "status": status,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["_sha256"],
        "selection_data": protocol["selection_data"],
        "checks": {key: bool(value) for key, value in checks.items()},
        "failed_checks": [key for key, value in checks.items() if not value],
        "cmip6_assets": len(cmip),
        "cmip6_historical_assets": int(cmip.experiment_id.eq("historical").sum()),
        "cmip6_future_assets": int(cmip.experiment_id.ne("historical").sum()),
        "provider_summary": provider_summary,
        "soil_policy_site_count": int(soil.site_count.sum()),
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
        "artifacts": artifacts,
    }
    atomic_json(output / "raw_archive_certification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.root.resolve(), args.protocol, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"].startswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
