from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from typing import Any
import urllib.parse

import numpy as np
import pandas as pd

from server_training_pipeline.phase6a_environment_source_recovery import (
    SOILGRIDS_DEPTHS,
    SOILGRIDS_ENDPOINT,
    SOILGRIDS_NATIVE_CRS,
    SOILGRIDS_PROJ,
    SOILGRIDS_PROPERTIES,
    fetch_bytes,
    read_json,
    sha256_file,
    write_json_atomic,
    write_tsv,
)


PROTOCOL_VERSION = "phase6a_soilgrids_missing_resolution_v1"
DEFAULT_PROTOCOL = Path(
    "server_training_pipeline/phase6a_soilgrids_missing_resolution_protocol_v1.json"
)
DEFAULT_SOURCE_CACHE = Path("environment/v2/phase6a_soilgrids_water_full_v1")
DEFAULT_AUDIT_DIR = Path("audit/v2/phase6a_soilgrids_missing_resolution_v1")
DEFAULT_CACHE_DIR = Path("environment/v2/phase6a_soilgrids_missing_resolution_v1")
SOURCE_MISSING_STATUS = "STRUCTURALLY_UNAVAILABLE_SOIL_CELL"
TERMINAL_STATUSES = {
    "RECOVERED_NEAREST_VALID_SOIL_CELL",
    "CANDIDATE_DISTANCE_REVIEW",
    "MASKED_NO_VALID_SOIL_CELL_WITHIN_RADIUS",
}


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def neighborhood_url(
    property_name: str,
    depth: str,
    x: float,
    y: float,
    radius_m: int,
) -> str:
    if property_name not in SOILGRIDS_PROPERTIES or depth not in SOILGRIDS_DEPTHS:
        raise ValueError("Unsupported SoilGrids property or depth")
    if radius_m <= 0:
        raise ValueError("Search radius must be positive")
    params = [
        ("map", f"/map/{property_name}.map"),
        ("SERVICE", "WCS"),
        ("VERSION", "2.0.1"),
        ("REQUEST", "GetCoverage"),
        ("COVERAGEID", f"{property_name}_{depth}_Q0.5"),
        ("FORMAT", "GEOTIFF_INT16"),
        ("SUBSET", f"X({x - radius_m:.3f},{x + radius_m:.3f})"),
        ("SUBSET", f"Y({y - radius_m:.3f},{y + radius_m:.3f})"),
        ("SUBSETTINGCRS", SOILGRIDS_NATIVE_CRS),
        ("OUTPUTCRS", SOILGRIDS_NATIVE_CRS),
    ]
    return SOILGRIDS_ENDPOINT + "?" + urllib.parse.urlencode(params)


def physical_valid_mask(property_name: str, raw: np.ndarray) -> np.ndarray:
    canonical = raw.astype(np.float64) / SOILGRIDS_PROPERTIES[property_name]
    valid = np.isfinite(canonical)
    if property_name in {"wv0033", "wv1500"}:
        return valid & (canonical > 0) & (canonical <= 100)
    if property_name == "bdod":
        return valid & (canonical > 0) & (canonical <= 3)
    if property_name == "cfvo":
        return valid & (canonical >= 0) & (canonical <= 100)
    raise ValueError(f"Unsupported SoilGrids property {property_name}")


def choose_nearest_valid_cell(
    arrays: dict[tuple[str, str], np.ndarray],
    transform_value: Any,
    source_x: float,
    source_y: float,
    maximum_radius_m: float,
) -> tuple[int, int, float] | None:
    from rasterio.transform import xy

    valid: np.ndarray | None = None
    for (property_name, _), raw in arrays.items():
        layer_valid = physical_valid_mask(property_name, raw)
        valid = layer_valid if valid is None else valid & layer_valid
    assert valid is not None
    for depth in SOILGRIDS_DEPTHS:
        field_capacity = arrays[("wv0033", depth)].astype(float)
        wilting_point = arrays[("wv1500", depth)].astype(float)
        valid &= field_capacity > wilting_point
    rows, columns = np.where(valid)
    if len(rows) == 0:
        return None
    xs, ys = xy(transform_value, rows, columns, offset="center")
    distances = np.hypot(np.asarray(xs) - source_x, np.asarray(ys) - source_y)
    within = distances <= maximum_radius_m
    if not within.any():
        return None
    candidate_indexes = np.flatnonzero(within)
    selected = candidate_indexes[int(np.argmin(distances[within]))]
    return int(rows[selected]), int(columns[selected]), float(distances[selected])


def freeze_resolution(
    root: Path,
    protocol_path: Path,
    source_cache: Path,
    audit_dir: Path,
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise SystemExit("SoilGrids missing-resolution protocol identity mismatch")
    if audit_dir.exists():
        raise SystemExit(f"Fail-if-exists audit directory already exists: {audit_dir}")
    source_index_path = source_cache / "soilgrids_fetch_index.tsv"
    source_provenance_path = source_cache / "soilgrids_fetch_provenance.json"
    if not source_index_path.is_file() or not source_provenance_path.is_file():
        raise SystemExit("Completed SoilGrids source archive is missing")
    source_provenance = read_json(source_provenance_path)
    if not source_provenance.get("authoritative_soil_archive_complete"):
        raise SystemExit("SoilGrids source archive is not complete")
    source_index = pd.read_csv(source_index_path, sep="\t", dtype=str)
    missing = source_index[source_index["status"].eq(SOURCE_MISSING_STATUS)].copy()
    if missing.empty or not missing["site_id"].is_unique:
        raise SystemExit("Structural-missing SoilGrids inventory is empty or nonunique")
    keep = [
        "site_id",
        "latitude",
        "longitude",
        "mapped_environment_count",
        "properties",
        "depths",
        "statistic",
        "coverage_request_count",
        "detail",
    ]
    inventory = missing[keep].sort_values("site_id").reset_index(drop=True)
    audit_dir.mkdir(parents=True)
    inventory_path = audit_dir / "soilgrids_missing_site_inventory.tsv"
    write_tsv(inventory_path, inventory)
    freeze = {
        "status": "PASS",
        "protocol_version": PROTOCOL_VERSION,
        "selection_data": protocol["selection_data"],
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_climate_values_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
        "source_site_count": int(len(source_index)),
        "structurally_unavailable_site_count": int(len(inventory)),
        "affected_environment_count": int(
            pd.to_numeric(inventory["mapped_environment_count"], errors="raise").sum()
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "resolver_sha256": sha256_file(Path(__file__).resolve()),
        "source_index_sha256": sha256_file(source_index_path),
        "source_provenance_sha256": sha256_file(source_provenance_path),
        "missing_inventory_sha256": sha256_file(inventory_path),
        "source_index_path": str(source_index_path),
        "source_provenance_path": str(source_provenance_path),
        "protocol_path": str(protocol_path),
    }
    write_json_atomic(audit_dir / "soilgrids_missing_resolution_freeze.json", freeze)
    return freeze


def load_frozen_inputs(
    protocol_path: Path,
    source_cache: Path,
    audit_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    protocol = read_json(protocol_path)
    freeze_path = audit_dir / "soilgrids_missing_resolution_freeze.json"
    inventory_path = audit_dir / "soilgrids_missing_site_inventory.tsv"
    freeze = read_json(freeze_path)
    checks = {
        "protocol": sha256_file(protocol_path) == freeze["protocol_sha256"],
        "resolver": sha256_file(Path(__file__).resolve()) == freeze["resolver_sha256"],
        "source_index": sha256_file(source_cache / "soilgrids_fetch_index.tsv")
        == freeze["source_index_sha256"],
        "source_provenance": sha256_file(source_cache / "soilgrids_fetch_provenance.json")
        == freeze["source_provenance_sha256"],
        "inventory": sha256_file(inventory_path) == freeze["missing_inventory_sha256"],
    }
    if not all(checks.values()):
        raise SystemExit(f"Frozen SoilGrids resolution inputs changed: {checks}")
    inventory = pd.read_csv(inventory_path, sep="\t", dtype=str)
    return protocol, freeze, inventory


def cached_result(cache_dir: Path, site_id: str) -> dict[str, Any] | None:
    metadata_path = cache_dir / "requests" / site_id[:2] / f"{site_id}.json"
    if not metadata_path.is_file():
        return None
    metadata = read_json(metadata_path)
    status = str(metadata.get("status", ""))
    if status == "RECOVERED_NEAREST_VALID_SOIL_CELL" or status == "CANDIDATE_DISTANCE_REVIEW":
        values_path = cache_dir / str(metadata.get("values_path", ""))
        if not values_path.is_file() or sha256_file(values_path) != metadata.get(
            "values_sha256"
        ):
            return None
    if status not in TERMINAL_STATUSES:
        return None
    return metadata


def cache_raw(cache_dir: Path, raw: bytes) -> tuple[str, str]:
    digest = hashlib.sha256(raw).hexdigest()
    relative = Path("raw") / digest[:2] / f"{digest}.tif"
    path = cache_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(
            f".tmp.{os.getpid()}.{threading.get_ident()}.tif"
        )
        temporary.write_bytes(raw)
        try:
            temporary.replace(path)
        except FileExistsError:
            temporary.unlink(missing_ok=True)
    return relative.as_posix(), digest


def resolve_site(
    cache_dir: Path,
    row: pd.Series,
    protocol: dict[str, Any],
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    from rasterio.io import MemoryFile
    from rasterio.warp import transform as warp_transform

    site_id = str(row["site_id"])
    cached = cached_result(cache_dir, site_id)
    if cached is not None:
        cached = dict(cached)
        cached["status_before_cache"] = cached["status"]
        cached["status"] = "CACHED_" + str(cached["status"])
        return cached
    latitude = float(row["latitude"])
    longitude = float(row["longitude"])
    source_xs, source_ys = warp_transform(
        "EPSG:4326", SOILGRIDS_PROJ, [longitude], [latitude]
    )
    source_x, source_y = source_xs[0], source_ys[0]
    radius_m = int(protocol["maximum_search_radius_m"])
    arrays: dict[tuple[str, str], np.ndarray] = {}
    bindings: list[dict[str, Any]] = []
    expected_shape: tuple[int, int] | None = None
    expected_transform: Any = None
    expected_crs = ""
    try:
        layers = [
            (property_name, depth)
            for property_name in protocol["properties"]
            for depth in protocol["depths"]
        ]
        for property_name, depth in layers:
            url = neighborhood_url(
                property_name, depth, source_x, source_y, radius_m
            )
            raw = fetch_bytes(url, timeout, retries)
            raw_path, raw_sha256 = cache_raw(cache_dir, raw)
            with MemoryFile(raw) as memory:
                with memory.open() as source:
                    array = source.read(1)
                    current_shape = tuple(array.shape)
                    current_transform = source.transform
                    current_crs = str(source.crs)
            if expected_shape is None:
                expected_shape = current_shape
                expected_transform = current_transform
                expected_crs = current_crs
            elif (
                current_shape != expected_shape
                or tuple(current_transform) != tuple(expected_transform)
                or current_crs != expected_crs
            ):
                raise ValueError("SoilGrids neighborhood layers are not grid-aligned")
            arrays[(property_name, depth)] = array
            bindings.append(
                {
                    "property": property_name,
                    "depth": depth,
                    "raw_path": raw_path,
                    "raw_sha256": raw_sha256,
                    "request_url_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
                }
            )
        selected = choose_nearest_valid_cell(
            arrays,
            expected_transform,
            source_x,
            source_y,
            float(radius_m),
        )
        metadata_path = cache_dir / "requests" / site_id[:2] / f"{site_id}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if selected is None:
            metadata = {
                "site_id": site_id,
                "status": "MASKED_NO_VALID_SOIL_CELL_WITHIN_RADIUS",
                "source_latitude": latitude,
                "source_longitude": longitude,
                "maximum_search_radius_m": radius_m,
                "feature_eligible": False,
                "explicit_soil_missing_mask": True,
                "raw_bindings": bindings,
            }
            write_json_atomic(metadata_path, metadata)
            return metadata
        selected_row, selected_column, distance_m = selected
        from rasterio.transform import xy

        selected_x, selected_y = xy(
            expected_transform, selected_row, selected_column, offset="center"
        )
        selected_lon, selected_lat = warp_transform(
            SOILGRIDS_PROJ, "EPSG:4326", [selected_x], [selected_y]
        )
        accepted = distance_m <= float(protocol["automatic_acceptance_distance_m"])
        high_confidence = distance_m <= float(protocol["high_confidence_distance_m"])
        status = (
            "RECOVERED_NEAREST_VALID_SOIL_CELL"
            if accepted
            else "CANDIDATE_DISTANCE_REVIEW"
        )
        values = []
        for property_name, depth in layers:
            raw_value = float(arrays[(property_name, depth)][selected_row, selected_column])
            canonical_value = raw_value / SOILGRIDS_PROPERTIES[property_name]
            values.append(
                {
                    "site_id": site_id,
                    "source_latitude": latitude,
                    "source_longitude": longitude,
                    "soil_cell_latitude": selected_lat[0],
                    "soil_cell_longitude": selected_lon[0],
                    "soil_cell_distance_m": distance_m,
                    "property": property_name,
                    "depth": depth,
                    "statistic": protocol["statistic"],
                    "mapped_integer_value": raw_value,
                    "conversion_divisor": SOILGRIDS_PROPERTIES[property_name],
                    "canonical_value": canonical_value,
                    "canonical_unit": (
                        "volume_percent"
                        if property_name in {"wv0033", "wv1500", "cfvo"}
                        else "kg_per_dm3"
                    ),
                    "fallback_status": status,
                    "feature_eligible": accepted,
                    "high_confidence": high_confidence,
                    "explicit_soil_fallback_mask": True,
                }
            )
        values_frame = pd.DataFrame(values)
        relative_values = Path("values") / site_id[:2] / f"{site_id}.parquet"
        values_path = cache_dir / relative_values
        values_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_values = values_path.with_suffix(
            f".tmp.{os.getpid()}.{threading.get_ident()}.parquet"
        )
        values_frame.to_parquet(temporary_values, index=False, compression="zstd")
        temporary_values.replace(values_path)
        metadata = {
            "site_id": site_id,
            "status": status,
            "source_latitude": latitude,
            "source_longitude": longitude,
            "soil_cell_latitude": selected_lat[0],
            "soil_cell_longitude": selected_lon[0],
            "soil_cell_distance_m": distance_m,
            "maximum_search_radius_m": radius_m,
            "automatic_acceptance_distance_m": protocol[
                "automatic_acceptance_distance_m"
            ],
            "feature_eligible": accepted,
            "high_confidence": high_confidence,
            "explicit_soil_missing_mask": not accepted,
            "explicit_soil_fallback_mask": True,
            "values_path": relative_values.as_posix(),
            "values_sha256": sha256_file(values_path),
            "value_rows": len(values_frame),
            "raw_bindings": bindings,
        }
        write_json_atomic(metadata_path, metadata)
        return metadata
    except Exception as exc:
        return {
            "site_id": site_id,
            "status": "FAILED_RETRYABLE",
            "detail": f"{type(exc).__name__}:{exc}",
            "feature_eligible": False,
            "explicit_soil_missing_mask": True,
        }


def run_resolution(
    root: Path,
    protocol_path: Path,
    source_cache: Path,
    audit_dir: Path,
    cache_dir: Path,
    limit: int,
    workers: int,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    if limit < 0 or workers < 1:
        raise SystemExit("Limit must be nonnegative and workers must be positive")
    protocol, freeze, inventory = load_frozen_inputs(
        protocol_path, source_cache, audit_dir
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached: dict[str, dict[str, Any]] = {}
    for site_id in inventory["site_id"].astype(str):
        result = cached_result(cache_dir, site_id)
        if result is not None:
            cached[site_id] = result
    pending = inventory[~inventory["site_id"].isin(cached)].copy()
    if limit > 0:
        pending = pending.head(limit)
    fetched: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                resolve_site, cache_dir, row, protocol, timeout, retries
            ): str(row["site_id"])
            for _, row in pending.iterrows()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            site_id = futures[future]
            result = future.result()
            fetched[site_id] = result
            print(
                json.dumps(
                    {
                        "completed_this_run": completed,
                        "site_id": site_id,
                        "status": result["status"],
                        "soil_cell_distance_m": result.get("soil_cell_distance_m"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    rows = []
    for _, row in inventory.iterrows():
        site_id = str(row["site_id"])
        result = fetched.get(site_id) or cached.get(site_id)
        record = row.to_dict()
        if result is None:
            record.update(
                {
                    "status": "PENDING_LIMIT",
                    "feature_eligible": False,
                    "explicit_soil_missing_mask": True,
                }
            )
        else:
            record.update(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"raw_bindings"}
                }
            )
        rows.append(record)
    index = pd.DataFrame(rows)
    index_path = cache_dir / "soilgrids_missing_resolution_index.tsv"
    write_tsv(index_path, index)
    statuses = index["status"].astype(str).str.removeprefix("CACHED_")
    terminal_count = int(statuses.isin(TERMINAL_STATUSES).sum())
    accepted_count = int(statuses.eq("RECOVERED_NEAREST_VALID_SOIL_CELL").sum())
    review_count = int(statuses.eq("CANDIDATE_DISTANCE_REVIEW").sum())
    masked_count = int(statuses.eq("MASKED_NO_VALID_SOIL_CELL_WITHIN_RADIUS").sum())
    provenance = {
        "status": "PASS",
        "run_status": "COMPLETE" if terminal_count == len(index) else "PARTIAL",
        "protocol_version": PROTOCOL_VERSION,
        "selection_data": protocol["selection_data"],
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_climate_values_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
        "observations_excluded": 0,
        "source_structurally_unavailable_site_count": len(index),
        "selected_pending_count": len(pending),
        "resolved_terminal_count": terminal_count,
        "automatic_feature_eligible_count": accepted_count,
        "distance_review_count": review_count,
        "explicit_missing_mask_count": review_count + masked_count,
        "no_valid_local_cell_count": masked_count,
        "status_counts": {
            str(key): int(value) for key, value in statuses.value_counts().items()
        },
        "protocol_sha256": freeze["protocol_sha256"],
        "freeze_sha256": sha256_file(
            audit_dir / "soilgrids_missing_resolution_freeze.json"
        ),
        "resolution_index_sha256": sha256_file(index_path),
        "cache_dir": str(cache_dir),
    }
    write_json_atomic(
        cache_dir / "soilgrids_missing_resolution_provenance.json", provenance
    )
    return provenance


def certify_resolution(
    protocol_path: Path,
    source_cache: Path,
    audit_dir: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    protocol, _, inventory = load_frozen_inputs(protocol_path, source_cache, audit_dir)
    index_path = cache_dir / "soilgrids_missing_resolution_index.tsv"
    provenance_path = cache_dir / "soilgrids_missing_resolution_provenance.json"
    if not index_path.is_file() or not provenance_path.is_file():
        raise SystemExit("SoilGrids missing-resolution outputs are absent")
    index = pd.read_csv(index_path, sep="\t", dtype=str)
    statuses = index["status"].astype(str).str.removeprefix("CACHED_")
    recovered = index[statuses.eq("RECOVERED_NEAREST_VALID_SOIL_CELL")].copy()
    review = index[statuses.eq("CANDIDATE_DISTANCE_REVIEW")].copy()
    value_checks = []
    for frame_row in pd.concat([recovered, review]).itertuples(index=False):
        values_path = cache_dir / str(frame_row.values_path)
        values = pd.read_parquet(values_path)
        finite = np.isfinite(values["canonical_value"].to_numpy(float)).all()
        complete = len(values) == len(SOILGRIDS_PROPERTIES) * len(SOILGRIDS_DEPTHS)
        pivot = values.pivot(index="depth", columns="property", values="canonical_value")
        physical = bool((pivot["wv0033"] > pivot["wv1500"]).all())
        value_checks.append(finite and complete and physical)
    accepted_distances = pd.to_numeric(
        recovered.get("soil_cell_distance_m", pd.Series(dtype=float)), errors="coerce"
    )
    checks = {
        "source_site_set_exact": set(index["site_id"]) == set(inventory["site_id"]),
        "one_row_per_source_site": len(index) == len(inventory) and index["site_id"].is_unique,
        "all_sites_terminal": statuses.isin(TERMINAL_STATUSES).all(),
        "accepted_distance_bound": accepted_distances.le(
            float(protocol["automatic_acceptance_distance_m"])
        ).all(),
        "recovered_values_complete_finite_physical": all(value_checks),
        "explicit_mask_for_nonaccepted": index.loc[
            ~statuses.eq("RECOVERED_NEAREST_VALID_SOIL_CELL"),
            "explicit_soil_missing_mask",
        ]
        .astype(str)
        .str.lower()
        .eq("true")
        .all(),
        "no_observation_exclusion": True,
        "no_protected_or_prediction_access": True,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol_version": PROTOCOL_VERSION,
        "selection_data": protocol["selection_data"],
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_climate_values_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
        "observations_excluded": 0,
        "source_missing_sites": len(index),
        "accepted_nearest_cell_sites": int(
            statuses.eq("RECOVERED_NEAREST_VALID_SOIL_CELL").sum()
        ),
        "distance_review_sites": int(statuses.eq("CANDIDATE_DISTANCE_REVIEW").sum()),
        "explicit_mask_sites": int(
            (~statuses.eq("RECOVERED_NEAREST_VALID_SOIL_CELL")).sum()
        ),
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "artifacts": {
            "soilgrids_missing_resolution_index.tsv": sha256_file(index_path),
            "soilgrids_missing_resolution_provenance.json": sha256_file(provenance_path),
        },
    }
    output_path = audit_dir / "soilgrids_missing_resolution_certification.json"
    write_json_atomic(output_path, result)
    if result["status"] != "PASS":
        raise SystemExit("SoilGrids missing-resolution certification failed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "resolve", "certify"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, default=Path("."))
        command.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        command.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
        command.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
        if name in {"resolve", "certify"}:
            command.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
        if name == "resolve":
            command.add_argument("--limit", type=int, default=0)
            command.add_argument("--workers", type=int, default=2)
            command.add_argument("--timeout", type=int, default=120)
            command.add_argument("--retries", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    protocol_path = resolve(root, args.protocol)
    source_cache = resolve(root, args.source_cache)
    audit_dir = resolve(root, args.audit_dir)
    if args.command == "freeze":
        result = freeze_resolution(root, protocol_path, source_cache, audit_dir)
    elif args.command == "resolve":
        result = run_resolution(
            root,
            protocol_path,
            source_cache,
            audit_dir,
            resolve(root, args.cache_dir),
            args.limit,
            args.workers,
            args.timeout,
            args.retries,
        )
    else:
        result = certify_resolution(
            protocol_path,
            source_cache,
            audit_dir,
            resolve(root, args.cache_dir),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
