from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PROTOCOL = Path(
    "server_training_pipeline/phase6a_cmip6_member_fetch_protocol_v1.json"
)
DEFAULT_AUDIT_DIR = Path("audit/v2/phase6a_cmip6_member_resolved_fetch_v1")
DEFAULT_CACHE_DIR = Path("environment/v2/phase6a_cmip6_member_resolved_daily_v1")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, sep="\t", index=False, lineterminator="\n")
    os.replace(temporary, path)


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_protocol(root: Path, path: Path) -> dict[str, Any]:
    protocol_path = resolve(root, path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["_protocol_path"] = str(protocol_path)
    protocol["_protocol_sha256"] = sha256_file(protocol_path)
    if protocol["status"] != "FROZEN_MEMBER_RESOLVED_VALUE_FETCH_ALLOWED":
        raise ValueError("CMIP6 value fetch protocol is not frozen and allowed")
    checks = {
        "parent_fetch_contract": protocol["parent_fetch_contract_sha256"],
        "selected_asset_manifest": protocol["selected_asset_manifest_sha256"],
        "site_inventory": protocol["site_inventory_sha256"],
    }
    for key, expected in checks.items():
        observed = sha256_file(resolve(root, protocol[key]))
        if observed != expected:
            raise ValueError(
                f"Frozen input checksum mismatch for {key}: observed={observed}; "
                f"expected={expected}"
            )
    if int(protocol["maximum_fetch_workers"]) != 1:
        raise ValueError("Member-resolved fetch must remain single-worker")
    return protocol


def normalize_version(values: pd.Series) -> pd.Series:
    return (
        values.fillna("")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.removeprefix("v")
    )


def download_once(url: str, path: Path, timeout: int = 600) -> str:
    if path.is_file():
        return "CACHED"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "WheatConformer-Phase6A-CMIP6/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as source, temporary.open(
            "wb"
        ) as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if temporary.stat().st_size == 0:
            raise ValueError("Downloaded CMIP6 catalog is empty")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return "FETCHED"


def asset_key_columns() -> list[str]:
    return [
        "institution_id",
        "source_id",
        "experiment_id",
        "member_id",
        "variable_id",
        "grid_label",
        "version",
    ]


def build_transport_inventory(
    assets: pd.DataFrame,
    catalog: pd.DataFrame,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    required_asset_columns = {
        "institution_id",
        "source_id",
        "experiment_id",
        "member_id",
        "variable",
        "grid_label",
        "version",
        "catalog_record_id",
        "calendar",
    }
    missing = required_asset_columns.difference(assets.columns)
    if missing:
        raise ValueError(f"Selected asset manifest is missing columns: {sorted(missing)}")
    cloud = catalog[catalog["table_id"].astype(str).eq(protocol["table_id"])].copy()
    cloud["version"] = normalize_version(cloud["version"])
    selected = assets.rename(columns={"variable": "variable_id"}).copy()
    selected["version"] = normalize_version(selected["version"])
    selected_keys = selected[asset_key_columns()].drop_duplicates()
    cloud = cloud.merge(selected_keys, on=asset_key_columns(), how="inner")
    duplicate_cloud = cloud.duplicated(asset_key_columns(), keep=False)
    if duplicate_cloud.any():
        duplicated = cloud.loc[duplicate_cloud, asset_key_columns()].drop_duplicates()
        raise ValueError(
            "Cloud catalog contains duplicate exact scientific assets: "
            f"rows={len(duplicated)}"
        )
    merged = selected.merge(
        cloud[asset_key_columns() + ["zstore"]],
        on=asset_key_columns(),
        how="left",
        validate="one_to_one",
    )
    merged["transport"] = np.where(
        merged["zstore"].notna(), "PANGEO_ZARR_EXACT", "ESGF_REPLICA_RESOLUTION_PENDING"
    )
    merged["transport_status"] = np.where(
        merged["zstore"].notna(), "READY_TO_FETCH", "PENDING_EXACT_REPLICA_RESOLUTION"
    )
    intervals = merged["experiment_id"].eq("historical")
    merged["fetch_start"] = np.where(
        intervals, protocol["historical_start"], protocol["future_start"]
    )
    merged["fetch_end"] = np.where(
        intervals, protocol["historical_end"], protocol["future_end"]
    )
    request_ids = []
    for row in merged.itertuples(index=False):
        identity = {
            "source_id": row.source_id,
            "experiment_id": row.experiment_id,
            "member_id": row.member_id,
            "variable": row.variable_id,
            "grid_label": row.grid_label,
            "version": row.version,
            "calendar": row.calendar,
            "fetch_start": row.fetch_start,
            "fetch_end": row.fetch_end,
            "site_inventory_sha256": protocol["site_inventory_sha256"],
        }
        request_ids.append(sha256_bytes(canonical_json(identity)))
    merged.insert(0, "request_id", request_ids)
    if merged["request_id"].duplicated().any():
        raise ValueError("CMIP6 request IDs are not unique")
    if len(merged) != 455:
        raise ValueError(f"Expected 455 frozen assets, observed {len(merged)}")
    order = [
        "request_id",
        "source_id",
        "institution_id",
        "experiment_id",
        "member_id",
        "variable_id",
        "grid_label",
        "version",
        "calendar",
        "fetch_start",
        "fetch_end",
        "transport",
        "transport_status",
        "zstore",
        "catalog_record_id",
    ]
    return merged[order].sort_values(
        ["source_id", "experiment_id", "variable_id"], kind="stable"
    ).reset_index(drop=True)


def prepare(
    root: Path,
    protocol_path: Path,
    audit_dir: Path,
) -> dict[str, Any]:
    protocol = load_protocol(root, protocol_path)
    audit = resolve(root, audit_dir)
    catalog_path = audit / "catalog_snapshot" / "pangeo-cmip6.csv"
    retrieval_mode = download_once(protocol["pangeo_catalog_url"], catalog_path)
    assets = pd.read_csv(resolve(root, protocol["selected_asset_manifest"]), sep="\t", dtype=str)
    catalog = pd.read_csv(catalog_path, dtype=str)
    inventory = build_transport_inventory(assets, catalog, protocol)
    inventory_path = audit / "cmip6_member_resolved_transport_inventory.tsv"
    atomic_tsv(inventory_path, inventory)
    ready = int(inventory["transport_status"].eq("READY_TO_FETCH").sum())
    pending = len(inventory) - ready
    provenance = {
        "status": "PASS_PARTIAL_TRANSPORT_READY" if pending else "PASS_TRANSPORT_READY",
        "protocol_version": protocol["protocol_version"],
        "selection_data": protocol["selection_data"],
        "selected_asset_count": len(inventory),
        "exact_cloud_transport_count": ready,
        "pending_exact_esgf_replica_resolution_count": pending,
        "site_count": len(
            pd.read_csv(resolve(root, protocol["site_inventory"]), sep="\t", dtype=str)
        ),
        "catalog_retrieval_mode": retrieval_mode,
        "catalog_snapshot_sha256": sha256_file(catalog_path),
        "transport_inventory_sha256": sha256_file(inventory_path),
        "protocol_sha256": protocol["_protocol_sha256"],
        "fetcher_sha256": sha256_file(Path(__file__)),
        "member_dimension_retained": True,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
    }
    atomic_json(audit / "cmip6_member_resolved_transport_provenance.json", provenance)
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)
    return provenance


def haversine_km(
    lat_a: np.ndarray, lon_a: np.ndarray, lat_b: np.ndarray, lon_b: np.ndarray
) -> np.ndarray:
    radius = 6371.0088
    phi_a = np.radians(lat_a)
    phi_b = np.radians(lat_b)
    d_phi = phi_b - phi_a
    d_lambda = np.radians(((lon_b - lon_a + 180.0) % 360.0) - 180.0)
    value = np.sin(d_phi / 2.0) ** 2 + np.cos(phi_a) * np.cos(phi_b) * np.sin(
        d_lambda / 2.0
    ) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def verify_identity(dataset: Any, row: pd.Series) -> None:
    expected = {
        "source_id": row.source_id,
        "experiment_id": row.experiment_id,
        "variant_label": row.member_id,
        "grid_label": row.grid_label,
        "variable_id": row.variable_id,
    }
    for key, value in expected.items():
        observed = str(dataset.attrs.get(key, ""))
        if observed != str(value):
            raise ValueError(
                f"Remote CMIP6 identity mismatch for {key}: observed={observed}; expected={value}"
            )


def site_subset(dataset: Any, row: pd.Series, sites: pd.DataFrame) -> Any:
    import xarray as xr

    if row.variable_id not in dataset.data_vars:
        raise ValueError(f"Remote store lacks variable {row.variable_id}")
    if "lat" not in dataset.coords or "lon" not in dataset.coords:
        raise ValueError("Only explicit latitude/longitude CMIP6 grids are supported")
    if dataset["lat"].ndim != 1 or dataset["lon"].ndim != 1:
        raise ValueError("Curvilinear CMIP6 grid requires a separately certified selector")
    target_lat = sites["latitude"].astype(float).to_numpy()
    target_lon = sites["longitude"].astype(float).to_numpy()
    source_lon = dataset["lon"].values.astype(float)
    selection_lon = target_lon.copy()
    if float(np.nanmax(source_lon)) > 180.0:
        selection_lon %= 360.0
    else:
        selection_lon = ((selection_lon + 180.0) % 360.0) - 180.0
    selected = dataset[row.variable_id].sel(
        time=slice(row.fetch_start, row.fetch_end)
    )
    selected = selected.sel(
        lat=xr.DataArray(target_lat, dims="site"),
        lon=xr.DataArray(selection_lon, dims="site"),
        method="nearest",
    )
    if selected.sizes.get("site") != len(sites) or selected.sizes.get("time", 0) == 0:
        raise ValueError("CMIP6 site/time subset is empty or incomplete")
    chosen_lat = selected["lat"].values.astype(float)
    chosen_lon = selected["lon"].values.astype(float)
    distance = haversine_km(target_lat, target_lon, chosen_lat, chosen_lon)
    selected = selected.drop_vars([name for name in ("lat", "lon") if name in selected.coords])
    selected = selected.assign_coords(
        site=("site", np.arange(len(sites), dtype=np.int32)),
        site_id=("site", sites["site_id"].astype(str).to_numpy(dtype="U64")),
        target_latitude=("site", target_lat),
        target_longitude=("site", target_lon),
        source_latitude=("site", chosen_lat),
        source_longitude=("site", chosen_lon),
        source_grid_distance_km=("site", distance),
    )
    return selected.to_dataset(name=row.variable_id)


def free_space_gib(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / (1024**3)


def valid_receipt(
    receipt_path: Path, output_path: Path, *, verify_hash: bool = False
) -> bool:
    if not receipt_path.is_file() or not output_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        structurally_valid = (
            receipt.get("status") == "FETCHED"
            and int(receipt.get("output_bytes", -1)) == output_path.stat().st_size
        )
        if not structurally_valid:
            return False
        return not verify_hash or receipt.get("output_sha256") == sha256_file(output_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def fetch_asset(
    row: pd.Series,
    sites: pd.DataFrame,
    cache: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    import gcsfs
    import xarray as xr

    request_id = str(row.request_id)
    output = cache / "assets" / request_id[:2] / f"{request_id}.nc"
    receipt_path = cache / "requests" / request_id[:2] / f"{request_id}.json"
    if valid_receipt(receipt_path, output):
        return {"status": "CACHED", "request_id": request_id}
    minimum_free = float(protocol["minimum_free_space_gib"])
    observed_free = free_space_gib(cache)
    if observed_free < minimum_free:
        raise OSError(
            f"Insufficient free space: observed_gib={observed_free:.3f}; minimum_gib={minimum_free:.3f}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.part")
    temporary.unlink(missing_ok=True)
    started = time.time()
    filesystem = gcsfs.GCSFileSystem(token="anon", access="read_only")
    mapper = filesystem.get_mapper(str(row.zstore))
    decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    dataset = xr.open_zarr(mapper, consolidated=True, decode_times=decoder)
    try:
        verify_identity(dataset, row)
        selected = site_subset(dataset, row, sites)
        time_count = int(selected.sizes["time"])
        site_count = int(selected.sizes["site"])
        selected.attrs.update(
            {
                "phase6a_protocol_version": protocol["protocol_version"],
                "phase6a_protocol_sha256": protocol["_protocol_sha256"],
                "fetcher_sha256": sha256_file(Path(__file__)),
                "selected_asset_manifest_sha256": protocol[
                    "selected_asset_manifest_sha256"
                ],
                "site_inventory_sha256": protocol["site_inventory_sha256"],
                "request_id": request_id,
                "transport": row.transport,
                "transport_source": row.zstore,
                "source_id": row.source_id,
                "experiment_id": row.experiment_id,
                "variant_label": row.member_id,
                "grid_label": row.grid_label,
                "version": row.version,
                "calendar": row.calendar,
                "future_covariate_matrix": "not_generated",
                "future_prediction": "not_generated",
            }
        )
        selected = selected.chunk(
            {"time": min(366, time_count), "site": min(128, site_count)}
        )
        encoding = {
            row.variable_id: {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "chunksizes": (min(366, time_count), min(128, site_count)),
            }
        }
        selected.to_netcdf(temporary, engine="h5netcdf", encoding=encoding)
    finally:
        dataset.close()
    with xr.open_dataset(temporary, engine="h5netcdf", decode_times=False) as check:
        if row.variable_id not in check or int(check.sizes.get("site", 0)) != len(sites):
            raise ValueError("Written CMIP6 site archive failed structural verification")
        observed_time = int(check.sizes.get("time", 0))
        if observed_time != time_count:
            raise ValueError("Written CMIP6 site archive changed the time dimension")
    output_sha256 = sha256_file(temporary)
    output_bytes = temporary.stat().st_size
    os.replace(temporary, output)
    receipt = {
        "status": "FETCHED",
        "request_id": request_id,
        "source_id": row.source_id,
        "experiment_id": row.experiment_id,
        "member_id": row.member_id,
        "variable": row.variable_id,
        "grid_label": row.grid_label,
        "version": row.version,
        "calendar": row.calendar,
        "fetch_start": row.fetch_start,
        "fetch_end": row.fetch_end,
        "site_count": len(sites),
        "time_count": time_count,
        "transport": row.transport,
        "transport_source_sha256": sha256_bytes(str(row.zstore).encode("utf-8")),
        "protocol_sha256": protocol["_protocol_sha256"],
        "fetcher_sha256": sha256_file(Path(__file__)),
        "output_path": str(output),
        "output_bytes": output_bytes,
        "output_sha256": output_sha256,
        "elapsed_seconds": round(time.time() - started, 3),
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def status(
    root: Path,
    protocol_path: Path,
    audit_dir: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    protocol = load_protocol(root, protocol_path)
    audit = resolve(root, audit_dir)
    cache = resolve(root, cache_dir)
    inventory_path = audit / "cmip6_member_resolved_transport_inventory.tsv"
    if not inventory_path.is_file():
        raise FileNotFoundError("Run prepare before requesting CMIP6 fetch status")
    inventory = pd.read_csv(inventory_path, sep="\t", dtype=str)
    rows = []
    for row in inventory.itertuples(index=False):
        output = cache / "assets" / row.request_id[:2] / f"{row.request_id}.nc"
        receipt = cache / "requests" / row.request_id[:2] / f"{row.request_id}.json"
        if valid_receipt(receipt, output):
            current = "FETCHED"
        elif row.transport_status != "READY_TO_FETCH":
            current = row.transport_status
        else:
            current = "PENDING"
        rows.append({"source_id": row.source_id, "experiment_id": row.experiment_id, "status": current})
    frame = pd.DataFrame(rows)
    counts = frame["status"].value_counts().sort_index().to_dict()
    summary = {
        "status": "PASS",
        "protocol_version": protocol["protocol_version"],
        "selected_asset_count": len(inventory),
        "site_count": len(
            pd.read_csv(resolve(root, protocol["site_inventory"]), sep="\t", dtype=str)
        ),
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "completed_asset_count": int(counts.get("FETCHED", 0)),
        "archive_complete": int(counts.get("FETCHED", 0)) == len(inventory),
        "member_dimension_retained": True,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
        "free_space_gib": round(free_space_gib(cache), 3),
    }
    atomic_json(cache / "cmip6_member_resolved_fetch_status.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def fetch(
    root: Path,
    protocol_path: Path,
    audit_dir: Path,
    cache_dir: Path,
    limit: int,
    source_id: str | None,
    experiment_id: str | None,
    variable: str | None,
    fail_fast: bool,
) -> None:
    protocol = load_protocol(root, protocol_path)
    audit = resolve(root, audit_dir)
    cache = resolve(root, cache_dir)
    inventory_path = audit / "cmip6_member_resolved_transport_inventory.tsv"
    inventory = pd.read_csv(inventory_path, sep="\t", dtype=str)
    selected = inventory[inventory["transport_status"].eq("READY_TO_FETCH")].copy()
    for column, value in (
        ("source_id", source_id),
        ("experiment_id", experiment_id),
        ("variable_id", variable),
    ):
        if value:
            selected = selected[selected[column].eq(value)]
    priorities = {name: index for index, name in enumerate(protocol["source_priority"])}
    selected["_priority"] = selected["source_id"].map(priorities).fillna(len(priorities))
    selected = selected.sort_values(
        ["_priority", "source_id", "experiment_id", "variable_id"], kind="stable"
    )
    sites = pd.read_csv(resolve(root, protocol["site_inventory"]), sep="\t", dtype=str)
    if len(sites) != 907 or sites["site_id"].duplicated().any():
        raise ValueError("Frozen Stage-1 v2 site inventory is not the certified 907-site set")
    attempted = 0
    for _, row in selected.iterrows():
        output = cache / "assets" / row.request_id[:2] / f"{row.request_id}.nc"
        receipt = cache / "requests" / row.request_id[:2] / f"{row.request_id}.json"
        if valid_receipt(receipt, output):
            continue
        if limit > 0 and attempted >= limit:
            break
        attempted += 1
        label = (
            f"{row.source_id}/{row.experiment_id}/{row.member_id}/"
            f"{row.variable_id}/{row.grid_label}/v{row.version}"
        )
        print(f"[{attempted}] FETCH {label}", flush=True)
        try:
            result = fetch_asset(row, sites, cache, protocol)
            print(json.dumps(result, sort_keys=True), flush=True)
        except Exception as exc:
            failure = {
                "status": "FAILED_RETRYABLE",
                "request_id": row.request_id,
                "label": label,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timestamp_epoch": time.time(),
            }
            atomic_json(cache / "failures" / row.request_id[:2] / f"{row.request_id}.json", failure)
            print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
            if fail_fast:
                raise
    status(root, protocol_path, audit_dir, cache_dir)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for name in ("prepare", "fetch", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, default=Path("."))
        command.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        command.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
        if name in {"fetch", "status"}:
            command.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
        if name == "fetch":
            command.add_argument("--limit", type=int, default=0)
            command.add_argument("--source-id")
            command.add_argument("--experiment-id")
            command.add_argument("--variable")
            command.add_argument("--fail-fast", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        prepare(root, args.protocol, args.audit_dir)
    elif args.command == "status":
        status(root, args.protocol, args.audit_dir, args.cache_dir)
    else:
        fetch(
            root,
            args.protocol,
            args.audit_dir,
            args.cache_dir,
            args.limit,
            args.source_id,
            args.experiment_id,
            args.variable,
            args.fail_fast,
        )


if __name__ == "__main__":
    main()
