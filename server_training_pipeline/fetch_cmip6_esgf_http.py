from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from server_training_pipeline.fetch_cmip6_member_resolved import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_PROTOCOL,
    atomic_json,
    free_space_gib,
    load_protocol,
    resolve,
    sha256_bytes,
    sha256_file,
    site_subset,
    valid_receipt,
    verify_identity,
)


DEFAULT_HOST_PRIORITY_PROTOCOL = Path(
    "server_training_pipeline/phase6a_cmip6_transport_host_priority_v1.json"
)


def checksum_file(path: Path, algorithm: str) -> str:
    normalized = algorithm.lower().replace("-", "")
    try:
        digest = hashlib.new(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported ESGF checksum algorithm: {algorithm}") from exc
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_urls(
    raw_urls: str, host_priority: list[str] | tuple[str, ...] = ()
) -> list[str]:
    urls = json.loads(raw_urls)
    result = []
    for url in urls:
        if str(url).startswith("http://"):
            result.append("https://" + str(url)[len("http://") :])
        result.append(str(url))
    unique = list(dict.fromkeys(result))
    if not host_priority:
        return unique
    ranks = {host.lower(): index for index, host in enumerate(host_priority)}

    def priority(url: str) -> tuple[int, int, int]:
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").lower()
        host_rank = ranks.get(host, len(ranks))
        scheme_rank = 0 if parsed.scheme.lower() == "https" else 1
        return host_rank, scheme_rank, unique.index(url)

    return sorted(unique, key=priority)


def asset_transport_priority(
    files: pd.DataFrame, host_priority: list[str] | tuple[str, ...]
) -> int:
    """Rank an asset by its slowest required file's best exact replica."""
    ranks = {host.lower(): index for index, host in enumerate(host_priority)}
    fallback_rank = len(ranks)
    file_ranks = []
    for raw_urls in files["http_urls_json"].astype(str):
        urls = candidate_urls(raw_urls, host_priority)
        file_ranks.append(
            min(
                (
                    ranks.get(
                        (urllib.parse.urlparse(url).hostname or "").lower(),
                        fallback_rank,
                    )
                    for url in urls
                ),
                default=fallback_rank,
            )
        )
    return max(file_ranks, default=fallback_rank)


def load_host_priority_protocol(root: Path, path: Path) -> dict[str, Any]:
    resolved = resolve(root, path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != "phase6a_cmip6_transport_host_priority_v1":
        raise ValueError("CMIP6 host-priority protocol identity mismatch")
    if payload.get("scientific_asset_identity_changed") is not False:
        raise ValueError("Host-priority protocol may not change scientific asset identity")
    if payload.get("exact_file_size_and_checksum_required") is not True:
        raise ValueError("Host-priority protocol must retain exact checksum validation")
    if int(payload.get("asset_concurrency", 0)) != 1:
        raise ValueError("Host-priority amendment retains one asset at a time")
    hosts = payload.get("host_priority")
    if not isinstance(hosts, list) or not hosts or len(hosts) != len(set(hosts)):
        raise ValueError("Host-priority registry is empty or nonunique")
    payload["_path"] = str(resolved)
    payload["_sha256"] = sha256_file(resolved)
    return payload


def download_verified(
    urls: list[str],
    destination: Path,
    expected_bytes: int,
    checksum_type: str,
    expected_checksum: str,
    retries: int,
    timeout: int,
) -> tuple[str, str]:
    if destination.is_file():
        if destination.stat().st_size == expected_bytes and checksum_file(
            destination, checksum_type
        ).lower() == expected_checksum.lower():
            return "CACHED", ""
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        for attempt in range(retries + 1):
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.{attempt}.part"
            )
            temporary.unlink(missing_ok=True)
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "WheatConformer-Phase6A-CMIP6/1.0"},
                )
                downloaded = 0
                next_report = 512 * 1024 * 1024
                with urllib.request.urlopen(request, timeout=timeout) as source, temporary.open(
                    "wb"
                ) as target:
                    while chunk := source.read(8 * 1024 * 1024):
                        target.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            print(
                                f"  downloaded_gib={downloaded / 1024**3:.3f} file={destination.name}",
                                flush=True,
                            )
                            next_report += 512 * 1024 * 1024
                if temporary.stat().st_size != expected_bytes:
                    raise ValueError(
                        f"Downloaded size mismatch: observed={temporary.stat().st_size}; "
                        f"expected={expected_bytes}"
                    )
                observed = checksum_file(temporary, checksum_type)
                if observed.lower() != expected_checksum.lower():
                    raise ValueError(
                        f"Downloaded checksum mismatch: observed={observed}; expected={expected_checksum}"
                    )
                os.replace(temporary, destination)
                return "FETCHED", url
            except Exception as exc:
                errors.append(f"{url}:{type(exc).__name__}:{exc}")
                temporary.unlink(missing_ok=True)
                if attempt < retries:
                    time.sleep(2 ** attempt)
    raise RuntimeError("Every exact ESGF HTTP replica failed: " + " | ".join(errors))


def write_subset_part(
    raw_path: Path,
    part_path: Path,
    row: pd.Series,
    sites: pd.DataFrame,
) -> dict[str, Any]:
    import xarray as xr

    decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(
        raw_path, engine="netcdf4", decode_times=decoder, chunks={"time": 366}
    ) as source:
        verify_identity(source, row)
        selected = site_subset(source, row, sites)
        time_count = int(selected.sizes.get("time", 0))
        if time_count == 0:
            return {"status": "NO_OVERLAPPING_DAYS", "time_count": 0}
        selected = selected.chunk(
            {"time": min(366, time_count), "site": min(128, len(sites))}
        )
        part_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = part_path.with_name(f".{part_path.name}.{os.getpid()}.part")
        temporary.unlink(missing_ok=True)
        selected.to_netcdf(
            temporary,
            engine="h5netcdf",
            encoding={
                row.variable_id: {
                    "zlib": True,
                    "complevel": 4,
                    "shuffle": True,
                    "chunksizes": (min(366, time_count), min(128, len(sites))),
                }
            },
        )
        os.replace(temporary, part_path)
    return {
        "status": "PART_READY",
        "time_count": time_count,
        "part_bytes": part_path.stat().st_size,
        "part_sha256": sha256_file(part_path),
    }


def part_is_valid(part_path: Path, metadata_path: Path, source_checksum: str) -> bool:
    if not part_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            metadata.get("status") == "PART_READY"
            and metadata.get("source_checksum") == source_checksum
            and int(metadata.get("part_bytes", -1)) == part_path.stat().st_size
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def assemble_asset(
    row: pd.Series,
    files: pd.DataFrame,
    sites: pd.DataFrame,
    cache: Path,
    protocol: dict[str, Any],
    retries: int,
    timeout: int,
) -> dict[str, Any]:
    import xarray as xr

    request_id = row.request_id
    output = cache / "assets" / request_id[:2] / f"{request_id}.nc"
    receipt_path = cache / "requests" / request_id[:2] / f"{request_id}.json"
    if valid_receipt(receipt_path, output):
        return {"status": "CACHED", "request_id": request_id}
    if free_space_gib(cache) < float(protocol["minimum_free_space_gib"]):
        raise OSError("Insufficient free space for ESGF raw-file staging")
    # Keep transient paths short enough for HDF5 on Windows, which does not
    # consistently honor extended-length path handling.
    staging = cache / "tmp_http" / request_id[:16]
    raw_dir = staging / "raw"
    part_dir = staging / "parts"
    part_paths = []
    transport_urls = []
    started = time.time()
    for index, file_row in enumerate(files.itertuples(index=False), 1):
        print(
            f"  [{index}/{len(files)}] SOURCE {file_row.title} "
            f"size_gib={int(file_row.size_bytes) / 1024**3:.3f}",
            flush=True,
        )
        digest = sha256_bytes(file_row.title.encode("utf-8"))
        raw_path = raw_dir / file_row.title
        part_path = part_dir / f"{digest[:16]}.nc"
        part_metadata = part_dir / f"{digest[:16]}.json"
        if not part_is_valid(part_path, part_metadata, file_row.checksum):
            _, used_url = download_verified(
                candidate_urls(
                    file_row.http_urls_json, protocol.get("_host_priority", [])
                ),
                raw_path,
                int(file_row.size_bytes),
                file_row.checksum_type,
                file_row.checksum,
                retries,
                timeout,
            )
            metadata = write_subset_part(raw_path, part_path, row, sites)
            raw_path.unlink(missing_ok=True)
            if metadata["status"] == "NO_OVERLAPPING_DAYS":
                continue
            metadata.update(
                {
                    "source_title": file_row.title,
                    "source_checksum_type": file_row.checksum_type,
                    "source_checksum": file_row.checksum,
                    "transport_url_sha256": sha256_bytes(used_url.encode("utf-8")),
                    "transport_host_priority_protocol_sha256": protocol.get(
                        "_host_priority_protocol_sha256", ""
                    ),
                }
            )
            atomic_json(part_metadata, metadata)
        part_paths.append(part_path)
        transport_urls.extend(
            candidate_urls(
                file_row.http_urls_json, protocol.get("_host_priority", [])
            )
        )
    if not part_paths:
        raise ValueError("No ESGF source file contributed days to the requested interval")
    decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
    parts = [
        xr.open_dataset(path, engine="h5netcdf", decode_times=decoder, chunks={"time": 366})
        for path in part_paths
    ]
    try:
        combined = xr.concat(parts, dim="time", data_vars="minimal", coords="minimal", compat="override")
        combined = combined.sortby("time")
        times = combined["time"].values
        if any(current <= previous for previous, current in zip(times[:-1], times[1:])):
            raise ValueError("ESGF subset parts contain duplicate or non-increasing time values")
        time_count = int(combined.sizes["time"])
        combined.attrs.update(
            {
                "phase6a_protocol_version": protocol["protocol_version"],
                "phase6a_protocol_sha256": protocol["_protocol_sha256"],
                "fetcher_sha256": sha256_file(Path(__file__)),
                "selected_asset_manifest_sha256": protocol[
                    "selected_asset_manifest_sha256"
                ],
                "site_inventory_sha256": protocol["site_inventory_sha256"],
                "request_id": request_id,
                "transport": "ESGF_HTTP_EXACT",
                "transport_host_priority_protocol_sha256": protocol.get(
                    "_host_priority_protocol_sha256", ""
                ),
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
        combined = combined.chunk(
            {"time": min(366, time_count), "site": min(128, len(sites))}
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.part")
        temporary.unlink(missing_ok=True)
        combined.to_netcdf(
            temporary,
            engine="h5netcdf",
            encoding={
                row.variable_id: {
                    "zlib": True,
                    "complevel": 4,
                    "shuffle": True,
                    "chunksizes": (min(366, time_count), min(128, len(sites))),
                }
            },
        )
    finally:
        for part in parts:
            part.close()
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
        "transport": "ESGF_HTTP_EXACT",
        "transport_source_count": len(set(transport_urls)),
        "transport_host_priority_protocol_sha256": protocol.get(
            "_host_priority_protocol_sha256", ""
        ),
        "output_path": str(output),
        "output_bytes": output_bytes,
        "output_sha256": output_sha256,
        "protocol_sha256": protocol["_protocol_sha256"],
        "fetcher_sha256": sha256_file(Path(__file__)),
        "elapsed_seconds": round(time.time() - started, 3),
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(receipt_path, receipt)
    shutil.rmtree(staging, ignore_errors=True)
    return receipt


def run(
    root: Path,
    protocol_path: Path,
    audit_dir: Path,
    cache_dir: Path,
    limit: int,
    retries: int,
    timeout: int,
    fail_fast: bool,
    host_priority_protocol_path: Path,
) -> None:
    protocol = load_protocol(root, protocol_path)
    host_protocol = load_host_priority_protocol(root, host_priority_protocol_path)
    protocol["_host_priority"] = host_protocol["host_priority"]
    protocol["_host_priority_protocol_sha256"] = host_protocol["_sha256"]
    audit = resolve(root, audit_dir)
    cache = resolve(root, cache_dir)
    summary = pd.read_csv(
        audit / "cmip6_esgf_exact_asset_transport_summary.tsv", sep="\t", dtype=str
    )
    files = pd.read_csv(
        audit / "cmip6_esgf_exact_file_transport_manifest.tsv", sep="\t", dtype=str
    )
    inventory = pd.read_csv(
        audit / "cmip6_member_resolved_transport_inventory.tsv", sep="\t", dtype=str
    )
    ready = summary[summary["transport_status"].eq("READY_ESGF_HTTP_EXACT")].merge(
        inventory.drop(columns=["transport_status"]), on="request_id", validate="one_to_one"
    )
    priorities = {name: index for index, name in enumerate(protocol["source_priority"])}
    ready["_priority"] = ready["source_id_x"].map(priorities).fillna(len(priorities))
    required_files = files[
        files["required_for_fetch_interval"].astype(str).str.lower().eq("true")
    ]
    request_transport_priority = {
        request_id: asset_transport_priority(group, protocol["_host_priority"])
        for request_id, group in required_files.groupby("request_id", sort=False)
    }
    ready["_transport_priority"] = (
        ready["request_id"]
        .map(request_transport_priority)
        .fillna(len(protocol["_host_priority"]))
    )
    ready = ready.sort_values(
        [
            "_transport_priority",
            "_priority",
            "source_id_x",
            "experiment_id_x",
            "variable_id_x",
        ],
        kind="stable",
    )
    sites = pd.read_csv(resolve(root, protocol["site_inventory"]), sep="\t", dtype=str)
    attempted = 0
    for _, merged in ready.iterrows():
        row = pd.Series(
            {
                "request_id": merged.request_id,
                "source_id": merged.source_id_x,
                "experiment_id": merged.experiment_id_x,
                "member_id": merged.member_id_x,
                "variable_id": merged.variable_id_x,
                "grid_label": merged.grid_label_x,
                "version": merged.version_x,
                "calendar": merged.calendar,
                "fetch_start": merged.fetch_start,
                "fetch_end": merged.fetch_end,
            }
        )
        output = cache / "assets" / row.request_id[:2] / f"{row.request_id}.nc"
        receipt = cache / "requests" / row.request_id[:2] / f"{row.request_id}.json"
        if valid_receipt(receipt, output):
            continue
        if limit > 0 and attempted >= limit:
            break
        attempted += 1
        current_files = files[
            files["request_id"].eq(row.request_id)
            & files["required_for_fetch_interval"].astype(str).str.lower().eq("true")
        ].sort_values(["file_start", "title"], kind="stable")
        label = (
            f"{row.source_id}/{row.experiment_id}/{row.member_id}/"
            f"{row.variable_id}/{row.grid_label}/v{row.version}"
        )
        print(f"[{attempted}] FETCH ESGF {label}", flush=True)
        try:
            result = assemble_asset(
                row, current_files, sites, cache, protocol, retries, timeout
            )
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
            atomic_json(
                cache / "failures_esgf_http" / row.request_id[:2] / f"{row.request_id}.json",
                failure,
            )
            print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
            if fail_fast:
                raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--host-priority-protocol",
        type=Path,
        default=DEFAULT_HOST_PRIORITY_PROTOCOL,
    )
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.protocol,
        args.audit_dir,
        args.cache_dir,
        args.limit,
        args.retries,
        args.timeout,
        args.fail_fast,
        args.host_priority_protocol,
    )


if __name__ == "__main__":
    main()
