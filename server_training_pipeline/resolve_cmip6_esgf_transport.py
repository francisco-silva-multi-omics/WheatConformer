from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from server_training_pipeline.fetch_cmip6_member_resolved import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_PROTOCOL,
    atomic_json,
    atomic_tsv,
    canonical_json,
    load_protocol,
    resolve,
    sha256_bytes,
    sha256_file,
)


ESGF_SEARCH = "https://esgf-node.llnl.gov/esg-search/search"
PERIOD_PATTERN = re.compile(r"_(\d{8})-(\d{8})\.nc$")


def scalar(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def service_urls(document: dict[str, Any], service: str) -> list[str]:
    result = []
    for entry in document.get("url", []) or []:
        parts = str(entry).split("|")
        if len(parts) < 3 or parts[-1].upper() != service.upper():
            continue
        url = parts[0]
        if service.upper() == "OPENDAP":
            url = url.removesuffix(".html")
        result.append(url)
    return sorted(set(result))


def file_period(title: str) -> tuple[str, str]:
    match = PERIOD_PATTERN.search(title)
    if not match:
        raise ValueError(f"CMIP6 daily filename has no terminal date range: {title}")
    return match.group(1), match.group(2)


def overlaps_interval(start: str, end: str, fetch_start: str, fetch_end: str) -> bool:
    normalized_start = fetch_start.replace("-", "")
    normalized_end = fetch_end.replace("-", "")
    return start <= normalized_end and end >= normalized_start


def exact_query_url(row: pd.Series) -> str:
    parameters = [
        ("project", "CMIP6"),
        ("type", "File"),
        ("source_id", row.source_id),
        ("experiment_id", row.experiment_id),
        ("member_id", row.member_id),
        ("grid_label", row.grid_label),
        ("variable_id", row.variable_id),
        ("frequency", "day"),
        ("version", row.version),
        ("distrib", "true"),
        ("format", "application/solr+json"),
        ("limit", "10000"),
    ]
    return ESGF_SEARCH + "?" + urllib.parse.urlencode(parameters)


def deterministic_gzip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(payload)
    os.replace(temporary, path)


def query_asset(
    row_dict: dict[str, str], snapshot_dir: Path, timeout: int, retries: int
) -> tuple[dict[str, str], dict[str, Any], str]:
    row = pd.Series(row_dict)
    url = exact_query_url(row)
    snapshot = snapshot_dir / f"{row.request_id}.json.gz"
    if snapshot.is_file():
        with gzip.open(snapshot, "rb") as stream:
            payload = stream.read()
        return row_dict, json.loads(payload), "CACHED"
    error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "WheatConformer-Phase6A-CMIP6/1.0"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            parsed = json.loads(payload)
            declared = int(parsed["response"]["numFound"])
            observed = len(parsed["response"].get("docs", []))
            if declared != observed:
                raise ValueError(
                    f"Incomplete ESGF file query: declared={declared}; observed={observed}"
                )
            deterministic_gzip(snapshot, payload)
            return row_dict, parsed, "FETCHED"
        except Exception as exc:
            error = exc
            if attempt < retries:
                time.sleep(2 ** attempt)
    assert error is not None
    raise error


def resolve_asset_files(row: pd.Series, response: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exact_instance = str(row.catalog_record_id).split("|", 1)[0]
    documents = [
        document
        for document in response["response"].get("docs", [])
        if scalar(document.get("dataset_id")).split("|", 1)[0] == exact_instance
    ]
    by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        by_title[scalar(document.get("title"))].append(document)
    file_rows = []
    checksum_conflicts = 0
    required_files = 0
    required_http_files = 0
    required_opendap_files = 0
    for title in sorted(by_title):
        replicas = by_title[title]
        start, end = file_period(title)
        required = overlaps_interval(start, end, row.fetch_start, row.fetch_end)
        checksums = {
            (scalar(document.get("checksum_type")).upper(), scalar(document.get("checksum")))
            for document in replicas
            if scalar(document.get("checksum"))
        }
        if len(checksums) > 1:
            checksum_conflicts += 1
        checksum_type, checksum = next(iter(checksums), ("", ""))
        http_urls = sorted(
            {
                url
                for document in replicas
                for url in service_urls(document, "HTTPServer")
            }
        )
        opendap_urls = sorted(
            {
                url
                for document in replicas
                for url in service_urls(document, "OPENDAP")
            }
        )
        sizes = {int(document.get("size") or 0) for document in replicas}
        if required:
            required_files += 1
            required_http_files += bool(http_urls)
            required_opendap_files += bool(opendap_urls)
        file_rows.append(
            {
                "request_id": row.request_id,
                "source_id": row.source_id,
                "experiment_id": row.experiment_id,
                "member_id": row.member_id,
                "variable_id": row.variable_id,
                "grid_label": row.grid_label,
                "version": row.version,
                "title": title,
                "file_start": start,
                "file_end": end,
                "required_for_fetch_interval": required,
                "replica_count": len(replicas),
                "size_bytes": max(sizes) if sizes else 0,
                "checksum_type": checksum_type,
                "checksum": checksum,
                "http_urls_json": json.dumps(http_urls, separators=(",", ":")),
                "opendap_urls_json": json.dumps(opendap_urls, separators=(",", ":")),
            }
        )
    if not documents:
        status = "BLOCKED_EXACT_DATASET_NOT_RESOLVED"
    elif checksum_conflicts:
        status = "BLOCKED_REPLICA_CHECKSUM_CONFLICT"
    elif required_files == 0:
        status = "BLOCKED_NO_FILES_OVERLAP_FETCH_INTERVAL"
    elif required_http_files == required_files:
        status = "READY_ESGF_HTTP_EXACT"
    else:
        status = "BLOCKED_REQUIRED_FILE_LACKS_HTTP_REPLICA"
    summary = {
        "request_id": row.request_id,
        "source_id": row.source_id,
        "experiment_id": row.experiment_id,
        "member_id": row.member_id,
        "variable_id": row.variable_id,
        "grid_label": row.grid_label,
        "version": row.version,
        "transport_status": status,
        "logical_file_count": len(by_title),
        "required_file_count": required_files,
        "required_http_file_count": required_http_files,
        "required_opendap_file_count": required_opendap_files,
        "checksum_conflict_count": checksum_conflicts,
        "required_raw_bytes": sum(
            int(item["size_bytes"])
            for item in file_rows
            if item["required_for_fetch_interval"]
        ),
    }
    return file_rows, summary


def run(
    root: Path,
    protocol_path: Path,
    audit_dir: Path,
    workers: int,
    timeout: int,
    retries: int,
) -> None:
    protocol = load_protocol(root, protocol_path)
    if workers < 1 or workers > 4:
        raise ValueError("ESGF metadata resolver workers must be between 1 and 4")
    audit = resolve(root, audit_dir)
    inventory_path = audit / "cmip6_member_resolved_transport_inventory.tsv"
    inventory = pd.read_csv(inventory_path, sep="\t", dtype=str)
    pending = inventory[
        inventory["transport_status"].eq("PENDING_EXACT_REPLICA_RESOLUTION")
    ].copy()
    snapshot_dir = audit / "esgf_catalog_snapshot"
    payloads = pending.to_dict("records")
    resolved: list[tuple[dict[str, str], dict[str, Any], str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(query_asset, row, snapshot_dir, timeout, retries)
            for row in payloads
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            resolved.append(future.result())
            if index % 10 == 0 or index == len(futures):
                print(f"Resolved ESGF metadata {index}/{len(futures)}", flush=True)
    file_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    retrieval_counts: dict[str, int] = defaultdict(int)
    by_request = {row["request_id"]: row for row in payloads}
    for row_dict, response, retrieval_mode in resolved:
        rows, summary = resolve_asset_files(pd.Series(by_request[row_dict["request_id"]]), response)
        file_rows.extend(rows)
        summaries.append(summary)
        retrieval_counts[retrieval_mode] += 1
    files = pd.DataFrame(file_rows).sort_values(
        ["source_id", "experiment_id", "variable_id", "file_start", "title"],
        kind="stable",
    )
    assets = pd.DataFrame(summaries).sort_values(
        ["source_id", "experiment_id", "variable_id"], kind="stable"
    )
    file_path = audit / "cmip6_esgf_exact_file_transport_manifest.tsv"
    asset_path = audit / "cmip6_esgf_exact_asset_transport_summary.tsv"
    atomic_tsv(file_path, files)
    atomic_tsv(asset_path, assets)
    status_counts = assets["transport_status"].value_counts().sort_index().to_dict()
    provenance = {
        "status": "PASS" if assets["transport_status"].eq("READY_ESGF_HTTP_EXACT").all() else "PASS_WITH_EXPLICIT_BLOCKERS",
        "protocol_version": "phase6a_cmip6_exact_esgf_transport_resolution_v1",
        "selection_data": "frozen_member_identity_and_public_esgf_file_metadata_only",
        "asset_count": len(assets),
        "file_count": len(files),
        "required_file_count": int(files["required_for_fetch_interval"].sum()),
        "required_raw_bytes": int(assets["required_raw_bytes"].sum()),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "retrieval_counts": dict(retrieval_counts),
        "selected_asset_manifest_sha256": protocol["selected_asset_manifest_sha256"],
        "transport_inventory_sha256": sha256_file(inventory_path),
        "file_transport_manifest_sha256": sha256_file(file_path),
        "asset_transport_summary_sha256": sha256_file(asset_path),
        "resolver_sha256": sha256_file(Path(__file__)),
        "climate_values_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
        "phenotype_values_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
    }
    atomic_json(audit / "cmip6_esgf_exact_transport_provenance.json", provenance)
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    run(args.root.resolve(), args.protocol, args.audit_dir, args.workers, args.timeout, args.retries)


if __name__ == "__main__":
    main()
