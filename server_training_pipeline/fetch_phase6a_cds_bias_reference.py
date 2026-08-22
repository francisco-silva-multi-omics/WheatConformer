from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any
import zipfile

import pandas as pd

from server_training_pipeline.fetch_cmip6_member_resolved import (
    atomic_json,
    atomic_tsv,
    canonical_json,
    resolve,
    sha256_file,
)


DEFAULT_PROTOCOL = Path(
    "server_training_pipeline/phase6a_cds_bias_reference_protocol_v1.json"
)
DEFAULT_TRANSPORT_AMENDMENT = Path(
    "server_training_pipeline/phase6a_cds_bias_reference_transport_amendment_v2.json"
)
DEFAULT_AUDIT = Path("audit/v2/phase6a_cds_bias_reference_v1")
DEFAULT_CACHE = Path("environment/v2/phase6a_cds_era5_land_bias_reference_v1")
SITE_INVENTORY = Path(
    "audit/v2/phase6a_environment_source_contract_v10/soilgrids_request_inventory.tsv"
)


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def load_protocol(root: Path, path: Path) -> dict[str, Any]:
    resolved = resolve(root, path)
    protocol = json.loads(resolved.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "phase6a_cds_bias_reference_v1":
        raise ValueError("CDS bias-reference protocol identity mismatch")
    for relative, expected in protocol["parent_artifacts"].items():
        if expected == "REPLACE_PROTOCOL_SHA256":
            raise ValueError("CDS bias-reference protocol contains an unresolved checksum")
        if sha256_file(resolve(root, Path(relative))) != expected:
            raise ValueError(f"Frozen parent artifact changed: {relative}")
    protocol["_path"] = str(resolved)
    protocol["_sha256"] = sha256_file(resolved)
    return protocol


def load_transport_amendment(
    root: Path,
    path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    resolved = resolve(root, path)
    amendment = json.loads(resolved.read_text(encoding="utf-8"))
    if amendment.get("protocol_version") not in {
        "phase6a_cds_bias_reference_transport_v1",
        "phase6a_cds_bias_reference_transport_v2",
    }:
        raise ValueError("CDS bias-reference transport amendment identity mismatch")
    if amendment.get("status") != "FROZEN_OPERATIONAL_AMENDMENT":
        raise ValueError("CDS bias-reference transport amendment is not frozen")
    if amendment.get("parent_protocol_sha256") != protocol["_sha256"]:
        raise ValueError("CDS bias-reference transport amendment parent mismatch")
    parent_amendment_path = amendment.get("parent_transport_amendment_path")
    parent_amendment_sha256 = amendment.get("parent_transport_amendment_sha256")
    if parent_amendment_path or parent_amendment_sha256:
        if not parent_amendment_path or not parent_amendment_sha256:
            raise ValueError("Transport amendment parent identity is incomplete")
        if sha256_file(resolve(root, Path(parent_amendment_path))) != parent_amendment_sha256:
            raise ValueError("Frozen parent transport amendment changed")
    maximum = int(amendment["maximum_pending_requests"])
    default = int(amendment["default_worker_count"])
    if maximum < 1 or maximum > 20 or default < 1 or default > maximum:
        raise ValueError("CDS bias-reference transport worker limits are invalid")
    amendment["_path"] = str(resolved)
    amendment["_sha256"] = sha256_file(resolved)
    return amendment


def effective_worker_count(
    requested: int | None,
    selected_count: int,
    amendment: dict[str, Any],
) -> int:
    workers = (
        int(amendment["default_worker_count"])
        if requested is None
        else int(requested)
    )
    maximum = int(amendment["maximum_pending_requests"])
    if workers < 1 or workers > maximum:
        raise ValueError(
            f"CDS worker count must be between 1 and {maximum}; observed={workers}"
        )
    return min(workers, selected_count) if selected_count else 0


def request_payload(protocol: dict[str, Any], latitude: float, longitude: float) -> dict[str, Any]:
    return {
        "variable": list(protocol["variables"]),
        "location": {"latitude": round(latitude, 5), "longitude": round(longitude, 5)},
        "date": f"{protocol['reference_start']}/{protocol['reference_end']}",
        "data_format": protocol["data_format"],
    }


def prepare(root: Path, protocol: dict[str, Any], audit: Path) -> dict[str, Any]:
    sites = pd.read_csv(root / SITE_INVENTORY, sep="\t", dtype=str)
    if len(sites) != int(protocol["site_count"]) or sites.site_id.duplicated().any():
        raise ValueError("Frozen bias-reference site inventory is not one-to-one")
    rows = []
    for row in sites.sort_values("site_id").itertuples(index=False):
        payload = request_payload(protocol, float(row.latitude), float(row.longitude))
        identity = {
            "protocol_version": protocol["protocol_version"],
            "site_id": row.site_id,
            "dataset": protocol["dataset"],
            "payload": payload,
        }
        rows.append(
            {
                "request_id": stable_sha256(identity),
                "site_id": row.site_id,
                "latitude": f"{float(row.latitude):.5f}",
                "longitude": f"{float(row.longitude):.5f}",
                "reference_start": protocol["reference_start"],
                "reference_end": protocol["reference_end"],
                "dataset": protocol["dataset"],
                "request_payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
            }
        )
    inventory = pd.DataFrame(rows)
    if inventory.request_id.duplicated().any():
        raise ValueError("Bias-reference request identifiers are not unique")
    audit.mkdir(parents=True, exist_ok=True)
    inventory_path = audit / "cds_bias_reference_request_inventory.tsv"
    atomic_tsv(inventory_path, inventory)
    freeze = {
        "status": "PASS_READY_TO_FETCH",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["_sha256"],
        "site_count": len(sites),
        "request_count": len(inventory),
        "reference_start": protocol["reference_start"],
        "reference_end": protocol["reference_end"],
        "request_concurrency": 1,
        "request_inventory_sha256": sha256_file(inventory_path),
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(audit / "cds_bias_reference_fetch_freeze.json", freeze)
    return freeze


def valid_receipt(cache: Path, request_id: str) -> dict[str, Any] | None:
    path = cache / "requests" / request_id[:2] / f"{request_id}.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        raw = cache / value["raw_path"]
        if (
            value.get("status") == "FETCHED_RAW"
            and raw.is_file()
            and raw.stat().st_size == int(value["raw_bytes"])
            and sha256_file(raw) == value["raw_sha256"]
        ):
            return value
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        pass
    return None


def payload_has_four_components(path: Path) -> bool:
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        return len(members) == 4 and all(item.file_size > 0 for item in members)


def fetch(
    root: Path,
    protocol: dict[str, Any],
    transport: dict[str, Any],
    audit: Path,
    cache: Path,
    limit: int,
    workers: int | None,
) -> dict[str, Any]:
    import cdsapi  # type: ignore

    freeze_path = audit / "cds_bias_reference_fetch_freeze.json"
    inventory_path = audit / "cds_bias_reference_request_inventory.tsv"
    if not freeze_path.is_file() or not inventory_path.is_file():
        prepare(root, protocol, audit)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("protocol_sha256") != protocol["_sha256"]
        or freeze.get("request_inventory_sha256") != sha256_file(inventory_path)
    ):
        raise ValueError("Bias-reference fetch freeze identity mismatch")
    inventory = pd.read_csv(inventory_path, sep="\t", dtype=str)
    completed_before = {
        request_id: valid_receipt(cache, request_id)
        for request_id in inventory.request_id
    }
    pending = inventory[[completed_before[value] is None for value in inventory.request_id]]
    selected = pending if limit == 0 else pending.head(limit)
    worker_count = effective_worker_count(workers, len(selected), transport)
    thread_state = threading.local()
    publication_lock = threading.Lock()

    def client_for_worker() -> Any:
        client = getattr(thread_state, "client", None)
        if client is None:
            client = cdsapi.Client(
                quiet=True,
                retry_max=int(protocol["retry_max"]),
                sleep_max=int(protocol["retry_sleep_max_seconds"]),
                timeout=int(protocol["request_timeout_seconds"]),
            )
            thread_state.client = client
        return client

    def fetch_one(row: dict[str, Any]) -> tuple[str, str | None]:
        request_id = str(row["request_id"])
        temporary = (
            cache
            / "staging"
            / f"{request_id}.{os.getpid()}.{threading.get_ident()}.zip"
        )
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.unlink(missing_ok=True)
        try:
            payload = json.loads(str(row["request_payload_json"]))
            client_for_worker().retrieve(protocol["dataset"], payload, str(temporary))
            if not temporary.is_file() or not payload_has_four_components(temporary):
                raise ValueError("CDS response is absent or lacks the four expected components")
            raw_sha256 = sha256_file(temporary)
            raw_bytes = temporary.stat().st_size
            relative = Path("raw") / raw_sha256[:2] / f"{raw_sha256}.zip"
            raw = cache / relative
            with publication_lock:
                raw.parent.mkdir(parents=True, exist_ok=True)
                if raw.is_file():
                    if raw.stat().st_size != raw_bytes or sha256_file(raw) != raw_sha256:
                        raise ValueError(
                            "Content-addressed CDS target conflicts with downloaded payload"
                        )
                    temporary.unlink(missing_ok=True)
                else:
                    os.replace(temporary, raw)
            receipt = {
                "status": "FETCHED_RAW",
                "protocol_version": protocol["protocol_version"],
                "protocol_sha256": protocol["_sha256"],
                "transport_amendment_sha256": transport["_sha256"],
                "request_worker_limit": worker_count,
                "request_id": request_id,
                "site_id": row["site_id"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "reference_start": row["reference_start"],
                "reference_end": row["reference_end"],
                "dataset": protocol["dataset"],
                "raw_path": relative.as_posix(),
                "raw_sha256": raw_sha256,
                "raw_bytes": raw_bytes,
                "credentials_or_token_logged": False,
                "phenotype_values_read": False,
                "outer_test_outcomes_read": False,
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
                "future_covariate_matrices_generated": 0,
                "future_predictions_generated": 0,
            }
            receipt_path = cache / "requests" / request_id[:2] / f"{request_id}.json"
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(receipt_path, receipt)
            return request_id, None
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            return request_id, f"{type(exc).__name__}: {exc}"

    fetched = 0
    failures: dict[str, str] = {}
    if worker_count:
        print(
            f"Starting CDS bias-reference fetch: requests={len(selected)} "
            f"workers={worker_count} account_pending_limit={transport['maximum_pending_requests']}",
            flush=True,
        )
        records = selected.to_dict(orient="records")
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="cds-bias-reference",
        ) as executor:
            future_to_row = {
                executor.submit(fetch_one, row): row for row in records
            }
            for number, future in enumerate(as_completed(future_to_row), start=1):
                row = future_to_row[future]
                request_id, error = future.result()
                if error is None:
                    fetched += 1
                    outcome = "FETCHED"
                else:
                    failures[request_id] = error
                    outcome = "FAILED_RETRYABLE"
                detail = "" if error is None else f" detail={error[:500]}"
                print(
                    f"CDS bias reference {number}/{len(selected)} "
                    f"site={row['site_id']} status={outcome}{detail}",
                    flush=True,
                )

    rows = []
    for row in inventory.itertuples(index=False):
        receipt = valid_receipt(cache, row.request_id)
        if receipt is not None:
            status = "FETCHED"
            detail = ""
        elif row.request_id in failures:
            status = "FAILED_RETRYABLE"
            detail = failures[row.request_id]
        else:
            status = "PENDING_LIMIT"
            detail = ""
        rows.append({**row._asdict(), "status": status, "detail": detail})
    index = pd.DataFrame(rows)
    cache.mkdir(parents=True, exist_ok=True)
    index_path = cache / "cds_bias_reference_fetch_index.tsv"
    atomic_tsv(index_path, index)
    counts = index.status.value_counts().sort_index().to_dict()
    complete = counts.get("FETCHED", 0) == len(index)
    result = {
        "status": "PASS",
        "run_status": "COMPLETE" if complete else "PARTIAL",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["_sha256"],
        "request_inventory_sha256": freeze["request_inventory_sha256"],
        "fetch_index_sha256": sha256_file(index_path),
        "request_count": len(index),
        "cached_before_run": sum(value is not None for value in completed_before.values()),
        "selected_pending_count": len(selected),
        "fetched_this_run": fetched,
        "completed_request_count": counts.get("FETCHED", 0),
        "archive_complete": complete,
        "request_concurrency": worker_count,
        "maximum_pending_requests": int(transport["maximum_pending_requests"]),
        "transport_amendment_sha256": transport["_sha256"],
        "status_counts": counts,
        "credentials_or_token_logged": False,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(cache / "cds_bias_reference_fetch_provenance.json", result)
    return result


def status(
    root: Path,
    protocol: dict[str, Any],
    transport: dict[str, Any],
    audit: Path,
    cache: Path,
) -> dict[str, Any]:
    inventory_path = audit / "cds_bias_reference_request_inventory.tsv"
    if not inventory_path.is_file():
        return {"status": "NOT_PREPARED", "completed_request_count": 0, "request_count": 0}
    inventory = pd.read_csv(inventory_path, sep="\t", dtype=str)
    completed = sum(valid_receipt(cache, value) is not None for value in inventory.request_id)
    return {
        "status": "PASS",
        "protocol_version": protocol["protocol_version"],
        "request_count": len(inventory),
        "completed_request_count": completed,
        "pending_request_count": len(inventory) - completed,
        "archive_complete": completed == len(inventory),
        "default_request_concurrency": int(transport["default_worker_count"]),
        "maximum_pending_requests": int(transport["maximum_pending_requests"]),
        "transport_amendment_sha256": transport["_sha256"],
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "fetch", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--transport-amendment",
        type=Path,
        default=DEFAULT_TRANSPORT_AMENDMENT,
    )
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = load_protocol(root, args.protocol)
    transport = load_transport_amendment(root, args.transport_amendment, protocol)
    audit = resolve(root, args.audit)
    cache = resolve(root, args.cache)
    if args.command == "prepare":
        result = prepare(root, protocol, audit)
    elif args.command == "fetch":
        result = fetch(
            root,
            protocol,
            transport,
            audit,
            cache,
            args.limit,
            args.workers,
        )
    else:
        result = status(root, protocol, transport, audit, cache)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
