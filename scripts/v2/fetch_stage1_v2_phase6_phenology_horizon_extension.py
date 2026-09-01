from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import pandas as pd

from server_training_pipeline.phase6a_environment_source_recovery import (
    CDS_DATASET,
    cds_api_payload,
    credential_present,
    read_json,
    sha256_file,
    write_json_atomic,
    write_tsv,
)


CONTRACT = Path(
    "audit/v2/stage1_v2_phase6_phenology_readiness_v1/horizon_extension_contract"
)
CDS_CACHE = Path("environment/v2/e_projection_daily_horizon_v2_cds_extension_v1")
MAXIMUM_CDS_WORKERS = 20


def _validate_contract(contract_dir: Path, inventory_name: str) -> tuple[dict, Path]:
    contract_path = contract_dir / "environment_source_contract.json"
    inventory_path = contract_dir / inventory_name
    contract = read_json(contract_path)
    expected = contract.get("artifacts", {}).get(inventory_name)
    if contract.get("status") != "PASS" or not expected:
        raise ValueError(f"Extension inventory is not bound: {inventory_name}")
    if sha256_file(inventory_path) != expected:
        raise ValueError(f"Extension inventory checksum changed: {inventory_name}")
    return contract, inventory_path


def _cached_cds_record(cache: Path, row: pd.Series) -> dict[str, Any] | None:
    request_id = str(row["request_id"])
    metadata_path = cache / "requests" / request_id[:2] / f"{request_id}.json"
    if not metadata_path.is_file():
        return None
    metadata = read_json(metadata_path)
    candidate = cache / str(metadata.get("raw_path", ""))
    if not candidate.is_file() or sha256_file(candidate) != metadata.get("raw_sha256"):
        return None
    return {
        **row.to_dict(),
        "status": "CACHED",
        "detail": "",
        "raw_path": str(metadata["raw_path"]),
        "raw_sha256": str(metadata["raw_sha256"]),
        "raw_bytes": int(candidate.stat().st_size),
    }


def _fetch_cds_one(cache: Path, row: pd.Series) -> dict[str, Any]:
    import cdsapi  # type: ignore

    request_id = str(row["request_id"])
    temporary = cache / f".{request_id}.download"
    if temporary.exists():
        temporary.unlink()
    try:
        client = cdsapi.Client(
            quiet=True,
            retry_max=5,
            sleep_max=30,
            timeout=120,
        )
        client.retrieve(CDS_DATASET, cds_api_payload(row), str(temporary))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise ValueError("CDS retrieval produced an empty target")
        raw_sha256 = sha256_file(temporary)
        relative = Path("raw") / raw_sha256[:2] / f"{raw_sha256}.bin"
        target = cache / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            temporary.unlink()
        else:
            temporary.replace(target)
        metadata_path = cache / "requests" / request_id[:2] / f"{request_id}.json"
        write_json_atomic(
            metadata_path,
            {
                "request_id": request_id,
                "dataset": CDS_DATASET,
                "request_payload": cds_api_payload(row),
                "raw_path": relative.as_posix(),
                "raw_sha256": raw_sha256,
                "raw_bytes": target.stat().st_size,
            },
        )
        return {
            **row.to_dict(),
            "status": "FETCHED_RAW",
            "detail": "",
            "raw_path": relative.as_posix(),
            "raw_sha256": raw_sha256,
            "raw_bytes": int(target.stat().st_size),
        }
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        return {
            **row.to_dict(),
            "status": "FAILED_RETRYABLE",
            "detail": f"{type(exc).__name__}:{exc}",
            "raw_path": "",
            "raw_sha256": "",
            "raw_bytes": 0,
        }


def fetch_cds(
    contract_dir: Path,
    cache: Path,
    limit: int,
    workers: int,
) -> dict[str, Any]:
    contract, inventory_path = _validate_contract(
        contract_dir, "cds_era5_land_request_inventory.tsv"
    )
    if not credential_present():
        raise ValueError("CDS credentials are unavailable")
    try:
        import cdsapi  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise ValueError("cdsapi is not installed") from exc
    if workers < 1 or workers > MAXIMUM_CDS_WORKERS:
        raise ValueError(f"CDS workers must be between 1 and {MAXIMUM_CDS_WORKERS}")
    cache.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(inventory_path, sep="\t", dtype=str)
    records: dict[str, dict[str, Any]] = {}
    pending: list[pd.Series] = []
    for _, row in inventory.iterrows():
        cached = _cached_cds_record(cache, row)
        if cached is None:
            pending.append(row)
        else:
            records[str(row["request_id"])] = cached
    selected = pending if limit <= 0 else pending[:limit]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_cds_one, cache, row): row for row in selected}
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records[str(record["request_id"])] = record
            detail = (
                f" detail={record['detail']}"
                if record["status"] == "FAILED_RETRYABLE"
                else ""
            )
            print(
                f"CDS {completed}/{len(selected)} {record['status']} "
                f"{record['request_id']}{detail}",
                flush=True,
            )
    selected_ids = {str(row["request_id"]) for row in selected}
    rows = []
    for _, row in inventory.iterrows():
        request_id = str(row["request_id"])
        record = records.get(request_id)
        if record is None:
            record = {
                **row.to_dict(),
                "status": "PENDING_LIMIT",
                "detail": "not selected in this bounded run",
                "raw_path": "",
                "raw_sha256": "",
                "raw_bytes": 0,
            }
        elif request_id not in selected_ids and record["status"] == "CACHED":
            record["detail"] = ""
        rows.append(record)
    index = pd.DataFrame(rows)
    index_path = cache / "cds_era5_land_fetch_index.tsv"
    write_tsv(index_path, index)
    completed_count = int(index["status"].isin(["FETCHED_RAW", "CACHED"]).sum())
    provenance = {
        "status": "PASS",
        "run_status": (
            "COMPLETE_RAW_ARCHIVE"
            if completed_count == len(index)
            else "PARTIAL"
        ),
        "protocol_version": "stage1_v2_phase6_phenology_daily_horizon_extension_fetch_v1",
        "selection_data": contract["selection_data"],
        "request_count": len(index),
        "selected_pending_count": len(selected),
        "completed_raw_request_count": completed_count,
        "raw_archive_complete": completed_count == len(index),
        "request_concurrency": workers,
        "credentials_or_token_logged": False,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "future_SSP_values_read": False,
        "future_predictions_generated": 0,
        "status_counts": {
            str(key): int(value) for key, value in index["status"].value_counts().items()
        },
        "contract_sha256": sha256_file(
            contract_dir / "environment_source_contract.json"
        ),
        "inventory_sha256": sha256_file(inventory_path),
        "index_sha256": sha256_file(index_path),
    }
    write_json_atomic(cache / "phenology_horizon_cds_fetch_provenance.json", provenance)
    return provenance


def status(root: Path, contract_dir: Path, cds_cache: Path) -> dict:
    contract, reuse_inventory = _validate_contract(
        contract_dir, "certified_CDS_reference_reuse_inventory.tsv"
    )
    _, cds_inventory = _validate_contract(
        contract_dir, "cds_era5_land_request_inventory.tsv"
    )
    _, masked_inventory = _validate_contract(
        contract_dir, "masked_CDS_reference_inventory.tsv"
    )
    _, all_inventory = _validate_contract(contract_dir, "daily_request_inventory.tsv")

    def resolved(path: Path, statuses: set[str]) -> int:
        if not path.is_file():
            return 0
        frame = pd.read_csv(path, sep="\t", dtype=str)
        return int(frame["status"].isin(statuses).sum())

    def cached_requests(cache: Path, request_ids: set[str]) -> int:
        request_root = cache / "requests"
        if not request_root.is_dir():
            return 0
        return sum(
            1
            for request_id in request_ids
            if (request_root / request_id[:2] / f"{request_id}.json").is_file()
        )

    cds = pd.read_csv(cds_inventory, sep="\t", dtype=str)
    reuse = pd.read_csv(reuse_inventory, sep="\t", dtype=str)
    masked = pd.read_csv(masked_inventory, sep="\t", dtype=str)
    all_requests = pd.read_csv(all_inventory, sep="\t", dtype=str)
    cds_ids = set(cds["request_id"])
    cds_resolved = max(
        resolved(
            cds_cache / "cds_era5_land_fetch_index.tsv",
            {"FETCHED_RAW", "CACHED"},
        ),
        cached_requests(cds_cache, cds_ids),
    )

    result = {
        "status": "PASS",
        "protocol_version": "stage1_v2_phase6_phenology_daily_horizon_extension_status_v2_reuse_first",
        "extension_request_count": len(all_requests),
        "certified_CDS_reference_reuse_count": len(reuse),
        "explicit_missing_weather_mask_count": len(masked),
        "new_CDS_fetch_resolved": cds_resolved,
        "new_CDS_fetch_total": len(cds),
        "new_OpenMeteo_fetch_total": 0,
        "cross_provider_certification_reused": contract.get(
            "cross_provider_certification_sha256"
        )
        is not None,
        "phenotype_values_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "future_predictions_generated": 0,
    }
    result["required_raw_archives_complete"] = (
        len(reuse) + result["new_CDS_fetch_resolved"]
        == len(all_requests) - len(masked)
    )
    result["extension_source_resolution_complete"] = (
        len(reuse) + len(masked) + result["new_CDS_fetch_resolved"]
        == len(all_requests)
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch the phenology daily-horizon extension")
    parser.add_argument("command", choices=["fetch-cds", "status"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--contract-dir", type=Path, default=CONTRACT)
    parser.add_argument("--cds-cache", type=Path, default=CDS_CACHE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    contract = resolve(args.contract_dir)
    cds_cache = resolve(args.cds_cache)
    if args.command == "fetch-cds":
        result = fetch_cds(contract, cds_cache, args.limit, args.workers)
    else:
        result = status(root, contract, cds_cache)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
