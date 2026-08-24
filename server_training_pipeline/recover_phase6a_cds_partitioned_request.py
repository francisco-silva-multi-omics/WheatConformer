from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import zipfile

import pandas as pd

from server_training_pipeline.phase6a_environment_source_recovery import (
    CDS_DATASET,
    CDS_REQUEST_TIMEOUT_SECONDS,
    cds_api_payload,
    read_json,
    sha256_file,
    write_json_atomic,
)


PROTOCOL_VERSION = "phase6a_cds_partitioned_request_recovery_v1"
DEFAULT_CONTRACT = Path("audit/v2/phase6a_environment_source_contract_v10")
DEFAULT_CACHE = Path("environment/v2/phase6a_cds_era5_land_daily_full_v1")


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def year_partitions(start: str, end: str) -> list[tuple[str, str]]:
    first = pd.Timestamp(start)
    last = pd.Timestamp(end)
    parts = []
    current = first
    while current <= last:
        part_end = min(last, pd.Timestamp(year=current.year, month=12, day=31))
        parts.append((current.strftime("%Y-%m-%d"), part_end.strftime("%Y-%m-%d")))
        current = part_end + pd.Timedelta(days=1)
    return parts


def deterministic_bundle(parts: list[dict], original_payload: dict) -> bytes:
    output = io.BytesIO()
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": CDS_DATASET,
        "original_request_payload": original_payload,
        "parts": [
            {
                key: value
                for key, value in part.items()
                if key not in {"bytes"}
            }
            for part in parts
        ],
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        manifest_info = zipfile.ZipInfo("manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
        manifest_info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(
            manifest_info,
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        for index, part in enumerate(parts):
            name = f"parts/{index:03d}_{part['raw_sha256']}.zip"
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, part["bytes"])
    return output.getvalue()


def recover(root: Path, contract_dir: Path, cache: Path, request_id: str) -> dict:
    import cdsapi  # type: ignore

    contract_path = contract_dir / "environment_source_contract.json"
    inventory_path = contract_dir / "cds_era5_land_request_inventory.tsv"
    contract = read_json(contract_path)
    expected = dict(contract.get("artifacts", {})).get(inventory_path.name)
    if contract.get("status") != "PASS" or sha256_file(inventory_path) != expected:
        raise SystemExit("CDS recovery input is not bound to the frozen source contract")
    inventory = pd.read_csv(inventory_path, sep="\t", dtype=str)
    selected = inventory[inventory["request_id"].eq(request_id)]
    if len(selected) != 1:
        raise SystemExit(f"Expected exactly one frozen CDS request for {request_id}")
    row = selected.iloc[0]
    metadata_path = cache / "requests" / request_id[:2] / f"{request_id}.json"
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        raw = cache / str(metadata.get("raw_path", ""))
        if raw.is_file() and sha256_file(raw) == metadata.get("raw_sha256"):
            return {"status": "CACHED", **metadata}

    original_payload = cds_api_payload(row)
    partitions = year_partitions(row["request_start_date"], row["request_end_date"])
    client = cdsapi.Client(quiet=True, retry_max=5, sleep_max=30, timeout=CDS_REQUEST_TIMEOUT_SECONDS)
    staging = cache / "partitioned_staging" / request_id
    staging.mkdir(parents=True, exist_ok=True)
    parts = []
    try:
        for index, (part_start, part_end) in enumerate(partitions):
            target = staging / f"part_{index:03d}.zip"
            payload = dict(original_payload)
            payload["date"] = f"{part_start}/{part_end}"
            print(
                json.dumps(
                    {
                        "request_id": request_id,
                        "part": index + 1,
                        "part_count": len(partitions),
                        "date": payload["date"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            client.retrieve(CDS_DATASET, payload, str(target))
            if not target.is_file() or target.stat().st_size == 0 or not zipfile.is_zipfile(target):
                raise ValueError(f"CDS partition {index} is absent, empty or not ZIP")
            with zipfile.ZipFile(target) as archive:
                members = [info for info in archive.infolist() if not info.is_dir()]
                if not members or any(info.file_size <= 0 for info in members):
                    raise ValueError(f"CDS partition {index} contains empty members")
            raw_bytes = target.read_bytes()
            raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            relative_part = Path("raw_parts") / raw_sha256[:2] / f"{raw_sha256}.zip"
            part_path = cache / relative_part
            part_path.parent.mkdir(parents=True, exist_ok=True)
            if not part_path.exists():
                temporary = part_path.with_suffix(f".tmp.{os.getpid()}.zip")
                temporary.write_bytes(raw_bytes)
                temporary.replace(part_path)
            parts.append(
                {
                    "part_index": index,
                    "date_start": part_start,
                    "date_end": part_end,
                    "request_payload": payload,
                    "raw_path": relative_part.as_posix(),
                    "raw_sha256": raw_sha256,
                    "raw_bytes": len(raw_bytes),
                    "member_count": len(members),
                    "bytes": raw_bytes,
                }
            )
        bundle = deterministic_bundle(parts, original_payload)
        bundle_sha256 = hashlib.sha256(bundle).hexdigest()
        relative = Path("raw") / bundle_sha256[:2] / f"{bundle_sha256}.bin"
        raw_path = cache / relative
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists():
            temporary = raw_path.with_suffix(f".tmp.{os.getpid()}.bin")
            temporary.write_bytes(bundle)
            temporary.replace(raw_path)
        metadata = {
            "request_id": request_id,
            "dataset": CDS_DATASET,
            "request_payload": original_payload,
            "raw_path": relative.as_posix(),
            "raw_sha256": bundle_sha256,
            "raw_bytes": len(bundle),
            "transport": "CDS_PARTITIONED_CALENDAR_YEAR_EXACT",
            "container_format": "deterministic_zip_of_original_cds_zip_parts",
            "partition_count": len(parts),
            "partitions": [
                {key: value for key, value in part.items() if key != "bytes"}
                for part in parts
            ],
            "protocol_version": PROTOCOL_VERSION,
            "recovery_sha256": sha256_file(Path(__file__)),
            "phenotype_values_read": False,
            "outer_test_outcomes_read": False,
            "outer_test_metrics_read": False,
            "final_holdout_outcomes_read": False,
            "future_covariate_matrices_generated": 0,
            "future_predictions_generated": 0,
        }
        write_json_atomic(metadata_path, metadata)
        return {"status": "FETCHED_RAW_PARTITIONED", **metadata}
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--request-id", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = recover(
        root,
        resolve(root, args.contract_dir),
        resolve(root, args.cache_dir),
        args.request_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
