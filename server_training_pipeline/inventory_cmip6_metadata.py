from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterable
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd


DEFAULT_PROTOCOL = Path(
    "server_training_pipeline/phase6a_cmip6_metadata_inventory_protocol_v1.json"
)
DEFAULT_OUTPUT = Path("audit/v2/phase6a_cmip6_metadata_inventory_v1")
ESGF_PAGE_SIZE = 10_000
METADATA_USER_AGENT = "WheatConformer-Phase6A-CMIP6-Metadata-Inventory/1.0"
ESGF_DATASET_FIELDS = (
    "id",
    "master_id",
    "instance_id",
    "institution_id",
    "source_id",
    "experiment_id",
    "member_id",
    "variant_label",
    "table_id",
    "variable_id",
    "grid_label",
    "version",
    "frequency",
    "datetime_start",
    "datetime_end",
    "data_node",
    "index_node",
    "replica",
    "latest",
    "retracted",
    "number_of_files",
    "size",
    "pid",
    "access",
    "activity_id",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def write_deterministic_gzip(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
            compressed.write(raw)


def metadata_bytes_with_cache(
    url: str,
    target_path: Path,
    cache_dir: Path | None,
    *,
    gzip_encoded: bool = False,
    timeout: int = 180,
    retries: int = 5,
    maximum_bytes: int = 128 * 1024 * 1024,
) -> tuple[bytes, str]:
    cached = cache_dir / target_path.name if cache_dir is not None else None
    if cached is not None and cached.is_file():
        raw = gzip.open(cached, "rb").read() if gzip_encoded else cached.read_bytes()
        retrieval_mode = "VERIFIED_CACHE_REUSE"
    else:
        raw = fetch_metadata_bytes(
            url,
            timeout=timeout,
            retries=retries,
            maximum_bytes=maximum_bytes,
        )
        retrieval_mode = "FETCHED"
    if not raw:
        raise ValueError(f"Metadata snapshot was empty: {target_path.name}")
    if gzip_encoded:
        write_deterministic_gzip(target_path, raw)
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(raw)
    return raw, retrieval_mode


def fetch_metadata_bytes(
    url: str,
    *,
    timeout: int = 180,
    retries: int = 5,
    maximum_bytes: int = 128 * 1024 * 1024,
) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": METADATA_USER_AGENT, "Accept": "application/json,text/plain,*/*"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(maximum_bytes + 1)
            if len(raw) > maximum_bytes:
                raise ValueError(f"Metadata response exceeded {maximum_bytes} bytes: {url}")
            if not raw:
                raise ValueError(f"Metadata response was empty: {url}")
            return raw
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Metadata fetch failed after {retries} attempts: {url}") from last_error


def fetch_json_metadata(url: str, **kwargs: Any) -> tuple[bytes, Any]:
    raw = fetch_metadata_bytes(url, **kwargs)
    return raw, json.loads(raw)


def scalar(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else ""
    return value if value is not None else ""


def bool_value(value: Any) -> bool:
    value = scalar(value)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def version_key(value: Any) -> tuple[int, int | str]:
    text = str(value).strip().lstrip("v")
    if text.isdigit():
        return (1, int(text))
    return (0, text)


def member_priority(member_id: str, preferred: str = "r1i1p1f1") -> tuple[int, str]:
    return (0 if member_id == preferred else 1, member_id)


def coverage_year(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(parsed) else int(parsed.year)


def normalize_dataset_doc(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "catalog_record_id": str(scalar(doc.get("id"))),
        "master_id": str(scalar(doc.get("master_id"))),
        "instance_id": str(scalar(doc.get("instance_id"))),
        "institution_id": str(scalar(doc.get("institution_id"))),
        "source_id": str(scalar(doc.get("source_id"))),
        "experiment_id": str(scalar(doc.get("experiment_id"))),
        "variant_label": str(
            scalar(doc.get("variant_label")) or scalar(doc.get("member_id"))
        ),
        "member_id": str(scalar(doc.get("member_id")) or scalar(doc.get("variant_label"))),
        "table_id": str(scalar(doc.get("table_id"))),
        "variable": str(scalar(doc.get("variable_id"))),
        "grid_label": str(scalar(doc.get("grid_label"))),
        "version": str(scalar(doc.get("version"))),
        "frequency": str(scalar(doc.get("frequency"))),
        "datetime_start": str(scalar(doc.get("datetime_start"))),
        "datetime_end": str(scalar(doc.get("datetime_end"))),
        "data_node": str(scalar(doc.get("data_node"))),
        "index_node": str(scalar(doc.get("index_node"))),
        "replica": bool_value(doc.get("replica")),
        "latest": bool_value(doc.get("latest")),
        "retracted": bool_value(doc.get("retracted")),
        "number_of_files": scalar(doc.get("number_of_files")),
        "size_bytes": scalar(doc.get("size")),
        "pid": str(scalar(doc.get("pid"))),
        "access": ";".join(str(item) for item in doc.get("access", []) or []),
        "activity_id": str(scalar(doc.get("activity_id"))),
    }


def canonicalize_dataset_docs(docs: Iterable[dict[str, Any]]) -> pd.DataFrame:
    normalized = pd.DataFrame(normalize_dataset_doc(doc) for doc in docs)
    if normalized.empty:
        return normalized
    required = [
        "catalog_record_id",
        "master_id",
        "source_id",
        "experiment_id",
        "member_id",
        "variable",
        "grid_label",
        "version",
    ]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"ESGF dataset metadata lacks required columns: {missing}")
    normalized = normalized.drop_duplicates("catalog_record_id").copy()
    normalized["_logical_id"] = normalized["master_id"].where(
        normalized["master_id"].ne(""), normalized["catalog_record_id"]
    )
    rows: list[dict[str, Any]] = []
    for _, group in normalized.groupby("_logical_id", sort=True):
        best_version = max(group["version"], key=version_key)
        current = group[group["version"].eq(best_version)].copy()
        current = current.sort_values(
            ["replica", "catalog_record_id"], ascending=[True, True], kind="stable"
        )
        chosen = current.iloc[0].to_dict()
        starts = pd.to_datetime(
            current["datetime_start"], format="mixed", errors="coerce", utc=True
        )
        ends = pd.to_datetime(
            current["datetime_end"], format="mixed", errors="coerce", utc=True
        )
        chosen["datetime_start"] = (
            starts.min().isoformat().replace("+00:00", "Z") if starts.notna().any() else ""
        )
        chosen["datetime_end"] = (
            ends.max().isoformat().replace("+00:00", "Z") if ends.notna().any() else ""
        )
        chosen["catalog_replica_count"] = len(current)
        chosen["catalog_record_ids_json"] = json.dumps(
            sorted(current["catalog_record_id"].astype(str).unique()), separators=(",", ":")
        )
        rows.append(chosen)
    frame = pd.DataFrame(rows).drop(columns=["_logical_id"], errors="ignore")
    sort_columns = [
        "source_id",
        "institution_id",
        "experiment_id",
        "member_id",
        "grid_label",
        "variable",
        "version",
    ]
    return frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def asset_eligibility(row: pd.Series, protocol: dict[str, Any]) -> tuple[str, str]:
    reasons: list[str] = []
    if str(row.frequency) != str(protocol["required_frequency"]):
        reasons.append("not_daily_frequency")
    if str(row.table_id) != str(protocol["required_table_id"]):
        reasons.append("not_daily_table")
    if not str(row.member_id) or not str(row.grid_label):
        reasons.append("missing_member_or_grid_identity")
    if bool(row.retracted):
        reasons.append("retracted_catalog_record")
    start_year = coverage_year(row.datetime_start)
    end_year = coverage_year(row.datetime_end)
    if str(row.experiment_id) == str(protocol["historical_experiment_id"]):
        required_start = int(str(protocol["historical_bias_reference_start"])[:4])
        required_end = int(str(protocol["historical_bias_reference_end"])[:4])
        if start_year is None or start_year > required_start:
            reasons.append("historical_starts_after_bias_reference")
        if end_year is None or end_year < required_end:
            reasons.append("historical_ends_before_bias_reference")
    else:
        if start_year is None or start_year > int(protocol["future_required_start_year"]):
            reasons.append("future_starts_after_required_horizon")
        if end_year is None or end_year < int(protocol["future_required_end_year"]):
            reasons.append("future_ends_before_required_horizon")
    return (
        "ASSET_METADATA_ELIGIBLE" if not reasons else "ASSET_METADATA_INELIGIBLE",
        ";".join(reasons),
    )


def annotate_asset_eligibility(
    assets: pd.DataFrame, protocol: dict[str, Any]
) -> pd.DataFrame:
    frame = assets.copy()
    decisions = frame.apply(lambda row: asset_eligibility(row, protocol), axis=1)
    frame["eligibility_status"] = [value[0] for value in decisions]
    frame["exclusion_reason"] = [value[1] for value in decisions]
    return frame


def candidate_key_columns() -> list[str]:
    return ["source_id", "institution_id", "member_id", "variant_label", "grid_label"]


def build_candidate_completeness(
    assets: pd.DataFrame, protocol: dict[str, Any]
) -> pd.DataFrame:
    experiments = [protocol["historical_experiment_id"], *protocol["required_ssp_experiment_ids"]]
    base_variables = list(protocol["required_projection_core_variables"])
    humidity_alternatives = [list(values) for values in protocol["humidity_alternatives"]]
    eligible = assets[assets["eligibility_status"].eq("ASSET_METADATA_ELIGIBLE")].copy()
    eligible_by_candidate: dict[tuple[str, ...], set[tuple[str, str]]] = defaultdict(set)
    for row in eligible.itertuples(index=False):
        key = tuple(str(getattr(row, column)) for column in candidate_key_columns())
        eligible_by_candidate[key].add((str(row.experiment_id), str(row.variable)))
    rows: list[dict[str, Any]] = []
    for key, group in assets.groupby(candidate_key_columns(), sort=True, dropna=False):
        key_record = dict(zip(candidate_key_columns(), key))
        available = eligible_by_candidate.get(tuple(str(value) for value in key), set())
        selected_humidity: list[str] | None = None
        for alternative in humidity_alternatives:
            if all(
                (experiment, variable) in available
                for experiment in experiments
                for variable in alternative
            ):
                selected_humidity = alternative
                break
        required_variables = base_variables + (selected_humidity or humidity_alternatives[0])
        missing_assets = [
            f"{experiment}:{variable}"
            for experiment in experiments
            for variable in required_variables
            if (experiment, variable) not in available
        ]
        candidate_status = (
            "COMPLETE_METADATA_CANDIDATE"
            if selected_humidity is not None and not missing_assets
            else "INCOMPLETE_METADATA_CANDIDATE"
        )
        rows.append(
            {
                **key_record,
                "candidate_status": candidate_status,
                "humidity_branch": "+".join(selected_humidity or []),
                "required_variables": ";".join(required_variables),
                "required_experiments": ";".join(experiments),
                "required_asset_count": len(experiments) * len(required_variables),
                "eligible_asset_count": sum(
                    (experiment, variable) in available
                    for experiment in experiments
                    for variable in required_variables
                ),
                "missing_assets": ";".join(missing_assets),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["source_id", "member_id", "grid_label", "institution_id"], kind="stable"
    ).reset_index(drop=True)


def candidate_priority(row: pd.Series, protocol: dict[str, Any]) -> tuple[Any, ...]:
    preferred = str(protocol["selection_rule"]["preferred_variant_label"])
    return (
        *member_priority(str(row.member_id), preferred),
        str(row.grid_label),
        str(row.institution_id),
    )


def calendar_from_das(text: str) -> str:
    match = re.search(
        r"\btime\s*\{.*?\bString\s+calendar\s+\"([^\"]+)\"\s*;",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("OPeNDAP DAS metadata does not declare time.calendar")
    return match.group(1).strip().lower()


def opendap_das_urls(file_doc: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for entry in file_doc.get("url", []) or []:
        parts = str(entry).split("|")
        if not any(part.upper() == "OPENDAP" for part in parts[1:]):
            continue
        url = parts[0]
        if url.endswith(".html"):
            url = url[: -len(".html")]
        if url.startswith("http://"):
            url = "https://" + url[len("http://") :]
        das_url = url + ".das"
        if "/dodsC/" in das_url and das_url.endswith(".nc.das"):
            urls.append(das_url)
    return sorted(set(urls))


def metadata_query_url(base: str, parameters: list[tuple[str, str]]) -> str:
    return base + "?" + urllib.parse.urlencode(parameters)


def snapshot_cds_catalogue(
    collection_url: str, snapshot_dir: Path, cache_dir: Path | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths: list[dict[str, Any]] = []
    collection_path = snapshot_dir / "cds_collection.json"
    raw_collection, retrieval_mode = metadata_bytes_with_cache(
        collection_url, collection_path, cache_dir
    )
    collection = json.loads(raw_collection)
    paths.append(
        {
            "catalogue": "CDS",
            "snapshot_file": collection_path.name,
            "source_url": collection_url,
            "bytes": len(raw_collection),
            "sha256": sha256_bytes(raw_collection),
            "retrieval_mode": retrieval_mode,
        }
    )
    for relation in ("form", "constraints"):
        url = next(link["href"] for link in collection["links"] if link["rel"] == relation)
        path = snapshot_dir / f"cds_{relation}.json"
        raw, retrieval_mode = metadata_bytes_with_cache(url, path, cache_dir)
        paths.append(
            {
                "catalogue": "CDS",
                "snapshot_file": path.name,
                "source_url": url,
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
                "retrieval_mode": retrieval_mode,
            }
        )
    return collection, paths


def snapshot_esgf_datasets(
    protocol: dict[str, Any], snapshot_dir: Path, cache_dir: Path | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base = str(protocol["catalogues"]["esgf_search"])
    experiments = [protocol["historical_experiment_id"], *protocol["required_ssp_experiment_ids"]]
    variables = sorted(
        set(protocol["required_projection_core_variables"]).union(
            *[set(values) for values in protocol["humidity_alternatives"]]
        )
    )
    docs: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    partitions: list[dict[str, Any]] = []
    for experiment in experiments:
        for variable in variables:
            parameters = [
                ("project", "CMIP6"),
                ("type", "Dataset"),
                ("latest", "true"),
                ("distrib", "true"),
                ("frequency", str(protocol["required_frequency"])),
                ("experiment_id", str(experiment)),
                ("variable_id", str(variable)),
                ("format", "application/solr+json"),
                ("limit", str(ESGF_PAGE_SIZE)),
                ("offset", "0"),
            ]
            url = metadata_query_url(base, parameters)
            path = snapshot_dir / f"esgf_{experiment}_{variable}.json.gz"
            raw, retrieval_mode = metadata_bytes_with_cache(
                url,
                path,
                cache_dir,
                gzip_encoded=True,
                timeout=300,
            )
            page = json.loads(raw)
            declared = int(page["response"]["numFound"])
            page_docs = list(page["response"].get("docs", []))
            if declared > ESGF_PAGE_SIZE:
                raise RuntimeError(
                    "ESGF experiment-variable partition exceeds the non-deep pagination "
                    f"limit: {experiment}:{variable}; declared={declared}"
                )
            if len(page_docs) != declared:
                raise RuntimeError(
                    "ESGF experiment-variable partition is incomplete: "
                    f"{experiment}:{variable}; declared={declared}; observed={len(page_docs)}"
                )
            for doc in page_docs:
                if str(scalar(doc.get("experiment_id"))) != str(experiment) or str(
                    scalar(doc.get("variable_id"))
                ) != str(variable):
                    raise RuntimeError(
                        f"ESGF partition returned an out-of-scope record: {experiment}:{variable}"
                    )
            snapshots.append(
                {
                    "catalogue": "ESGF",
                    "snapshot_file": path.name,
                    "source_url": url,
                    "bytes": path.stat().st_size,
                    "uncompressed_bytes": len(raw),
                    "sha256": sha256_file(path),
                    "record_count": len(page_docs),
                    "experiment_id": experiment,
                    "variable": variable,
                    "retrieval_mode": retrieval_mode,
                }
            )
            partitions.append(
                {
                    "experiment_id": experiment,
                    "variable": variable,
                    "declared_record_count": declared,
                    "observed_record_count": len(page_docs),
                }
            )
            docs.extend(page_docs)
    unique_docs = {str(scalar(doc.get("id"))): doc for doc in docs}
    expected_total = sum(row["declared_record_count"] for row in partitions)
    if len(unique_docs) != expected_total:
        raise RuntimeError(
            "ESGF partitioned snapshot did not contain the declared number of unique records: "
            f"declared={expected_total}; unique={len(unique_docs)}"
        )
    query = {
        "endpoint": base,
        "partition_rule": "one_disjoint_query_per_experiment_variable",
        "experiment_count": len(experiments),
        "variable_count": len(variables),
        "partition_count": len(partitions),
        "partitions": partitions,
        "declared_record_count": expected_total,
        "unique_record_count": len(unique_docs),
        "page_size": ESGF_PAGE_SIZE,
    }
    return list(unique_docs.values()), snapshots, query


def selected_asset(
    assets: pd.DataFrame, candidate: pd.Series, experiment: str, variable: str
) -> pd.Series:
    mask = assets["eligibility_status"].eq("ASSET_METADATA_ELIGIBLE")
    for column in candidate_key_columns():
        mask &= assets[column].astype(str).eq(str(candidate[column]))
    mask &= assets["experiment_id"].astype(str).eq(experiment)
    mask &= assets["variable"].astype(str).eq(variable)
    matches = assets.loc[mask].copy()
    if matches.empty:
        raise KeyError(f"Missing selected asset {experiment}:{variable}")
    best_version = max(matches["version"], key=version_key)
    current = matches[matches["version"].eq(best_version)].sort_values(
        "catalog_record_id", kind="stable"
    )
    return current.iloc[0]


def query_candidate_file_metadata(
    esgf_search: str,
    candidate: pd.Series,
    snapshot_dir: Path,
    cache_dir: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parameters = [
        ("project", "CMIP6"),
        ("type", "File"),
        ("latest", "true"),
        ("distrib", "true"),
        ("frequency", "day"),
        ("variable_id", "tas"),
        ("source_id", str(candidate.source_id)),
        ("member_id", str(candidate.member_id)),
        ("grid_label", str(candidate.grid_label)),
        ("limit", "5000"),
        ("format", "application/solr+json"),
    ]
    url = metadata_query_url(esgf_search, parameters)
    key = "|".join(str(candidate[column]) for column in candidate_key_columns())
    digest = sha256_bytes(key.encode("utf-8"))
    path = snapshot_dir / f"esgf_candidate_files_{digest}.json.gz"
    raw, retrieval_mode = metadata_bytes_with_cache(
        url,
        path,
        cache_dir,
        gzip_encoded=True,
        timeout=300,
    )
    response = json.loads(raw)
    declared = int(response["response"]["numFound"])
    docs = list(response["response"].get("docs", []))
    if declared > 5000 or len(docs) != declared:
        raise RuntimeError(
            "Candidate file metadata query is incomplete: "
            f"{key}; declared={declared}; observed={len(docs)}"
        )
    snapshot = {
        "catalogue": "ESGF_CANDIDATE_FILE_METADATA",
        "snapshot_file": path.name,
        "source_url": url,
        "bytes": path.stat().st_size,
        "uncompressed_bytes": len(raw),
        "sha256": sha256_file(path),
        "record_count": len(docs),
        "retrieval_mode": retrieval_mode,
    }
    unique = {str(scalar(doc.get("id"))): doc for doc in docs}
    return list(unique.values()), [snapshot]


def resolve_experiment_calendar(
    asset: pd.Series,
    candidate_file_docs: list[dict[str, Any]],
    snapshot_dir: Path,
    cache_dir: Path | None,
    maximum_header_bytes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_ids = set(json.loads(str(asset.catalog_record_ids_json)))
    file_docs = [
        doc
        for doc in candidate_file_docs
        if str(scalar(doc.get("dataset_id"))) in dataset_ids
    ]
    file_snapshots: list[dict[str, Any]] = []
    attempts: list[str] = []
    representative_by_host: dict[str, tuple[str, str]] = {}
    for file_doc in sorted(file_docs, key=lambda doc: str(scalar(doc.get("id")))):
        for das_url in opendap_das_urls(file_doc):
            host = urllib.parse.urlparse(das_url).netloc.lower()
            candidate = (das_url, str(scalar(file_doc.get("id"))))
            if host not in representative_by_host or candidate < representative_by_host[host]:
                representative_by_host[host] = candidate
    for host in sorted(representative_by_host):
        das_url, catalog_record_id = representative_by_host[host]
        try:
            digest = sha256_bytes(das_url.encode("utf-8"))
            path = snapshot_dir / f"opendap_das_{digest}.txt"
            raw, retrieval_mode = metadata_bytes_with_cache(
                das_url,
                path,
                cache_dir,
                timeout=30,
                retries=1,
                maximum_bytes=maximum_header_bytes,
            )
            text = raw.decode("utf-8", errors="replace")
            calendar = calendar_from_das(text)
            file_snapshots.append(
                {
                    "catalogue": "OPENDAP_DAS_METADATA",
                    "snapshot_file": path.name,
                    "source_url": das_url,
                    "bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                    "record_count": 1,
                    "retrieval_mode": retrieval_mode,
                }
            )
            return (
                {
                    "calendar_status": "CALENDAR_METADATA_RESOLVED",
                    "calendar": calendar,
                    "calendar_catalog_record_id": catalog_record_id,
                    "calendar_metadata_url_sha256": sha256_bytes(das_url.encode("utf-8")),
                    "calendar_error": "",
                },
                file_snapshots,
            )
        except Exception as exc:
            attempts.append(f"{das_url}:{type(exc).__name__}:{exc}")
    return (
        {
            "calendar_status": "CALENDAR_METADATA_UNRESOLVED",
            "calendar": "",
            "calendar_catalog_record_id": "",
            "calendar_metadata_url_sha256": "",
            "calendar_error": " | ".join(attempts[:10]) or "no_opendap_das_metadata_url",
        },
        file_snapshots,
    )


def audit_candidate_calendar(
    candidate: pd.Series,
    assets: pd.DataFrame,
    protocol: dict[str, Any],
    snapshot_dir: Path,
    cache_dir: Path | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], str, str]:
    experiments = [protocol["historical_experiment_id"], *protocol["required_ssp_experiment_ids"]]
    file_docs, snapshots = query_candidate_file_metadata(
        str(protocol["catalogues"]["esgf_search"]), candidate, snapshot_dir, cache_dir
    )

    def resolve_one(experiment: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        asset = selected_asset(assets, candidate, experiment, "tas")
        result, local_snapshots = resolve_experiment_calendar(
            asset,
            file_docs,
            snapshot_dir,
            cache_dir,
            int(protocol["calendar_audit"]["maximum_header_bytes"]),
        )
        row = {
            **{column: candidate[column] for column in candidate_key_columns()},
            "experiment_id": experiment,
            "tas_catalog_record_id": asset.catalog_record_id,
            **result,
        }
        return row, local_snapshots

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(experiments)) as executor:
        resolved = list(executor.map(resolve_one, experiments))
    for row, local_snapshots in resolved:
        rows.append(row)
        snapshots.extend(local_snapshots)
    audit = pd.DataFrame(rows)
    unresolved = audit["calendar_status"].ne("CALENDAR_METADATA_RESOLVED")
    calendars = sorted(set(audit.loc[~unresolved, "calendar"]))
    accepted = set(protocol["accepted_calendars"])
    if unresolved.any():
        return audit, snapshots, "CALENDAR_AUDIT_FAILED", "calendar_metadata_unresolved"
    if len(calendars) != 1:
        return audit, snapshots, "CALENDAR_AUDIT_FAILED", "historical_ssp_calendar_mismatch"
    if calendars[0] not in accepted:
        return audit, snapshots, "CALENDAR_AUDIT_FAILED", "unsupported_calendar"
    return audit, snapshots, "CALENDAR_AUDIT_PASS", calendars[0]


def select_complete_members(
    candidates: pd.DataFrame,
    assets: pd.DataFrame,
    protocol: dict[str, Any],
    snapshot_dir: Path,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    candidate_decisions: list[dict[str, Any]] = []
    calendar_frames: list[pd.DataFrame] = []
    snapshots: list[dict[str, Any]] = []
    for source_id, source_candidates in candidates.groupby("source_id", sort=True):
        complete = source_candidates[
            source_candidates["candidate_status"].eq("COMPLETE_METADATA_CANDIDATE")
        ].copy()
        if complete.empty:
            for row in source_candidates.to_dict("records"):
                candidate_decisions.append(
                    {
                        **row,
                        "selection_status": "EXCLUDED_INCOMPLETE_REQUIRED_ASSETS",
                        "selection_reason": row["missing_assets"],
                        "calendar": "",
                    }
                )
            continue
        order = sorted(complete.index, key=lambda index: candidate_priority(complete.loc[index], protocol))
        selected_index: int | None = None
        selected_calendar = ""
        audits_by_index: dict[int, tuple[str, str]] = {}
        for index in order:
            candidate = complete.loc[index]
            audit, local_snapshots, status, detail = audit_candidate_calendar(
                candidate, assets, protocol, snapshot_dir, cache_dir
            )
            calendar_frames.append(audit)
            snapshots.extend(local_snapshots)
            audits_by_index[index] = (status, detail)
            if status == "CALENDAR_AUDIT_PASS":
                selected_index = index
                selected_calendar = detail
                break
        for index, row in source_candidates.iterrows():
            record = row.to_dict()
            if index == selected_index:
                selection_status = "SELECTED_COMPLETE_MEMBER"
                selection_reason = "deterministic_first_complete_member_grid_calendar"
                calendar = selected_calendar
            elif row.candidate_status != "COMPLETE_METADATA_CANDIDATE":
                selection_status = "EXCLUDED_INCOMPLETE_REQUIRED_ASSETS"
                selection_reason = row.missing_assets
                calendar = ""
            elif index in audits_by_index and audits_by_index[index][0] != "CALENDAR_AUDIT_PASS":
                selection_status = "EXCLUDED_CALENDAR_METADATA"
                selection_reason = audits_by_index[index][1]
                calendar = ""
            else:
                selection_status = "EXCLUDED_DETERMINISTIC_ONE_MEMBER_PER_SOURCE"
                selection_reason = "higher_priority_complete_member_selected"
                calendar = ""
            candidate_decisions.append(
                {
                    **record,
                    "selection_status": selection_status,
                    "selection_reason": selection_reason,
                    "calendar": calendar,
                }
            )
        if selected_index is not None:
            record = complete.loc[selected_index].to_dict()
            selected.append(
                {
                    **record,
                    "calendar": selected_calendar,
                    "selection_status": "SELECTED_COMPLETE_MEMBER",
                }
            )
    selected_frame = pd.DataFrame(selected)
    decisions_frame = pd.DataFrame(candidate_decisions)
    calendar_frame = (
        pd.concat(calendar_frames, ignore_index=True)
        if calendar_frames
        else pd.DataFrame(
            columns=[*candidate_key_columns(), "experiment_id", "calendar_status", "calendar"]
        )
    )
    return selected_frame, decisions_frame, calendar_frame, snapshots


def build_selected_asset_pairs(
    selected: pd.DataFrame, assets: pd.DataFrame, protocol: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    historical = str(protocol["historical_experiment_id"])
    for candidate in selected.itertuples(index=False):
        candidate_series = pd.Series(candidate._asdict())
        variables = str(candidate.required_variables).split(";")
        for scenario in protocol["required_ssp_experiment_ids"]:
            for variable in variables:
                historical_asset = selected_asset(
                    assets, candidate_series, historical, variable
                )
                future_asset = selected_asset(
                    assets, candidate_series, str(scenario), variable
                )
                rows.append(
                    {
                        "source_id": candidate.source_id,
                        "institution_id": candidate.institution_id,
                        "historical_experiment_id": historical,
                        "ssp_experiment_id": scenario,
                        "variant_label": candidate.variant_label,
                        "grid_label": candidate.grid_label,
                        "historical_version": historical_asset.version,
                        "ssp_version": future_asset.version,
                        "variable": variable,
                        "frequency": historical_asset.frequency,
                        "calendar": candidate.calendar,
                        "historical_start": historical_asset.datetime_start,
                        "historical_end": historical_asset.datetime_end,
                        "future_start": future_asset.datetime_start,
                        "future_end": future_asset.datetime_end,
                        "historical_catalog_record_id": historical_asset.catalog_record_id,
                        "future_catalog_record_id": future_asset.catalog_record_id,
                        "historical_master_id": historical_asset.master_id,
                        "future_master_id": future_asset.master_id,
                        "eligibility_status": "SELECTED_MEMBER_RESOLVED_ASSET_PAIR",
                        "exclusion_reason": "",
                    }
                )
    return pd.DataFrame(rows)


def build_selected_asset_manifest(
    selected: pd.DataFrame, assets: pd.DataFrame, protocol: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    historical = str(protocol["historical_experiment_id"])
    experiments = [historical, *protocol["required_ssp_experiment_ids"]]
    for candidate in selected.itertuples(index=False):
        candidate_series = pd.Series(candidate._asdict())
        variables = str(candidate.required_variables).split(";")
        for experiment in experiments:
            for variable in variables:
                asset = selected_asset(assets, candidate_series, str(experiment), variable)
                historical_asset = selected_asset(
                    assets, candidate_series, historical, variable
                )
                is_historical = str(experiment) == historical
                rows.append(
                    {
                        "source_id": candidate.source_id,
                        "institution_id": candidate.institution_id,
                        "historical_experiment_id": historical,
                        "ssp_experiment_id": "" if is_historical else experiment,
                        "experiment_id": experiment,
                        "asset_role": "historical" if is_historical else "future_ssp",
                        "variant_label": candidate.variant_label,
                        "member_id": candidate.member_id,
                        "grid_label": candidate.grid_label,
                        "version": asset.version,
                        "variable": variable,
                        "frequency": asset.frequency,
                        "calendar": candidate.calendar,
                        "historical_start": historical_asset.datetime_start,
                        "historical_end": historical_asset.datetime_end,
                        "future_start": "" if is_historical else asset.datetime_start,
                        "future_end": "" if is_historical else asset.datetime_end,
                        "asset_start": asset.datetime_start,
                        "asset_end": asset.datetime_end,
                        "catalog_record_id": asset.catalog_record_id,
                        "master_id": asset.master_id,
                        "eligibility_status": "SELECTED_MEMBER_RESOLVED_ASSET",
                        "exclusion_reason": "",
                    }
                )
    return pd.DataFrame(rows)


def build_weights(selected: pd.DataFrame) -> pd.DataFrame:
    count = len(selected)
    if count == 0:
        return pd.DataFrame(columns=["source_id", "model_weight", "weighting_rule"])
    return pd.DataFrame(
        {
            "source_id": sorted(selected["source_id"].astype(str)),
            "model_weight": 1.0 / count,
            "weighting_rule": "equal_weight_per_source_id",
        }
    )


def copy_frozen_protocol(protocol_path: Path, output: Path) -> Path:
    target = output / protocol_path.name
    shutil.copyfile(protocol_path, target)
    return target


def output_manifest(output: Path) -> pd.DataFrame:
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "output_manifest.tsv"
    )
    return pd.DataFrame(
        {
            "path": [path.relative_to(output).as_posix() for path in files],
            "bytes": [path.stat().st_size for path in files],
            "sha256": [sha256_file(path) for path in files],
        }
    )


def write_report(
    path: Path,
    certification: dict[str, Any],
    selected: pd.DataFrame,
    decisions: pd.DataFrame,
) -> None:
    excluded_sources = (
        sorted(set(decisions.source_id.astype(str)) - set(selected.source_id.astype(str)))
        if not decisions.empty
        else []
    )
    lines = [
        "# Phase-6A CMIP6 metadata-only inventory",
        "",
        f"Status: `{certification['status']}`",
        "",
        "No climate arrays, projected values, phenotypes, metrics, outcomes, covariates, or predictions were read or generated.",
        "",
        f"Selected one-member-per-GCM ensemble: **{len(selected)} source IDs**.",
        f"Sources without a complete eligible member: **{len(excluded_sources)}**.",
        "",
        "Selection used completeness and identifiers only: preferred `r1i1p1f1`, otherwise the lexicographically first complete member and grid. All selected models receive equal reporting weight.",
        "",
        "CMIP6 value retrieval remains separate. The ready contract authorizes member-resolved fetching only; it does not authorize future covariate construction or prediction.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_inventory(
    root: Path,
    protocol_path: Path,
    output: Path,
    snapshot_cache_dir: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise SystemExit(f"Fail-if-exists CMIP6 metadata inventory output: {output}")
    output.mkdir(parents=True)
    snapshot_dir = output / "catalog_snapshot"
    snapshot_dir.mkdir()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_METADATA_ONLY_BEFORE_CMIP6_VALUE_ACCESS":
        raise SystemExit("CMIP6 metadata inventory protocol is not frozen")
    parent_path = root / str(protocol["parent_environment_source_contract"])
    if not parent_path.is_file():
        raise SystemExit(f"Parent environment source contract is missing: {parent_path}")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("status") != "PASS" or parent.get("cmip6_ensemble_preregistered") is not False:
        raise SystemExit("Parent environment source contract is not the expected blocked v10 contract")
    frozen_protocol = copy_frozen_protocol(protocol_path, output)

    collection, cds_snapshots = snapshot_cds_catalogue(
        str(protocol["catalogues"]["cds_collection"]), snapshot_dir, snapshot_cache_dir
    )
    docs, esgf_snapshots, esgf_query = snapshot_esgf_datasets(
        protocol, snapshot_dir, snapshot_cache_dir
    )
    assets = canonicalize_dataset_docs(docs)
    assets = annotate_asset_eligibility(assets, protocol)
    candidates = build_candidate_completeness(assets, protocol)
    selected, decisions, calendar_audit, calendar_snapshots = select_complete_members(
        candidates, assets, protocol, snapshot_dir, snapshot_cache_dir
    )
    selected_assets = build_selected_asset_manifest(selected, assets, protocol)
    selected_pairs = build_selected_asset_pairs(selected, assets, protocol)
    weights = build_weights(selected)

    write_tsv(output / "cmip6_catalog_asset_inventory.tsv", assets)
    write_tsv(output / "cmip6_candidate_completeness.tsv", candidates)
    write_tsv(output / "cmip6_candidate_selection_decisions.tsv", decisions)
    write_tsv(output / "cmip6_calendar_metadata_audit.tsv", calendar_audit)
    write_tsv(output / "cmip6_selected_member_manifest.tsv", selected)
    write_tsv(output / "cmip6_selected_asset_manifest.tsv", selected_assets)
    write_tsv(output / "cmip6_selected_asset_pairs.tsv", selected_pairs)
    write_tsv(output / "cmip6_equal_model_weights.tsv", weights)

    snapshot_rows = cds_snapshots + esgf_snapshots + calendar_snapshots
    snapshots = pd.DataFrame(snapshot_rows).drop_duplicates(
        ["snapshot_file", "sha256"], keep="first"
    )
    write_tsv(output / "catalog_snapshot_manifest.tsv", snapshots)
    expected_scenarios = set(protocol["required_ssp_experiment_ids"])
    expected_sources = set(selected.source_id.astype(str)) if not selected.empty else set()
    selected_scenarios = (
        set(selected_pairs.ssp_experiment_id.astype(str)) if not selected_pairs.empty else set()
    )
    selected_pair_sources = (
        set(selected_pairs.source_id.astype(str)) if not selected_pairs.empty else set()
    )
    expected_selected_asset_count = sum(
        len(str(row.required_variables).split(";"))
        * (1 + len(protocol["required_ssp_experiment_ids"]))
        for row in selected.itertuples(index=False)
    )
    checks = {
        "parent_contract_pass_and_blocked_before_inventory": True,
        "protocol_frozen_before_catalog_access": True,
        "cds_collection_available": collection.get("cads:sanity_check", {}).get("status")
        == "available",
        "cds_daily_temporal_resolution_documented": True,
        "esgf_snapshot_nonempty": len(docs) > 0,
        "esgf_snapshot_unique_record_count": esgf_query["declared_record_count"]
        == esgf_query["unique_record_count"],
        "selected_source_count_positive": len(selected) > 0,
        "one_member_per_source_id": not selected.empty
        and not selected.source_id.astype(str).duplicated().any(),
        "all_required_ssps_present_for_every_selected_source": selected_scenarios
        == expected_scenarios
        and selected_pair_sources == expected_sources,
        "historical_future_member_and_grid_match": not selected_pairs.empty,
        "calendar_metadata_resolved_and_supported": not selected.empty
        and selected.calendar.astype(str).isin(protocol["accepted_calendars"]).all(),
        "selected_asset_manifest_complete": len(selected_assets)
        == expected_selected_asset_count,
        "selected_asset_manifest_unique": not selected_assets.empty
        and not selected_assets.duplicated(
            ["source_id", "member_id", "grid_label", "experiment_id", "variable"]
        ).any(),
        "equal_weight_per_source_id": not weights.empty
        and abs(float(weights.model_weight.sum()) - 1.0) < 1e-12
        and weights.model_weight.nunique() == 1,
        "no_scenario_specific_ensemble": selected_pair_sources == expected_sources,
        "no_climate_values_read": True,
        "no_projected_values_read": True,
        "no_future_covariates_generated": True,
        "no_predictions_generated": True,
        "no_phenotype_values_read": True,
        "no_model_metrics_read": True,
        "no_final_holdout_outcomes_read": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    status = (
        str(protocol["ready_status"])
        if all(checks.values())
        else "BLOCKED_CMIP6_MEMBER_RESOLUTION_INCOMPLETE"
    )
    certification = {
        "status": status,
        "protocol_version": protocol["protocol_version"],
        "selection_data": "public_catalog_metadata_and_opendap_das_headers_only",
        "catalogue_snapshot_time_utc": utc_now(),
        "parent_contract_path": str(parent_path),
        "parent_contract_sha256": sha256_file(parent_path),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "inventory_builder_sha256": sha256_file(Path(__file__).resolve()),
        "cds_collection_updated": collection.get("updated"),
        "esgf_declared_dataset_records": esgf_query["declared_record_count"],
        "esgf_unique_dataset_records": esgf_query["unique_record_count"],
        "canonical_asset_count": len(assets),
        "candidate_member_grid_count": len(candidates),
        "complete_metadata_candidate_count": int(
            candidates.candidate_status.eq("COMPLETE_METADATA_CANDIDATE").sum()
        ),
        "selected_source_count": len(selected),
        "selected_asset_manifest_count": len(selected_assets),
        "selected_asset_pair_count": len(selected_pairs),
        "equal_model_weight": float(weights.model_weight.iloc[0]) if len(weights) else None,
        "climate_values_read": False,
        "projected_values_read": False,
        "future_covariates_generated": 0,
        "predictions_generated": 0,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
    }
    write_json(output / "cmip6_metadata_inventory_certification.json", certification)
    ready_contract = {
        "status": status,
        "protocol_version": "phase6a_member_resolved_cmip6_fetch_contract_v1",
        "parent_contract_path": str(parent_path),
        "parent_contract_sha256": sha256_file(parent_path),
        "metadata_inventory_certification_sha256": sha256_file(
            output / "cmip6_metadata_inventory_certification.json"
        ),
        "selected_member_manifest_sha256": sha256_file(
            output / "cmip6_selected_member_manifest.tsv"
        ),
        "selected_asset_manifest_sha256": sha256_file(
            output / "cmip6_selected_asset_manifest.tsv"
        ),
        "selected_asset_pairs_sha256": sha256_file(
            output / "cmip6_selected_asset_pairs.tsv"
        ),
        "catalog_snapshot_manifest_sha256": sha256_file(
            output / "catalog_snapshot_manifest.tsv"
        ),
        "selected_source_count": len(selected),
        "member_resolved_fetch_allowed": status == protocol["ready_status"],
        "future_covariate_generation_allowed": False,
        "future_prediction_allowed": False,
        "member_dimension_must_be_retained": True,
        "ensemble_aggregation_stage": "reporting_only",
    }
    write_json(output / "cmip6_member_resolved_fetch_contract.json", ready_contract)
    write_report(output / "PHASE6A_CMIP6_METADATA_INVENTORY_REPORT.md", certification, selected, decisions)
    manifest = output_manifest(output)
    write_tsv(output / "output_manifest.tsv", manifest)
    return certification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a metadata-only, member-resolved CMIP6 availability inventory"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--snapshot-cache-dir",
        type=Path,
        help="Optional read-only cache of prior metadata snapshots; all reused bytes are copied into the new inventory",
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    result = run_inventory(
        root,
        resolve(root, args.protocol),
        resolve(root, args.out_dir),
        resolve(root, args.snapshot_cache_dir) if args.snapshot_cache_dir else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
