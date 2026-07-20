from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd

from server_genotype_recovery.fetch_brapi_pedigree_markers import (
    build_query_terms,
    clean,
    read_table,
    sha256_file,
    write_json_atomic,
    write_tsv,
)


USER_AGENT = "WheatConformer-CIMMYT-Dataverse-recovery/1.0"
SEARCH_COLUMNS = [
    "query_scope",
    "query_id",
    "query_kind",
    "query_text",
    "item_type",
    "name",
    "global_id",
    "entity_id",
    "dataset_persistent_id",
    "dataset_name",
    "url",
    "description",
    "published_at",
    "raw_json",
]
FILE_COLUMNS = [
    "dataset_persistent_id",
    "dataset_version",
    "datafile_id",
    "filename",
    "content_type",
    "filesize",
    "storage_identifier",
    "restricted",
    "candidate_role",
    "candidate_reason",
    "description",
    "checksum_type",
    "checksum_value",
]
REQUEST_COLUMNS = [
    "context",
    "method",
    "url",
    "status",
    "elapsed_seconds",
    "response_bytes",
    "cache_path",
]
FAILURE_COLUMNS = ["context", "method", "url", "status", "error_type", "error"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(value: object) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", clean(value)).strip("._")
    return name[:180] or "unnamed"


def classify_candidate_file(filename: object, description: object, content_type: object) -> tuple[str, str]:
    text = " ".join([clean(filename), clean(description), clean(content_type)]).lower()
    marker_terms = (
        "genotyp", "marker", "snp", "vcf", "hapmap", "plink", "dart", "gbs",
        "80k", "90k", "35k", "allele", "variant", "dosage",
    )
    pedigree_terms = (
        "pedigree", "cross", "parent", "lineage", "selection history", "germplasm",
        "accession", "passport",
    )
    marker = [term for term in marker_terms if term in text]
    pedigree = [term for term in pedigree_terms if term in text]
    if marker and pedigree:
        return "marker_and_pedigree", "|".join(marker + pedigree)
    if marker:
        return "marker", "|".join(marker)
    if pedigree:
        return "pedigree", "|".join(pedigree)
    return "none", ""


def response_data(payload: dict | None) -> object:
    if not isinstance(payload, dict) or payload.get("status") not in {None, "OK"}:
        return None
    return payload.get("data")


JsonTransport = Callable[[str, str, dict[str, str], int], dict]


def urllib_json_transport(method: str, url: str, headers: dict[str, str], timeout: int) -> dict:
    request = urllib.request.Request(url, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class DataverseClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        cache_dir: Path,
        request_log: list[dict[str, object]],
        failures: list[dict[str, object]],
        timeout: int = 30,
        transport: JsonTransport = urllib_json_transport,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_log = request_log
        self.failures = failures
        self.timeout = timeout
        self.transport = transport

    def headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            "X-Dataverse-key": self.token,
        }

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json"

    def request_json(
        self,
        path: str,
        params: list[tuple[str, object]] | dict[str, object] | None,
        context: str,
        use_cache: bool = True,
    ) -> dict | None:
        pairs = list(params.items()) if isinstance(params, dict) else (params or [])
        query = urllib.parse.urlencode(pairs, doseq=True)
        url = f"{self.base_url}/{path.lstrip('/')}" + (f"?{query}" if query else "")
        cache_path = self._cache_path(url)
        started = time.monotonic()
        if use_cache and cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.request_log.append(
                {
                    "context": context,
                    "method": "GET",
                    "url": url,
                    "status": "CACHE_HIT",
                    "elapsed_seconds": time.monotonic() - started,
                    "response_bytes": cache_path.stat().st_size,
                    "cache_path": str(cache_path),
                }
            )
            return payload
        try:
            payload = self.transport("GET", url, self.headers(), self.timeout)
            encoded = json.dumps(payload, sort_keys=True)
            cache_path.write_text(encoded, encoding="utf-8")
            self.request_log.append(
                {
                    "context": context,
                    "method": "GET",
                    "url": url,
                    "status": "OK",
                    "elapsed_seconds": time.monotonic() - started,
                    "response_bytes": len(encoded.encode("utf-8")),
                    "cache_path": str(cache_path),
                }
            )
            return payload
        except Exception as exc:
            code = getattr(exc, "code", None)
            status = "AUTH_REQUIRED" if code in (401, 403) else (f"HTTP_{code}" if code else "ERROR")
            request_row = {
                "context": context,
                "method": "GET",
                "url": url,
                "status": status,
                "elapsed_seconds": time.monotonic() - started,
                "response_bytes": 0,
                "cache_path": str(cache_path),
            }
            failure_row = {
                "context": context,
                "method": "GET",
                "url": url,
                "status": status,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self.request_log.append(request_row)
            self.failures.append(failure_row)
            return None

    def download_file(
        self,
        datafile_id: str,
        destination: Path,
        context: str,
        max_bytes: int | None = None,
    ) -> tuple[bool, str]:
        url = f"{self.base_url}/api/access/datafile/{urllib.parse.quote(str(datafile_id), safe='')}?format=original"
        started = time.monotonic()
        temporary = destination.with_name(f".{destination.name}.part")
        try:
            request = urllib.request.Request(url, headers=self.headers(), method="GET")
            bytes_written = 0
            with urllib.request.urlopen(request, timeout=self.timeout) as response, temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if max_bytes is not None and bytes_written > max_bytes:
                        raise ValueError(
                            f"download exceeded enforced byte limit ({bytes_written}>{max_bytes})"
                        )
                    handle.write(chunk)
            temporary.replace(destination)
            self.request_log.append(
                {
                    "context": context,
                    "method": "GET",
                    "url": url,
                    "status": "OK",
                    "elapsed_seconds": time.monotonic() - started,
                    "response_bytes": destination.stat().st_size,
                    "cache_path": str(destination),
                }
            )
            return True, ""
        except Exception as exc:
            if temporary.exists():
                temporary.unlink()
            code = getattr(exc, "code", None)
            status = "AUTH_REQUIRED" if code in (401, 403) else (f"HTTP_{code}" if code else "ERROR")
            self.request_log.append(
                {
                    "context": context,
                    "method": "GET",
                    "url": url,
                    "status": status,
                    "elapsed_seconds": time.monotonic() - started,
                    "response_bytes": 0,
                    "cache_path": str(destination),
                }
            )
            self.failures.append(
                {
                    "context": context,
                    "method": "GET",
                    "url": url,
                    "status": status,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return False, str(exc)


def normalize_search_item(
    item: dict, query_scope: str, query_id: str, query_kind: str, query_text: str
) -> dict[str, object]:
    return {
        "query_scope": query_scope,
        "query_id": query_id,
        "query_kind": query_kind,
        "query_text": query_text,
        "item_type": clean(item.get("type")),
        "name": clean(item.get("name")),
        "global_id": clean(item.get("global_id")),
        "entity_id": clean(item.get("entity_id") or item.get("id") or item.get("datafile_id")),
        "dataset_persistent_id": clean(item.get("dataset_persistent_id")),
        "dataset_name": clean(item.get("dataset_name")),
        "url": clean(item.get("url")),
        "description": clean(item.get("description")),
        "published_at": clean(item.get("published_at")),
        "raw_json": json.dumps(item, sort_keys=True)[:30000],
    }


def dataset_file_rows(payload: dict | None, persistent_id: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    data = response_data(payload)
    if not isinstance(data, dict):
        return {}, []
    version = data.get("latestVersion") or {}
    dataset_row = {
        "dataset_persistent_id": persistent_id,
        "dataset_id": clean(data.get("id")),
        "version": f"{clean(version.get('versionNumber'))}.{clean(version.get('versionMinorNumber'))}",
        "version_state": clean(version.get("versionState")),
        "release_time": clean(version.get("releaseTime")),
        "file_count": len(version.get("files") or []),
        "raw_json": json.dumps(data, sort_keys=True)[:50000],
    }
    rows: list[dict[str, object]] = []
    for entry in version.get("files") or []:
        if not isinstance(entry, dict):
            continue
        datafile = entry.get("dataFile") or {}
        checksum = datafile.get("checksum") or {}
        role, reason = classify_candidate_file(
            datafile.get("filename"), entry.get("description"), datafile.get("contentType")
        )
        rows.append(
            {
                "dataset_persistent_id": persistent_id,
                "dataset_version": dataset_row["version"],
                "datafile_id": clean(datafile.get("id")),
                "filename": clean(datafile.get("filename")),
                "content_type": clean(datafile.get("contentType")),
                "filesize": int(datafile.get("filesize") or 0),
                "storage_identifier": clean(datafile.get("storageIdentifier")),
                "restricted": bool(entry.get("restricted", False)),
                "candidate_role": role,
                "candidate_reason": reason,
                "description": clean(entry.get("description")),
                "checksum_type": clean(checksum.get("type")),
                "checksum_value": clean(checksum.get("value")),
            }
        )
    return dataset_row, rows


def text_streams(path: Path) -> Iterator[tuple[str, Iterator[str]]]:
    lower = path.name.lower()
    if lower.endswith(".zip"):
        archive = zipfile.ZipFile(path)
        try:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                handle = archive.open(member)
                yield member.filename, (line.decode("utf-8", errors="replace") for line in handle)
                handle.close()
        finally:
            archive.close()
        return
    if lower.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield path.name, iter(handle)
        return
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        yield path.name, iter(handle)


def exact_term_pattern(value: object) -> re.Pattern[str] | None:
    chunks = re.findall(r"[A-Z0-9]+", clean(value).upper())
    if not chunks:
        return None
    expression = r"(?<![A-Z0-9])" + r"[^A-Z0-9]*".join(
        re.escape(chunk) for chunk in chunks
    ) + r"(?![A-Z0-9])"
    return re.compile(expression)


def scan_file_for_terms(
    path: Path,
    terms: list[dict[str, object]],
    max_hits_per_term: int = 3,
) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    counts: dict[tuple[str, str, str], int] = {}
    prepared = [
        (term, exact_term_pattern(term["query_text"]))
        for term in terms
        if clean(term["query_text"])
    ]
    try:
        for member, lines in text_streams(path):
            for line_number, line in enumerate(lines, start=1):
                upper = line.upper()
                for term, pattern in prepared:
                    key = (str(term["query_id"]), str(term["query_kind"]), str(term["query_text"]))
                    if counts.get(key, 0) >= max_hits_per_term:
                        continue
                    if pattern is None or pattern.search(upper) is None:
                        continue
                    counts[key] = counts.get(key, 0) + 1
                    hits.append(
                        {
                            "query_id": term["query_id"],
                            "query_kind": term["query_kind"],
                            "query_text": term["query_text"],
                            "path": str(path),
                            "archive_member": member,
                            "line_number": line_number,
                            "match_excerpt": line.strip()[:1000],
                        }
                    )
    except (UnicodeError, OSError, zipfile.BadZipFile) as exc:
        hits.append(
            {
                "query_id": "",
                "query_kind": "scan_error",
                "query_text": "",
                "path": str(path),
                "archive_member": "",
                "line_number": 0,
                "match_excerpt": f"{type(exc).__name__}: {exc}",
            }
        )
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Authenticated, bounded CIMMYT Dataverse recovery for genotype and pedigree evidence."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--resolver-query",
        type=Path,
        default=Path("genotype_panels/germplasm_resolver/germplasm_cross_query.tsv"),
    )
    parser.add_argument("--base-url", default="https://data.cimmyt.org")
    parser.add_argument("--token-env", default="CIMMYT_DATAVERSE_TOKEN")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--per-page", type=int, default=25)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--discovery-query", action="append", default=[])
    parser.add_argument("--download-candidates", action="store_true")
    parser.add_argument("--include-restricted", action="store_true")
    parser.add_argument("--max-download-files", type=int, default=10)
    parser.add_argument("--max-file-bytes", type=int, default=25 * 1024 * 1024)
    parser.add_argument("--max-total-download-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("genotype_panels/cimmyt_dataverse_recovery_v1")
    )
    args = parser.parse_args()

    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise RuntimeError(
            f"API token is absent. Export it only through environment variable {args.token_env}."
        )
    root = args.root.resolve()
    resolver = args.resolver_query if args.resolver_query.is_absolute() else root / args.resolver_query
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not resolver.is_file() or resolver.stat().st_size == 0:
        raise FileNotFoundError(f"Resolver query is missing or empty: {resolver}")

    resolver_frame = read_table(resolver)
    selected, terms = build_query_terms(resolver_frame, args.limit, args.offset)
    discovery = args.discovery_query or [
        "wheat genotypic",
        "wheat pedigree",
        "Seeds of Discovery wheat",
        "DArTseq wheat",
    ]
    write_tsv(terms, out_dir / "dataverse_query_terms.tsv", [
        "source_row", "query_id", "query_kind", "query_text", "normalized_query",
        "marker_probe_eligible", "selection_stage_count", "selection_stages",
    ])

    request_log: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    client = DataverseClient(
        args.base_url,
        token,
        out_dir / "response_cache",
        request_log,
        failures,
        timeout=args.timeout,
    )
    identity = client.request_json("api/users/:me", None, "token_validation", use_cache=False)
    token_valid = isinstance(response_data(identity), dict)
    if not token_valid:
        write_tsv(request_log, out_dir / "dataverse_request_log.tsv", REQUEST_COLUMNS)
        write_tsv(failures, out_dir / "dataverse_failures.tsv", FAILURE_COLUMNS)
        raise RuntimeError("CIMMYT Dataverse token validation failed; see dataverse_failures.tsv")

    search_rows: list[dict[str, object]] = []
    query_specs = [
        ("resolver", str(term["query_id"]), str(term["query_kind"]), str(term["query_text"]))
        for term in terms
    ]
    query_specs.extend(("discovery", "", "discovery", value) for value in discovery)
    for query_index, (scope, query_id, query_kind, query_text) in enumerate(query_specs, start=1):
        print(
            f"[{utc_now()}] SEARCH {query_index}/{len(query_specs)} scope={scope} "
            f"kind={query_kind} query_id={query_id}",
            flush=True,
        )
        for page in range(args.max_pages):
            payload = client.request_json(
                "api/search",
                [
                    ("q", query_text),
                    ("type", "dataset"),
                    ("type", "file"),
                    ("per_page", args.per_page),
                    ("start", page * args.per_page),
                ],
                f"search:{scope}:{query_kind}:{query_text}:page_{page}",
            )
            data = response_data(payload)
            items = data.get("items") if isinstance(data, dict) else []
            for item in items or []:
                if isinstance(item, dict):
                    search_rows.append(
                        normalize_search_item(item, scope, query_id, query_kind, query_text)
                    )
            if not items or len(items) < args.per_page:
                break
        time.sleep(args.sleep)

    write_tsv(search_rows, out_dir / "dataverse_search_results.tsv", SEARCH_COLUMNS)
    persistent_ids = {
        clean(row["global_id"])
        for row in search_rows
        if row["item_type"] == "dataset" and clean(row["global_id"])
    }
    persistent_ids.update(
        clean(row["dataset_persistent_id"])
        for row in search_rows
        if clean(row["dataset_persistent_id"])
    )
    dataset_rows: list[dict[str, object]] = []
    file_rows: list[dict[str, object]] = []
    for index, persistent_id in enumerate(sorted(persistent_ids), start=1):
        print(f"[{utc_now()}] DATASET {index}/{len(persistent_ids)} {persistent_id}", flush=True)
        payload = client.request_json(
            "api/datasets/:persistentId/",
            {"persistentId": persistent_id},
            f"dataset:{persistent_id}",
        )
        dataset, files = dataset_file_rows(payload, persistent_id)
        if dataset:
            dataset_rows.append(dataset)
            file_rows.extend(files)

    write_tsv(dataset_rows, out_dir / "dataverse_dataset_metadata.tsv", [
        "dataset_persistent_id", "dataset_id", "version", "version_state", "release_time",
        "file_count", "raw_json",
    ])
    write_tsv(file_rows, out_dir / "dataverse_candidate_files.tsv", FILE_COLUMNS)

    downloads: list[dict[str, object]] = []
    content_hits: list[dict[str, object]] = []
    total_downloaded = 0
    downloaded_files = 0
    download_dir = out_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    if args.download_candidates:
        candidates = [row for row in file_rows if row["candidate_role"] != "none"]
        candidates.sort(key=lambda row: (bool(row["restricted"]), int(row["filesize"])))
        for row in candidates:
            reason = ""
            size = int(row["filesize"])
            if downloaded_files >= args.max_download_files:
                reason = "max_download_files_reached"
            elif bool(row["restricted"]) and not args.include_restricted:
                reason = "restricted_file_not_requested"
            elif size > args.max_file_bytes:
                reason = "file_exceeds_max_file_bytes"
            elif total_downloaded + size > args.max_total_download_bytes:
                reason = "total_download_budget_exceeded"
            if reason:
                downloads.append({**row, "download_status": "SKIPPED", "local_path": "", "detail": reason})
                continue
            destination = download_dir / f"{row['datafile_id']}_{safe_filename(row['filename'])}"
            enforced_limit = min(
                args.max_file_bytes,
                args.max_total_download_bytes - total_downloaded,
            )
            ok, detail = client.download_file(
                str(row["datafile_id"]),
                destination,
                f"download:{row['datafile_id']}",
                max_bytes=enforced_limit,
            )
            if ok:
                downloaded_files += 1
                total_downloaded += destination.stat().st_size
                content_hits.extend(scan_file_for_terms(destination, terms))
            downloads.append(
                {
                    **row,
                    "download_status": "DOWNLOADED" if ok else "FAILED",
                    "local_path": str(destination) if ok else "",
                    "detail": detail,
                }
            )

    write_tsv(downloads, out_dir / "dataverse_downloads.tsv", FILE_COLUMNS + [
        "download_status", "local_path", "detail",
    ])
    write_tsv(content_hits, out_dir / "dataverse_content_matches.tsv", [
        "query_id", "query_kind", "query_text", "path", "archive_member", "line_number",
        "match_excerpt",
    ])
    write_tsv(request_log, out_dir / "dataverse_request_log.tsv", REQUEST_COLUMNS)
    write_tsv(failures, out_dir / "dataverse_failures.tsv", FAILURE_COLUMNS)

    qc = [
        {"metric": "run_status", "value": "complete_with_failures" if failures else "complete"},
        {"metric": "token_valid", "value": token_valid},
        {"metric": "resolver_rows_available", "value": len(resolver_frame)},
        {"metric": "resolver_rows_selected", "value": len(selected)},
        {"metric": "resolver_query_terms", "value": len(terms)},
        {"metric": "discovery_queries", "value": len(discovery)},
        {"metric": "search_result_rows", "value": len(search_rows)},
        {"metric": "datasets_resolved", "value": len(dataset_rows)},
        {"metric": "dataset_file_rows", "value": len(file_rows)},
        {"metric": "candidate_marker_files", "value": sum(row["candidate_role"] in {"marker", "marker_and_pedigree"} for row in file_rows)},
        {"metric": "candidate_pedigree_files", "value": sum(row["candidate_role"] in {"pedigree", "marker_and_pedigree"} for row in file_rows)},
        {"metric": "downloaded_files", "value": downloaded_files},
        {"metric": "downloaded_bytes", "value": total_downloaded},
        {"metric": "content_match_rows", "value": len([row for row in content_hits if row["query_kind"] != "scan_error"])},
        {"metric": "matched_query_ids", "value": len({row["query_id"] for row in content_hits if row["query_id"]})},
        {"metric": "failure_rows", "value": len(failures)},
        {"metric": "phenotype_values_read", "value": False},
        {"metric": "outer_test_metrics_read", "value": False},
        {"metric": "final_holdout_outcomes_read", "value": False},
    ]
    write_tsv(qc, out_dir / "dataverse_recovery_qc.tsv", ["metric", "value"])
    outputs = sorted(path for path in out_dir.glob("dataverse_*") if path.is_file())
    provenance = {
        "status": "complete_with_failures" if failures else "complete",
        "created_at_utc": utc_now(),
        "module": "server_genotype_recovery.fetch_cimmyt_dataverse_recovery",
        "base_url": args.base_url,
        "token_env": args.token_env,
        "token_present": True,
        "token_value_recorded": False,
        "resolver_query": {"path": str(resolver), "sha256": sha256_file(resolver)},
        "parameters": {
            "limit": args.limit,
            "offset": args.offset,
            "per_page": args.per_page,
            "max_pages": args.max_pages,
            "download_candidates": args.download_candidates,
            "include_restricted": args.include_restricted,
            "max_download_files": args.max_download_files,
            "max_file_bytes": args.max_file_bytes,
            "max_total_download_bytes": args.max_total_download_bytes,
        },
        "selection_data": "identifiers_and_authenticated_repository_metadata_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "output_sha256": {str(path.relative_to(out_dir)): sha256_file(path) for path in outputs},
    }
    write_json_atomic(provenance, out_dir / "dataverse_recovery_provenance.json")
    print(pd.DataFrame(qc).to_string(index=False))


if __name__ == "__main__":
    main()
