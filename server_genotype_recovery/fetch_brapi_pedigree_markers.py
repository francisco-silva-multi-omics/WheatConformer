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
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd


USER_AGENT = "WheatConformer-BrAPI-recovery/1.0"
EMPTY_TEXT = {"", "NA", "N/A", "NAN", "NONE", "NULL", "<NA>"}
QUERY_FIELDS = (
    "sample_id",
    "bcid",
    "selection_history",
    "cross_name",
    "parent1",
    "parent2",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.upper() in EMPTY_TEXT else text


def normalized_identifier(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", clean(value).upper())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_selection_history(value: object) -> dict[str, object]:
    history = clean(value)
    if not history:
        return {"selection_history": "", "bcid": "", "selection_stages": "", "stage_count": 0}
    tokens = [token.strip() for token in history.split("-") if token.strip()]
    bcid = tokens[0] if tokens else ""
    stages = tokens[1:]
    return {
        "selection_history": history,
        "bcid": bcid,
        "selection_stages": "|".join(stages),
        "stage_count": len(stages),
    }


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, dtype=str, low_memory=False)
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def write_tsv(rows: list[dict[str, object]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, sep="\t", index=False)


def write_tsv_gz(rows: list[dict[str, object]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        pd.DataFrame(rows, columns=columns).to_csv(handle, sep="\t", index=False)


def response_result(payload: dict) -> object:
    return payload.get("result", payload)


def response_data(payload: dict) -> list[dict]:
    result = response_result(payload)
    if isinstance(result, list):
        return [row for row in result if isinstance(row, dict)]
    if not isinstance(result, dict):
        return []
    for key in ("data", "results"):
        rows = result.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def search_handle(payload: dict) -> str:
    result = response_result(payload)
    if isinstance(result, dict):
        return clean(result.get("searchResultsDbId") or result.get("searchResultDbId"))
    return clean(payload.get("searchResultsDbId") or payload.get("searchResultDbId"))


def record_names(record: dict) -> list[str]:
    values: list[str] = []
    for key in (
        "germplasmName",
        "defaultDisplayName",
        "accessionNumber",
        "germplasmPUI",
        "seedSource",
    ):
        value = clean(record.get(key))
        if value:
            values.append(value)
    synonyms = record.get("synonyms") or []
    if isinstance(synonyms, str):
        values.append(synonyms)
    elif isinstance(synonyms, list):
        for synonym in synonyms:
            if isinstance(synonym, dict):
                value = clean(
                    synonym.get("synonym")
                    or synonym.get("value")
                    or synonym.get("name")
                )
            else:
                value = clean(synonym)
            if value:
                values.append(value)
    for external in record.get("externalReferences") or []:
        if isinstance(external, dict):
            for key in ("referenceID", "referenceId", "referenceSource"):
                value = clean(external.get(key))
                if value:
                    values.append(value)
    return list(dict.fromkeys(values))


def exact_record_match(query: str, record: dict) -> bool:
    target = normalized_identifier(query)
    return bool(target) and target in {normalized_identifier(value) for value in record_names(record)}


def server_calls(payload: dict) -> set[str]:
    calls: set[str] = set()
    result = response_result(payload)
    if not isinstance(result, dict):
        return calls
    for row in result.get("calls") or []:
        if not isinstance(row, dict):
            continue
        call = clean(row.get("call") or row.get("path"))
        if call:
            calls.add(call.strip("/").lower())
    return calls


def capability_advertised(calls: set[str], resource: str) -> bool:
    needle = resource.strip("/").lower()
    return any(needle in call for call in calls)


@dataclass(frozen=True)
class ServerSpec:
    name: str
    base_url: str
    token: str | None = None


Transport = Callable[[str, str, dict | None, dict[str, str], int], dict]


def urllib_transport(
    method: str,
    url: str,
    payload: dict | None,
    headers: dict[str, str],
    timeout: int,
) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


class BrAPIClient:
    def __init__(
        self,
        spec: ServerSpec,
        cache_dir: Path,
        request_log: list[dict[str, object]],
        failures: list[dict[str, object]],
        timeout: int = 30,
        poll_attempts: int = 3,
        poll_sleep: float = 0.5,
        transport: Transport = urllib_transport,
    ) -> None:
        self.spec = spec
        self.base = spec.base_url.rstrip("/")
        self.cache_dir = cache_dir / spec.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_log = request_log
        self.failures = failures
        self.timeout = timeout
        self.poll_attempts = poll_attempts
        self.poll_sleep = poll_sleep
        self.transport = transport

    def _cache_path(self, method: str, url: str, payload: dict | None) -> Path:
        identity = json.dumps(
            {"method": method, "url": url, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return self.cache_dir / f"{hashlib.sha256(identity).hexdigest()}.json"

    def request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        context: str = "",
        use_cache: bool = True,
    ) -> dict | None:
        url = path if path.startswith("http") else f"{self.base}/{path.lstrip('/')}"
        cache_path = self._cache_path(method, url, payload)
        started = time.monotonic()
        if use_cache and cache_path.is_file():
            result = json.loads(cache_path.read_text(encoding="utf-8"))
            self.request_log.append(
                {
                    "server": self.spec.name,
                    "context": context,
                    "method": method,
                    "url": url,
                    "status": "CACHE_HIT",
                    "elapsed_seconds": time.monotonic() - started,
                    "response_bytes": cache_path.stat().st_size,
                    "cache_path": str(cache_path),
                }
            )
            return result
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self.spec.token:
            headers["Authorization"] = f"Bearer {self.spec.token}"
        try:
            result = self.transport(method, url, payload, headers, self.timeout)
            encoded = json.dumps(result, sort_keys=True)
            cache_path.write_text(encoded, encoding="utf-8")
            self.request_log.append(
                {
                    "server": self.spec.name,
                    "context": context,
                    "method": method,
                    "url": url,
                    "status": "OK",
                    "elapsed_seconds": time.monotonic() - started,
                    "response_bytes": len(encoded.encode("utf-8")),
                    "cache_path": str(cache_path),
                }
            )
            return result
        except Exception as exc:
            status = "ERROR"
            code = getattr(exc, "code", None)
            if code in (401, 403):
                status = "AUTH_REQUIRED"
            elif isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
                status = "TIMEOUT"
            self.request_log.append(
                {
                    "server": self.spec.name,
                    "context": context,
                    "method": method,
                    "url": url,
                    "status": status,
                    "elapsed_seconds": time.monotonic() - started,
                    "response_bytes": 0,
                    "cache_path": str(cache_path),
                }
            )
            self.failures.append(
                {
                    "server": self.spec.name,
                    "context": context,
                    "method": method,
                    "url": url,
                    "status": status,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return None

    def search_response(self, resource: str, payload: dict, context: str) -> dict | None:
        initial = self.request("POST", f"search/{resource}", payload, context=context)
        if initial is None:
            return None
        rows = response_data(initial)
        handle = search_handle(initial)
        if rows or not handle:
            return initial
        for attempt in range(self.poll_attempts):
            if attempt:
                time.sleep(self.poll_sleep)
            completed = self.request(
                "GET",
                f"search/{resource}/{urllib.parse.quote(handle, safe='')}",
                context=f"{context}:poll_{attempt + 1}",
                use_cache=False,
            )
            if completed is None:
                return None
            rows = response_data(completed)
            result = response_result(completed)
            if rows or (isinstance(result, dict) and result.get("dataMatrices")):
                return completed
        self.failures.append(
            {
                "server": self.spec.name,
                "context": context,
                "method": "GET",
                "url": f"{self.base}/search/{resource}/{handle}",
                "status": "ASYNC_EMPTY",
                "error_type": "BrAPISearchIncomplete",
                "error": f"No data after {self.poll_attempts} poll attempts",
            }
        )
        return None

    def search(self, resource: str, payload: dict, context: str) -> list[dict]:
        response = self.search_response(resource, payload, context)
        return [] if response is None else response_data(response)

    def get_collection(
        self, resource: str, params: dict[str, object], context: str
    ) -> list[dict]:
        query = urllib.parse.urlencode({key: value for key, value in params.items() if value not in (None, "")})
        payload = self.request("GET", f"{resource}?{query}", context=context)
        return [] if payload is None else response_data(payload)


def parse_server_specs(
    values: list[str], token_env_values: list[str]
) -> list[ServerSpec]:
    token_envs: dict[str, str] = {}
    for value in token_env_values:
        if "=" not in value:
            raise ValueError(f"--token-env must be NAME=ENV_VAR, received {value!r}")
        name, env_var = value.split("=", 1)
        token_envs[name.strip()] = env_var.strip()
    specs: list[ServerSpec] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--server must be NAME=URL, received {value!r}")
        name, base_url = value.split("=", 1)
        name = name.strip()
        base_url = base_url.strip().rstrip("/")
        if not name or not base_url:
            raise ValueError(f"Invalid server specification: {value!r}")
        env_var = token_envs.get(name)
        specs.append(ServerSpec(name, base_url, os.environ.get(env_var) if env_var else None))
    return specs


def build_query_terms(
    frame: pd.DataFrame, limit: int, offset: int = 0
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if "sample_id" not in frame.columns:
        raise ValueError("Resolver query must contain sample_id")
    selected = frame.drop_duplicates().iloc[offset : offset + limit].copy()
    terms: list[dict[str, object]] = []
    for source_row, record in selected.reset_index(drop=True).iterrows():
        sample_id = clean(record.get("sample_id"))
        parsed = parse_selection_history(record.get("selection_history"))
        values = {
            "sample_id": sample_id,
            "bcid": clean(parsed["bcid"]),
            "selection_history": clean(parsed["selection_history"]),
            "cross_name": clean(record.get("cross_name")),
            "parent1": clean(record.get("parent1")),
            "parent2": clean(record.get("parent2")),
        }
        seen: set[str] = set()
        for kind in QUERY_FIELDS:
            value = values[kind]
            normalized = normalized_identifier(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            terms.append(
                {
                    "source_row": source_row,
                    "query_id": sample_id,
                    "query_kind": kind,
                    "query_text": value,
                    "normalized_query": normalized,
                    "marker_probe_eligible": kind in {"sample_id", "bcid"},
                    "selection_stage_count": parsed["stage_count"],
                    "selection_stages": parsed["selection_stages"],
                }
            )
    return selected, terms


def germplasm_search(client: BrAPIClient, term: dict[str, object], page_size: int) -> list[dict]:
    text = str(term["query_text"])
    context = f"germplasm:{term['query_id']}:{term['query_kind']}:{text}"
    rows = client.search(
        "germplasm",
        {"germplasmNames": [text], "commonCropNames": ["wheat"], "pageSize": page_size},
        context,
    )
    if rows:
        return rows
    return client.get_collection(
        "germplasm",
        {"germplasmName": text, "pageSize": page_size},
        f"{context}:get_fallback",
    )


def germplasm_row(
    server: str, term: dict[str, object], record: dict, match_status: str
) -> dict[str, object]:
    return {
        "server": server,
        "query_id": term["query_id"],
        "query_kind": term["query_kind"],
        "query_text": term["query_text"],
        "match_status": match_status,
        "germplasmDbId": clean(record.get("germplasmDbId")),
        "germplasmName": clean(record.get("germplasmName")),
        "defaultDisplayName": clean(record.get("defaultDisplayName")),
        "accessionNumber": clean(record.get("accessionNumber")),
        "germplasmPUI": clean(record.get("germplasmPUI")),
        "synonyms": "|".join(record_names({"synonyms": record.get("synonyms") or []})),
        "raw_json": json.dumps(record, sort_keys=True)[:20000],
    }


def parent_records(payload: dict) -> list[dict[str, str]]:
    parents: list[dict[str, str]] = []

    def add(value: object, relation: str) -> None:
        if not isinstance(value, dict):
            return
        dbid = clean(value.get("germplasmDbId") or value.get("parentGermplasmDbId"))
        name = clean(
            value.get("germplasmName")
            or value.get("defaultDisplayName")
            or value.get("parentName")
        )
        if dbid or name:
            parents.append({"germplasmDbId": dbid, "name": name, "relation": relation})

    bodies: list[dict] = []
    result = response_result(payload)
    if isinstance(result, dict):
        bodies.append(result)
    bodies.extend(response_data(payload))
    for body in bodies:
        for key in ("parent1", "parent2", "femaleParent", "maleParent"):
            add(body.get(key), key)
        for value in body.get("parents") or []:
            relation = clean(value.get("parentType")) if isinstance(value, dict) else "parent"
            add(value, relation or "parent")
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for parent in parents:
        key = (parent["germplasmDbId"], parent["name"], parent["relation"])
        unique[key] = parent
    return list(unique.values())


def traverse_pedigree(
    client: BrAPIClient,
    roots: list[tuple[str, str, str]],
    max_depth: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    queue = deque((query_id, dbid, name, 0) for query_id, dbid, name in roots if dbid)
    visited: set[tuple[str, str]] = set()
    while queue:
        query_id, dbid, fallback_name, depth = queue.popleft()
        key = (query_id, dbid)
        if key in visited:
            continue
        visited.add(key)
        detail = client.request("GET", f"germplasm/{urllib.parse.quote(dbid, safe='')}", context=f"detail:{query_id}:{dbid}")
        records = response_data(detail or {})
        record = records[0] if records else (response_result(detail or {}) if isinstance(response_result(detail or {}), dict) else {})
        nodes.append(
            {
                "server": client.spec.name,
                "query_id": query_id,
                "germplasmDbId": dbid,
                "germplasmName": clean(record.get("germplasmName")) or fallback_name,
                "defaultDisplayName": clean(record.get("defaultDisplayName")),
                "accessionNumber": clean(record.get("accessionNumber")),
                "depth": depth,
                "raw_json": json.dumps(record, sort_keys=True)[:20000],
            }
        )
        if depth >= max_depth:
            continue
        pedigree = client.request(
            "GET",
            f"germplasm/{urllib.parse.quote(dbid, safe='')}/pedigree",
            context=f"pedigree:{query_id}:{dbid}:depth_{depth}",
        )
        if pedigree is None:
            continue
        for parent in parent_records(pedigree):
            edges.append(
                {
                    "server": client.spec.name,
                    "query_id": query_id,
                    "child_germplasmDbId": dbid,
                    "parent_germplasmDbId": parent["germplasmDbId"],
                    "parent_name": parent["name"],
                    "parent_relation": parent["relation"],
                    "child_depth": depth,
                    "parent_depth": depth + 1,
                }
            )
            if parent["germplasmDbId"]:
                queue.append((query_id, parent["germplasmDbId"], parent["name"], depth + 1))
    return nodes, edges


def search_crosses(client: BrAPIClient, terms: list[dict[str, object]], page_size: int) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for term in terms:
        if term["query_kind"] != "cross_name":
            continue
        text = str(term["query_text"])
        rows = client.get_collection(
            "crosses",
            {"crossName": text, "pageSize": page_size},
            f"crosses:{term['query_id']}:{text}",
        )
        for record in rows:
            output.append(
                {
                    "server": client.spec.name,
                    "query_id": term["query_id"],
                    "query_text": text,
                    "crossDbId": clean(record.get("crossDbId")),
                    "crossName": clean(record.get("crossName")),
                    "crossType": clean(record.get("crossType")),
                    "parent1_germplasmDbId": clean((record.get("parent1") or {}).get("germplasmDbId")) if isinstance(record.get("parent1"), dict) else "",
                    "parent2_germplasmDbId": clean((record.get("parent2") or {}).get("germplasmDbId")) if isinstance(record.get("parent2"), dict) else "",
                    "raw_json": json.dumps(record, sort_keys=True)[:20000],
                }
            )
    return output


def sample_row(server: str, query_id: str, germplasm_dbid: str, record: dict) -> dict[str, object]:
    return {
        "server": server,
        "query_id": query_id,
        "germplasmDbId": clean(record.get("germplasmDbId")) or germplasm_dbid,
        "sampleDbId": clean(record.get("sampleDbId")),
        "sampleName": clean(record.get("sampleName")),
        "studyDbId": clean(record.get("studyDbId")),
        "observationUnitDbId": clean(record.get("observationUnitDbId")),
        "plateDbId": clean(record.get("plateDbId")),
        "raw_json": json.dumps(record, sort_keys=True)[:20000],
    }


def find_samples(
    client: BrAPIClient,
    exact_matches: list[dict[str, object]],
    marker_terms: list[dict[str, object]],
    page_size: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(query_id: str, germplasm_dbid: str, rows: list[dict]) -> None:
        for record in rows:
            row = sample_row(client.spec.name, query_id, germplasm_dbid, record)
            key = (str(row["query_id"]), str(row["sampleDbId"]), str(row["sampleName"]))
            if key not in seen:
                seen.add(key)
                output.append(row)

    for match in exact_matches:
        dbid = str(match["germplasmDbId"])
        query_id = str(match["query_id"])
        if not dbid:
            continue
        rows = client.search(
            "samples",
            {"germplasmDbIds": [dbid], "pageSize": page_size},
            f"samples:germplasm:{query_id}:{dbid}",
        )
        if not rows:
            rows = client.get_collection(
                "samples",
                {"germplasmDbId": dbid, "pageSize": page_size},
                f"samples:germplasm:{query_id}:{dbid}:get_fallback",
            )
        add(query_id, dbid, rows)
    for term in marker_terms:
        text = str(term["query_text"])
        rows = client.search(
            "samples",
            {"sampleNames": [text], "pageSize": page_size},
            f"samples:name:{term['query_id']}:{text}",
        )
        add(str(term["query_id"]), "", rows)
    return output


def callset_row(server: str, query_id: str, sample_dbid: str, record: dict) -> dict[str, object]:
    return {
        "server": server,
        "query_id": query_id,
        "sampleDbId": clean(record.get("sampleDbId")) or sample_dbid,
        "callSetDbId": clean(record.get("callSetDbId")),
        "callSetName": clean(record.get("callSetName")),
        "variantSetDbIds": "|".join(map(str, record.get("variantSetDbIds") or [])),
        "created": clean(record.get("created")),
        "updated": clean(record.get("updated")),
        "raw_json": json.dumps(record, sort_keys=True)[:20000],
    }


def find_callsets(
    client: BrAPIClient,
    samples: list[dict[str, object]],
    marker_terms: list[dict[str, object]],
    page_size: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    def add(query_id: str, sample_dbid: str, rows: list[dict]) -> None:
        for record in rows:
            row = callset_row(client.spec.name, query_id, sample_dbid, record)
            key = (str(row["callSetDbId"]), str(row["callSetName"]))
            if key not in seen:
                seen.add(key)
                output.append(row)

    for sample in samples:
        sample_dbid = str(sample["sampleDbId"])
        if not sample_dbid:
            continue
        rows = client.search(
            "callsets",
            {"sampleDbIds": [sample_dbid], "pageSize": page_size},
            f"callsets:sample:{sample['query_id']}:{sample_dbid}",
        )
        if not rows:
            rows = client.get_collection(
                "callsets",
                {"sampleDbId": sample_dbid, "pageSize": page_size},
                f"callsets:sample:{sample['query_id']}:{sample_dbid}:get_fallback",
            )
        add(str(sample["query_id"]), sample_dbid, rows)
    for term in marker_terms:
        text = str(term["query_text"])
        rows = client.search(
            "callsets",
            {"callSetNames": [text], "pageSize": page_size},
            f"callsets:name:{term['query_id']}:{text}",
        )
        add(str(term["query_id"]), "", rows)
    return output


def find_calls(
    client: BrAPIClient,
    callsets: list[dict[str, object]],
    max_calls_per_callset: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for callset in callsets:
        callset_dbid = str(callset["callSetDbId"])
        if not callset_dbid:
            continue
        rows = client.get_collection(
            f"callsets/{urllib.parse.quote(callset_dbid, safe='')}/calls",
            {"pageSize": max_calls_per_callset},
            f"calls:{callset['query_id']}:{callset_dbid}",
        )
        for record in rows[:max_calls_per_callset]:
            genotype = record.get("genotype")
            if isinstance(genotype, list):
                genotype = "/".join(map(str, genotype))
            output.append(
                {
                    "server": client.spec.name,
                    "query_id": callset["query_id"],
                    "sampleDbId": callset["sampleDbId"],
                    "callSetDbId": callset_dbid,
                    "callSetName": callset["callSetName"],
                    "call_source": "callsets/{callSetDbId}/calls",
                    "variantDbId": clean(record.get("variantDbId")),
                    "variantName": clean(record.get("variantName")),
                    "genotype": clean(genotype),
                    "genotypeValue": clean(record.get("genotypeValue")),
                    "phaseSet": clean(record.get("phaseSet")),
                    "raw_json": json.dumps(record, sort_keys=True)[:20000],
                }
            )
    return output


def find_allele_matrix_calls(
    client: BrAPIClient,
    callsets: list[dict[str, object]],
    max_calls_per_callset: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for callset in callsets:
        callset_dbid = str(callset["callSetDbId"])
        if not callset_dbid:
            continue
        payload = client.search_response(
            "allelematrix",
            {"callSetDbIds": [callset_dbid], "pageSize": max_calls_per_callset},
            f"allelematrix:{callset['query_id']}:{callset_dbid}",
        )
        result = response_result(payload or {})
        if not isinstance(result, dict):
            continue
        callset_ids = [str(value) for value in result.get("callSetDbIds") or [callset_dbid]]
        variant_ids = [str(value) for value in result.get("variantDbIds") or []]
        matrices = result.get("dataMatrices") or []
        if not isinstance(matrices, list):
            continue
        for matrix in matrices:
            if not isinstance(matrix, dict):
                continue
            matrix_name = clean(matrix.get("dataMatrixName") or matrix.get("dataMatrixAbbreviation"))
            if matrix_name.lower() not in {"", "genotype", "gt"}:
                continue
            values = matrix.get("dataMatrix") or []
            if not isinstance(values, list):
                continue
            for variant_index, row_values in enumerate(values[:max_calls_per_callset]):
                if not isinstance(row_values, list):
                    row_values = [row_values]
                variant_dbid = variant_ids[variant_index] if variant_index < len(variant_ids) else ""
                for callset_index, genotype in enumerate(row_values):
                    matrix_callset = callset_ids[callset_index] if callset_index < len(callset_ids) else callset_dbid
                    if matrix_callset != callset_dbid:
                        continue
                    if isinstance(genotype, list):
                        genotype = "/".join(map(str, genotype))
                    output.append(
                        {
                            "server": client.spec.name,
                            "query_id": callset["query_id"],
                            "sampleDbId": callset["sampleDbId"],
                            "callSetDbId": callset_dbid,
                            "callSetName": callset["callSetName"],
                            "call_source": "search/allelematrix",
                            "variantDbId": variant_dbid,
                            "variantName": "",
                            "genotype": clean(genotype),
                            "genotypeValue": clean(genotype),
                            "phaseSet": "",
                            "raw_json": json.dumps(
                                {"dataMatrixName": matrix_name, "value": genotype},
                                sort_keys=True,
                            ),
                        }
                    )
    return output


def capability_rows(client: BrAPIClient, payload: dict | None) -> list[dict[str, object]]:
    calls = server_calls(payload or {})
    resources = ["germplasm", "pedigree", "progeny", "crosses", "samples", "callsets", "calls", "variants", "variantsets"]
    return [
        {
            "server": client.spec.name,
            "base_url": client.base,
            "authenticated": bool(client.spec.token),
            "serverinfo_available": payload is not None,
            "resource": resource,
            "advertised": capability_advertised(calls, resource),
            "advertised_call_count": len(calls),
        }
        for resource in resources
    ]


def metric(rows: list[dict[str, object]], name: str, value: object) -> None:
    rows.append({"metric": name, "value": value})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded BrAPI recovery of germplasm ancestry and actual marker calls."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--resolver-query",
        type=Path,
        default=Path("genotype_panels/germplasm_resolver/germplasm_cross_query.tsv"),
    )
    parser.add_argument(
        "--classification",
        type=Path,
        default=Path("genotype_panels/germplasm_resolver/germplasm_recovery_classification.tsv"),
    )
    parser.add_argument("--server", action="append", required=True, help="NAME=BrAPI-v2-base-URL")
    parser.add_argument("--token-env", action="append", default=[], help="NAME=environment-variable")
    parser.add_argument("--limit", type=int, default=10, help="Maximum resolver rows for this bounded pilot")
    parser.add_argument("--offset", type=int, default=0, help="Resolver-row offset for resumable batches")
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--max-pedigree-depth", type=int, default=3)
    parser.add_argument("--max-matched-germplasm", type=int, default=100)
    parser.add_argument("--fetch-calls", action="store_true")
    parser.add_argument("--max-calls-per-callset", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--poll-attempts", type=int, default=3)
    parser.add_argument("--poll-sleep", type=float, default=0.5)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("genotype_panels/brapi_recovery_v1")
    )
    args = parser.parse_args()

    root = args.root.resolve()
    resolver_query = args.resolver_query if args.resolver_query.is_absolute() else root / args.resolver_query
    classification = args.classification if args.classification.is_absolute() else root / args.classification
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "response_cache"
    specs = parse_server_specs(args.server, args.token_env)
    if not resolver_query.is_file() or resolver_query.stat().st_size == 0:
        raise FileNotFoundError(f"Resolver query is missing or empty: {resolver_query}")

    query = read_table(resolver_query)
    if classification.is_file() and classification.stat().st_size:
        classes = read_table(classification)
        if "sample_id" in classes.columns and "recovery_class" in classes.columns:
            query = query.merge(
                classes[["sample_id", "recovery_class"]].drop_duplicates("sample_id"),
                on="sample_id",
                how="left",
            )
    selected, query_terms = build_query_terms(query, args.limit, args.offset)
    marker_terms = [term for term in query_terms if term["marker_probe_eligible"]]

    request_log: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    capabilities: list[dict[str, object]] = []
    matches: list[dict[str, object]] = []
    cross_matches: list[dict[str, object]] = []
    pedigree_nodes: list[dict[str, object]] = []
    pedigree_edges: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    callsets: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []

    for spec in specs:
        client = BrAPIClient(
            spec,
            cache_dir,
            request_log,
            failures,
            timeout=args.timeout,
            poll_attempts=args.poll_attempts,
            poll_sleep=args.poll_sleep,
        )
        serverinfo = client.request("GET", "serverinfo", context="serverinfo")
        capabilities.extend(capability_rows(client, serverinfo))
        server_matches: list[dict[str, object]] = []
        for term in query_terms:
            hits = germplasm_search(client, term, args.page_size)
            for hit in hits:
                status = "exact" if exact_record_match(str(term["query_text"]), hit) else "review_candidate"
                server_matches.append(germplasm_row(spec.name, term, hit, status))
            time.sleep(args.sleep)
        matches.extend(server_matches)
        cross_matches.extend(search_crosses(client, query_terms, args.page_size))
        exact = [row for row in server_matches if row["match_status"] == "exact"]
        unique_roots: list[tuple[str, str, str]] = []
        seen_roots: set[tuple[str, str]] = set()
        for row in exact:
            key = (str(row["query_id"]), str(row["germplasmDbId"]))
            if row["germplasmDbId"] and key not in seen_roots:
                seen_roots.add(key)
                unique_roots.append((key[0], key[1], str(row["germplasmName"])))
        nodes, edges = traverse_pedigree(
            client, unique_roots[: args.max_matched_germplasm], args.max_pedigree_depth
        )
        pedigree_nodes.extend(nodes)
        pedigree_edges.extend(edges)
        server_samples = find_samples(client, exact, marker_terms, args.page_size)
        samples.extend(server_samples)
        server_callsets = find_callsets(client, server_samples, marker_terms, args.page_size)
        callsets.extend(server_callsets)
        if args.fetch_calls:
            calls.extend(find_calls(client, server_callsets, args.max_calls_per_callset))
            calls.extend(
                find_allele_matrix_calls(client, server_callsets, args.max_calls_per_callset)
            )

    query_columns = [
        "source_row", "query_id", "query_kind", "query_text", "normalized_query",
        "marker_probe_eligible", "selection_stage_count", "selection_stages",
    ]
    write_tsv(query_terms, out_dir / "brapi_query_terms.tsv", query_columns)
    write_tsv(capabilities, out_dir / "brapi_capability_audit.tsv", [
        "server", "base_url", "authenticated", "serverinfo_available", "resource",
        "advertised", "advertised_call_count",
    ])
    write_tsv(matches, out_dir / "brapi_germplasm_matches.tsv", [
        "server", "query_id", "query_kind", "query_text", "match_status",
        "germplasmDbId", "germplasmName", "defaultDisplayName", "accessionNumber",
        "germplasmPUI", "synonyms", "raw_json",
    ])
    write_tsv(cross_matches, out_dir / "brapi_cross_matches.tsv", [
        "server", "query_id", "query_text", "crossDbId", "crossName", "crossType",
        "parent1_germplasmDbId", "parent2_germplasmDbId", "raw_json",
    ])
    write_tsv(pedigree_nodes, out_dir / "brapi_pedigree_nodes.tsv", [
        "server", "query_id", "germplasmDbId", "germplasmName", "defaultDisplayName",
        "accessionNumber", "depth", "raw_json",
    ])
    write_tsv(pedigree_edges, out_dir / "brapi_pedigree_edges.tsv", [
        "server", "query_id", "child_germplasmDbId", "parent_germplasmDbId",
        "parent_name", "parent_relation", "child_depth", "parent_depth",
    ])
    write_tsv(samples, out_dir / "brapi_samples.tsv", [
        "server", "query_id", "germplasmDbId", "sampleDbId", "sampleName", "studyDbId",
        "observationUnitDbId", "plateDbId", "raw_json",
    ])
    write_tsv(callsets, out_dir / "brapi_callsets.tsv", [
        "server", "query_id", "sampleDbId", "callSetDbId", "callSetName",
        "variantSetDbIds", "created", "updated", "raw_json",
    ])
    write_tsv_gz(calls, out_dir / "brapi_marker_calls.tsv.gz", [
        "server", "query_id", "sampleDbId", "callSetDbId", "callSetName", "call_source",
        "variantDbId", "variantName", "genotype", "genotypeValue", "phaseSet", "raw_json",
    ])
    write_tsv(request_log, out_dir / "brapi_request_log.tsv", [
        "server", "context", "method", "url", "status", "elapsed_seconds",
        "response_bytes", "cache_path",
    ])
    write_tsv(failures, out_dir / "brapi_failures.tsv", [
        "server", "context", "method", "url", "status", "error_type", "error",
    ])

    qc: list[dict[str, object]] = []
    metric(qc, "resolver_rows_available", len(query))
    metric(qc, "resolver_rows_selected", len(selected))
    metric(qc, "query_terms", len(query_terms))
    metric(qc, "marker_probe_terms", len(marker_terms))
    metric(qc, "server_count", len(specs))
    metric(qc, "germplasm_match_rows", len(matches))
    metric(qc, "germplasm_exact_match_rows", sum(row["match_status"] == "exact" for row in matches))
    metric(qc, "cross_match_rows", len(cross_matches))
    metric(qc, "pedigree_node_rows", len(pedigree_nodes))
    metric(qc, "pedigree_edge_rows", len(pedigree_edges))
    metric(qc, "sample_rows", len(samples))
    metric(qc, "callset_rows", len(callsets))
    metric(qc, "marker_call_rows", len(calls))
    metric(qc, "marker_discovery_attempted", True)
    metric(qc, "marker_call_fetch_requested", bool(args.fetch_calls))
    metric(qc, "auth_failure_count", sum(row["status"] == "AUTH_REQUIRED" for row in failures))
    metric(qc, "timeout_count", sum(row["status"] == "TIMEOUT" for row in failures))
    metric(qc, "failure_rows", len(failures))
    metric(qc, "phenotype_values_read", False)
    metric(qc, "outer_test_metrics_read", False)
    metric(qc, "final_holdout_outcomes_read", False)
    write_tsv(qc, out_dir / "brapi_recovery_qc.tsv", ["metric", "value"])

    output_paths = sorted(
        path for path in out_dir.glob("brapi_*") if path.is_file() and path.name != "brapi_recovery_provenance.json"
    )
    provenance = {
        "status": "complete_with_failures" if failures else "complete",
        "created_at_utc": utc_now(),
        "module": "server_genotype_recovery.fetch_brapi_pedigree_markers",
        "input": {"path": str(resolver_query), "sha256": sha256_file(resolver_query)},
        "classification": (
            {"path": str(classification), "sha256": sha256_file(classification)}
            if classification.is_file()
            else None
        ),
        "servers": [
            {"name": spec.name, "base_url": spec.base_url, "authenticated": bool(spec.token)}
            for spec in specs
        ],
        "parameters": {
            "limit": args.limit,
            "offset": args.offset,
            "page_size": args.page_size,
            "max_pedigree_depth": args.max_pedigree_depth,
            "max_matched_germplasm": args.max_matched_germplasm,
            "fetch_calls": bool(args.fetch_calls),
            "max_calls_per_callset": args.max_calls_per_callset,
            "timeout": args.timeout,
            "poll_attempts": args.poll_attempts,
        },
        "selection_data": "identifiers_and_public_api_metadata_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "output_sha256": {str(path.relative_to(out_dir)): sha256_file(path) for path in output_paths},
    }
    (out_dir / "brapi_recovery_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(qc).to_string(index=False))


if __name__ == "__main__":
    main()
