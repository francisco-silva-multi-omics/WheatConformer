from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def request_json(url: str, method: str = "GET", payload: dict | None = None, token: str | None = None, timeout: int = 60) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json", "User-Agent": "wheat-germplasm-api-recovery/1.0"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def brapi_search(base_url: str, query: str, token: str | None = None) -> tuple[list[dict], str]:
    base = base_url.rstrip("/")
    endpoints = [
        (
            f"{base}/brapi/v2/search/germplasm",
            "POST",
            {"germplasmNames": [query], "commonCropNames": ["wheat"], "pageSize": 20},
        ),
        (
            f"{base}/brapi/v2/germplasm?" + urllib.parse.urlencode({"germplasmName": query, "pageSize": 20}),
            "GET",
            None,
        ),
        (
            f"{base}/brapi/v1/germplasm-search?" + urllib.parse.urlencode({"germplasmName": query, "pageSize": 20}),
            "GET",
            None,
        ),
    ]
    errors = []
    for url, method, payload in endpoints:
        try:
            data = request_json(url, method=method, payload=payload, token=token)
            result = data.get("result", {})
            if isinstance(result, dict):
                rows = result.get("data") or result.get("results") or []
            elif isinstance(result, list):
                rows = result
            else:
                rows = []
            return rows if isinstance(rows, list) else [], url
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(" ; ".join(errors))


def genesys_search(query: str) -> tuple[list[dict], str]:
    # Genesys primarily exposes accession/passport metadata. This endpoint may
    # change; failures are recorded and are not fatal to the local workflow.
    url = "https://api.genesys-pgr.org/api/v1/acn/filter"
    payload = {"_text": query, "crop": ["wheat"], "size": 20}
    data = request_json(url, method="POST", payload=payload)
    rows = data.get("content") or data.get("result", {}).get("data") or []
    return rows if isinstance(rows, list) else [], url


def normalize_brapi_record(record: dict, source: str, query_id: str, query_text: str, endpoint: str) -> dict:
    return {
        "query_id": query_id,
        "query_text": query_text,
        "source": source,
        "endpoint": endpoint,
        "germplasmDbId": record.get("germplasmDbId", ""),
        "germplasmName": record.get("germplasmName", ""),
        "germplasmPUI": record.get("germplasmPUI", ""),
        "accessionNumber": record.get("accessionNumber", ""),
        "defaultDisplayName": record.get("defaultDisplayName", ""),
        "synonyms": "|".join(map(str, record.get("synonyms", []) or [])),
        "raw_json": json.dumps(record, sort_keys=True)[:5000],
    }


def normalize_genesys_record(record: dict, query_id: str, query_text: str, endpoint: str) -> dict:
    return {
        "query_id": query_id,
        "query_text": query_text,
        "source": "genesys",
        "endpoint": endpoint,
        "germplasmDbId": record.get("uuid", ""),
        "germplasmName": record.get("acceNumb", ""),
        "germplasmPUI": record.get("doi", ""),
        "accessionNumber": record.get("acceNumb", ""),
        "defaultDisplayName": record.get("acceNumb", ""),
        "synonyms": "",
        "raw_json": json.dumps(record, sort_keys=True)[:5000],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional public/private germplasm API alias enrichment.")
    parser.add_argument("--resolver-query", type=Path, default=Path("genotype_panels/germplasm_resolver/germplasm_cross_query.tsv"))
    parser.add_argument("--classification", type=Path, default=Path("genotype_panels/germplasm_resolver/germplasm_recovery_classification.tsv"))
    parser.add_argument("--brapi-base-url", action="append", default=[], help="Base URL of a BrAPI server.")
    parser.add_argument("--brapi-token", default=None, help="Optional bearer token for BrAPI.")
    parser.add_argument("--include-genesys", action="store_true", help="Also try public Genesys accession search.")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path, default=Path("genotype_panels/germplasm_api_recovery"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    query = pd.read_csv(args.resolver_query, sep="\t", dtype=str, low_memory=False)
    cls = pd.read_csv(args.classification, sep="\t", dtype=str, low_memory=False) if args.classification.exists() else pd.DataFrame()
    if not cls.empty and "sample_id" in cls.columns:
        query = query.merge(cls[["sample_id", "recovery_class"]], on="sample_id", how="left")
    candidates = query.copy()
    if "recovery_class" in candidates.columns:
        candidates = candidates[candidates["recovery_class"].isin(["pedigree_only_cross_available", "unresolved_no_cross", "family_or_cross_genotyped"])].copy()
    query_cols = [c for c in ["sample_id", "cross_name", "parent1", "parent2", "selection_history"] if c in candidates.columns]
    candidates = candidates[query_cols].drop_duplicates().head(args.limit)

    rows = []
    failures = []
    for _, rec in candidates.iterrows():
        query_id = clean(rec.get("sample_id", ""))
        terms = [clean(rec.get(c, "")) for c in ["sample_id", "cross_name", "parent1", "parent2", "selection_history"]]
        terms = [t for t in dict.fromkeys(terms) if t]
        for term in terms:
            for base in args.brapi_base_url:
                try:
                    hits, endpoint = brapi_search(base, term, args.brapi_token)
                    for hit in hits:
                        rows.append(normalize_brapi_record(hit, "brapi", query_id, term, endpoint))
                except Exception as exc:
                    failures.append({"query_id": query_id, "query_text": term, "source": "brapi", "endpoint": base, "error": str(exc)})
                time.sleep(args.sleep)
            if args.include_genesys:
                try:
                    hits, endpoint = genesys_search(term)
                    for hit in hits:
                        rows.append(normalize_genesys_record(hit, query_id, term, endpoint))
                except Exception as exc:
                    failures.append({"query_id": query_id, "query_text": term, "source": "genesys", "endpoint": "api.genesys-pgr.org", "error": str(exc)})
                time.sleep(args.sleep)

    pd.DataFrame(rows).to_csv(args.out_dir / "germplasm_api_alias_matches.tsv", sep="\t", index=False)
    pd.DataFrame(failures).to_csv(args.out_dir / "germplasm_api_alias_failures.tsv", sep="\t", index=False)
    qc = pd.DataFrame(
        [
            {"metric": "query_rows_considered", "value": len(candidates)},
            {"metric": "brapi_base_url_count", "value": len(args.brapi_base_url)},
            {"metric": "genesys_enabled", "value": bool(args.include_genesys)},
            {"metric": "match_rows", "value": len(rows)},
            {"metric": "failure_rows", "value": len(failures)},
            {"metric": "genotype_call_fetch_attempted", "value": False},
            {"metric": "note", "value": "Public mode collects aliases/passport/pedigree metadata only; marker calls require a configured BrAPI genotyping endpoint and credentials."},
        ]
    )
    qc.to_csv(args.out_dir / "germplasm_api_recovery_qc.tsv", sep="\t", index=False)
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
