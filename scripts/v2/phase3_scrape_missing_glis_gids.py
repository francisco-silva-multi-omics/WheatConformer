"""Resolve trial DOI tokens absent from the supplied clean GLIS registry.

Only DOI tokens present in the frozen local DOI ledger are queried. Every response
is cached and hashed. A GID is accepted only when the returned page names the same
DOI and contains exactly one ``GID <integer>`` token.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pyarrow.parquet as pq
import requests


DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.I)
GID_RE = re.compile(r"\bGID\s+([0-9]+)\b", re.I)


def normalized_text(payload: str) -> str:
    without_scripts = re.sub(r"<(script|style)\b.*?</\1>", " ", payload, flags=re.I | re.S)
    without_tags = re.sub(r"<[^>]+>", " ", without_scripts)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def parse_glis_page(requested_doi: str, payload: str) -> tuple[str, list[str], str]:
    text = normalized_text(payload)
    doi_present = requested_doi.upper() in text.upper()
    gids = sorted(set(GID_RE.findall(text)), key=int)
    if not doi_present:
        return "REJECT_PAGE_DOI_MISMATCH", gids, ""
    if len(gids) == 0:
        return "REJECT_NO_GID_TOKEN", gids, ""
    if len(gids) > 1:
        return "REJECT_MULTIPLE_GID_TOKENS", gids, ""
    return "ACCEPT_EXACT_PAGE_DOI_SINGLE_GID", gids, gids[0]


def valid_doi(value: str) -> bool:
    return bool(DOI_RE.fullmatch(str(value).strip()))


def request_url(doi: str) -> str:
    prefix, suffix = doi.split("/", 1)
    return f"https://glis.fao.org/glis/doi/{prefix}/{quote(suffix, safe='')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doi-ledger", type=Path, required=True)
    parser.add_argument("--clean-resolver", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = result_dir / "response_cache"
    cache_dir.mkdir()

    doi_rows = pq.read_table(args.doi_ledger.resolve()).to_pandas().fillna("")
    doi_rows["DOI"] = doi_rows["DOI"].astype(str).str.strip()
    local = doi_rows[doi_rows["DOI"].map(valid_doi)].copy()
    clean = pd.read_csv(args.clean_resolver.resolve(), sep="\t", dtype=str, low_memory=False).fillna("")
    clean["DOI"] = clean["DOI"].astype(str).str.strip()
    clean["glis_gid"] = clean["glis_gid"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    if clean["DOI"].eq("").any() or clean["glis_gid"].eq("").any():
        raise RuntimeError("Supplied clean GLIS resolver contains blank DOI or GID")
    if clean["DOI"].duplicated().any():
        raise RuntimeError("Supplied clean GLIS resolver contains duplicate DOI")
    if clean["glis_gid"].duplicated().any():
        raise RuntimeError("Supplied clean GLIS resolver contains duplicate GID")

    local_dois = sorted(set(local["DOI"]))
    clean_dois = set(clean["DOI"])
    missing = [doi for doi in local_dois if doi not in clean_dois]
    source_map = (
        local.groupby("DOI", sort=False)["doi_source_file"]
        .agg(lambda values: ";".join(sorted(set(map(str, values)))))
        .to_dict()
    )

    session = requests.Session()
    session.headers.update({
        "User-Agent": "WheatConformer-Stage1-v2-reproducibility-audit/1.0",
        "Accept": "text/html,application/xhtml+xml",
    })
    response_rows: list[dict[str, object]] = []
    accepted_rows: list[dict[str, object]] = []
    for index, doi in enumerate(missing, start=1):
        url = request_url(doi)
        response = None
        error = ""
        attempts = 0
        for attempts in range(1, args.max_attempts + 1):
            try:
                response = session.get(url, timeout=args.timeout_seconds, allow_redirects=True)
                if response.status_code < 500:
                    break
                error = f"HTTP_{response.status_code}"
            except requests.RequestException as exc:
                error = f"{type(exc).__name__}: {exc}"
            if attempts < args.max_attempts:
                time.sleep(min(2.0 ** attempts, 8.0))

        queried_at = datetime.now(timezone.utc).isoformat()
        payload = b"" if response is None else response.content
        content_sha = hashlib.sha256(payload).hexdigest()
        cache_name = hashlib.sha256(doi.encode("utf-8")).hexdigest() + ".html"
        cache_path = cache_dir / cache_name
        cache_path.write_bytes(payload)
        http_status = "" if response is None else response.status_code
        final_url = "" if response is None else response.url
        parser_status = "REJECT_REQUEST_FAILED"
        gids: list[str] = []
        accepted_gid = ""
        if response is not None and response.status_code == 200:
            parser_status, gids, accepted_gid = parse_glis_page(doi, response.text)
        elif response is not None:
            parser_status = f"REJECT_HTTP_{response.status_code}"
        response_rows.append({
            "DOI": doi,
            "requested_url": url,
            "final_url": final_url,
            "queried_at_utc": queried_at,
            "attempts": attempts,
            "http_status": http_status,
            "response_bytes": len(payload),
            "response_sha256": content_sha,
            "cache_file": f"response_cache/{cache_name}",
            "gid_candidates": ";".join(gids),
            "parser_status": parser_status,
            "accepted_gid": accepted_gid,
            "request_error": error,
            "source_files": source_map.get(doi, ""),
        })
        if accepted_gid:
            accepted_rows.append({
                "DOI": doi,
                "glis_url": url,
                "glis_gid": accepted_gid,
                "glis_status": "OK_LIVE_PHASE3",
                "source_files": source_map.get(doi, ""),
                "glis_panel_sample_id": "GID" + accepted_gid,
                "resolver_source": "PHASE3_LIVE_GLIS_HTML_SINGLE_GID",
                "response_sha256": content_sha,
                "queried_at_utc": queried_at,
            })
        if index < len(missing):
            time.sleep(args.delay_seconds)
        if index % 50 == 0 or index == len(missing):
            print(f"GLIS DOI queries completed: {index}/{len(missing)}", flush=True)

    responses = pd.DataFrame(response_rows)
    responses.to_csv(result_dir / "glis_response_ledger.tsv", sep="\t", index=False)
    accepted = pd.DataFrame(accepted_rows)
    if accepted.empty:
        accepted = pd.DataFrame(columns=[
            "DOI", "glis_url", "glis_gid", "glis_status", "source_files",
            "glis_panel_sample_id", "resolver_source", "response_sha256", "queried_at_utc",
        ])
    accepted.to_csv(result_dir / "glis_live_recoveries.tsv", sep="\t", index=False)

    clean_out = clean.copy()
    clean_out["resolver_source"] = "SUPPLIED_CLEAN_GLIS_GID_OK"
    clean_out["response_sha256"] = ""
    clean_out["queried_at_utc"] = ""
    combined = pd.concat([clean_out, accepted], ignore_index=True, sort=False)
    if combined["DOI"].duplicated().any():
        raise RuntimeError("Combined GLIS resolver contains duplicate DOI")
    doi_gid_conflicts = (
        combined.groupby("DOI")["glis_gid"].nunique().gt(1).sum()
    )
    if doi_gid_conflicts:
        raise RuntimeError(f"Combined GLIS resolver contains {doi_gid_conflicts} DOI-to-GID conflicts")
    combined.sort_values("DOI").to_csv(result_dir / "glis_resolver_v2.tsv", sep="\t", index=False)

    resolved_dois = set(combined["DOI"])
    unresolved = responses[~responses["DOI"].isin(resolved_dois)].copy()
    unresolved.to_csv(result_dir / "glis_unresolved_dois.tsv", sep="\t", index=False)
    summary = {
        "status": "PASS_GLIS_RESOLVER_BUILT",
        "local_valid_doi_rows": len(local),
        "local_unique_valid_dois": len(local_dois),
        "supplied_clean_resolver_rows": len(clean),
        "local_dois_already_in_clean_resolver": len(set(local_dois) & clean_dois),
        "live_queries_attempted": len(missing),
        "live_single_gid_recoveries": len(accepted),
        "live_unresolved_dois": len(unresolved),
        "final_resolver_rows": len(combined),
        "local_dois_resolved_final": len(set(local_dois) & resolved_dois),
        "local_dois_unresolved_final": len(set(local_dois) - resolved_dois),
        "final_duplicate_dois": int(combined["DOI"].duplicated().sum()),
        "final_doi_to_multiple_gid_conflicts": int(doi_gid_conflicts),
    }
    (result_dir / "glis_resolver_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
