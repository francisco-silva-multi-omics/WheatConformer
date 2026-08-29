#!/usr/bin/env python3
"""Fetch identifier-bearing metadata for the CIMMYT 130K GBS release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


PROTOCOL_VERSION = "cimmyt_130k_identifier_metadata_fetch_v1"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_RUNINFO = "https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo"
ENA_FILEREPORT = "https://www.ebi.ac.uk/ena/portal/api/filereport"
USER_AGENT = "WheatConformer-CIMMYT-130K-identifier-recovery/1.0"
GID_RE = re.compile(r"^GID[0-9]+$", re.IGNORECASE)
WGE_RE = re.compile(r"^WGE[0-9]+$", re.IGNORECASE)
PAIR_RE = re.compile(r"\(([^(),]+),([ACGT]+)\)", re.IGNORECASE)
RUN_RE = re.compile(r"^(?:SRR|ERR|DRR)[0-9]+$", re.IGNORECASE)

ENA_FIELDS = ",".join(
    [
        "run_accession",
        "experiment_accession",
        "experiment_alias",
        "run_alias",
        "library_name",
        "sample_accession",
        "secondary_sample_accession",
        "sample_alias",
        "sample_title",
        "sample_description",
        "study_accession",
        "secondary_study_accession",
        "center_name",
        "broker_name",
        "submitted_ftp",
        "submitted_md5",
        "submitted_bytes",
    ]
)


@dataclass(frozen=True)
class FetchSpec:
    source: str
    accession: str
    url: str
    path: Path
    validator: Callable[[bytes], None]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_identifier(value: object) -> str:
    return normalize_text(value).upper()


def run_library_key(filename: object) -> str:
    """Return the GBS run key encoded in the TASSEL filename.

    The suffix before ``x`` distinguishes resequenced library parts such as
    GBS0752R and GBS1287F. A terminal R1 is the read marker used by newer
    filenames and is not part of the library key.
    """

    basename = Path(normalize_text(filename)).name.upper()
    modern = re.match(r"^(GBS[0-9]+)R1X", basename)
    if modern:
        return modern.group(1)
    legacy = re.match(r"^(GBS[0-9]+[A-Z]?)[X_]", basename)
    return legacy.group(1) if legacy else ""


def key_library_key(value: object, valid_run_keys: set[str]) -> str:
    label = normalize_identifier(value)
    if label in valid_run_keys:
        return label
    if label and label[-1:] in {"A", "B", "C", "D"} and label[:-1] in valid_run_keys:
        return label[:-1]
    return ""


def validate_xml(payload: bytes) -> None:
    ET.fromstring(payload)


def validate_runinfo(payload: bytes) -> None:
    text = payload.decode("utf-8-sig", "replace")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows or "Run" not in rows[0]:
        raise ValueError("NCBI RunInfo response has no run row")


def validate_ena(payload: bytes) -> None:
    value = json.loads(payload)
    if not isinstance(value, list) or not value or not value[0].get("run_accession"):
        raise ValueError("ENA response has no run row")


def request_bytes(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def cache_valid(spec: FetchSpec) -> bool:
    if not spec.path.is_file():
        return False
    try:
        spec.validator(spec.path.read_bytes())
    except Exception:
        return False
    return True


def fetch_cached(
    spec: FetchSpec,
    *,
    timeout: int,
    retry_max: int,
    retry_sleep: float,
) -> tuple[str, str]:
    if cache_valid(spec):
        payload = spec.path.read_bytes()
        return "CACHED", sha256_bytes(payload)
    spec.path.unlink(missing_ok=True)

    error = ""
    for attempt in range(1, retry_max + 1):
        try:
            payload = request_bytes(spec.url, timeout)
            spec.validator(payload)
            atomic_write(spec.path, payload)
            return "FETCHED", sha256_bytes(payload)
        except (OSError, ValueError, ET.ParseError, json.JSONDecodeError, urllib.error.URLError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < retry_max:
                time.sleep(retry_sleep * attempt)
    return "FAILED_RETRYABLE", error


def ncbi_url(database: str, accession: str, api_key: str) -> str:
    parameters = {"db": database, "id": accession, "retmode": "xml"}
    if api_key:
        parameters["api_key"] = api_key
    return NCBI_EFETCH + "?" + urllib.parse.urlencode(parameters)


def run_specs(out_dir: Path, run: str, api_key: str) -> list[FetchSpec]:
    return [
        FetchSpec(
            "NCBI_SRA_EXPERIMENT_XML",
            run,
            ncbi_url("sra", run, api_key),
            out_dir / "raw" / "ncbi_sra_xml" / f"{run}.xml",
            validate_xml,
        ),
        FetchSpec(
            "NCBI_SRA_RUNINFO",
            run,
            NCBI_RUNINFO + "?" + urllib.parse.urlencode({"acc": run}),
            out_dir / "raw" / "ncbi_runinfo" / f"{run}.csv",
            validate_runinfo,
        ),
        FetchSpec(
            "ENA_READ_RUN",
            run,
            ENA_FILEREPORT
            + "?"
            + urllib.parse.urlencode(
                {
                    "accession": run,
                    "result": "read_run",
                    "format": "json",
                    "fields": ENA_FIELDS,
                }
            ),
            out_dir / "raw" / "ena_read_run" / f"{run}.json",
            validate_ena,
        ),
    ]


def supporting_specs(
    out_dir: Path,
    biosamples: Iterable[str],
    bioprojects: Iterable[str],
    api_key: str,
) -> list[FetchSpec]:
    specs = []
    for accession in sorted(set(biosamples)):
        specs.append(
            FetchSpec(
                "NCBI_BIOSAMPLE_XML",
                accession,
                ncbi_url("biosample", accession, api_key),
                out_dir / "raw" / "ncbi_biosample_xml" / f"{accession}.xml",
                validate_xml,
            )
        )
    for accession in sorted(set(bioprojects)):
        specs.append(
            FetchSpec(
                "NCBI_BIOPROJECT_XML",
                accession,
                ncbi_url("bioproject", accession, api_key),
                out_dir / "raw" / "ncbi_bioproject_xml" / f"{accession}.xml",
                validate_xml,
            )
        )
    return specs


def parse_sra_xml(path: Path, expected_run: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    root = ET.parse(path).getroot()
    experiment = root.find(".//EXPERIMENT")
    run = root.find(".//RUN")
    sample = root.find(".//SAMPLE")
    description = ""
    if experiment is not None:
        description = normalize_text(experiment.findtext(".//DESIGN_DESCRIPTION"))
    pairs = []
    seen = set()
    for identifier, barcode in PAIR_RE.findall(description):
        key = (normalize_identifier(identifier), normalize_identifier(barcode))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            {
                "run_accession": expected_run,
                "submitted_identifier": key[0],
                "submitted_barcode": key[1],
                "evidence_source": "NCBI_EXPERIMENT_DESIGN_DESCRIPTION",
                "source_xml_sha256": sha256_file(path),
            }
        )
    summary = {
        "run_accession": expected_run,
        "experiment_accession": normalize_text(experiment.get("accession") if experiment is not None else ""),
        "experiment_alias": normalize_text(experiment.get("alias") if experiment is not None else ""),
        "run_alias": normalize_text(run.get("alias") if run is not None else ""),
        "sample_accession": normalize_text(sample.get("accession") if sample is not None else ""),
        "submitted_pair_count": str(len(pairs)),
        "source_xml_sha256": sha256_file(path),
    }
    return pairs, summary


def read_hmp_axis(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            member = next(name for name in archive.namelist() if not name.startswith("__MACOSX"))
            line = archive.open(member).readline().decode("utf-8-sig", "replace")
    else:
        with path.open("rt", encoding="utf-8-sig", errors="replace") as handle:
            line = handle.readline()
    return {normalize_identifier(value) for value in line.rstrip("\r\n").split("\t")[11:]}


def crosswalk_class(dryad_identifier: str, submitted_identifier: str) -> str:
    if not dryad_identifier:
        return "NCBI_ONLY_BARCODE"
    if not submitted_identifier:
        return "DRYAD_ONLY_BARCODE"
    if dryad_identifier == submitted_identifier:
        return "EXACT_IDENTIFIER_AND_BARCODE"
    if GID_RE.fullmatch(dryad_identifier) and WGE_RE.fullmatch(submitted_identifier):
        return "GID_TO_WGE_ALIAS_CANDIDATE"
    if WGE_RE.fullmatch(dryad_identifier) and GID_RE.fullmatch(submitted_identifier):
        return "WGE_TO_GID_ALIAS_CANDIDATE"
    if GID_RE.fullmatch(dryad_identifier) and GID_RE.fullmatch(submitted_identifier):
        return "CONFLICTING_GID_FOR_BARCODE"
    return "OTHER_IDENTIFIER_MISMATCH"


def build_crosswalk(
    key: pd.DataFrame,
    pairs: pd.DataFrame,
    matrix_axis: set[str],
) -> pd.DataFrame:
    dryad = key[
        [
            "run_accession",
            "run_library_key",
            "Flowcell",
            "Lane",
            "Barcode",
            "FullSampleName",
            "LibraryPlateID",
            "DNA_Plate",
            "SampleDNA_Well",
            "sample_id",
        ]
    ].copy()
    dryad["dryad_identifier"] = dryad["FullSampleName"].map(normalize_identifier)
    dryad["barcode_norm"] = dryad["Barcode"].map(normalize_identifier)
    submitted = pairs.copy()
    if submitted.empty:
        submitted = pd.DataFrame(
            columns=["run_accession", "submitted_identifier", "submitted_barcode", "evidence_source", "source_xml_sha256"]
        )
    submitted["barcode_norm"] = submitted["submitted_barcode"].map(normalize_identifier)
    merged = dryad.merge(
        submitted,
        on=["run_accession", "barcode_norm"],
        how="outer",
        validate="one_to_one",
    )
    for column in ("dryad_identifier", "submitted_identifier"):
        merged[column] = merged[column].fillna("").map(normalize_identifier)
    merged["crosswalk_class"] = [
        crosswalk_class(dryad, submitted)
        for dryad, submitted in zip(
            merged["dryad_identifier"], merged["submitted_identifier"], strict=True
        )
    ]
    merged["dryad_identifier_in_matrix_axis"] = merged["dryad_identifier"].isin(matrix_axis)
    merged["submitted_identifier_in_matrix_axis"] = merged["submitted_identifier"].isin(matrix_axis)
    return merged.sort_values(["run_accession", "barcode_norm"], na_position="last")


def prepare_inputs(
    key_workbook: Path,
    sra_workbook: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = pd.read_excel(key_workbook, dtype=str).fillna("")
    sra = pd.read_excel(sra_workbook, dtype=str).fillna("")
    required_key = {
        "Flowcell",
        "Lane",
        "Barcode",
        "FullSampleName",
        "LibraryPlateID",
        "DNA_Plate",
        "SampleDNA_Well",
        "sample_id",
    }
    required_sra = {"Accession", "BioProject", "BioSample", "file_name_for_Tassel"}
    if missing := required_key - set(key):
        raise ValueError(f"Key workbook is missing columns: {sorted(missing)}")
    if missing := required_sra - set(sra):
        raise ValueError(f"SRA workbook is missing columns: {sorted(missing)}")

    sra["run_accession"] = sra["Accession"].map(normalize_identifier)
    if not sra["run_accession"].map(lambda value: bool(RUN_RE.fullmatch(value))).all():
        raise ValueError("SRA workbook contains an invalid run accession")
    if sra["run_accession"].duplicated().any():
        raise ValueError("SRA workbook contains duplicate run accessions")
    sra["run_library_key"] = sra["file_name_for_Tassel"].map(run_library_key)
    if sra["run_library_key"].eq("").any():
        raise ValueError("Could not derive a library key from every TASSEL filename")
    if sra["run_library_key"].duplicated().any():
        duplicate = sorted(sra.loc[sra["run_library_key"].duplicated(False), "run_library_key"].unique())
        raise ValueError(f"Run library keys are not unique: {duplicate}")

    valid_run_keys = set(sra["run_library_key"])
    key["run_library_key"] = key["LibraryPlateID"].map(
        lambda value: key_library_key(value, valid_run_keys)
    )
    run_map = sra.set_index("run_library_key")["run_accession"]
    key["run_accession"] = key["run_library_key"].map(run_map).fillna("")
    return key, sra


def output_artifacts(out_dir: Path) -> dict[str, str]:
    artifacts = {}
    for path in sorted(out_dir.glob("*")):
        if path.is_file() and path.name not in {
            "metadata_fetch_provenance.json",
            "metadata_fetch_status.json",
        }:
            artifacts[path.name] = sha256_file(path)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--key-workbook", type=Path, required=True)
    parser.add_argument("--sra-workbook", type=Path, required=True)
    parser.add_argument("--hmp", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("audit/v2/cimmyt_130k_identifier_metadata_fetch_v1"),
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum pending runs; 0 means all")
    parser.add_argument(
        "--run-accession",
        action="append",
        default=[],
        help="Restrict the fetch to one or more runs; omission selects all runs",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retry-max", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--request-delay", type=float, default=0.35)
    parser.add_argument("--ncbi-api-key-env", default="NCBI_API_KEY")
    args = parser.parse_args()

    root = args.root.resolve()
    key_workbook = (root / args.key_workbook).resolve() if not args.key_workbook.is_absolute() else args.key_workbook.resolve()
    sra_workbook = (root / args.sra_workbook).resolve() if not args.sra_workbook.is_absolute() else args.sra_workbook.resolve()
    hmp = None
    if args.hmp:
        hmp = (root / args.hmp).resolve() if not args.hmp.is_absolute() else args.hmp.resolve()
    out_dir = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    key, sra = prepare_inputs(key_workbook, sra_workbook)
    api_key = os.environ.get(args.ncbi_api_key_env, "").strip()
    inventory_runs = sra["run_accession"].tolist()
    requested_runs = [normalize_identifier(value) for value in args.run_accession]
    unknown_runs = sorted(set(requested_runs) - set(inventory_runs))
    if unknown_runs:
        raise ValueError(f"Requested runs are absent from the SRA workbook: {unknown_runs}")
    all_runs = list(dict.fromkeys(requested_runs)) if requested_runs else inventory_runs
    pending = []
    for run in all_runs:
        if any(not cache_valid(spec) for spec in run_specs(out_dir, run, api_key)):
            pending.append(run)
    selected = pending[: args.limit] if args.limit > 0 else pending

    request_log: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for index, run in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] FETCH identifier metadata {run}", flush=True)
        for spec in run_specs(out_dir, run, api_key):
            status, detail = fetch_cached(
                spec,
                timeout=args.timeout,
                retry_max=args.retry_max,
                retry_sleep=args.retry_sleep,
            )
            request_log.append(
                {
                    "source": spec.source,
                    "accession": spec.accession,
                    "status": status,
                    "detail_or_sha256": detail,
                }
            )
            if status == "FAILED_RETRYABLE":
                failures.append(request_log[-1])
            time.sleep(args.request_delay)

    selected_sra = sra[sra["run_accession"].isin(all_runs)]
    support_specs = supporting_specs(
        out_dir,
        selected_sra["BioSample"].map(normalize_identifier),
        selected_sra["BioProject"].map(normalize_identifier),
        api_key,
    )
    for spec in support_specs:
        status, detail = fetch_cached(
            spec,
            timeout=args.timeout,
            retry_max=args.retry_max,
            retry_sleep=args.retry_sleep,
        )
        request_log.append(
            {
                "source": spec.source,
                "accession": spec.accession,
                "status": status,
                "detail_or_sha256": detail,
            }
        )
        if status == "FAILED_RETRYABLE":
            failures.append(request_log[-1])
        time.sleep(args.request_delay)

    pair_rows: list[dict[str, str]] = []
    run_summaries: list[dict[str, str]] = []
    completed_runs = []
    for run in all_runs:
        xml_path = out_dir / "raw" / "ncbi_sra_xml" / f"{run}.xml"
        runinfo_path = out_dir / "raw" / "ncbi_runinfo" / f"{run}.csv"
        ena_path = out_dir / "raw" / "ena_read_run" / f"{run}.json"
        if not (xml_path.is_file() and runinfo_path.is_file() and ena_path.is_file()):
            continue
        try:
            validate_xml(xml_path.read_bytes())
            validate_runinfo(runinfo_path.read_bytes())
            validate_ena(ena_path.read_bytes())
            pairs, summary = parse_sra_xml(xml_path, run)
        except Exception:
            continue
        pair_rows.extend(pairs)
        run_summaries.append(summary)
        completed_runs.append(run)

    completed_key = key[key["run_accession"].isin(completed_runs)].copy()
    pairs = pd.DataFrame(pair_rows)
    matrix_axis = read_hmp_axis(hmp)
    crosswalk = build_crosswalk(completed_key, pairs, matrix_axis)
    candidates = crosswalk[
        crosswalk["crosswalk_class"].isin(
            {"GID_TO_WGE_ALIAS_CANDIDATE", "WGE_TO_GID_ALIAS_CANDIDATE"}
        )
    ].copy()

    source_manifest = pd.DataFrame(
        [
            {
                "source": "DRYAD_KEY_WORKBOOK",
                "path": str(key_workbook),
                "bytes": key_workbook.stat().st_size,
                "sha256": sha256_file(key_workbook),
            },
            {
                "source": "DRYAD_SRA_WORKBOOK",
                "path": str(sra_workbook),
                "bytes": sra_workbook.stat().st_size,
                "sha256": sha256_file(sra_workbook),
            },
        ]
    )
    if hmp and hmp.is_file():
        source_manifest.loc[len(source_manifest)] = {
            "source": "DRYAD_HAPMAP_AXIS",
            "path": str(hmp),
            "bytes": hmp.stat().st_size,
            "sha256": sha256_file(hmp),
        }

    source_manifest.to_csv(out_dir / "source_input_manifest.tsv", sep="\t", index=False)
    sra.to_csv(out_dir / "run_library_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(run_summaries).to_csv(out_dir / "ncbi_run_metadata_summary.tsv", sep="\t", index=False)
    pairs.to_csv(out_dir / "ncbi_submitted_barcode_keys.tsv.gz", sep="\t", index=False, compression="gzip")
    crosswalk.to_csv(out_dir / "dryad_ncbi_barcode_crosswalk.tsv.gz", sep="\t", index=False, compression="gzip")
    candidates.to_csv(out_dir / "gid_wge_crosswalk_candidates.tsv", sep="\t", index=False)
    pd.DataFrame(request_log, columns=["source", "accession", "status", "detail_or_sha256"]).to_csv(
        out_dir / "metadata_request_log.tsv", sep="\t", index=False
    )
    pd.DataFrame(failures, columns=["source", "accession", "status", "detail_or_sha256"]).to_csv(
        out_dir / "metadata_fetch_failures.tsv", sep="\t", index=False
    )

    unmatched_key_rows = int(key["run_accession"].eq("").sum())
    status_counts = crosswalk["crosswalk_class"].value_counts().sort_index().to_dict()
    support_complete = all(cache_valid(spec) for spec in support_specs)
    complete = len(completed_runs) == len(all_runs) and support_complete and not failures
    status = {
        "status": "PASS" if not failures else "PASS_WITH_RETRYABLE_FAILURES",
        "run_status": "COMPLETE" if complete else "PARTIAL",
        "protocol_version": PROTOCOL_VERSION,
        "selection_data": "public_identifier_metadata_only",
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "sequence_payloads_downloaded": 0,
        "run_count": len(all_runs),
        "completed_run_count": len(completed_runs),
        "pending_run_count": len(all_runs) - len(completed_runs),
        "selected_pending_run_count": len(selected),
        "supporting_metadata_complete": support_complete,
        "dryad_key_rows": len(key),
        "dryad_key_rows_without_run_mapping": unmatched_key_rows,
        "submitted_barcode_key_rows": len(pairs),
        "crosswalk_rows": len(crosswalk),
        "gid_wge_crosswalk_candidate_rows": len(candidates),
        "matrix_axis_loaded": bool(matrix_axis),
        "matrix_axis_rows": len(matrix_axis),
        "crosswalk_class_counts": status_counts,
        "checks": {
            "source_workbooks_hashed": True,
            "run_accessions_unique": not sra["run_accession"].duplicated().any(),
            "run_library_keys_unique": not sra["run_library_key"].duplicated().any(),
            "sequence_payloads_not_downloaded": True,
            "phenotypes_and_outcomes_unread": True,
        },
    }
    write_json(out_dir / "metadata_fetch_status.json", status)
    provenance = {
        **status,
        "completed_at_utc": utc_now(),
        "source_endpoints": {
            "ncbi_efetch": NCBI_EFETCH,
            "ncbi_runinfo": NCBI_RUNINFO,
            "ena_filereport": ENA_FILEREPORT,
        },
        "artifacts": output_artifacts(out_dir),
    }
    write_json(out_dir / "metadata_fetch_provenance.json", provenance)
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
