from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from server_genotype_recovery.fetch_brapi_pedigree_markers import clean


NO_LOCAL_MATCH = "NO_LOCAL_MATCH"
LOCAL_EXACT_CHECKSUM = "LOCAL_EXACT_CHECKSUM"
LOCAL_NAME_SIZE_REVIEW = "LOCAL_NAME_SIZE_REVIEW"
LOCAL_CHECKSUM_MISMATCH = "LOCAL_CHECKSUM_MISMATCH"
LOCAL_DERIVED_REPRESENTATION_REVIEW = "LOCAL_DERIVED_REPRESENTATION_REVIEW"
LOCAL_DATASET_DIRECTORY_REVIEW = "LOCAL_DATASET_DIRECTORY_REVIEW"

LOCAL_REVIEW_STATUSES = {
    LOCAL_NAME_SIZE_REVIEW,
    LOCAL_CHECKSUM_MISMATCH,
    LOCAL_DERIVED_REPRESENTATION_REVIEW,
    LOCAL_DATASET_DIRECTORY_REVIEW,
}

COMPRESSION_SUFFIXES = (".gz", ".zip", ".7z", ".bz2", ".xz", ".tar")
GENERIC_DATASET_TOKENS = {
    "and",
    "array",
    "cimmyt",
    "data",
    "dataset",
    "derived",
    "for",
    "genotype",
    "genotypic",
    "genotyping",
    "germplasm",
    "high",
    "international",
    "marker",
    "markers",
    "panel",
    "nursery",
    "results",
    "screening",
    "selection",
    "semi",
    "spring",
    "snp",
    "snps",
    "the",
    "trial",
    "wheat",
    "yield",
}
STRONG_DATASET_TOKENS = {
    "35k",
    "80k",
    "90k",
    "dartag",
    "hibap",
    "iwyp",
    "masagro",
    "seeds",
}

TRIAL_FAMILY_PATTERNS = {
    "IBWSN": (
        r"(?<![a-z])ibwsn(?![a-z])",
        r"international\s+bread\s+wheat\s+screening\s+nursery",
    ),
    "SAWSN": (r"(?<![a-z])sawsn(?![a-z])", r"semi[- ]arid\s+wheat\s+screening\s+nursery"),
    "SAWYT": (r"(?<![a-z])sawyt(?![a-z])", r"semi[- ]arid\s+wheat\s+yield\s+trial"),
    "ESWYT": (r"(?<![a-z])eswyt(?![a-z])", r"elite\s+spring\s+wheat\s+yield\s+trial"),
    "HTWYT": (
        r"(?<![a-z])(?:htwyt|thwyt)(?![a-z])",
        r"high\s+temperature\s+wheat\s+yield\s+trial",
    ),
    "HRWYT": (r"(?<![a-z])hrwyt(?![a-z])", r"high\s+rainfall\s+wheat\s+yield\s+trial"),
    "IDSN": (r"(?<![a-z])idsn(?![a-z])", r"international\s+durum\s+screening\s+nursery"),
}


def normalized_filename(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean(value).lower())


def normalized_payload_filename(value: object) -> str:
    name = clean(value).lower()
    changed = True
    while changed:
        changed = False
        for suffix in COMPRESSION_SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
                break
    return normalized_filename(name)


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", clean(value).lower())
        if len(token) >= 3 and token not in GENERIC_DATASET_TOKENS
    }


def _trial_numbers(value: object) -> set[int]:
    text = clean(value).lower().replace("_", " ")
    numbers: set[int] = set()
    for start, finish in re.findall(
        r"(?<![a-z0-9])(\d{1,3})(?:st|nd|rd|th)?\s*(?:to|-)\s*"
        r"(\d{1,3})(?:st|nd|rd|th)?(?![a-z0-9])",
        text,
    ):
        lower, upper = sorted((int(start), int(finish)))
        numbers.update(range(lower, upper + 1))
    numbers.update(
        int(value)
        for value in re.findall(
            r"(?<![a-z0-9])(\d{1,3})(?:st|nd|rd|th)(?![a-z0-9])", text
        )
    )
    numbers.update(
        int(value)
        for value in re.findall(
            r"(?<![a-z0-9])(?:c)?(\d{1,3})(?=(?:ibwsn|sawsn|sawyt|eswyt|"
            r"htwyt|thwyt|hrwyt|idsn)\b)",
            text,
        )
    )
    return numbers


def _trial_families(value: object) -> set[str]:
    text = re.sub(r"[_]+", " ", clean(value).lower())
    return {
        family
        for family, patterns in TRIAL_FAMILY_PATTERNS.items()
        if any(re.search(pattern, text) for pattern in patterns)
    }


def _compact_phrase(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean(value).lower())


def dataset_directory_match(dataset_name: object, relative_parent: object) -> bool:
    dataset_name = clean(dataset_name)
    relative_parent = clean(relative_parent)
    if not dataset_name or not relative_parent:
        return False
    dataset_numbers = _trial_numbers(dataset_name)
    parent_numbers = _trial_numbers(relative_parent)
    if dataset_numbers and parent_numbers and dataset_numbers.isdisjoint(parent_numbers):
        return False
    dataset_families = _trial_families(dataset_name)
    parent_families = _trial_families(relative_parent)
    if (
        dataset_families
        and parent_families
        and dataset_families.isdisjoint(parent_families)
    ):
        return False
    if (
        dataset_numbers & parent_numbers
        and dataset_families & parent_families
    ):
        return True

    dataset_phrase = _compact_phrase(dataset_name)
    parent_phrase = _compact_phrase(relative_parent)
    if len(dataset_phrase) >= 12 and (
        dataset_phrase in parent_phrase or parent_phrase in dataset_phrase
    ):
        return True

    dataset_tokens = _tokens(dataset_name)
    parent_tokens = _tokens(relative_parent)
    shared = dataset_tokens & parent_tokens
    if shared & STRONG_DATASET_TOKENS:
        return True
    return len(shared) >= 2 and len(shared) / max(1, len(dataset_tokens)) >= 0.6


def inventory_local_files(local_roots: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root in local_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            rows.append(
                {
                    "local_root": str(root.resolve()),
                    "local_path": str(path.resolve()),
                    "local_relative_path": str(path.relative_to(root)),
                    "local_relative_parent": str(path.relative_to(root).parent),
                    "local_filename": path.name,
                    "local_filesize": int(stat.st_size),
                    "local_mtime_ns": int(stat.st_mtime_ns),
                    "normalized_filename": normalized_filename(path.name),
                    "normalized_payload_filename": normalized_payload_filename(path.name),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "local_root",
            "local_path",
            "local_relative_path",
            "local_relative_parent",
            "local_filename",
            "local_filesize",
            "local_mtime_ns",
            "normalized_filename",
            "normalized_payload_filename",
        ],
    )


def _hash_algorithm(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", clean(value).lower())
    return {
        "md5": "md5",
        "sha1": "sha1",
        "sha256": "sha256",
    }.get(normalized, "")


def _file_digest(path: str, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def reconcile_local_files(
    candidates: pd.DataFrame,
    local_roots: list[Path],
    *,
    max_hash_bytes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = inventory_local_files(local_roots)
    annotated = candidates.copy()
    result_rows: list[dict[str, object]] = []
    digest_cache: dict[tuple[str, str, int, int], str] = {}

    if local.empty:
        local_by_name_size: dict[tuple[str, int], list[dict[str, object]]] = {}
        local_by_payload: dict[str, list[dict[str, object]]] = {}
        local_records: list[dict[str, object]] = []
    else:
        local_records = local.to_dict("records")
        local_by_name_size = {}
        local_by_payload = {}
        for record in local_records:
            local_by_name_size.setdefault(
                (record["normalized_filename"], int(record["local_filesize"])), []
            ).append(record)
            local_by_payload.setdefault(
                record["normalized_payload_filename"], []
            ).append(record)
    local_parents = sorted(
        {
            (
                record["local_relative_parent"],
                str(Path(record["local_path"]).parent),
            )
            for record in local_records
        }
    )

    for candidate in annotated.to_dict("records"):
        remote_name = normalized_filename(candidate.get("filename"))
        payload_name = normalized_payload_filename(candidate.get("filename"))
        remote_size = int(float(clean(candidate.get("filesize")) or 0))
        exact = local_by_name_size.get((remote_name, remote_size), [])
        algorithm = _hash_algorithm(candidate.get("checksum_type"))
        expected = clean(candidate.get("checksum_value")).lower()
        status = NO_LOCAL_MATCH
        detail = ""
        match_paths: list[str] = []
        observed_checksum = ""

        if exact:
            match_paths = [record["local_path"] for record in exact]
            if algorithm and expected and remote_size <= max_hash_bytes:
                digests = []
                for record in exact:
                    key = (
                        record["local_path"],
                        algorithm,
                        int(record["local_filesize"]),
                        int(record["local_mtime_ns"]),
                    )
                    if key not in digest_cache:
                        digest_cache[key] = _file_digest(record["local_path"], algorithm)
                    digests.append(digest_cache[key])
                observed_checksum = ";".join(sorted(set(digests)))
                if expected in digests:
                    status = LOCAL_EXACT_CHECKSUM
                    detail = f"{algorithm} and byte size match Dataverse metadata"
                    match_paths = [
                        record["local_path"]
                        for record, digest in zip(exact, digests)
                        if digest == expected
                    ]
                else:
                    status = LOCAL_CHECKSUM_MISMATCH
                    detail = "filename and byte size match but checksum differs"
            else:
                status = LOCAL_NAME_SIZE_REVIEW
                detail = (
                    "filename and byte size match; checksum unavailable or exceeds "
                    "the configured hashing limit"
                )
        else:
            payload_matches = local_by_payload.get(payload_name, [])
            if payload_matches:
                status = LOCAL_DERIVED_REPRESENTATION_REVIEW
                detail = "compression-normalized filename matches local data"
                match_paths = [record["local_path"] for record in payload_matches]
            else:
                directory_matches = [
                    absolute_parent
                    for relative_parent, absolute_parent in local_parents
                    if dataset_directory_match(
                        candidate.get("dataset_name"),
                        relative_parent,
                    )
                ]
                if directory_matches:
                    status = LOCAL_DATASET_DIRECTORY_REVIEW
                    detail = "local directory appears to represent the same dataset"
                    match_paths = sorted(set(directory_matches))

        result_rows.append(
            {
                "local_reconciliation_status": status,
                "local_reconciliation_detail": detail,
                "local_match_paths": ";".join(match_paths),
                "local_match_count": len(match_paths),
                "local_observed_checksum": observed_checksum,
                "local_reuse_verified": status == LOCAL_EXACT_CHECKSUM,
                "local_equivalence_review_required": status in LOCAL_REVIEW_STATUSES,
            }
        )

    result = pd.DataFrame(result_rows, index=annotated.index)
    for column in result.columns:
        annotated[column] = result[column]
    return annotated, local
