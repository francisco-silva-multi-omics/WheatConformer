from __future__ import annotations

import re

from server_genotype_recovery.fetch_brapi_pedigree_markers import clean


WHEAT_CONFIRMED = "WHEAT_CONFIRMED"
NON_WHEAT_EXCLUDED = "NON_WHEAT_EXCLUDED"
AMBIGUOUS_REVIEW = "AMBIGUOUS_REVIEW"

WHEAT_PATTERNS = {
    "wheat": r"\bwheat\b",
    "triticum": r"\btriticum\b",
    "IBWSN": r"(?<![a-z])ibwsn(?![a-z])",
    "SAWSN": r"(?<![a-z])sawsn(?![a-z])",
    "SAWYT": r"(?<![a-z])sawyt(?![a-z])",
    "ESWYT": r"(?<![a-z])eswyt(?![a-z])",
    "HRWYT": r"(?<![a-z])hrwyt(?![a-z])",
    "WYCYT": r"(?<![a-z])wycyt(?![a-z])",
    "IDYN": r"(?<![a-z])idyn(?![a-z])",
    "IDSN": r"(?<![a-z])idsn(?![a-z])",
    "ISEPTON": r"(?<![a-z])isepton(?![a-z])",
    "KBSN": r"(?<![a-z])kbsn(?![a-z])",
    "FHBSN": r"(?<![a-z])fhbsn(?![a-z])",
    "HTWYT": r"(?<![a-z])htwyt(?![a-z])",
    "IWYP": r"\biwyp\b",
    "HiBAP": r"\bhibap\b",
}

NON_WHEAT_PATTERNS = {
    "maize": r"\bmaize\b",
    "zea": r"\bzea(?:\s+mays)?\b",
    "CML": r"\bcml(?:\d+|[_ -]|\b)",
    "rice": r"\brice\b|\boryza\b",
    "barley": r"\bbarley\b|\bhordeum\b",
    "groundnut": r"\bgroundnut\b|\barachis\b",
    "sorghum": r"\bsorghum\b",
}


def _hits(text: str, patterns: dict[str, str]) -> list[str]:
    return [label for label, pattern in patterns.items() if re.search(pattern, text)]


def classify_crop_scope(
    dataset_name: object,
    filename: object = "",
    description: object = "",
) -> tuple[str, str]:
    title = clean(dataset_name).lower()
    file_text = f"{clean(filename)} {clean(description)}".lower()
    title = f"{title} {re.sub(r'[^a-z0-9]+', ' ', title)}"
    file_text = f"{file_text} {re.sub(r'[^a-z0-9]+', ' ', file_text)}"
    title_non_wheat = _hits(title, NON_WHEAT_PATTERNS)
    title_wheat = _hits(title, WHEAT_PATTERNS)
    file_non_wheat = _hits(file_text, NON_WHEAT_PATTERNS)
    file_wheat = _hits(file_text, WHEAT_PATTERNS)

    non_wheat = title_non_wheat or file_non_wheat
    wheat = title_wheat or file_wheat
    evidence = ";".join(
        [
            *(f"title_non_wheat={value}" for value in title_non_wheat),
            *(f"file_non_wheat={value}" for value in file_non_wheat),
            *(f"title_wheat={value}" for value in title_wheat),
            *(f"file_wheat={value}" for value in file_wheat),
        ]
    )
    if non_wheat:
        return NON_WHEAT_EXCLUDED, evidence
    if wheat:
        return WHEAT_CONFIRMED, evidence
    return AMBIGUOUS_REVIEW, "no_explicit_crop_evidence"
