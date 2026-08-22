"""Strict, provenance-preserving identifier semantics for the Phase-3G audit.

This module is diagnostic-only.  It is deliberately not imported by the frozen
Phase-3 pipeline and therefore cannot change the delivered Stage-1 v2 data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable


AUTHORITATIVE_GID_CONTEXTS = frozenset(
    {
        "authoritative_gid_column",
        "canonical_gid_column",
        "documented_gid_row",
        "explicit_crosswalk_gid_target",
        "glis_other_gid_field",
        "frozen_gid_order",
    }
)

OPAQUE_SAMPLE_CONTEXTS = frozenset(
    {
        "panel_sample_id",
        "sample_id",
        "taxa",
        "accession_id",
        "row_name",
        "line_number",
        "marker_matrix_label",
    }
)

_PREFIXED_GID = re.compile(r"(?i)^GID\s*:?\s*0*([0-9]+)$")
_PLAIN_INTEGER = re.compile(r"^0*([0-9]+)$")
_EXCEL_INTEGER = re.compile(r"^0*([0-9]+)\.0+$")
_SID = re.compile(r"(?i)^SID\s*:?\s*(.+)$")


@dataclass(frozen=True)
class IdentifierDecision:
    raw_value: str
    identifier_context: str
    identifier_type: str
    canonical_gid_candidate: str
    decision: str
    normalization_rule: str
    coercion_recorded: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def parse_identifier(
    value: object,
    *,
    context: str,
    excel_derived: bool = False,
) -> IdentifierDecision:
    """Classify *value* without crossing identifier namespaces.

    Numeric content is accepted as a GID only in an explicitly authoritative
    GID context.  Even a literal ``GID`` prefix remains opaque when the source
    schema says that the field is merely a panel sample label.
    """

    raw = _text(value)
    if not raw:
        return IdentifierDecision(
            raw, context, "MISSING", "", "NO_VALUE", "NONE", False
        )

    if context in OPAQUE_SAMPLE_CONTEXTS:
        decision = "UNTYPED_NUMERIC_IDENTIFIER" if _PLAIN_INTEGER.fullmatch(raw) else "OPAQUE_SAMPLE_IDENTIFIER"
        return IdentifierDecision(
            raw,
            context,
            "PANEL_SAMPLE_ID",
            "",
            decision,
            "PRESERVE_EXACT_OPAQUE_STRING",
            False,
        )

    sid_match = _SID.fullmatch(raw)
    if context == "sid" or sid_match:
        return IdentifierDecision(
            raw,
            context,
            "SID",
            "",
            "SID_REQUIRES_EXPLICIT_CROSSWALK",
            "PRESERVE_TYPED_SID",
            False,
        )

    if context == "doi":
        return IdentifierDecision(
            raw,
            context,
            "DOI",
            "",
            "DOI_REQUIRES_OFFICIAL_OTHER_GID_EVIDENCE",
            "PRESERVE_EXACT_DOI",
            False,
        )

    if context not in AUTHORITATIVE_GID_CONTEXTS:
        return IdentifierDecision(
            raw,
            context,
            "UNTYPED_IDENTIFIER",
            "",
            "NO_TYPED_GID_AUTHORITY",
            "PRESERVE_EXACT_UNTYPED_STRING",
            False,
        )

    match = _PREFIXED_GID.fullmatch(raw)
    if match:
        return IdentifierDecision(
            raw,
            context,
            "GID",
            f"GID{int(match.group(1))}",
            "TYPED_CANONICAL_GID_CANDIDATE",
            "PARSE_DOCUMENTED_GID_PREFIX_GRAMMAR",
            False,
        )

    match = _PLAIN_INTEGER.fullmatch(raw)
    if match:
        return IdentifierDecision(
            raw,
            context,
            "GID",
            f"GID{int(match.group(1))}",
            "TYPED_CANONICAL_GID_CANDIDATE",
            "PARSE_PLAIN_INTEGER_IN_AUTHORITATIVE_GID_FIELD",
            False,
        )

    match = _EXCEL_INTEGER.fullmatch(raw)
    if match and excel_derived:
        return IdentifierDecision(
            raw,
            context,
            "GID",
            f"GID{int(match.group(1))}",
            "TYPED_CANONICAL_GID_CANDIDATE",
            "SAFE_EXCEL_INTEGER_REPAIR_IN_AUTHORITATIVE_GID_FIELD",
            True,
        )

    return IdentifierDecision(
        raw,
        context,
        "GID_FIELD_INVALID_VALUE",
        "",
        "INVALID_AUTHORITATIVE_GID_VALUE",
        "NO_COERCION",
        False,
    )


def panel_sample_key(panel_id: object, raw_sample_id: object) -> str:
    panel = _text(panel_id)
    sample = _text(raw_sample_id)
    if not panel or not sample:
        raise ValueError("panel_id and raw_sample_id must both be non-empty")
    return f"{panel}::{sample}"


def resolve_gid_candidates(candidates: Iterable[str]) -> tuple[str, str, str]:
    """Resolve explicit candidates without inventing a tie-break rule."""

    values = sorted({str(value).strip() for value in candidates if str(value).strip()})
    if not values:
        return "", "NO_CANONICAL_MATCH", ""
    if len(values) == 1:
        return values[0], "ACCEPTED_AUTHORITATIVE_CROSSWALK", values[0]
    return "", "CONFLICTING_EVIDENCE", ";".join(values)
