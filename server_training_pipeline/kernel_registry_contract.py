from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


TRUE_VALUES = {"1", "true", "yes", "y"}


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUE_VALUES


def csv_set(value: str | Iterable[str] | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = value
    return {str(item).strip() for item in values if str(item).strip()}


def eligible_trait_set(value: object) -> set[str] | None:
    text = str(value).strip()
    if not text or text == "*":
        return None
    return {item.strip().upper() for item in text.split(",") if item.strip()}


def active_kernel_names(
    registry: pd.DataFrame,
    *,
    include_disabled: str | Iterable[str] | None = None,
    exclude: str | Iterable[str] | None = None,
    retained_traits: Iterable[str] | None = None,
) -> set[str]:
    required = {"kernel", "enabled_default", "eligible_traits"}
    missing = sorted(required.difference(registry.columns))
    if missing:
        raise ValueError(f"Kernel registry is missing columns: {missing}")
    names = registry["kernel"].fillna("").astype(str).str.strip()
    if names.eq("").any() or names.duplicated().any():
        raise ValueError("Kernel registry contains empty or duplicate kernel names")

    enabled = registry["enabled_default"].map(parse_bool)
    included = csv_set(include_disabled)
    excluded = csv_set(exclude)
    if included:
        enabled |= names.isin(included)
    if excluded:
        enabled &= ~names.isin(excluded)

    active = registry[enabled].copy()
    if retained_traits is not None:
        retained = {str(value).strip().upper() for value in retained_traits if str(value).strip()}
        active = active[
            active["eligible_traits"].map(
                lambda value: eligible_trait_set(value) is None
                or bool(eligible_trait_set(value) & retained)
            )
        ]
    return set(active["kernel"].astype(str))


def require_active_kernel_contract(
    active: Iterable[str],
    *,
    required: str | Iterable[str] | None = None,
    forbidden: str | Iterable[str] | None = None,
) -> None:
    active_set = {str(value).strip() for value in active if str(value).strip()}
    missing = sorted(csv_set(required) - active_set)
    unexpected = sorted(csv_set(forbidden) & active_set)
    if missing or unexpected:
        raise ValueError(
            "Active-kernel contract failed: "
            f"missing_required={missing}; active_forbidden={unexpected}; "
            f"active={sorted(active_set)}"
        )


def content_identity(value: dict[str, Any]) -> dict[str, object]:
    identity = {key: value.get(key) for key in ["bytes", "sha256"]}
    if identity["bytes"] is None or not identity["sha256"]:
        raise ValueError(f"Incomplete certified content identity: {identity}")
    return identity


def training_input_identities(
    certification: dict[str, Any], active: Iterable[str]
) -> dict[str, object]:
    if certification.get("status") != "PASS":
        raise ValueError("Kernel certification must be PASS before recording input identities")
    active_names = sorted({str(value).strip() for value in active if str(value).strip()})

    def selected(section: str, *, require_all: bool) -> dict[str, dict[str, object]]:
        values = certification.get(section, {})
        missing = sorted(set(active_names) - set(values))
        if require_all and missing:
            raise ValueError(f"Certification section {section} is missing active kernels: {missing}")
        return {
            name: content_identity(values[name])
            for name in active_names
            if name in values
        }

    return {
        "ledger": content_identity(certification.get("ledger_identity", {})),
        "registry": content_identity(certification.get("registry_identity", {})),
        "kernels": selected("kernel_identities", require_all=True),
        "orders": selected("order_identities", require_all=True),
        "coverage": selected("coverage_identities", require_all=False),
    }
