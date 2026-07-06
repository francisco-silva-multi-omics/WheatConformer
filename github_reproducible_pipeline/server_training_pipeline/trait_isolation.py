from __future__ import annotations

import re

import pandas as pd


MULTITRAIT_ERROR = "Multiple phenotype traits detected. Run one trait per model using --trait. Mixed-trait training is disabled."


def clean_trait_values(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def select_single_trait(observations: pd.DataFrame, requested_traits: list[str] | None) -> tuple[pd.DataFrame, str]:
    if "trait_name_canonical" not in observations.columns:
        raise SystemExit("Observation table is missing trait_name_canonical")

    traits = clean_trait_values(observations["trait_name_canonical"])
    available = sorted({trait for trait in traits if trait})
    requested = [str(trait).strip() for trait in (requested_traits or []) if str(trait).strip()]
    requested_norm = {trait.upper() for trait in requested}

    if len(requested_norm) > 1:
        raise SystemExit("One model invocation supports exactly one --trait value.")
    if not requested_norm:
        if len(available) > 1:
            raise SystemExit(MULTITRAIT_ERROR)
        if not available:
            raise SystemExit("No non-empty phenotype trait is available for training.")
        selected = available[0]
    else:
        selected_norm = next(iter(requested_norm))
        matches = [trait for trait in available if trait.upper() == selected_norm]
        if not matches:
            raise SystemExit(f"Requested trait has zero rows: {requested[0]}")
        selected = matches[0]

    filtered = observations[traits.str.upper().eq(selected.upper())].copy()
    if filtered.empty:
        raise SystemExit(f"Requested trait has zero rows: {selected}")
    return filtered, selected


def sanitize_trait_name(trait: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(trait).strip())
    slug = re.sub(r"_+", "_", slug).strip("_.")
    return slug or "trait"
