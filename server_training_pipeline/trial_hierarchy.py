from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_CANDIDATE_FIELDS = {
    "name",
    "trial_effect_enabled",
    "environment_intercept_enabled",
    "minimum_trial_trait_training_rows",
    "minimum_environment_trait_training_rows",
    "trial_penalty",
    "environment_penalty",
}


def validate_hierarchy_candidate(candidate: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_CANDIDATE_FIELDS.difference(candidate))
    if missing:
        raise ValueError(f"Trial-hierarchy candidate is missing fields: {missing}")
    for field in (
        "minimum_trial_trait_training_rows",
        "minimum_environment_trait_training_rows",
    ):
        if int(candidate[field]) < 0:
            raise ValueError(f"{field} must be non-negative")
    for field in ("trial_penalty", "environment_penalty"):
        value = float(candidate[field])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and non-negative")
    if bool(candidate["trial_effect_enabled"]) != (
        int(candidate["minimum_trial_trait_training_rows"]) > 0
        and float(candidate["trial_penalty"]) > 0
    ):
        raise ValueError("Trial-effect support and penalty disagree with enabled status")
    if bool(candidate["environment_intercept_enabled"]) != (
        int(candidate["minimum_environment_trait_training_rows"]) > 0
        and float(candidate["environment_penalty"]) > 0
    ):
        raise ValueError(
            "Environment-intercept support and penalty disagree with enabled status"
        )


def environment_ids(frame: pd.DataFrame) -> pd.Series:
    column = "environment_id" if "environment_id" in frame else "env_kernel_id"
    if column not in frame:
        raise ValueError("Trial hierarchy requires environment_id or env_kernel_id")
    values = frame[column].fillna("").astype(str).str.strip()
    if values.eq("").any():
        raise ValueError("Trial hierarchy encountered empty environment IDs")
    return values


def trial_ids(frame: pd.DataFrame) -> pd.Series:
    environments = environment_ids(frame)
    fallback = environments.str.split("|", n=1).str[0].str.strip()
    if "trial_name" not in frame:
        return fallback
    preferred = frame["trial_name"].fillna("").astype(str).str.strip()
    values = preferred.where(preferred.ne(""), fallback)
    if values.eq("").any():
        raise ValueError("Trial hierarchy encountered empty trial IDs")
    return values


def fit_hierarchy_support(
    training: pd.DataFrame,
    trait_names: list[str],
    candidate: dict[str, Any],
) -> tuple[dict[str, dict[str, int]], pd.DataFrame]:
    validate_hierarchy_candidate(candidate)
    required = {"trait_name_canonical"}
    missing = sorted(required.difference(training.columns))
    if missing:
        raise ValueError(f"Hierarchy training identifiers are missing: {missing}")
    if training.empty:
        raise ValueError("Hierarchy support cannot be fitted from an empty partition")
    local = training.reset_index(drop=True).copy()
    local["_trial_id"] = trial_ids(local)
    local["_environment_id"] = environment_ids(local)
    local["_trait"] = (
        local["trait_name_canonical"].fillna("").astype(str).str.strip()
    )
    expected_traits = set(trait_names)
    if not set(local["_trait"]).issubset(expected_traits) or local["_trait"].eq("").any():
        raise ValueError("Hierarchy support contains unknown or empty traits")
    trial_per_environment = local.groupby("_environment_id")["_trial_id"].nunique()
    if trial_per_environment.gt(1).any():
        raise ValueError("A certified environment maps to multiple trial identities")

    maps: dict[str, dict[str, int]] = {"trial": {}, "environment": {}}
    rows: list[dict[str, object]] = []
    specifications = (
        (
            "trial",
            "_trial_id",
            bool(candidate["trial_effect_enabled"]),
            int(candidate["minimum_trial_trait_training_rows"]),
        ),
        (
            "environment",
            "_environment_id",
            bool(candidate["environment_intercept_enabled"]),
            int(candidate["minimum_environment_trait_training_rows"]),
        ),
    )
    for axis, column, enabled, minimum in specifications:
        counts = (
            local.groupby([column, "_trait"], sort=True)
            .size()
            .rename("training_rows")
            .reset_index()
        )
        retained_ids = sorted(
            set(counts.loc[counts["training_rows"].ge(minimum), column])
            if enabled
            else set()
        )
        maps[axis] = {value: index for index, value in enumerate(retained_ids)}
        for entity in sorted(set(local[column])):
            by_trait = counts[counts[column].eq(entity)].set_index("_trait")[
                "training_rows"
            ]
            for trait in trait_names:
                count = int(by_trait.get(trait, 0))
                rows.append(
                    {
                        "axis": axis,
                        "entity_id": entity,
                        "trait_name_canonical": trait,
                        "training_rows": count,
                        "minimum_training_rows": minimum,
                        "candidate_enabled": enabled,
                        "entity_retained": entity in maps[axis],
                        "entity_trait_supported": enabled and count >= minimum,
                    }
                )
        if enabled and not maps[axis]:
            raise ValueError(f"No {axis} IDs meet the frozen hierarchy support floor")
    return maps, pd.DataFrame(rows)


def hierarchy_indices(
    frame: pd.DataFrame, maps: dict[str, dict[str, int]]
) -> tuple[np.ndarray, np.ndarray]:
    trial = trial_ids(frame).map(maps["trial"]).fillna(-1).to_numpy(dtype=np.int32)
    environment = (
        environment_ids(frame)
        .map(maps["environment"])
        .fillna(-1)
        .to_numpy(dtype=np.int32)
    )
    return trial, environment


def support_matrix(
    support: pd.DataFrame,
    axis: str,
    mapping: dict[str, int],
    trait_names: list[str],
) -> np.ndarray:
    matrix = np.zeros((len(mapping), len(trait_names)), dtype=bool)
    if not mapping:
        return matrix
    trait_index = {value: index for index, value in enumerate(trait_names)}
    selected = support[
        support["axis"].eq(axis) & support["entity_trait_supported"].astype(bool)
    ]
    for row in selected.itertuples(index=False):
        entity = str(row.entity_id)
        trait = str(row.trait_name_canonical)
        if entity in mapping and trait in trait_index:
            matrix[mapping[entity], trait_index[trait]] = True
    return matrix
