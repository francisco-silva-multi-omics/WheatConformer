from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def clean_identifier(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").fillna("").str.strip()
    return cleaned.mask(cleaned.str.lower().isin({"", "nan", "none", "<na>"}), "")


def add_hierarchy_indices(
    frame: pd.DataFrame,
    trait_names: Sequence[str],
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    local = frame.copy()
    local["_trial_key"] = clean_identifier(local["trial_id"])
    local["_environment_key"] = clean_identifier(local["environment_id"])
    training = local.loc[
        local["selection_role"].eq("TRAINING") & local["loss_weight"].gt(0)
    ].copy()
    contract = protocol["trial_environment_hierarchy"]
    rows: list[dict[str, object]] = []

    def build(
        column: str, minimum: int
    ) -> tuple[dict[str, int], np.ndarray, list[dict[str, object]]]:
        counts = (
            training.loc[training[column].ne("")]
            .groupby([column, "trait"], as_index=False)
            .size()
        )
        supported = counts.loc[counts["size"].ge(minimum)].copy()
        entities = sorted(supported[column].astype(str).unique())
        lookup = {value: index for index, value in enumerate(entities)}
        trait_lookup = {value: index for index, value in enumerate(trait_names)}
        matrix = np.zeros((len(entities), len(trait_names)), dtype=bool)
        support_rows: list[dict[str, object]] = []
        for _, row in counts.iterrows():
            entity = str(row[column])
            trait = str(row["trait"])
            count = int(row["size"])
            eligible = entity in lookup and count >= minimum
            if eligible:
                matrix[lookup[entity], trait_lookup[trait]] = True
            support_rows.append(
                {
                    "level": column.removeprefix("_"),
                    "entity_id": entity,
                    "trait_name_canonical": trait,
                    "positive_weight_training_rows": count,
                    "minimum_rows": minimum,
                    "eligible": eligible,
                }
            )
        return lookup, matrix, support_rows

    trial_lookup, trial_support, trial_rows = build(
        "_trial_key", int(contract["minimum_positive_weight_trial_trait_rows"])
    )
    environment_lookup, environment_support, environment_rows = build(
        "_environment_key",
        int(contract["minimum_positive_weight_environment_trait_rows"]),
    )
    rows.extend(trial_rows)
    rows.extend(environment_rows)
    local["trial_hierarchy_index"] = (
        local["_trial_key"].map(trial_lookup).fillna(-1).astype(np.int32)
    )
    local["environment_hierarchy_index"] = (
        local["_environment_key"].map(environment_lookup).fillna(-1).astype(np.int32)
    )
    local = local.drop(columns=["_trial_key", "_environment_key"])
    return local, trial_support, environment_support, pd.DataFrame(rows)


def fit_positive_calibration(
    training: pd.DataFrame,
    prediction: np.ndarray,
    active: np.ndarray,
    trait_names: Sequence[str],
    protocol: dict[str, Any],
) -> pd.DataFrame:
    contract = protocol["positive_training_calibration"]
    rows = []
    trait_index_values = training["trait_index"].to_numpy(dtype=int)
    loss_weight_values = training["loss_weight"].to_numpy(dtype=float)
    for trait_index, trait in enumerate(trait_names):
        mask = active & (trait_index_values == trait_index) & (loss_weight_values > 0)
        count = int(mask.sum())
        slope = 1.0
        intercept = 0.0
        status = "IDENTITY_INSUFFICIENT_SUPPORT"
        if count >= int(contract["minimum_rows_per_trait"]):
            x = prediction[mask].astype(float)
            y = training.loc[mask, "y_scaled"].to_numpy(dtype=float)
            weight = training.loc[mask, "loss_weight"].to_numpy(dtype=float)
            weight_sum = float(weight.sum())
            x_mean = float(np.sum(weight * x) / weight_sum)
            y_mean = float(np.sum(weight * y) / weight_sum)
            covariance = float(
                np.sum(weight * (x - x_mean) * (y - y_mean)) / weight_sum
            )
            variance = float(np.sum(weight * np.square(x - x_mean)) / weight_sum)
            raw_slope = covariance / (
                variance + float(contract["weighted_ridge"])
            )
            slope = float(
                np.clip(
                    raw_slope,
                    float(contract["minimum_slope"]),
                    float(contract["maximum_slope"]),
                )
            )
            intercept = y_mean - slope * x_mean
            status = "FITTED_POSITIVE_SLOPE"
        rows.append(
            {
                "trait_name_canonical": trait,
                "training_rows": count,
                "intercept": intercept,
                "slope": slope,
                "status": status,
                "validation_values_used": False,
            }
        )
    return pd.DataFrame(rows)


def apply_calibration(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    active: np.ndarray,
    calibration: pd.DataFrame,
) -> np.ndarray:
    result = prediction.copy()
    lookup = calibration.set_index("trait_name_canonical")
    for trait, positions in frame.groupby("trait", sort=False).groups.items():
        index = np.asarray(list(positions), dtype=np.int64)
        index = index[active[index]]
        if not len(index):
            continue
        row = lookup.loc[trait]
        result[index] = float(row["intercept"]) + float(row["slope"]) * result[index]
    return result
