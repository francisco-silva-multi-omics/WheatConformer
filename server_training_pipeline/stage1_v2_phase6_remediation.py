from __future__ import annotations

import hashlib
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


def _weighted_calibration_coefficients(
    prediction: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    contract: dict[str, Any],
) -> tuple[float, float]:
    weight_sum = float(weight.sum())
    if weight_sum <= 0:
        return 0.0, 1.0
    x_mean = float(np.sum(weight * prediction) / weight_sum)
    y_mean = float(np.sum(weight * target) / weight_sum)
    covariance = float(
        np.sum(weight * (prediction - x_mean) * (target - y_mean)) / weight_sum
    )
    variance = float(
        np.sum(weight * np.square(prediction - x_mean)) / weight_sum
    )
    raw_slope = covariance / (variance + float(contract["weighted_ridge"]))
    slope = float(
        np.clip(
            raw_slope,
            float(contract["minimum_slope"]),
            float(contract["maximum_slope"]),
        )
    )
    return y_mean - slope * x_mean, slope


def _group_fold(value: str, fold_count: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % fold_count


def fit_group_crossfitted_trait_calibration(
    training: pd.DataFrame,
    prediction: np.ndarray,
    active: np.ndarray,
    trait_names: Sequence[str],
    protocol: dict[str, Any],
    *,
    target_trait: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a robust training-only calibrator across held-out identifier groups.

    The predictive model is unchanged. Calibration coefficients are fitted on
    four grouped partitions at a time, audited on the fifth, and aggregated by
    the preregistered componentwise median. Inner-validation values never enter
    this procedure.
    """

    evidence_columns = [
        "trait_name_canonical",
        "crossfit_fold",
        "fit_rows",
        "heldout_rows",
        "fit_group_count",
        "heldout_group_count",
        "intercept",
        "slope",
        "heldout_calibration_slope",
        "heldout_scaled_rmse",
        "status",
        "validation_values_used",
    ]
    calibration = fit_positive_calibration(
        training, prediction, active, trait_names, protocol
    )
    calibration["method"] = "standard_full_training"
    calibration["crossfit_valid_folds"] = 0
    contract = protocol["test_weight_group_crossfit"]
    if target_trait not in trait_names:
        raise ValueError(f"Cross-fit target trait is absent: {target_trait}")
    trait_index = list(trait_names).index(target_trait)
    trait_values = training["trait_index"].to_numpy(dtype=int)
    weights = training["loss_weight"].to_numpy(dtype=float)
    target_mask = active & (trait_values == trait_index) & (weights > 0)
    local = training.loc[target_mask].copy()
    local_prediction = np.asarray(prediction, dtype=float)[target_mask]
    if len(local) < int(contract["minimum_fit_rows_per_fold"]):
        calibration.loc[
            calibration["trait_name_canonical"].eq(target_trait),
            ["intercept", "slope", "status", "method"],
        ] = [0.0, 1.0, "IDENTITY_CROSSFIT_INSUFFICIENT_SUPPORT", "identity"]
        return calibration, pd.DataFrame(columns=evidence_columns)

    trial = clean_identifier(local["trial_id"])
    environment = clean_identifier(local["environment_id"])
    row_id = clean_identifier(local["phase4_adjusted_row_id"])
    group = pd.Series(index=local.index, dtype="string")
    group.loc[trial.ne("")] = "TRIAL:" + trial.loc[trial.ne("")]
    missing = group.isna()
    group.loc[missing & environment.ne("")] = (
        "ENV:" + environment.loc[missing & environment.ne("")]
    )
    missing = group.isna()
    group.loc[missing] = "ROW:" + row_id.loc[missing]
    fold_count = int(contract["fold_count"])
    folds = group.map(lambda value: _group_fold(str(value), fold_count)).to_numpy()
    target = local["y_scaled"].to_numpy(dtype=float)
    local_weight = local["loss_weight"].to_numpy(dtype=float)
    evidence: list[dict[str, object]] = []
    coefficients: list[tuple[float, float]] = []
    positive = protocol["positive_training_calibration"]
    for fold in range(fold_count):
        heldout = folds == fold
        fit = ~heldout
        fit_rows = int(fit.sum())
        heldout_rows = int(heldout.sum())
        valid = (
            fit_rows >= int(contract["minimum_fit_rows_per_fold"])
            and heldout_rows >= int(contract["minimum_heldout_rows_per_fold"])
        )
        intercept = 0.0
        slope = 1.0
        heldout_slope = np.nan
        heldout_rmse = np.nan
        if valid:
            intercept, slope = _weighted_calibration_coefficients(
                local_prediction[fit], target[fit], local_weight[fit], positive
            )
            calibrated = intercept + slope * local_prediction[heldout]
            _, heldout_slope = _weighted_calibration_coefficients(
                calibrated,
                target[heldout],
                local_weight[heldout],
                {**positive, "weighted_ridge": 0.0},
            )
            heldout_rmse = float(
                np.sqrt(
                    np.average(
                        np.square(target[heldout] - calibrated),
                        weights=local_weight[heldout],
                    )
                )
            )
            coefficients.append((intercept, slope))
        evidence.append(
            {
                "trait_name_canonical": target_trait,
                "crossfit_fold": fold,
                "fit_rows": fit_rows,
                "heldout_rows": heldout_rows,
                "fit_group_count": int(pd.Series(group.to_numpy()[fit]).nunique()),
                "heldout_group_count": int(
                    pd.Series(group.to_numpy()[heldout]).nunique()
                ),
                "intercept": intercept,
                "slope": slope,
                "heldout_calibration_slope": heldout_slope,
                "heldout_scaled_rmse": heldout_rmse,
                "status": "PASS" if valid else "INSUFFICIENT_FOLD_SUPPORT",
                "validation_values_used": False,
            }
        )
    target_row = calibration["trait_name_canonical"].eq(target_trait)
    if len(coefficients) == fold_count:
        intercept = float(np.median([value[0] for value in coefficients]))
        slope = float(np.median([value[1] for value in coefficients]))
        calibration.loc[
            target_row,
            [
                "intercept",
                "slope",
                "status",
                "method",
                "crossfit_valid_folds",
            ],
        ] = [
            intercept,
            slope,
            "FITTED_GROUP_CROSSFIT_MEDIAN",
            "group_crossfit_median",
            fold_count,
        ]
    else:
        calibration.loc[
            target_row,
            [
                "intercept",
                "slope",
                "status",
                "method",
                "crossfit_valid_folds",
            ],
        ] = [
            0.0,
            1.0,
            "IDENTITY_CROSSFIT_INCOMPLETE",
            "identity",
            len(coefficients),
        ]
    return calibration, pd.DataFrame(evidence, columns=evidence_columns)


def set_trait_identity_calibration(
    calibration: pd.DataFrame, trait_name: str
) -> pd.DataFrame:
    result = calibration.copy()
    if "method" not in result:
        result["method"] = "standard_full_training"
    if "crossfit_valid_folds" not in result:
        result["crossfit_valid_folds"] = 0
    target = result["trait_name_canonical"].eq(trait_name)
    if int(target.sum()) != 1:
        raise ValueError(f"Identity calibration target is not unique: {trait_name}")
    result.loc[
        target,
        ["intercept", "slope", "status", "method", "crossfit_valid_folds"],
    ] = [0.0, 1.0, "IDENTITY_PREREGISTERED", "identity", 0]
    return result
