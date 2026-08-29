from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np
import pandas as pd


EVIDENCE_COLUMNS = [
    "trait_name_canonical",
    "calibration_method",
    "crossfit_fold",
    "fit_rows",
    "heldout_rows",
    "fit_environment_count",
    "heldout_environment_count",
    "excluded_missing_environment_rows",
    "intercept",
    "slope",
    "heldout_calibration_slope",
    "heldout_scaled_rmse",
    "huber_iterations",
    "status",
    "validation_values_used",
]


def _environment_fold(value: str, fold_count: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % fold_count


def _weighted_affine(
    prediction: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    contract: dict[str, Any],
    *,
    robust: bool,
) -> tuple[float, float, int]:
    x = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    base_weight = np.asarray(weight, dtype=np.float64)
    if len(x) == 0 or float(base_weight.sum()) <= 0:
        return 0.0, 1.0, 0

    ridge = float(contract["slope_ridge_toward_one"])
    minimum_slope = float(contract["minimum_slope"])
    maximum_slope = float(contract["maximum_slope"])

    def solve(local_weight: np.ndarray) -> tuple[float, float]:
        weight_sum = float(local_weight.sum())
        x_mean = float(np.sum(local_weight * x) / weight_sum)
        y_mean = float(np.sum(local_weight * y) / weight_sum)
        covariance = float(
            np.sum(local_weight * (x - x_mean) * (y - y_mean)) / weight_sum
        )
        variance = float(
            np.sum(local_weight * np.square(x - x_mean)) / weight_sum
        )
        # The penalty is centered on slope 1, not slope 0.
        slope = (covariance + ridge) / (variance + ridge)
        slope = float(np.clip(slope, minimum_slope, maximum_slope))
        return y_mean - slope * x_mean, slope

    intercept, slope = solve(base_weight)
    if not robust:
        return intercept, slope, 0

    delta = float(contract["huber_delta"])
    maximum_iterations = int(contract["huber_maximum_iterations"])
    tolerance = float(contract["huber_convergence_tolerance"])
    completed = 0
    for iteration in range(1, maximum_iterations + 1):
        residual = y - (intercept + slope * x)
        residual_center = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - residual_center)))
        if not np.isfinite(scale) or scale <= 1e-12:
            completed = iteration
            break
        threshold = delta * scale
        robust_weight = np.ones(len(residual), dtype=np.float64)
        large = np.abs(residual) > threshold
        robust_weight[large] = threshold / np.abs(residual[large])
        next_intercept, next_slope = solve(base_weight * robust_weight)
        completed = iteration
        if max(abs(next_intercept - intercept), abs(next_slope - slope)) <= tolerance:
            intercept, slope = next_intercept, next_slope
            break
        intercept, slope = next_intercept, next_slope
    return intercept, slope, completed


def _heldout_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    intercept: float,
    slope: float,
) -> tuple[float, float]:
    calibrated = intercept + slope * prediction
    weight_sum = float(weight.sum())
    rmse = float(np.sqrt(np.sum(weight * np.square(calibrated - target)) / weight_sum))
    calibrated_mean = float(np.sum(weight * calibrated) / weight_sum)
    target_mean = float(np.sum(weight * target) / weight_sum)
    covariance = float(
        np.sum(weight * (calibrated - calibrated_mean) * (target - target_mean))
        / weight_sum
    )
    variance = float(
        np.sum(weight * np.square(calibrated - calibrated_mean)) / weight_sum
    )
    calibration_slope = covariance / variance if variance > 0 else np.nan
    return float(calibration_slope), rmse


def fit_test_weight_calibration(
    training: pd.DataFrame,
    prediction: np.ndarray,
    active: np.ndarray,
    trait_names: Sequence[str],
    base_calibration: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    candidate: str,
    target_trait: str = "TEST_WEIGHT",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive one preregistered TEST_WEIGHT calibrator from training rows only."""

    calibration = base_calibration.copy()
    calibration["method"] = "standard_full_training"
    calibration["crossfit_valid_folds"] = 0
    candidate_contract = protocol["calibration_candidates"][candidate]
    method = str(candidate_contract["method"])
    target_mask = calibration["trait_name_canonical"].eq(target_trait)
    if int(target_mask.sum()) != 1:
        raise ValueError(f"Calibration table does not contain exactly one {target_trait} row")

    if method == "identity":
        calibration.loc[
            target_mask,
            ["intercept", "slope", "status", "method", "crossfit_valid_folds"],
        ] = [0.0, 1.0, "IDENTITY_PREREGISTERED_REFERENCE", method, 0]
        evidence = pd.DataFrame(
            [
                {
                    "trait_name_canonical": target_trait,
                    "calibration_method": method,
                    "crossfit_fold": -1,
                    "fit_rows": 0,
                    "heldout_rows": 0,
                    "fit_environment_count": 0,
                    "heldout_environment_count": 0,
                    "excluded_missing_environment_rows": 0,
                    "intercept": 0.0,
                    "slope": 1.0,
                    "heldout_calibration_slope": np.nan,
                    "heldout_scaled_rmse": np.nan,
                    "huber_iterations": 0,
                    "status": "IDENTITY_REFERENCE",
                    "validation_values_used": False,
                }
            ],
            columns=EVIDENCE_COLUMNS,
        )
        return calibration, evidence

    if method not in {"environment_oof_affine_ridge", "environment_oof_huber"}:
        raise ValueError(f"Unknown calibration method: {method}")
    if target_trait not in trait_names:
        raise ValueError(f"Calibration target trait is absent: {target_trait}")

    contract = protocol["test_weight_environment_oof_calibration"]
    trait_index = list(trait_names).index(target_trait)
    trait_values = training["trait_index"].to_numpy(dtype=int)
    weights = training["loss_weight"].to_numpy(dtype=float)
    selected = np.asarray(active, dtype=bool) & (trait_values == trait_index) & (weights > 0)
    local = training.loc[selected].copy()
    local_prediction = np.asarray(prediction, dtype=np.float64)[selected]
    environment = local["environment_id"].astype("string").fillna("").str.strip()
    valid_environment = environment.ne("")
    excluded_missing = int((~valid_environment).sum())
    local = local.loc[valid_environment].reset_index(drop=True)
    local_prediction = local_prediction[valid_environment.to_numpy()]
    environment = environment.loc[valid_environment].reset_index(drop=True)

    fold_count = int(contract["fold_count"])
    if (
        len(local) < int(contract["minimum_total_rows"])
        or environment.nunique() < int(contract["minimum_total_environments"])
    ):
        calibration.loc[
            target_mask,
            ["intercept", "slope", "status", "method", "crossfit_valid_folds"],
        ] = [0.0, 1.0, "IDENTITY_ENVIRONMENT_OOF_INSUFFICIENT_SUPPORT", method, 0]
        return calibration, pd.DataFrame(columns=EVIDENCE_COLUMNS)

    folds = environment.map(lambda value: _environment_fold(str(value), fold_count)).to_numpy()
    target = local["y_scaled"].to_numpy(dtype=np.float64)
    local_weight = local["loss_weight"].to_numpy(dtype=np.float64)
    robust = method == "environment_oof_huber"
    evidence_rows: list[dict[str, object]] = []
    coefficients: list[tuple[float, float]] = []
    for fold in range(fold_count):
        heldout = folds == fold
        fit = ~heldout
        fit_rows = int(fit.sum())
        heldout_rows = int(heldout.sum())
        fit_environments = int(environment.loc[fit].nunique())
        heldout_environments = int(environment.loc[heldout].nunique())
        valid = (
            fit_rows >= int(contract["minimum_fit_rows_per_fold"])
            and heldout_rows >= int(contract["minimum_heldout_rows_per_fold"])
            and fit_environments >= int(contract["minimum_fit_environments_per_fold"])
            and heldout_environments
            >= int(contract["minimum_heldout_environments_per_fold"])
        )
        intercept = 0.0
        slope = 1.0
        heldout_slope = np.nan
        heldout_rmse = np.nan
        iterations = 0
        status = "INSUFFICIENT_FOLD_SUPPORT"
        if valid:
            intercept, slope, iterations = _weighted_affine(
                local_prediction[fit],
                target[fit],
                local_weight[fit],
                contract,
                robust=robust,
            )
            heldout_slope, heldout_rmse = _heldout_metrics(
                local_prediction[heldout],
                target[heldout],
                local_weight[heldout],
                intercept,
                slope,
            )
            coefficients.append((intercept, slope))
            status = "PASS"
        evidence_rows.append(
            {
                "trait_name_canonical": target_trait,
                "calibration_method": method,
                "crossfit_fold": fold,
                "fit_rows": fit_rows,
                "heldout_rows": heldout_rows,
                "fit_environment_count": fit_environments,
                "heldout_environment_count": heldout_environments,
                "excluded_missing_environment_rows": excluded_missing,
                "intercept": intercept,
                "slope": slope,
                "heldout_calibration_slope": heldout_slope,
                "heldout_scaled_rmse": heldout_rmse,
                "huber_iterations": iterations,
                "status": status,
                "validation_values_used": False,
            }
        )

    if len(coefficients) != fold_count:
        calibration.loc[
            target_mask,
            ["intercept", "slope", "status", "method", "crossfit_valid_folds"],
        ] = [0.0, 1.0, "IDENTITY_ENVIRONMENT_OOF_INCOMPLETE", method, len(coefficients)]
    else:
        intercept = float(np.median([value[0] for value in coefficients]))
        slope = float(np.median([value[1] for value in coefficients]))
        calibration.loc[
            target_mask,
            ["intercept", "slope", "status", "method", "crossfit_valid_folds"],
        ] = [
            intercept,
            slope,
            "FITTED_ENVIRONMENT_OOF_COMPONENTWISE_MEDIAN",
            method,
            len(coefficients),
        ]
    return calibration, pd.DataFrame(evidence_rows, columns=EVIDENCE_COLUMNS)


def non_target_calibration_signature(
    calibration: pd.DataFrame, target_trait: str = "TEST_WEIGHT"
) -> str:
    local = calibration.loc[
        ~calibration["trait_name_canonical"].eq(target_trait),
        ["trait_name_canonical", "training_rows", "intercept", "slope", "status"],
    ].sort_values("trait_name_canonical")
    payload = local.to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
