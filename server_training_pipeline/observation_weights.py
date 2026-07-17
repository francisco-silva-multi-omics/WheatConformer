from __future__ import annotations

import numpy as np
import pandas as pd


def _owned_float_array(values: pd.Series) -> np.ndarray:
    """Materialize a writable array from NumPy- or Arrow-backed pandas data."""
    return values.to_numpy(dtype=np.float64, copy=True)


def _quantile(values: np.ndarray, q: float, fallback: float) -> float:
    if values.size == 0:
        return float(fallback)
    value = float(np.quantile(values, q))
    return value if np.isfinite(value) and value > 0 else float(fallback)


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights[np.isfinite(weights) & (weights > 0)]
    if weights.size == 0:
        return 0.0
    denominator = float(np.sum(np.square(weights)))
    return float(np.square(np.sum(weights)) / denominator) if denominator > 0 else 0.0


def top_weight_share(weights: np.ndarray, fraction: float = 0.01) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights[np.isfinite(weights) & (weights > 0)]
    if weights.size == 0:
        return 0.0
    top_n = max(1, int(np.ceil(weights.size * fraction)))
    return float(np.sort(weights)[-top_n:].sum() / weights.sum())


def _normalized_power_weights(weights: np.ndarray, power: float) -> np.ndarray:
    powered = np.power(np.asarray(weights, dtype=np.float64), power)
    mean_weight = float(np.mean(powered))
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("Weight tempering produced an invalid mean")
    return powered / mean_weight


def _temper_power_to_constraints(
    weights: np.ndarray,
    *,
    requested_power: float,
    min_effective_sample_fraction: float,
    max_top_1pct_share: float,
) -> tuple[np.ndarray, float]:
    """Use the largest requested power that satisfies concentration limits."""

    def valid(candidate: np.ndarray) -> bool:
        ess_fraction = effective_sample_size(candidate) / len(candidate)
        return (
            ess_fraction + 1e-12 >= min_effective_sample_fraction
            and top_weight_share(candidate) <= max_top_1pct_share + 1e-12
        )

    requested = _normalized_power_weights(weights, requested_power)
    if valid(requested):
        return requested, requested_power

    uniform = _normalized_power_weights(weights, 0.0)
    if not valid(uniform):
        raise ValueError(
            "Weight concentration constraints are impossible for this trait size: "
            f"minimum ESS fraction={min_effective_sample_fraction}, "
            f"maximum top-1% share={max_top_1pct_share}"
        )

    low = 0.0
    high = requested_power
    for _ in range(60):
        midpoint = (low + high) / 2.0
        candidate = _normalized_power_weights(weights, midpoint)
        if valid(candidate):
            low = midpoint
        else:
            high = midpoint
    return _normalized_power_weights(weights, low), low


def stabilize_precision_weights(
    frame: pd.DataFrame,
    *,
    trait_col: str = "trait_name_canonical",
    variance_col: str = "var_g_e",
    source_weight_col: str = "weight_g_e",
    output_weight_col: str = "weight_g_e",
    floor_quantile: float = 0.01,
    missing_variance_quantile: float = 0.75,
    clip_quantile: float = 0.99,
    weight_power: float = 1.0,
    min_effective_sample_fraction: float = 0.0,
    max_top_1pct_share: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create robust, trait-normalized precision weights and per-trait QC."""
    for name, value in {
        "floor_quantile": floor_quantile,
        "missing_variance_quantile": missing_variance_quantile,
        "clip_quantile": clip_quantile,
        "weight_power": weight_power,
        "min_effective_sample_fraction": min_effective_sample_fraction,
        "max_top_1pct_share": max_top_1pct_share,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]; found {value}")
    missing = [c for c in [trait_col, variance_col] if c not in frame.columns]
    if missing:
        raise ValueError(f"Weight stabilization is missing required columns: {missing}")

    out = frame.copy()
    raw_variance = pd.to_numeric(out[variance_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if source_weight_col in out.columns:
        source_weight = pd.to_numeric(out[source_weight_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    else:
        source_weight = pd.Series(np.nan, index=out.index, dtype=float)
    inverse_variance = pd.Series(np.nan, index=out.index, dtype=float)
    positive_variance = raw_variance.notna() & raw_variance.gt(0)
    inverse_variance.loc[positive_variance] = 1.0 / raw_variance.loc[positive_variance]
    out["raw_var_g_e"] = raw_variance
    out["raw_weight_g_e"] = inverse_variance
    out["source_weight_g_e"] = source_weight
    out["weight_variance_imputed"] = False
    out["weight_variance_floored"] = False

    stabilized = pd.Series(np.nan, index=out.index, dtype=float)
    adjusted_variance = pd.Series(np.nan, index=out.index, dtype=float)
    qc_rows: list[dict[str, float | int | str]] = []
    trait_values = out[trait_col].fillna("").astype(str).str.strip()

    for trait, index in trait_values.groupby(trait_values, sort=True).groups.items():
        index = pd.Index(index)
        variance = _owned_float_array(raw_variance.loc[index])
        positive = variance[np.isfinite(variance) & (variance > 0)]
        fallback = float(np.median(positive)) if positive.size else 1.0
        variance_floor = _quantile(positive, floor_quantile, fallback)
        missing_fill = _quantile(positive, missing_variance_quantile, fallback)

        missing_mask = ~np.isfinite(variance) | (variance <= 0)
        working_variance = variance.copy()
        working_variance[missing_mask] = missing_fill
        floor_mask = working_variance < variance_floor
        working_variance = np.maximum(working_variance, variance_floor)
        weights = 1.0 / working_variance

        weight_cap = _quantile(weights, clip_quantile, float(np.max(weights)))
        weights = np.minimum(weights, weight_cap)
        pre_temper_weights = _normalized_power_weights(weights, weight_power)
        pre_temper_ess_fraction = effective_sample_size(pre_temper_weights) / len(pre_temper_weights)
        pre_temper_top_share = top_weight_share(pre_temper_weights)
        weights, effective_power = _temper_power_to_constraints(
            weights,
            requested_power=weight_power,
            min_effective_sample_fraction=min_effective_sample_fraction,
            max_top_1pct_share=max_top_1pct_share,
        )

        adjusted_variance.loc[index] = working_variance
        stabilized.loc[index] = weights
        out.loc[index, "weight_variance_imputed"] = missing_mask
        out.loc[index, "weight_variance_floored"] = floor_mask & ~missing_mask

        ess = effective_sample_size(weights)
        qc_rows.append(
            {
                "trait_name_canonical": trait,
                "rows": int(len(index)),
                "finite_positive_raw_variance_rows": int(positive.size),
                "imputed_variance_rows": int(np.sum(missing_mask)),
                "floored_variance_rows": int(np.sum(floor_mask & ~missing_mask)),
                "variance_floor": variance_floor,
                "missing_variance_fill": missing_fill,
                "weight_cap_before_normalization": weight_cap,
                "requested_weight_power": weight_power,
                "effective_weight_power": effective_power,
                "weight_power_tempered": bool(effective_power < weight_power - 1e-10),
                "pre_temper_effective_sample_fraction": pre_temper_ess_fraction,
                "pre_temper_top_1pct_weight_share": pre_temper_top_share,
                "normalized_weight_mean": float(np.mean(weights)),
                "normalized_weight_median": float(np.median(weights)),
                "normalized_weight_p99": float(np.quantile(weights, 0.99)),
                "normalized_weight_max": float(np.max(weights)),
                "effective_sample_size": ess,
                "effective_sample_fraction": ess / len(weights),
                "top_1pct_weight_share": top_weight_share(weights),
            }
        )

    out["stabilized_var_g_e"] = adjusted_variance
    out[output_weight_col] = stabilized
    return out, pd.DataFrame(qc_rows)


def fit_precision_weight_transform(
    frame: pd.DataFrame,
    *,
    trait_col: str = "trait_name_canonical",
    variance_col: str = "var_g_e",
    floor_quantile: float = 0.01,
    missing_variance_quantile: float = 0.75,
    clip_quantile: float = 0.99,
    weight_power: float = 1.0,
    min_effective_sample_fraction: float = 0.0,
    max_top_1pct_share: float = 1.0,
) -> pd.DataFrame:
    """Fit precision-weight parameters using training observations only."""
    if trait_col not in frame or variance_col not in frame:
        raise ValueError(f"Weight fitting requires {trait_col} and {variance_col}")
    rows: list[dict[str, float | int | str]] = []
    traits = frame[trait_col].fillna("").astype(str).str.strip()
    variance_all = pd.to_numeric(frame[variance_col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    for trait, index in traits.groupby(traits, sort=True).groups.items():
        variance = _owned_float_array(variance_all.loc[index])
        positive = variance[np.isfinite(variance) & (variance > 0)]
        fallback = float(np.median(positive)) if positive.size else 1.0
        variance_floor = _quantile(positive, floor_quantile, fallback)
        missing_fill = _quantile(positive, missing_variance_quantile, fallback)
        working = variance.copy()
        missing = ~np.isfinite(working) | (working <= 0)
        working[missing] = missing_fill
        working = np.maximum(working, variance_floor)
        raw_weight = 1.0 / working
        weight_cap = _quantile(raw_weight, clip_quantile, float(np.max(raw_weight)))
        capped = np.minimum(raw_weight, weight_cap)
        _, effective_power = _temper_power_to_constraints(
            capped,
            requested_power=weight_power,
            min_effective_sample_fraction=min_effective_sample_fraction,
            max_top_1pct_share=max_top_1pct_share,
        )
        powered = np.power(capped, effective_power)
        normalization_mean = float(np.mean(powered))
        normalized = powered / normalization_mean
        rows.append(
            {
                "trait_name_canonical": trait,
                "training_rows": int(len(index)),
                "finite_positive_training_variance_rows": int(positive.size),
                "variance_floor": variance_floor,
                "missing_variance_fill": missing_fill,
                "weight_cap_before_power": weight_cap,
                "requested_weight_power": weight_power,
                "effective_weight_power": effective_power,
                "training_normalization_mean": normalization_mean,
                "training_effective_sample_fraction": effective_sample_size(normalized)
                / len(normalized),
                "training_top_1pct_weight_share": top_weight_share(normalized),
            }
        )
    return pd.DataFrame(rows)


def apply_precision_weight_transform(
    frame: pd.DataFrame,
    parameters: pd.DataFrame,
    *,
    trait_col: str = "trait_name_canonical",
    variance_col: str = "var_g_e",
    output_weight_col: str = "weight_g_e",
) -> pd.DataFrame:
    """Apply a train-fitted precision-weight transformation to any partition."""
    required = {
        "trait_name_canonical",
        "variance_floor",
        "missing_variance_fill",
        "weight_cap_before_power",
        "effective_weight_power",
        "training_normalization_mean",
    }
    missing = sorted(required.difference(parameters.columns))
    if missing:
        raise ValueError(f"Weight-transform parameters are missing columns: {missing}")
    if parameters["trait_name_canonical"].duplicated().any():
        raise ValueError("Weight-transform parameters contain duplicate traits")
    lookup = parameters.set_index("trait_name_canonical").to_dict("index")
    out = frame.copy()
    variance = pd.to_numeric(out[variance_col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    result = pd.Series(np.nan, index=out.index, dtype=np.float64)
    imputed = pd.Series(False, index=out.index)
    floored = pd.Series(False, index=out.index)
    traits = out[trait_col].fillna("").astype(str).str.strip()
    for trait, index in traits.groupby(traits, sort=True).groups.items():
        if trait not in lookup:
            raise ValueError(f"No train-fitted weight parameters for trait {trait!r}")
        params = lookup[trait]
        local = _owned_float_array(variance.loc[index])
        missing_mask = ~np.isfinite(local) | (local <= 0)
        local[missing_mask] = float(params["missing_variance_fill"])
        floor_mask = local < float(params["variance_floor"])
        local = np.maximum(local, float(params["variance_floor"]))
        weights = np.minimum(
            1.0 / local, float(params["weight_cap_before_power"])
        )
        weights = np.power(weights, float(params["effective_weight_power"]))
        weights /= float(params["training_normalization_mean"])
        result.loc[index] = weights
        imputed.loc[index] = missing_mask
        floored.loc[index] = floor_mask & ~missing_mask
    if not np.isfinite(result.to_numpy(dtype=float)).all() or (result <= 0).any():
        raise ValueError("Fold-local weight application produced invalid weights")
    out[output_weight_col] = result
    out["weight_variance_imputed"] = imputed
    out["weight_variance_floored"] = floored
    return out
