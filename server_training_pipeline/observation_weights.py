from __future__ import annotations

import numpy as np
import pandas as pd


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create robust, trait-normalized precision weights and per-trait QC."""
    for name, value in {
        "floor_quantile": floor_quantile,
        "missing_variance_quantile": missing_variance_quantile,
        "clip_quantile": clip_quantile,
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
        variance = raw_variance.loc[index].to_numpy(dtype=np.float64)
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
        mean_weight = float(np.mean(weights))
        if not np.isfinite(mean_weight) or mean_weight <= 0:
            raise ValueError(f"Trait {trait!r} produced invalid stabilized weights")
        weights = weights / mean_weight

        adjusted_variance.loc[index] = working_variance
        stabilized.loc[index] = weights
        out.loc[index, "weight_variance_imputed"] = missing_mask
        out.loc[index, "weight_variance_floored"] = floor_mask & ~missing_mask

        sorted_weights = np.sort(weights)[::-1]
        top_n = max(1, int(np.ceil(len(sorted_weights) * 0.01)))
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
                "normalized_weight_mean": float(np.mean(weights)),
                "normalized_weight_median": float(np.median(weights)),
                "normalized_weight_p99": float(np.quantile(weights, 0.99)),
                "normalized_weight_max": float(np.max(weights)),
                "effective_sample_size": ess,
                "effective_sample_fraction": ess / len(weights),
                "top_1pct_weight_share": float(np.sum(sorted_weights[:top_n]) / np.sum(sorted_weights)),
            }
        )

    out["stabilized_var_g_e"] = adjusted_variance
    out[output_weight_col] = stabilized
    return out, pd.DataFrame(qc_rows)
