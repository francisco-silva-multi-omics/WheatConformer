from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_POLICY_FIELDS = {
    "name",
    "environment_count_exponent",
    "genotype_count_exponent",
    "minimum_relative_weight",
    "maximum_relative_weight",
}


def validate_loss_balance_policy(policy: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_POLICY_FIELDS.difference(policy))
    if missing:
        raise ValueError(f"Loss-balance policy is missing fields: {missing}")
    for field in ("environment_count_exponent", "genotype_count_exponent"):
        value = float(policy[field])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and non-negative")
    lower = policy["minimum_relative_weight"]
    upper = policy["maximum_relative_weight"]
    if lower is not None and (not math.isfinite(float(lower)) or float(lower) <= 0):
        raise ValueError("minimum_relative_weight must be positive or null")
    if upper is not None and (not math.isfinite(float(upper)) or float(upper) <= 0):
        raise ValueError("maximum_relative_weight must be positive or null")
    if lower is not None and upper is not None and float(lower) > float(upper):
        raise ValueError("minimum_relative_weight exceeds maximum_relative_weight")


def fold_local_balanced_loss_weights(
    frame: pd.DataFrame,
    policy: dict[str, Any],
    *,
    trait_column: str = "trait_name_canonical",
    environment_column: str = "environment_id",
    genotype_column: str = "genotype_id",
    precision_column: str = "weight_g_e",
) -> np.ndarray:
    validate_loss_balance_policy(policy)
    required = {
        trait_column,
        environment_column,
        genotype_column,
        precision_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Loss-balance input is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Loss-balance input is empty")

    local = frame.reset_index(drop=True)
    traits = local[trait_column].fillna("").astype(str).str.strip()
    environments = local[environment_column].fillna("").astype(str).str.strip()
    genotypes = local[genotype_column].fillna("").astype(str).str.strip()
    if traits.eq("").any() or environments.eq("").any() or genotypes.eq("").any():
        raise ValueError("Loss-balance identifiers must be non-empty")
    precision = pd.to_numeric(local[precision_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(precision).all() or np.any(precision <= 0):
        raise ValueError("Loss-balance precision weights must be positive and finite")

    environment_counts = (
        pd.DataFrame({"trait": traits, "environment": environments})
        .groupby(["trait", "environment"])["environment"]
        .transform("size")
        .to_numpy(dtype=np.float64)
    )
    genotype_counts = (
        pd.DataFrame({"trait": traits, "genotype": genotypes})
        .groupby(["trait", "genotype"])["genotype"]
        .transform("size")
        .to_numpy(dtype=np.float64)
    )
    alpha_environment = float(policy["environment_count_exponent"])
    alpha_genotype = float(policy["genotype_count_exponent"])
    raw = precision.copy()
    if alpha_environment:
        raw /= np.power(environment_counts, alpha_environment)
    if alpha_genotype:
        raw /= np.power(genotype_counts, alpha_genotype)
    if not np.isfinite(raw).all() or np.any(raw <= 0):
        raise ValueError("Loss-balance transformation produced invalid weights")

    output = np.zeros(len(local), dtype=np.float64)
    unique_traits = sorted(set(traits))
    target_trait_mass = len(local) / len(unique_traits)
    for trait in unique_traits:
        positions = np.flatnonzero(traits.eq(trait).to_numpy())
        relative = raw[positions] / float(np.mean(raw[positions]))
        lower = policy["minimum_relative_weight"]
        upper = policy["maximum_relative_weight"]
        if lower is not None or upper is not None:
            relative = np.clip(
                relative,
                -np.inf if lower is None else float(lower),
                np.inf if upper is None else float(upper),
            )
        output[positions] = relative * target_trait_mass / float(relative.sum())
    if not np.isfinite(output).all() or np.any(output <= 0):
        raise ValueError("Normalized loss-balance weights are invalid")
    return output.astype(np.float32)


def loss_weight_diagnostics(
    frame: pd.DataFrame,
    weights: np.ndarray,
    *,
    policy_name: str,
) -> pd.DataFrame:
    local = frame.reset_index(drop=True).copy()
    values = np.asarray(weights, dtype=np.float64)
    if len(values) != len(local) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Diagnostic weights do not align with the training frame")
    local["loss_weight"] = values
    rows: list[dict[str, object]] = []
    for trait, group in local.groupby("trait_name_canonical", sort=True):
        total = float(group["loss_weight"].sum())
        normalized = group["loss_weight"].to_numpy(dtype=np.float64) / total
        entity_shares: dict[str, float] = {}
        entity_counts: dict[str, int] = {}
        for label, column in (
            ("environment", "environment_id"),
            ("genotype", "genotype_id"),
        ):
            shares = group.groupby(column, dropna=False)["loss_weight"].sum() / total
            entity_shares[label] = float(shares.max())
            entity_counts[label] = int(len(shares))
        top_count = max(1, int(math.ceil(0.01 * len(group))))
        rows.append(
            {
                "loss_balance_policy": policy_name,
                "trait_name_canonical": trait,
                "training_rows": int(len(group)),
                "unique_training_environments": entity_counts["environment"],
                "unique_training_genotypes": entity_counts["genotype"],
                "loss_weight_sum": total,
                "effective_observation_count": float(1.0 / np.square(normalized).sum()),
                "top_1pct_row_weight_share": float(
                    np.sort(normalized)[-top_count:].sum()
                ),
                "maximum_environment_weight_share": entity_shares["environment"],
                "maximum_genotype_weight_share": entity_shares["genotype"],
                "loss_weight_min": float(group["loss_weight"].min()),
                "loss_weight_median": float(group["loss_weight"].median()),
                "loss_weight_max": float(group["loss_weight"].max()),
            }
        )
    return pd.DataFrame(rows)
