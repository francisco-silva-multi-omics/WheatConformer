from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


REFERENCE = "current_huber_authoritative_row_mass"


def apply_trait_mass_policy(
    frame: pd.DataFrame,
    *,
    candidate: str,
    candidate_policy: dict[str, Any],
    primary_traits: Sequence[str],
    exploratory_traits: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"selection_role", "trait", "loss_weight"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Trait-mass frame is missing columns: {sorted(missing)}")
    local = frame.copy()
    training = local["selection_role"].eq("TRAINING")
    positive = training & local["loss_weight"].gt(0)
    original = local["loss_weight"].to_numpy(dtype=np.float64, copy=True)
    traits = [*primary_traits, *exploratory_traits]
    if set(local.loc[training, "trait"].unique()).difference(traits):
        raise ValueError("Trait-mass policy encountered an unregistered training trait")
    policy = str(candidate_policy["policy"])
    if candidate == REFERENCE:
        if policy != "unchanged_authoritative_row_mass":
            raise ValueError("Reference trait-mass policy is not unchanged")
        target = {
            trait: float(local.loc[positive & local["trait"].eq(trait), "loss_weight"].sum())
            for trait in traits
        }
    elif policy == "fixed_total_mass_by_trait":
        primary_mass = float(candidate_policy["primary_trait_mass"])
        exploratory_mass = float(candidate_policy["exploratory_trait_mass"])
        if primary_mass <= 0 or exploratory_mass <= 0:
            raise ValueError("Trait target masses must be positive")
        target = {
            trait: primary_mass if trait in primary_traits else exploratory_mass
            for trait in traits
        }
    else:
        raise ValueError(f"Unknown trait-mass policy: {policy}")

    original_total = float(local.loc[positive, "loss_weight"].sum())
    if original_total <= 0:
        raise ValueError("Trait-mass policy has no positive training weight")
    target_sum = float(sum(target.values()))
    diagnostics: list[dict[str, object]] = []
    for trait in traits:
        mask = positive & local["trait"].eq(trait)
        base_mass = float(local.loc[mask, "loss_weight"].sum())
        if base_mass <= 0:
            raise ValueError(f"Trait-mass policy lacks positive support: {trait}")
        desired_mass = (
            base_mass
            if candidate == REFERENCE
            else original_total * float(target[trait]) / target_sum
        )
        multiplier = desired_mass / base_mass
        local.loc[mask, "loss_weight"] = (
            local.loc[mask, "loss_weight"].to_numpy(dtype=np.float64) * multiplier
        )
        final_mass = float(local.loc[mask, "loss_weight"].sum())
        diagnostics.append(
            {
                "candidate": candidate,
                "trait_name_canonical": trait,
                "training_rows": int((training & local["trait"].eq(trait)).sum()),
                "positive_weight_training_rows": int(mask.sum()),
                "base_loss_weight_mass": base_mass,
                "target_relative_mass": float(target[trait]),
                "loss_weight_multiplier": multiplier,
                "final_loss_weight_mass": final_mass,
                "final_loss_weight_share": final_mass / original_total,
            }
        )
    if not np.array_equal(
        original[~training.to_numpy()],
        local.loc[~training, "loss_weight"].to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise ValueError("Trait-mass policy changed non-training loss weights")
    if not np.allclose(
        local.loc[training & frame["loss_weight"].eq(0), "loss_weight"], 0.0
    ):
        raise ValueError("Trait-mass policy activated zero authoritative weights")
    final_total = float(local.loc[positive, "loss_weight"].sum())
    if not np.isclose(final_total, original_total, rtol=1e-12, atol=1e-8):
        raise ValueError(
            f"Trait-mass policy changed total weight: {final_total} != {original_total}"
        )
    return local, pd.DataFrame(diagnostics)
