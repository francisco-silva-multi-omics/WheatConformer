from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_CANDIDATE_FIELDS = {
    "name",
    "environment_mean_loss_weight",
    "genotype_deviation_loss_weight",
    "ranking_loss_weight",
    "ranking_temperature",
    "minimum_standardized_pair_gap",
    "uncertainty_gap_multiplier",
    "maximum_pairs_per_environment_trait",
}


def validate_candidate(candidate: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_CANDIDATE_FIELDS.difference(candidate))
    if missing:
        raise ValueError(f"Within-environment candidate is missing fields: {missing}")
    for field in (
        "environment_mean_loss_weight",
        "genotype_deviation_loss_weight",
        "ranking_loss_weight",
        "minimum_standardized_pair_gap",
        "uncertainty_gap_multiplier",
    ):
        value = float(candidate[field])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and non-negative")
    if float(candidate["ranking_temperature"]) <= 0:
        raise ValueError("ranking_temperature must be positive")
    if int(candidate["maximum_pairs_per_environment_trait"]) < 1:
        raise ValueError("maximum_pairs_per_environment_trait must be positive")


def _weighted_group_sums(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    local = frame[keys + ["y_scaled", "weight_g_e"]].copy()
    local["_wy"] = local["y_scaled"].to_numpy(float) * local["weight_g_e"].to_numpy(float)
    return (
        local.groupby(keys, sort=False, dropna=False)
        .agg(_weight_sum=("weight_g_e", "sum"), _weighted_y_sum=("_wy", "sum"))
        .reset_index()
    )


def leave_one_genotype_out_targets(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trait_name_canonical",
        "environment_id",
        "genotype_id",
        "y_scaled",
        "weight_g_e",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Environment-deviation target input is missing: {missing}")
    local = frame.reset_index(drop=True).copy()
    if local.empty:
        return pd.DataFrame(
            columns=[
                "environment_mean_target",
                "genotype_deviation_target",
                "decomposition_weight",
            ]
        )
    numeric = local[["y_scaled", "weight_g_e"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all() or numeric["weight_g_e"].le(0).any():
        raise ValueError("Environment-deviation targets require finite values and positive weights")

    environment_keys = ["trait_name_canonical", "environment_id"]
    genotype_keys = [*environment_keys, "genotype_id"]
    environment = _weighted_group_sums(local, environment_keys).rename(
        columns={
            "_weight_sum": "_environment_weight",
            "_weighted_y_sum": "_environment_weighted_y",
        }
    )
    genotype = _weighted_group_sums(local, genotype_keys).rename(
        columns={
            "_weight_sum": "_genotype_weight",
            "_weighted_y_sum": "_genotype_weighted_y",
        }
    )
    joined = local[genotype_keys + ["y_scaled", "weight_g_e"]].merge(
        environment, on=environment_keys, how="left", validate="many_to_one"
    ).merge(genotype, on=genotype_keys, how="left", validate="many_to_one")
    remaining_weight = joined["_environment_weight"] - joined["_genotype_weight"]
    remaining_sum = joined["_environment_weighted_y"] - joined["_genotype_weighted_y"]
    eligible = remaining_weight.gt(0) & np.isfinite(remaining_sum)
    mean_target = np.zeros(len(joined), dtype=np.float32)
    mean_target[eligible] = (
        remaining_sum[eligible].to_numpy(float) / remaining_weight[eligible].to_numpy(float)
    ).astype(np.float32)
    deviation_target = joined["y_scaled"].to_numpy(np.float32) - mean_target
    decomposition_weight = np.where(
        eligible,
        joined["weight_g_e"].to_numpy(float),
        0.0,
    ).astype(np.float32)
    return pd.DataFrame(
        {
            "environment_mean_target": mean_target,
            "genotype_deviation_target": deviation_target,
            "decomposition_weight": decomposition_weight,
        }
    )


def _trait_scale(frame: pd.DataFrame) -> pd.Series:
    scales: dict[str, float] = {}
    for trait, group in frame.groupby("trait_name_canonical", sort=False):
        raw = pd.to_numeric(group["phenotype_value"], errors="coerce").to_numpy(float)
        standardized = pd.to_numeric(group["y_scaled"], errors="coerce").to_numpy(float)
        finite = np.isfinite(raw) & np.isfinite(standardized)
        if finite.sum() < 2 or np.std(standardized[finite]) <= 0:
            scales[str(trait)] = 1.0
            continue
        slope = np.cov(raw[finite], standardized[finite], ddof=0)[0, 1] / np.var(
            standardized[finite]
        )
        scales[str(trait)] = float(abs(slope)) if np.isfinite(slope) and slope != 0 else 1.0
    return frame["trait_name_canonical"].astype(str).map(scales).astype(float)


def deterministic_pair_assignments(
    frame: pd.DataFrame,
    candidate: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_candidate(candidate)
    required = {
        "trait_name_canonical",
        "environment_id",
        "genotype_id",
        "phenotype_value",
        "y_scaled",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Pair assignment input is missing: {missing}")
    local = frame.reset_index(drop=True).copy()
    partner = np.arange(len(local), dtype=np.int32)
    direction = np.zeros(len(local), dtype=np.float32)
    pair_weight = np.zeros(len(local), dtype=np.float32)
    scale = _trait_scale(local).to_numpy(float)
    variance = (
        pd.to_numeric(local["var_g_e"], errors="coerce").to_numpy(float)
        if "var_g_e" in local
        else np.full(len(local), np.nan)
    )
    standardized_variance = variance / np.square(np.maximum(scale, 1e-12))
    y = pd.to_numeric(local["y_scaled"], errors="coerce").to_numpy(float)
    maximum_pairs = int(candidate["maximum_pairs_per_environment_trait"])
    minimum_gap = float(candidate["minimum_standardized_pair_gap"])
    uncertainty_multiplier = float(candidate["uncertainty_gap_multiplier"])
    diagnostics: list[dict[str, object]] = []

    for (trait, environment), positions in local.groupby(
        ["trait_name_canonical", "environment_id"], sort=True, dropna=False
    ).groups.items():
        indices = np.asarray(list(positions), dtype=np.int32)
        genotype = local.loc[indices, "genotype_id"].fillna("").astype(str).to_numpy()
        if pd.Series(genotype).nunique() < 2:
            diagnostics.append(
                {
                    "trait_name_canonical": trait,
                    "environment_id": environment,
                    "rows": len(indices),
                    "unique_genotypes": int(pd.Series(genotype).nunique()),
                    "selected_pairs": 0,
                    "maximum_pairs": maximum_pairs,
                    "attempts": 0,
                    "minimum_standardized_pair_gap": minimum_gap,
                    "uncertainty_gap_multiplier": uncertainty_multiplier,
                    "median_pair_standard_error": float("nan"),
                }
            )
            continue
        label = f"{seed}|{trait}|{environment}"
        local_seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little")
        rng = np.random.default_rng(local_seed)
        primary_order = rng.permutation(indices)
        local_position = {int(value): position for position, value in enumerate(indices)}
        selected: list[tuple[int, int, float, float]] = []
        used: set[tuple[int, int]] = set()
        used_primary: set[int] = set()
        attempts = 0
        maximum_attempts = max(100, min(len(indices), maximum_pairs) * 40)
        while (
            len(selected) < min(len(indices), maximum_pairs)
            and attempts < maximum_attempts
            and len(indices) > 1
        ):
            left = int(primary_order[attempts % len(primary_order)])
            right = int(indices[int(rng.integers(0, len(indices)))])
            attempts += 1
            if left == right:
                continue
            if left in used_primary:
                continue
            left_local = local_position[left]
            right_local = local_position[right]
            if genotype[left_local] == genotype[right_local]:
                continue
            key = tuple(sorted((left, right)))
            if key in used:
                continue
            gap = abs(y[left] - y[right])
            pair_variance = standardized_variance[left] + standardized_variance[right]
            pair_se = math_sqrt_or_nan(pair_variance)
            threshold = minimum_gap
            if np.isfinite(pair_se):
                threshold = max(threshold, uncertainty_multiplier * pair_se)
            if not np.isfinite(gap) or gap <= threshold:
                continue
            selected.append((left, right, float(np.sign(y[left] - y[right])), pair_se))
            used.add(key)
            used_primary.add(left)
        if selected:
            group_weight = 1.0 / len(selected)
            for left, right, sign, _ in selected:
                partner[left] = right
                direction[left] = sign
                pair_weight[left] = group_weight
        diagnostics.append(
            {
                "trait_name_canonical": trait,
                "environment_id": environment,
                "rows": len(indices),
                "unique_genotypes": int(pd.Series(genotype).nunique()),
                "selected_pairs": len(selected),
                "maximum_pairs": maximum_pairs,
                "attempts": attempts,
                "minimum_standardized_pair_gap": minimum_gap,
                "uncertainty_gap_multiplier": uncertainty_multiplier,
                "median_pair_standard_error": float(
                    np.nanmedian([value[3] for value in selected])
                )
                if selected
                else float("nan"),
            }
        )
    selected_groups_by_trait = (
        pd.DataFrame(diagnostics)
        .loc[lambda value: value["selected_pairs"].gt(0)]
        .groupby("trait_name_canonical", sort=False)
        .size()
        .to_dict()
    )
    for trait, group_count in selected_groups_by_trait.items():
        positions = local.index[local["trait_name_canonical"].eq(trait)].to_numpy(int)
        pair_weight[positions] /= float(group_count)
    trait_count = len(selected_groups_by_trait)
    if trait_count:
        pair_weight *= float(len(local)) / float(trait_count)
    assignments = pd.DataFrame(
        {
            "partner_position": partner,
            "pair_direction": direction,
            "pair_weight": pair_weight,
        }
    )
    return assignments, pd.DataFrame(diagnostics)


def math_sqrt_or_nan(value: float) -> float:
    return float(np.sqrt(value)) if np.isfinite(value) and value >= 0 else float("nan")


def objective_artifact_digest(*frames: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return digest.hexdigest()


def candidate_json(candidate: dict[str, Any]) -> str:
    validate_candidate(candidate)
    return json.dumps(candidate, sort_keys=True, separators=(",", ":"))
