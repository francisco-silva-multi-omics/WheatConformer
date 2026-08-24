"""Phase 4 phenotype reconstruction and within-environment signal assessment.

This module is intentionally independent of certified-v1 artifacts and protected
validation outcomes.  It consumes only the frozen Stage-1-v2 canonical layer and
fits within-environment models for the seven predeclared modelling traits.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr


PHASE4_VERSION = "phase4_phenotype_reconstruction_2026_08_01_v1"
SELECTED_TRAITS = (
    "1000_GRAIN_WEIGHT",
    "ABOVE_GROUND_BIOMASS",
    "DAYS_TO_HEADING",
    "DAYS_TO_MATURITY",
    "GRAIN_YIELD",
    "PLANT_HEIGHT",
    "TEST_WEIGHT",
)
GROUP_COLS = (
    "canonical_environment_id",
    "canonical_trial_name",
    "cycle",
    "occ",
    "loc_no",
    "country",
    "loc_desc",
    "accepted_canonical_trait",
    "trait_name_original",
    "standardized_unit",
)
EXPECTED_SELECTED_CONTRIBUTORS = 4_226_848
EXPECTED_SELECTED_STAGE1_ROWS = 3_193_677


def stable_id(prefix: str, *values: object) -> str:
    token = "|".join("" if value is None else str(value) for value in values)
    return prefix + hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def factor_tokens(values: pd.Series) -> np.ndarray:
    tokens = clean(values).replace("", "__MISSING__")
    return tokens.to_numpy(dtype=object)


def safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    return float(spearmanr(x, y).statistic)


def aicc_from_rss(rss: float, n: int, p: int, logdet_correlation: float = 0.0) -> tuple[float, float]:
    if not np.isfinite(rss) or rss <= 0 or n <= p:
        return math.nan, math.nan
    log_likelihood = -0.5 * (
        n * (math.log(2.0 * math.pi) + 1.0 + math.log(rss / n)) + logdet_correlation
    )
    aic = -2.0 * log_likelihood + 2.0 * p
    if n - p - 1 <= 0:
        return aic, math.inf
    return aic, aic + (2.0 * p * (p + 1.0)) / (n - p - 1.0)


@dataclass
class Design:
    matrix: np.ndarray
    column_names: list[str]
    genotype_levels: list[str]
    genotype_column_count: int
    nuisance_column_indices: list[int]
    terms: list[str]
    formula: str


@dataclass
class Fit:
    model_name: str
    status: str
    design: Design
    beta: np.ndarray
    covariance: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    n: int
    p: int
    rank: int
    df_resid: int
    rss: float
    sigma2: float
    aic: float
    aicc: float
    rho: float = math.nan
    adjacent_pairs: int = 0


def _append_treatment_columns(
    arrays: list[np.ndarray], names: list[str], tokens: np.ndarray, prefix: str
) -> None:
    levels = sorted(set(tokens.tolist()))
    for level in levels[1:]:
        arrays.append((tokens == level).astype(float))
        names.append(f"{prefix}[{level}]")


def build_design(group: pd.DataFrame, include_field_design: bool, include_spline: bool) -> Design:
    gids = factor_tokens(group["resolved_gid_v2"])
    genotype_levels = sorted(set(gids.tolist()))
    arrays: list[np.ndarray] = [np.ones(len(group), dtype=float)]
    names = ["Intercept"]
    for gid in genotype_levels[1:]:
        arrays.append((gids == gid).astype(float))
        names.append(f"gid[{gid}]")
    genotype_column_count = len(arrays)
    terms = ["genotype_fixed"]

    rep_tokens = factor_tokens(group["rep"])
    block_tokens = factor_tokens(group["subblock"])
    rep_coverage = float(np.mean(rep_tokens != "__MISSING__"))
    block_coverage = float(np.mean(block_tokens != "__MISSING__"))
    rep_usable = include_field_design and rep_coverage >= 0.8 and len(set(rep_tokens)) >= 2
    block_usable = include_field_design and block_coverage >= 0.8 and len(set(block_tokens)) >= 2
    if rep_usable:
        _append_treatment_columns(arrays, names, rep_tokens, "rep")
        terms.append("rep_fixed")
    if block_usable:
        if rep_usable:
            # Parameterize blocks within replication by omitting the first block
            # in each replication, avoiding the deterministic rep/block alias.
            for rep_level in sorted(set(rep_tokens.tolist())):
                mask = rep_tokens == rep_level
                levels = sorted(set(block_tokens[mask].tolist()))
                for level in levels[1:]:
                    arrays.append((mask & (block_tokens == level)).astype(float))
                    names.append(f"block_within_rep[{rep_level}:{level}]")
            terms.append("block_within_rep_fixed")
        else:
            _append_treatment_columns(arrays, names, block_tokens, "block")
            terms.append("block_fixed")

    if include_spline:
        plot = pd.to_numeric(group["plot"], errors="coerce").to_numpy(dtype=float)
        coverage = float(np.mean(np.isfinite(plot)))
        finite = plot[np.isfinite(plot)]
        if coverage >= 0.8 and len(np.unique(finite)) >= 8:
            fill = float(np.median(finite))
            missing = ~np.isfinite(plot)
            plot = np.where(missing, fill, plot)
            low, high = float(np.min(finite)), float(np.max(finite))
            z = (plot - low) / max(high - low, np.finfo(float).eps)
            basis = [z, z**2, z**3]
            basis.extend(np.maximum(z - knot, 0.0) ** 3 for knot in (0.25, 0.5, 0.75))
            for index, values in enumerate(basis, start=1):
                centered = values - np.mean(values)
                if np.std(centered) > 1e-12:
                    arrays.append(centered)
                    names.append(f"plot_spline_{index}")
            if np.any(missing) and np.any(~missing):
                arrays.append(missing.astype(float) - float(np.mean(missing)))
                names.append("plot_coordinate_missing")
            terms.append("plot_order_cubic_regression_spline")

    matrix = np.column_stack(arrays)
    nuisance = list(range(genotype_column_count, matrix.shape[1]))
    formula = "value_standardized ~ " + " + ".join(terms)
    return Design(matrix, names, genotype_levels, genotype_column_count, nuisance, terms, formula)


def fit_ols(y: np.ndarray, design: Design, model_name: str) -> Fit:
    x = design.matrix
    n, p = x.shape
    try:
        beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        rank = 0
        beta = np.full(p, np.nan)
    if rank != p:
        nan_matrix = np.full((p, p), np.nan)
        return Fit(model_name, "REJECT_RANK_DEFICIENT", design, beta, nan_matrix,
                   np.full(n, np.nan), np.full(n, np.nan), n, p, int(rank), n - int(rank),
                   math.nan, math.nan, math.nan, math.nan)
    fitted = x @ beta
    residual = y - fitted
    rss = float(residual @ residual)
    df = n - p
    sigma2 = rss / df if df > 0 else math.nan
    # Candidate comparison does not require covariance.  It is computed once,
    # after model selection, for the selected design only.
    covariance = np.full((p, p), np.nan)
    aic, aicc = aicc_from_rss(rss, n, p)
    status = "ELIGIBLE" if df >= 3 and np.isfinite(aicc) else "REJECT_INSUFFICIENT_RESIDUAL_DF"
    return Fit(model_name, status, design, beta, covariance, fitted, residual,
               n, p, int(rank), df, rss, sigma2, aic, aicc)


def estimate_ar1_rho(group: pd.DataFrame, residual: np.ndarray) -> tuple[float, int]:
    frame = pd.DataFrame({
        "residual": residual,
        "plot_num": pd.to_numeric(group["plot"], errors="coerce"),
        "rep": clean(group["rep"]),
        "row_order": np.arange(len(group)),
    })
    frame = frame[frame["plot_num"].notna()].copy()
    if len(frame) < 10 or frame["plot_num"].nunique() < 8:
        return math.nan, 0
    frame["sequence"] = frame["rep"].where(frame["rep"].ne(""), "__ALL__")
    previous: list[float] = []
    current: list[float] = []
    for _, part in frame.sort_values(["sequence", "plot_num", "row_order"]).groupby("sequence", sort=True):
        values = part["residual"].to_numpy(dtype=float)
        if len(values) >= 2:
            previous.extend(values[:-1])
            current.extend(values[1:])
    if len(previous) < 8:
        return math.nan, len(previous)
    prev = np.asarray(previous)
    curr = np.asarray(current)
    denominator = float(prev @ prev)
    if denominator <= 0:
        return math.nan, len(previous)
    return float(np.clip((prev @ curr) / denominator, -0.90, 0.90)), len(previous)


def fit_ar1(group: pd.DataFrame, y: np.ndarray, base: Fit) -> Fit:
    rho, adjacent_pairs = estimate_ar1_rho(group, base.residual)
    if not np.isfinite(rho) or adjacent_pairs < 8:
        return Fit("PLOT_ORDER_AR1_GLS", "REJECT_AR1_NOT_IDENTIFIABLE", base.design,
                   base.beta, base.covariance, base.fitted, base.residual, base.n, base.p,
                   base.rank, base.df_resid, base.rss, base.sigma2, base.aic, math.nan,
                   rho, adjacent_pairs)
    x = base.design.matrix
    xw = x.copy()
    yw = y.copy()
    plot_num = pd.to_numeric(group["plot"], errors="coerce").to_numpy(dtype=float)
    reps = factor_tokens(group["rep"])
    transformed = np.zeros(len(group), dtype=bool)
    logdet = 0.0
    for rep_level in sorted(set(reps.tolist())):
        indices = np.flatnonzero((reps == rep_level) & np.isfinite(plot_num))
        if len(indices) < 2:
            continue
        indices = indices[np.argsort(plot_num[indices], kind="mergesort")]
        first = indices[0]
        xw[first] = math.sqrt(1.0 - rho**2) * x[first]
        yw[first] = math.sqrt(1.0 - rho**2) * y[first]
        transformed[first] = True
        for previous, current in zip(indices[:-1], indices[1:], strict=True):
            xw[current] = x[current] - rho * x[previous]
            yw[current] = y[current] - rho * y[previous]
            transformed[current] = True
        logdet += (len(indices) - 1) * math.log(1.0 - rho**2)
    if int(transformed.sum()) < 10:
        return Fit("PLOT_ORDER_AR1_GLS", "REJECT_AR1_NOT_IDENTIFIABLE", base.design,
                   base.beta, base.covariance, base.fitted, base.residual, base.n, base.p,
                   base.rank, base.df_resid, base.rss, base.sigma2, base.aic, math.nan,
                   rho, adjacent_pairs)
    try:
        beta, _, rank, _ = np.linalg.lstsq(xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        rank = 0
        beta = np.full(base.p, np.nan)
    if rank != base.p:
        return Fit("PLOT_ORDER_AR1_GLS", "REJECT_RANK_DEFICIENT", base.design,
                   beta, np.full((base.p, base.p), np.nan), np.full(base.n, np.nan),
                   np.full(base.n, np.nan), base.n, base.p, int(rank), base.n - int(rank),
                   math.nan, math.nan, math.nan, math.nan, rho, adjacent_pairs)
    residual_w = yw - xw @ beta
    rss = float(residual_w @ residual_w)
    df = base.n - base.p
    sigma2 = rss / df if df > 0 else math.nan
    covariance = np.full((base.p, base.p), np.nan)
    fitted = x @ beta
    residual = y - fitted
    aic, aicc = aicc_from_rss(rss, base.n, base.p + 1, logdet)
    status = "ELIGIBLE" if df >= 3 and np.isfinite(aicc) else "REJECT_INSUFFICIENT_RESIDUAL_DF"
    return Fit("PLOT_ORDER_AR1_GLS", status, base.design, beta, covariance, fitted,
               residual, base.n, base.p, int(rank), df, rss, sigma2, aic, aicc,
               rho, adjacent_pairs)


def huber_fit(y: np.ndarray, x: np.ndarray, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.full(x.shape[1], np.nan), np.full(len(y), np.nan), np.full(len(y), np.nan), "FAILED"
    weights = np.ones(len(y))
    for _ in range(max_iter):
        residual = y - x @ beta
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            return beta, residual, weights, "CONVERGED_ZERO_MAD"
        standardized = np.abs(residual) / scale
        new_weights = np.minimum(1.0, 1.345 / np.maximum(standardized, np.finfo(float).eps))
        root = np.sqrt(new_weights)
        try:
            new_beta = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]
        except np.linalg.LinAlgError:
            return beta, residual, weights, "FAILED"
        if float(np.max(np.abs(new_beta - beta))) <= 1e-8 * (1.0 + float(np.max(np.abs(beta)))):
            beta = new_beta
            weights = new_weights
            return beta, y - x @ beta, weights, "CONVERGED"
        beta, weights = new_beta, new_weights
    return beta, y - x @ beta, weights, "MAX_ITER"


def selected_covariance(group: pd.DataFrame, fit: Fit) -> np.ndarray:
    if not np.isfinite(fit.sigma2):
        return np.full((fit.p, fit.p), np.nan)
    x = fit.design.matrix
    if fit.model_name != "PLOT_ORDER_AR1_GLS" or not np.isfinite(fit.rho):
        return np.linalg.pinv(x.T @ x) * fit.sigma2
    plot_num = pd.to_numeric(group["plot"], errors="coerce").to_numpy(dtype=float)
    reps = factor_tokens(group["rep"])
    xw = x.copy()
    for rep_level in sorted(set(reps.tolist())):
        indices = np.flatnonzero((reps == rep_level) & np.isfinite(plot_num))
        if len(indices) < 2:
            continue
        indices = indices[np.argsort(plot_num[indices], kind="mergesort")]
        first = indices[0]
        xw[first] = math.sqrt(1.0 - fit.rho**2) * x[first]
        for previous, current in zip(indices[:-1], indices[1:], strict=True):
            xw[current] = x[current] - fit.rho * x[previous]
    return np.linalg.pinv(xw.T @ xw) * fit.sigma2


def genotype_contrasts(design: Design) -> np.ndarray:
    l = np.zeros((len(design.genotype_levels), design.matrix.shape[1]), dtype=float)
    l[:, 0] = 1.0
    if design.nuisance_column_indices:
        l[:, design.nuisance_column_indices] = np.mean(
            design.matrix[:, design.nuisance_column_indices], axis=0
        )
    for index in range(1, len(design.genotype_levels)):
        l[index, index] = 1.0
    return l


def model_record(group_id: str, fit: Fit, selected: bool) -> dict[str, Any]:
    max_std = math.nan
    if np.isfinite(fit.sigma2) and fit.sigma2 > 0 and np.isfinite(fit.residual).any():
        max_std = float(np.nanmax(np.abs(fit.residual) / math.sqrt(fit.sigma2)))
    return {
        "phase4_group_id": group_id,
        "candidate_model": fit.model_name,
        "candidate_status": fit.status,
        "selected_model": selected,
        "formula": fit.design.formula,
        "terms": ";".join(fit.design.terms),
        "n_observations": fit.n,
        "n_parameters": fit.p,
        "matrix_rank": fit.rank,
        "df_resid": fit.df_resid,
        "rss": fit.rss,
        "sigma2": fit.sigma2,
        "aic": fit.aic,
        "aicc": fit.aicc,
        "ar1_rho": fit.rho,
        "ar1_adjacent_pairs": fit.adjacent_pairs,
        "max_absolute_standardized_residual": max_std,
        "phase4_version": PHASE4_VERSION,
    }


def _rank_ceiling(
    group: pd.DataFrame, adjusted_observation: np.ndarray
) -> dict[str, Any]:
    raw_a: list[float] = []
    raw_b: list[float] = []
    adjusted_a: list[float] = []
    adjusted_b: list[float] = []
    for _, part in group.assign(_adjusted=adjusted_observation).groupby("resolved_gid_v2", sort=True):
        if len(part) < 2:
            continue
        part = part.sort_values(["rep", "subblock", "plot", "raw_source_row_id"], kind="mergesort")
        left = part.iloc[::2]
        right = part.iloc[1::2]
        if left.empty or right.empty:
            continue
        raw_a.append(float(left["value_standardized"].mean()))
        raw_b.append(float(right["value_standardized"].mean()))
        adjusted_a.append(float(left["_adjusted"].mean()))
        adjusted_b.append(float(right["_adjusted"].mean()))
    n = len(raw_a)
    raw_r = safe_correlation(np.asarray(raw_a), np.asarray(raw_b), "spearman")
    adjusted_r = safe_correlation(np.asarray(adjusted_a), np.asarray(adjusted_b), "spearman")

    def corrected(value: float) -> float:
        if not np.isfinite(value):
            return math.nan
        return float(np.clip((2.0 * value) / (1.0 + value), 0.0, 1.0)) if value > -1.0 else 0.0

    return {
        "n_entries_split": n,
        "raw_split_half_spearman": raw_r,
        "raw_spearman_brown_ceiling": corrected(raw_r),
        "adjusted_split_half_spearman": adjusted_r,
        "adjusted_spearman_brown_ceiling": corrected(adjusted_r),
        "ranking_ceiling_status": "ESTIMATED" if n >= 5 and np.isfinite(adjusted_r) else "NOT_ESTIMABLE_LT5_REPLICATED_ENTRIES",
        "split_rule": "within-entry deterministic alternate rows after rep/subblock/plot/source-row sort",
    }


def fit_group(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    group = group.copy().reset_index(drop=True)
    group_key = [str(group.iloc[0][column]) for column in GROUP_COLS]
    group_id = stable_id("P4G_", *group_key)
    y = pd.to_numeric(group["value_standardized"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"Nonfinite standardized phenotype in eligible group {group_id}")
    n = len(group)
    n_genotypes = clean(group["resolved_gid_v2"]).nunique()

    mean_design = build_design(group, include_field_design=False, include_spline=False)
    candidates: list[Fit] = [fit_ols(y, mean_design, "UNADJUSTED_GENOTYPE_MEANS")]
    field_design = build_design(group, include_field_design=True, include_spline=False)
    if field_design.matrix.shape[1] > mean_design.matrix.shape[1]:
        candidates.append(fit_ols(y, field_design, "REP_BLOCK_ADJUSTED_BLUE"))
    spline_design = build_design(group, include_field_design=True, include_spline=True)
    if spline_design.matrix.shape[1] > field_design.matrix.shape[1]:
        spline_fit = fit_ols(y, spline_design, "PLOT_ORDER_SPLINE_BLUE")
        candidates.append(spline_fit)
        if spline_fit.status == "ELIGIBLE":
            candidates.append(fit_ar1(group, y, spline_fit))

    eligible = [(index, fit) for index, fit in enumerate(candidates)
                if fit.status == "ELIGIBLE" and np.isfinite(fit.aicc)]
    if eligible:
        # AICc selects entirely within the environment/trait.  Ties favor the
        # simpler design in the declared candidate order.
        _, selected = min(eligible, key=lambda item: (item[1].aicc, item[0]))
        selection_status = "SELECTED_BY_WITHIN_GROUP_AICC"
    else:
        selected = candidates[0]
        selection_status = "UNADJUSTED_FALLBACK_NO_ESTIMABLE_RESIDUAL_VARIANCE"

    residual_mad = 1.4826 * float(np.median(np.abs(selected.residual - np.median(selected.residual))))
    robust_trigger = (
        np.isfinite(residual_mad) and residual_mad > np.finfo(float).eps
        and float(np.max(np.abs(selected.residual))) / residual_mad > 3.5
    )
    if robust_trigger:
        robust_beta, robust_residual, robust_weights, robust_status = huber_fit(
            y, selected.design.matrix, max_iter=25
        )
    else:
        robust_beta = selected.beta.copy()
        robust_residual = selected.residual.copy()
        robust_weights = np.ones(n)
        robust_status = "NOT_TRIGGERED_NO_EXTREME_RESIDUAL"
    selected.covariance = selected_covariance(group, selected)
    contrasts = genotype_contrasts(selected.design)
    blue = contrasts @ selected.beta
    robust_blue = contrasts @ robust_beta if np.isfinite(robust_beta).all() else np.full(len(blue), np.nan)
    if np.isfinite(selected.covariance).all():
        pev = np.einsum("ij,jk,ik->i", contrasts, selected.covariance, contrasts)
        pev = np.maximum(pev, 0.0)
    else:
        pev = np.full(len(blue), np.nan)

    gid_tokens = clean(group["resolved_gid_v2"])
    raw_summary = group.assign(_gid=gid_tokens).groupby("_gid", sort=True)["value_standardized"].agg(
        ["mean", "std", "count"]
    )
    genotype_variance = math.nan
    finite_pev = pev[np.isfinite(pev)]
    if len(blue) >= 2 and len(finite_pev):
        genotype_variance = max(float(np.var(blue, ddof=1) - np.mean(finite_pev)), 0.0)
    if np.isfinite(genotype_variance):
        reliability = np.divide(
            genotype_variance,
            genotype_variance + pev,
            out=np.full_like(pev, np.nan),
            where=np.isfinite(pev) & ((genotype_variance + pev) > 0),
        )
    else:
        reliability = np.full(len(blue), np.nan)
    grand_mean = float(np.mean(blue))
    blup = grand_mean + reliability * (blue - grand_mean)
    precision = np.divide(1.0, pev, out=np.full_like(pev, np.nan), where=np.isfinite(pev) & (pev > 0))

    entry_rows: list[dict[str, Any]] = []
    for index, gid in enumerate(selected.design.genotype_levels):
        part = group[gid_tokens.eq(gid)]
        names = clean(part["genotype_name"])
        genotype_name = names[names.ne("")].iloc[0] if names.ne("").any() else ""
        entry_rows.append({
            "phase4_entry_id": stable_id("P4E_", group_id, gid),
            "phase4_group_id": group_id,
            "resolved_gid": gid,
            "canonical_germplasm_key": f"GID{gid}",
            "genotype_name": genotype_name,
            "raw_unadjusted_mean": float(raw_summary.loc[gid, "mean"]),
            "raw_unadjusted_sd": float(raw_summary.loc[gid, "std"]) if raw_summary.loc[gid, "count"] > 1 else 0.0,
            "n_plot_records": int(raw_summary.loc[gid, "count"]),
            "adjusted_blue": float(blue[index]),
            "robust_adjusted_blue": float(robust_blue[index]) if np.isfinite(robust_blue[index]) else math.nan,
            "blue_sampling_variance_pev_proxy": float(pev[index]) if np.isfinite(pev[index]) else math.nan,
            "blue_se": float(math.sqrt(pev[index])) if np.isfinite(pev[index]) else math.nan,
            "estimated_genetic_variance": genotype_variance,
            "reliability": float(reliability[index]) if np.isfinite(reliability[index]) else math.nan,
            "reliability_weight": float(reliability[index]) if np.isfinite(reliability[index]) else math.nan,
            "raw_precision_weight": float(precision[index]) if np.isfinite(precision[index]) else math.nan,
            "adjusted_blup": float(blup[index]) if np.isfinite(blup[index]) else math.nan,
            "recommended_target": "ADJUSTED_BLUE",
            "deregression_required_for_recommended_target": False,
            "blup_requires_deregression_if_used_as_target": True,
            "selected_model": selected.model_name,
            "selection_status": selection_status,
            "check_status": "JOIN_PENDING",
            "phase4_version": PHASE4_VERSION,
            **dict(zip(GROUP_COLS, group_key, strict=True)),
        })
    entries = pd.DataFrame(entry_rows)

    nuisance = selected.design.nuisance_column_indices
    adjusted_observation = y.copy()
    if nuisance:
        nuisance_effect = selected.design.matrix[:, nuisance] @ selected.beta[nuisance]
        adjusted_observation = y - nuisance_effect + float(np.mean(nuisance_effect))
    ceiling = {"phase4_group_id": group_id, **dict(zip(GROUP_COLS, group_key, strict=True)),
               **_rank_ceiling(group, adjusted_observation), "phase4_version": PHASE4_VERSION}

    centered_raw = entries["raw_unadjusted_mean"].to_numpy() - float(entries["raw_unadjusted_mean"].mean())
    centered_blue = entries["adjusted_blue"].to_numpy() - float(entries["adjusted_blue"].mean())
    raw_sd = float(np.std(centered_raw, ddof=1)) if len(centered_raw) > 1 else math.nan
    mean_pev = float(np.nanmean(pev)) if np.isfinite(pev).any() else math.nan
    h2 = (
        genotype_variance / (genotype_variance + mean_pev)
        if np.isfinite(genotype_variance) and np.isfinite(mean_pev) and genotype_variance + mean_pev > 0
        else math.nan
    )
    repeatability = (
        genotype_variance / (genotype_variance + selected.sigma2)
        if np.isfinite(genotype_variance) and np.isfinite(selected.sigma2)
        and genotype_variance + selected.sigma2 > 0 else math.nan
    )
    group_report = {
        "phase4_group_id": group_id,
        **dict(zip(GROUP_COLS, group_key, strict=True)),
        "n_plot_records": n,
        "n_genotypes": n_genotypes,
        "n_replicated_genotypes": int((raw_summary["count"] >= 2).sum()),
        "rep_levels": int(clean(group["rep"])[clean(group["rep"]).ne("")].nunique()),
        "block_levels": int(clean(group["subblock"])[clean(group["subblock"]).ne("")].nunique()),
        "numeric_plot_levels": int(pd.to_numeric(group["plot"], errors="coerce").nunique()),
        "field_row_status": "SOURCE_NOT_PROVIDED",
        "field_column_status": "SOURCE_NOT_PROVIDED",
        "ar1_by_ar1_status": "NOT_IDENTIFIABLE_NO_INDEPENDENT_ROW_COLUMN_COORDINATES",
        "selected_model": selected.model_name,
        "selection_status": selection_status,
        "selected_formula": selected.design.formula,
        "selected_sigma2": selected.sigma2,
        "entry_mean_heritability": h2,
        "plot_repeatability": repeatability,
        "mean_pev_proxy": mean_pev,
        "mean_reliability": float(np.nanmean(reliability)) if np.isfinite(reliability).any() else math.nan,
        "robust_fit_status": robust_status,
        "raw_vs_adjusted_pearson": safe_correlation(centered_raw, centered_blue, "pearson"),
        "raw_vs_adjusted_spearman": safe_correlation(centered_raw, centered_blue, "spearman"),
        "centered_rms_adjustment": float(np.sqrt(np.mean((centered_blue - centered_raw) ** 2))),
        "centered_rms_adjustment_in_raw_sd": (
            float(np.sqrt(np.mean((centered_blue - centered_raw) ** 2)) / raw_sd)
            if np.isfinite(raw_sd) and raw_sd > 0 else math.nan
        ),
        "adjusted_to_raw_variance_ratio": (
            float(np.var(centered_blue, ddof=1) / np.var(centered_raw, ddof=1))
            if len(centered_raw) > 1 and np.var(centered_raw, ddof=1) > 0 else math.nan
        ),
        "extreme_residual_count_abs_gt4sigma": (
            int(np.sum(np.abs(selected.residual) > 4.0 * math.sqrt(selected.sigma2)))
            if np.isfinite(selected.sigma2) and selected.sigma2 > 0 else 0
        ),
        "observations_removed_as_outliers": 0,
        "phase4_version": PHASE4_VERSION,
    }

    diagnostic = pd.DataFrame({
        "raw_source_row_id": group["raw_source_row_id"].astype(str),
        "canonical_row_id": group["canonical_row_id"].astype(str),
        "phase4_group_id": group_id,
        "selected_model": selected.model_name,
        "selected_fitted_value": selected.fitted,
        "selected_residual": selected.residual,
        "selected_standardized_residual": (
            selected.residual / math.sqrt(selected.sigma2)
            if np.isfinite(selected.sigma2) and selected.sigma2 > 0 else np.full(n, np.nan)
        ),
        "robust_residual": robust_residual,
        "robust_weight": robust_weights,
        "quality_flag_abs_residual_gt4sigma": (
            np.abs(selected.residual) > 4.0 * math.sqrt(selected.sigma2)
            if np.isfinite(selected.sigma2) and selected.sigma2 > 0 else np.zeros(n, dtype=bool)
        ),
        "outlier_excluded": False,
        "phase4_version": PHASE4_VERSION,
    })
    model_rows = [model_record(group_id, fit, fit is selected) for fit in candidates]
    model_rows.append({
        "phase4_group_id": group_id,
        "candidate_model": "HUBER_ROBUST_SENSITIVITY",
        "candidate_status": robust_status,
        "selected_model": False,
        "formula": selected.design.formula,
        "terms": ";".join(selected.design.terms),
        "n_observations": n,
        "n_parameters": selected.p,
        "matrix_rank": selected.rank,
        "df_resid": selected.df_resid,
        "rss": float(robust_residual @ robust_residual) if np.isfinite(robust_residual).all() else math.nan,
        "sigma2": math.nan,
        "aic": math.nan,
        "aicc": math.nan,
        "ar1_rho": math.nan,
        "ar1_adjacent_pairs": 0,
        "max_absolute_standardized_residual": math.nan,
        "phase4_version": PHASE4_VERSION,
    })
    return entries, diagnostic, model_rows, group_report, ceiling


class SafeParquetWriter:
    def __init__(self, path: Path):
        self.path = path
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        elif table.schema != self.schema:
            table = table.cast(self.schema, safe=False)
        self.writer.write_table(table, row_group_size=100_000)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _complete_groups_from_batches(path: Path):
    reader = pq.ParquetFile(path)
    carry = pd.DataFrame()
    for batch in reader.iter_batches(batch_size=100_000):
        frame = pa.Table.from_batches([batch]).to_pandas()
        if not carry.empty:
            frame = pd.concat([carry, frame], ignore_index=True)
            carry = pd.DataFrame()
        if frame.empty:
            continue
        last_key = tuple(frame.iloc[-1][column] for column in GROUP_COLS)
        last_mask = np.ones(len(frame), dtype=bool)
        for column, value in zip(GROUP_COLS, last_key, strict=True):
            last_mask &= frame[column].eq(value).to_numpy()
        carry = frame.loc[last_mask].copy()
        complete = frame.loc[~last_mask]
        if not complete.empty:
            for _, group in complete.groupby(list(GROUP_COLS), sort=False, dropna=False):
                yield group
    if not carry.empty:
        yield carry


def _fit_worker(groups: list[pd.DataFrame]):
    return [fit_group(group) for group in groups]


def _chunked_groups(path: Path, chunk_size: int):
    chunk: list[pd.DataFrame] = []
    for group in _complete_groups_from_batches(path):
        chunk.append(group)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _bounded_ordered_map(executor: ProcessPoolExecutor, chunks, max_pending: int):
    """Preserve deterministic order without eagerly materializing all groups."""
    pending = deque()
    iterator = iter(chunks)
    for _ in range(max_pending):
        try:
            pending.append(executor.submit(_fit_worker, next(iterator)))
        except StopIteration:
            break
    while pending:
        future = pending.popleft()
        yield future.result()
        try:
            pending.append(executor.submit(_fit_worker, next(iterator)))
        except StopIteration:
            pass


def write_opening_freeze(result_dir: Path, inputs: list[Path]) -> None:
    manifest = []
    for path in inputs:
        manifest.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "role": "READ_ONLY_INPUT",
        })
    pd.DataFrame(manifest).to_csv(result_dir / "input_freeze_manifest.tsv", sep="\t", index=False)
    protocol = {
        "phase4_version": PHASE4_VERSION,
        "scope": "seven predeclared modelling traits; every exact environment/canonical-trait/original-trait/unit group",
        "selected_traits": list(SELECTED_TRAITS),
        "group_columns": list(GROUP_COLS),
        "protected_outcome_policy": "outer-test outcomes and sealed final holdout not opened, queried, summarized, or used",
        "row_column_policy": "independent field row/column were not supplied; do not infer them from plot or subblock",
        "ar1_by_ar1_policy": "not identifiable for all groups",
        "spatial_alternatives": ["plot-order cubic regression spline", "plot-order AR1 GLS"],
        "selection_rule": "minimum within-group AICc among identifiable Gaussian candidates; deterministic complexity tie-break",
        "robust_policy": "Huber fit is a sensitivity analysis; no observation is deleted",
        "check_policy": "only exact 0 and exact 1 receive literal noncheck/check labels; 100 and other codes remain unconfirmed/ambiguous",
        "recommended_target_rule": "selected-model adjusted BLUE with PEV proxy and reliability; no deregression",
        "randomness": "none; replicate split uses deterministic alternating order",
    }
    (result_dir / "phase4_frozen_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    dependencies = [
        ("Python", platform.python_version(), "existing isolated environment"),
        ("numpy", np.__version__, "existing isolated environment"),
        ("pandas", pd.__version__, "existing isolated environment"),
        ("pyarrow", pa.__version__, "existing isolated environment"),
        ("duckdb", duckdb.__version__, "existing isolated environment"),
        ("scipy", __import__("scipy").__version__, "existing isolated environment"),
        ("statsmodels", __import__("statsmodels").__version__, "added in Phase 4; diagnostic compatibility"),
        ("patsy", __import__("patsy").__version__, "added as statsmodels dependency"),
    ]
    pd.DataFrame(dependencies, columns=["dependency", "exact_version", "provenance"]).to_csv(
        result_dir / "dependencies_added.tsv", sep="\t", index=False
    )


def build_sorted_input(con: duckdb.DuckDBPyConnection, canonical: Path, sorted_path: Path) -> int:
    source = str(canonical).replace("'", "''")
    target = str(sorted_path).replace("'", "''")
    traits = ",".join(f"'{trait}'" for trait in SELECTED_TRAITS)
    columns = list(GROUP_COLS) + [
        "raw_source_row_id", "canonical_row_id", "resolved_gid_v2", "genotype_name",
        "value_standardized", "raw_value_token", "raw_unit", "rep", "subblock", "plot",
        "quality_flags_v2", "source_file", "source_member", "source_physical_row",
    ]
    quoted = ",".join(f'"{column}"' for column in columns)
    ordered = ",".join(f'"{column}"' for column in list(GROUP_COLS) + ["raw_source_row_id"])
    con.execute(
        f"COPY (SELECT {quoted} FROM read_parquet('{source}') "
        f"WHERE row_disposition_v2='ELIGIBLE_STAGE1_CONTRIBUTOR' "
        f"AND accepted_canonical_trait IN ({traits}) ORDER BY {ordered}) "
        f"TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    return int(con.execute("SELECT count(*) FROM read_parquet(?)", [str(sorted_path)]).fetchone()[0])


def build_check_reconstruction(con: duckdb.DuckDBPyConnection, canonical: Path, output: Path) -> None:
    source = str(canonical).replace("'", "''")
    target = str(output).replace("'", "''")
    con.execute(f"""
        COPY (
          WITH base AS (
            SELECT canonical_environment_id, resolved_gid_v2 AS resolved_gid,
                   value_standardized, raw_value_token, raw_unit,
                   raw_source_row_id, canonical_row_id, source_file, source_member, source_physical_row
            FROM read_parquet('{source}')
            WHERE row_disposition_v2='ELIGIBLE_STAGE1_CONTRIBUTOR'
              AND accepted_canonical_trait='SELECTED_CHECK_MARK'
          ), agg AS (
            SELECT canonical_environment_id, resolved_gid,
                   count(*) AS source_rows,
                   count(DISTINCT value_standardized) AS distinct_numeric_codes,
                   string_agg(DISTINCT cast(value_standardized AS VARCHAR), ';' ORDER BY cast(value_standardized AS VARCHAR)) AS numeric_codes,
                   string_agg(DISTINCT raw_value_token, ';' ORDER BY raw_value_token) AS raw_tokens,
                   string_agg(DISTINCT raw_unit, ';' ORDER BY raw_unit) AS raw_units,
                   string_agg(raw_source_row_id, ';' ORDER BY raw_source_row_id) AS source_row_ids
            FROM base GROUP BY ALL
          )
          SELECT *, CASE
            WHEN distinct_numeric_codes > 1 THEN 'AMBIGUOUS_CONFLICTING_CHECK_CODES'
            WHEN numeric_codes='1.0' THEN 'CHECK_EXACT_1'
            WHEN numeric_codes='0.0' THEN 'NONCHECK_EXACT_0'
            WHEN numeric_codes='100.0' THEN 'CHECK_CODE_100_UNCONFIRMED'
            ELSE 'AMBIGUOUS_NONBINARY_CHECK_CODE'
          END AS check_status,
          '{PHASE4_VERSION}' AS phase4_version
          FROM agg ORDER BY canonical_environment_id, resolved_gid
        ) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)


def postprocess_outputs(
    con: duckdb.DuckDBPyConnection,
    canonical: Path,
    check_path: Path,
    entries_temp: Path,
    diagnostics: Path,
    entries_final: Path,
    plot_final: Path,
    reliability_final: Path,
) -> None:
    check = str(check_path).replace("'", "''")
    entries = str(entries_temp).replace("'", "''")
    entries_out = str(entries_final).replace("'", "''")
    reliability_out = str(reliability_final).replace("'", "''")
    con.execute(f"""
      COPY (
        SELECT e.* EXCLUDE(check_status), coalesce(c.check_status, 'NO_CHECK_MARK_RECORD') AS check_status
        FROM read_parquet('{entries}') e
        LEFT JOIN read_parquet('{check}') c USING (canonical_environment_id, resolved_gid)
        ORDER BY phase4_group_id, resolved_gid
      ) TO '{entries_out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)
    con.execute(f"""
      COPY (
        SELECT phase4_entry_id, phase4_group_id, canonical_environment_id,
               accepted_canonical_trait, trait_name_original, standardized_unit,
               resolved_gid, blue_sampling_variance_pev_proxy, blue_se,
               estimated_genetic_variance, reliability, reliability_weight,
               raw_precision_weight, adjusted_blup,
               deregression_required_for_recommended_target,
               blup_requires_deregression_if_used_as_target, phase4_version
        FROM read_parquet('{entries_out}')
      ) TO '{reliability_out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)
    source = str(canonical).replace("'", "''")
    diagnostic = str(diagnostics).replace("'", "''")
    plot_out = str(plot_final).replace("'", "''")
    traits = ",".join(f"'{trait}'" for trait in SELECTED_TRAITS)
    con.execute(f"""
      COPY (
        SELECT c.raw_source_row_id, c.canonical_row_id, c.source_file, c.source_member,
               c.source_physical_row, c.canonical_environment_id, c.canonical_trial_name,
               c.cycle, c.occ, c.loc_no, c.country, c.loc_desc, c.resolved_gid_v2 AS resolved_gid,
               c.canonical_germplasm_key, c.genotype_name, c.accepted_canonical_trait,
               c.trait_name_original, c.raw_value_token, c.raw_unit, c.value_standardized,
               c.standardized_unit, c.rep, c.subblock, c.plot,
               '' AS field_row, '' AS field_column,
               'SOURCE_NOT_PROVIDED' AS field_row_status,
               'SOURCE_NOT_PROVIDED' AS field_column_status,
               coalesce(ch.check_status, 'NO_CHECK_MARK_RECORD') AS check_status,
               c.quality_flags_v2 AS canonical_quality_flags,
               d.phase4_group_id, d.selected_model, d.selected_fitted_value,
               d.selected_residual, d.selected_standardized_residual,
               d.robust_residual, d.robust_weight, d.quality_flag_abs_residual_gt4sigma,
               d.outlier_excluded, d.phase4_version
        FROM read_parquet('{source}') c
        JOIN read_parquet('{diagnostic}') d USING (raw_source_row_id, canonical_row_id)
        LEFT JOIN read_parquet('{check}') ch
          ON c.canonical_environment_id=ch.canonical_environment_id
         AND c.resolved_gid_v2=ch.resolved_gid
        WHERE c.row_disposition_v2='ELIGIBLE_STAGE1_CONTRIBUTOR'
          AND c.accepted_canonical_trait IN ({traits})
        ORDER BY d.phase4_group_id, c.raw_source_row_id
      ) TO '{plot_out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)


def write_delivery_tables(
    result_dir: Path,
    group_reports: list[dict[str, Any]],
    ceilings: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    con: duckdb.DuckDBPyConnection,
    entries: Path,
    plot_design: Path,
    stage1: Path,
    canonical_count: int,
) -> dict[str, Any]:
    groups = pd.DataFrame(group_reports)
    ceiling = pd.DataFrame(ceilings)
    models = pd.DataFrame(model_rows)
    groups.to_csv(result_dir / "trial_trait_spatial_model_selection_report.tsv", sep="\t", index=False)
    ceiling.to_csv(result_dir / "ranking_ceiling_estimates.tsv", sep="\t", index=False)
    models.to_csv(result_dir / "candidate_model_comparison.tsv", sep="\t", index=False)
    signal_columns = [
        "phase4_group_id", *GROUP_COLS, "selected_model", "raw_vs_adjusted_pearson",
        "raw_vs_adjusted_spearman", "centered_rms_adjustment",
        "centered_rms_adjustment_in_raw_sd", "adjusted_to_raw_variance_ratio",
    ]
    groups[signal_columns].to_csv(
        result_dir / "centered_genetic_signal_change.tsv", sep="\t", index=False
    )
    merged = groups.merge(
        ceiling[["phase4_group_id", "n_entries_split", "adjusted_spearman_brown_ceiling", "ranking_ceiling_status"]],
        on="phase4_group_id", how="left", validate="one_to_one",
    )
    merged["ranking_claim_status"] = np.select(
        [
            merged["entry_mean_heritability"].isna(),
            merged["mean_reliability"].isna(),
            merged["mean_reliability"].lt(0.30),
            merged["adjusted_spearman_brown_ceiling"].notna()
            & merged["adjusted_spearman_brown_ceiling"].lt(0.30),
        ],
        [
            "TOO_UNRELIABLE_HERITABILITY_NOT_ESTIMABLE",
            "TOO_UNRELIABLE_RELIABILITY_NOT_ESTIMABLE",
            "TOO_UNRELIABLE_MEAN_RELIABILITY_LT_0_30",
            "TOO_UNRELIABLE_RANKING_CEILING_LT_0_30",
        ],
        default="RANKING_SIGNAL_USABLE_WITH_REPORTED_UNCERTAINTY",
    )
    unreliable = merged[merged["ranking_claim_status"].str.startswith("TOO_UNRELIABLE")].copy()
    unreliable.to_csv(result_dir / "unreliable_environment_trait_groups.tsv", sep="\t", index=False)

    entry_path = str(entries)
    plot_path = str(plot_design)
    stage1_path = str(stage1)
    entry_count = int(con.execute("SELECT count(*) FROM read_parquet(?)", [entry_path]).fetchone()[0])
    plot_count = int(con.execute("SELECT count(*) FROM read_parquet(?)", [plot_path]).fetchone()[0])
    selected_stage1_count = int(con.execute(
        "SELECT count(*) FROM read_parquet(?) WHERE accepted_canonical_trait IN (?,?,?,?,?,?,?)",
        [stage1_path, *SELECTED_TRAITS],
    ).fetchone()[0])
    if canonical_count != EXPECTED_SELECTED_CONTRIBUTORS or plot_count != canonical_count:
        raise AssertionError(
            f"Plot reconciliation failed: expected {EXPECTED_SELECTED_CONTRIBUTORS}, canonical={canonical_count}, output={plot_count}"
        )
    if selected_stage1_count != EXPECTED_SELECTED_STAGE1_ROWS or entry_count != selected_stage1_count:
        raise AssertionError(
            f"Entry reconciliation failed: expected {EXPECTED_SELECTED_STAGE1_ROWS}, Stage1={selected_stage1_count}, Phase4={entry_count}"
        )
    duplicate_plot_ids = int(con.execute(
        "SELECT count(*)-count(DISTINCT raw_source_row_id) FROM read_parquet(?)", [plot_path]
    ).fetchone()[0])
    duplicate_entry_ids = int(con.execute(
        "SELECT count(*)-count(DISTINCT phase4_entry_id) FROM read_parquet(?)", [entry_path]
    ).fetchone()[0])
    if duplicate_plot_ids or duplicate_entry_ids:
        raise AssertionError(f"ID uniqueness failed: plot={duplicate_plot_ids}, entry={duplicate_entry_ids}")

    before_after = con.execute(
        """
        WITH p AS (
          SELECT accepted_canonical_trait, count(*) AS canonical_plot_rows,
                 count(DISTINCT phase4_group_id) AS phase4_groups
          FROM read_parquet(?) GROUP BY ALL
        ), e AS (
          SELECT accepted_canonical_trait, count(*) AS phase3_stage1_entry_rows
          FROM read_parquet(?) WHERE accepted_canonical_trait IN (?,?,?,?,?,?,?) GROUP BY ALL
        ), a AS (
          SELECT accepted_canonical_trait, count(*) AS phase4_adjusted_entry_rows,
                 sum((reliability IS NOT NULL)::INTEGER) AS entries_with_reliability
          FROM read_parquet(?) GROUP BY ALL
        )
        SELECT p.*, e.phase3_stage1_entry_rows, a.phase4_adjusted_entry_rows,
               a.entries_with_reliability,
               p.canonical_plot_rows-a.phase4_adjusted_entry_rows AS plot_to_entry_reduction
        FROM p JOIN e USING (accepted_canonical_trait) JOIN a USING (accepted_canonical_trait)
        ORDER BY accepted_canonical_trait
        """,
        [plot_path, stage1_path, *SELECTED_TRAITS, entry_path],
    ).fetchdf()
    before_after.to_csv(result_dir / "before_after_counts.tsv", sep="\t", index=False)

    selection_counts = groups["selected_model"].value_counts(dropna=False).rename_axis("selected_model").reset_index(name="groups")
    reliability_summary = groups.groupby("accepted_canonical_trait", dropna=False).agg(
        groups=("phase4_group_id", "size"),
        median_entry_mean_heritability=("entry_mean_heritability", "median"),
        median_plot_repeatability=("plot_repeatability", "median"),
        median_mean_reliability=("mean_reliability", "median"),
        median_raw_vs_adjusted_spearman=("raw_vs_adjusted_spearman", "median"),
    ).reset_index()
    reliability_summary.to_csv(result_dir / "trait_reliability_summary.tsv", sep="\t", index=False)
    summary = {
        "phase4_version": PHASE4_VERSION,
        "status": "PASS_PHASE4_RECONSTRUCTION_AND_SIGNAL_ASSESSMENT",
        "canonical_selected_plot_records": canonical_count,
        "plot_design_output_rows": plot_count,
        "phase3_selected_stage1_rows": selected_stage1_count,
        "phase4_adjusted_entry_rows": entry_count,
        "trial_trait_groups": len(groups),
        "groups_with_estimable_heritability": int(groups["entry_mean_heritability"].notna().sum()),
        "groups_with_ranking_ceiling": int(ceiling["ranking_ceiling_status"].eq("ESTIMATED").sum()),
        "groups_too_unreliable_for_ranking_claims": len(unreliable),
        "groups_by_selected_model": dict(zip(selection_counts["selected_model"], selection_counts["groups"].astype(int), strict=True)),
        "ar1_by_ar1_fitted_groups": 0,
        "observations_excluded_as_outliers": 0,
        "recommended_v2_target": "within-group selected adjusted BLUE with PEV proxy and reliability",
        "deregression_recommendation": "not required for recommended BLUE; required if adjusted BLUP is substituted",
    }
    (result_dir / "phase4_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--groups-per-task", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1 or args.groups_per_task < 1:
        raise ValueError("workers and groups-per-task must be positive")
    canonical = args.canonical.resolve()
    stage1 = args.stage1.resolve()
    bridge = args.bridge.resolve()
    for path in (canonical, stage1, bridge):
        if not path.is_file():
            raise FileNotFoundError(path)
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    write_opening_freeze(result_dir, [canonical, stage1, bridge])
    canonical_hash_before = file_sha256(canonical)

    work_dir = result_dir / "work"
    work_dir.mkdir()
    sorted_input = work_dir / "selected_contributors_sorted.parquet"
    check_path = result_dir / "check_reconstruction_v1.parquet"
    entries_temp = work_dir / "adjusted_phenotypes_without_check.parquet"
    diagnostics = result_dir / "plot_model_diagnostics_v1.parquet"
    entries_final = result_dir / "adjusted_phenotypes_v1.parquet"
    plot_final = result_dir / "plot_design_reconstruction_v1.parquet"
    reliability_final = result_dir / "reliability_pev_v1.parquet"

    con = duckdb.connect(str(work_dir / "phase4.duckdb"))
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA preserve_insertion_order=false")
    canonical_count = build_sorted_input(con, canonical, sorted_input)
    if canonical_count != EXPECTED_SELECTED_CONTRIBUTORS:
        raise AssertionError(f"Unexpected selected contributor count: {canonical_count}")
    build_check_reconstruction(con, canonical, check_path)

    entry_writer = SafeParquetWriter(entries_temp)
    diagnostic_writer = SafeParquetWriter(diagnostics)
    model_rows: list[dict[str, Any]] = []
    group_reports: list[dict[str, Any]] = []
    ceilings: list[dict[str, Any]] = []
    processed_groups = 0
    processed_plots = 0
    processed_entries = 0

    chunks = _chunked_groups(sorted_input, args.groups_per_task)
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        results_iter = (
            _bounded_ordered_map(executor, chunks, max_pending=args.workers * 2)
            if executor else map(_fit_worker, chunks)
        )
        for task_results in results_iter:
            for entries, diagnostic, models, report, ceiling in task_results:
                entry_writer.write(entries)
                diagnostic_writer.write(diagnostic)
                model_rows.extend(models)
                group_reports.append(report)
                ceilings.append(ceiling)
                processed_groups += 1
                processed_plots += len(diagnostic)
                processed_entries += len(entries)
            if processed_groups and processed_groups % 500 < args.groups_per_task:
                print(
                    f"processed groups={processed_groups:,} plots={processed_plots:,} entries={processed_entries:,}",
                    flush=True,
                )
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=False)
        entry_writer.close()
        diagnostic_writer.close()

    if processed_groups != 37_206 or processed_plots != canonical_count:
        raise AssertionError(
            f"Streaming reconciliation failed groups={processed_groups}, plots={processed_plots}"
        )
    postprocess_outputs(
        con, canonical, check_path, entries_temp, diagnostics, entries_final, plot_final,
        reliability_final,
    )
    summary = write_delivery_tables(
        result_dir, group_reports, ceilings, model_rows, con, entries_final, plot_final,
        stage1, canonical_count,
    )
    con.close()
    if file_sha256(canonical) != canonical_hash_before:
        raise AssertionError("Canonical input changed during Phase 4")

    # Remove only private working files created beneath the new Phase-4 root.
    for path in (sorted_input, entries_temp, work_dir / "phase4.duckdb"):
        if path.exists():
            path.unlink()
    try:
        work_dir.rmdir()
    except OSError:
        pass

    output_manifest = []
    for path in sorted(result_dir.glob("*")):
        if path.is_file():
            output_manifest.append({
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })
    pd.DataFrame(output_manifest).to_csv(result_dir / "output_manifest.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
