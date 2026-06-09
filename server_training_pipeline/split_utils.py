from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


SPLIT_ALIASES = {
    "random": "cv2_random_observation",
    "cv2": "cv2_random_observation",
    "loeo": "gho_environment",
    "loyo": "gho_cycle",
    "loto": "gho_trial",
    "loco": "gho_country",
    "lofo": "gho_family",
}

CANONICAL_SPLIT_MODES = [
    "cv2_random_observation", "gho_environment", "gho_cycle", "gho_trial",
    "gho_country", "gho_family", "cv1_genotype", "cv1_environment",
    "cv0_genotype_environment", "group_kfold",
]


def canonical_split_mode(mode: str, warn: bool = False) -> str:
    normalized = mode.strip().lower()
    canonical = SPLIT_ALIASES.get(normalized, normalized)
    if canonical not in CANONICAL_SPLIT_MODES:
        raise SystemExit(f"Unknown split mode {mode!r}; choose from {CANONICAL_SPLIT_MODES}")
    if warn and canonical != normalized:
        warnings.warn(f"Split mode {mode!r} is a legacy alias; recording canonical mode {canonical!r}.", stacklevel=2)
    return canonical


def grouped_holdout(df: pd.DataFrame, group_col: str, seed: int, test_fraction: float, val_fraction: float):
    if group_col not in df.columns:
        raise SystemExit(f"Grouped holdout requires group column {group_col}")
    rng = np.random.default_rng(seed)
    groups = np.asarray(df[group_col].fillna("").astype(str).unique(), dtype=object).copy()
    if len(groups) < 3:
        raise SystemExit(f"Grouped holdout requires at least three groups in {group_col}; found {len(groups)}")
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_fraction)))
    n_val = max(1, int(round(len(groups) * val_fraction)))
    if n_test + n_val >= len(groups):
        n_test = n_val = 1
    test_groups, val_groups = set(groups[:n_test]), set(groups[n_test:n_test + n_val])
    values = df[group_col].fillna("").astype(str)
    return (
        np.where(~values.isin(test_groups | val_groups))[0],
        np.where(values.isin(val_groups))[0],
        np.where(values.isin(test_groups))[0],
    )


def cv0_split(df: pd.DataFrame, seed: int, test_fraction: float, val_fraction: float):
    rng = np.random.default_rng(seed)
    geno = df["panel_sample_id"].fillna("").astype(str)
    env = df["env_kernel_id"].fillna("").astype(str)
    def partition(values):
        groups = np.asarray(values.unique(), dtype=object).copy()
        if len(groups) < 3:
            raise SystemExit("cv0_genotype_environment requires at least three groups on each axis")
        rng.shuffle(groups)
        n_test, n_val = max(1, round(len(groups) * test_fraction)), max(1, round(len(groups) * val_fraction))
        if n_test + n_val >= len(groups):
            n_test = n_val = 1
        return set(groups[:n_test]), set(groups[n_test:n_test + n_val])
    test_g, val_g = partition(geno)
    test_e, val_e = partition(env)
    return (
        np.where(~geno.isin(test_g | val_g) & ~env.isin(test_e | val_e))[0],
        np.where(geno.isin(val_g) & env.isin(val_e))[0],
        np.where(geno.isin(test_g) & env.isin(test_e))[0],
    )


def make_split(df, mode, seed, test_fraction, val_fraction, group_col=None):
    mode = canonical_split_mode(mode)
    if mode == "cv2_random_observation":
        idx = np.arange(len(df))
        np.random.default_rng(seed).shuffle(idx)
        n_test, n_val = max(1, round(len(df) * test_fraction)), max(1, round(len(df) * val_fraction))
        return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]
    if mode == "cv0_genotype_environment":
        return cv0_split(df, seed, test_fraction, val_fraction)
    return grouped_holdout(df, group_col, seed, test_fraction, val_fraction)


def split_leakage_record(df, repeat, split_mode, train, val, test, group_col=None):
    mode = canonical_split_mode(split_mode)
    geno_col = "panel_sample_id" if "panel_sample_id" in df else "geno_kernel_index"
    env_col = "env_kernel_id" if "env_kernel_id" in df else "env_kernel_index"
    geno, env = df[geno_col].fillna("").astype(str), df[env_col].fillna("").astype(str)
    train_g, val_g, test_g = set(geno.iloc[train]), set(geno.iloc[val]), set(geno.iloc[test])
    train_e, val_e, test_e = set(env.iloc[train]), set(env.iloc[val]), set(env.iloc[test])
    expected_g = "zero" if mode in {"cv1_genotype", "cv0_genotype_environment"} else "allowed"
    expected_e = "zero" if mode in {"gho_environment", "cv1_environment", "cv0_genotype_environment"} else "allowed"
    if mode == "group_kfold":
        expected_g = "zero" if group_col in {geno_col, "panel_sample_id"} else expected_g
        expected_e = "zero" if group_col in {env_col, "env_kernel_id"} else expected_e
    go, eo = len(train_g & test_g), len(train_e & test_e)
    failed = (expected_g == "zero" and go) or (expected_e == "zero" and eo)
    return {
        "repeat": repeat, "split_mode": mode, "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
        "train_unique_genotypes": len(train_g), "val_unique_genotypes": len(val_g), "test_unique_genotypes": len(test_g),
        "train_unique_environments": len(train_e), "val_unique_environments": len(val_e), "test_unique_environments": len(test_e),
        "geno_overlap_train_test": go, "env_overlap_train_test": eo,
        "expected_geno_overlap": expected_g, "expected_env_overlap": expected_e,
        "leakage_status": "fail" if failed else "pass",
    }
