from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd


ALIASES = {
    "random": "cv2_random_observation",
    "cv2": "cv2_random_observation",
    "cv2_random": "cv2_random_observation",
    "loeo": "gho_environment",
    "loyo": "gho_cycle",
    "loco": "gho_country",
    "gho_env": "gho_environment",
}

SUPPORTED_SPLITS = {
    "cv2_random_observation",
    "gho_environment",
    "gho_cycle",
    "gho_trial",
    "gho_country",
    "gho_family",
    "cv1_genotype",
    "cv0_genotype_environment",
}


def canonical_split_mode(mode: str, warn: bool = False) -> str:
    raw = str(mode or "").strip()
    key = raw.lower()
    canonical = ALIASES.get(key, key)
    if canonical not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split mode: {mode!r}")
    if warn and raw and canonical != raw:
        warnings.warn(f"Split mode {raw!r} is a legacy alias; recording canonical mode {canonical!r}.")
    return canonical


def _as_group_key(df: pd.DataFrame, group_col: str | list[str]) -> pd.Series:
    if isinstance(group_col, list):
        missing = [c for c in group_col if c not in df.columns]
        if missing:
            raise ValueError(f"Missing split group columns: {missing}")
        return df[group_col].fillna("").astype(str).agg("|".join, axis=1)
    if group_col not in df.columns:
        raise ValueError(f"Missing split group column: {group_col}")
    return df[group_col].fillna("").astype(str)


def make_split(
    df: pd.DataFrame,
    mode: str,
    seed: int,
    test_fraction: float,
    val_fraction: float,
    group_col: str | list[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    canonical = canonical_split_mode(mode)
    rng = np.random.default_rng(seed)
    n = len(df)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if canonical == "cv2_random_observation":
        idx = np.arange(n, dtype=np.int64)
        rng.shuffle(idx)
        n_test = max(1, int(round(n * test_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]

    if group_col is None:
        raise ValueError(f"Split mode {canonical!r} requires a group column")
    group_series = _as_group_key(df, group_col)
    groups = np.asarray(group_series.unique(), dtype=object)
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_fraction)))
    n_val = max(1, int(round(len(groups) * val_fraction)))
    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test : n_test + n_val])
    test = np.where(group_series.isin(test_groups))[0]
    val = np.where(group_series.isin(val_groups))[0]
    train = np.where(~group_series.isin(test_groups | val_groups))[0]
    return train.astype(np.int64), val.astype(np.int64), test.astype(np.int64)


def split_group_column(mode: str) -> str | list[str] | None:
    canonical = canonical_split_mode(mode)
    return {
        "cv2_random_observation": None,
        "gho_environment": "env_kernel_id",
        "gho_cycle": "cycle",
        "gho_trial": "trial_name",
        "gho_country": "country",
        "gho_family": "family_id",
        "cv1_genotype": "panel_sample_id",
        "cv0_genotype_environment": ["panel_sample_id", "env_kernel_id"],
    }[canonical]


def split_leakage_record(
    df: pd.DataFrame,
    repeat: int,
    split_mode: str,
    train_idx: Iterable[int],
    val_idx: Iterable[int],
    test_idx: Iterable[int],
    group_col: str | list[str] | None = None,
) -> dict[str, object]:
    canonical = canonical_split_mode(split_mode)
    train_idx = np.asarray(list(train_idx), dtype=np.int64)
    val_idx = np.asarray(list(val_idx), dtype=np.int64)
    test_idx = np.asarray(list(test_idx), dtype=np.int64)
    rec: dict[str, object] = {
        "repeat": repeat,
        "split_mode": canonical,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "leakage_status": "pass",
    }
    if group_col is None or canonical == "cv2_random_observation":
        return rec
    groups = _as_group_key(df, group_col)
    train_groups = set(groups.iloc[train_idx])
    val_groups = set(groups.iloc[val_idx])
    test_groups = set(groups.iloc[test_idx])
    overlaps = {
        "train_val_overlap": len(train_groups & val_groups),
        "train_test_overlap": len(train_groups & test_groups),
        "val_test_overlap": len(val_groups & test_groups),
    }
    rec.update(overlaps)
    if any(v > 0 for v in overlaps.values()):
        rec["leakage_status"] = "fail"
    return rec
