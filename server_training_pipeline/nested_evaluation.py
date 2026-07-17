from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .final_evaluation_contract import file_sha256
from .split_utils import split_leakage_record


SCENARIO_MODES = {
    "unseen_environments": "gho_environment",
    "unseen_genotypes": "cv1_genotype",
    "unseen_genotypes_and_environments": "cv0_genotype_environment",
    "temporal_holdout": "gho_cycle",
    "country_holdout": "gho_country",
}

AXIS_COLUMNS = {
    "environment": "env_kernel_id",
    "genotype": "panel_sample_id",
    "cycle": "cycle",
    "country": "country",
}


def stable_bucket(value: str, salt: str, buckets: int) -> int:
    digest = hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def cycle_year(value: object) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", "" if pd.isna(value) else str(value))
    return int(match.group(0)) if match else None


def _values(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"Evaluation ledger is missing required column {column}")
    return frame[column].fillna("").astype(str).str.strip()


def _axis_rows(
    scenario: str,
    outer_fold: int,
    inner_fold: int,
    axis: str,
    partition: str,
    values: set[str] | list[str],
) -> list[dict[str, object]]:
    return [
        {
            "scenario": scenario,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "axis": axis,
            "partition": partition,
            "entity_id": value,
        }
        for value in sorted(set(values))
        if value
    ]


def manifest_identity(manifest_path: Path, contract_path: Path) -> dict[str, str]:
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": file_sha256(contract_path),
    }


def verify_manifest_contract(manifest_path: Path, contract_path: Path) -> dict[str, object]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    observed = file_sha256(manifest_path)
    if observed != contract.get("entity_manifest_sha256"):
        raise ValueError(
            "Immutable evaluation manifest hash mismatch: "
            f"expected={contract.get('entity_manifest_sha256')} observed={observed}"
        )
    if contract.get("status") != "frozen":
        raise ValueError("Evaluation manifest contract is not frozen")
    return contract


def assign_nested_split(
    ledger: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    scenario: str,
    outer_fold: int,
    inner_fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if scenario not in SCENARIO_MODES:
        raise ValueError(f"Unknown nested-evaluation scenario {scenario!r}")
    required = {"scenario", "outer_fold", "inner_fold", "axis", "partition", "entity_id"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Evaluation manifest is missing columns: {missing}")
    selected = manifest[
        manifest["scenario"].eq(scenario)
        & pd.to_numeric(manifest["outer_fold"], errors="coerce").eq(outer_fold)
        & pd.to_numeric(manifest["inner_fold"], errors="coerce").eq(inner_fold)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No manifest rows for scenario={scenario} outer_fold={outer_fold} "
            f"inner_fold={inner_fold}"
        )
    masks: dict[tuple[str, str], np.ndarray] = {}
    for (axis, partition), group in selected.groupby(["axis", "partition"]):
        if axis not in AXIS_COLUMNS:
            raise ValueError(f"Unsupported evaluation axis {axis!r}")
        ids = set(group["entity_id"].fillna("").astype(str))
        masks[(axis, partition)] = _values(ledger, AXIS_COLUMNS[axis]).isin(ids).to_numpy()

    final_mask = masks.get(("environment", "final_holdout"), np.zeros(len(ledger), dtype=bool))
    excluded = final_mask.copy()
    for axis in AXIS_COLUMNS:
        excluded |= masks.get((axis, "excluded"), np.zeros(len(ledger), dtype=bool))

    if scenario == "unseen_genotypes_and_environments":
        test_g = masks.get(("genotype", "outer_test"), np.zeros(len(ledger), dtype=bool))
        test_e = masks.get(("environment", "outer_test"), np.zeros(len(ledger), dtype=bool))
        val_g = masks.get(("genotype", "inner_validation"), np.zeros(len(ledger), dtype=bool))
        val_e = masks.get(("environment", "inner_validation"), np.zeros(len(ledger), dtype=bool))
        test_mask = test_g & test_e & ~excluded
        val_mask = val_g & val_e & ~excluded & ~test_mask
        blocked_axes = test_g | test_e | val_g | val_e
        train_mask = ~blocked_axes & ~excluded
    else:
        axis = {
            "unseen_environments": "environment",
            "unseen_genotypes": "genotype",
            "temporal_holdout": "cycle",
            "country_holdout": "country",
        }[scenario]
        test_mask = masks.get((axis, "outer_test"), np.zeros(len(ledger), dtype=bool))
        val_mask = masks.get((axis, "inner_validation"), np.zeros(len(ledger), dtype=bool))
        test_mask &= ~excluded
        val_mask &= ~excluded & ~test_mask
        train_mask = ~(test_mask | val_mask | excluded)

    train = np.flatnonzero(train_mask)
    val = np.flatnonzero(val_mask)
    test = np.flatnonzero(test_mask)
    omitted = np.flatnonzero(~(train_mask | val_mask | test_mask))
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(
            f"Nested split is empty: train={len(train)} val={len(val)} test={len(test)}"
        )
    mode = SCENARIO_MODES[scenario]
    group_col = {
        "gho_environment": "env_kernel_id",
        "cv1_genotype": "panel_sample_id",
        "gho_cycle": "cycle",
        "gho_country": "country",
    }.get(mode)
    leakage = split_leakage_record(
        ledger, f"outer{outer_fold}_inner{inner_fold}", mode, train, val, test, group_col
    )
    leakage.update(
        {
            "scenario": scenario,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "omitted_rows": len(omitted),
            "final_holdout_rows": int(final_mask.sum()),
        }
    )
    if leakage["leakage_status"] != "pass":
        raise ValueError(f"Nested split leakage detected: {leakage}")
    return train, val, test, omitted, leakage
