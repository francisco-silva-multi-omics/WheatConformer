#!/usr/bin/env python3
"""Shared constants and deterministic helpers for the namespace/R3 releases."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("STAGE1_V2_DATA_ROOT", REPOSITORY_ROOT)).resolve()
TRIAL_ROOT = DATA_ROOT / "TRIALS_AND_NURSERIES_DATA"
GENOTYPE_ROOT = DATA_ROOT / "GENOTYPIC_DATA"
STAGE1_ROOT = DATA_ROOT / "audit/v2/phase3_stage1_v2_reconstruction_v1"
PHASE3G_R2_ROOT = DATA_ROOT / "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2"
PHASE4_ROOT = DATA_ROOT / "audit/v2/phase4_integrated_spatial_promotion_release_v1"
PHASE5_ROOT = DATA_ROOT / "audit/v2/phase5_kernel_validation_v1"

PHASE4_NS_ROOT = DATA_ROOT / "audit/v2/phase4_namespace_corrected_release_v1"
PHASE3G_R3_ROOT = DATA_ROOT / "audit/v2/phase3g_r3_identity_recovery_v1"
STAGE1_R3_ROOT = DATA_ROOT / "audit/v2/stage1_r3_recovery_reconstruction_v1"
PHASE4_R3_ROOT = DATA_ROOT / "audit/v2/phase4_r3_recovery_promotion_v1"

PHASE4_NS_RELEASE_ID = "P4NSC_20260808_V1_274E41DF"
PHASE3G_R3_RELEASE_ID = "P3GR3_20260808_V1_274E41DF"
STAGE1_R3_RELEASE_ID = "S1R3_20260808_V1_274E41DF"
PHASE4_R3_RELEASE_ID = "P4R3_20260808_V1_274E41DF"
OVERALL_RELEASE_ID = "NSR3_20260808_V1_274E41DF"

PINNED_R2_HASHES = {
    "unresolved_phenotype_identity_candidates.tsv": "b9b4c976d60ce7e3d74fa0c09af7eb43314233770f33a8a21682859f0a0da34c",
    "canonical_gid_panel_coverage.tsv": "0772ae9a521a46c39df31217b102f59c6504af2e01d0d6d15a5b1e524e6b7257",
    "r2_protocol.json": "c35a6e7cde41f93b13d8d3cdef641be79c28705758b98043d14f27661eb1949a",
    "phase3g_r2_build_summary.json": "c402a09a2dcba9a4a6f7ddf7a100dd40b149f8da40b096bbdc2911d814a1f17a",
}

SELECTED_TRAITS = (
    "1000_GRAIN_WEIGHT",
    "ABOVE_GROUND_BIOMASS",
    "DAYS_TO_HEADING",
    "DAYS_TO_MATURITY",
    "GRAIN_YIELD",
    "PLANT_HEIGHT",
    "TEST_WEIGHT",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, *values: Any) -> str:
    payload = "\x1f".join("" if value is None else str(value) for value in values)
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def index_signature(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(values):
        digest.update(f"{index}\t{clean(value)}\n".encode("utf-8"))
    return digest.hexdigest()


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def clean_id(value: Any) -> str:
    return re.sub(r"\.0$", "", clean(value))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: pd.DataFrame | Iterable[dict[str, Any]]) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def q(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def output_manifest(release_root: Path, release_id: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in sorted(p for p in release_root.rglob("*") if p.is_file() and p.name != "output_manifest.tsv"):
        records.append(
            {
                "release_id": release_id,
                "relative_path": path.relative_to(release_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    frame = pd.DataFrame(records)
    write_tsv(release_root / "output_manifest.tsv", frame)
    return frame
