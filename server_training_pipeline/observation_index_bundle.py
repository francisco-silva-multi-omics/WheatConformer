from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


BUNDLE_COLUMNS = {
    "geno_kernel_index": ("geno_kernel_index", np.int32, False),
    "env_kernel_index": ("env_kernel_index", np.int32, False),
    "y": ("phenotype_value", np.float32, False),
    "weight": ("weight_g_e", np.float32, True),
    "var": ("var_g_e", np.float32, True),
    "se": ("SE_g_e", np.float32, True),
}


def observation_index_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    missing = sorted(
        source_col
        for source_col, _dtype, _allow_nan in BUNDLE_COLUMNS.values()
        if source_col not in frame.columns
    )
    if missing:
        raise ValueError(f"Observation ledger is missing NPZ source columns: {missing}")

    arrays: dict[str, np.ndarray] = {}
    for output_name, (source_col, dtype, allow_nan) in BUNDLE_COLUMNS.items():
        values = pd.to_numeric(frame[source_col], errors="coerce").to_numpy(dtype=np.float64)
        if np.isinf(values).any() or (not allow_nan and np.isnan(values).any()):
            raise ValueError(f"Observation ledger column {source_col} contains unsupported non-finite values")
        if np.issubdtype(dtype, np.integer):
            if not np.equal(values, np.floor(values)).all() or np.any(values < 0):
                raise ValueError(
                    f"Observation ledger index column {source_col} must contain nonnegative integers"
                )
        arrays[output_name] = values.astype(dtype, copy=False)
    return arrays


def write_observation_index_bundle(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    arrays = observation_index_arrays(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "rows": len(frame),
        "keys": sorted(arrays),
        "bytes": path.stat().st_size,
    }


def compare_observation_index_bundle(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    expected = observation_index_arrays(frame)
    with np.load(path) as bundle:
        schema_match = set(expected).issubset(bundle.files)
        rows_match = bool(
            schema_match and all(len(bundle[key]) == len(frame) for key in expected)
        )
        values_match = bool(
            rows_match
            and all(np.array_equal(bundle[key], expected[key], equal_nan=True) for key in expected)
        )
    return {
        "schema_match": schema_match,
        "rows_match": rows_match,
        "values_match": values_match,
    }
