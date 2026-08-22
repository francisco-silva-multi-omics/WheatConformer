#!/usr/bin/env python3
"""Independent Phase-5 kernel mathematics.

This module deliberately imports no production kernel builder.  It provides a
small, deterministic reference implementation for analytical tests and sampled
real-data comparisons; it never materializes observation-by-observation kernels.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


UNKNOWN = {"", ".", "0", "NA", "NAN", "NONE", "NULL", "UNKNOWN"}


def clean_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = " ".join(str(value).strip().split())
    return "" if text.upper() in UNKNOWN else text


def additive_relationship(
    records: Iterable[tuple[object, object, object]],
    requested_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    alternatives: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for child_raw, parent1_raw, parent2_raw in records:
        child = clean_id(child_raw)
        if not child:
            continue
        parent1, parent2 = clean_id(parent1_raw), clean_id(parent2_raw)
        if parent1 == child:
            parent1 = ""
        if parent2 == child:
            parent2 = ""
        alternatives[child].add((parent1, parent2))
    conflicts = {key: values for key, values in alternatives.items() if len(values) != 1}
    if conflicts:
        raise ValueError(f"Conflicting pedigree records: {list(conflicts)[:5]}")
    parents = {key: next(iter(values)) for key, values in alternatives.items()}
    nodes = set(parents)
    nodes.update(parent for pair in parents.values() for parent in pair if parent)
    for node in nodes:
        parents.setdefault(node, ("", ""))

    topological: list[str] = []
    unresolved = set(nodes)
    while unresolved:
        ready = sorted(
            node for node in unresolved
            if all(not parent or parent in topological for parent in parents[node])
        )
        if not ready:
            raise ValueError(f"Pedigree cycle: {sorted(unresolved)[:5]}")
        topological.extend(ready)
        unresolved.difference_update(ready)

    index = {node: position for position, node in enumerate(topological)}
    relationship = np.zeros((len(topological), len(topological)), dtype=np.float64)
    for child in topological:
        i = index[child]
        parent1, parent2 = parents[child]
        i1 = index[parent1] if parent1 else None
        i2 = index[parent2] if parent2 else None
        for j in range(i):
            value = 0.5 * (
                (relationship[i1, j] if i1 is not None else 0.0)
                + (relationship[i2, j] if i2 is not None else 0.0)
            )
            relationship[i, j] = relationship[j, i] = value
        relationship[i, i] = 1.0 + (
            0.5 * relationship[i1, i2] if i1 is not None and i2 is not None else 0.0
        )

    if requested_order is not None:
        order = [clean_id(item) for item in requested_order]
        missing = [item for item in order if item not in index]
        if missing:
            raise ValueError(f"Requested pedigree nodes absent: {missing[:5]}")
        selection = np.asarray([index[item] for item in order], dtype=np.int64)
        relationship = relationship[np.ix_(selection, selection)]
        topological = order
    return relationship, topological


def fit_marker_transform(
    marker_matrix: np.ndarray, fit_rows: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    matrix = np.asarray(marker_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Marker matrix must be nonempty samples-by-markers")
    finite_or_nan = np.isfinite(matrix) | np.isnan(matrix)
    if not finite_or_nan.all() or np.any(matrix[np.isfinite(matrix)] < 0) or np.any(matrix[np.isfinite(matrix)] > 2):
        raise ValueError("Finite marker dosages must lie in [0, 2]")
    fit = np.asarray(sorted(set(int(i) for i in fit_rows)), dtype=np.int64)
    if fit.size < 2 or fit.min(initial=0) < 0 or fit.max(initial=-1) >= len(matrix):
        raise ValueError("At least two valid training rows are required")
    train = matrix[fit]
    finite_counts = np.isfinite(train).sum(axis=0)
    keep = finite_counts > 0
    if not keep.any():
        raise ValueError("No marker has a training-finite call")
    train = train[:, keep]
    transformed = matrix[:, keep].copy()
    means = np.nanmean(train, axis=0)
    missing = ~np.isfinite(transformed)
    transformed[missing] = means[np.where(missing)[1]]
    p = np.nanmean(train, axis=0) / 2.0
    dosage_std = np.nanstd(train, axis=0, ddof=0)
    variable = np.isfinite(p) & (p > 0) & (p < 1) & np.isfinite(dosage_std) & (dosage_std > 0)
    transformed, p = transformed[:, variable], p[variable]
    keep_indices = np.flatnonzero(keep)[variable]
    denominator = float(np.sum(2.0 * p * (1.0 - p)))
    if transformed.shape[1] == 0 or not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Training marker denominator is not positive")
    return transformed, p, keep_indices, denominator


def vanraden(
    marker_matrix: np.ndarray, fit_rows: Sequence[int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    matrix = np.asarray(marker_matrix, dtype=np.float64)
    fit = np.arange(len(matrix), dtype=np.int64) if fit_rows is None else np.asarray(fit_rows)
    transformed, p, kept, denominator = fit_marker_transform(matrix, fit)
    centered = transformed - 2.0 * p
    kernel = centered @ centered.T / denominator
    return kernel, p, kept, denominator


def environment_linear_kernel(
    feature_matrix: np.ndarray, fit_rows: Sequence[int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Environment matrix must be nonempty environments-by-features")
    fit = np.arange(len(matrix), dtype=np.int64) if fit_rows is None else np.asarray(fit_rows, dtype=np.int64)
    if fit.size < 2:
        raise ValueError("At least two training environments are required")
    reference = matrix[fit]
    mean = np.nanmean(reference, axis=0)
    filled_reference = np.where(np.isfinite(reference), reference, mean)
    std = filled_reference.std(axis=0, ddof=1)
    keep = np.isfinite(mean) & np.isfinite(std) & (std > 0)
    if not keep.any():
        raise ValueError("No variable environment feature in training scope")
    mean, std = mean[keep], std[keep]
    transformed = matrix[:, keep]
    transformed = np.where(np.isfinite(transformed), transformed, mean)
    standardized = (transformed - mean) / std
    kernel = standardized @ standardized.T / standardized.shape[1]
    training_mean_diagonal = float(np.mean(np.diag(kernel)[fit]))
    if not np.isfinite(training_mean_diagonal) or training_mean_diagonal <= 0:
        raise ValueError("Invalid environment-kernel diagonal scale")
    return kernel / training_mean_diagonal, standardized, mean, std


def gxe_elements(
    genotype_kernel: np.ndarray,
    environment_kernel: np.ndarray,
    genotype_indices: Sequence[int],
    environment_indices: Sequence[int],
    pairs: Iterable[tuple[int, int]],
) -> list[float]:
    gi = np.asarray(genotype_indices, dtype=np.int64)
    ei = np.asarray(environment_indices, dtype=np.int64)
    if gi.shape != ei.shape:
        raise ValueError("Genotype and environment observation indices are misaligned")
    output = []
    for i, j in pairs:
        output.append(float(genotype_kernel[gi[i], gi[j]] * environment_kernel[ei[i], ei[j]]))
    return output


def index_signature(values: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(values):
        digest.update(f"{index}\t{clean_id(value)}\n".encode("utf-8"))
    return digest.hexdigest()


def assert_same_index(expected: Sequence[object], observed: Sequence[object], label: str) -> None:
    left = [clean_id(value) for value in expected]
    right = [clean_id(value) for value in observed]
    if left != right:
        mismatch = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None)
        raise ValueError(f"{label} index mismatch at {mismatch}; lengths={len(left)},{len(right)}")


def assert_many_to_one(left: pd.DataFrame, right: pd.DataFrame, key: str) -> None:
    if right[key].isna().any() or right[key].astype(str).str.strip().eq("").any():
        raise ValueError(f"Right-side {key} contains null/empty keys")
    if right[key].duplicated().any():
        raise ValueError(f"Prohibited one-to-many or many-to-many join on {key}")
    left.merge(right, on=key, how="left", validate="many_to_one")


def synthetic_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pedigree = [
        ("A", "", ""), ("B", "", ""), ("C", "A", "B"),
        ("D", "C", "A"), ("E", "C", "C"), ("F", "D", ""),
    ]
    ka, order = additive_relationship(pedigree, ["A", "B", "C", "D", "E", "F"])
    expected = np.asarray([
        [1, 0, .5, .75, .5, .375],
        [0, 1, .5, .25, .5, .125],
        [.5, .5, 1, .75, 1, .375],
        [.75, .25, .75, 1.25, .75, .625],
        [.5, .5, 1, .75, 1.5, .375],
        [.375, .125, .375, .625, .375, 1],
    ])
    rows.append({"test": "synthetic_K_A", "max_abs_error": float(np.max(np.abs(ka - expected))), "status": "PASS" if np.allclose(ka, expected) else "FAIL"})

    markers = np.asarray([[0, 0, 1, np.nan, 1], [0, 0, 1, 1, 1], [2, 2, 1, 2, 1], [1, 1, 1, 1, 1]], dtype=float)
    kg, _, kept, _ = vanraden(markers)
    rows.append({"test": "synthetic_K_G", "max_abs_error": float(np.max(np.abs(kg - kg.T))), "status": "PASS" if kept.size == 3 and np.isfinite(kg).all() else "FAIL"})

    environment = np.asarray([[1, 2, 7], [1, 2, 7], [9, np.nan, 7]], dtype=float)
    ke, _, _, _ = environment_linear_kernel(environment)
    rows.append({"test": "synthetic_K_E", "max_abs_error": float(abs(ke[0, 1] - ke[0, 0])), "status": "PASS" if np.allclose(ke[0], ke[1]) else "FAIL"})

    pairs = [(0, 0), (0, 1), (1, 2)]
    gxe = gxe_elements(np.asarray([[1, .2], [.2, 1.]]), np.asarray([[1, .4], [.4, 1.]]), [0, 0, 1], [0, 1, 0], pairs)
    expected_gxe = [1.0, .4, .08]
    rows.append({"test": "synthetic_GxE", "max_abs_error": float(np.max(np.abs(np.asarray(gxe) - expected_gxe))), "status": "PASS" if np.allclose(gxe, expected_gxe) else "FAIL"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = synthetic_results()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, sep="\t", index=False)
    if not result["status"].eq("PASS").all():
        raise SystemExit("Independent synthetic reconstruction failed")


if __name__ == "__main__":
    main()
