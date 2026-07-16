from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NULL_IDS = {"", "NA", "NAN", "NONE", "NULL", ".", "-", "UNKNOWN", "0"}


def normalize_identifier(value: object, *, gid_prefix: bool = False) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).strip())
    if text.upper() in NULL_IDS:
        return ""
    text = re.sub(r"\.0$", "", text) if re.fullmatch(r"[0-9]+\.0", text) else text
    if gid_prefix and re.fullmatch(r"(?i)GID[0-9]+", text):
        return "GID" + text[3:]
    return text


def canonical_gid(value: object) -> str:
    text = normalize_identifier(value, gid_prefix=True)
    if not text:
        return ""
    if re.fullmatch(r"[0-9]+", text):
        return f"GID{text}"
    return text if text.upper().startswith("GID") else ""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def assert_unique(values: pd.Series, label: str, *, allow_empty: bool = False) -> None:
    normalized = values.fillna("").astype(str).str.strip()
    if not allow_empty and normalized.eq("").any():
        raise AssertionError(f"{label} contains empty identifiers")
    duplicated = normalized[normalized.ne("") & normalized.duplicated(keep=False)]
    if not duplicated.empty:
        raise AssertionError(f"{label} contains duplicate identifiers: {duplicated.head(5).tolist()}")


def join_cardinality(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> dict[str, int | str]:
    left_counts = left.groupby(keys, dropna=False).size().rename("left_n")
    right_counts = right.groupby(keys, dropna=False).size().rename("right_n")
    pairs = left_counts.to_frame().join(right_counts, how="outer").fillna(0).astype(np.int64)
    matched = pairs[(pairs.left_n > 0) & (pairs.right_n > 0)]
    expanded = int((matched.left_n * matched.right_n).sum())
    left_matched_rows = int(matched.left_n.sum())
    return {
        "keys": ";".join(keys),
        "left_rows": len(left),
        "right_rows": len(right),
        "left_unique_keys": len(left_counts),
        "right_unique_keys": len(right_counts),
        "unmatched_left_keys": int(((pairs.left_n > 0) & (pairs.right_n == 0)).sum()),
        "unmatched_right_keys": int(((pairs.right_n > 0) & (pairs.left_n == 0)).sum()),
        "one_to_one_keys": int(((matched.left_n == 1) & (matched.right_n == 1)).sum()),
        "one_to_many_keys": int(((matched.left_n == 1) & (matched.right_n > 1)).sum()),
        "many_to_one_keys": int(((matched.left_n > 1) & (matched.right_n == 1)).sum()),
        "many_to_many_keys": int(((matched.left_n > 1) & (matched.right_n > 1)).sum()),
        "left_rows_matched": left_matched_rows,
        "joined_rows_expected": expanded,
        "rows_duplicated_by_join": max(0, expanded - left_matched_rows),
    }


def independent_vanraden(marker_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    matrix = np.asarray(marker_matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Marker matrix must be a finite samples-by-markers matrix")
    if np.any((matrix < 0) | (matrix > 2)):
        raise ValueError("Marker dosages must lie in [0, 2]")
    allele_frequency = matrix.mean(axis=0) / 2.0
    denominator = float(np.sum(2.0 * allele_frequency * (1.0 - allele_frequency)))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("VanRaden denominator must be finite and positive")
    centered = matrix - 2.0 * allele_frequency
    kernel = centered @ centered.T / denominator
    return kernel, allele_frequency, denominator


def mean_impute_markers(marker_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(marker_matrix, dtype=np.float64).copy()
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("At least one marker has no finite genotype calls")
    means = np.where(finite, matrix, 0.0).sum(axis=0) / counts
    missing_rows, missing_cols = np.where(~finite)
    matrix[missing_rows, missing_cols] = means[missing_cols]
    return matrix, means


def independent_additive_relationship(
    records: Iterable[tuple[str, str, str]], requested_order: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    parent_map: dict[str, tuple[str, str]] = {}
    conflicts: dict[str, set[tuple[str, str]]] = {}
    for child_raw, p1_raw, p2_raw in records:
        child = normalize_identifier(child_raw)
        p1 = normalize_identifier(p1_raw)
        p2 = normalize_identifier(p2_raw)
        if not child:
            continue
        parents = ("" if p1 == child else p1, "" if p2 == child else p2)
        conflicts.setdefault(child, set()).add(parents)
    conflicting = {child: values for child, values in conflicts.items() if len(values) > 1}
    if conflicting:
        examples = {key: sorted(value) for key, value in list(conflicting.items())[:5]}
        raise ValueError(f"Conflicting pedigree records: {examples}")
    parent_map = {child: next(iter(values)) for child, values in conflicts.items()}
    all_ids = set(parent_map)
    all_ids.update(parent for parents in parent_map.values() for parent in parents if parent)
    for identifier in all_ids:
        parent_map.setdefault(identifier, ("", ""))

    resolved: list[str] = []
    unresolved = set(parent_map)
    while unresolved:
        ready = sorted(
            child
            for child in unresolved
            if all(not parent or parent in resolved for parent in parent_map[child])
        )
        if not ready:
            raise ValueError(f"Pedigree contains a cycle involving {sorted(unresolved)[:5]}")
        for child in ready:
            resolved.append(child)
            unresolved.remove(child)

    index = {identifier: position for position, identifier in enumerate(resolved)}
    matrix = np.zeros((len(resolved), len(resolved)), dtype=np.float64)
    for child in resolved:
        i = index[child]
        p1, p2 = parent_map[child]
        i1 = index[p1] if p1 else None
        i2 = index[p2] if p2 else None
        for j in range(i):
            matrix[i, j] = matrix[j, i] = 0.5 * (
                (matrix[i1, j] if i1 is not None else 0.0)
                + (matrix[i2, j] if i2 is not None else 0.0)
            )
        matrix[i, i] = 1.0 + (0.5 * matrix[i1, i2] if i1 is not None and i2 is not None else 0.0)

    if requested_order is not None:
        missing = [identifier for identifier in requested_order if identifier not in index]
        if missing:
            raise ValueError(f"Requested pedigree IDs are absent: {missing[:5]}")
        selection = [index[identifier] for identifier in requested_order]
        matrix = matrix[np.ix_(selection, selection)]
        resolved = list(requested_order)
    return matrix, resolved


def independent_environment_kernel(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.array(features, dtype=np.float64, copy=True)
    if matrix.ndim != 2:
        raise ValueError("Environment features must be two-dimensional")
    matrix[~np.isfinite(matrix)] = np.nan
    finite_counts = np.sum(np.isfinite(matrix), axis=0)
    finite_column_mask = finite_counts > 0
    matrix = matrix[:, finite_column_mask]
    if matrix.shape[1] == 0:
        raise ValueError("Environment feature matrix has no finite columns")
    means = np.nanmean(matrix, axis=0)
    missing = ~np.isfinite(matrix)
    matrix[missing] = np.take(means, np.where(missing)[1])
    std = matrix.std(axis=0, ddof=0)
    variable_column_mask = np.isfinite(std) & (std > 0)
    matrix, means, std = (
        matrix[:, variable_column_mask],
        means[variable_column_mask],
        std[variable_column_mask],
    )
    if matrix.shape[1] == 0:
        raise ValueError("Environment feature matrix has no variable finite columns")
    standardized = (matrix - means) / std
    kernel = standardized @ standardized.T / standardized.shape[1]
    mean_diagonal = float(np.mean(np.diag(kernel)))
    if not np.isfinite(mean_diagonal) or mean_diagonal <= 0:
        raise ValueError("Environment kernel has invalid mean diagonal")
    retained = np.zeros(features.shape[1], dtype=bool)
    retained[np.flatnonzero(finite_column_mask)[variable_column_mask]] = True
    return kernel / mean_diagonal, standardized, retained


def independent_observation_gxe(
    genotype_kernel: np.ndarray,
    environment_kernel: np.ndarray,
    genotype_indices: np.ndarray,
    environment_indices: np.ndarray,
) -> np.ndarray:
    genotype_indices = np.asarray(genotype_indices, dtype=np.int64)
    environment_indices = np.asarray(environment_indices, dtype=np.int64)
    if genotype_indices.shape != environment_indices.shape:
        raise ValueError("Genotype and environment observation indices must align")
    genotype_observation = np.asarray(genotype_kernel)[np.ix_(genotype_indices, genotype_indices)]
    environment_observation = np.asarray(environment_kernel)[
        np.ix_(environment_indices, environment_indices)
    ]
    return genotype_observation * environment_observation


def sampled_kernel_diagnostics(
    path: Path,
    *,
    order_path: Path | None = None,
    sample_size: int = 512,
    seed: int = 20260715,
) -> dict[str, object]:
    kernel = np.load(path, mmap_mode="r")
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        return {"path": str(path), "shape": str(kernel.shape), "status": "not_square"}
    n = kernel.shape[0]
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(n, size=min(n, sample_size), replace=False))
    block = np.asarray(kernel[np.ix_(selected, selected)], dtype=np.float64)
    diagonal = np.asarray(kernel.diagonal(), dtype=np.float64)
    finite = True
    symmetry_error = 0.0
    for start in range(0, n, 256):
        rows = np.asarray(kernel[start : min(start + 256, n)], dtype=np.float64)
        finite = finite and bool(np.isfinite(rows).all())
        cols = np.asarray(kernel[:, start : min(start + 256, n)], dtype=np.float64).T
        symmetry_error = max(symmetry_error, float(np.max(np.abs(rows - cols))))
    eigenvalues = np.linalg.eigvalsh((block + block.T) / 2.0)
    positive = eigenvalues[eigenvalues > max(1e-12, np.max(eigenvalues) * 1e-10)]
    effective_rank = (
        float(np.square(np.sum(np.maximum(eigenvalues, 0))) / np.sum(np.square(np.maximum(eigenvalues, 0))))
        if np.any(eigenvalues > 0)
        else 0.0
    )
    order_rows = None
    order_unique = None
    if order_path and order_path.exists():
        order = pd.read_csv(order_path, sep="\t", dtype=str)
        order_rows = len(order)
        order_unique = int(order.iloc[:, 0].nunique(dropna=False))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "shape": f"{n}x{n}",
        "dtype": str(kernel.dtype),
        "all_finite": finite,
        "symmetry_max_abs": symmetry_error,
        "diagonal_mean": float(np.mean(diagonal)),
        "diagonal_min": float(np.min(diagonal)),
        "diagonal_max": float(np.max(diagonal)),
        "trace": float(np.sum(diagonal)),
        "sampled_n": len(selected),
        "sampled_min_eigenvalue": float(np.min(eigenvalues)),
        "sampled_material_negative_count": int(np.sum(eigenvalues < -1e-6)),
        "sampled_effective_rank": effective_rank,
        "sampled_condition_number": float(np.max(positive) / np.min(positive)) if len(positive) else math.inf,
        "order_path": str(order_path) if order_path else "",
        "order_rows": order_rows,
        "order_unique": order_unique,
        "dimension_matches_order": order_rows == n if order_rows is not None else None,
        "status": "PASS" if finite and symmetry_error <= 1e-5 and (order_rows in {None, n}) else "FAIL",
    }


def deterministic_signature(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    payload = frame[columns].fillna("").astype(str).agg("\x1f".join, axis=1)
    return payload.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())


def relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def iter_source_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and not any(part in {".git", ".audit-venv", "audit"} for part in path.parts):
            yield path
