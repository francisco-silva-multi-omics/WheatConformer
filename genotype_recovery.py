from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MISSING_CALLS = {"", "-", ".", "?", "N", "NA", "N/A", "NN", "NULL", "NONE", "0/0"}
IUPAC_TO_BASES = {
    "R": {"A", "G"},
    "Y": {"C", "T"},
    "S": {"C", "G"},
    "W": {"A", "T"},
    "K": {"G", "T"},
    "M": {"A", "C"},
}


def normalize_identifier(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).strip())
    return text


def canonical_gid(value: object) -> str:
    text = normalize_identifier(value)
    match = re.fullmatch(r"(?i)(?:GID)?\s*0*([0-9]+)(?:\.0)?", text)
    return f"GID{match.group(1)}" if match else ""


def load_canonical_catalog(path: Path) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    catalog = pd.read_csv(path, dtype=str)
    gid_col = "canonical_gid" if "canonical_gid" in catalog.columns else "canonical_id"
    if gid_col not in catalog.columns:
        raise SystemExit(f"{path} does not contain canonical_gid or canonical_id")
    catalog = catalog.rename(columns={gid_col: "canonical_gid"})
    catalog["canonical_gid"] = catalog["canonical_gid"].map(canonical_gid)
    catalog = catalog[catalog["canonical_gid"].ne("")].drop_duplicates("canonical_gid")
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in catalog.itertuples(index=False):
        gid = row.canonical_gid
        aliases[gid.upper()].add(gid)
        raw = getattr(row, "raw_identifiers", "")
        try:
            values = json.loads(raw) if raw else {}
        except (TypeError, json.JSONDecodeError):
            values = {}
        for group in values.values():
            for value in group:
                normalized = normalize_identifier(value)
                if normalized:
                    aliases[normalized.upper()].add(gid)
    return catalog.reset_index(drop=True), aliases


def add_mapping(
    lookup: dict[str, set[str]], sample_identifier: object, gid_value: object
) -> None:
    sample = normalize_identifier(sample_identifier)
    gid = canonical_gid(gid_value)
    if sample and gid:
        lookup.setdefault(sample.upper(), set()).add(gid)


def load_explicit_sample_gid_mappings(genotypic_root: Path) -> tuple[dict[str, set[str]], pd.DataFrame]:
    lookup: dict[str, set[str]] = defaultdict(set)
    evidence: list[dict[str, object]] = []

    known_tables = [
        (
            genotypic_root
            / "DArTseq-derived_SNPs_for_wheat_Mexican_landrace_accessions"
            / "Mexican_landrace_samples_for_Germinate.txt",
            "SampleID",
            "GID",
            "tab",
        ),
        (
            genotypic_root
            / "Seeds_of_Discovery_-_MasAgro_Biodiversidad_Wheat_DArTseq-Derived_SNP_Data_Beta_Recall_Results_From_2011-2014"
            / "SampleIDvsGID_45610samples.txt",
            "SampleID",
            "GID",
            "tab",
        ),
    ]
    for path, sample_col, gid_col, delimiter in known_tables:
        if not path.exists():
            continue
        frame = pd.read_csv(path, sep="\t" if delimiter == "tab" else ",", dtype=str)
        for sample, gid_value in frame[[sample_col, gid_col]].dropna().itertuples(index=False, name=None):
            add_mapping(lookup, sample, gid_value)
            evidence.append(
                {
                    "dataset": path.relative_to(genotypic_root).parts[0],
                    "file_path": str(path),
                    "sample_identifier": normalize_identifier(sample),
                    "canonical_gid": canonical_gid(gid_value),
                    "mapping_method": "explicit_sample_gid_sidecar",
                }
            )
    return lookup, pd.DataFrame(evidence)


def marker_alleles(marker_id: str, explicit: str = "") -> tuple[str, str] | None:
    text = normalize_identifier(explicit).upper().replace("|", "/")
    match = re.search(r"([ACGT])\s*[/|>]\s*([ACGT])", text)
    if not match:
        match = re.search(r":([ACGT])>([ACGT])(?:$|[^ACGT])", str(marker_id).upper())
    if not match or match.group(1) == match.group(2):
        return None
    return match.group(1), match.group(2)


def genotype_call_to_dosage(call: object, ref: str, alt: str) -> int:
    raw = normalize_identifier(call).upper()
    if raw in MISSING_CALLS:
        return -1
    compact = raw.replace("/", "").replace("|", "").replace(":", "")
    if compact in {ref, ref * 2}:
        return 0
    if compact in {alt, alt * 2}:
        return 2
    if IUPAC_TO_BASES.get(compact) == {ref, alt}:
        return 1
    bases = set(compact)
    if bases == {ref, alt}:
        return 1
    return -1


def validate_kernel(kernel: np.ndarray, *, name: str, atol: float = 1e-5) -> dict[str, float | int | str]:
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"{name} is not square: {kernel.shape}")
    if not np.isfinite(kernel).all():
        raise ValueError(f"{name} contains non-finite values")
    symmetry = float(np.max(np.abs(kernel - kernel.T))) if kernel.size else 0.0
    if symmetry > atol:
        raise ValueError(f"{name} is not symmetric; max_abs_diff={symmetry}")
    diagonal = np.diag(kernel).astype(np.float64)
    if np.any(diagonal <= 0):
        raise ValueError(f"{name} has non-positive diagonal entries")
    eig_sample = kernel
    if len(kernel) > 1024:
        rng = np.random.default_rng(20260715)
        index = np.sort(rng.choice(len(kernel), size=1024, replace=False))
        eig_sample = kernel[np.ix_(index, index)]
    min_eigenvalue = float(np.linalg.eigvalsh(eig_sample.astype(np.float64)).min())
    if min_eigenvalue < -1e-4:
        raise ValueError(f"{name} is not sampled-PSD; min_eigenvalue={min_eigenvalue}")
    return {
        "kernel": name,
        "dimension": int(len(kernel)),
        "finite": "true",
        "max_abs_symmetry_difference": symmetry,
        "mean_diagonal": float(diagonal.mean()),
        "min_diagonal": float(diagonal.min()),
        "max_diagonal": float(diagonal.max()),
        "sampled_min_eigenvalue": min_eigenvalue,
    }


def rbf_from_linear_kernel(kernel: np.ndarray) -> tuple[np.ndarray, float]:
    diagonal = np.diag(kernel).astype(np.float64)
    distance = diagonal[:, None] + diagonal[None, :] - 2.0 * kernel.astype(np.float64)
    distance = np.maximum(distance, 0.0)
    upper = distance[np.triu_indices(len(distance), k=1)]
    positive = upper[np.isfinite(upper) & (upper > 0)]
    if not len(positive):
        raise ValueError("Cannot construct RBF kernel because all pairwise distances are zero")
    gamma = 1.0 / float(np.median(positive))
    rbf = np.exp(-gamma * distance).astype(np.float32)
    rbf = ((rbf + rbf.T) * 0.5).astype(np.float32)
    np.fill_diagonal(rbf, 1.0)
    return rbf, gamma


def first_unique(values: Iterable[str]) -> str:
    unique = sorted({normalize_identifier(value) for value in values if normalize_identifier(value)})
    return unique[0] if len(unique) == 1 else ""
