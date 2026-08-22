from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


RELEASE_ID = "P5SBK_20260808_V1_274E41DF"
SEED = 20260808
SCENARIOS = ("GNEW_EOBS", "GOBS_ENEW", "GNEW_ENEW")
OUTER_FOLDS = tuple(range(1, 6))
INNER_FOLDS = tuple(range(1, 6))

SPLIT_ALLOWED_COLUMNS = (
    "phase4_adjusted_row_id",
    "phase4_group_id",
    "trial_id",
    "cycle",
    "environment_id",
    "year",
    "trait",
    "typed_source_genotype_id",
    "canonical_gid",
    "identity_status",
    "canonical_gid_eligible",
    "primary_weighted_training_eligible",
    "secondary_unweighted_training_eligible",
    "continuous_error_evaluation_eligible",
    "correlation_evaluation_eligible",
    "ranking_evaluation_eligible",
    "phenotype_release_eligible",
    "loc_no",
    "country",
    "loc_desc",
    "standardized_unit",
    "namespace_correction_status",
)

PROHIBITED_SPLIT_COLUMNS = frozenset(
    {
        "adjusted_value",
        "raw_unadjusted_mean",
        "pev_proxy",
        "reliability",
        "reliability_weight",
        "raw_precision_weight",
        "entry_mean_heritability",
        "plot_repeatability",
        "ranking_ceiling",
        "robust_adjusted_blue",
        "adjusted_blup",
        "estimated_genetic_variance",
        "check_status",
        "huber_status",
    }
)

MISSING_PARENT_CODES = frozenset(
    {"", "0", "-", ".", "NA", "N/A", "NAN", "NONE", "NULL", "UNKNOWN", "UNK"}
)

IUPAC = {
    "A": frozenset("A"),
    "C": frozenset("C"),
    "G": frozenset("G"),
    "T": frozenset("T"),
    "R": frozenset("AG"),
    "Y": frozenset("CT"),
    "S": frozenset("CG"),
    "W": frozenset("AT"),
    "K": frozenset("GT"),
    "M": frozenset("AC"),
}


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def index_signature(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_tsv(path: Path, value: pd.DataFrame | Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(list(value))
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def canonical_gid(value: object) -> str:
    text = clean_text(value).upper()
    if re.fullmatch(r"GID\d+", text):
        return text
    if re.fullmatch(r"\d+", text):
        return f"GID{text}"
    return ""


def deterministic_tie(seed: int, salt: str, entity: str, fold: int | None = None) -> str:
    suffix = "" if fold is None else f"|{fold}"
    return sha256_text(f"{seed}|{salt}|{entity}{suffix}")


def assign_balanced_entities(
    summary: pd.DataFrame,
    entity_col: str,
    scenario: str,
    seed: int = SEED,
) -> pd.DataFrame:
    """Freeze primary assignments, then add secondary-only entities without reassignment."""
    required = {entity_col, "primary_rows", "secondary_rows"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Entity summary lacks {sorted(missing)}")
    work = summary[list(required)].copy()
    work[entity_col] = work[entity_col].astype(str)
    work["primary_rows"] = work["primary_rows"].fillna(0).astype("int64")
    work["secondary_rows"] = work["secondary_rows"].fillna(0).astype("int64")
    if work[entity_col].duplicated().any():
        raise ValueError(f"Duplicate entities in {entity_col} summary")

    assignments: list[dict[str, Any]] = []
    primary_totals = {fold: 0 for fold in OUTER_FOLDS}
    secondary_totals = {fold: 0 for fold in OUTER_FOLDS}
    entity_totals = {fold: 0 for fold in OUTER_FOLDS}

    primary = work[work["primary_rows"] > 0].copy()
    primary["tie"] = primary[entity_col].map(
        lambda entity: deterministic_tie(seed, f"{scenario}|PRIMARY", entity)
    )
    primary = primary.sort_values(["primary_rows", "tie", entity_col], ascending=[False, True, True])
    for row in primary.itertuples(index=False):
        entity = str(getattr(row, entity_col))
        fold = min(
            OUTER_FOLDS,
            key=lambda candidate: (
                primary_totals[candidate],
                entity_totals[candidate],
                deterministic_tie(seed, f"{scenario}|PRIMARY_FOLD", entity, candidate),
            ),
        )
        primary_rows = int(row.primary_rows)
        secondary_rows = int(row.secondary_rows)
        primary_totals[fold] += primary_rows
        secondary_totals[fold] += secondary_rows
        entity_totals[fold] += 1
        assignments.append(
            {
                "entity_id": entity,
                "assigned_fold": fold,
                "assignment_stage": "PRIMARY_FROZEN",
                "primary_rows": primary_rows,
                "secondary_rows": secondary_rows,
            }
        )

    secondary_only = work[(work["primary_rows"] == 0) & (work["secondary_rows"] > 0)].copy()
    secondary_only["tie"] = secondary_only[entity_col].map(
        lambda entity: deterministic_tie(seed, f"{scenario}|SECONDARY_ONLY", entity)
    )
    secondary_only = secondary_only.sort_values(
        ["secondary_rows", "tie", entity_col], ascending=[False, True, True]
    )
    for row in secondary_only.itertuples(index=False):
        entity = str(getattr(row, entity_col))
        fold = min(
            OUTER_FOLDS,
            key=lambda candidate: (
                secondary_totals[candidate],
                entity_totals[candidate],
                deterministic_tie(seed, f"{scenario}|SECONDARY_FOLD", entity, candidate),
            ),
        )
        secondary_rows = int(row.secondary_rows)
        secondary_totals[fold] += secondary_rows
        entity_totals[fold] += 1
        assignments.append(
            {
                "entity_id": entity,
                "assigned_fold": fold,
                "assignment_stage": "SECONDARY_ONLY_AFTER_PRIMARY_FREEZE",
                "primary_rows": 0,
                "secondary_rows": secondary_rows,
            }
        )
    result = pd.DataFrame(assignments).sort_values("entity_id").reset_index(drop=True)
    if len(result) != len(work[work["secondary_rows"] > 0]):
        raise AssertionError("Not every secondary entity received a fold")
    return result


def outer_role(
    scenario: str,
    fold: int,
    gid_fold: int,
    env_fold: int,
    other_seen_in_training: bool = True,
) -> str:
    if scenario == "GNEW_EOBS":
        if gid_fold != fold:
            return "TRAIN"
        return "TEST" if other_seen_in_training else "EMBARGO_OTHER_ENTITY_UNSEEN"
    if scenario == "GOBS_ENEW":
        if env_fold != fold:
            return "TRAIN"
        return "TEST" if other_seen_in_training else "EMBARGO_OTHER_ENTITY_UNSEEN"
    if scenario == "GNEW_ENEW":
        gid_new = gid_fold == fold
        env_new = env_fold == fold
        if gid_new and env_new:
            return "TEST"
        if not gid_new and not env_new:
            return "TRAIN"
        return "EMBARGO_SINGLE_NOVELTY"
    raise ValueError(f"Unknown scenario {scenario}")


def normalize_pedigree(text: object) -> str:
    value = clean_text(text).upper().replace("\\", "/")
    value = re.sub(r"\s+[Xx]\s+", "/", value)
    value = re.sub(r"\s*/\s*", "/", value)
    return value if value not in MISSING_PARENT_CODES else ""


def _node_id(kind: str, label: str) -> str:
    return f"{kind}_{sha256_text(label)[:24]}"


def parse_purdy_expression(
    expression: str,
    nodes: dict[str, dict[str, str]],
) -> str:
    """Parse exact Purdy-style slash levels without mapping names to GIDs."""
    expression = normalize_pedigree(expression)
    if not expression:
        return ""
    matches = list(re.finditer(r"/+", expression))
    if not matches:
        node = _node_id("PEDLEAF", expression)
        nodes.setdefault(node, {"node_kind": "PEDIGREE_LEAF", "label": expression, "parent1": "", "parent2": ""})
        return node
    max_width = max(len(match.group(0)) for match in matches)
    candidates = [match for match in matches if len(match.group(0)) == max_width]
    split = candidates[-1]
    left = expression[: split.start()]
    right = expression[split.end() :]
    if not left or not right:
        return ""
    p1 = parse_purdy_expression(left, nodes)
    p2 = parse_purdy_expression(right, nodes)
    if not p1 or not p2:
        return ""
    p1, p2 = sorted((p1, p2))
    label = f"{p1}|{p2}"
    node = _node_id("PEDCROSS", label)
    nodes.setdefault(node, {"node_kind": "PEDIGREE_CROSS", "label": expression, "parent1": p1, "parent2": p2})
    return node


def build_pedigree_parent_map(
    manifest: pd.DataFrame,
    accepted_gids: set[str],
) -> tuple[dict[str, tuple[str, str]], pd.DataFrame, pd.DataFrame]:
    required = {"resolved_gid", "cross_name"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Pedigree source lacks {sorted(required - set(manifest.columns))}")
    source = manifest.copy()
    source["canonical_gid"] = source["resolved_gid"].map(canonical_gid)
    source["normalized_cross_name"] = source["cross_name"].map(normalize_pedigree)
    source = source[source["canonical_gid"].isin(accepted_gids)]
    source = source[source["normalized_cross_name"] != ""]

    parent_map: dict[str, tuple[str, str]] = {}
    nodes: dict[str, dict[str, str]] = {}
    ledger: list[dict[str, Any]] = []
    for gid in sorted(accepted_gids):
        gid_rows = source[source["canonical_gid"] == gid]
        crosses = sorted(set(gid_rows["normalized_cross_name"]))
        if not crosses:
            ledger.append({"canonical_gid": gid, "disposition": "NO_PEDIGREE_SOURCE", "candidate_cross_count": 0, "normalized_cross_names": ""})
            continue
        if len(crosses) > 1:
            ledger.append({"canonical_gid": gid, "disposition": "CONFLICTING_PEDIGREE_STRINGS", "candidate_cross_count": len(crosses), "normalized_cross_names": ";".join(crosses)})
            continue
        root = parse_purdy_expression(crosses[0], nodes)
        if not root or nodes[root]["node_kind"] != "PEDIGREE_CROSS":
            ledger.append({"canonical_gid": gid, "disposition": "UNPARSEABLE_OR_SINGLE_LINE_PEDIGREE", "candidate_cross_count": 1, "normalized_cross_names": crosses[0]})
            continue
        p1, p2 = nodes[root]["parent1"], nodes[root]["parent2"]
        parent_map[gid] = tuple(sorted((p1, p2)))
        ledger.append({"canonical_gid": gid, "disposition": "ACCEPTED_EXACT_UNIQUE_PEDIGREE", "candidate_cross_count": 1, "normalized_cross_names": crosses[0]})

    for node_id, value in nodes.items():
        parent_map.setdefault(node_id, (value["parent1"], value["parent2"]))
    ledger_frame = pd.DataFrame(ledger).sort_values("canonical_gid").reset_index(drop=True)
    nodes_frame = pd.DataFrame(
        [
            {
                "node_id": node_id,
                **value,
                "is_observed_gid": False,
            }
            for node_id, value in sorted(nodes.items())
        ]
        + [
            {
                "node_id": gid,
                "node_kind": "OBSERVED_CANONICAL_GID",
                "label": gid,
                "parent1": parent_map[gid][0],
                "parent2": parent_map[gid][1],
                "is_observed_gid": True,
            }
            for gid in sorted(parent_map)
            if gid.startswith("GID")
        ]
    )
    return parent_map, nodes_frame, ledger_frame


def topological_order(parent_map: Mapping[str, tuple[str, str]]) -> list[str]:
    resolved: list[str] = []
    resolved_set: set[str] = set()
    unresolved = set(parent_map)
    while unresolved:
        available = []
        for node in unresolved:
            p1, p2 = parent_map[node]
            if (not p1 or p1 in resolved_set) and (not p2 or p2 in resolved_set):
                available.append(node)
        if not available:
            raise ValueError(f"Pedigree cycle/unresolved dependency: {sorted(unresolved)[:10]}")
        for node in sorted(available):
            resolved.append(node)
            resolved_set.add(node)
            unresolved.remove(node)
    return resolved


def sparse_weighted_dot(
    left: Mapping[int, float],
    right: Mapping[int, float],
    d_values: Sequence[float],
) -> float:
    if len(left) > len(right):
        left, right = right, left
    return float(sum(value * right.get(index, 0.0) * d_values[index] for index, value in left.items()))


def build_pedigree_factor(
    parent_map: Mapping[str, tuple[str, str]],
) -> tuple[sparse.csr_matrix, np.ndarray, list[str], np.ndarray]:
    order = topological_order(parent_map)
    index = {node: i for i, node in enumerate(order)}
    coefficient_rows: list[dict[int, float]] = []
    d_values: list[float] = []
    diagonals: list[float] = []
    for i, node in enumerate(order):
        p1, p2 = parent_map[node]
        parent_indices = [index[parent] for parent in (p1, p2) if parent]
        coefficients: defaultdict[int, float] = defaultdict(float)
        coefficients[i] = 1.0
        for parent_index in parent_indices:
            for key, value in coefficient_rows[parent_index].items():
                coefficients[key] += 0.5 * value
        if len(parent_indices) == 0:
            d_value = 1.0
        elif len(parent_indices) == 1:
            parent_diag = diagonals[parent_indices[0]]
            d_value = 1.0 - 0.25 * parent_diag
        else:
            d_value = 1.0 - 0.25 * diagonals[parent_indices[0]] - 0.25 * diagonals[parent_indices[1]]
        if d_value < -1e-10:
            raise ValueError(f"Negative Mendelian variance for {node}: {d_value}")
        d_value = max(d_value, 0.0)
        d_values.append(d_value)
        coefficient_rows.append(dict(coefficients))
        diagonals.append(sparse_weighted_dot(coefficients, coefficients, d_values))
    row_index: list[int] = []
    col_index: list[int] = []
    values: list[float] = []
    for row, coefficients in enumerate(coefficient_rows):
        for col, value in sorted(coefficients.items()):
            row_index.append(row)
            col_index.append(col)
            values.append(value)
    factor = sparse.csr_matrix((values, (row_index, col_index)), shape=(len(order), len(order)), dtype=np.float64)
    return factor, np.asarray(d_values, dtype=np.float64), order, np.asarray(diagonals, dtype=np.float64)


def relationship_element(
    factor: sparse.csr_matrix,
    d_values: np.ndarray,
    left_index: int,
    right_index: int,
) -> float:
    left = factor.getrow(left_index)
    right = factor.getrow(right_index)
    left_map = dict(zip(left.indices.tolist(), left.data.tolist()))
    right_map = dict(zip(right.indices.tolist(), right.data.tolist()))
    return sparse_weighted_dot(left_map, right_map, d_values)


def relationship_block(
    factor: sparse.csr_matrix,
    d_values: np.ndarray,
    indices: Sequence[int],
) -> np.ndarray:
    block_factor = factor[np.asarray(indices, dtype=int), :]
    weighted = block_factor.multiply(np.sqrt(d_values))
    return np.asarray((weighted @ weighted.T).toarray(), dtype=np.float64)


def decode_biallelic_call(call: object, alleles: object) -> float:
    allele_text = clean_text(alleles).upper()
    parts = [part for part in re.split(r"[/|]", allele_text) if part]
    if len(parts) != 2 or any(part not in {"A", "C", "G", "T"} for part in parts) or parts[0] == parts[1]:
        return math.nan
    token = clean_text(call).upper()
    if token in {"", "N", "-", ".", "NA", "N/A", "?"}:
        return math.nan
    observed = IUPAC.get(token)
    if observed is None or not observed.issubset(set(parts)):
        return math.nan
    if observed == frozenset(parts[0]):
        return 0.0
    if observed == frozenset(parts[1]):
        return 2.0
    if observed == frozenset(parts):
        return 1.0
    return math.nan


def consensus_dosage(sample_matrix: np.ndarray) -> tuple[np.ndarray, int]:
    if sample_matrix.ndim != 2:
        raise ValueError("Replicate sample matrix must be two-dimensional")
    if sample_matrix.shape[0] == 1:
        return sample_matrix[0].copy(), 0
    result = np.full(sample_matrix.shape[1], np.nan, dtype=np.float64)
    conflicts = 0
    for marker in range(sample_matrix.shape[1]):
        values = sample_matrix[:, marker]
        nonmissing = values[np.isfinite(values)]
        unique = np.unique(nonmissing)
        if len(unique) == 1:
            result[marker] = unique[0]
        elif len(unique) > 1:
            conflicts += 1
    return result, conflicts


def fit_vanraden(
    dosage: np.ndarray,
    entity_ids: Sequence[str],
    training_ids: set[str],
) -> dict[str, Any]:
    ids = [str(value) for value in entity_ids]
    training_index = np.asarray([i for i, value in enumerate(ids) if value in training_ids], dtype=int)
    if len(training_index) < 2:
        raise ValueError("At least two training genotypes are required for VanRaden fitting")
    training = dosage[training_index, :]
    with np.errstate(invalid="ignore"):
        p = np.nanmean(training, axis=0) / 2.0
        training_variance = np.nanvar(training, axis=0)
    retained = np.isfinite(p) & np.isfinite(training_variance) & (training_variance > 0.0) & (p > 0.0) & (p < 1.0)
    if not retained.any():
        raise ValueError("No polymorphic training markers")
    p_retained = p[retained]
    transformed = dosage[:, retained].astype(np.float64, copy=True)
    means = 2.0 * p_retained
    missing = ~np.isfinite(transformed)
    transformed[missing] = np.broadcast_to(means, transformed.shape)[missing]
    transformed -= means
    denominator = float(2.0 * np.sum(p_retained * (1.0 - p_retained)))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"Invalid VanRaden denominator {denominator}")
    factor = transformed / math.sqrt(denominator)
    return {
        "training_index": training_index,
        "retained_mask": retained,
        "allele_frequency": p_retained,
        "imputation_value": means,
        "denominator": denominator,
        "factor": factor,
    }


def kernel_diagnostics(factor: np.ndarray) -> dict[str, float | int | bool]:
    kernel = np.asarray(factor @ factor.T, dtype=np.float64)
    symmetric = (kernel + kernel.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetric)
    diagonal = np.diag(symmetric)
    trace = float(np.trace(symmetric))
    effective_rank = 0.0
    positive = eigenvalues[eigenvalues > 1e-12]
    if positive.size and positive.sum() > 0:
        probabilities = positive / positive.sum()
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    upper = symmetric[np.triu_indices_from(symmetric, k=1)]
    return {
        "dimension": int(kernel.shape[0]),
        "all_finite": bool(np.isfinite(kernel).all()),
        "max_symmetry_error": float(np.max(np.abs(kernel - kernel.T))),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "materially_negative_eigenvalues": int(np.sum(eigenvalues < -1e-8)),
        "trace": trace,
        "mean_diagonal": float(diagonal.mean()),
        "minimum_diagonal": float(diagonal.min()),
        "maximum_diagonal": float(diagonal.max()),
        "off_diagonal_q05": float(np.quantile(upper, 0.05)) if upper.size else math.nan,
        "off_diagonal_q50": float(np.quantile(upper, 0.50)) if upper.size else math.nan,
        "off_diagonal_q95": float(np.quantile(upper, 0.95)) if upper.size else math.nan,
        "effective_rank": effective_rank,
        "condition_proxy": float(positive.max() / positive.min()) if positive.size else math.inf,
    }


def geo_factor(
    environment_ids: Sequence[str],
    location_keys: Sequence[str],
    training_environment_ids: set[str],
) -> tuple[sparse.csr_matrix, list[str], float]:
    pairs = [(str(env), str(location)) for env, location in zip(environment_ids, location_keys)]
    training_levels = sorted(
        {location for env, location in pairs if env in training_environment_ids and location}
    )
    level_index = {value: index for index, value in enumerate(training_levels)}
    rows: list[int] = []
    cols: list[int] = []
    for row, (_, location) in enumerate(pairs):
        if location in level_index:
            rows.append(row)
            cols.append(level_index[location])
    values = np.ones(len(rows), dtype=np.float64)
    factor = sparse.csr_matrix((values, (rows, cols)), shape=(len(pairs), len(training_levels)))
    training_rows = [i for i, (env, _) in enumerate(pairs) if env in training_environment_ids]
    if not training_rows:
        raise ValueError("No training environments for geography factor")
    diagonal = np.asarray(factor[training_rows].multiply(factor[training_rows]).sum(axis=1)).ravel()
    positive = diagonal[diagonal > 0]
    scale = float(positive.mean()) if positive.size else 1.0
    return factor / math.sqrt(scale), training_levels, scale


def assert_many_to_one(left: pd.DataFrame, right: pd.DataFrame, key: str) -> None:
    if right[key].duplicated().any():
        raise ValueError(f"Right table is not unique on {key}")
    before = len(left)
    joined = left.merge(right, on=key, how="left", validate="many_to_one")
    if len(joined) != before:
        raise ValueError(f"Many-to-one join changed row count on {key}")
