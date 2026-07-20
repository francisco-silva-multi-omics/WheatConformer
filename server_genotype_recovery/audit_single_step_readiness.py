from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from server_genotype_recovery.build_regulatory_eligibility_manifest import (
    detect_column,
    read_table,
    sha256_file,
)


MISSING = {"", "NA", "NAN", "NONE", "NULL", ".", "-", "UNKNOWN", "0"}


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).strip())
    return "" if text.upper() in MISSING else text


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "reviewed"}


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (root / value).resolve()


def add_metric(
    rows: list[dict[str, object]], section: str, metric: str, value: object
) -> None:
    rows.append({"section": section, "metric": metric, "value": value})


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parent_table(path: Path) -> pd.DataFrame:
    frame = read_table(path)
    columns = {
        "sample_id": detect_column(frame, ["sample_id", "panel_sample_id", "genotype_id"]),
        "parent1": detect_column(frame, ["parent1", "female_parent", "mother", "dam"]),
        "parent2": detect_column(frame, ["parent2", "male_parent", "father", "sire"]),
    }
    missing = [name for name, column in columns.items() if column is None]
    if missing:
        raise ValueError(f"Pedigree parent table is missing columns: {missing}")
    output = pd.DataFrame(
        {
            name: frame[column].map(clean) for name, column in columns.items() if column
        }
    )
    output = output[output["sample_id"].ne("")].drop_duplicates().reset_index(drop=True)
    return output


def source_lineage_conflicts(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(), {
            "source_manifest_present": False,
            "source_manifest_rows": 0,
            "source_unique_children": 0,
            "source_children_with_multiple_lineages": 0,
        }
    frame = read_table(path)
    id_col = detect_column(
        frame, ["sample_id", "panel_sample_id_expected", "panel_sample_id", "genotype_id"]
    )
    p1_col = detect_column(frame, ["parent1", "female_parent", "mother", "dam"])
    p2_col = detect_column(frame, ["parent2", "male_parent", "father", "sire"])
    cross_col = detect_column(frame, ["cross_name", "cross", "pedigree", "designation"])
    if id_col is None or (cross_col is None and p1_col is None and p2_col is None):
        raise ValueError(f"Source pedigree manifest has no usable lineage columns: {path}")
    work = pd.DataFrame({"sample_id": frame[id_col].map(clean)})
    if p1_col is not None or p2_col is not None:
        work["lineage_signature"] = (
            (frame[p1_col].map(clean) if p1_col else "")
            + "|"
            + (frame[p2_col].map(clean) if p2_col else "")
        )
    else:
        work["lineage_signature"] = frame[cross_col].map(clean).str.upper()
    work = work[work["sample_id"].ne("") & work["lineage_signature"].ne("")]
    unique = work.drop_duplicates(["sample_id", "lineage_signature"])
    counts = unique.groupby("sample_id")["lineage_signature"].nunique()
    conflict_ids = set(counts[counts > 1].index)
    conflicts = unique[unique["sample_id"].isin(conflict_ids)].sort_values(
        ["sample_id", "lineage_signature"]
    )
    return conflicts.reset_index(drop=True), {
        "source_manifest_present": True,
        "source_manifest_rows": len(frame),
        "source_unique_children": work["sample_id"].nunique(),
        "source_children_with_multiple_lineages": len(conflict_ids),
    }


def curated_parent_tokens(path: Path | None) -> set[str]:
    if path is None or not path.is_file() or path.stat().st_size == 0:
        return set()
    frame = read_table(path)
    token_col = detect_column(frame, ["parent_token", "source_parent_id", "parent_id"])
    stable_col = detect_column(frame, ["stable_parent_id", "canonical_parent_id"])
    reviewed_col = detect_column(frame, ["reviewed", "accepted", "status"])
    if token_col is None or stable_col is None or reviewed_col is None:
        raise ValueError(f"Curated parent registry has an invalid schema: {path}")
    valid = frame[reviewed_col].map(bool_value) & frame[stable_col].map(clean).ne("")
    return set(frame.loc[valid, token_col].map(clean))


def pedigree_structure(
    pedigree: pd.DataFrame,
    *,
    child_pattern: re.Pattern[str],
    parent_pattern: re.Pattern[str],
    curated_tokens: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    assignment_counts = pedigree.groupby("sample_id").size()
    conflict_ids = set(assignment_counts[assignment_counts > 1].index)
    conflicts = pedigree[pedigree["sample_id"].isin(conflict_ids)].sort_values(
        ["sample_id", "parent1", "parent2"]
    )
    children = set(pedigree["sample_id"])
    parent_values = [
        value
        for column in ["parent1", "parent2"]
        for value in pedigree[column]
        if value
    ]
    parents = set(parent_values)
    invalid_children = sorted(child for child in children if not child_pattern.fullmatch(child))
    noncanonical = sorted(
        parent
        for parent in parents
        if not parent_pattern.fullmatch(parent)
    )
    uncurated = [parent for parent in noncanonical if parent not in curated_tokens]
    curated_aliases = [parent for parent in noncanonical if parent in curated_tokens]
    parent_counts = Counter(parent_values)
    parent_issues = pd.DataFrame(
        [
            {
                "parent_token": parent,
                "child_count": parent_counts[parent],
                "contains_pedigree_delimiter": bool(re.search(r"[/\\*]", parent)),
                "status": (
                    "curated_alias_requires_canonical_pedigree_rebuild"
                    if parent in curated_tokens
                    else "uncurated_noncanonical_parent_token"
                ),
            }
            for parent in noncanonical
        ]
    )
    self_parent = pedigree[
        pedigree.apply(
            lambda row: row["sample_id"] in {row["parent1"], row["parent2"]}, axis=1
        )
    ]
    duplicate_parent = pedigree[
        pedigree["parent1"].ne("") & pedigree["parent1"].eq(pedigree["parent2"])
    ]

    nodes = children | parents
    adjacency: dict[str, set[str]] = defaultdict(set)
    indegree = {node: 0 for node in nodes}
    directed_children: dict[str, set[str]] = defaultdict(set)
    for row in pedigree.itertuples(index=False):
        for parent in [row.parent1, row.parent2]:
            if not parent or parent == row.sample_id:
                continue
            adjacency[parent].add(row.sample_id)
            adjacency[row.sample_id].add(parent)
            if row.sample_id not in directed_children[parent]:
                directed_children[parent].add(row.sample_id)
                indegree[row.sample_id] += 1
    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    depth = {node: 0 for node in queue}
    visited = []
    while queue:
        node = queue.popleft()
        visited.append(node)
        for child in sorted(directed_children[node]):
            depth[child] = max(depth.get(child, 0), depth[node] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    cycle_nodes = sorted(nodes - set(visited))

    component_rows = []
    unseen = set(nodes)
    while unseen:
        start = min(unseen)
        stack = [start]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        unseen -= component
        component_rows.append(
            {
                "component_id": len(component_rows),
                "node_count": len(component),
                "child_count": len(component & children),
                "parent_token_count": len(component & parents),
            }
        )
    components = pd.DataFrame(component_rows).sort_values(
        "node_count", ascending=False, kind="stable"
    )
    parent_pair_sizes = (
        pedigree.assign(parent_pair=pedigree["parent1"] + "|" + pedigree["parent2"])
        .query("parent_pair != '|'", engine="python")
        .groupby("parent_pair")
        .size()
    )
    depth_values = np.asarray([depth.get(child, 0) for child in children], dtype=float)
    metrics = {
        "pedigree_rows": len(pedigree),
        "unique_children": len(children),
        "unique_parent_tokens": len(parents),
        "children_with_conflicting_parent_assignments": len(conflict_ids),
        "invalid_child_ids": len(invalid_children),
        "noncanonical_parent_tokens": len(noncanonical),
        "uncurated_noncanonical_parent_tokens": len(uncurated),
        "curated_parent_aliases_requiring_rebuild": len(curated_aliases),
        "self_parent_rows": len(self_parent),
        "duplicate_parent_rows": len(duplicate_parent),
        "founder_rows": int(pedigree[["parent1", "parent2"]].eq("").all(axis=1).sum()),
        "one_parent_rows": int(pedigree[["parent1", "parent2"]].ne("").sum(axis=1).eq(1).sum()),
        "two_parent_rows": int(pedigree[["parent1", "parent2"]].ne("").all(axis=1).sum()),
        "cycle_node_count": len(cycle_nodes),
        "connected_component_count": len(components),
        "largest_component_nodes": int(components["node_count"].max()) if len(components) else 0,
        "pedigree_depth_median": float(np.median(depth_values)) if len(depth_values) else 0.0,
        "pedigree_depth_max": int(np.max(depth_values)) if len(depth_values) else 0,
        "full_sib_family_count": int((parent_pair_sizes >= 2).sum()),
        "largest_full_sib_family": int(parent_pair_sizes.max()) if len(parent_pair_sizes) else 0,
    }
    issues = pd.concat(
        [
            conflicts.assign(issue="conflicting_parent_assignment"),
            self_parent.assign(issue="self_parent"),
            duplicate_parent.assign(issue="duplicate_parent"),
        ],
        ignore_index=True,
    )
    if invalid_children:
        invalid = pd.DataFrame({"sample_id": invalid_children, "issue": "invalid_child_id"})
        issues = pd.concat([issues, invalid], ignore_index=True)
    cycle_frame = pd.DataFrame({"node_id": cycle_nodes, "issue": "pedigree_cycle"})
    return issues, parent_issues, components, {**metrics, "cycle_nodes": cycle_frame}


def load_order(path: Path) -> list[str]:
    frame = read_table(path)
    id_col = detect_column(frame, ["sample_id", "panel_sample_id", "genotype_id"])
    if id_col is None:
        raise ValueError(f"Kernel order lacks a recognized ID column: {path}")
    values = frame[id_col].map(clean).tolist()
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError(f"Kernel order contains empty or duplicate IDs: {path}")
    return values


def kernel_integrity(path: Path, order: list[str], sample_size: int) -> dict[str, object]:
    kernel = np.load(path, mmap_mode="r")
    square = kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1]
    dimension_matches = square and kernel.shape[0] == len(order)
    if not dimension_matches or not order:
        return {
            "square": square,
            "dimension_matches_order": dimension_matches,
            "nonempty": bool(order),
            "finite": False,
            "status": "FAIL",
        }
    finite = True
    for start in range(0, len(order), 512):
        if not np.isfinite(np.asarray(kernel[start : start + 512])).all():
            finite = False
            break
    diagonal = np.asarray(kernel.diagonal(), dtype=np.float64)
    selected = np.linspace(0, len(order) - 1, min(len(order), sample_size), dtype=int)
    sample = np.asarray(kernel[np.ix_(selected, selected)], dtype=np.float64)
    symmetry = float(np.max(np.abs(sample - sample.T)))
    eigenvalues = np.linalg.eigvalsh((sample + sample.T) * 0.5) if finite else np.asarray([])
    min_eigenvalue = float(eigenvalues.min()) if len(eigenvalues) else float("nan")
    tolerance = max(1e-5, 1e-6 * float(np.trace(sample))) if finite else float("nan")
    status = (
        "PASS"
        if finite
        and symmetry <= 1e-5
        and np.all(diagonal > 0)
        and min_eigenvalue >= -tolerance
        else "FAIL"
    )
    return {
        "shape": f"{kernel.shape[0]}x{kernel.shape[1]}",
        "square": square,
        "dimension_matches_order": dimension_matches,
        "nonempty": True,
        "finite": finite,
        "symmetry_max_abs_sampled": symmetry,
        "diagonal_mean": float(diagonal.mean()),
        "diagonal_min": float(diagonal.min()),
        "diagonal_max": float(diagonal.max()),
        "sampled_min_eigenvalue": min_eigenvalue,
        "sampled_psd_tolerance": tolerance,
        "status": status,
    }


def overlap_moments(kernel: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    diagonal = np.asarray(kernel[indices, indices], dtype=np.float64)
    total = 0.0
    for start in range(0, len(indices), 256):
        rows = indices[start : start + 256]
        total += float(np.asarray(kernel[np.ix_(rows, indices)], dtype=np.float64).sum())
    off_count = len(indices) * (len(indices) - 1)
    off_mean = (total - float(diagonal.sum())) / off_count if off_count else 0.0
    return float(diagonal.mean()), float(off_mean)


def spectral_metrics(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    eigenvalues = np.linalg.eigvalsh((values + values.T) * 0.5)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = max(1e-8, 1e-8 * scale)
    positive = eigenvalues[eigenvalues > tolerance]
    total = float(positive.sum())
    effective_rank = (
        float(total * total / np.square(positive).sum()) if len(positive) else 0.0
    )
    condition = (
        float(positive.max() / positive.min()) if len(positive) else float("inf")
    )
    return {
        f"{prefix}_min_eigenvalue": float(eigenvalues.min()),
        f"{prefix}_max_eigenvalue": float(eigenvalues.max()),
        f"{prefix}_positive_eigenvalue_count": int(len(positive)),
        f"{prefix}_effective_rank": effective_rank,
        f"{prefix}_condition_number_positive_spectrum": condition,
    }


def a22_hmp_compatibility(
    ka_path: Path,
    ka_order: list[str],
    kg_path: Path,
    kg_order: list[str],
    *,
    sample_size: int,
    blend_fraction: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ka_lookup = {value: index for index, value in enumerate(ka_order)}
    kg_lookup = {value: index for index, value in enumerate(kg_order)}
    overlap = sorted(set(ka_lookup) & set(kg_lookup))
    ai = np.asarray([ka_lookup[value] for value in overlap], dtype=int)
    gi = np.asarray([kg_lookup[value] for value in overlap], dtype=int)
    ka = np.load(ka_path, mmap_mode="r")
    kg = np.load(kg_path, mmap_mode="r")
    overlap_frame = pd.DataFrame(
        {
            "sample_id": overlap,
            "K_A_index": ai,
            "K_G_HMP_index": gi,
        }
    )
    if len(overlap) < 2:
        return overlap_frame, {
            "overlap_genotypes": len(overlap),
            "A22_mean_diagonal": (
                float(ka[ai[0], ai[0]]) if len(overlap) == 1 else float("nan")
            ),
            "A22_mean_offdiagonal": float("nan"),
            "G_mean_diagonal": (
                float(kg[gi[0], gi[0]]) if len(overlap) == 1 else float("nan")
            ),
            "G_mean_offdiagonal": float("nan"),
            "alignment_alpha": float("nan"),
            "alignment_beta": float("nan"),
            "sampled_offdiagonal_correlation": float("nan"),
            "sampled_informative_pair_count": 0,
            "sampled_informative_pair_correlation": float("nan"),
            "blend_fraction_A22": blend_fraction,
            "sampled_scaled_G_min_eigenvalue": float("nan"),
            "sampled_blended_G_min_eigenvalue": float("nan"),
            "sampled_A22_condition_number_positive_spectrum": float("nan"),
            "sampled_scaled_G_condition_number_positive_spectrum": float("nan"),
            "sampled_blended_G_condition_number_positive_spectrum": float("nan"),
        }
    a_diag, a_off = overlap_moments(ka, ai)
    g_diag, g_off = overlap_moments(kg, gi)
    denominator = g_diag - g_off
    beta = (a_diag - a_off) / denominator if denominator > 1e-12 else float("nan")
    alpha = a_off - beta * g_off if np.isfinite(beta) else float("nan")
    chosen = np.linspace(0, len(overlap) - 1, min(len(overlap), sample_size), dtype=int)
    a_sample = np.asarray(ka[np.ix_(ai[chosen], ai[chosen])], dtype=np.float64)
    g_sample = np.asarray(kg[np.ix_(gi[chosen], gi[chosen])], dtype=np.float64)
    upper = np.triu_indices(len(chosen), k=1)
    a_upper = a_sample[upper]
    g_upper = g_sample[upper]
    correlation = (
        float(np.corrcoef(a_upper, g_upper)[0, 1])
        if len(a_upper) and np.std(a_upper) > 0 and np.std(g_upper) > 0
        else float("nan")
    )
    informative = a_upper >= 0.125
    informative_correlation = (
        float(np.corrcoef(a_upper[informative], g_upper[informative])[0, 1])
        if informative.sum() >= 3
        and np.std(a_upper[informative]) > 0
        and np.std(g_upper[informative]) > 0
        else float("nan")
    )
    if np.isfinite(beta):
        g_scaled = alpha + beta * g_sample
        g_blended = (1.0 - blend_fraction) * g_scaled + blend_fraction * a_sample
        a_spectrum = spectral_metrics(a_sample, "sampled_A22")
        scaled_spectrum = spectral_metrics(g_scaled, "sampled_scaled_G")
        blended_spectrum = spectral_metrics(g_blended, "sampled_blended_G")
    else:
        a_spectrum = spectral_metrics(a_sample, "sampled_A22")
        scaled_spectrum = {
            "sampled_scaled_G_min_eigenvalue": float("nan"),
            "sampled_scaled_G_max_eigenvalue": float("nan"),
            "sampled_scaled_G_positive_eigenvalue_count": 0,
            "sampled_scaled_G_effective_rank": 0.0,
            "sampled_scaled_G_condition_number_positive_spectrum": float("nan"),
        }
        blended_spectrum = {
            "sampled_blended_G_min_eigenvalue": float("nan"),
            "sampled_blended_G_max_eigenvalue": float("nan"),
            "sampled_blended_G_positive_eigenvalue_count": 0,
            "sampled_blended_G_effective_rank": 0.0,
            "sampled_blended_G_condition_number_positive_spectrum": float("nan"),
        }
    metrics = {
        "overlap_genotypes": len(overlap),
        "A22_mean_diagonal": a_diag,
        "A22_mean_offdiagonal": a_off,
        "G_mean_diagonal": g_diag,
        "G_mean_offdiagonal": g_off,
        "alignment_alpha": alpha,
        "alignment_beta": beta,
        "sampled_offdiagonal_correlation": correlation,
        "sampled_informative_pair_count": int(informative.sum()),
        "sampled_informative_pair_correlation": informative_correlation,
        "blend_fraction_A22": blend_fraction,
        **a_spectrum,
        **scaled_spectrum,
        **blended_spectrum,
    }
    return overlap_frame, metrics


def readiness_decision(
    source_metrics: dict[str, object],
    structure: dict[str, object],
    ka_qc: dict[str, object],
    kg_qc: dict[str, object],
    compatibility: dict[str, object],
    *,
    minimum_overlap: int,
) -> tuple[str, list[str], list[str]]:
    blocking: list[str] = []
    warnings: list[str] = []
    if not source_metrics["source_manifest_present"]:
        blocking.append("source_pedigree_manifest_absent")
    if source_metrics["source_children_with_multiple_lineages"]:
        blocking.append("source_children_have_multiple_lineages")
    for metric, reason in [
        ("children_with_conflicting_parent_assignments", "parent_assignments_conflict"),
        ("invalid_child_ids", "child_ids_are_not_canonical"),
        (
            "uncurated_noncanonical_parent_tokens",
            "parent_tokens_are_not_curated_stable_ids",
        ),
        (
            "curated_parent_aliases_requiring_rebuild",
            "curated_parent_aliases_require_canonical_K_A_rebuild",
        ),
        ("self_parent_rows", "self_parent_relationships_present"),
        ("duplicate_parent_rows", "duplicate_parent_relationships_present"),
        ("cycle_node_count", "pedigree_cycles_present"),
    ]:
        if structure.get(metric, 0):
            blocking.append(reason)
    if ka_qc["status"] != "PASS":
        blocking.append("K_A_integrity_failed")
    if kg_qc["status"] != "PASS":
        blocking.append("K_G_HMP_integrity_failed")
    if compatibility["overlap_genotypes"] < minimum_overlap:
        blocking.append("insufficient_A22_HMP_overlap")
    if (
        not np.isfinite(compatibility["alignment_beta"])
        or compatibility["alignment_beta"] <= 0
    ):
        blocking.append("A22_HMP_scaling_is_invalid")
    blended_min = compatibility["sampled_blended_G_min_eigenvalue"]
    if np.isfinite(blended_min) and blended_min < -1e-4:
        blocking.append("sampled_blended_G_is_not_PSD")
    for metric, reason in [
        ("K_A_order_missing_pedigree_nodes", "K_A_order_omits_pedigree_nodes"),
        ("K_A_order_unexpected_nodes", "K_A_order_contains_unexpected_nodes"),
    ]:
        if structure.get(metric, 0):
            blocking.append(reason)
    if structure["pedigree_depth_max"] <= 1:
        warnings.append("pedigree_depth_is_shallow")
    if compatibility["sampled_informative_pair_count"] < 30:
        warnings.append("few_informative_A22_pairs_in_sample")
    correlation = compatibility["sampled_offdiagonal_correlation"]
    if not np.isfinite(correlation) or correlation < 0.1:
        warnings.append("weak_A22_HMP_relationship_concordance")
    condition = compatibility.get(
        "sampled_blended_G_condition_number_positive_spectrum", float("nan")
    )
    if np.isfinite(condition) and condition > 1e8:
        warnings.append("sampled_blended_G_is_ill_conditioned")
    status = "BLOCKED" if blocking else ("WARN" if warnings else "PASS")
    return status, blocking, warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit pedigree provenance and A22/HMP compatibility before single-step H."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--pedigree-parent-table",
        type=Path,
        default=Path("genotype_panels/pedigree/pedigree_parent_table.tsv"),
    )
    parser.add_argument(
        "--pedigree-source-manifest",
        type=Path,
        default=Path("genotype_panels/pedigree/trial_derived_pedigree_manifest.tsv"),
    )
    parser.add_argument("--curated-parent-registry", type=Path)
    parser.add_argument(
        "--k-a", type=Path, default=Path("genotype_panels/pedigree/K_A.npy")
    )
    parser.add_argument(
        "--k-a-order",
        type=Path,
        default=Path("genotype_panels/pedigree/K_A_sample_order.tsv"),
    )
    parser.add_argument(
        "--k-g-hmp",
        type=Path,
        default=Path("genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy"),
    )
    parser.add_argument(
        "--k-g-hmp-order",
        type=Path,
        default=Path("genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv"),
    )
    parser.add_argument(
        "--regulatory-certification",
        type=Path,
        default=Path(
            "model_kernels/regulatory_eligibility_v1/"
            "regulatory_eligibility_certification.json"
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("model_kernels/single_step_readiness_v1")
    )
    parser.add_argument("--minimum-overlap", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--blend-fraction", type=float, default=0.05)
    parser.add_argument("--child-id-regex", default=r"^GID[0-9]+$")
    parser.add_argument("--parent-id-regex", default=r"^GID[0-9]+$")
    args = parser.parse_args()
    if args.minimum_overlap < 2:
        raise SystemExit("--minimum-overlap must be at least 2")
    if args.sample_size < 2:
        raise SystemExit("--sample-size must be at least 2")
    if not 0.0 <= args.blend_fraction <= 1.0:
        raise SystemExit("--blend-fraction must be between 0 and 1")
    root = args.root.resolve()
    paths = {
        "pedigree_parent_table": resolve(root, args.pedigree_parent_table),
        "pedigree_source_manifest": resolve(root, args.pedigree_source_manifest),
        "K_A": resolve(root, args.k_a),
        "K_A_order": resolve(root, args.k_a_order),
        "K_G_HMP": resolve(root, args.k_g_hmp),
        "K_G_HMP_order": resolve(root, args.k_g_hmp_order),
        "regulatory_certification": resolve(root, args.regulatory_certification),
    }
    curated_path = (
        resolve(root, args.curated_parent_registry) if args.curated_parent_registry else None
    )
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    required = [
        paths["pedigree_parent_table"],
        paths["K_A"],
        paths["K_A_order"],
        paths["K_G_HMP"],
        paths["K_G_HMP_order"],
        paths["regulatory_certification"],
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit(f"Required single-step readiness inputs are missing: {missing}")
    regulatory = json.loads(paths["regulatory_certification"].read_text(encoding="utf-8"))
    if regulatory.get("status") != "PASS":
        raise SystemExit("Regulatory eligibility certification is not PASS")

    pedigree = parent_table(paths["pedigree_parent_table"])
    source_conflicts, source_metrics = source_lineage_conflicts(
        paths["pedigree_source_manifest"]
    )
    curated = curated_parent_tokens(curated_path)
    issues, parent_issues, components, structure = pedigree_structure(
        pedigree,
        child_pattern=re.compile(args.child_id_regex),
        parent_pattern=re.compile(args.parent_id_regex),
        curated_tokens=curated,
    )
    cycle_nodes = structure.pop("cycle_nodes")
    ka_order = load_order(paths["K_A_order"])
    kg_order = load_order(paths["K_G_HMP_order"])
    pedigree_nodes = set(pedigree["sample_id"])
    pedigree_nodes.update(value for value in pedigree["parent1"] if value)
    pedigree_nodes.update(value for value in pedigree["parent2"] if value)
    ka_order_set = set(ka_order)
    missing_ka_nodes = sorted(pedigree_nodes - ka_order_set)
    unexpected_ka_nodes = sorted(ka_order_set - pedigree_nodes)
    structure["pedigree_relationship_node_count"] = len(pedigree_nodes)
    structure["K_A_order_node_count"] = len(ka_order)
    structure["K_A_order_missing_pedigree_nodes"] = len(missing_ka_nodes)
    structure["K_A_order_unexpected_nodes"] = len(unexpected_ka_nodes)
    ka_order_mismatches = pd.concat(
        [
            pd.DataFrame(
                {
                    "node_id": missing_ka_nodes,
                    "status": "pedigree_node_missing_from_K_A_order",
                }
            ),
            pd.DataFrame(
                {
                    "node_id": unexpected_ka_nodes,
                    "status": "unexpected_node_in_K_A_order",
                }
            ),
        ],
        ignore_index=True,
    )
    ka_qc = kernel_integrity(paths["K_A"], ka_order, args.sample_size)
    kg_qc = kernel_integrity(paths["K_G_HMP"], kg_order, args.sample_size)
    overlap, compatibility = a22_hmp_compatibility(
        paths["K_A"],
        ka_order,
        paths["K_G_HMP"],
        kg_order,
        sample_size=args.sample_size,
        blend_fraction=args.blend_fraction,
    )

    status, blocking, warnings = readiness_decision(
        source_metrics,
        structure,
        ka_qc,
        kg_qc,
        compatibility,
        minimum_overlap=args.minimum_overlap,
    )

    metric_rows: list[dict[str, object]] = []
    for key, value in source_metrics.items():
        add_metric(metric_rows, "source_lineage", key, value)
    for key, value in structure.items():
        add_metric(metric_rows, "pedigree_structure", key, value)
    for key, value in ka_qc.items():
        add_metric(metric_rows, "K_A_integrity", key, value)
    for key, value in kg_qc.items():
        add_metric(metric_rows, "K_G_HMP_integrity", key, value)
    for key, value in compatibility.items():
        add_metric(metric_rows, "A22_HMP_compatibility", key, value)
    pd.DataFrame(metric_rows).to_csv(
        out_dir / "single_step_readiness_metrics.tsv", sep="\t", index=False
    )
    pedigree.to_csv(out_dir / "pedigree_parent_table_audited.tsv", sep="\t", index=False)
    source_conflicts.to_csv(
        out_dir / "source_lineage_conflicts.tsv", sep="\t", index=False
    )
    issues.to_csv(out_dir / "pedigree_structural_issues.tsv", sep="\t", index=False)
    parent_issues.to_csv(
        out_dir / "uncurated_parent_tokens.tsv", sep="\t", index=False
    )
    components.to_csv(out_dir / "pedigree_components.tsv", sep="\t", index=False)
    cycle_nodes.to_csv(out_dir / "pedigree_cycle_nodes.tsv", sep="\t", index=False)
    ka_order_mismatches.to_csv(
        out_dir / "K_A_pedigree_order_mismatches.tsv", sep="\t", index=False
    )
    overlap.to_csv(out_dir / "A22_HMP_overlap_order.tsv", sep="\t", index=False)
    decision = {
        "status": status,
        "single_step_H_construction_allowed": not blocking,
        "selection_data": "pedigree_identifiers_and_relationship_kernels_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "blocking_reasons": blocking,
        "warnings": warnings,
        "regulatory_certification": str(paths["regulatory_certification"]),
        "regulatory_certification_sha256": sha256_file(
            paths["regulatory_certification"]
        ),
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
            if path.is_file()
        },
        "curated_parent_registry": (
            {"path": str(curated_path), "sha256": sha256_file(curated_path)}
            if curated_path is not None and curated_path.is_file()
            else None
        ),
        "pedigree_summary": structure,
        "source_summary": source_metrics,
        "K_A_integrity": ka_qc,
        "K_G_HMP_integrity": kg_qc,
        "A22_HMP_compatibility": compatibility,
        "interpretation_contract": {
            "mathematical_kernel_validity_implies_curated_pedigree": False,
            "cross_string_parent_token_implies_canonical_parent_identity": False,
            "curated_alias_without_canonical_K_A_rebuild_is_sufficient": False,
            "single_step_H_requires_reviewed_stable_parent_ids": True,
            "no_existing_kernel_was_modified": True,
        },
    }
    safe_decision = json_safe(decision)
    (out_dir / "single_step_readiness_decision.json").write_text(
        json.dumps(safe_decision, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(safe_decision, indent=2, allow_nan=False))
    print("\n=== READINESS METRICS ===")
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
