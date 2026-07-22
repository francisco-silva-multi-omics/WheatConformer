from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMNS = ("sample_id", "genotype_id", "panel_sample_id")


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_order(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    id_col = next((column for column in ID_COLUMNS if column in frame.columns), None)
    if id_col is None:
        raise ValueError(f"Kernel order has no recognized ID column: {path}")
    ids = frame[id_col].fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError(f"Kernel order contains empty or duplicate IDs: {path}")
    if "compact_kernel_index" in frame.columns:
        compact = pd.to_numeric(frame["compact_kernel_index"], errors="raise").astype(int)
        if not np.array_equal(np.sort(compact), np.arange(len(frame), dtype=int)):
            raise ValueError(f"compact_kernel_index is not a zero-based permutation: {path}")
        frame = frame.assign(_compact=compact).sort_values("_compact", kind="stable")
        ids = ids.loc[frame.index]
    return pd.DataFrame({"sample_id": ids.to_numpy()}).reset_index(drop=True)


def offdiagonal_mean(values: np.ndarray) -> float:
    if values.shape[0] < 2:
        return float("nan")
    return float((values.sum() - np.trace(values)) / (values.size - len(values)))


def relationship_moments(values: np.ndarray) -> tuple[float, float]:
    return float(np.diag(values).mean()), offdiagonal_mean(values)


def tune_and_blend_genomic_relationship(
    a22: np.ndarray,
    genomic: np.ndarray,
    *,
    pedigree_blend_fraction: float,
    eigen_floor_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    a_diag, a_off = relationship_moments(a22)
    g_diag, g_off = relationship_moments(genomic)
    denominator = g_diag - g_off
    if not np.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError("Genomic relationship has no usable diagonal/off-diagonal contrast")
    beta = (a_diag - a_off) / denominator
    alpha = a_off - beta * g_off
    if not np.isfinite(beta) or beta <= 0 or not np.isfinite(alpha):
        raise ValueError(f"Invalid genomic-to-pedigree tuning coefficients: alpha={alpha}, beta={beta}")
    scaled = alpha + beta * genomic
    blended = (
        (1.0 - pedigree_blend_fraction) * scaled
        + pedigree_blend_fraction * a22
    )
    blended = (blended + blended.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(blended)
    eigen_floor = max(
        1e-10,
        eigen_floor_fraction * max(float(np.diag(a22).mean()), 1.0),
    )
    clipped = np.maximum(eigenvalues, eigen_floor)
    working = (eigenvectors * clipped) @ eigenvectors.T
    working = (working + working.T) * 0.5
    return working, {
        "A22_mean_diagonal": a_diag,
        "A22_mean_offdiagonal": a_off,
        "G_mean_diagonal": g_diag,
        "G_mean_offdiagonal": g_off,
        "alignment_alpha": float(alpha),
        "alignment_beta": float(beta),
        "pedigree_blend_fraction": float(pedigree_blend_fraction),
        "pre_floor_min_eigenvalue": float(eigenvalues.min()),
        "eigen_floor": float(eigen_floor),
        "floored_eigenvalue_count": int((eigenvalues < eigen_floor).sum()),
        "working_G_mean_diagonal": float(np.diag(working).mean()),
        "working_G_mean_offdiagonal": offdiagonal_mean(working),
    }


def positive_inverse(values: np.ndarray, *, eigen_floor_fraction: float) -> tuple[np.ndarray, dict[str, float]]:
    symmetric = (values + values.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    floor = max(
        1e-10,
        eigen_floor_fraction * max(float(np.diag(symmetric).mean()), 1.0),
    )
    clipped = np.maximum(eigenvalues, floor)
    inverse = (eigenvectors * (1.0 / clipped)) @ eigenvectors.T
    inverse = (inverse + inverse.T) * 0.5
    return inverse, {
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
        "eigen_floor": float(floor),
        "floored_eigenvalue_count": int((eigenvalues < floor).sum()),
        "condition_number_after_floor": float(clipped.max() / clipped.min()),
    }


def construct_single_step_submatrix(
    pedigree: np.ndarray,
    target_indices: np.ndarray,
    genomic_indices: np.ndarray,
    working_genomic: np.ndarray,
    *,
    eigen_floor_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    a22 = np.asarray(pedigree[np.ix_(genomic_indices, genomic_indices)], dtype=np.float64)
    abb = np.asarray(pedigree[np.ix_(target_indices, target_indices)], dtype=np.float64)
    ab2 = np.asarray(pedigree[np.ix_(target_indices, genomic_indices)], dtype=np.float64)
    a22_inverse, inverse_qc = positive_inverse(
        a22, eigen_floor_fraction=eigen_floor_fraction
    )
    propagation = ab2 @ a22_inverse
    delta = working_genomic - a22
    correction = (propagation @ delta) @ propagation.T
    relationship = abb + correction
    relationship = (relationship + relationship.T) * 0.5
    return relationship, {
        "A22_inverse_min_eigenvalue": inverse_qc["min_eigenvalue"],
        "A22_inverse_eigen_floor": inverse_qc["eigen_floor"],
        "A22_inverse_floored_eigenvalue_count": inverse_qc[
            "floored_eigenvalue_count"
        ],
        "A22_inverse_condition_number_after_floor": inverse_qc[
            "condition_number_after_floor"
        ],
        "correction_frobenius_norm": float(np.linalg.norm(correction)),
        "pedigree_target_frobenius_norm": float(np.linalg.norm(abb)),
        "relative_correction_frobenius_norm": float(
            np.linalg.norm(correction) / max(np.linalg.norm(abb), 1e-12)
        ),
    }


def sampled_kernel_qc(values: np.ndarray, sample_size: int) -> dict[str, object]:
    if not np.isfinite(values).all():
        raise ValueError("Constructed H contains non-finite values")
    diagonal = np.diag(values)
    if np.any(diagonal <= 0):
        raise ValueError("Constructed H contains a non-positive diagonal")
    selected = np.linspace(
        0, len(values) - 1, min(len(values), sample_size), dtype=int
    )
    sample = np.asarray(values[np.ix_(selected, selected)], dtype=np.float64)
    sample = (sample + sample.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(sample)
    tolerance = max(1e-5, 1e-6 * max(float(np.trace(sample)), 1.0))
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError(
            "Constructed H failed sampled PSD certification: "
            f"min_eigenvalue={eigenvalues.min()} tolerance={tolerance}"
        )
    positive = eigenvalues[eigenvalues > max(1e-8, tolerance * 1e-3)]
    effective_rank = (
        float(np.square(positive.sum()) / np.square(positive).sum())
        if len(positive)
        else 0.0
    )
    return {
        "shape": f"{values.shape[0]}x{values.shape[1]}",
        "all_finite": True,
        "symmetry_max_abs": float(np.max(np.abs(values - values.T))),
        "diagonal_mean": float(diagonal.mean()),
        "diagonal_min": float(diagonal.min()),
        "diagonal_max": float(diagonal.max()),
        "sampled_size": int(len(selected)),
        "sampled_min_eigenvalue": float(eigenvalues.min()),
        "sampled_max_eigenvalue": float(eigenvalues.max()),
        "sampled_psd_tolerance": float(tolerance),
        "sampled_effective_rank": effective_rank,
    }


def require_readiness(path: Path, k_a_path: Path, k_a_order_path: Path) -> dict[str, object]:
    decision = json.loads(path.read_text(encoding="utf-8"))
    if not decision.get("single_step_H_construction_allowed", False):
        raise ValueError(
            "Single-step readiness did not allow construction: "
            f"{decision.get('blocking_reasons', [])}"
        )
    inputs = decision.get("inputs", {})
    expected = {
        "K_A": (k_a_path, inputs.get("K_A", {}).get("sha256")),
        "K_A_order": (k_a_order_path, inputs.get("K_A_order", {}).get("sha256")),
    }
    for label, (source, digest) in expected.items():
        if not digest:
            raise ValueError(f"Readiness decision lacks a certified {label} digest")
        observed = sha256_file(source)
        if observed != digest:
            raise ValueError(
                f"Readiness {label} digest is stale: expected={digest} observed={observed}"
            )
    return decision


def write_checksum_manifest(paths: list[Path], output: Path) -> None:
    lines = [f"{sha256_file(path)}  {path.name}" for path in paths]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a panel-specific single-step H submatrix for model genotypes "
            "using the covariance form equivalent to H inverse updating."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--k-a", type=Path, required=True)
    parser.add_argument("--k-a-order", type=Path, required=True)
    parser.add_argument("--k-g", type=Path, required=True)
    parser.add_argument("--k-g-order", type=Path, required=True)
    parser.add_argument("--target-order", type=Path, required=True)
    parser.add_argument("--readiness-decision", type=Path, required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument(
        "--genomic-relationship-method",
        default="certified_precomputed_VanRaden_relationship",
    )
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pedigree-blend-fraction", type=float, default=0.05)
    parser.add_argument("--eigen-floor-fraction", type=float, default=1e-6)
    parser.add_argument("--minimum-overlap", type=int, default=100)
    parser.add_argument("--minimum-target-overlap", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=1024)
    args = parser.parse_args()
    if not 0.0 < args.pedigree_blend_fraction < 1.0:
        raise SystemExit("--pedigree-blend-fraction must be strictly between 0 and 1")
    if args.eigen_floor_fraction <= 0:
        raise SystemExit("--eigen-floor-fraction must be positive")
    if (
        args.minimum_overlap < 2
        or args.minimum_target_overlap < 2
        or args.sample_size < 2
    ):
        raise SystemExit(
            "--minimum-overlap, --minimum-target-overlap, and --sample-size must be at least 2"
        )

    root = args.root.resolve()
    paths = {
        "K_A": resolve(root, args.k_a),
        "K_A_order": resolve(root, args.k_a_order),
        "K_G": resolve(root, args.k_g),
        "K_G_order": resolve(root, args.k_g_order),
        "target_order": resolve(root, args.target_order),
        "readiness_decision": resolve(root, args.readiness_decision),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"Required single-step inputs are missing: {missing}")
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readiness = require_readiness(
        paths["readiness_decision"], paths["K_A"], paths["K_A_order"]
    )

    ka_order = load_order(paths["K_A_order"])
    kg_order = load_order(paths["K_G_order"])
    target_order = load_order(paths["target_order"])
    ka_lookup = {value: index for index, value in enumerate(ka_order["sample_id"])}
    kg_lookup = {value: index for index, value in enumerate(kg_order["sample_id"])}
    missing_target = sorted(set(target_order["sample_id"]) - set(ka_lookup))
    if missing_target:
        raise SystemExit(
            f"Target order contains {len(missing_target)} IDs absent from K_A; "
            f"examples={missing_target[:10]}"
        )
    overlap = [value for value in kg_order["sample_id"] if value in ka_lookup]
    if len(overlap) < args.minimum_overlap:
        raise SystemExit(
            f"Only {len(overlap)} panel genotypes overlap K_A; minimum={args.minimum_overlap}"
        )
    ai = np.asarray([ka_lookup[value] for value in overlap], dtype=int)
    gi = np.asarray([kg_lookup[value] for value in overlap], dtype=int)
    bi = np.asarray([ka_lookup[value] for value in target_order["sample_id"]], dtype=int)

    ka = np.load(paths["K_A"], mmap_mode="r")
    kg = np.load(paths["K_G"], mmap_mode="r")
    if ka.ndim != 2 or ka.shape != (len(ka_order), len(ka_order)):
        raise SystemExit("K_A shape does not match its order")
    if kg.ndim != 2 or kg.shape != (len(kg_order), len(kg_order)):
        raise SystemExit("K_G shape does not match its order")
    a22 = np.asarray(ka[np.ix_(ai, ai)], dtype=np.float64)
    genomic = np.asarray(kg[np.ix_(gi, gi)], dtype=np.float64)
    if not np.isfinite(a22).all() or not np.isfinite(genomic).all():
        raise SystemExit("A22 or G contains non-finite values")
    working_genomic, tuning_qc = tune_and_blend_genomic_relationship(
        a22,
        genomic,
        pedigree_blend_fraction=args.pedigree_blend_fraction,
        eigen_floor_fraction=args.eigen_floor_fraction,
    )
    relationship, construction_qc = construct_single_step_submatrix(
        ka,
        bi,
        ai,
        working_genomic,
        eigen_floor_fraction=args.eigen_floor_fraction,
    )
    kernel_qc = sampled_kernel_qc(relationship, args.sample_size)

    target_lookup = {value: index for index, value in enumerate(target_order["sample_id"])}
    target_overlap = [value for value in overlap if value in target_lookup]
    if len(target_overlap) < args.minimum_target_overlap:
        raise SystemExit(
            f"Only {len(target_overlap)} directly genotyped IDs occur in the target order; "
            f"minimum={args.minimum_target_overlap}"
        )
    target_positions = np.asarray([target_lookup[value] for value in target_overlap], dtype=int)
    overlap_lookup = {value: index for index, value in enumerate(overlap)}
    overlap_positions = np.asarray(
        [overlap_lookup[value] for value in target_overlap], dtype=int
    )
    h22_residual = float(
        np.max(
            np.abs(
                relationship[np.ix_(target_positions, target_positions)]
                - working_genomic[np.ix_(overlap_positions, overlap_positions)]
            )
        )
    )
    h22_tolerance = max(1e-4, 100 * args.eigen_floor_fraction)
    if np.isfinite(h22_residual) and h22_residual > h22_tolerance:
        raise SystemExit(
            f"Single-step H22 replacement residual is too large: {h22_residual}"
        )

    kernel_path = out_dir / f"{args.prefix}.npy"
    order_path = out_dir / f"{args.prefix}_sample_order.tsv"
    overlap_path = out_dir / f"{args.prefix}_genotyped_overlap_order.tsv"
    qc_path = out_dir / f"{args.prefix}_construction_qc.tsv"
    provenance_path = out_dir / f"{args.prefix}_construction.json"
    checksum_path = out_dir / f"{args.prefix}_artifacts.sha256"
    np.save(kernel_path, relationship.astype(np.float32))
    pd.DataFrame(
        {
            "sample_id": target_order["sample_id"],
            "source_K_A_index": bi,
            "compact_kernel_index": np.arange(len(target_order), dtype=int),
        }
    ).to_csv(order_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "sample_id": overlap,
            "K_A_index": ai,
            "K_G_index": gi,
            "present_in_target_order": [value in target_lookup for value in overlap],
        }
    ).to_csv(overlap_path, sep="\t", index=False)
    metrics = {
        "status": "PASS",
        "panel": args.panel,
        "genomic_relationship_method": args.genomic_relationship_method,
        "pedigree_nodes": len(ka_order),
        "target_model_genotypes": len(target_order),
        "panel_genotypes": len(kg_order),
        "panel_K_A_overlap_genotypes": len(overlap),
        "overlap_genotypes_in_target_order": len(target_overlap),
        "H22_replacement_max_abs_residual": h22_residual,
        "H22_replacement_tolerance": h22_tolerance,
        **tuning_qc,
        **construction_qc,
        **kernel_qc,
    }
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    ).to_csv(qc_path, sep="\t", index=False)
    provenance = {
        "status": "PASS",
        "method": "single_step_H_target_marginal_covariance_update",
        "equivalent_inverse_definition": "H^-1=A^-1+block(G_working^-1-A22^-1)",
        "panel": args.panel,
        "genomic_relationship_method": args.genomic_relationship_method,
        "selection_data": "pedigree_and_genomic_relationships_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "readiness_status": readiness.get("status"),
        "parameters": {
            "pedigree_blend_fraction": args.pedigree_blend_fraction,
            "eigen_floor_fraction": args.eigen_floor_fraction,
            "minimum_overlap": args.minimum_overlap,
            "minimum_target_overlap": args.minimum_target_overlap,
            "sample_size": args.sample_size,
        },
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "outputs": {
            "kernel": str(kernel_path),
            "order": str(order_path),
            "genotyped_overlap_order": str(overlap_path),
            "qc": str(qc_path),
        },
        "metrics": metrics,
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_checksum_manifest(
        [kernel_path, order_path, overlap_path, qc_path, provenance_path], checksum_path
    )
    print(json.dumps(provenance, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
