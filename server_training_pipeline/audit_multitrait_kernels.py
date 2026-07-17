from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except ImportError:
            fallback = path.with_suffix(".tsv.gz")
            if fallback.exists():
                return pd.read_csv(fallback, sep="\t", low_memory=False)
            raise
    return pd.read_csv(path, sep="\t", low_memory=False)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(path),
    }


def extremal_eigenvalues(kernel: np.ndarray, exact_limit: int, seed: int) -> tuple[float, float, str]:
    n = kernel.shape[0]
    if n <= exact_limit:
        values = np.linalg.eigvalsh(np.asarray(kernel, dtype=np.float64))
        return float(values[0]), float(values[-1]), "exact"
    try:
        from scipy.sparse.linalg import eigsh

        matrix = np.asarray(kernel, dtype=np.float64)
        minimum = float(eigsh(matrix, k=1, which="SA", return_eigenvectors=False, tol=1e-5)[0])
        maximum = float(eigsh(matrix, k=1, which="LA", return_eigenvectors=False, tol=1e-5)[0])
        return minimum, maximum, "lanczos"
    except (ImportError, ValueError, RuntimeError, np.linalg.LinAlgError):
        rng = np.random.default_rng(seed)
        index = np.sort(rng.choice(n, size=min(exact_limit, n), replace=False))
        values = np.linalg.eigvalsh(np.asarray(kernel[np.ix_(index, index)], dtype=np.float64))
        return float(values[0]), float(values[-1]), f"sampled_{len(index)}"


def certify_kernel(
    *,
    name: str,
    biological_role: str,
    axis: str,
    kernel_path: Path,
    order_path: Path,
    id_col: str,
    ledger: pd.DataFrame,
    ledger_index_col: str | None,
    ledger_id_col: str,
    symmetry_tolerance: float,
    mean_diag_tolerance: float,
    exact_eigen_limit: int,
    seed: int,
    eligible_traits: str = "*",
    minimum_ledger_coverage: float = 1.0,
    coverage_basis: str = "observation_rows",
    allow_partial: bool = False,
    coverage_path: Path | None = None,
    coverage_id_col: str = "",
    coverage_column: str = "",
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    checks: list[dict[str, object]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append({"kernel": name, "check": check, "status": "PASS" if passed else "FAIL", "detail": detail})

    if not kernel_path.exists() or not order_path.exists():
        add("files_present", False, f"kernel={kernel_path.exists()}; order={order_path.exists()}")
        return checks, {}, {}
    add("files_present", True, f"kernel={kernel_path}; order={order_path}")

    kernel = np.load(kernel_path, mmap_mode="r")
    order = pd.read_csv(order_path, sep="\t", dtype=str)
    square = kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1]
    add("square", square, f"shape={kernel.shape}; dtype={kernel.dtype}")
    if not square:
        return checks, {}, {}
    n = kernel.shape[0]
    add("order_length", len(order) == n, f"order_rows={len(order)}; kernel_dimension={n}")
    required_order = {id_col, "compact_kernel_index"}
    missing_order = sorted(required_order.difference(order.columns))
    add("order_columns", not missing_order, f"missing={missing_order}")
    if missing_order or len(order) != n:
        return checks, {}, {}

    ids = order[id_col].fillna("").astype(str)
    compact = pd.to_numeric(order["compact_kernel_index"], errors="coerce")
    add("order_ids_unique_nonempty", bool(ids.ne("").all() and not ids.duplicated().any()), f"unique={ids.nunique()}")
    compact_ok = bool(
        compact.notna().all()
        and np.array_equal(np.sort(compact.astype(int).to_numpy()), np.arange(n, dtype=int))
    )
    add("compact_index_sequence", compact_ok, f"compact_min={compact.min()}; compact_max={compact.max()}")
    if coverage_path is not None:
        coverage_present = coverage_path.exists()
        add("coverage_mask_present", coverage_present, f"path={coverage_path}")
        if not coverage_present:
            return checks, {}, {}
        coverage_frame = pd.read_csv(coverage_path, sep="\t", dtype=str)
        missing_coverage = sorted(
            {coverage_id_col, coverage_column}.difference(coverage_frame.columns)
        )
        add("coverage_mask_columns", not missing_coverage, f"missing={missing_coverage}")
        if missing_coverage:
            return checks, {}, {}
        coverage_ids = coverage_frame[coverage_id_col].fillna("").astype(str)
        unique_coverage = bool(coverage_ids.ne("").all() and not coverage_ids.duplicated().any())
        add("coverage_mask_ids_unique_nonempty", unique_coverage, f"unique={coverage_ids.nunique()}")
        available_ids = set(
            coverage_ids[coverage_frame[coverage_column].map(parse_bool)]
        )
        order_covered = ids.isin(available_ids)
        add(
            "kernel_order_respects_coverage_mask",
            bool(order_covered.all()),
            f"covered={int(order_covered.sum())}/{len(order_covered)}",
        )

    finite = True
    max_asymmetry = 0.0
    sum_squares = 0.0
    chunk_size = max(1, min(512, n))
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = np.asarray(kernel[start:stop, :], dtype=np.float64)
        finite = finite and bool(np.isfinite(block).all())
        sum_squares += float(np.sum(np.square(block)))
        transpose_block = np.asarray(kernel[:, start:stop], dtype=np.float64).T
        max_asymmetry = max(max_asymmetry, float(np.max(np.abs(block - transpose_block))))
    add("finite_values", finite, f"all_finite={finite}")
    add("symmetry", max_asymmetry <= symmetry_tolerance, f"max_abs_difference={max_asymmetry:.8g}")

    diagonal = np.asarray(np.diag(kernel), dtype=np.float64)
    diag_finite_positive = bool(np.isfinite(diagonal).all() and np.all(diagonal > 0))
    mean_diag = float(np.mean(diagonal))
    add("positive_finite_diagonal", diag_finite_positive, f"min={diagonal.min():.8g}; max={diagonal.max():.8g}")
    add(
        "mean_diagonal_near_one",
        abs(mean_diag - 1.0) <= mean_diag_tolerance,
        f"mean={mean_diag:.8g}; tolerance={mean_diag_tolerance}",
    )

    minimum_eigenvalue, maximum_eigenvalue, eigen_method = extremal_eigenvalues(kernel, exact_eigen_limit, seed)
    psd_tolerance = max(1e-6, abs(maximum_eigenvalue) * 1e-6)
    add(
        "positive_semidefinite",
        minimum_eigenvalue >= -psd_tolerance,
        f"min_eigenvalue={minimum_eigenvalue:.8g}; tolerance={psd_tolerance:.8g}; method={eigen_method}",
    )

    eligible = ledger
    if eligible_traits != "*":
        requested = {value.strip().upper() for value in eligible_traits.split(",") if value.strip()}
        eligible = ledger[
            ledger["trait_name_canonical"].fillna("").astype(str).str.upper().isin(requested)
        ]
    if coverage_basis not in {"observation_rows", "unique_entities"}:
        raise ValueError(
            f"Unsupported coverage basis {coverage_basis!r} for kernel {name}"
        )
    observed = eligible[ledger_id_col].fillna("").astype(str)
    order_id_set = set(ids)
    mapped_by_id = observed.isin(order_id_set)
    id_match_count = int(mapped_by_id.sum())
    observation_coverage = float(mapped_by_id.mean()) if len(mapped_by_id) else 0.0
    observed_entity_ids = set(observed[observed.ne("")])
    mapped_entity_ids = observed_entity_ids.intersection(order_id_set)
    unique_entity_coverage = (
        len(mapped_entity_ids) / len(observed_entity_ids) if observed_entity_ids else 0.0
    )
    coverage = (
        observation_coverage
        if coverage_basis == "observation_rows"
        else unique_entity_coverage
    )
    coverage_ok = coverage >= minimum_ledger_coverage and (
        allow_partial or bool(mapped_by_id.all())
    )
    add(
        "ledger_id_coverage",
        coverage_ok,
        f"basis={coverage_basis}; selected_coverage={coverage:.8g}; "
        f"mapped_rows={id_match_count}/{len(observed)}; "
        f"observation_coverage={observation_coverage:.8g}; "
        f"mapped_unique_entities={len(mapped_entity_ids)}/{len(observed_entity_ids)}; "
        f"unique_entity_coverage={unique_entity_coverage:.8g}; "
        f"minimum={minimum_ledger_coverage}; allow_partial={allow_partial}",
    )

    if ledger_index_col and ledger_index_col in eligible.columns:
        index = pd.to_numeric(eligible[ledger_index_col], errors="coerce")
        in_range = index.notna() & index.ge(0) & index.lt(n)
        index_ok = bool(in_range.all()) if not allow_partial else bool((in_range | ~mapped_by_id).all())
        add(
            "ledger_indices_in_range",
            index_ok,
            f"in_range={int(in_range.sum())}/{len(index)}",
        )
        if bool(in_range.any()):
            order_by_compact = dict(zip(compact.astype(int), ids))
            expected = index[in_range].astype(int).map(order_by_compact).fillna("").astype(str)
            direct_observed = observed[in_range]
            id_match = expected.eq(direct_observed)
            add(
                "ledger_ids_match_order",
                bool(id_match.all()),
                f"matched={int(id_match.sum())}/{len(id_match)}",
            )

    trace = float(np.sum(diagonal))
    trace_effective_rank = trace / maximum_eigenvalue if maximum_eigenvalue > 0 else 0.0
    participation_ratio = trace**2 / sum_squares if sum_squares > 0 else 0.0
    registry = {
        "kernel": name,
        "biological_role": biological_role,
        "axis": axis,
        "kernel_path": str(kernel_path.resolve()),
        "kernel_sha256": file_sha256(kernel_path),
        "order_path": str(order_path.resolve()),
        "order_sha256": file_sha256(order_path),
        "id_col": id_col,
        "ledger_index_col": ledger_index_col,
        "ledger_id_col": ledger_id_col,
        "eligible_traits": eligible_traits,
        "minimum_ledger_coverage": minimum_ledger_coverage,
        "coverage_basis": coverage_basis,
        "ledger_id_coverage": coverage,
        "ledger_observation_coverage": observation_coverage,
        "ledger_unique_entity_coverage": unique_entity_coverage,
        "ledger_unique_entity_matches": len(mapped_entity_ids),
        "ledger_unique_entity_count": len(observed_entity_ids),
        "dimension": n,
        "dtype": str(kernel.dtype),
        "certification_status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL",
        "coverage_path": str(coverage_path.resolve()) if coverage_path is not None else "",
        "coverage_id_col": coverage_id_col if coverage_path is not None else "",
        "coverage_column": coverage_column if coverage_path is not None else "",
    }
    spectrum = {
        "kernel": name,
        "dimension": n,
        "mean_diagonal": mean_diag,
        "min_diagonal": float(np.min(diagonal)),
        "max_diagonal": float(np.max(diagonal)),
        "min_eigenvalue": minimum_eigenvalue,
        "max_eigenvalue": maximum_eigenvalue,
        "eigen_method": eigen_method,
        "trace_effective_rank": trace_effective_rank,
        "participation_ratio": participation_ratio,
        "max_abs_asymmetry": max_asymmetry,
        "ledger_id_matches": id_match_count,
        "ledger_unique_entity_matches": len(mapped_entity_ids),
        "ledger_unique_entity_count": len(observed_entity_ids),
        "ledger_observation_coverage": observation_coverage,
        "ledger_unique_entity_coverage": unique_entity_coverage,
        "coverage_basis": coverage_basis,
    }
    return checks, registry, spectrum


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify all registered multi-trait kernel experts.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--symmetry-tolerance", type=float, default=1e-5)
    parser.add_argument("--mean-diag-tolerance", type=float, default=0.05)
    parser.add_argument("--exact-eigen-limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    root = args.root.resolve()
    ledger_path = args.ledger if args.ledger.is_absolute() else root / args.ledger
    registry_path = args.registry if args.registry.is_absolute() else root / args.registry
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = read_table(ledger_path)
    source_registry = pd.read_csv(registry_path, sep="\t")
    required_columns = {
        "kernel",
        "biological_role",
        "axis",
        "kernel_path",
        "order_path",
        "id_col",
        "eligible_traits",
        "minimum_ledger_coverage",
    }
    missing = sorted(required_columns.difference(source_registry.columns))
    if missing:
        raise SystemExit(f"Kernel registry is missing columns: {missing}")
    if source_registry["kernel"].duplicated().any():
        raise SystemExit("Kernel registry contains duplicate kernel names")
    all_checks: list[dict[str, object]] = []
    registry_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    for _, row in source_registry.iterrows():
        axis = str(row["axis"])
        if axis not in {"genotype", "environment"}:
            raise SystemExit(f"Unsupported kernel axis {axis!r} for {row['kernel']}")
        specification = {
            "name": str(row["kernel"]),
            "biological_role": str(row["biological_role"]),
            "axis": axis,
            "kernel_path": Path(str(row["kernel_path"])),
            "order_path": Path(str(row["order_path"])),
            "id_col": str(row["id_col"]),
            "ledger_index_col": None,
            "ledger_id_col": "genotype_id" if axis == "genotype" else "environment_id",
            "eligible_traits": str(row["eligible_traits"]),
            "minimum_ledger_coverage": float(row["minimum_ledger_coverage"]),
            "coverage_basis": (
                "observation_rows"
                if pd.isna(row.get("coverage_basis"))
                else str(row.get("coverage_basis", "observation_rows")).strip()
                or "observation_rows"
            ),
            "allow_partial": float(row["minimum_ledger_coverage"]) < 1.0,
            "coverage_path": (
                None
                if pd.isna(row.get("coverage_path")) or not str(row.get("coverage_path", "")).strip()
                else Path(str(row["coverage_path"]))
            ),
            "coverage_id_col": ""
            if pd.isna(row.get("coverage_id_col"))
            else str(row.get("coverage_id_col", "")),
            "coverage_column": ""
            if pd.isna(row.get("coverage_column"))
            else str(row.get("coverage_column", "")),
        }
        checks, registry, spectrum = certify_kernel(
            **specification,
            ledger=ledger,
            symmetry_tolerance=args.symmetry_tolerance,
            mean_diag_tolerance=args.mean_diag_tolerance,
            exact_eigen_limit=args.exact_eigen_limit,
            seed=args.seed,
        )
        all_checks.extend(checks)
        if registry:
            registry_rows.append({**row.to_dict(), **registry})
        if spectrum:
            spectrum_rows.append(spectrum)

    checks_frame = pd.DataFrame(all_checks)
    registry_frame = pd.DataFrame(registry_rows)
    spectrum_frame = pd.DataFrame(spectrum_rows)
    checks_frame.to_csv(out_dir / "multitrait_kernel_certification_checks.tsv", sep="\t", index=False)
    registry_frame.to_csv(out_dir / "multitrait_kernel_registry.tsv", sep="\t", index=False)
    spectrum_frame.to_csv(out_dir / "multitrait_kernel_spectrum_summary.tsv", sep="\t", index=False)
    result = {
        "status": "PASS" if not checks_frame.empty and checks_frame["status"].eq("PASS").all() else "FAIL",
        "checks": int(len(checks_frame)),
        "failed_checks": int(checks_frame["status"].eq("FAIL").sum()) if not checks_frame.empty else 1,
        "ledger_identity": file_identity(ledger_path),
        "registry_identity": file_identity(registry_path),
        "kernel_identities": {
            row["kernel"]: file_identity(Path(row["kernel_path"])) for row in registry_rows
        },
        "order_identities": {
            row["kernel"]: file_identity(Path(row["order_path"])) for row in registry_rows
        },
        "coverage_identities": {
            row["kernel"]: file_identity(Path(row["coverage_path"]))
            for row in registry_rows
            if str(row.get("coverage_path", "")).strip()
        },
    }
    (out_dir / "multitrait_kernel_certification_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(checks_frame.to_string(index=False), flush=True)
    if result["status"] != "PASS":
        failed = checks_frame[checks_frame["status"].eq("FAIL")]
        print("\n=== FAILED KERNEL CERTIFICATION CHECKS ===", flush=True)
        print(failed.to_string(index=False), flush=True)
        raise SystemExit("Multi-trait kernel certification failed")


if __name__ == "__main__":
    main()
