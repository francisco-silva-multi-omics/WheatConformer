from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
LOCAL_DEPS = BASE / "local_python_deps"
if platform.system() == "Windows" and LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

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


def clean_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def add_report(rows: list[dict[str, Any]], component: str, status: str, path: Path | str, detail: str, required: bool) -> None:
    rows.append(
        {
            "component": component,
            "status": status,
            "required": required,
            "path": str(path),
            "detail": detail,
        }
    )


def path_status(rows: list[dict[str, Any]], component: str, path: Path, required: bool) -> bool:
    if path.exists() and path.stat().st_size > 0:
        add_report(rows, component, "PASS", path, f"exists; bytes={path.stat().st_size}", required)
        return True
    status = "FAIL" if required else "WARN"
    add_report(rows, component, status, path, "missing or empty", required)
    return False


def load_order(path: Path, preferred_col: str | None, rows: list[dict[str, Any]], component: str, required: bool) -> tuple[pd.DataFrame | None, str | None]:
    if not path_status(rows, f"{component}_order_file", path, required):
        return None, None
    try:
        order = pd.read_csv(path, sep="\t", dtype=str)
    except Exception as exc:
        add_report(rows, f"{component}_order_read", "FAIL" if required else "WARN", path, f"could not read order: {exc}", required)
        return None, None
    if order.empty:
        add_report(rows, f"{component}_order_nonempty", "FAIL" if required else "WARN", path, "order file has zero rows", required)
        return None, None
    candidates = [preferred_col, "sample_id", "env_id", "future_env_id", "panel_sample_id"]
    selected = next((c for c in candidates if c and c in order.columns), None)
    if selected is None:
        selected = order.columns[0]
    values = clean_series(order[selected])
    dup = int(values.duplicated().sum())
    if dup:
        add_report(rows, f"{component}_order_unique", "FAIL" if required else "WARN", path, f"{dup} duplicated IDs in {selected}", required)
    else:
        add_report(rows, f"{component}_order_unique", "PASS", path, f"{len(order)} unique IDs in {selected}", required)
    return order, selected


def sampled_indices(n: int, max_n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.int64)
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, max_n, dtype=np.int64))


def validate_kernel(
    kernel_path: Path,
    order_path: Path,
    order_col: str | None,
    component: str,
    rows: list[dict[str, Any]],
    required: bool,
    sample_n: int,
) -> dict[str, Any] | None:
    if not path_status(rows, f"{component}_kernel_file", kernel_path, required):
        return None
    order, selected_col = load_order(order_path, order_col, rows, component, required)
    try:
        K = np.load(kernel_path, mmap_mode="r")
    except Exception as exc:
        add_report(rows, f"{component}_kernel_read", "FAIL" if required else "WARN", kernel_path, f"could not load npy: {exc}", required)
        return None
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        add_report(rows, f"{component}_kernel_square", "FAIL" if required else "WARN", kernel_path, f"shape={K.shape}", required)
        return None
    add_report(rows, f"{component}_kernel_square", "PASS", kernel_path, f"shape={K.shape}; dtype={K.dtype}", required)
    if order is not None and len(order) != K.shape[0]:
        add_report(
            rows,
            f"{component}_kernel_order_length",
            "FAIL" if required else "WARN",
            order_path,
            f"order rows={len(order)} but kernel dimension={K.shape[0]}",
            required,
        )
    elif order is not None:
        add_report(rows, f"{component}_kernel_order_length", "PASS", order_path, f"order rows match kernel dimension={K.shape[0]}", required)

    idx = sampled_indices(K.shape[0], sample_n)
    diag = np.asarray(K[idx, idx], dtype=np.float64) if len(idx) else np.array([], dtype=np.float64)
    if len(diag) and np.all(np.isfinite(diag)):
        detail = f"sampled_diag_n={len(diag)} mean={float(np.mean(diag)):.6g} min={float(np.min(diag)):.6g} max={float(np.max(diag)):.6g}"
        add_report(rows, f"{component}_kernel_diag_finite", "PASS", kernel_path, detail, required)
    else:
        add_report(rows, f"{component}_kernel_diag_finite", "FAIL" if required else "WARN", kernel_path, "sampled diagonal contains non-finite values", required)

    sym_idx = sampled_indices(K.shape[0], min(sample_n, 512))
    if len(sym_idx):
        block = np.asarray(K[np.ix_(sym_idx, sym_idx)], dtype=np.float64)
        max_abs = float(np.nanmax(np.abs(block - block.T)))
        status = "PASS" if math.isfinite(max_abs) and max_abs <= 1e-4 else ("FAIL" if required else "WARN")
        add_report(rows, f"{component}_kernel_sample_symmetry", status, kernel_path, f"sampled_n={len(sym_idx)} max_abs_diff={max_abs:.6g}", required)
        symmetric_block = (block + block.T) / 2.0
        min_eigenvalue = float(np.linalg.eigvalsh(symmetric_block)[0])
        psd_status = "PASS" if math.isfinite(min_eigenvalue) and min_eigenvalue >= -1e-4 else ("FAIL" if required else "WARN")
        add_report(
            rows,
            f"{component}_kernel_sample_psd",
            psd_status,
            kernel_path,
            f"sampled_n={len(sym_idx)} min_eigenvalue={min_eigenvalue:.6g}",
            required,
        )

    return {"path": str(kernel_path), "shape": list(K.shape), "dtype": str(K.dtype), "order_path": str(order_path), "order_col": selected_col}


def validate_observations(args: argparse.Namespace, rows: list[dict[str, Any]], full_g_shape: int | None, full_e_shape: int | None) -> dict[str, Any] | None:
    obs_path = args.observations
    if not path_status(rows, "observations_file", obs_path, True):
        return None
    try:
        obs = read_table(obs_path)
    except Exception as exc:
        add_report(rows, "observations_read", "FAIL", obs_path, f"could not read observations: {exc}", True)
        return None
    required_cols = {
        "observation_index",
        "geno_kernel_index",
        "env_kernel_index",
        "phenotype_value",
        "weight_g_e",
        "trait_name_canonical",
    }
    missing = sorted(required_cols.difference(obs.columns))
    if missing:
        add_report(rows, "observations_required_columns", "FAIL", obs_path, f"missing columns: {missing}", True)
        return None
    add_report(rows, "observations_required_columns", "PASS", obs_path, f"columns present; rows={len(obs)}", True)
    if obs.empty:
        add_report(rows, "observations_nonempty", "FAIL", obs_path, "zero rows", True)
        return None
    add_report(rows, "observations_nonempty", "PASS", obs_path, f"rows={len(obs)}", True)

    y = pd.to_numeric(obs["phenotype_value"], errors="coerce")
    w = pd.to_numeric(obs["weight_g_e"], errors="coerce")
    gi = pd.to_numeric(obs["geno_kernel_index"], errors="coerce")
    ei = pd.to_numeric(obs["env_kernel_index"], errors="coerce")
    add_report(rows, "observations_response_finite", "PASS" if y.notna().all() and np.isfinite(y).all() else "FAIL", obs_path, f"finite={int(np.isfinite(y).sum())}/{len(y)}", True)
    weight_ok = w.notna() & np.isfinite(w) & (w > 0)
    add_report(rows, "observations_weight_positive", "PASS" if bool(weight_ok.all()) else "FAIL", obs_path, f"positive_finite={int(weight_ok.sum())}/{len(w)}", True)
    index_ok = gi.notna() & ei.notna() & (gi >= 0) & (ei >= 0)
    if full_g_shape is not None:
        index_ok = index_ok & (gi < full_g_shape)
    if full_e_shape is not None:
        index_ok = index_ok & (ei < full_e_shape)
    add_report(rows, "observations_kernel_indices_in_range", "PASS" if bool(index_ok.all()) else "FAIL", obs_path, f"in_range={int(index_ok.sum())}/{len(obs)}", True)

    trait_counts = clean_series(obs["trait_name_canonical"]).value_counts().head(20)
    add_report(rows, "observations_trait_coverage", "PASS", obs_path, f"traits={obs['trait_name_canonical'].nunique()}; top20={trait_counts.to_dict()}", True)

    return {
        "path": str(obs_path),
        "rows": int(len(obs)),
        "unique_genotypes": int(obs["geno_kernel_index"].nunique()),
        "unique_environments": int(obs["env_kernel_index"].nunique()),
        "unique_traits": int(obs["trait_name_canonical"].nunique()),
    }


def validate_indices(args: argparse.Namespace, rows: list[dict[str, Any]], obs_rows: int | None) -> dict[str, Any] | None:
    path = args.indices
    if not path_status(rows, "observation_indices_npz", path, True):
        return None
    try:
        z = np.load(path)
    except Exception as exc:
        add_report(rows, "observation_indices_npz_read", "FAIL", path, f"could not read npz: {exc}", True)
        return None
    required = ["geno_kernel_index", "env_kernel_index", "y", "weight", "var", "se"]
    missing = [k for k in required if k not in z.files]
    if missing:
        add_report(rows, "observation_indices_npz_arrays", "FAIL", path, f"missing arrays: {missing}", True)
        return None
    lengths = {k: int(len(z[k])) for k in required}
    ok = len(set(lengths.values())) == 1 and (obs_rows is None or next(iter(lengths.values())) == obs_rows)
    add_report(rows, "observation_indices_npz_lengths", "PASS" if ok else "FAIL", path, f"lengths={lengths}; observation_rows={obs_rows}", True)
    return {"path": str(path), "lengths": lengths}


def validate_compact_orders(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    obs_summary: dict[str, Any] | None,
    sample_n: int,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    kg = validate_kernel(args.compact_g_kernel, args.compact_g_order, None, "compact_K_G", rows, True, sample_n)
    kg_rbf = validate_kernel(args.compact_g_rbf_kernel, args.compact_g_order, None, "compact_K_G_RBF", rows, True, sample_n)
    ke = validate_kernel(args.compact_e_kernel, args.compact_e_order, None, "compact_K_E", rows, True, sample_n)
    if kg:
        manifest["compact_K_G"] = kg
    if kg_rbf:
        manifest["compact_K_G_RBF"] = kg_rbf
    if ke:
        manifest["compact_K_E"] = ke
    for label, kernel, order, expected_unique in [
        ("compact_K_G", kg, args.compact_g_order, "unique_genotypes"),
        ("compact_K_G_RBF", kg_rbf, args.compact_g_order, "unique_genotypes"),
        ("compact_K_E", ke, args.compact_e_order, "unique_environments"),
    ]:
        if kernel and obs_summary:
            n = kernel["shape"][0]
            expected = int(obs_summary[expected_unique])
            status = "PASS" if n == expected else "FAIL"
            add_report(rows, f"{label}_dimension_matches_observations", status, order, f"kernel_dim={n}; observation_{expected_unique}={expected}", True)
    return manifest


def validate_optional_kernel(
    path: Path,
    order_path: Path,
    order_col: str | None,
    component: str,
    rows: list[dict[str, Any]],
    observed_ids: set[str],
    sample_n: int,
) -> dict[str, Any] | None:
    if not path.exists() and not order_path.exists():
        add_report(rows, f"{component}_optional_presence", "WARN", path, "optional kernel absent", False)
        return None
    info = validate_kernel(path, order_path, order_col, component, rows, False, sample_n)
    if info is None:
        return None
    order, selected = load_order(order_path, order_col, rows, f"{component}_coverage", False)
    if order is not None and selected:
        ids = set(clean_series(order[selected]))
        covered = len(observed_ids.intersection(ids))
        add_report(rows, f"{component}_observed_genotype_coverage", "PASS" if covered == len(observed_ids) else "WARN", order_path, f"covered={covered}/{len(observed_ids)} observed genotype IDs", False)
    return info


def validate_gaussian_qc(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not path_status(rows, "full_K_G_RBF_qc_file", path, True):
        return None
    try:
        qc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        add_report(rows, "full_K_G_RBF_qc_read", "FAIL", path, f"could not parse JSON: {exc}", True)
        return None
    gamma = float(qc.get("gamma", float("nan")))
    median_d2 = float(qc.get("sampled_median_squared_distance", float("nan")))
    min_eigenvalue = float(qc.get("sampled_min_eigenvalue", float("nan")))
    valid = (
        math.isfinite(gamma)
        and gamma > 0
        and math.isfinite(median_d2)
        and median_d2 > 0
        and math.isfinite(min_eigenvalue)
        and min_eigenvalue >= -1e-4
    )
    add_report(
        rows,
        "full_K_G_RBF_qc_values",
        "PASS" if valid else "FAIL",
        path,
        f"gamma={gamma:.6g}; median_d2={median_d2:.6g}; sampled_min_eigenvalue={min_eigenvalue:.6g}",
        True,
    )
    return qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model-ready matrices against the thesis multikernel methodology.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--model-dir", type=Path, default=Path("model_kernels/stage1_hmp_env"))
    parser.add_argument("--prefix", default="stage1_hmp_env")
    parser.add_argument("--out-dir", type=Path, default=Path("model_kernels/readiness"))
    parser.add_argument("--sample-n", type=int, default=2048)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--indices", type=Path)
    parser.add_argument("--geno-kernel", type=Path, default=Path("genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy"))
    parser.add_argument("--geno-rbf-kernel", type=Path, default=Path("genotype_panels/hmp/K_HMP.QCfiltered.gaussian.npy"))
    parser.add_argument("--geno-rbf-qc", type=Path, default=Path("genotype_panels/hmp/K_HMP.QCfiltered.gaussian.qc.json"))
    parser.add_argument("--geno-order", type=Path, default=Path("genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv"))
    parser.add_argument("--env-kernel", type=Path, default=Path("environment/K_E.npy"))
    parser.add_argument("--env-order", type=Path, default=Path("environment/env_kernel_sample_order.tsv"))
    parser.add_argument("--compact-g-kernel", type=Path)
    parser.add_argument("--compact-g-rbf-kernel", type=Path)
    parser.add_argument("--compact-g-order", type=Path)
    parser.add_argument("--compact-e-kernel", type=Path)
    parser.add_argument("--compact-e-order", type=Path)
    parser.add_argument("--k-a", type=Path, default=Path("genotype_panels/pedigree/K_A.npy"))
    parser.add_argument("--k-a-order", type=Path, default=Path("genotype_panels/pedigree/K_A_sample_order.tsv"))
    parser.add_argument("--k-z", type=Path, default=Path("model_kernels/K_z.npy"))
    parser.add_argument("--k-z-order", type=Path, default=Path("model_kernels/K_z_sample_order.tsv"))
    parser.add_argument("--pangenome-output", type=Path, action="append", default=[])
    args = parser.parse_args()

    args.root = args.root.resolve()
    args.model_dir = (args.root / args.model_dir).resolve() if not args.model_dir.is_absolute() else args.model_dir
    args.out_dir = (args.root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    args.observations = args.observations or args.model_dir / f"{args.prefix}_model_ready_stage1_observations.parquet"
    if not args.observations.exists() and args.observations.suffix == ".parquet":
        fallback_observations = args.observations.with_suffix(".tsv.gz")
        if fallback_observations.exists():
            args.observations = fallback_observations
    args.indices = args.indices or args.model_dir / f"{args.prefix}_observation_kernel_indices.npz"
    args.compact_g_kernel = args.compact_g_kernel or args.model_dir / f"{args.prefix}_K_G_unique.npy"
    args.compact_g_rbf_kernel = args.compact_g_rbf_kernel or args.model_dir / f"{args.prefix}_K_G_RBF_unique.npy"
    args.compact_g_order = args.compact_g_order or args.model_dir / f"{args.prefix}_K_G_unique_order.tsv"
    args.compact_e_kernel = args.compact_e_kernel or args.model_dir / f"{args.prefix}_K_E_unique.npy"
    args.compact_e_order = args.compact_e_order or args.model_dir / f"{args.prefix}_K_E_unique_order.tsv"

    for attr in [
        "observations",
        "indices",
        "geno_kernel",
        "geno_rbf_kernel",
        "geno_rbf_qc",
        "geno_order",
        "env_kernel",
        "env_order",
        "compact_g_kernel",
        "compact_g_rbf_kernel",
        "compact_g_order",
        "compact_e_kernel",
        "compact_e_order",
        "k_a",
        "k_a_order",
        "k_z",
        "k_z_order",
    ]:
        path = getattr(args, attr)
        if isinstance(path, Path) and not path.is_absolute():
            setattr(args, attr, (args.root / path).resolve())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"root": str(args.root), "methodology": "non-pangenome model matrices; pangenome external"}

    full_g = validate_kernel(args.geno_kernel, args.geno_order, "sample_id", "full_K_G", rows, True, args.sample_n)
    full_g_rbf = validate_kernel(args.geno_rbf_kernel, args.geno_order, "sample_id", "full_K_G_RBF", rows, True, args.sample_n)
    full_e = validate_kernel(args.env_kernel, args.env_order, "env_id", "full_K_E", rows, True, args.sample_n)
    if full_g:
        manifest["full_K_G"] = full_g
    if full_g_rbf:
        manifest["full_K_G_RBF"] = full_g_rbf
    gaussian_qc = validate_gaussian_qc(args.geno_rbf_qc, rows)
    if gaussian_qc:
        manifest["full_K_G_RBF_qc"] = gaussian_qc
    if full_e:
        manifest["full_K_E"] = full_e

    obs_summary = validate_observations(
        args,
        rows,
        full_g["shape"][0] if full_g else None,
        full_e["shape"][0] if full_e else None,
    )
    if obs_summary:
        manifest["observations"] = obs_summary
    idx_summary = validate_indices(args, rows, obs_summary["rows"] if obs_summary else None)
    if idx_summary:
        manifest["observation_indices"] = idx_summary
    manifest.update(validate_compact_orders(args, rows, obs_summary, args.sample_n))

    observed_ids: set[str] = set()
    try:
        obs = read_table(args.observations)
        if "panel_sample_id" in obs.columns:
            observed_ids = set(clean_series(obs["panel_sample_id"]))
    except Exception:
        observed_ids = set()

    ka = validate_optional_kernel(args.k_a, args.k_a_order, "sample_id", "K_A_pedigree", rows, observed_ids, args.sample_n)
    kz = validate_optional_kernel(args.k_z, args.k_z_order, "sample_id", "K_z_regulatory", rows, observed_ids, args.sample_n)
    if ka:
        manifest["K_A_optional"] = ka
    if kz:
        manifest["K_z_optional"] = kz

    pangenome_entries = []
    for path in args.pangenome_output:
        resolved = path if path.is_absolute() else (args.root / path).resolve()
        exists = resolved.exists() and resolved.stat().st_size > 0
        add_report(rows, "pangenome_external_artifact", "PASS" if exists else "WARN", resolved, "external artifact referenced; not recomputed", False)
        pangenome_entries.append({"path": str(resolved), "exists": exists})
    manifest["pangenome_external_artifacts"] = pangenome_entries

    report = pd.DataFrame(rows)
    report_path = args.out_dir / "model_input_readiness_report.tsv"
    manifest_path = args.out_dir / "model_input_manifest.json"
    report.to_csv(report_path, sep="\t", index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    required_failures = report[(report["required"]) & (report["status"] == "FAIL")]
    print(report[["component", "status", "required", "detail"]].to_string(index=False))
    print(f"Wrote: {report_path}")
    print(f"Wrote: {manifest_path}")
    if not required_failures.empty:
        raise SystemExit(f"Required model-input checks failed: {len(required_failures)}")


if __name__ == "__main__":
    main()
