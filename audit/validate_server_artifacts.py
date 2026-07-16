from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from .audit_common import sampled_kernel_diagnostics
except ImportError:
    from audit_common import sampled_kernel_diagnostics

from server_training_pipeline.observation_index_bundle import compare_observation_index_bundle


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_artifacts(root: Path, extra_directories: list[Path] | None = None) -> list[dict[str, object]]:
    rows = []
    patterns = [
        "genotype_panels/pedigree/K_A.npy",
        "model_kernels/**/*_model_ready_stage1_observations.parquet",
        "model_kernels/**/*_observation_kernel_indices.npz",
        "model_kernels/**/*_K_G_unique.npy",
        "model_kernels/**/*_K_E_unique.npy",
        "model_kernels/**/*_K_GE_hadamard.npy",
        "model_kernels/multitrait_*/**/*registry*.tsv",
        "model_kernels/multitrait_*/**/*ledger*.parquet",
    ]
    seen = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                rows.append({"path": str(path.resolve()), "relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size, "suffix": "".join(path.suffixes)})
    for directory in extra_directories or []:
        if not directory.exists():
            continue
        for path in directory.glob("*"):
            resolved = path.resolve()
            if path.is_file() and resolved not in seen:
                seen.add(resolved)
                try:
                    relative_path = str(resolved.relative_to(root))
                except ValueError:
                    relative_path = str(resolved)
                rows.append(
                    {
                        "path": str(resolved),
                        "relative_path": relative_path,
                        "bytes": path.stat().st_size,
                        "suffix": "".join(path.suffixes),
                    }
                )
    return rows


def validate_explicit_kernel_order(
    kernel_path: Path,
    order_path: Path,
    id_col: str,
    component: str,
) -> dict[str, object]:
    record: dict[str, object] = {
        "component": component,
        "kernel_path": str(kernel_path),
        "order_path": str(order_path),
    }
    if not kernel_path.exists() or not order_path.exists():
        record.update({"status": "FAIL", "detail": "kernel or order file missing"})
        return record
    kernel = np.load(kernel_path, mmap_mode="r")
    order = pd.read_csv(order_path, sep="\t", dtype=str)
    ids = order[id_col] if id_col in order else order.iloc[:, 0]
    record.update(
        {
            "kernel_rows": int(kernel.shape[0]) if kernel.ndim == 2 else -1,
            "kernel_columns": int(kernel.shape[1]) if kernel.ndim == 2 else -1,
            "order_rows": len(order),
            "order_unique": int(ids.nunique(dropna=False)),
        }
    )
    checks = [
        kernel.ndim == 2,
        kernel.shape[0] == kernel.shape[1] if kernel.ndim == 2 else False,
        kernel.shape[0] == len(order) if kernel.ndim == 2 else False,
        ids.notna().all(),
        ids.is_unique,
    ]
    record["status"] = "PASS" if all(checks) else "FAIL"
    record["detail"] = "explicit kernel/order alignment"
    return record


def validate_compact_axis(
    obs: pd.DataFrame,
    kernel_path: Path,
    order_path: Path,
    order_id_col: str,
    obs_index_col: str,
    obs_id_col: str,
    kernel_label: str,
    obs_label: str,
) -> tuple[dict[str, object], list[bool]]:
    record: dict[str, object] = {
        f"{kernel_label}_kernel_present": kernel_path.exists(),
        f"{kernel_label}_order_present": order_path.exists(),
    }
    checks = [kernel_path.exists(), order_path.exists()]
    if not all(checks):
        return record, checks

    kernel = np.load(kernel_path, mmap_mode="r")
    order = pd.read_csv(order_path, sep="\t", dtype=str)
    kernel_square = kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1]
    required_order_cols = {order_id_col, "source_kernel_index", "compact_kernel_index"}
    order_schema = required_order_cols.issubset(order.columns)
    record.update(
        {
            f"{kernel_label}_dim": int(kernel.shape[0]) if kernel.ndim == 2 else -1,
            f"{kernel_label}_order_rows": len(order),
            f"{kernel_label}_kernel_square": kernel_square,
            f"{kernel_label}_order_schema": order_schema,
        }
    )
    checks.extend([kernel_square, order_schema])
    if not order_schema:
        return record, checks

    source = pd.to_numeric(order["source_kernel_index"], errors="coerce")
    compact = pd.to_numeric(order["compact_kernel_index"], errors="coerce")
    source_values = source.to_numpy(dtype=float)
    compact_values = compact.to_numpy(dtype=float)
    source_integral = bool(
        np.isfinite(source_values).all() and np.equal(source_values, np.floor(source_values)).all()
    )
    compact_integral = bool(
        np.isfinite(compact_values).all() and np.equal(compact_values, np.floor(compact_values)).all()
    )
    id_values = order[order_id_col].fillna("").astype(str)
    id_unique = bool(id_values.ne("").all() and id_values.is_unique)
    source_unique = bool(source_integral and source.is_unique)
    compact_zero_based = False
    if compact_integral:
        compact_int = compact.astype(np.int64)
        compact_zero_based = bool(
            np.array_equal(
                np.sort(compact_int.to_numpy()),
                np.arange(len(order), dtype=np.int64),
            )
        )
    order_match = bool(kernel_square and kernel.shape[0] == len(order))
    record.update(
        {
            f"{kernel_label}_order_id_unique": id_unique,
            f"{kernel_label}_source_index_unique": source_unique,
            f"{kernel_label}_compact_index_zero_based": compact_zero_based,
            f"{kernel_label}_order_match": order_match,
        }
    )
    checks.extend([id_unique, source_unique, compact_zero_based, order_match])

    obs_index_present = obs_index_col in obs.columns
    obs_id_present = obs_id_col in obs.columns
    record[f"{obs_label}_source_index_present"] = obs_index_present
    record[f"{obs_label}_id_present"] = obs_id_present
    checks.extend([obs_index_present, obs_id_present])
    if not (obs_index_present and obs_id_present and source_unique and compact_zero_based):
        return record, checks

    obs_source = pd.to_numeric(obs[obs_index_col], errors="coerce")
    obs_source_values = obs_source.to_numpy(dtype=float)
    obs_source_integral = bool(
        np.isfinite(obs_source_values).all()
        and np.equal(obs_source_values, np.floor(obs_source_values)).all()
    )
    record[f"{obs_label}_source_indices_numeric_integer"] = obs_source_integral
    checks.append(obs_source_integral)
    if not obs_source_integral:
        return record, checks

    source_int = source.astype(np.int64)
    compact_int = compact.astype(np.int64)
    obs_source_int = obs_source.astype(np.int64)
    source_to_compact = dict(zip(source_int, compact_int))
    source_to_id = dict(zip(source_int, id_values))
    mapped_compact = obs_source_int.map(source_to_compact)
    mapped_ids = obs_source_int.map(source_to_id)
    mapped_complete = bool(mapped_compact.notna().all())
    mapped_in_range = bool(
        mapped_complete
        and mapped_compact.between(0, max(len(order) - 1, 0)).all()
    )
    observed_ids = obs[obs_id_col].fillna("").astype(str)
    ids_match = bool(mapped_complete and mapped_ids.astype(str).eq(observed_ids).all())
    record.update(
        {
            f"{obs_label}_source_indices_mapped": mapped_complete,
            f"{obs_label}_unmapped_count": int(mapped_compact.isna().sum()),
            f"{obs_label}_compact_indices_in_range": mapped_in_range,
            f"{obs_label}_ids_match_order": ids_match,
            f"{obs_label}_id_mismatch_count": int((mapped_ids.astype(str) != observed_ids).sum()),
        }
    )
    checks.extend([mapped_complete, mapped_in_range, ids_match])
    return record, checks


def validate_model_dir(path: Path, out_rows: list[dict[str, object]]) -> None:
    for obs_path in path.glob("*_model_ready_stage1_observations.parquet"):
        prefix = obs_path.name.replace("_model_ready_stage1_observations.parquet", "")
        obs = pd.read_parquet(obs_path)
        record: dict[str, object] = {
            "model_dir": str(path),
            "prefix": prefix,
            "observation_rows": len(obs),
        }
        checks: list[bool] = [len(obs) > 0]
        g_record, g_checks = validate_compact_axis(
            obs,
            path / f"{prefix}_K_G_unique.npy",
            path / f"{prefix}_K_G_unique_order.tsv",
            "sample_id",
            "geno_kernel_index",
            "panel_sample_id",
            "K_G",
            "geno",
        )
        e_record, e_checks = validate_compact_axis(
            obs,
            path / f"{prefix}_K_E_unique.npy",
            path / f"{prefix}_K_E_unique_order.tsv",
            "env_id",
            "env_kernel_index",
            "env_kernel_id",
            "K_E",
            "env",
        )
        record.update(g_record)
        record.update(e_record)
        checks.extend(g_checks + e_checks)

        warnings: list[str] = []
        index_path = path / f"{prefix}_observation_kernel_indices.npz"
        record["observation_index_bundle_present"] = index_path.exists()
        if index_path.exists():
            try:
                comparison = compare_observation_index_bundle(obs, index_path)
                record["observation_index_bundle_schema"] = comparison["schema_match"]
                record["observation_index_bundle_rows_match"] = comparison["rows_match"]
                record["observation_index_bundle_values_match"] = comparison["values_match"]
                if not all(comparison.values()):
                    warnings.append("auxiliary observation index NPZ does not match the Parquet ledger")
            except Exception as exc:
                record["observation_index_bundle_schema"] = False
                record["observation_index_bundle_rows_match"] = False
                record["observation_index_bundle_values_match"] = False
                warnings.append(f"could not validate auxiliary observation index NPZ: {type(exc).__name__}: {exc}")
        else:
            warnings.append("auxiliary observation index NPZ is absent")

        record["warning_count"] = len(warnings)
        record["warnings"] = "; ".join(warnings)
        record["status"] = "PASS" if all(checks) else "FAIL"
        out_rows.append(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate server-only stage-1 and multitrait artifacts")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("audit/server_artifacts"))
    parser.add_argument("--model-dir", type=Path, action="append", default=[])
    parser.add_argument("--environment-dir", type=Path)
    parser.add_argument("--pedigree-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    explicit_model_dirs = [(root / path).resolve() if not path.is_absolute() else path.resolve() for path in args.model_dir]
    environment_dir = (
        (root / args.environment_dir).resolve() if args.environment_dir and not args.environment_dir.is_absolute()
        else args.environment_dir.resolve() if args.environment_dir else None
    )
    pedigree_dir = (
        (root / args.pedigree_dir).resolve() if args.pedigree_dir and not args.pedigree_dir.is_absolute()
        else args.pedigree_dir.resolve() if args.pedigree_dir else None
    )
    extra_dirs = explicit_model_dirs + [path for path in [environment_dir, pedigree_dir] if path is not None]
    inventory = find_artifacts(root, extra_directories=extra_dirs)
    write_csv(out_dir / "server_artifact_inventory.csv", inventory)
    model_rows: list[dict[str, object]] = []
    discovered_model_dirs = {Path(row["path"]).parent for row in inventory if "model_kernels" in row["relative_path"]}
    for directory in sorted(discovered_model_dirs | set(explicit_model_dirs)):
        validate_model_dir(directory, model_rows)
    write_csv(out_dir / "server_model_alignment_audit.csv", model_rows)
    kernel_rows = []
    for row in inventory:
        path = Path(row["path"])
        if path.suffix == ".npy" and ("K_" in path.name):
            try:
                kernel_rows.append(sampled_kernel_diagnostics(path))
            except Exception as exc:
                kernel_rows.append({"path": str(path), "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
    write_csv(out_dir / "server_kernel_diagnostics.csv", kernel_rows)
    explicit_rows = []
    if environment_dir is not None:
        explicit_rows.append(
            validate_explicit_kernel_order(
                environment_dir / "K_E.npy",
                environment_dir / "env_kernel_sample_order.tsv",
                "env_id",
                "K_E",
            )
        )
    if pedigree_dir is not None:
        explicit_rows.append(
            validate_explicit_kernel_order(
                pedigree_dir / "K_A.npy",
                pedigree_dir / "K_A_sample_order.tsv",
                "sample_id",
                "K_A",
            )
        )
    explicit_rows.extend(
        {
            "component": f"compact_model:{row.get('prefix', '')}",
            "status": row.get("status", "FAIL"),
            "detail": json.dumps(row, sort_keys=True),
        }
        for row in model_rows
        if Path(row["model_dir"]).resolve() in set(explicit_model_dirs)
    )
    pd.DataFrame(explicit_rows).to_csv(
        out_dir / "server_artifact_validation.tsv", sep="\t", index=False, lineterminator="\n"
    )
    summary = {
        "artifacts": len(inventory),
        "model_directories": len(model_rows),
        "model_failures": sum(row.get("status") == "FAIL" for row in model_rows),
        "model_warnings": sum(int(row.get("warning_count", 0)) for row in model_rows),
        "kernel_failures": sum(row.get("status") in {"FAIL", "ERROR"} for row in kernel_rows),
        "explicit_validation_failures": sum(row.get("status") != "PASS" for row in explicit_rows),
    }
    write_json(out_dir / "server_audit_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if explicit_rows and summary["explicit_validation_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
