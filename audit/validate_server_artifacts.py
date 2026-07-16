from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .audit_common import sampled_kernel_diagnostics
except ImportError:
    from audit_common import sampled_kernel_diagnostics


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


def validate_model_dir(path: Path, out_rows: list[dict[str, object]]) -> None:
    for obs_path in path.glob("*_model_ready_stage1_observations.parquet"):
        prefix = obs_path.name.replace("_model_ready_stage1_observations.parquet", "")
        obs = pd.read_parquet(obs_path)
        g_order_path = path / f"{prefix}_K_G_unique_order.tsv"
        e_order_path = path / f"{prefix}_K_E_unique_order.tsv"
        g_path = path / f"{prefix}_K_G_unique.npy"
        e_path = path / f"{prefix}_K_E_unique.npy"
        record = {"model_dir": str(path), "prefix": prefix, "observation_rows": len(obs)}
        if g_path.exists() and g_order_path.exists():
            g_order = pd.read_csv(g_order_path, sep="\t", dtype=str)
            kg = np.load(g_path, mmap_mode="r")
            record.update({"K_G_dim": kg.shape[0], "K_G_order_rows": len(g_order), "K_G_order_unique": g_order.iloc[:, 0].nunique(), "K_G_order_match": len(g_order) == kg.shape[0]})
        if e_path.exists() and e_order_path.exists():
            e_order = pd.read_csv(e_order_path, sep="\t", dtype=str)
            ke = np.load(e_path, mmap_mode="r")
            record.update({"K_E_dim": ke.shape[0], "K_E_order_rows": len(e_order), "K_E_order_unique": e_order.iloc[:, 0].nunique(), "K_E_order_match": len(e_order) == ke.shape[0]})
        if "geno_kernel_index" in obs and "env_kernel_index" in obs:
            record["geno_indices_in_range"] = bool(obs["geno_kernel_index"].between(0, record.get("K_G_dim", 0) - 1).all())
            record["env_indices_in_range"] = bool(obs["env_kernel_index"].between(0, record.get("K_E_dim", 0) - 1).all())
        record["status"] = "PASS" if all(value is not False for key, value in record.items() if key.endswith(("_match", "_in_range"))) else "FAIL"
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
        "kernel_failures": sum(row.get("status") in {"FAIL", "ERROR"} for row in kernel_rows),
        "explicit_validation_failures": sum(row.get("status") != "PASS" for row in explicit_rows),
    }
    write_json(out_dir / "server_audit_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if explicit_rows and summary["explicit_validation_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
