from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_trait_environment_candidates(
    manifest_path: Path,
    *,
    root: Path,
    base_e_order: pd.DataFrame,
) -> list[dict[str, object]]:
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    required = {
        "kernel",
        "biological_role",
        "kernel_path",
        "order_path",
        "eligible_traits",
        "enabled_default",
        "interaction_enabled",
        "rank",
        "minimum_ledger_coverage",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise SystemExit(f"{manifest_path} is missing columns: {missing}")
    names = manifest["kernel"].fillna("").astype(str).str.strip()
    if names.eq("").any():
        raise SystemExit(f"{manifest_path} contains an empty kernel name")
    if names.duplicated().any():
        raise SystemExit(f"{manifest_path} contains duplicate kernel names")
    candidates = []
    for _, row in manifest.iterrows():
        candidates.append(
            {
                "kernel": str(row["kernel"]).strip(),
                "axis": "environment",
                "biological_role": str(row["biological_role"]).strip(),
                "source_kernel": resolve(root, Path(str(row["kernel_path"]))),
                "source_order": resolve(root, Path(str(row["order_path"]))),
                "source_id_col": "env_id",
                "target_order": base_e_order,
                "target_id_col": "env_id",
                "eligible_traits": str(row["eligible_traits"]).strip(),
                "enabled_default": parse_bool(row["enabled_default"]),
                "interaction_enabled": parse_bool(row["interaction_enabled"]),
                "rank": int(row["rank"]),
                "minimum_ledger_coverage": float(row["minimum_ledger_coverage"]),
            }
        )
    return candidates


def load_order(path: Path, id_col: str) -> pd.DataFrame:
    order = pd.read_csv(path, sep="\t", dtype=str)
    if id_col not in order.columns:
        raise SystemExit(f"{path} does not contain ID column {id_col!r}")
    ids = order[id_col].fillna("").astype(str).str.strip()
    if not ids.ne("").all() or ids.duplicated().any():
        raise SystemExit(f"{path} has empty or duplicate IDs in {id_col}")
    if "compact_kernel_index" in order.columns:
        compact = pd.to_numeric(order["compact_kernel_index"], errors="raise").astype(int)
        if not np.array_equal(np.sort(compact), np.arange(len(order), dtype=int)):
            raise SystemExit(f"{path} compact_kernel_index is not a zero-based permutation")
        order = order.assign(_source_row=compact).sort_values("_source_row", kind="stable")
    else:
        order = order.assign(_source_row=np.arange(len(order), dtype=int))
    order[id_col] = ids.loc[order.index].to_numpy()
    return order.reset_index(drop=True)


def compact_kernel(
    *,
    name: str,
    source_kernel_path: Path,
    source_order_path: Path,
    source_id_col: str,
    target_order: pd.DataFrame,
    target_id_col: str,
    out_dir: Path,
    diagonal_epsilon: float,
) -> tuple[Path, Path, dict[str, object]]:
    source_order = load_order(source_order_path, source_id_col)
    source_kernel = np.load(source_kernel_path, mmap_mode="r")
    if source_kernel.ndim != 2 or source_kernel.shape[0] != source_kernel.shape[1]:
        raise SystemExit(f"{source_kernel_path} is not square: {source_kernel.shape}")
    if source_kernel.shape[0] != len(source_order):
        raise SystemExit(
            f"{name} order/kernel mismatch: order={len(source_order)}; kernel={source_kernel.shape}"
        )

    source_lookup = dict(zip(source_order[source_id_col], source_order["_source_row"].astype(int)))
    target_ids = target_order[target_id_col].fillna("").astype(str)
    mapped_source = target_ids.map(source_lookup)
    mapped_mask = mapped_source.notna().to_numpy()
    mapped_positions = np.flatnonzero(mapped_mask)
    source_positions = mapped_source[mapped_mask].astype(int).to_numpy()
    if not len(source_positions):
        raise SystemExit(f"{name} has no IDs in common with the base ledger order")

    source_diagonal = np.asarray(np.diag(source_kernel), dtype=np.float64)
    positive = np.isfinite(source_diagonal[source_positions]) & (
        source_diagonal[source_positions] > diagonal_epsilon
    )
    mapped_positions = mapped_positions[positive]
    source_positions = source_positions[positive]
    if not len(source_positions):
        raise SystemExit(f"{name} has no mapped IDs with a positive finite diagonal")

    compact = np.asarray(
        source_kernel[np.ix_(source_positions, source_positions)], dtype=np.float64
    )
    if not np.isfinite(compact).all():
        raise SystemExit(f"{name} contains non-finite values after compaction")
    compact = (compact + compact.T) * 0.5
    diagonal = np.diag(compact).copy()
    scale = np.sqrt(diagonal)
    compact = compact / np.outer(scale, scale)
    compact = (compact + compact.T) * 0.5
    np.fill_diagonal(compact, 1.0)

    kernel_path = out_dir / f"{name}.npy"
    order_path = out_dir / f"{name}_order.tsv"
    np.save(kernel_path, compact.astype(np.float32))
    compact_order = pd.DataFrame(
        {
            target_id_col: target_ids.iloc[mapped_positions].to_numpy(),
            "source_kernel_index": source_positions,
            "base_kernel_index": mapped_positions,
            "compact_kernel_index": np.arange(len(mapped_positions), dtype=int),
        }
    )
    compact_order.to_csv(order_path, sep="\t", index=False)
    qc = {
        "kernel": name,
        "source_kernel": str(source_kernel_path),
        "source_order": str(source_order_path),
        "source_dimension": int(source_kernel.shape[0]),
        "base_dimension": int(len(target_order)),
        "mapped_before_diagonal_filter": int(mapped_mask.sum()),
        "compact_dimension": int(len(compact_order)),
        "base_id_coverage": float(len(compact_order) / len(target_order)),
        "removed_nonpositive_or_nonfinite_diagonal": int(mapped_mask.sum() - len(compact_order)),
        "mean_diagonal": float(np.mean(np.diag(compact))),
        "kernel_sha256": file_sha256(kernel_path),
        "order_sha256": file_sha256(order_path),
    }
    return kernel_path, order_path, qc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare compact, aligned kernel experts for the multi-trait model."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--base-model-dir", type=Path, default=Path("model_kernels/stage1_pedigree_env")
    )
    parser.add_argument("--base-prefix", default="stage1_pedigree_env")
    parser.add_argument(
        "--hmp-model-dir",
        type=Path,
        default=Path("model_kernels/stage1_hmp_env_ke_diag_norm"),
    )
    parser.add_argument("--hmp-prefix", default="stage1_hmp_env")
    parser.add_argument(
        "--gbs-model-dir",
        type=Path,
        default=Path("model_kernels/stage1_gbs_sawyt_env_ke_diag_norm"),
    )
    parser.add_argument("--gbs-prefix", default="stage1_gbs_sawyt_env")
    parser.add_argument(
        "--dth-model-dir",
        type=Path,
        default=Path("model_kernels/stage1_pedigree_env_dth_v2"),
    )
    parser.add_argument(
        "--trait-environment-manifest",
        type=Path,
        default=Path(
            "model_kernels/trait_environment_v2/trait_environment_kernel_manifest.tsv"
        ),
    )
    parser.add_argument("--require-trait-environment-manifest", action="store_true")
    parser.add_argument("--environment-dir", type=Path, default=Path("environment"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("model_kernels/multitrait_kernel_experts")
    )
    parser.add_argument("--diagonal-epsilon", type=float, default=1e-8)
    parser.add_argument("--allow-missing-experts", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    base_dir = resolve(root, args.base_model_dir)
    hmp_dir = resolve(root, args.hmp_model_dir)
    gbs_dir = resolve(root, args.gbs_model_dir)
    dth_dir = resolve(root, args.dth_model_dir)
    trait_environment_manifest = resolve(root, args.trait_environment_manifest)
    environment_dir = resolve(root, args.environment_dir)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_g_order_path = base_dir / f"{args.base_prefix}_K_G_unique_order.tsv"
    base_e_order_path = base_dir / f"{args.base_prefix}_K_E_unique_order.tsv"
    base_g_order = load_order(base_g_order_path, "sample_id")
    base_e_order = load_order(base_e_order_path, "env_id")

    candidates = [
        {
            "kernel": "K_A",
            "axis": "genotype",
            "biological_role": "pedigree_additive_relationship",
            "source_kernel": base_dir / f"{args.base_prefix}_K_G_unique.npy",
            "source_order": base_g_order_path,
            "source_id_col": "sample_id",
            "target_order": base_g_order,
            "target_id_col": "sample_id",
            "eligible_traits": "*",
            "enabled_default": True,
            "interaction_enabled": True,
            "rank": 128,
            "minimum_ledger_coverage": 1.0,
        },
        {
            "kernel": "K_G_HMP_LINEAR",
            "axis": "genotype",
            "biological_role": "HMP_marker_linear_genomic_relationship",
            "source_kernel": hmp_dir / f"{args.hmp_prefix}_K_G_unique.npy",
            "source_order": hmp_dir / f"{args.hmp_prefix}_K_G_unique_order.tsv",
            "source_id_col": "sample_id",
            "target_order": base_g_order,
            "target_id_col": "sample_id",
            "eligible_traits": "*",
            "enabled_default": True,
            "interaction_enabled": True,
            "rank": 128,
            "minimum_ledger_coverage": 0.01,
        },
        {
            "kernel": "K_G_HMP_RBF",
            "axis": "genotype",
            "biological_role": "HMP_marker_gaussian_RBF",
            "source_kernel": hmp_dir / f"{args.hmp_prefix}_K_G_RBF_unique.npy",
            "source_order": hmp_dir / f"{args.hmp_prefix}_K_G_unique_order.tsv",
            "source_id_col": "sample_id",
            "target_order": base_g_order,
            "target_id_col": "sample_id",
            "eligible_traits": "*",
            "enabled_default": True,
            "interaction_enabled": True,
            "rank": 128,
            "minimum_ledger_coverage": 0.01,
        },
        {
            "kernel": "K_G_GBS_LINEAR",
            "axis": "genotype",
            "biological_role": "GBS_SAWYT_marker_linear_genomic_relationship",
            "source_kernel": gbs_dir / f"{args.gbs_prefix}_K_G_unique.npy",
            "source_order": gbs_dir / f"{args.gbs_prefix}_K_G_unique_order.tsv",
            "source_id_col": "sample_id",
            "target_order": base_g_order,
            "target_id_col": "sample_id",
            "eligible_traits": "*",
            "enabled_default": True,
            "interaction_enabled": True,
            "rank": 128,
            "minimum_ledger_coverage": 0.01,
        },
        {
            "kernel": "K_G_GBS_RBF",
            "axis": "genotype",
            "biological_role": "GBS_SAWYT_marker_gaussian_RBF",
            "source_kernel": gbs_dir / f"{args.gbs_prefix}_K_G_RBF_unique.npy",
            "source_order": gbs_dir / f"{args.gbs_prefix}_K_G_unique_order.tsv",
            "source_id_col": "sample_id",
            "target_order": base_g_order,
            "target_id_col": "sample_id",
            "eligible_traits": "*",
            "enabled_default": True,
            "interaction_enabled": True,
            "rank": 128,
            "minimum_ledger_coverage": 0.01,
        },
        {
            "kernel": "K_E_GENERIC",
            "axis": "environment",
            "biological_role": "legacy_equal_weight_combined_environment",
            "source_kernel": base_dir / f"{args.base_prefix}_K_E_unique.npy",
            "source_order": base_e_order_path,
            "source_id_col": "env_id",
            "target_order": base_e_order,
            "target_id_col": "env_id",
            "eligible_traits": "*",
            "enabled_default": False,
            "interaction_enabled": True,
            "rank": 64,
            "minimum_ledger_coverage": 0.95,
        },
    ]
    for component, minimum_coverage in [
        ("geo", 0.95),
        ("weather", 0.50),
        ("stress", 0.50),
        ("mgmt", 0.90),
    ]:
        candidates.append(
            {
                "kernel": f"K_E_{component.upper()}",
                "axis": "environment",
                "biological_role": f"environment_{component}_component",
                "source_kernel": environment_dir / f"K_{component}.npy",
                "source_order": environment_dir / "env_kernel_sample_order.tsv",
                "source_id_col": "env_id",
                "target_order": base_e_order,
                "target_id_col": "env_id",
                "eligible_traits": "*",
                "enabled_default": True,
                "interaction_enabled": True,
                "rank": 64,
                "minimum_ledger_coverage": minimum_coverage,
            }
        )
    candidates.append(
        {
            "kernel": "K_E_DTH_V2",
            "axis": "environment",
            "biological_role": "DTH_fixed_sowing_window_weather_and_trial_metadata",
            "source_kernel": dth_dir / f"{args.base_prefix}_K_E_unique.npy",
            "source_order": dth_dir / f"{args.base_prefix}_K_E_unique_order.tsv",
            "source_id_col": "env_id",
            "target_order": base_e_order,
            "target_id_col": "env_id",
            "eligible_traits": "DAYS_TO_HEADING",
            "enabled_default": False,
            "interaction_enabled": True,
            "rank": 64,
            "minimum_ledger_coverage": 0.95,
        }
    )
    if trait_environment_manifest.exists():
        extra_candidates = load_trait_environment_candidates(
            trait_environment_manifest,
            root=root,
            base_e_order=base_e_order,
        )
        existing = {str(candidate["kernel"]) for candidate in candidates}
        duplicates = sorted(
            str(candidate["kernel"])
            for candidate in extra_candidates
            if str(candidate["kernel"]) in existing
        )
        if duplicates:
            raise SystemExit(
                f"Trait-environment manifest duplicates built-in kernels: {duplicates}"
            )
        candidates.extend(extra_candidates)
    elif args.require_trait_environment_manifest:
        raise SystemExit(
            f"Required trait-environment manifest is absent: {trait_environment_manifest}"
        )

    missing = []
    registry_rows = []
    qc_rows = []
    for candidate in candidates:
        source_kernel = Path(candidate["source_kernel"])
        source_order = Path(candidate["source_order"])
        if not source_kernel.exists() or not source_order.exists():
            missing.append(
                {
                    "kernel": candidate["kernel"],
                    "source_kernel_exists": source_kernel.exists(),
                    "source_order_exists": source_order.exists(),
                    "source_kernel": str(source_kernel),
                    "source_order": str(source_order),
                }
            )
            continue
        kernel_path, order_path, qc = compact_kernel(
            name=str(candidate["kernel"]),
            source_kernel_path=source_kernel,
            source_order_path=source_order,
            source_id_col=str(candidate["source_id_col"]),
            target_order=candidate["target_order"],
            target_id_col=str(candidate["target_id_col"]),
            out_dir=out_dir,
            diagonal_epsilon=args.diagonal_epsilon,
        )
        qc_rows.append(qc)
        registry_rows.append(
            {
                "kernel": candidate["kernel"],
                "axis": candidate["axis"],
                "biological_role": candidate["biological_role"],
                "kernel_path": str(kernel_path),
                "order_path": str(order_path),
                "id_col": candidate["target_id_col"],
                "eligible_traits": candidate["eligible_traits"],
                "enabled_default": candidate["enabled_default"],
                "interaction_enabled": candidate["interaction_enabled"],
                "rank": candidate["rank"],
                "minimum_ledger_coverage": candidate["minimum_ledger_coverage"],
                "dimension": qc["compact_dimension"],
                "base_id_coverage": qc["base_id_coverage"],
                "source_kernel_path": str(source_kernel),
                "source_order_path": str(source_order),
            }
        )

    missing_frame = pd.DataFrame(missing)
    if not missing_frame.empty:
        missing_frame.to_csv(out_dir / "missing_kernel_experts.tsv", sep="\t", index=False)
        if not args.allow_missing_experts:
            raise SystemExit(
                "Required kernel experts are missing. See "
                f"{out_dir / 'missing_kernel_experts.tsv'}"
            )
    registry = pd.DataFrame(registry_rows)
    if registry.empty:
        raise SystemExit("No kernel experts were prepared")
    registry_path = out_dir / "multitrait_kernel_registry.tsv"
    registry.to_csv(registry_path, sep="\t", index=False)
    pd.DataFrame(qc_rows).to_csv(out_dir / "multitrait_kernel_preparation_qc.tsv", sep="\t", index=False)
    lineage = {
        "base_genotype_order": str(base_g_order_path),
        "base_genotype_order_sha256": file_sha256(base_g_order_path),
        "base_environment_order": str(base_e_order_path),
        "base_environment_order_sha256": file_sha256(base_e_order_path),
        "registry": str(registry_path),
        "registry_sha256": file_sha256(registry_path),
        "prepared_kernels": registry["kernel"].tolist(),
        "trait_environment_manifest": str(trait_environment_manifest)
        if trait_environment_manifest.exists()
        else "",
        "trait_environment_manifest_sha256": file_sha256(trait_environment_manifest)
        if trait_environment_manifest.exists()
        else "",
        "missing_kernels": missing,
    }
    (out_dir / "multitrait_kernel_registry_lineage.json").write_text(
        json.dumps(lineage, indent=2), encoding="utf-8"
    )
    print(registry.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
