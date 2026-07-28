from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_dth_env_features_v2 import (
    base_env_table,
    build_geo,
    build_observed_envdata,
    build_window_features,
    feature_export_frame,
    kernel_from_features,
    read_order,
)
from build_trait_environment_kernels import (
    TRAIT_SPECS,
    observed_features_for_trait,
)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_frozen_scaling(
    raw_features: pd.DataFrame,
    scaling: pd.DataFrame,
    expected_columns: list[str],
) -> pd.DataFrame:
    if raw_features.index.has_duplicates:
        raise ValueError("Raw environment feature IDs are duplicated")
    if scaling["feature"].astype(str).duplicated().any():
        raise ValueError("Frozen scaling contains duplicate feature names")

    required_scaling_columns = {"feature", "mean", "std"}
    missing_scaling_columns = sorted(required_scaling_columns - set(scaling.columns))
    if missing_scaling_columns:
        raise ValueError(
            f"Frozen scaling lacks required columns: {missing_scaling_columns}"
        )
    base_features = [
        column for column in expected_columns if not column.endswith("__missing")
    ]
    if not base_features:
        raise ValueError("Certified standardized matrix contains no base features")

    scaling_by_feature = scaling.assign(
        feature=scaling["feature"].fillna("").astype(str)
    ).set_index("feature")
    parts: dict[str, pd.Series] = {}
    for feature in base_features:
        if feature not in scaling_by_feature.index:
            raise ValueError(f"Frozen scaling is absent for certified feature: {feature}")
        if feature not in raw_features.columns:
            raise ValueError(f"Frozen feature is absent from reconstructed inputs: {feature}")
        row = scaling_by_feature.loc[feature]
        mean = float(row["mean"])
        std = float(row["std"])
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid frozen scaling for {feature}: mean={mean}; std={std}")
        values = pd.to_numeric(raw_features[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        missing = values.isna()
        parts[feature] = ((values.fillna(mean) - mean) / std).rename(feature)
        missing_column = f"{feature}__missing"
        if missing_column in expected_columns:
            parts[missing_column] = missing.astype(np.float32).rename(missing_column)

    if not parts:
        raise ValueError("Frozen scaling retained no environment features")
    missing = sorted(set(expected_columns) - set(parts))
    extra = sorted(set(parts) - set(expected_columns))
    if missing or extra:
        raise ValueError(
            "Reconstructed frozen feature columns disagree with the certified matrix: "
            f"missing={missing[:10]}; extra={extra[:10]}"
        )
    standardized = pd.concat(
        [parts[column] for column in expected_columns], axis=1
    ).astype(np.float32)
    if not np.isfinite(standardized.to_numpy(dtype=np.float64)).all():
        raise ValueError("Frozen feature projection produced nonfinite values")
    return standardized


def extend_standardized_kernel(
    *,
    source_kernel: np.ndarray,
    source_order: pd.DataFrame,
    source_features: pd.DataFrame,
    target_order: pd.DataFrame,
    projected_features: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    source_ids = source_order["env_id"].astype(str).tolist()
    target_ids = target_order["env_id"].astype(str).tolist()
    if source_kernel.shape != (len(source_ids), len(source_ids)):
        raise ValueError(
            "Source kernel/order mismatch: "
            f"kernel={source_kernel.shape}; order={len(source_ids)}"
        )
    missing_source_ids = sorted(set(source_ids) - set(target_ids))
    if missing_source_ids:
        raise ValueError(
            "Recovered target order lost frozen source environments: "
            f"{missing_source_ids[:10]}"
        )

    source = source_features.copy()
    if "env_id" not in source.columns:
        raise ValueError("Certified standardized features lack env_id")
    source["env_id"] = source["env_id"].fillna("").astype(str).str.strip()
    if source["env_id"].duplicated().any() or set(source["env_id"]) != set(source_ids):
        raise ValueError("Certified standardized feature IDs disagree with source order")
    feature_columns = [column for column in source.columns if column != "env_id"]
    if projected_features.columns.tolist() != feature_columns:
        raise ValueError("Projected feature columns are not in the frozen feature order")
    if set(projected_features.index.astype(str)) != set(target_ids):
        raise ValueError("Projected feature IDs disagree with the recovered target order")

    source = source.set_index("env_id").loc[source_ids, feature_columns].astype(np.float32)
    extended = projected_features.loc[target_ids, feature_columns].astype(np.float32).copy()
    extended.loc[source_ids, feature_columns] = source.to_numpy(dtype=np.float32)
    kernel = kernel_from_features(extended)
    positions = np.asarray([target_ids.index(value) for value in source_ids], dtype=int)
    source_block = kernel[np.ix_(positions, positions)]
    max_abs_delta = float(
        np.max(np.abs(source_block.astype(np.float64) - source_kernel.astype(np.float64)))
    )
    return kernel, extended, max_abs_delta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extend a frozen trait-specific environment kernel to a recovered "
            "environment order without refitting its feature transformation."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-order", type=Path, required=True)
    parser.add_argument("--envdata", type=Path, required=True)
    parser.add_argument("--locdata", type=Path, required=True)
    parser.add_argument("--window-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kernel", default="K_E_TGW_V2", choices=sorted(TRAIT_SPECS))
    parser.add_argument("--original-block-tolerance", type=float, default=5e-6)
    args = parser.parse_args()

    root = args.root.resolve()
    resolve = lambda value: value.resolve() if value.is_absolute() else (root / value).resolve()
    source_manifest_path = resolve(args.source_manifest)
    target_order_path = resolve(args.target_order)
    envdata_path = resolve(args.envdata)
    locdata_path = resolve(args.locdata)
    window_path = resolve(args.window_features)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(source_manifest_path, sep="\t", dtype=str)
    selected = manifest[manifest["kernel"].astype(str).eq(args.kernel)]
    if len(selected) != 1:
        raise ValueError(f"Expected one {args.kernel} source manifest row; found {len(selected)}")
    source_row = selected.iloc[0].copy()
    source_kernel_path = resolve(Path(str(source_row["kernel_path"])))
    source_order_path = resolve(Path(str(source_row["order_path"])))
    source_dir = source_kernel_path.parent
    source_features_path = source_dir / f"{args.kernel}_features.parquet"
    source_scaling_path = source_dir / f"{args.kernel}_scaling.tsv"
    source_feature_manifest_path = source_dir / f"{args.kernel}_feature_manifest.tsv"
    required = [
        source_kernel_path,
        source_order_path,
        source_features_path,
        source_scaling_path,
        source_feature_manifest_path,
        target_order_path,
        envdata_path,
        locdata_path,
        window_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    source_order = read_order(source_order_path)
    target_order = read_order(target_order_path)
    source_kernel = np.load(source_kernel_path, mmap_mode="r")
    source_features = pd.read_parquet(source_features_path)
    scaling = pd.read_csv(source_scaling_path, sep="\t")
    target_ids = target_order["env_id"].astype(str).reset_index(drop=True)

    spec = TRAIT_SPECS[args.kernel]
    trait = str(spec["trait"])
    envdata = pd.read_csv(envdata_path, sep="\t", dtype=str, low_memory=False)
    locdata = pd.read_csv(locdata_path, sep="\t", dtype=str, low_memory=False)
    env_base = base_env_table(envdata, target_ids)
    feature_sets = {
        "geo": build_geo(env_base, locdata).reindex(target_ids),
        "observed_envdata": observed_features_for_trait(
            build_observed_envdata(envdata, target_ids), trait
        ),
        "api_sowing_windows": build_window_features(
            window_path,
            target_ids,
            allowed_labels=set(spec["windows"]),
            allowed_metrics=set(spec["metrics"]),
        ),
    }
    raw_features = pd.concat(feature_sets.values(), axis=1)
    raw_features.index = target_ids
    raw_features.index.name = "env_id"
    certified_columns = [
        column for column in source_features.columns if column != "env_id"
    ]
    projected = apply_frozen_scaling(raw_features, scaling, certified_columns)
    kernel, extended_features, original_block_delta = extend_standardized_kernel(
        source_kernel=source_kernel,
        source_order=source_order,
        source_features=source_features,
        target_order=target_order,
        projected_features=projected,
    )
    if original_block_delta > args.original_block_tolerance:
        raise ValueError(
            "Frozen trait-kernel block changed during extension: "
            f"max_abs_delta={original_block_delta:.9g}; "
            f"tolerance={args.original_block_tolerance:.9g}"
        )

    output_kernel = out_dir / f"{args.kernel}.npy"
    output_order = out_dir / f"{args.kernel}_order.tsv"
    output_features = out_dir / f"{args.kernel}_features.parquet"
    output_manifest = out_dir / "trait_environment_kernel_manifest.tsv"
    output_qc = out_dir / f"{args.kernel}_extension_qc.json"
    np.save(output_kernel, kernel)
    target_order.to_csv(output_order, sep="\t", index=False)
    feature_export_frame(extended_features).to_parquet(output_features, index=False)

    output_row = source_row.copy()
    output_row["kernel_path"] = str(output_kernel)
    output_row["order_path"] = str(output_order)
    output_row["extension_policy"] = "frozen_feature_projection_preserve_original_block"
    output_row["extension_qc_path"] = str(output_qc)
    pd.DataFrame([output_row]).to_csv(output_manifest, sep="\t", index=False)

    source_ids = set(source_order["env_id"].astype(str))
    added_ids = [value for value in target_ids if value not in source_ids]
    added = extended_features.loc[added_ids] if added_ids else extended_features.iloc[0:0]
    added_nonzero = int(
        np.count_nonzero(np.linalg.norm(added.to_numpy(dtype=np.float64), axis=1) > 0)
    )
    qc = {
        "status": "PASS",
        "protocol_version": "trait_environment_frozen_extension_v1",
        "selection_data": "environment_identifiers_and_frozen_environment_features_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "kernel": args.kernel,
        "trait": trait,
        "source_environment_count": len(source_order),
        "target_environment_count": len(target_order),
        "added_environment_count": len(added_ids),
        "added_environment_nonzero_feature_count": added_nonzero,
        "original_block_max_abs_delta": original_block_delta,
        "original_block_tolerance": args.original_block_tolerance,
        "frozen_feature_count": len(certified_columns),
        "source_artifacts": {
            str(path): file_sha256(path)
            for path in [
                source_manifest_path,
                source_kernel_path,
                source_order_path,
                source_features_path,
                source_scaling_path,
                source_feature_manifest_path,
                target_order_path,
                envdata_path,
                locdata_path,
                window_path,
            ]
        },
        "output_artifacts": {
            str(path): file_sha256(path)
            for path in [output_kernel, output_order, output_features, output_manifest]
        },
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    output_qc.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
