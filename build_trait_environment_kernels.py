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
    zscore_with_missing,
)


ALL_WEATHER_METRICS = {
    "n_days",
    "temperature_mean_c",
    "temperature_max_c",
    "temperature_min_c",
    "gdd_base0_sum",
    "gdd_base5_sum",
    "cold_days_tmin_lt_0",
    "chill_days_tmean_0_10",
    "heat_days_tmax_ge_30",
    "heat_days_tmax_ge_35",
    "precipitation_total_mm",
    "dry_days_precip_lt_1mm",
    "solar_radiation_total_mj_m2",
    "solar_radiation_mean_daily_mj_m2",
    "relative_humidity_mean_pct",
    "wind_speed_2m_mean_m_s",
    "vpd_mean_kpa",
    "vpd_max_kpa",
    "high_vpd_days_gt_1_5",
    "drought_days_precip_lt_1mm_and_vpd_gt_1_5",
}

WATER_ENERGY_METRICS = {
    "n_days",
    "temperature_mean_c",
    "temperature_max_c",
    "temperature_min_c",
    "gdd_base0_sum",
    "gdd_base5_sum",
    "heat_days_tmax_ge_30",
    "heat_days_tmax_ge_35",
    "precipitation_total_mm",
    "dry_days_precip_lt_1mm",
    "solar_radiation_total_mj_m2",
    "solar_radiation_mean_daily_mj_m2",
    "relative_humidity_mean_pct",
    "vpd_mean_kpa",
    "vpd_max_kpa",
    "high_vpd_days_gt_1_5",
    "drought_days_precip_lt_1mm_and_vpd_gt_1_5",
}

TRAIT_SPECS: dict[str, dict[str, object]] = {
    "K_E_DTM_V2": {
        "trait": "DAYS_TO_MATURITY",
        "slug": "dtm",
        "biological_role": "DTM_fixed_sowing_window_thermal_water_and_terminal_stress",
        "windows": [
            "d0_30",
            "d30_60",
            "d60_90",
            "d90_120",
            "d120_150",
            "d150_180",
            "d0_120",
            "d0_150",
            "d0_180",
        ],
        "metrics": ALL_WEATHER_METRICS,
    },
    "K_E_GY_V2": {
        "trait": "GRAIN_YIELD",
        "slug": "gy",
        "biological_role": "grain_yield_fixed_sowing_window_water_energy_and_terminal_stress",
        "windows": [
            "d0_30",
            "d30_60",
            "d60_90",
            "d90_120",
            "d120_150",
            "d150_180",
            "d0_90",
            "d0_120",
            "d0_150",
            "d0_180",
        ],
        "metrics": WATER_ENERGY_METRICS,
    },
    "K_E_TGW_V2": {
        "trait": "1000_GRAIN_WEIGHT",
        "slug": "tgw",
        "biological_role": "thousand_grain_weight_fixed_grain_filling_weather_windows",
        "windows": [
            "d60_90",
            "d90_120",
            "d120_150",
            "d150_180",
            "d0_120",
            "d0_150",
            "d0_180",
        ],
        "metrics": WATER_ENERGY_METRICS,
    },
    "K_E_PH_V2": {
        "trait": "PLANT_HEIGHT",
        "slug": "ph",
        "biological_role": "plant_height_fixed_early_vegetative_weather_windows",
        "windows": ["d0_30", "d30_60", "d60_90", "d90_120", "d0_90", "d0_120"],
        "metrics": WATER_ENERGY_METRICS,
    },
}


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observed_features_for_trait(observed: pd.DataFrame, trait: str) -> pd.DataFrame:
    if trait in {"DAYS_TO_MATURITY", "GRAIN_YIELD"}:
        return observed.copy()
    common_tokens = (
        "PRECIP",
        "PPN_",
        "MOISTURE",
        "IRRIG",
        "sowing_",
        "has_sowing",
        "weather_comment_",
        "has_weather_comment",
    )
    columns = [column for column in observed.columns if any(token in str(column) for token in common_tokens)]
    return observed[columns].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build standalone trait-specific environment kernel experts."
    )
    parser.add_argument(
        "--base-model-dir", type=Path, default=Path("model_kernels/stage1_pedigree_env")
    )
    parser.add_argument("--prefix", default="stage1_pedigree_env")
    parser.add_argument(
        "--window-features",
        type=Path,
        default=Path("environment/agronomic_api_weather_windows.tsv"),
    )
    parser.add_argument("--envdata", type=Path, default=Path("environment/envdata.tsv"))
    parser.add_argument("--locdata", type=Path, default=Path("environment/locdata.tsv"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("model_kernels/trait_environment_v2")
    )
    parser.add_argument("--kernel", action="append", choices=sorted(TRAIT_SPECS))
    parser.add_argument("--minimum-api-coverage", type=float, default=0.50)
    args = parser.parse_args()
    if not 0 <= args.minimum_api_coverage <= 1:
        raise SystemExit("--minimum-api-coverage must be between 0 and 1")

    requested = args.kernel or sorted(TRAIT_SPECS)
    order_path = args.base_model_dir / f"{args.prefix}_K_E_unique_order.tsv"
    order = read_order(order_path)
    env_ids = order["env_id"].astype(str).reset_index(drop=True)
    envdata = pd.read_csv(args.envdata, sep="\t", dtype=str, low_memory=False)
    locdata = pd.read_csv(args.locdata, sep="\t", dtype=str, low_memory=False)
    window_rows = pd.read_csv(
        args.window_features,
        sep="\t",
        dtype=str,
        usecols=lambda column: column in {"env_id", "window_label", "fetch_status"},
        low_memory=False,
    )
    required_window_columns = {"env_id", "window_label"}
    missing_window_columns = sorted(required_window_columns - set(window_rows.columns))
    if missing_window_columns:
        raise SystemExit(
            f"{args.window_features} is missing columns: {missing_window_columns}"
        )
    if "fetch_status" in window_rows.columns:
        window_rows = window_rows[
            window_rows["fetch_status"].astype(str).str.lower().eq("ok")
        ].copy()
    available_windows = set(window_rows["window_label"].dropna().astype(str))
    env_base = base_env_table(envdata, env_ids)
    geo = build_geo(env_base, locdata).reindex(env_ids)
    observed = build_observed_envdata(envdata, env_ids)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    lineage: dict[str, object] = {
        "base_environment_order": str(order_path),
        "base_environment_order_sha256": file_sha256(order_path),
        "weather_window_features": str(args.window_features),
        "weather_window_features_sha256": file_sha256(args.window_features),
        "envdata": str(args.envdata),
        "envdata_sha256": file_sha256(args.envdata),
        "locdata": str(args.locdata),
        "locdata_sha256": file_sha256(args.locdata),
        "kernels": {},
    }

    for kernel_name in requested:
        spec = TRAIT_SPECS[kernel_name]
        trait = str(spec["trait"])
        missing_windows = sorted(set(spec["windows"]) - available_windows)
        if missing_windows:
            raise SystemExit(
                f"{kernel_name} is missing successful weather windows: {missing_windows}"
            )
        model_env_ids = set(env_ids)
        window_coverage = {
            label: float(
                window_rows.loc[
                    window_rows["window_label"].astype(str).eq(label)
                    & window_rows["env_id"].astype(str).isin(model_env_ids),
                    "env_id",
                ].nunique()
                / len(env_ids)
            )
            for label in spec["windows"]
        }
        minimum_window_coverage = min(window_coverage.values())
        if minimum_window_coverage < args.minimum_api_coverage:
            raise SystemExit(
                f"{kernel_name} minimum successful API-window coverage "
                f"{minimum_window_coverage:.3f} is below {args.minimum_api_coverage:.3f}: "
                f"{window_coverage}"
            )
        api_features = build_window_features(
            args.window_features,
            env_ids,
            allowed_labels=set(spec["windows"]),
            allowed_metrics=set(spec["metrics"]),
        )
        feature_sets = {
            "geo": geo,
            "observed_envdata": observed_features_for_trait(observed, trait),
            "api_sowing_windows": api_features,
        }
        features = pd.concat(feature_sets.values(), axis=1)
        features.index = env_ids
        z, scaling = zscore_with_missing(features)
        if z.empty:
            raise SystemExit(f"{kernel_name} has no usable features")
        kernel = kernel_from_features(z)
        sample_n = min(len(kernel), 512)
        sample_index = np.linspace(0, len(kernel) - 1, sample_n, dtype=int)
        sampled_min_eigenvalue = float(
            np.linalg.eigvalsh(kernel[np.ix_(sample_index, sample_index)]).min()
        )

        kernel_path = args.out_dir / f"{kernel_name}.npy"
        kernel_order_path = args.out_dir / f"{kernel_name}_order.tsv"
        feature_path = args.out_dir / f"{kernel_name}_features.parquet"
        np.save(kernel_path, kernel)
        order.to_csv(kernel_order_path, sep="\t", index=False)
        exported_features = feature_export_frame(z)
        exported_features.to_parquet(feature_path, index=False)
        exported_features.to_csv(
            args.out_dir / f"{kernel_name}_features.tsv.gz", sep="\t", index=False
        )
        scaling.to_csv(args.out_dir / f"{kernel_name}_scaling.tsv", sep="\t", index=False)

        feature_manifest_rows = []
        for group, frame in feature_sets.items():
            for column in frame.dropna(axis=1, how="all").columns:
                feature_manifest_rows.append(
                    {"kernel": kernel_name, "trait": trait, "feature_group": group, "feature": column}
                )
        pd.DataFrame(feature_manifest_rows).to_csv(
            args.out_dir / f"{kernel_name}_feature_manifest.tsv", sep="\t", index=False
        )

        manifest_rows.append(
            {
                "kernel": kernel_name,
                "biological_role": spec["biological_role"],
                "kernel_path": str(kernel_path),
                "order_path": str(kernel_order_path),
                "eligible_traits": trait,
                # Candidate experts remain opt-in until the repeated-seed
                # ablation accepts them for the certified baseline.
                "enabled_default": False,
                "interaction_enabled": True,
                "rank": 64,
                "minimum_ledger_coverage": 0.95,
            }
        )
        qc_rows.append(
            {
                "kernel": kernel_name,
                "trait": trait,
                "environment_count": len(env_ids),
                "feature_count": z.shape[1],
                "window_count": len(spec["windows"]),
                "minimum_window_environment_coverage": minimum_window_coverage,
                "api_environment_coverage": float(api_features.notna().any(axis=1).mean()),
                "mean_diagonal": float(np.diag(kernel).mean()),
                "min_diagonal": float(np.diag(kernel).min()),
                "max_diagonal": float(np.diag(kernel).max()),
                "symmetry_max_abs": float(np.max(np.abs(kernel - kernel.T))),
                "sampled_eigenvalue_n": sample_n,
                "sampled_min_eigenvalue": sampled_min_eigenvalue,
                "kernel_sha256": file_sha256(kernel_path),
            }
        )
        lineage["kernels"][kernel_name] = {
            "trait": trait,
            "windows": spec["windows"],
            "window_environment_coverage": window_coverage,
            "weather_metrics": sorted(spec["metrics"]),
            "kernel_sha256": file_sha256(kernel_path),
            "order_sha256": file_sha256(kernel_order_path),
        }

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.out_dir / "trait_environment_kernel_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(qc_rows).to_csv(
        args.out_dir / "trait_environment_kernel_qc.tsv", sep="\t", index=False
    )
    (args.out_dir / "trait_environment_kernel_lineage.json").write_text(
        json.dumps(lineage, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(qc_rows).to_string(index=False), flush=True)
    print(f"Wrote {args.out_dir / 'trait_environment_kernel_manifest.tsv'}", flush=True)


if __name__ == "__main__":
    main()
