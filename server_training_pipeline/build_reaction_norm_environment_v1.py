from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from build_dth_env_features_v2 import (
    build_observed_envdata,
    build_window_features,
)
from build_environment_component_kernels import (
    ID_COLS,
    assert_kernel_valid,
    build_env_trait_matrix,
    build_fetched_weather_feature_sets,
    build_geo_features,
    scale_kernel_mean_diagonal,
    trait_group_columns,
)
from .final_evaluation_contract import file_sha256


TRAITS = [
    "DAYS_TO_HEADING",
    "DAYS_TO_MATURITY",
    "PLANT_HEIGHT",
    "GRAIN_YIELD",
    "1000_GRAIN_WEIGHT",
    "ABOVE_GROUND_BIOMASS",
    "TEST_WEIGHT",
]

WINDOW_TRAITS = {
    "d0_30": {"DAYS_TO_HEADING", "DAYS_TO_MATURITY", "PLANT_HEIGHT", "GRAIN_YIELD", "ABOVE_GROUND_BIOMASS"},
    "d30_60": {"DAYS_TO_HEADING", "DAYS_TO_MATURITY", "PLANT_HEIGHT", "GRAIN_YIELD", "ABOVE_GROUND_BIOMASS"},
    "d60_90": set(TRAITS) - {"TEST_WEIGHT"},
    "d90_120": set(TRAITS),
    "d120_150": {"DAYS_TO_MATURITY", "GRAIN_YIELD", "1000_GRAIN_WEIGHT", "TEST_WEIGHT"},
    "d150_180": {"DAYS_TO_MATURITY", "GRAIN_YIELD", "1000_GRAIN_WEIGHT", "TEST_WEIGHT"},
    "d0_90": {"DAYS_TO_HEADING", "DAYS_TO_MATURITY", "PLANT_HEIGHT", "GRAIN_YIELD", "ABOVE_GROUND_BIOMASS"},
    "d0_120": {"DAYS_TO_HEADING", "DAYS_TO_MATURITY", "PLANT_HEIGHT", "GRAIN_YIELD", "ABOVE_GROUND_BIOMASS"},
    "d0_150": {"DAYS_TO_MATURITY", "GRAIN_YIELD", "1000_GRAIN_WEIGHT"},
    "d0_180": {"DAYS_TO_MATURITY", "GRAIN_YIELD", "1000_GRAIN_WEIGHT", "TEST_WEIGHT"},
}


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def read_ids(path: Path, preferred: str) -> pd.Index:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    column = preferred if preferred in frame.columns else frame.columns[0]
    values = frame[column].fillna("").astype(str).str.strip()
    if values.eq("").any() or values.duplicated().any():
        raise ValueError(f"{path} contains empty or duplicate IDs")
    return pd.Index(values)


def source_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "sha256": file_sha256(path),
    }


def daylength_hours(latitude: pd.Series, day_of_year: pd.Series) -> pd.Series:
    lat = np.deg2rad(pd.to_numeric(latitude, errors="coerce").clip(-66.0, 66.0))
    day = pd.to_numeric(day_of_year, errors="coerce")
    declination = 0.409 * np.sin((2.0 * np.pi * day / 365.0) - 1.39)
    argument = -np.tan(lat) * np.tan(declination)
    omega = np.arccos(np.clip(argument, -1.0, 1.0))
    result = 24.0 * omega / np.pi
    return pd.Series(result, index=latitude.index, dtype=float)


def feature_block(column: str) -> str:
    name = column.lower()
    if any(token in name for token in ("solar", "radiation")):
        return "radiation"
    if any(token in name for token in ("heat", "temperature_max", "vpd", "hot")):
        return "heat"
    if any(token in name for token in ("precip", "dry", "drought", "rain", "water", "moisture", "irrig")):
        return "water"
    if any(token in name for token in ("gdd", "chill", "cold", "frost", "temperature_min", "temperature_mean")):
        return "development"
    return "development"


def regulatory_treatment(column: str) -> str:
    name = column.lower()
    if any(token in name for token in ("heat", "temperature_max", "hot")):
        return "heat"
    if any(token in name for token in ("cold", "chill", "frost", "vernal")):
        return "cold_vernalization"
    if any(token in name for token in ("heavy_rain", "flood", "waterlog")):
        return "water_excess_flood"
    if any(token in name for token in ("dry", "drought", "vpd", "water_balance")):
        return "water_deficit_ABA"
    if "salin" in name:
        return "salinity_NaCl"
    return "none"


def traits_for_feature(block: str, source_feature: str) -> list[str]:
    match = re.search(r"api_(d\d+(?:_\d+)?)_", source_feature)
    if match:
        return sorted(WINDOW_TRAITS.get(match.group(1), set(TRAITS)))
    if block in {"geo", "management", "confidence"}:
        return TRAITS.copy()
    return TRAITS.copy()


def read_climatology(path: Path, env_ids: pd.Index) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(index=env_ids)
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    if "env_id" not in frame:
        raise ValueError(f"{path} is missing env_id")
    metadata = {
        "weather_climatology",
        "climatology_method",
        "climatology_donor_count",
        "climatology_window_inferred",
        "climatology_location_key",
        "climatology_location_key_source",
        "climatology_confidence",
    }
    columns = [column for column in frame.columns if column != "env_id" and column not in metadata]
    result = frame.drop_duplicates("env_id").set_index("env_id")[columns].reindex(env_ids)
    return result.apply(pd.to_numeric, errors="coerce")


def read_confidence(path: Path, env_ids: pd.Index) -> pd.DataFrame:
    expected = [
        "weather_observed",
        "weather_climatology",
        "weather_any_available",
        "window_inferred",
        "climatology_window_inferred",
        "coordinates_inferred",
    ]
    if not path.exists():
        return pd.DataFrame(0.0, index=env_ids, columns=expected)
    frame = pd.read_csv(path, sep="\t", dtype=str).drop_duplicates("env_id").set_index("env_id")
    output = pd.DataFrame(index=env_ids)
    for column in expected:
        values = frame.get(column, pd.Series("", index=frame.index)).reindex(env_ids)
        output[column] = values.fillna("").str.lower().isin({"1", "true", "yes", "y"}).astype(float)
    return output


def standardize_fold_local(
    features: pd.DataFrame, fit_ids: pd.Index
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference = features.loc[fit_ids].replace([np.inf, -np.inf], np.nan)
    target = features.replace([np.inf, -np.inf], np.nan)
    parts: list[pd.Series] = []
    scaling_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for column in target.columns:
        values = pd.to_numeric(target[column], errors="coerce")
        fit_values = pd.to_numeric(reference[column], errors="coerce")
        finite_fit = fit_values[np.isfinite(fit_values)]
        row = {
            "feature": column,
            "fit_environment_count": len(reference),
            "fit_finite_count": len(finite_fit),
            "target_missing_count": int(values.isna().sum()),
            "imputation": "outer_training_median",
            "imputation_value": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "status": "",
        }
        if finite_fit.empty:
            row["status"] = "dropped_no_finite_outer_training_values"
            scaling_rows.append(row)
            continue
        fill = float(finite_fit.median())
        filled = values.fillna(fill).astype(float)
        fit_filled = fit_values.fillna(fill).astype(float)
        mean = float(fit_filled.mean())
        std = float(fit_filled.std(ddof=0))
        row.update({"imputation_value": fill, "mean": mean, "std": std})
        if not np.isfinite(std) or std <= 0:
            row["status"] = "dropped_constant_in_outer_training"
            scaling_rows.append(row)
            continue
        parts.append(((filled - mean) / std).astype(np.float32).rename(column))
        row["status"] = "retained"
        scaling_rows.append(row)
        missing = values.isna().astype(np.float32)
        fit_missing = missing.loc[fit_ids]
        missing_std = float(fit_missing.std(ddof=0))
        if missing.sum() > 0 and np.isfinite(missing_std) and missing_std > 0:
            missing_name = f"{column}__missing"
            parts.append(((missing - float(fit_missing.mean())) / missing_std).rename(missing_name))
            missing_rows.append(
                {
                    "feature": missing_name,
                    "source_feature": column,
                    "fit_missing_fraction": float(fit_missing.mean()),
                    "target_missing_count": int(missing.sum()),
                }
            )
    if not parts:
        raise ValueError("No environment features remain after outer-training scaling")
    standardized = pd.concat(parts, axis=1).astype(np.float32).copy()
    if not np.isfinite(standardized.to_numpy(dtype=float)).all():
        raise ValueError("E_REACTION_NORM_V1 contains non-finite standardized values")
    return standardized, pd.DataFrame(scaling_rows), pd.DataFrame(missing_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a phenotype-blind fold-local reaction-norm environment matrix."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--environment-input-dir", type=Path, default=Path("environment"))
    parser.add_argument("--weather-dir", type=Path, required=True)
    parser.add_argument("--fold-environment-dir", type=Path, required=True)
    parser.add_argument(
        "--window-features",
        type=Path,
        default=Path("environment/agronomic_api_weather_windows.tsv"),
    )
    parser.add_argument("--fit-environment-ids", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    protocol_path = resolve(root, args.protocol)
    environment_input = resolve(root, args.environment_input_dir)
    weather_dir = resolve(root, args.weather_dir)
    fold_environment = resolve(root, args.fold_environment_dir)
    window_path = resolve(root, args.window_features)
    fit_path = resolve(root, args.fit_environment_ids)
    out_dir = resolve(root, args.out_dir)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_inner_validation":
        raise SystemExit("Environment protocol is not frozen before inner validation")
    if protocol.get("phenotype_values_read") is not False:
        raise SystemExit("Environment protocol does not declare phenotype-blind construction")

    order_path = fold_environment / "env_kernel_sample_order.tsv"
    env_ids = read_ids(order_path, "env_id")
    fit_ids = read_ids(fit_path, "env_id")
    unknown_fit = fit_ids.difference(env_ids)
    if len(unknown_fit):
        raise SystemExit(f"Outer-training environment manifest has {len(unknown_fit)} unknown IDs")
    fit_ids = env_ids[env_ids.isin(fit_ids)]
    if len(fit_ids) < 2:
        raise SystemExit("At least two outer-training environments are required")
    component_qc_path = fold_environment / "K_E.qc.json"
    if not component_qc_path.exists():
        raise SystemExit("Fold-local generic environment component provenance is absent")
    component_qc = json.loads(component_qc_path.read_text(encoding="utf-8"))
    if component_qc.get("feature_fit_scope") != "training_environments_only" or component_qc.get(
        "fit_environment_ids_sha256"
    ) != file_sha256(fit_path):
        raise SystemExit(
            "Fold-local generic environment components were not fitted with the requested outer-training IDs"
        )

    envdata_path = environment_input / "envdata.tsv"
    locdata_path = environment_input / "locdata.tsv"
    envdata = pd.read_csv(envdata_path, sep="\t", dtype=str, low_memory=False)
    locdata = pd.read_csv(locdata_path, sep="\t", dtype=str, low_memory=False)
    env_base = envdata[ID_COLS].drop_duplicates().copy()
    env_base["env_id"] = env_base[ID_COLS].apply(
        lambda row: "|".join("" if pd.isna(value) else str(value) for value in row), axis=1
    )
    env_base = env_base[env_base["env_id"].isin(env_ids)]

    geo = build_geo_features(envdata, locdata, env_ids).rename(columns={"altitude": "elevation_m"})
    observed = build_observed_envdata(envdata, pd.Series(env_ids, dtype=str)).reindex(env_ids)
    geo["photoperiod_at_sowing_hours"] = daylength_hours(
        geo["latitude"], observed["sowing_dayofyear"]
    )

    trait_matrix = build_env_trait_matrix(envdata).reindex(env_ids)
    management_columns = trait_group_columns(trait_matrix.columns, "mgmt")
    management = trait_matrix[management_columns].copy()
    management["management_missing_fraction"] = management.isna().mean(axis=1)

    weather, stress = build_fetched_weather_feature_sets(env_ids, environment_dir=weather_dir)
    climatology_path = fold_environment / "trial_weather_features_climatology.tsv"
    climatology_lineage_path = fold_environment / "weather_climatology_lineage.json"
    if climatology_path.exists():
        if not climatology_lineage_path.exists():
            raise SystemExit("Fold climatology exists without donor/scaling lineage")
        climatology_lineage = json.loads(
            climatology_lineage_path.read_text(encoding="utf-8")
        )
        fit_sha256 = file_sha256(fit_path)
        if (
            climatology_lineage.get("donor_scope") != "outer_training_only"
            or climatology_lineage.get("donor_environment_ids_sha256") != fit_sha256
            or climatology_lineage.get("fit_environment_ids_sha256") != fit_sha256
        ):
            raise SystemExit(
                "Fold climatology donors or scaling were not restricted to the requested outer-training IDs"
            )
    climatology = read_climatology(climatology_path, env_ids)
    generic = pd.concat([weather, stress], axis=1)
    generic = generic.loc[:, ~generic.columns.duplicated()]
    if not climatology.empty:
        generic = generic.combine_first(climatology.reindex(columns=generic.columns))

    windows = build_window_features(window_path, pd.Series(env_ids, dtype=str))
    confidence_path = fold_environment / "environment_expert_coverage.tsv"
    confidence = read_confidence(confidence_path, env_ids)
    confidence["management_missing"] = management.drop(
        columns=["management_missing_fraction"], errors="ignore"
    ).isna().all(axis=1).astype(float)
    confidence["coordinates_missing"] = geo[["latitude", "longitude"]].isna().any(axis=1).astype(float)

    blocks: dict[str, pd.DataFrame] = {
        "geo": geo,
        "management": management,
        "confidence": confidence,
    }
    dynamic_blocks: dict[str, list[pd.Series]] = {}
    for source_name, source in (("generic", generic), ("window", windows)):
        for column in source.columns:
            block = feature_block(str(column))
            dynamic_blocks.setdefault(block, []).append(
                pd.to_numeric(source[column], errors="coerce").rename(
                    f"{source_name}__{column}"
                )
            )
    for block, parts in dynamic_blocks.items():
        dynamic = pd.concat(parts, axis=1)
        blocks[block] = pd.concat(
            [blocks.get(block, pd.DataFrame(index=env_ids)), dynamic], axis=1
        ).copy()
    development_observed = observed[
        [column for column in observed.columns if "sowing" in str(column).lower()]
    ].copy()
    development_observed.columns = [
        f"observed__{column}" for column in development_observed.columns
    ]
    blocks["development"] = pd.concat(
        [
            blocks.get("development", pd.DataFrame(index=env_ids)),
            development_observed,
        ],
        axis=1,
    ).copy()
    observed_water_columns = [
        column
        for column in observed.columns
        if any(token in str(column).upper() for token in ("PRECIP", "PPN_", "MOISTURE", "IRRIG"))
    ]
    if observed_water_columns:
        observed_water = observed[observed_water_columns].copy()
        observed_water.columns = [
            f"observed__{column}" for column in observed_water_columns
        ]
        blocks["water"] = pd.concat(
            [blocks.get("water", pd.DataFrame(index=env_ids)), observed_water],
            axis=1,
        ).copy()

    raw_parts = []
    base_manifest: dict[str, dict[str, object]] = {}
    for block, frame in blocks.items():
        frame = frame.reindex(env_ids)
        for column in frame.columns:
            exported = f"{block}__{column}"
            column_text = str(column)
            if block == "geo":
                source_artifact = "envdata.tsv+locdata.tsv"
            elif block == "management" or column_text.startswith("observed__"):
                source_artifact = "envdata.tsv"
            elif block == "confidence":
                source_artifact = "environment_expert_coverage.tsv"
            elif column_text.startswith("window__"):
                source_artifact = window_path.name
            else:
                source_artifact = "observed_api_weather_or_fold_climatology"
            raw_parts.append(pd.to_numeric(frame[column], errors="coerce").rename(exported))
            base_manifest[exported] = {
                "feature_block": block,
                "source_feature": column_text,
                "source_artifact": source_artifact,
                "eligible_traits": ",".join(traits_for_feature(block, column_text)),
                "regulatory_treatment": regulatory_treatment(column_text),
            }
    raw = pd.concat(raw_parts, axis=1)
    raw = (
        raw.loc[:, ~raw.columns.duplicated()]
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )
    standardized, scaling, missing_manifest = standardize_fold_local(raw, fit_ids)

    manifest_rows = []
    for column in standardized.columns:
        source = column.removesuffix("__missing")
        metadata = base_manifest[source]
        manifest_rows.append(
            {
                "feature": column,
                "source_feature": metadata["source_feature"],
                "source_artifact": metadata["source_artifact"],
                "feature_block": metadata["feature_block"],
                "eligible_traits": metadata["eligible_traits"],
                "regulatory_treatment": metadata["regulatory_treatment"],
                "is_missingness_indicator": column.endswith("__missing"),
                "phenotype_derived": False,
                "fit_partition": "outer_training_environments_only",
            }
        )
    manifest = pd.DataFrame(manifest_rows)

    matrix = standardized.to_numpy(dtype=np.float32)
    kernel_raw = ((matrix @ matrix.T) / max(matrix.shape[1], 1)).astype(np.float32)
    kernel_raw = ((kernel_raw + kernel_raw.T) * np.float32(0.5)).astype(np.float32)
    position = pd.Series(np.arange(len(env_ids), dtype=int), index=env_ids)
    fit_positions = position.loc[fit_ids].to_numpy(dtype=int)
    kernel, raw_mean_diag, scaled_mean_diag = scale_kernel_mean_diagonal(
        kernel_raw, reference_indices=fit_positions
    )
    assert_kernel_valid(kernel, "K_E_REACTION_NORM_V1")

    out_dir.mkdir(parents=True, exist_ok=True)
    order = pd.DataFrame(
        {"env_id": env_ids, "compact_kernel_index": np.arange(len(env_ids), dtype=np.int32)}
    )
    raw.reset_index(names="env_id").to_parquet(
        out_dir / "E_REACTION_NORM_V1_raw.parquet", index=False
    )
    standardized.reset_index(names="env_id").to_parquet(
        out_dir / "E_REACTION_NORM_V1.parquet", index=False
    )
    order.to_csv(out_dir / "E_REACTION_NORM_V1_order.tsv", sep="\t", index=False)
    order.to_csv(out_dir / "K_E_REACTION_NORM_V1_order.tsv", sep="\t", index=False)
    manifest.to_csv(out_dir / "E_REACTION_NORM_V1_feature_manifest.tsv", sep="\t", index=False)
    scaling.to_csv(out_dir / "E_REACTION_NORM_V1_scaling.tsv", sep="\t", index=False)
    missing_manifest.to_csv(
        out_dir / "E_REACTION_NORM_V1_missingness_indicators.tsv", sep="\t", index=False
    )
    np.save(out_dir / "K_E_REACTION_NORM_V1.raw.npy", kernel_raw)
    np.save(out_dir / "K_E_REACTION_NORM_V1.npy", kernel)
    pd.DataFrame(
        [
            {
                "kernel": "K_E_REACTION_NORM_V1",
                "biological_role": "explicit_phenotype_blind_reaction_norm_environment_axes",
                "kernel_path": str((out_dir / "K_E_REACTION_NORM_V1.npy").resolve()),
                "order_path": str((out_dir / "K_E_REACTION_NORM_V1_order.tsv").resolve()),
                "eligible_traits": "*",
                "enabled_default": False,
                "interaction_enabled": True,
                "rank": min(128, standardized.shape[1]),
                "minimum_ledger_coverage": 0.95,
                "coverage_basis": "unique_entities",
                "minimum_eligible_entities": 2,
                "minimum_training_entities": 2,
            }
        ]
    ).to_csv(
        out_dir / "reaction_norm_environment_kernel_manifest.tsv",
        sep="\t",
        index=False,
    )

    missingness_rows = []
    for block, columns in manifest.groupby("feature_block")["feature"]:
        source_columns = [value for value in columns if not value.endswith("__missing")]
        missingness_rows.append(
            {
                "feature_block": block,
                "retained_feature_count": len(columns),
                "source_feature_count": len(source_columns),
                "outer_training_environment_count": len(fit_ids),
                "target_environment_count": len(env_ids),
                "raw_missing_fraction_outer_training": float(
                    raw.loc[fit_ids, source_columns].isna().to_numpy().mean()
                ) if source_columns else 0.0,
                "raw_missing_fraction_all": float(raw[source_columns].isna().to_numpy().mean())
                if source_columns else 0.0,
            }
        )
    pd.DataFrame(missingness_rows).to_csv(
        out_dir / "E_REACTION_NORM_V1_block_missingness.tsv", sep="\t", index=False
    )

    sources = {
        "protocol": source_identity(protocol_path),
        "environment_order": source_identity(order_path),
        "fit_environment_ids": source_identity(fit_path),
        "envdata": source_identity(envdata_path),
        "locdata": source_identity(locdata_path),
        "window_features": source_identity(window_path),
        "generic_environment_provenance": source_identity(component_qc_path),
    }
    for label, path in {
        "climatology": climatology_path,
        "confidence": confidence_path,
        "climatology_lineage": climatology_lineage_path,
    }.items():
        if path.exists():
            sources[label] = source_identity(path)
    provenance = {
        "status": "BUILT_PENDING_CERTIFICATION",
        "protocol_version": protocol["protocol_version"],
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "actual_heading_or_maturity_dates_used": False,
        "weather_window_policy": "fixed_sowing_relative_only",
        "feature_scaling_fit_partition": "outer_training_environments_only",
        "climatology_donor_partition": "outer_training_environments_only",
        "environment_count": len(env_ids),
        "fit_environment_count": len(fit_ids),
        "feature_count": standardized.shape[1],
        "kernel_mean_diagonal_raw_fit": raw_mean_diag,
        "kernel_mean_diagonal_scaled_fit": scaled_mean_diag,
        "sources": sources,
    }
    (out_dir / "E_REACTION_NORM_V1_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
