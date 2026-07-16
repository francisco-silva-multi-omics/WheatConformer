from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from build_environment_component_kernels import (
    assert_kernel_valid,
    build_fetched_weather_feature_sets,
    scale_kernel_mean_diagonal,
    standardized_kernel,
)


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def normalized_key(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]+", "_", str(value).strip().upper()).strip("_")


def build_location_keys(
    manifest: pd.DataFrame, registry_path: Path | None
) -> pd.DataFrame:
    output = manifest[["env_id"]].copy()
    country = manifest.get("Country", pd.Series("", index=manifest.index)).map(normalized_key)
    loc_no = manifest.get("Loc_no", pd.Series("", index=manifest.index)).map(normalized_key)
    trial_dir = manifest.get("trial_dir", pd.Series("", index=manifest.index)).map(normalized_key)
    stable = np.where(
        country.ne("") & loc_no.ne(""),
        "COUNTRY_LOCNO:" + country + "|" + loc_no,
        np.where(
            trial_dir.ne("") & loc_no.ne(""),
            "TRIAL_LOCNO:" + trial_dir + "|" + loc_no,
            "",
        ),
    )
    output["location_key"] = stable
    output["location_key_source"] = np.where(output["location_key"].ne(""), "stable_loc_no", "missing")

    if registry_path is not None and registry_path.exists():
        registry = pd.read_csv(registry_path, sep="\t", dtype=str)
        required = {"env_id", "curated_location_id", "review_status"}
        missing = sorted(required.difference(registry.columns))
        if missing:
            raise ValueError(f"{registry_path} is missing columns: {missing}")
        approved = registry[
            registry["review_status"].fillna("").str.lower().isin({"approved", "reviewed"})
        ][["env_id", "curated_location_id"]].drop_duplicates("env_id")
        output = output.merge(approved, on="env_id", how="left")
        curated = output["curated_location_id"].fillna("").map(normalized_key)
        use_curated = curated.ne("")
        output.loc[use_curated, "location_key"] = "CURATED:" + curated.loc[use_curated]
        output.loc[use_curated, "location_key_source"] = "curated_location_registry"
        output = output.drop(columns=["curated_location_id"])
    return output


def group_statistics(
    donors: pd.DataFrame, keys: list[str], feature_columns: list[str]
) -> tuple[dict[tuple[object, ...], np.ndarray], dict[tuple[object, ...], int]]:
    means: dict[tuple[object, ...], np.ndarray] = {}
    counts: dict[tuple[object, ...], int] = {}
    for key, group in donors.groupby(keys, dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        means[key_tuple] = group[feature_columns].mean(axis=0).to_numpy(dtype=np.float32)
        counts[key_tuple] = int(len(group))
    return means, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a separate location-season climatology kernel for API-missing environments."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--environment-dir", type=Path, default=Path("environment"))
    parser.add_argument("--weather-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--location-registry", type=Path, default=None)
    parser.add_argument("--minimum-donors", type=int, default=3)
    args = parser.parse_args()

    root = args.root.resolve()
    environment_dir = resolve(root, args.environment_dir)
    weather_dir = resolve(root, args.weather_dir)
    audit_dir = resolve(root, args.audit_dir)
    out_dir = resolve(root, args.out_dir)
    registry_path = (
        None if args.location_registry is None else resolve(root, args.location_registry)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    order = pd.read_csv(environment_dir / "env_kernel_sample_order.tsv", sep="\t", dtype=str)
    env_ids = pd.Index(order["env_id"].fillna("").astype(str))
    manifest = pd.read_csv(
        weather_dir / "trial_weather_fetch_manifest.tsv", sep="\t", dtype=str, low_memory=False
    ).drop_duplicates("env_id", keep="first")
    audit = pd.read_csv(
        audit_dir / "weather_recovery_environment_audit.tsv", sep="\t", dtype=str, low_memory=False
    ).drop_duplicates("env_id", keep="first")
    observed = audit.set_index("env_id")["weather_observed"].str.lower().eq("true").reindex(env_ids).fillna(False)

    weather, stress = build_fetched_weather_feature_sets(env_ids, environment_dir=weather_dir)
    features = pd.concat([weather, stress], axis=1)
    features = features.loc[:, ~features.columns.duplicated()].apply(pd.to_numeric, errors="coerce")
    feature_columns = features.columns.tolist()
    if not feature_columns:
        raise SystemExit("No observed API weather/stress features are available for climatology donors")

    metadata = order.merge(manifest, on="env_id", how="left", validate="one_to_one")
    metadata = metadata.merge(build_location_keys(metadata, registry_path), on="env_id", how="left")
    metadata["season_month"] = pd.to_datetime(
        metadata.get("weather_start_date"), errors="coerce"
    ).dt.month
    donor_frame = metadata[["env_id", "location_key", "season_month"]].set_index("env_id")
    donor_frame = donor_frame.join(features)
    donor_frame["weather_observed"] = observed
    donors = donor_frame[
        donor_frame["weather_observed"]
        & donor_frame["location_key"].fillna("").ne("")
        & donor_frame[feature_columns].notna().any(axis=1)
    ].copy()
    donors["season_quarter"] = ((donors["season_month"] - 1) // 3 + 1).astype("Int64")

    month_mode = (
        donors.dropna(subset=["season_month"])
        .groupby("location_key")["season_month"]
        .agg(lambda values: int(values.mode().iloc[0]))
        .to_dict()
    )
    exact_means, exact_counts = group_statistics(
        donors.dropna(subset=["season_month"]),
        ["location_key", "season_month"],
        feature_columns,
    )
    quarter_means, quarter_counts = group_statistics(
        donors.dropna(subset=["season_quarter"]),
        ["location_key", "season_quarter"],
        feature_columns,
    )

    climatology = pd.DataFrame(np.nan, index=env_ids, columns=feature_columns, dtype=np.float32)
    provenance_rows = []
    target_metadata = metadata.set_index("env_id")
    for env_id in env_ids[~observed.to_numpy()]:
        row = target_metadata.loc[env_id]
        location_key = str(row.get("location_key", "") or "")
        month = pd.to_numeric(pd.Series([row.get("season_month")]), errors="coerce").iloc[0]
        window_inferred = False
        if pd.isna(month) and location_key in month_mode:
            month = month_mode[location_key]
            window_inferred = True
        donor_count = 0
        method = "unrecovered"
        values = None
        if location_key and pd.notna(month):
            exact_key = (location_key, float(month))
            donor_count = exact_counts.get(exact_key, 0)
            if donor_count >= args.minimum_donors:
                values = exact_means[exact_key]
                method = "location_calendar_month_climatology"
            else:
                quarter = int((int(month) - 1) // 3 + 1)
                quarter_key = (location_key, quarter)
                donor_count = quarter_counts.get(quarter_key, 0)
                if donor_count >= args.minimum_donors:
                    values = quarter_means[quarter_key]
                    method = "location_calendar_quarter_climatology"
        if values is not None:
            climatology.loc[env_id, feature_columns] = values
        provenance_rows.append(
            {
                "env_id": env_id,
                "weather_climatology": values is not None,
                "climatology_method": method,
                "climatology_donor_count": donor_count,
                "climatology_window_inferred": window_inferred,
                "climatology_location_key": location_key,
                "climatology_location_key_source": row.get("location_key_source", "missing"),
                "climatology_confidence": (
                    "high" if donor_count >= 5 and values is not None else "medium" if values is not None else "none"
                ),
            }
        )

    provenance = pd.DataFrame(provenance_rows).set_index("env_id").reindex(env_ids)
    climatology_available = provenance["weather_climatology"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"}
    )
    if not climatology_available.any():
        raise SystemExit("No missing environments met the location-season climatology criteria")

    kernel_raw, standardized, scaling = standardized_kernel(climatology)
    kernel, raw_mean_diag, scaled_mean_diag = scale_kernel_mean_diagonal(kernel_raw)
    assert_kernel_valid(kernel, "K_climatology")
    np.save(out_dir / "K_climatology.raw.npy", kernel_raw)
    np.save(out_dir / "K_climatology.npy", kernel)
    order.to_csv(out_dir / "env_kernel_sample_order.tsv", sep="\t", index=False)
    standardized_output = standardized.reset_index(names="env_id")
    try:
        standardized_output.to_parquet(
            out_dir / "env_features_climatology.parquet", index=False
        )
    except Exception as exc:
        standardized_output.to_csv(
            out_dir / "env_features_climatology.tsv.gz", sep="\t", index=False
        )
        (out_dir / "env_features_climatology_parquet_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
    scaling.to_csv(out_dir / "climatology_feature_scaling.tsv", sep="\t", index=False)

    feature_output = climatology.reset_index(names="env_id").merge(
        provenance.reset_index(), on="env_id", how="left", validate="one_to_one"
    )
    feature_output.to_csv(
        out_dir / "trial_weather_features_climatology.tsv", sep="\t", index=False
    )
    coverage = audit[["env_id", "weather_observed", "window_inferred", "coordinates_inferred"]].copy()
    for column in ["weather_observed", "window_inferred", "coordinates_inferred"]:
        coverage[column] = coverage[column].fillna("").astype(str).str.lower().eq("true")
    coverage = coverage.merge(
        provenance.reset_index()[
            [
                "env_id",
                "weather_climatology",
                "climatology_method",
                "climatology_donor_count",
                "climatology_window_inferred",
                "climatology_confidence",
            ]
        ],
        on="env_id",
        how="left",
        validate="one_to_one",
    )
    coverage["weather_climatology"] = coverage["weather_climatology"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"}
    )
    coverage["weather_api_available"] = coverage["weather_observed"]
    coverage["weather_any_available"] = (
        coverage["weather_observed"] | coverage["weather_climatology"]
    )
    coverage.to_csv(out_dir / "environment_expert_coverage.tsv", sep="\t", index=False)

    qc = pd.DataFrame(
        [
            {"metric": "environment_count", "value": len(env_ids)},
            {"metric": "api_observed_environment_count", "value": int(observed.sum())},
            {"metric": "api_missing_environment_count", "value": int((~observed).sum())},
            {"metric": "climatology_recovered_environment_count", "value": int(climatology_available.sum())},
            {
                "metric": "still_unrecovered_environment_count",
                "value": int((~observed).sum() - climatology_available.sum()),
            },
            {"metric": "climatology_feature_count", "value": standardized.shape[1]},
            {"metric": "K_climatology_mean_diagonal_raw", "value": raw_mean_diag},
            {"metric": "K_climatology_mean_diagonal_scaled", "value": scaled_mean_diag},
            {"metric": "minimum_donors", "value": args.minimum_donors},
        ]
    )
    qc.to_csv(out_dir / "weather_climatology_qc.tsv", sep="\t", index=False)
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
