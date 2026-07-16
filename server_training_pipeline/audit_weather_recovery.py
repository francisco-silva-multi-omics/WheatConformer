from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .split_utils import make_split, split_group_column, split_leakage_record
except ImportError:
    from split_utils import make_split, split_group_column, split_leakage_record


EVALUATION_SPLITS = ("train", "val", "test")


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[column].fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y"}
    )


def feature_status(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["env_id", f"observed_{source}"])
    frame = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    if "env_id" not in frame.columns:
        raise ValueError(f"{path} is missing env_id")
    ok = (
        frame["fetch_status"].fillna("").astype(str).str.lower().eq("ok")
        if "fetch_status" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    output = pd.DataFrame(
        {
            "env_id": frame["env_id"].fillna("").astype(str),
            f"observed_{source}": ok,
        }
    )
    return output.groupby("env_id", as_index=False)[f"observed_{source}"].max()


def failure_request_ids(environment_dir: Path) -> set[str]:
    values: set[str] = set()
    for path in [
        environment_dir / "trial_weather_fetch_nasa_power_failures.tsv",
        environment_dir / "trial_weather_fetch_openmeteo_failures.tsv",
    ]:
        if not path.exists():
            continue
        frame = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
        if "weather_request_id" in frame.columns:
            values.update(frame["weather_request_id"].dropna().astype(str))
    return values


def request_id(frame: pd.DataFrame) -> pd.Series:
    latitude = pd.to_numeric(frame.get("latitude"), errors="coerce")
    longitude = pd.to_numeric(frame.get("longitude"), errors="coerce")
    start = frame.get("weather_start_date", pd.Series("", index=frame.index)).fillna("")
    end = frame.get("weather_end_date", pd.Series("", index=frame.index)).fillna("")
    output = pd.Series("", index=frame.index, dtype=object)
    valid = latitude.notna() & longitude.notna() & start.ne("") & end.ne("")
    output.loc[valid] = [
        f"{lat:.5f}|{lon:.5f}|{begin}|{finish}"
        for lat, lon, begin, finish in zip(
            latitude.loc[valid], longitude.loc[valid], start.loc[valid], end.loc[valid]
        )
    ]
    return output


def classify_environment_coverage(
    order: pd.DataFrame,
    manifest: pd.DataFrame,
    nasa: pd.DataFrame,
    openmeteo: pd.DataFrame,
    failures: set[str],
    model_env_ids: set[str],
    nasa_start: pd.Timestamp,
) -> pd.DataFrame:
    audit = order[["env_id"]].copy()
    audit["env_id"] = audit["env_id"].fillna("").astype(str)
    manifest = manifest.drop_duplicates("env_id", keep="first").copy()
    audit = audit.merge(manifest, on="env_id", how="left", indicator="manifest_merge")
    audit = audit.merge(nasa, on="env_id", how="left")
    audit = audit.merge(openmeteo, on="env_id", how="left")
    audit["observed_nasa"] = bool_series(audit, "observed_nasa")
    audit["observed_openmeteo"] = bool_series(audit, "observed_openmeteo")
    audit["weather_observed"] = audit["observed_nasa"] | audit["observed_openmeteo"]
    audit["weather_source"] = np.select(
        [audit["observed_nasa"], audit["observed_openmeteo"]],
        ["nasa_power_daily", "openmeteo_era5"],
        default="missing",
    )
    audit["manifest_present"] = audit["manifest_merge"].eq("both")
    audit["has_fetch_window"] = bool_series(audit, "has_fetch_window")
    audit["has_fetch_coordinates"] = bool_series(audit, "has_fetch_coordinates")
    audit["ready_to_fetch"] = bool_series(audit, "ready_to_fetch")
    audit["window_inferred"] = bool_series(audit, "window_inferred")
    audit["coordinates_inferred"] = bool_series(audit, "coordinates_inferred")
    audit["request_id"] = request_id(audit)
    audit["fetch_failed"] = audit["request_id"].isin(failures)
    start = pd.to_datetime(audit.get("weather_start_date"), errors="coerce")
    audit["outside_nasa_coverage"] = start.notna() & start.lt(nasa_start)

    missing_window = ~audit["has_fetch_window"]
    missing_coordinates = ~audit["has_fetch_coordinates"]
    audit["coverage_cause"] = np.select(
        [
            audit["weather_observed"],
            ~audit["manifest_present"],
            missing_window & missing_coordinates,
            missing_window,
            missing_coordinates,
            audit["outside_nasa_coverage"],
            audit["fetch_failed"],
            audit["ready_to_fetch"],
        ],
        [
            "observed_api",
            "missing_manifest",
            "missing_window_and_coordinates",
            "missing_fetch_window",
            "missing_coordinates",
            "dates_outside_nasa_coverage",
            "fetch_failed",
            "ready_not_fetched",
        ],
        default="not_ready_unclassified",
    )
    audit["used_by_pedigree_model"] = audit["env_id"].isin(model_env_ids)
    audit["weather_climatology"] = False
    audit["weather_any_available"] = audit["weather_observed"]
    audit = audit.drop(columns=["manifest_merge"])
    return audit


def apply_climatology_coverage(audit: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    if path is None:
        return audit
    coverage = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    required = {"env_id", "weather_climatology"}
    missing = sorted(required.difference(coverage.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if coverage["env_id"].fillna("").duplicated().any():
        raise ValueError(f"{path} contains duplicate env_id values")
    climate = pd.DataFrame(
        {
            "env_id": coverage["env_id"].fillna("").astype(str),
            "weather_climatology_mask": bool_series(coverage, "weather_climatology"),
        }
    )
    output = audit.drop(columns=["weather_climatology", "weather_any_available"]).merge(
        climate, on="env_id", how="left", validate="one_to_one"
    )
    output["weather_climatology"] = output["weather_climatology_mask"].fillna(False)
    output["weather_any_available"] = (
        output["weather_observed"] | output["weather_climatology"]
    )
    return output.drop(columns=["weather_climatology_mask"])


def scope_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, selected in [
        ("all_environments", audit),
        ("pedigree_model_environments", audit[audit["used_by_pedigree_model"]]),
    ]:
        counts = selected["coverage_cause"].value_counts(dropna=False)
        for cause, count in counts.items():
            rows.append(
                {
                    "scope": scope,
                    "coverage_cause": cause,
                    "environment_count": int(count),
                    "scope_environment_count": int(len(selected)),
                    "fraction": float(count / len(selected)) if len(selected) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def availability_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, selected in [
        ("all_environments", audit),
        ("pedigree_model_environments", audit[audit["used_by_pedigree_model"]]),
    ]:
        total = len(selected)
        for status in ["weather_observed", "weather_climatology", "weather_any_available"]:
            count = int(selected[status].sum())
            rows.append(
                {
                    "scope": scope,
                    "availability": status,
                    "environment_count": count,
                    "scope_environment_count": total,
                    "fraction": float(count / total) if total else np.nan,
                }
            )
        rows.append(
            {
                "scope": scope,
                "availability": "weather_still_missing",
                "environment_count": int((~selected["weather_any_available"]).sum()),
                "scope_environment_count": total,
                "fraction": float((~selected["weather_any_available"]).mean()) if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def dimension_summary(audit: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = audit.copy()
    frame[column] = frame.get(column, pd.Series("", index=frame.index)).fillna("").astype(str)
    return (
        frame.groupby(["used_by_pedigree_model", column], dropna=False)
        .agg(
            environment_count=("env_id", "nunique"),
            observed_environment_count=("weather_observed", "sum"),
            climatology_environment_count=("weather_climatology", "sum"),
            any_weather_environment_count=("weather_any_available", "sum"),
        )
        .reset_index()
        .assign(
            observed_fraction=lambda value: value["observed_environment_count"]
            / value["environment_count"],
            any_weather_fraction=lambda value: value["any_weather_environment_count"]
            / value["environment_count"],
        )
    )


def prepare_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    ledger = ledger.copy()
    if "environment_id" not in ledger.columns:
        if "env_kernel_id" in ledger.columns:
            ledger["environment_id"] = ledger["env_kernel_id"]
        else:
            raise ValueError("Ledger must contain environment_id or env_kernel_id")
    if "env_kernel_id" not in ledger.columns:
        ledger["env_kernel_id"] = ledger["environment_id"]
    if "panel_sample_id" not in ledger.columns and "genotype_id" in ledger.columns:
        ledger["panel_sample_id"] = ledger["genotype_id"]
    return ledger


def trait_coverage_summary(ledger: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    status = audit.set_index("env_id")[[
        "weather_observed", "weather_climatology", "weather_any_available"
    ]]
    frame = ledger[["trait_name_canonical", "environment_id"]].drop_duplicates().copy()
    for column in status.columns:
        frame[column] = frame["environment_id"].map(status[column]).fillna(False)
    return (
        frame.groupby("trait_name_canonical", dropna=False)
        .agg(
            environment_count=("environment_id", "nunique"),
            observed_environment_count=("weather_observed", "sum"),
            climatology_environment_count=("weather_climatology", "sum"),
            any_weather_environment_count=("weather_any_available", "sum"),
        )
        .reset_index()
        .assign(
            observed_fraction=lambda value: value["observed_environment_count"]
            / value["environment_count"],
            any_weather_fraction=lambda value: value["any_weather_environment_count"]
            / value["environment_count"],
        )
    )


def split_coverage(
    ledger: pd.DataFrame,
    audit: pd.DataFrame,
    seeds: list[int],
    test_fraction: float,
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    status = audit.set_index("env_id")[[
        "weather_observed", "weather_climatology", "weather_any_available"
    ]]
    coverage_rows = []
    leakage_rows = []
    group_col = split_group_column("gho_environment")
    for seed in seeds:
        train, val, test = make_split(
            ledger, "gho_environment", seed, test_fraction, val_fraction, group_col
        )
        leakage = split_leakage_record(
            ledger, seed, "gho_environment", train, val, test, group_col
        )
        leakage_rows.append(leakage)
        for split, indices in zip(EVALUATION_SPLITS, [train, val, test]):
            selected = ledger.iloc[indices]
            environments = selected["environment_id"].drop_duplicates()
            observed = environments.map(status["weather_observed"]).fillna(False)
            climatology = environments.map(status["weather_climatology"]).fillna(False)
            any_weather = environments.map(status["weather_any_available"]).fillna(False)
            coverage_rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "observation_rows": len(selected),
                    "environment_count": len(environments),
                    "observed_environment_count": int(observed.sum()),
                    "missing_environment_count": int((~observed).sum()),
                    "observed_fraction": float(observed.mean()) if len(observed) else np.nan,
                    "climatology_environment_count": int(climatology.sum()),
                    "any_weather_environment_count": int(any_weather.sum()),
                    "still_missing_environment_count": int((~any_weather).sum()),
                    "any_weather_fraction": float(any_weather.mean()) if len(any_weather) else np.nan,
                }
            )
    leakage_frame = pd.DataFrame(leakage_rows)
    if not leakage_frame["leakage_status"].eq("pass").all():
        raise ValueError("Weather coverage audit detected split leakage")
    return pd.DataFrame(coverage_rows), leakage_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit weather recovery causes and model overlap.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--environment-dir", type=Path, default=Path("environment"))
    parser.add_argument(
        "--weather-dir",
        type=Path,
        default=None,
        help="Optional isolated directory containing the recovery manifest and weather tables.",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("model_kernels/stage1_pedigree_env")
    )
    parser.add_argument("--model-prefix", default="stage1_pedigree_env")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(
            "model_kernels/multitrait_pedigree_env_uniform_tgw_certified/"
            "multitrait_pedigree_uniform_tgw_certified_observations.parquet"
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("model_kernels/weather_recovery_audit")
    )
    parser.add_argument("--seeds", default="2026,2027,2028,2029")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--nasa-start", default="1981-01-01")
    parser.add_argument("--coverage-file", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    environment_dir = resolve(root, args.environment_dir)
    weather_dir = environment_dir if args.weather_dir is None else resolve(root, args.weather_dir)
    model_dir = resolve(root, args.model_dir)
    ledger_path = resolve(root, args.ledger)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    order = pd.read_csv(environment_dir / "env_kernel_sample_order.tsv", sep="\t", dtype=str)
    manifest = pd.read_csv(
        weather_dir / "trial_weather_fetch_manifest.tsv", sep="\t", dtype=str, low_memory=False
    )
    model_order = pd.read_csv(
        model_dir / f"{args.model_prefix}_K_E_unique_order.tsv", sep="\t", dtype=str
    )
    audit = classify_environment_coverage(
        order=order,
        manifest=manifest,
        nasa=feature_status(
            weather_dir / "trial_weather_features_nasa_power.tsv", "nasa"
        ),
        openmeteo=feature_status(
            weather_dir / "trial_weather_features_openmeteo.tsv", "openmeteo"
        ),
        failures=failure_request_ids(weather_dir),
        model_env_ids=set(model_order["env_id"].fillna("").astype(str)),
        nasa_start=pd.Timestamp(args.nasa_start),
    )
    coverage_path = (
        None if args.coverage_file is None else resolve(root, args.coverage_file)
    )
    audit = apply_climatology_coverage(audit, coverage_path)
    ledger = prepare_ledger(read_table(ledger_path))
    split_frame, leakage_frame = split_coverage(
        ledger,
        audit,
        seeds=[int(value) for value in args.seeds.split(",") if value.strip()],
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
    )

    audit.to_csv(out_dir / "weather_recovery_environment_audit.tsv", sep="\t", index=False)
    scope_summary(audit).to_csv(
        out_dir / "weather_recovery_cause_summary.tsv", sep="\t", index=False
    )
    availability_summary(audit).to_csv(
        out_dir / "weather_recovery_availability_summary.tsv", sep="\t", index=False
    )
    trait_coverage_summary(ledger, audit).to_csv(
        out_dir / "weather_recovery_by_trait.tsv", sep="\t", index=False
    )
    dimension_summary(audit, "Country").to_csv(
        out_dir / "weather_recovery_by_country.tsv", sep="\t", index=False
    )
    dimension_summary(audit, "Cycle").to_csv(
        out_dir / "weather_recovery_by_cycle.tsv", sep="\t", index=False
    )
    split_frame.to_csv(out_dir / "weather_recovery_by_split.tsv", sep="\t", index=False)
    leakage_frame.to_csv(out_dir / "weather_recovery_split_leakage.tsv", sep="\t", index=False)
    audit.loc[~audit["weather_observed"]].to_csv(
        out_dir / "weather_recovery_targets_all.tsv", sep="\t", index=False
    )
    audit.loc[audit["used_by_pedigree_model"] & ~audit["weather_observed"]].to_csv(
        out_dir / "weather_recovery_targets_model.tsv", sep="\t", index=False
    )

    print(scope_summary(audit).to_string(index=False))
    print("\n", availability_summary(audit).to_string(index=False))
    print("\nModel environment count:", int(audit["used_by_pedigree_model"].sum()))
    print("Split leakage: pass")


if __name__ == "__main__":
    main()
