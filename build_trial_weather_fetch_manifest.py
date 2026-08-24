from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd


ID_COLS = ["Trial_name", "Occ", "Loc_no", "Country", "Loc_desc", "Cycle"]
MAX_REASONABLE_DATE = pd.Timestamp.today().normalize()


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")


def clean_key(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace(".0", "")


def normalized_trial_dir(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\\", "/").rstrip("/").lower()


def env_id(df: pd.DataFrame) -> pd.Series:
    return df[ID_COLS].apply(lambda row: "|".join(row.map(lambda x: "" if pd.isna(x) else str(x))), axis=1)


def decimal_degrees(deg: pd.Series, minutes: pd.Series, hemi: pd.Series) -> pd.Series:
    d = pd.to_numeric(deg, errors="coerce")
    m = pd.to_numeric(minutes, errors="coerce").fillna(0)
    val = d + m / 60.0
    h = hemi.fillna("").astype(str).str.upper().str.strip()
    sign = np.where(h.isin(["S", "W"]), -1.0, 1.0)
    return val * sign


def parse_date(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if not text or text.upper() in {"UNKNOWN", "NA", "N/A", "NONE", "-", "?"}:
        return pd.NaT
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=False)
    if pd.isna(parsed) or parsed > MAX_REASONABLE_DATE:
        return pd.NaT
    return parsed


def build_location_table(loc: pd.DataFrame) -> pd.DataFrame:
    loc = loc.copy()
    loc["Loc_no_key"] = loc["Loc_no"].map(clean_key)
    loc["Country_key"] = loc.get("Country", pd.Series("", index=loc.index)).map(clean_key).str.upper()
    loc["trial_dir_key"] = loc.get("trial_dir", pd.Series("", index=loc.index)).map(
        normalized_trial_dir
    )
    loc["latitude"] = decimal_degrees(loc["Lat_degress"], loc["Lat_minutes"], loc["Latitud"])
    loc["longitude"] = decimal_degrees(loc["Long_degress"], loc["Long_minutes"], loc["Longitude"])
    loc["altitude_m"] = pd.to_numeric(loc["Altitude"], errors="coerce")
    loc["has_decimal_coordinates"] = loc["latitude"].notna() & loc["longitude"].notna()
    loc["coordinate_source"] = "Loc_data"
    return loc


def apply_date_supplement(manifest: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    if path is None:
        return manifest
    supplement = pd.read_csv(path, sep=None, engine="python", dtype=str)
    if "env_id" not in supplement.columns:
        raise ValueError(f"{path} must contain env_id")
    date_columns = [
        "sowing_date",
        "emergence_date",
        "harvest_start_date",
        "harvest_finish_date",
    ]
    available = [column for column in date_columns if column in supplement.columns]
    if not available:
        raise ValueError(f"{path} must contain at least one of {date_columns}")
    if supplement["env_id"].fillna("").duplicated().any():
        raise ValueError(f"{path} contains duplicate env_id values")
    source = (
        supplement["provenance"].fillna("").astype(str)
        if "provenance" in supplement.columns
        else pd.Series(f"curated_date_supplement:{path.name}", index=supplement.index)
    )
    work = supplement[["env_id", *available]].copy()
    for column in available:
        work[column] = work[column].map(parse_date)
        work[f"{column}_supplement_source"] = source
    manifest = manifest.merge(work, on="env_id", how="left", validate="one_to_one")
    for column in date_columns:
        source_column = f"{column}_source"
        if source_column not in manifest.columns:
            manifest[source_column] = np.where(
                manifest[column].notna(), "trial_envdata_fieldbook", "missing"
            )
        supplement_column = f"{column}_y"
        if supplement_column not in manifest.columns:
            continue
        base_column = f"{column}_x"
        use = manifest[base_column].isna() & manifest[supplement_column].notna()
        manifest[column] = manifest[base_column].fillna(manifest[supplement_column])
        manifest.loc[use, source_column] = manifest.loc[
            use, f"{column}_supplement_source"
        ].replace("", f"curated_date_supplement:{path.name}")
        manifest = manifest.drop(
            columns=[base_column, supplement_column, f"{column}_supplement_source"]
        )
    return manifest


def apply_curated_locations(manifest: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    if path is None:
        return manifest
    registry = pd.read_csv(path, sep=None, engine="python", dtype=str)
    required = {"latitude", "longitude", "review_status"}
    missing = sorted(required.difference(registry.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    registry = registry[
        registry["review_status"].fillna("").str.strip().str.lower().isin(
            {"approved", "reviewed"}
        )
    ].copy()
    registry["latitude"] = pd.to_numeric(registry["latitude"], errors="coerce")
    registry["longitude"] = pd.to_numeric(registry["longitude"], errors="coerce")
    registry = registry[registry["latitude"].notna() & registry["longitude"].notna()]
    manifest = manifest.copy()
    if "env_id" in registry.columns:
        by_env = registry[["env_id", "latitude", "longitude"]].drop_duplicates("env_id")
        by_env = by_env.rename(
            columns={"latitude": "curated_latitude", "longitude": "curated_longitude"}
        )
        manifest = manifest.merge(by_env, on="env_id", how="left", validate="one_to_one")
        use = (
            (manifest["latitude"].isna() | manifest["longitude"].isna())
            & manifest["curated_latitude"].notna()
            & manifest["curated_longitude"].notna()
        )
        manifest.loc[use, "latitude"] = manifest.loc[use, "curated_latitude"]
        manifest.loc[use, "longitude"] = manifest.loc[use, "curated_longitude"]
        manifest.loc[use, "coordinate_source"] = "curated_location_registry_env_id"
        manifest = manifest.drop(columns=["curated_latitude", "curated_longitude"])
    if {"Country", "Loc_no"}.issubset(registry.columns):
        registry["Country_key"] = registry["Country"].map(clean_key).str.upper()
        registry["Loc_no_key"] = registry["Loc_no"].map(clean_key)
        by_location = registry[
            ["Country_key", "Loc_no_key", "latitude", "longitude"]
        ].drop_duplicates(["Country_key", "Loc_no_key"])
        by_location = by_location.rename(
            columns={"latitude": "curated_latitude", "longitude": "curated_longitude"}
        )
        manifest = manifest.merge(
            by_location,
            on=["Country_key", "Loc_no_key"],
            how="left",
            validate="many_to_one",
        )
        use = (
            (manifest["latitude"].isna() | manifest["longitude"].isna())
            & manifest["curated_latitude"].notna()
            & manifest["curated_longitude"].notna()
        )
        manifest.loc[use, "latitude"] = manifest.loc[use, "curated_latitude"]
        manifest.loc[use, "longitude"] = manifest.loc[use, "curated_longitude"]
        manifest.loc[use, "coordinate_source"] = (
            "curated_location_registry_country_locno"
        )
        manifest = manifest.drop(columns=["curated_latitude", "curated_longitude"])
    return manifest


def extract_env_dates(env: pd.DataFrame) -> pd.DataFrame:
    date_traits = {
        "SOWING_DATE": "sowing_date",
        "EMERGENCE_DATE": "emergence_date",
        "HARVEST_STARTING_DATE": "harvest_start_date",
        "HARVEST_FINISHING_DATE": "harvest_finish_date",
    }
    tmp = env[["trial_dir", *ID_COLS, "Trait_name", "Value"]].copy()
    tmp = tmp[tmp["Trait_name"].isin(date_traits)].copy()
    tmp["date_field"] = tmp["Trait_name"].map(date_traits)
    tmp["parsed_date"] = tmp["Value"].map(parse_date)
    tmp["env_id"] = env_id(tmp)
    wide = (
        tmp.pivot_table(
            index=["trial_dir", *ID_COLS, "env_id"],
            columns="date_field",
            values="parsed_date",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    for col in date_traits.values():
        if col not in wide.columns:
            wide[col] = pd.NaT
    return wide


def open_meteo_url(row: pd.Series) -> str:
    hourly = ",".join(
        [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "shortwave_radiation",
            "et0_fao_evapotranspiration",
            "vapour_pressure_deficit",
            "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm",
        ]
    )
    params = {
        "latitude": f"{row['latitude']:.5f}",
        "longitude": f"{row['longitude']:.5f}",
        "start_date": row["weather_start_date"],
        "end_date": row["weather_end_date"],
        "hourly": hourly,
        "timezone": "auto",
        "models": "era5",
    }
    return "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)


def nasa_power_url(row: pd.Series) -> str:
    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN,WS2M",
        "community": "AG",
        "longitude": f"{row['longitude']:.5f}",
        "latitude": f"{row['latitude']:.5f}",
        "start": row["weather_start_date"].replace("-", ""),
        "end": row["weather_end_date"].replace("-", ""),
        "format": "JSON",
    }
    return "https://power.larc.nasa.gov/api/temporal/daily/point?" + urlencode(params)


def main(
    environment_dir: Path | str = Path("environment"),
    output_dir: Path | str | None = None,
    date_supplement: Path | str | None = None,
    location_registry: Path | str | None = None,
) -> None:
    environment_dir = Path(environment_dir)
    output_dir = Path(output_dir) if output_dir is not None else environment_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    env = pd.read_csv(environment_dir / "envdata.tsv", sep="\t", dtype=str, low_memory=False)
    loc = pd.read_csv(environment_dir / "locdata.tsv", sep="\t", dtype=str, low_memory=False)

    loc_work = build_location_table(loc)
    loc_by_trial = (
        loc_work.sort_values("has_decimal_coordinates", ascending=False)
        .drop_duplicates(["trial_dir_key", "Loc_no_key"])
        [["trial_dir_key", "Loc_no_key", "latitude", "longitude", "altitude_m", "coordinate_source"]]
    )
    loc_by_id = (
        loc_work[loc_work["has_decimal_coordinates"]]
        .groupby(["Country_key", "Loc_no_key"], as_index=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            altitude_m=("altitude_m", "mean"),
            latitude_span=("latitude", lambda values: float(values.max() - values.min())),
            longitude_span=("longitude", lambda values: float(values.max() - values.min())),
        )
    )
    loc_by_id = loc_by_id[
        loc_by_id["latitude_span"].le(0.1) & loc_by_id["longitude_span"].le(0.1)
    ].drop(columns=["latitude_span", "longitude_span"])
    loc_by_id["coordinate_source"] = "Loc_data_Country_Loc_no_stable_mean"

    dates = extract_env_dates(env)
    env_base = env[["trial_dir", *ID_COLS]].drop_duplicates().copy()
    env_base["env_id"] = env_id(env_base)
    env_base = env_base.sort_values(["env_id", "trial_dir"], kind="stable").drop_duplicates(
        "env_id", keep="first"
    )
    date_columns = [
        "env_id",
        "sowing_date",
        "emergence_date",
        "harvest_start_date",
        "harvest_finish_date",
    ]
    date_summary = dates[date_columns].groupby("env_id", as_index=False).first()
    manifest = env_base.merge(date_summary, on="env_id", how="left", validate="one_to_one")
    manifest["Loc_no_key"] = manifest["Loc_no"].map(clean_key)
    manifest["Country_key"] = manifest["Country"].map(clean_key).str.upper()
    manifest["trial_dir_key"] = manifest["trial_dir"].map(normalized_trial_dir)
    manifest = manifest.merge(loc_by_trial, on=["trial_dir_key", "Loc_no_key"], how="left")
    missing_coord = manifest["latitude"].isna() | manifest["longitude"].isna()
    fallback = manifest.loc[missing_coord, ["Country_key", "Loc_no_key"]].merge(
        loc_by_id, on=["Country_key", "Loc_no_key"], how="left"
    )
    for col in ["latitude", "longitude", "altitude_m", "coordinate_source"]:
        manifest.loc[missing_coord, col] = fallback[col].to_numpy()

    manifest = apply_curated_locations(
        manifest, None if location_registry is None else Path(location_registry)
    )
    for column in [
        "sowing_date",
        "emergence_date",
        "harvest_start_date",
        "harvest_finish_date",
    ]:
        manifest[f"{column}_source"] = np.where(
            manifest[column].notna(), "trial_envdata_fieldbook", "missing"
        )
    manifest = apply_date_supplement(
        manifest, None if date_supplement is None else Path(date_supplement)
    )

    manifest["weather_start"] = manifest["sowing_date"].fillna(manifest["emergence_date"])
    manifest["weather_start_source"] = np.where(
        manifest["sowing_date"].notna(),
        "sowing_date:" + manifest["sowing_date_source"].astype(str),
        np.where(
            manifest["emergence_date"].notna(),
            "emergence_date:" + manifest["emergence_date_source"].astype(str),
            "missing",
        ),
    )
    manifest["weather_end"] = manifest["harvest_finish_date"].fillna(manifest["harvest_start_date"])
    manifest["weather_end_source"] = np.where(
        manifest["harvest_finish_date"].notna(),
        "harvest_finish_date:" + manifest["harvest_finish_date_source"].astype(str),
        np.where(
            manifest["harvest_start_date"].notna(),
            "harvest_start_date:" + manifest["harvest_start_date_source"].astype(str),
            "missing",
        ),
    )
    inferred_start = manifest["weather_start"].isna() & manifest["weather_end"].notna()
    manifest.loc[inferred_start, "weather_start"] = (
        manifest.loc[inferred_start, "weather_end"] - pd.Timedelta(days=180)
    )
    manifest.loc[inferred_start, "weather_start_source"] = "end_minus_180_days"
    inferred_end = manifest["weather_end"].isna() & manifest["weather_start"].notna()
    manifest.loc[inferred_end, "weather_end"] = (
        manifest.loc[inferred_end, "weather_start"] + pd.Timedelta(days=180)
    )
    manifest.loc[inferred_end, "weather_end_source"] = "start_plus_180_days"
    bad_window = manifest["weather_start"].notna() & manifest["weather_end"].notna() & (
        manifest["weather_end"] < manifest["weather_start"]
    )
    manifest.loc[bad_window, "weather_end"] = manifest.loc[bad_window, "weather_start"] + pd.Timedelta(days=180)
    manifest.loc[bad_window, "weather_end_source"] = "repaired_end_before_start_plus_180_days"
    long_window = manifest["weather_start"].notna() & manifest["weather_end"].notna() & (
        (manifest["weather_end"] - manifest["weather_start"]).dt.days > 365
    )
    manifest.loc[long_window, "weather_end"] = manifest.loc[long_window, "weather_start"] + pd.Timedelta(days=180)
    manifest.loc[long_window, "weather_end_source"] = "repaired_window_over_365_days_plus_180_days"
    manifest["weather_start_date"] = manifest["weather_start"].dt.strftime("%Y-%m-%d")
    manifest["weather_end_date"] = manifest["weather_end"].dt.strftime("%Y-%m-%d")
    manifest["has_fetch_window"] = manifest["weather_start_date"].notna() & manifest["weather_end_date"].notna()
    manifest["has_fetch_coordinates"] = manifest["latitude"].notna() & manifest["longitude"].notna()
    manifest["ready_to_fetch"] = manifest["has_fetch_window"] & manifest["has_fetch_coordinates"]
    manifest["window_inferred"] = ~(
        manifest["weather_start_source"].str.startswith("sowing_date:trial_envdata_fieldbook")
        & manifest["weather_end_source"].str.startswith(
            ("harvest_finish_date:trial_envdata_fieldbook", "harvest_start_date:trial_envdata_fieldbook")
        )
    )
    manifest["coordinates_inferred"] = ~manifest["coordinate_source"].eq("Loc_data")

    ready = manifest["ready_to_fetch"]
    manifest["open_meteo_era5_url"] = ""
    manifest["nasa_power_daily_url"] = ""
    manifest.loc[ready, "open_meteo_era5_url"] = manifest.loc[ready].apply(open_meteo_url, axis=1)
    manifest.loc[ready, "nasa_power_daily_url"] = manifest.loc[ready].apply(nasa_power_url, axis=1)

    out_cols = [
        "env_id",
        "trial_dir",
        "Trial_name",
        "Cycle",
        "Occ",
        "Loc_no",
        "Country",
        "Loc_desc",
        "latitude",
        "longitude",
        "altitude_m",
        "coordinate_source",
        "sowing_date",
        "emergence_date",
        "harvest_start_date",
        "harvest_finish_date",
        "sowing_date_source",
        "emergence_date_source",
        "harvest_start_date_source",
        "harvest_finish_date_source",
        "weather_start_date",
        "weather_end_date",
        "weather_start_source",
        "weather_end_source",
        "window_inferred",
        "coordinates_inferred",
        "has_fetch_window",
        "has_fetch_coordinates",
        "ready_to_fetch",
        "open_meteo_era5_url",
        "nasa_power_daily_url",
    ]
    write_tsv(manifest[out_cols], output_dir / "trial_weather_fetch_manifest.tsv")

    qc = pd.DataFrame(
        [
            {"metric": "locdata_rows", "value": len(loc_work)},
            {"metric": "locdata_rows_with_coordinates", "value": int(loc_work["has_decimal_coordinates"].sum())},
            {"metric": "env_records_with_date_traits", "value": len(dates)},
            {"metric": "environment_ids_total", "value": manifest["env_id"].nunique()},
            {"metric": "env_records_with_coordinates", "value": int(manifest["has_fetch_coordinates"].sum())},
            {"metric": "env_records_with_fetch_window", "value": int(manifest["has_fetch_window"].sum())},
            {"metric": "env_records_ready_to_fetch", "value": int(manifest["ready_to_fetch"].sum())},
            {"metric": "unique_env_id_ready_to_fetch", "value": manifest.loc[manifest["ready_to_fetch"], "env_id"].nunique()},
        ]
    )
    write_tsv(qc, output_dir / "trial_weather_fetch_manifest_qc.tsv")

    coord_qc = (
        loc_work.groupby(["Country"], dropna=False)
        .agg(
            locdata_rows=("Loc_no", "size"),
            rows_with_coordinates=("has_decimal_coordinates", "sum"),
            unique_loc_no=("Loc_no_key", "nunique"),
        )
        .reset_index()
        .sort_values(["locdata_rows"], ascending=False)
    )
    write_tsv(coord_qc, output_dir / "locdata_coordinate_qc.tsv")

    print(qc.to_string(index=False))


def cli() -> None:
    parser = argparse.ArgumentParser(description="Build a provenance-aware trial weather fetch manifest.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--environment-dir", type=Path, default=Path("environment"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--date-supplement", type=Path, default=None)
    parser.add_argument("--location-registry", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    environment_dir = (
        args.environment_dir.resolve()
        if args.environment_dir.is_absolute()
        else (root / args.environment_dir).resolve()
    )
    output_dir = (
        environment_dir
        if args.out_dir is None
        else (
            args.out_dir.resolve()
            if args.out_dir.is_absolute()
            else (root / args.out_dir).resolve()
        )
    )
    date_supplement = (
        None
        if args.date_supplement is None
        else (
            args.date_supplement.resolve()
            if args.date_supplement.is_absolute()
            else (root / args.date_supplement).resolve()
        )
    )
    location_registry = (
        None
        if args.location_registry is None
        else (
            args.location_registry.resolve()
            if args.location_registry.is_absolute()
            else (root / args.location_registry).resolve()
        )
    )
    main(environment_dir, output_dir, date_supplement, location_registry)


if __name__ == "__main__":
    cli()
