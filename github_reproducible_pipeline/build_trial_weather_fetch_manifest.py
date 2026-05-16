from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
OUT = BASE / "environment"
ID_COLS = ["Trial_name", "Occ", "Loc_no", "Country", "Loc_desc", "Cycle"]
MAX_REASONABLE_DATE = pd.Timestamp("2026-05-12")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")


def clean_key(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace(".0", "")


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
    loc["latitude"] = decimal_degrees(loc["Lat_degress"], loc["Lat_minutes"], loc["Latitud"])
    loc["longitude"] = decimal_degrees(loc["Long_degress"], loc["Long_minutes"], loc["Longitude"])
    loc["altitude_m"] = pd.to_numeric(loc["Altitude"], errors="coerce")
    loc["has_decimal_coordinates"] = loc["latitude"].notna() & loc["longitude"].notna()
    loc["coordinate_source"] = "Loc_data"
    return loc


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


def main() -> None:
    env = pd.read_csv(OUT / "envdata.tsv", sep="\t", dtype=str, low_memory=False)
    loc = pd.read_csv(OUT / "locdata.tsv", sep="\t", dtype=str, low_memory=False)

    loc_work = build_location_table(loc)
    loc_by_trial = (
        loc_work.sort_values("has_decimal_coordinates", ascending=False)
        .drop_duplicates(["trial_dir", "Loc_no_key"])
        [["trial_dir", "Loc_no_key", "latitude", "longitude", "altitude_m", "coordinate_source"]]
    )
    loc_by_id = (
        loc_work[loc_work["has_decimal_coordinates"]]
        .groupby("Loc_no_key", as_index=False)
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"), altitude_m=("altitude_m", "mean"))
    )
    loc_by_id["coordinate_source"] = "Loc_data_Loc_no_mean"

    dates = extract_env_dates(env)
    dates["Loc_no_key"] = dates["Loc_no"].map(clean_key)
    manifest = dates.merge(loc_by_trial, on=["trial_dir", "Loc_no_key"], how="left")
    missing_coord = manifest["latitude"].isna() | manifest["longitude"].isna()
    fallback = manifest.loc[missing_coord, ["Loc_no_key"]].merge(loc_by_id, on="Loc_no_key", how="left")
    for col in ["latitude", "longitude", "altitude_m", "coordinate_source"]:
        manifest.loc[missing_coord, col] = fallback[col].to_numpy()

    manifest["weather_start"] = manifest["sowing_date"].fillna(manifest["emergence_date"])
    manifest["weather_end"] = manifest["harvest_finish_date"].fillna(manifest["harvest_start_date"])
    manifest["weather_end"] = manifest["weather_end"].fillna(manifest["weather_start"] + pd.Timedelta(days=180))
    bad_window = manifest["weather_start"].notna() & manifest["weather_end"].notna() & (
        manifest["weather_end"] < manifest["weather_start"]
    )
    manifest.loc[bad_window, "weather_end"] = manifest.loc[bad_window, "weather_start"] + pd.Timedelta(days=180)
    long_window = manifest["weather_start"].notna() & manifest["weather_end"].notna() & (
        (manifest["weather_end"] - manifest["weather_start"]).dt.days > 365
    )
    manifest.loc[long_window, "weather_end"] = manifest.loc[long_window, "weather_start"] + pd.Timedelta(days=180)
    manifest["weather_start_date"] = manifest["weather_start"].dt.strftime("%Y-%m-%d")
    manifest["weather_end_date"] = manifest["weather_end"].dt.strftime("%Y-%m-%d")
    manifest["has_fetch_window"] = manifest["weather_start_date"].notna() & manifest["weather_end_date"].notna()
    manifest["has_fetch_coordinates"] = manifest["latitude"].notna() & manifest["longitude"].notna()
    manifest["ready_to_fetch"] = manifest["has_fetch_window"] & manifest["has_fetch_coordinates"]

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
        "weather_start_date",
        "weather_end_date",
        "ready_to_fetch",
        "open_meteo_era5_url",
        "nasa_power_daily_url",
    ]
    write_tsv(manifest[out_cols], OUT / "trial_weather_fetch_manifest.tsv")

    qc = pd.DataFrame(
        [
            {"metric": "locdata_rows", "value": len(loc_work)},
            {"metric": "locdata_rows_with_coordinates", "value": int(loc_work["has_decimal_coordinates"].sum())},
            {"metric": "env_records_with_date_traits", "value": len(dates)},
            {"metric": "env_records_with_coordinates", "value": int(manifest["has_fetch_coordinates"].sum())},
            {"metric": "env_records_with_fetch_window", "value": int(manifest["has_fetch_window"].sum())},
            {"metric": "env_records_ready_to_fetch", "value": int(manifest["ready_to_fetch"].sum())},
            {"metric": "unique_env_id_ready_to_fetch", "value": manifest.loc[manifest["ready_to_fetch"], "env_id"].nunique()},
        ]
    )
    write_tsv(qc, OUT / "trial_weather_fetch_manifest_qc.tsv")

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
    write_tsv(coord_qc, OUT / "locdata_coordinate_qc.tsv")

    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
