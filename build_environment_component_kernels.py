from __future__ import annotations

import argparse
import os
import re
import hashlib
import json
import platform
import sys
from datetime import date
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOCAL_DEPS = BASE / "local_python_deps"
if platform.system() == "Windows" and LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd

OUT = BASE / "environment"
ID_COLS = ["Trial_name", "Occ", "Loc_no", "Country", "Loc_desc", "Cycle"]
UNRESOLVED_LOCATION_PAYLOAD_FIELDS = [
    "Trial_name_key",
    "Country_key",
    "Loc_desc_key",
    "Loc_no_key",
    "Cycle",
    "Occ",
    "source_file",
]


def env_id_from_frame(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ID_COLS if c in df.columns]
    return df[cols].apply(lambda row: "|".join(row.map(lambda x: "" if pd.isna(x) else str(x))), axis=1)


def parse_value(value: object, trait: str) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    trait_upper = trait.upper()
    categorical_tokens = ("PRODUCT", "SPECIFY", "_TEXT", "FERTILIZER_1", "FERTILIZER_2", "FERTILIZER_3")
    if any(token in trait_upper for token in categorical_tokens) and "DATE" not in trait_upper:
        return np.nan

    if "DATE" in trait_upper or trait_upper in {"SOWING_OLD"}:
        parsed_date: date | None = None
        month_lookup = {
            "JAN": 1,
            "FEB": 2,
            "MAR": 3,
            "APR": 4,
            "MAY": 5,
            "JUN": 6,
            "JUL": 7,
            "AUG": 8,
            "SEP": 9,
            "OCT": 10,
            "NOV": 11,
            "DEC": 12,
        }
        month_first = re.fullmatch(r"([A-Za-z]{3,9})\s+(\d{1,2})\s+(\d{4})", text)
        day_first = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})", text)
        numeric_day_first = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", text)
        numeric_year_first = re.fullmatch(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", text)
        try:
            if month_first and month_first.group(1)[:3].upper() in month_lookup:
                parsed_date = date(
                    int(month_first.group(3)),
                    month_lookup[month_first.group(1)[:3].upper()],
                    int(month_first.group(2)),
                )
            elif day_first and day_first.group(2)[:3].upper() in month_lookup:
                parsed_date = date(
                    int(day_first.group(3)),
                    month_lookup[day_first.group(2)[:3].upper()],
                    int(day_first.group(1)),
                )
            elif numeric_day_first:
                year_text = numeric_day_first.group(3)
                year = int(year_text)
                if len(year_text) == 2:
                    year += 2000 if year < 69 else 1900
                parsed_date = date(year, int(numeric_day_first.group(2)), int(numeric_day_first.group(1)))
            elif numeric_year_first:
                parsed_date = date(
                    int(numeric_year_first.group(1)),
                    int(numeric_year_first.group(2)),
                    int(numeric_year_first.group(3)),
                )
        except ValueError:
            parsed_date = None
        return float(parsed_date.timetuple().tm_yday) if parsed_date is not None else np.nan

    upper = text.upper()
    if upper in {"YES", "Y", "TRUE", "T", "IRRIGATED", "APPLIED"}:
        return 1.0
    if upper in {"NO", "N", "FALSE", "NONE", "NIL", "NOT APPLIED"}:
        return 0.0

    normalized = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
    match = re.fullmatch(rf"\s*({number})\s*([%A-Za-z°/_().^\-\s]*)", normalized)
    if not match:
        return np.nan
    parsed = float(match.group(1))
    return parsed if np.isfinite(parsed) else np.nan


def normalize_loc_no(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.upper()


def normalize_country(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"[^0-9A-Z]+", "_", regex=True)
        .str.strip("_")
    )

def normalize_location_text(s: pd.Series) -> pd.Series:
    return normalize_country(s)


def stable_unresolved_location_key(row: pd.Series) -> str:
    payload = "|".join(str(row.get(field, "")).strip() for field in UNRESOLVED_LOCATION_PAYLOAD_FIELDS)
    return "UNRESOLVED_LOCATION|" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def add_location_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Loc_no_key"] = normalize_loc_no(out.get("Loc_no", pd.Series("", index=out.index)))
    out["Country_key"] = normalize_country(out.get("Country", pd.Series("", index=out.index)))
    out["Loc_desc_key"] = normalize_location_text(out.get("Loc_desc", pd.Series("", index=out.index)))
    out["Trial_name_key"] = normalize_location_text(out.get("Trial_name", pd.Series("", index=out.index)))
    out["loc_no_missing"] = out["Loc_no_key"].eq("")
    keys, methods = [], []
    for _, row in out.iterrows():
        country, loc_no, desc, trial = row["Country_key"], row["Loc_no_key"], row["Loc_desc_key"], row["Trial_name_key"]
        if country and loc_no:
            key, method = f"{country}|{loc_no}", "country_loc_no"
        elif country and desc:
            key, method = f"{country}|{desc}", "country_loc_desc_fallback"
        elif trial and desc:
            key, method = f"{trial}|{country}|{desc}", "trial_country_loc_desc_fallback"
        elif loc_no:
            key, method = loc_no, "country_missing_loc_no_fallback"
        else:
            key = stable_unresolved_location_key(row)
            method = "unresolved_location_hash_fallback"
        keys.append(key)
        methods.append(method)
    out["location_key"] = keys
    out["location_key_method"] = methods
    out["location_key_fallback"] = out["location_key_method"].ne("country_loc_no")
    return out


def decimal_degrees(deg: pd.Series, minutes: pd.Series, hemi: pd.Series) -> pd.Series:
    d = pd.to_numeric(deg, errors="coerce")
    m = pd.to_numeric(minutes, errors="coerce").fillna(0)
    val = d + m / 60.0
    h = hemi.astype(str).str.upper().str.strip()
    sign = np.where(h.isin(["S", "W"]), -1.0, 1.0)
    return val * sign


def location_collision_audit(
    loc: pd.DataFrame,
    degree_tolerance: float = 0.05,
    altitude_tolerance: float = 50.0,
) -> pd.DataFrame:
    work = add_location_keys(loc)
    work["latitude"] = decimal_degrees(work["Lat_degress"], work["Lat_minutes"], work["Latitud"])
    work["longitude"] = decimal_degrees(work["Long_degress"], work["Long_minutes"], work["Longitude"])
    work["altitude"] = pd.to_numeric(work["Altitude"], errors="coerce")
    country_counts = work[work["Country_key"].ne("")].groupby("Loc_no_key")["Country_key"].nunique()
    rows = []
    for location_key, group in work.groupby("location_key", dropna=False, sort=True):
        lat_dispersion = float(group["latitude"].max() - group["latitude"].min()) if group["latitude"].notna().any() else 0.0
        lon_dispersion = float(group["longitude"].max() - group["longitude"].min()) if group["longitude"].notna().any() else 0.0
        alt_dispersion = float(group["altitude"].max() - group["altitude"].min()) if group["altitude"].notna().any() else 0.0
        loc_numbers = sorted({value for value in group["Loc_no_key"] if value})
        countries = sorted({value for value in group["Country_key"] if value})
        descriptions = sorted({value for value in group["Loc_desc_key"] if value})
        trials = sorted({value for value in group["Trial_name_key"] if value})
        reasons = []
        if group["Country_key"].eq("").any():
            reasons.append("country_missing_fallback")
        if group["loc_no_missing"].any():
            reasons.append("loc_no_missing_fallback")
        if group["location_key_method"].eq("unresolved_location_hash_fallback").any():
            reasons.append("unresolved_location_hash_fallback")
        if any(country_counts.get(loc_no, 0) > 1 for loc_no in loc_numbers):
            reasons.append("loc_no_multiple_countries")
        if lat_dispersion > degree_tolerance or lon_dispersion > degree_tolerance or alt_dispersion > altitude_tolerance:
            reasons.append("coordinate_dispersion_collision")
        rows.append(
            {
                "location_key": location_key,
                "location_key_method": ";".join(sorted(set(group["location_key_method"]))),
                "n_rows": len(group),
                "n_unique_latitude": group["latitude"].nunique(dropna=True),
                "n_unique_longitude": group["longitude"].nunique(dropna=True),
                "n_unique_altitude": group["altitude"].nunique(dropna=True),
                "countries": ";".join(countries),
                "loc_numbers": ";".join(loc_numbers),
                "loc_descriptions": ";".join(descriptions),
                "trials": ";".join(trials),
                "loc_no_missing": bool(group["loc_no_missing"].any()),
                "unresolved_hash_payload_fields": (
                    ";".join(UNRESOLVED_LOCATION_PAYLOAD_FIELDS)
                    if group["location_key_method"].eq("unresolved_location_hash_fallback").any()
                    else ""
                ),
                "unresolved_duplicate_count": (
                    int(len(group) - 1)
                    if group["location_key_method"].eq("unresolved_location_hash_fallback").all()
                    else 0
                ),
                "collision_status": ";".join(reasons) if reasons else "ok",
            }
        )
    return pd.DataFrame(rows)


def build_env_trait_matrix(env: pd.DataFrame, parsing_qc_path: Path | None = None) -> pd.DataFrame:
    tmp = env[[*ID_COLS, "Trait_name", "Value"]].copy()
    tmp["env_id"] = env_id_from_frame(tmp)
    tmp["feature_value"] = [
        parse_value(value, trait) for value, trait in zip(tmp["Value"].to_numpy(), tmp["Trait_name"].to_numpy())
    ]
    if parsing_qc_path is not None:
        raw_present = tmp["Value"].fillna("").astype(str).str.strip().ne("")
        tmp["raw_value_present"] = raw_present
        tmp["parsed_finite"] = np.isfinite(tmp["feature_value"])
        qc = (
            tmp.groupby("Trait_name", dropna=False)
            .agg(
                rows=("Value", "size"),
                raw_values_present=("raw_value_present", "sum"),
                finite_values_parsed=("parsed_finite", "sum"),
                unique_raw_values=("Value", "nunique"),
            )
            .reset_index()
        )
        qc["unparsed_present_values"] = qc["raw_values_present"] - qc["finite_values_parsed"]
        qc["parse_fraction_of_present"] = np.where(
            qc["raw_values_present"] > 0,
            qc["finite_values_parsed"] / qc["raw_values_present"],
            np.nan,
        )
        qc.to_csv(parsing_qc_path, sep="\t", index=False)
    return tmp.pivot_table(index="env_id", columns="Trait_name", values="feature_value", aggfunc="mean")


def build_location_fallbacks(loc_work: pd.DataFrame) -> pd.DataFrame:
    country_counts = loc_work[loc_work["Country_key"].ne("")].groupby("Loc_no_key")["Country_key"].nunique()
    all_loc_numbers = pd.Index(loc_work["Loc_no_key"].drop_duplicates())
    safe_loc_numbers = all_loc_numbers[
        all_loc_numbers.to_series().ne("").to_numpy()
        & country_counts.reindex(all_loc_numbers, fill_value=0).le(1).to_numpy()
    ]
    return (
        loc_work[
            loc_work["Loc_no_key"].ne("")
            & loc_work["Loc_no_key"].isin(safe_loc_numbers)
        ]
        .groupby("Loc_no_key")[["latitude", "longitude", "altitude"]]
        .mean()
        .add_suffix("_fallback")
    )


def build_geo_features(env: pd.DataFrame, loc: pd.DataFrame, env_ids: pd.Index) -> pd.DataFrame:
    env_base = env[[*ID_COLS]].drop_duplicates().copy()
    env_base["env_id"] = env_id_from_frame(env_base)
    env_base = add_location_keys(env_base)

    loc_work = add_location_keys(loc)
    loc_work["latitude"] = decimal_degrees(loc_work["Lat_degress"], loc_work["Lat_minutes"], loc_work["Latitud"])
    loc_work["longitude"] = decimal_degrees(loc_work["Long_degress"], loc_work["Long_minutes"], loc_work["Longitude"])
    loc_work["altitude"] = pd.to_numeric(loc_work["Altitude"], errors="coerce")
    loc_by_id = loc_work.groupby("location_key")[["latitude", "longitude", "altitude"]].mean()
    fallback_by_loc = build_location_fallbacks(loc_work)

    geo = env_base[["env_id", "location_key", "Loc_no_key"]].merge(loc_by_id, left_on="location_key", right_index=True, how="left")
    geo = geo.merge(fallback_by_loc, left_on="Loc_no_key", right_index=True, how="left")
    for coordinate in ["latitude", "longitude", "altitude"]:
        geo[coordinate] = geo[coordinate].fillna(geo[f"{coordinate}_fallback"])
    geo = geo.drop_duplicates("env_id").set_index("env_id")[["latitude", "longitude", "altitude"]].reindex(env_ids)

    gps = env[ID_COLS + ["Trait_name", "Value"]].copy()
    gps["env_id"] = env_id_from_frame(gps)
    gps = gps[gps["Trait_name"].astype(str).str.startswith("GPS ")]
    if not gps.empty:
        wide = gps.pivot_table(index="env_id", columns="Trait_name", values="Value", aggfunc="first").reindex(env_ids)
        gps_lat = decimal_degrees(
            wide.get("GPS Latitude (Degrees)", pd.Series(index=env_ids, dtype=object)),
            wide.get("GPS Latitude (Minutes)", pd.Series(index=env_ids, dtype=object)),
            wide.get("GPS Latitude (N or S)", pd.Series(index=env_ids, dtype=object)),
        )
        gps_lon = decimal_degrees(
            wide.get("GPS Longitude (Degress)", pd.Series(index=env_ids, dtype=object)),
            wide.get("GPS Longitude (Minutes)", pd.Series(index=env_ids, dtype=object)),
            wide.get("GPS Longitude ( E or W)", pd.Series(index=env_ids, dtype=object)),
        )
        gps_alt = pd.to_numeric(wide.get("GPS Altitude", pd.Series(index=env_ids, dtype=object)), errors="coerce")
        geo["latitude"] = geo["latitude"].fillna(gps_lat)
        geo["longitude"] = geo["longitude"].fillna(gps_lon)
        geo["altitude"] = geo["altitude"].fillna(gps_alt)

    return geo


def trait_group_columns(all_cols: pd.Index, group: str) -> list[str]:
    cols = []
    for col in all_cols:
        name = str(col).upper()
        if "COMMENT" in name or "NOTE" in name or "EMAIL" in name:
            continue
        if group == "weather":
            match = any(k in name for k in ["TEMP", "RAIN", "PPN", "PRECIPIT", "RADIATION", "HUMID", "WEATHER"])
        elif group == "stress":
            match = any(k in name for k in ["HEAT", "DROUGHT", "VAPOR", "VPD", "DRY", "MOISTURE", "FROST", "HAIL"])
        elif group == "mgmt":
            if any(k in name for k in ["PRECIPIT", "RAIN", "MOISTURE", "DROUGHT", "DRY", "WEATHER"]):
                match = False
            else:
                match = any(
                    k in name
                    for k in [
                        "SOWING",
                        "IRRIGATION",
                        "IRRIGATED",
                        "FERTILIZER",
                        "HERBICIDE",
                        "FUNGICIDE",
                        "PESTICIDE",
                        "WEEDING",
                        "ROWS",
                        "AREA_",
                        "CROP_STAND",
                    ]
                )
        else:
            match = False
        if match:
            cols.append(col)
    return cols


def build_fetched_weather_feature_sets(
    env_ids: pd.Index, environment_dir: Path = OUT
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weather_cols = [
        "n_days_weather",
        "temperature_mean_c",
        "temperature_max_c",
        "temperature_min_c",
        "relative_humidity_mean_pct",
        "precipitation_total_mm",
        "precipitation_mean_daily_mm",
        "solar_radiation_total_mj_m2",
        "solar_radiation_mean_daily_mj_m2",
        "wind_speed_2m_mean_m_s",
    ]
    stress_cols = [
        "vpd_mean_kpa",
        "vpd_max_kpa",
        "heat_days_tmax_ge_30",
        "heat_days_tmax_ge_35",
        "extreme_heat_days_tmax_ge_40",
        "dry_days_precip_lt_1mm",
        "heavy_rain_days_precip_ge_20mm",
        "high_vpd_days_gt_1_5",
        "high_vpd_days_gt_2_0",
        "drought_days_precip_lt_1mm_and_vpd_gt_1_5",
        "radiation_vpd_stress_index",
    ]

    nasa_weather, nasa_stress = build_nasa_power_feature_sets(
        env_ids, weather_cols, stress_cols, environment_dir=environment_dir
    )
    openmeteo_weather, openmeteo_stress = build_openmeteo_feature_sets(
        env_ids, weather_cols, stress_cols, environment_dir=environment_dir
    )
    weather = nasa_weather.combine_first(openmeteo_weather).add_prefix("weather_api_")
    stress = nasa_stress.combine_first(openmeteo_stress).add_prefix("weather_api_")
    return weather, stress


def read_weather_feature_table(path: Path, env_ids: pd.Index) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(index=env_ids)
    fetched = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    if "fetch_status" in fetched.columns:
        fetched = fetched[fetched["fetch_status"].astype(str).str.lower().eq("ok")].copy()
    if fetched.empty:
        return pd.DataFrame(index=env_ids)
    fetched = fetched.drop_duplicates("env_id", keep="first").set_index("env_id").reindex(env_ids)
    for col in fetched.columns:
        if col not in {"fetch_status", "weather_request_id", "openmeteo_timezone"}:
            fetched[col] = pd.to_numeric(fetched[col], errors="coerce")
    return fetched


def build_nasa_power_feature_sets(
    env_ids: pd.Index,
    weather_cols: list[str],
    stress_cols: list[str],
    environment_dir: Path = OUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nasa_path = environment_dir / "trial_weather_features_nasa_power.tsv"
    fetched = read_weather_feature_table(nasa_path, env_ids)
    weather = fetched[[c for c in weather_cols if c in fetched.columns]].copy()
    stress = fetched[[c for c in stress_cols if c in fetched.columns]].copy()
    return weather, stress


def build_openmeteo_feature_sets(
    env_ids: pd.Index,
    weather_cols: list[str],
    stress_cols: list[str],
    environment_dir: Path = OUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    openmeteo_path = environment_dir / "trial_weather_features_openmeteo.tsv"
    fetched = read_weather_feature_table(openmeteo_path, env_ids)
    if fetched.empty:
        return pd.DataFrame(index=env_ids), pd.DataFrame(index=env_ids)

    if "shortwave_radiation_total_wm2h" in fetched.columns:
        fetched["solar_radiation_total_mj_m2"] = fetched["shortwave_radiation_total_wm2h"] * 0.0036
    if "shortwave_radiation_mean_daily_wm2h" in fetched.columns:
        fetched["solar_radiation_mean_daily_mj_m2"] = fetched["shortwave_radiation_mean_daily_wm2h"] * 0.0036
    if "solar_radiation_mean_daily_mj_m2" in fetched.columns and "vpd_mean_kpa" in fetched.columns:
        fetched["radiation_vpd_stress_index"] = fetched["solar_radiation_mean_daily_mj_m2"] * fetched["vpd_mean_kpa"]

    weather = fetched[[c for c in weather_cols if c in fetched.columns]].copy()
    stress = fetched[[c for c in stress_cols if c in fetched.columns]].copy()
    return weather, stress


def standardize_environment_features(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = features.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    features = features.replace([np.inf, -np.inf], np.nan)
    all_columns = features.columns.copy()
    finite_counts = features.notna().sum(axis=0).reindex(all_columns, fill_value=0)
    mean = features.mean(axis=0).reindex(all_columns)
    filled = features.fillna(mean)
    std = filled.std(axis=0).reindex(all_columns)
    retained = (finite_counts > 0) & np.isfinite(mean) & np.isfinite(std) & std.gt(0)
    drop_reason = pd.Series("", index=all_columns, dtype=object)
    drop_reason.loc[finite_counts.eq(0)] = "no_finite_values"
    drop_reason.loc[finite_counts.gt(0) & ~np.isfinite(mean)] = "nonfinite_mean"
    drop_reason.loc[finite_counts.gt(0) & np.isfinite(mean) & (~np.isfinite(std) | std.le(0))] = "constant_or_nonfinite_std"
    scaling = pd.DataFrame(
        {
            "feature": all_columns,
            "mean": mean.to_numpy(),
            "std": std.to_numpy(),
            "n_nonmissing": finite_counts.to_numpy(),
            "retained": retained.to_numpy(),
            "drop_reason": drop_reason.to_numpy(),
        }
    )
    if not retained.any():
        return features.iloc[:, 0:0], scaling
    kept_columns = all_columns[retained]
    z = ((filled[kept_columns] - mean[kept_columns]) / std[kept_columns]).astype(np.float32)
    if not np.isfinite(z.to_numpy()).all():
        raise ValueError("Environment standardization produced nonfinite values")
    return z, scaling


def standardized_kernel(features: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    z, scaling = standardize_environment_features(features)
    if z.shape[1] == 0:
        return np.zeros((len(features), len(features)), dtype=np.float32), z, scaling
    K = (z.to_numpy(dtype=np.float32) @ z.to_numpy(dtype=np.float32).T) / max(z.shape[1], 1)
    K = ((K + K.T) * np.float32(0.5)).astype(np.float32)
    if not np.isfinite(K).all():
        raise ValueError("Environment kernel contains nonfinite values")
    return K.astype(np.float32), z, scaling


def assert_kernel_valid(kernel: np.ndarray, label: str, sample_size: int = 512) -> None:
    matrix = np.asarray(kernel)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{label} must be square; found {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains nonfinite values")
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    if symmetry_error > 1e-5:
        raise ValueError(f"{label} is asymmetric; max_abs_diff={symmetry_error}")
    selected = np.linspace(0, len(matrix) - 1, min(len(matrix), sample_size), dtype=int)
    block = np.asarray(matrix[np.ix_(selected, selected)], dtype=np.float64)
    min_eigenvalue = float(np.linalg.eigvalsh((block + block.T) / 2.0).min())
    if min_eigenvalue < -1e-4:
        raise ValueError(f"{label} is materially non-PSD; sampled_min_eigenvalue={min_eigenvalue}")


def scale_kernel_mean_diagonal(kernel: np.ndarray) -> tuple[np.ndarray, float, float]:
    raw = np.asarray(kernel, dtype=np.float32)
    mean_diag_raw = float(np.mean(np.diag(raw))) if raw.size else 0.0
    if not np.isfinite(mean_diag_raw) or mean_diag_raw <= 0:
        return raw.copy(), mean_diag_raw, mean_diag_raw
    scaled = (raw / mean_diag_raw).astype(np.float32)
    return scaled, mean_diag_raw, float(np.mean(np.diag(scaled)))


def component_weights(nonempty: list[str], component_names: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    raw = {}
    for name in component_names:
        if name not in nonempty:
            raw[name] = 0.0
            continue
        variable = f"ENV_WEIGHT_{name.upper()}"
        value = os.environ.get(variable, "1.0")
        try:
            raw[name] = float(value)
        except ValueError as exc:
            raise SystemExit(f"{variable} must be numeric; found {value!r}") from exc
        if not np.isfinite(raw[name]) or raw[name] < 0:
            raise SystemExit(f"{variable} must be finite and nonnegative; found {raw[name]}")
    total = sum(raw.values())
    if total <= 0:
        raise SystemExit("At least one non-empty environment component must have a positive weight")
    return raw, {name: raw[name] / total for name in component_names}


def component_activity(feature_count: int, mean_diag_raw: float) -> tuple[bool, str]:
    if feature_count <= 0:
        return False, "no_features"
    if not np.isfinite(mean_diag_raw):
        return False, "nonfinite_mean_diagonal"
    if mean_diag_raw <= 0:
        return False, "zero_variance_kernel"
    return True, ""


def main(
    environment_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    weather_dir: Path | str | None = None,
    require_fetched_weather: bool = False,
) -> None:
    environment_dir = Path(environment_dir) if environment_dir is not None else OUT
    output_dir = Path(output_dir) if output_dir is not None else OUT
    weather_dir = Path(weather_dir) if weather_dir is not None else environment_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    env = pd.read_csv(environment_dir / "envdata.tsv", sep="\t", dtype=str, low_memory=False)
    loc = pd.read_csv(environment_dir / "locdata.tsv", sep="\t", dtype=str, low_memory=False)
    keyed_locations = add_location_keys(loc)
    location_collision_audit(loc).to_csv(output_dir / "qc_location_key_collisions.tsv", sep="\t", index=False)

    env_base = env[[*ID_COLS]].drop_duplicates().copy()
    env_base["env_id"] = env_id_from_frame(env_base)
    env_ids = pd.Index(env_base["env_id"].drop_duplicates())

    environment_order = pd.DataFrame({"env_id": env_ids})
    environment_order.to_csv(output_dir / "env_kernel_sample_order.tsv", sep="\t", index=False)
    environment_order.assign(row_index=np.arange(len(environment_order), dtype=np.int32))[
        ["row_index", "env_id"]
    ].to_csv(output_dir / "env_kernel_row_order.tsv", sep="\t", index=False)
    environment_order.assign(column_index=np.arange(len(environment_order), dtype=np.int32))[
        ["column_index", "env_id"]
    ].to_csv(output_dir / "env_kernel_column_order.tsv", sep="\t", index=False)

    trait_matrix = build_env_trait_matrix(env, output_dir / "env_feature_value_parsing_qc.tsv")
    trait_matrix = trait_matrix.reindex(env_ids)
    fetched_weather, fetched_stress = build_fetched_weather_feature_sets(
        env_ids, environment_dir=weather_dir
    )
    if require_fetched_weather:
        weather_feature_count = fetched_weather.dropna(axis=1, how="all").shape[1]
        stress_feature_count = fetched_stress.dropna(axis=1, how="all").shape[1]
        if weather_feature_count == 0 or stress_feature_count == 0:
            raise SystemExit(
                "Recovered build requires nonempty fetched weather and stress features; "
                f"weather={weather_feature_count}; stress={stress_feature_count}"
            )

    feature_sets = {
        "geo": build_geo_features(env, loc, env_ids),
        "weather": fetched_weather
        if fetched_weather.dropna(axis=1, how="all").shape[1] > 0
        else trait_matrix[trait_group_columns(trait_matrix.columns, "weather")],
        "stress": fetched_stress
        if fetched_stress.dropna(axis=1, how="all").shape[1] > 0
        else trait_matrix[trait_group_columns(trait_matrix.columns, "stress")],
        "mgmt": trait_matrix[trait_group_columns(trait_matrix.columns, "mgmt")],
    }

    kernels: dict[str, np.ndarray] = {}
    component_stats: dict[str, dict[str, float | int]] = {}
    manifest_rows = []
    coverage_rows = []
    scaling_rows = []
    for name, features in feature_sets.items():
        K_raw, z, scaling = standardized_kernel(features)
        feature_count = z.shape[1]
        covered_envs = int(features[z.columns].notna().any(axis=1).sum()) if feature_count else 0
        coverage_rows.append(
            {
                "kernel": name,
                "env_id_total": len(features),
                "env_id_with_any_feature": covered_envs,
                "env_id_without_any_feature": len(features) - covered_envs,
                "feature_count": feature_count,
                "input_feature_count": features.shape[1],
                "dropped_feature_count": features.shape[1] - feature_count,
            }
        )
        K_scaled, mean_diag_raw, mean_diag_scaled = scale_kernel_mean_diagonal(K_raw)
        active, inactive_reason = component_activity(feature_count, mean_diag_raw)
        kernels[name] = K_scaled
        assert_kernel_valid(K_scaled, f"K_{name}")
        component_stats[name] = {
            "feature_count": feature_count,
            "mean_diag_raw": mean_diag_raw,
            "mean_diag_scaled": mean_diag_scaled,
            "coverage_env_count": covered_envs,
            "active_component": active,
            "inactive_reason": inactive_reason,
        }
        np.save(output_dir / f"K_{name}.raw.npy", K_raw)
        np.save(output_dir / f"K_{name}.npy", K_scaled)
        z.reset_index(names="env_id").to_parquet(output_dir / f"env_features_{name}.parquet", index=False)
        if not scaling.empty:
            scaling = scaling.copy()
            scaling.insert(0, "kernel", name)
            scaling_rows.extend(scaling.to_dict("records"))
        for col in z.columns:
            manifest_rows.append({"kernel": name, "feature": col})
        print(f"K_{name}", K_scaled.shape, "features", z.shape[1], "mean_diag", mean_diag_scaled)

    active_components = [name for name in feature_sets if component_stats[name]["active_component"]]
    raw_weights, weights = component_weights(active_components, list(feature_sets))
    K_E_raw = sum(weights[name] * kernels[name] for name in feature_sets).astype(np.float32)
    K_E, environment_mean_diag_raw, environment_mean_diag_scaled = scale_kernel_mean_diagonal(K_E_raw)
    assert_kernel_valid(K_E, "K_E")
    np.save(output_dir / "K_E.raw.npy", K_E_raw)
    np.save(output_dir / "K_E.npy", K_E)
    builder_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output_dir / "K_E.qc.json").write_text(json.dumps({
        "builder_sha256": builder_sha256,
        "environment_input_dir": str(environment_dir.resolve()),
        "environment_output_dir": str(output_dir.resolve()),
        "weather_feature_input_dir": str(weather_dir.resolve()),
        "active_components": active_components,
        "environment_mean_diag_raw": environment_mean_diag_raw,
        "environment_mean_diag_scaled": environment_mean_diag_scaled,
        "loc_no_missing_count": int(keyed_locations["loc_no_missing"].sum()),
        "location_hash_fallback_count": int(keyed_locations["location_key_method"].eq("unresolved_location_hash_fallback").sum()),
        "country_loc_desc_fallback_count": int(keyed_locations["location_key_method"].eq("country_loc_desc_fallback").sum()),
        "empty_loc_no_excluded_from_fallback": True,
    }, indent=2), encoding="utf-8")

    pd.DataFrame(
        [
            {
                "kernel": name,
                "raw_weight": raw_weights[name],
                "normalized_weight": weights[name],
                **component_stats[name],
                "environment_mean_diag_raw": environment_mean_diag_raw,
                "environment_mean_diag_scaled": environment_mean_diag_scaled,
            }
            for name in feature_sets
        ]
    ).to_csv(output_dir / "env_kernel_component_weights.tsv", sep="\t", index=False)
    pd.DataFrame(manifest_rows).to_csv(output_dir / "env_kernel_feature_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(coverage_rows).to_csv(output_dir / "env_kernel_coverage_summary.tsv", sep="\t", index=False)
    pd.DataFrame(scaling_rows).to_csv(output_dir / "env_feature_scaling_parameters.tsv", sep="\t", index=False)

    print("K_E", K_E.shape, "weights", weights, "mean_diag", float(np.mean(np.diag(K_E))))


def cli() -> None:
    parser = argparse.ArgumentParser(description="Build validated environment component kernels.")
    parser.add_argument(
        "--environment-dir",
        type=Path,
        default=OUT,
        help="Directory containing envdata.tsv, locdata.tsv, and fetched weather tables.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT,
        help="Output directory. Use a new path for non-destructive regeneration.",
    )
    parser.add_argument(
        "--weather-dir",
        type=Path,
        default=None,
        help="Optional directory containing recovered fetched-weather tables.",
    )
    parser.add_argument(
        "--require-fetched-weather",
        action="store_true",
        help="Fail instead of falling back to trial traits when fetched weather is absent.",
    )
    args = parser.parse_args()
    main(
        environment_dir=args.environment_dir,
        output_dir=args.out_dir,
        weather_dir=args.weather_dir,
        require_fetched_weather=args.require_fetched_weather,
    )


if __name__ == "__main__":
    cli()
