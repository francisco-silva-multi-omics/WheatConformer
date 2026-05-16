from __future__ import annotations

import re
import platform
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOCAL_DEPS = BASE / ".codex_deps"
if platform.system() == "Windows" and LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd

OUT = BASE / "environment"
ID_COLS = ["Trial_name", "Occ", "Loc_no", "Country", "Loc_desc", "Cycle"]


def env_id_from_frame(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ID_COLS if c in df.columns]
    return df[cols].apply(lambda row: "|".join(row.map(lambda x: "" if pd.isna(x) else str(x))), axis=1)


def parse_value(value: object, trait: str) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    upper = text.upper()
    if upper in {"YES", "Y", "TRUE", "T", "IRRIGATED", "APPLIED"}:
        return 1.0
    if upper in {"NO", "N", "FALSE", "NONE", "NIL", "NOT APPLIED"}:
        return 0.0

    if "DATE" in trait.upper() or trait.upper() in {"SOWING_OLD"}:
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
        if not pd.isna(parsed):
            return float(parsed.dayofyear)

    cleaned = re.sub(r"[^0-9.+Ee-]", "", text.replace(",", ""))
    if cleaned in {"", ".", "-", "+", "+.", "-."}:
        return np.nan
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def normalize_loc_no(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def decimal_degrees(deg: pd.Series, minutes: pd.Series, hemi: pd.Series) -> pd.Series:
    d = pd.to_numeric(deg, errors="coerce")
    m = pd.to_numeric(minutes, errors="coerce").fillna(0)
    val = d + m / 60.0
    h = hemi.astype(str).str.upper().str.strip()
    sign = np.where(h.isin(["S", "W"]), -1.0, 1.0)
    return val * sign


def build_env_trait_matrix(env: pd.DataFrame) -> pd.DataFrame:
    tmp = env[[*ID_COLS, "Trait_name", "Value"]].copy()
    tmp["env_id"] = env_id_from_frame(tmp)
    tmp["feature_value"] = [
        parse_value(value, trait) for value, trait in zip(tmp["Value"].to_numpy(), tmp["Trait_name"].to_numpy())
    ]
    return tmp.pivot_table(index="env_id", columns="Trait_name", values="feature_value", aggfunc="mean")


def build_geo_features(env: pd.DataFrame, loc: pd.DataFrame, env_ids: pd.Index) -> pd.DataFrame:
    env_base = env[[*ID_COLS]].drop_duplicates().copy()
    env_base["env_id"] = env_id_from_frame(env_base)
    env_base["Loc_no_key"] = normalize_loc_no(env_base["Loc_no"])

    loc_work = loc.copy()
    loc_work["Loc_no_key"] = normalize_loc_no(loc_work["Loc_no"])
    loc_work["latitude"] = decimal_degrees(loc_work["Lat_degress"], loc_work["Lat_minutes"], loc_work["Latitud"])
    loc_work["longitude"] = decimal_degrees(loc_work["Long_degress"], loc_work["Long_minutes"], loc_work["Longitude"])
    loc_work["altitude"] = pd.to_numeric(loc_work["Altitude"], errors="coerce")
    loc_by_id = loc_work.groupby("Loc_no_key")[["latitude", "longitude", "altitude"]].mean()

    geo = env_base[["env_id", "Loc_no_key"]].merge(loc_by_id, left_on="Loc_no_key", right_index=True, how="left")
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


def build_fetched_weather_feature_sets(env_ids: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    nasa_weather, nasa_stress = build_nasa_power_feature_sets(env_ids, weather_cols, stress_cols)
    openmeteo_weather, openmeteo_stress = build_openmeteo_feature_sets(env_ids, weather_cols, stress_cols)
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
    env_ids: pd.Index, weather_cols: list[str], stress_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nasa_path = OUT / "trial_weather_features_nasa_power.tsv"
    fetched = read_weather_feature_table(nasa_path, env_ids)
    weather = fetched[[c for c in weather_cols if c in fetched.columns]].copy()
    stress = fetched[[c for c in stress_cols if c in fetched.columns]].copy()
    return weather, stress


def build_openmeteo_feature_sets(
    env_ids: pd.Index, weather_cols: list[str], stress_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    openmeteo_path = OUT / "trial_weather_features_openmeteo.tsv"
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


def standardized_kernel(features: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    features = features.copy()
    features = features.dropna(axis=1, how="all")
    finite_counts = features.notna().sum(axis=0)
    features = features.loc[:, finite_counts > 0]
    if features.shape[1] == 0:
        n = len(features)
        scaling = pd.DataFrame(columns=["feature", "mean", "std", "n_nonmissing"])
        return np.zeros((n, n), dtype=np.float32), features, scaling
    features = features.astype(np.float32)
    mean = features.mean(axis=0)
    filled = features.fillna(mean)
    std = filled.std(axis=0).replace(0, np.nan)
    z = ((filled - mean) / std).fillna(0.0).astype(np.float32)
    K = (z.to_numpy(dtype=np.float32) @ z.to_numpy(dtype=np.float32).T) / max(z.shape[1], 1)
    scaling = pd.DataFrame(
        {
            "feature": features.columns,
            "mean": mean.reindex(features.columns).to_numpy(),
            "std": std.reindex(features.columns).to_numpy(),
            "n_nonmissing": finite_counts.reindex(features.columns).to_numpy(),
        }
    )
    return K.astype(np.float32), z, scaling


def main() -> None:
    env = pd.read_csv(OUT / "envdata.tsv", sep="\t", dtype=str, low_memory=False)
    loc = pd.read_csv(OUT / "locdata.tsv", sep="\t", dtype=str, low_memory=False)

    env_base = env[[*ID_COLS]].drop_duplicates().copy()
    env_base["env_id"] = env_id_from_frame(env_base)
    env_ids = pd.Index(env_base["env_id"].drop_duplicates())

    pd.DataFrame({"env_id": env_ids}).to_csv(OUT / "env_kernel_sample_order.tsv", sep="\t", index=False)

    trait_matrix = build_env_trait_matrix(env)
    trait_matrix = trait_matrix.reindex(env_ids)
    fetched_weather, fetched_stress = build_fetched_weather_feature_sets(env_ids)

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
    manifest_rows = []
    coverage_rows = []
    scaling_rows = []
    for name, features in feature_sets.items():
        feature_count = features.dropna(axis=1, how="all").shape[1]
        covered_envs = int(features.notna().any(axis=1).sum()) if feature_count else 0
        coverage_rows.append(
            {
                "kernel": name,
                "env_id_total": len(features),
                "env_id_with_any_feature": covered_envs,
                "env_id_without_any_feature": len(features) - covered_envs,
                "feature_count": feature_count,
            }
        )
        K, z, scaling = standardized_kernel(features)
        kernels[name] = K
        np.save(OUT / f"K_{name}.npy", K)
        z.reset_index(names="env_id").to_parquet(OUT / f"env_features_{name}.parquet", index=False)
        if not scaling.empty:
            scaling = scaling.copy()
            scaling.insert(0, "kernel", name)
            scaling_rows.extend(scaling.to_dict("records"))
        for col in z.columns:
            manifest_rows.append({"kernel": name, "feature": col})
        print(f"K_{name}", K.shape, "features", z.shape[1], "mean_diag", float(np.mean(np.diag(K))))

    nonempty = [name for name, features in feature_sets.items() if features.dropna(axis=1, how="all").shape[1] > 0]
    weights = {name: (1.0 / len(nonempty) if name in nonempty else 0.0) for name in feature_sets}
    K_E = sum(weights[name] * kernels[name] for name in feature_sets).astype(np.float32)
    np.save(OUT / "K_E.npy", K_E)

    pd.DataFrame({"kernel": list(weights.keys()), "weight": list(weights.values())}).to_csv(
        OUT / "env_kernel_component_weights.tsv", sep="\t", index=False
    )
    pd.DataFrame(manifest_rows).to_csv(OUT / "env_kernel_feature_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(coverage_rows).to_csv(OUT / "env_kernel_coverage_summary.tsv", sep="\t", index=False)
    pd.DataFrame(scaling_rows).to_csv(OUT / "env_feature_scaling_parameters.tsv", sep="\t", index=False)

    print("K_E", K_E.shape, "weights", weights, "mean_diag", float(np.mean(np.diag(K_E))))


if __name__ == "__main__":
    main()
