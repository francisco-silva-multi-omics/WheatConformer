from __future__ import annotations

import re
import sys
from pathlib import Path

cwd = Path(__file__).resolve().parent

import numpy as np
import pandas as pd

OUT = cwd / "environment"
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


def standardized_kernel(features: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    features = features.copy()
    features = features.dropna(axis=1, how="all")
    finite_counts = features.notna().sum(axis=0)
    features = features.loc[:, finite_counts > 0]
    if features.shape[1] == 0:
        n = len(features)
        return np.zeros((n, n), dtype=np.float32), features
    features = features.astype(np.float32)
    features = features.fillna(features.mean(axis=0))
    std = features.std(axis=0).replace(0, np.nan)
    z = ((features - features.mean(axis=0)) / std).fillna(0.0).astype(np.float32)
    K = (z.to_numpy(dtype=np.float32) @ z.to_numpy(dtype=np.float32).T) / max(z.shape[1], 1)
    return K.astype(np.float32), z


def main() -> None:
    env = pd.read_csv(OUT / "envdata.tsv", sep="\t", dtype=str, low_memory=False)
    loc = pd.read_csv(OUT / "locdata.tsv", sep="\t", dtype=str, low_memory=False)

    env_base = env[[*ID_COLS]].drop_duplicates().copy()
    env_base["env_id"] = env_id_from_frame(env_base)
    env_ids = pd.Index(env_base["env_id"].drop_duplicates())

    pd.DataFrame({"env_id": env_ids}).to_csv(OUT / "env_kernel_sample_order.tsv", sep="\t", index=False)

    trait_matrix = build_env_trait_matrix(env)
    trait_matrix = trait_matrix.reindex(env_ids)

    feature_sets = {
        "geo": build_geo_features(env, loc, env_ids),
        "weather": trait_matrix[trait_group_columns(trait_matrix.columns, "weather")],
        "stress": trait_matrix[trait_group_columns(trait_matrix.columns, "stress")],
        "mgmt": trait_matrix[trait_group_columns(trait_matrix.columns, "mgmt")],
    }

    kernels: dict[str, np.ndarray] = {}
    manifest_rows = []
    for name, features in feature_sets.items():
        K, z = standardized_kernel(features)
        kernels[name] = K
        np.save(OUT / f"K_{name}.npy", K)
        z.reset_index(names="env_id").to_parquet(OUT / f"env_features_{name}.parquet", index=False)
        for col in z.columns:
            manifest_rows.append({"kernel": name, "feature": col})
        print(f"K_{name}", K.shape, "características", z.shape[1], "diag_promedio", float(np.mean(np.diag(K))))

    nonempty = [name for name, features in feature_sets.items() if features.dropna(axis=1, how="all").shape[1] > 0]
    weights = {name: (1.0 / len(nonempty) if name in nonempty else 0.0) for name in feature_sets}
    K_E = sum(weights[name] * kernels[name] for name in feature_sets).astype(np.float32)
    np.save(OUT / "K_E.npy", K_E)

    pd.DataFrame({"kernel": list(weights.keys()), "weight": list(weights.values())}).to_csv(
        OUT / "env_kernel_component_weights.tsv", sep="\t", index=False
    )
    pd.DataFrame(manifest_rows).to_csv(OUT / "env_kernel_feature_manifest.tsv", sep="\t", index=False)

    print("K_E", K_E.shape, "pesos", weights, "diag_promedio", float(np.mean(np.diag(K_E))))


if __name__ == "__main__":
    main()
