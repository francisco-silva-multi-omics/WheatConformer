from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLS = ["Trial_name", "Occ", "Loc_no", "Country", "Loc_desc", "Cycle"]
OBSERVED_TRAITS = [
    "PRECIPITATION_FROM_SOWING_TO_MATURITY",
    "PRECIPITATION_ON_CROP",
    "TOTAL_PRECIPIT_IN_12_MONTHS",
    "ESTIMATE_TOTAL_PRECIPIT_IN_12_MONTHS",
    "PRECIPITATION_AVAILABLE_TO_CROP_AFTER_SOWING_OLD",
    "MOISTURE_AVAILB_BEFORE_SOWING_EXCL_PRE_IRRIGATION",
    "MOISTURE_AVAILABLE_IN_FULL_ROOT_ZONE_AT_SOWING_OLD",
    "NO_OF_RAINS_DURING_CYCLE_OLD",
    "PPN_MONTH_OF_HARVESTED",
    "PPN_1ST_MO_BEFORE_HARVESTED",
    "PPN_2ND_MO_BEFORE_HARVESTED",
    "PPN_3RD_MO_BEFORE_HARVESTED",
    "PPN_4TH_MO_BEFORE_HARVESTED",
    "PPN_5TH_MO_BEFORE_HARVESTED",
    "PPN_6TH_MO_BEFORE_HARVESTED",
    "PPN_7TH_MO_BEFORE_HARVESTED",
    "PPN_8TH_MO_BEFORE_HARVESTED",
    "PPN_9TH_MO_BEFORE_HARVESTED",
    "PPN_10TH_MO_BEFORE_HARVESTED",
    "PPN_11TH_MO_BEFORE_HARVESTED",
    "IRRIGATED",
    "NUMBER_POST_SOWING_IRRIGATIONS",
    "NUMBER_PRE_SOWING_IRRIGATIONS",
    "IRRIGATION_AFTER_SOWING",
    "PRE_SOWING_IRRIGATION",
]


def read_order(path: Path) -> pd.DataFrame:
    order = pd.read_csv(path, sep="\t", dtype=str)
    id_col = "env_id" if "env_id" in order.columns else order.columns[0]
    order = order.rename(columns={id_col: "env_id"})
    return order


def env_id_from_parts(df: pd.DataFrame) -> pd.Series:
    return df[ID_COLS].apply(lambda row: "|".join("" if pd.isna(x) else str(x) for x in row), axis=1)


def parse_numeric(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    upper = text.upper()
    if upper in {"YES", "Y", "TRUE", "IRRIGATED", "APPLIED"}:
        return 1.0
    if upper in {"NO", "N", "FALSE", "NONE", "NIL", "NOT APPLIED"}:
        return 0.0
    cleaned = re.sub(r"[^0-9.+Ee-]", "", text.replace(",", ""))
    if cleaned in {"", ".", "-", "+", "+.", "-."}:
        return np.nan
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def parse_date(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    return pd.to_datetime(str(value).strip(), errors="coerce", dayfirst=False)


def decimal_degrees(deg: pd.Series, minutes: pd.Series, hemi: pd.Series) -> pd.Series:
    d = pd.to_numeric(deg, errors="coerce")
    m = pd.to_numeric(minutes, errors="coerce").fillna(0)
    val = d + m / 60.0
    h = hemi.fillna("").astype(str).str.upper().str.strip()
    sign = np.where(h.isin(["S", "W"]), -1.0, 1.0)
    return val * sign


def normalize_loc_no(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def base_env_table(envdata: pd.DataFrame, env_ids: pd.Series) -> pd.DataFrame:
    parts = env_ids.str.split("|", expand=True)
    cols = ["Trial_name", "Occ", "Loc_no", "Country", "Loc_desc", "Cycle"]
    out = pd.DataFrame({c: parts[i] if i in parts.columns else "" for i, c in enumerate(cols)})
    out.insert(0, "env_id", env_ids.to_numpy())
    return out


def build_geo(env_base: pd.DataFrame, locdata: pd.DataFrame) -> pd.DataFrame:
    loc = locdata.copy()
    loc["Loc_no_key"] = normalize_loc_no(loc["Loc_no"])
    loc["latitude"] = decimal_degrees(loc["Lat_degress"], loc["Lat_minutes"], loc["Latitud"])
    loc["longitude"] = decimal_degrees(loc["Long_degress"], loc["Long_minutes"], loc["Longitude"])
    loc["altitude_m"] = pd.to_numeric(loc["Altitude"], errors="coerce")
    loc_by_id = loc.groupby("Loc_no_key")[["latitude", "longitude", "altitude_m"]].mean()
    out = env_base[["env_id", "Loc_no"]].copy()
    out["Loc_no_key"] = normalize_loc_no(out["Loc_no"])
    out = out.merge(loc_by_id, left_on="Loc_no_key", right_index=True, how="left")
    return out.set_index("env_id")[["latitude", "longitude", "altitude_m"]]


def build_observed_envdata(envdata: pd.DataFrame, env_ids: pd.Series) -> pd.DataFrame:
    x = envdata[envdata["Trait_name"].isin(OBSERVED_TRAITS + ["WEATHER_COMMENTS", "SOWING_DATE"])].copy()
    x["env_id"] = env_id_from_parts(x)
    x["value_num"] = [parse_numeric(v) for v in x["Value"]]
    numeric = x[x["Trait_name"].isin(OBSERVED_TRAITS)].copy()
    wide = numeric.pivot_table(index="env_id", columns="Trait_name", values="value_num", aggfunc="mean")
    wide = wide.reindex(env_ids)

    comments = x[x["Trait_name"].eq("WEATHER_COMMENTS")]
    wide["has_weather_comment"] = wide.index.isin(set(comments["env_id"].astype(str))).astype(float)
    for key in ["DROUGHT", "DRY", "HEAT", "HOT", "RAIN", "FROST", "COLD", "HAIL"]:
        env_with_key = set(comments.loc[comments["Value"].astype(str).str.upper().str.contains(key, na=False), "env_id"])
        wide[f"weather_comment_{key.lower()}"] = wide.index.isin(env_with_key).astype(float)

    sow = x[x["Trait_name"].eq("SOWING_DATE")].copy()
    sow["parsed"] = sow["Value"].map(parse_date)
    sow = sow.dropna(subset=["parsed"]).drop_duplicates("env_id").set_index("env_id")["parsed"].reindex(env_ids)
    doy = sow.dt.dayofyear.astype(float)
    wide["sowing_dayofyear"] = doy
    wide["sowing_dayofyear_sin"] = np.sin(2 * math.pi * doy / 365.25)
    wide["sowing_dayofyear_cos"] = np.cos(2 * math.pi * doy / 365.25)
    wide["has_sowing_date"] = doy.notna().astype(float)
    return wide


def build_window_features(path: Path, env_ids: pd.Series) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(index=env_ids)
    x = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    if "fetch_status" in x.columns:
        x = x[x["fetch_status"].astype(str).str.lower().eq("ok")].copy()
    if x.empty:
        return pd.DataFrame(index=env_ids)
    id_cols = {"env_id", "window_label", "weather_request_id", "fetch_status", "window_start_date", "window_end_date"}
    numeric_cols = [c for c in x.columns if c not in id_cols]
    for col in numeric_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    pieces = []
    for label, sub in x.groupby("window_label", dropna=False):
        label = str(label)
        agg = sub.drop_duplicates("env_id").set_index("env_id")[numeric_cols].add_prefix(f"api_{label}_")
        pieces.append(agg)
    if not pieces:
        return pd.DataFrame(index=env_ids)
    return pd.concat(pieces, axis=1).reindex(env_ids)


def zscore_with_missing(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = features.copy().dropna(axis=1, how="all")
    z_parts = []
    scaling = []
    for col in features.columns:
        x = pd.to_numeric(features[col], errors="coerce")
        n = int(x.notna().sum())
        if n == 0:
            continue
        miss = x.isna().astype(np.float32)
        mean = float(x.mean())
        filled = x.fillna(mean)
        std = float(filled.std(ddof=0))
        if not np.isfinite(std) or std == 0:
            continue
        z_parts.append(((filled - mean) / std).rename(str(col)))
        if miss.sum() > 0:
            z_parts.append(miss.rename(f"{col}__missing"))
        scaling.append({"feature": col, "mean": mean, "std": std, "n_nonmissing": n})
    if not z_parts:
        return pd.DataFrame(index=features.index), pd.DataFrame(columns=["feature", "mean", "std", "n_nonmissing"])
    return pd.concat(z_parts, axis=1).astype(np.float32), pd.DataFrame(scaling)


def kernel_from_features(z: pd.DataFrame) -> np.ndarray:
    X = z.to_numpy(dtype=np.float64)
    K = (X @ X.T) / max(X.shape[1], 1)
    K = (K + K.T) / 2
    diag = np.diag(K).copy()
    good = np.isfinite(diag) & (diag > 0)
    if good.any():
        diag[~good] = np.nanmedian(diag[good])
        K = K / np.sqrt(np.outer(diag, diag))
    np.fill_diagonal(K, 1.0)
    return ((K + K.T) / 2).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DTH-specific environment features and K_E variant.")
    parser.add_argument("--base-model-dir", type=Path, default=Path("model_kernels/stage1_pedigree_env"))
    parser.add_argument("--prefix", default="stage1_pedigree_env")
    parser.add_argument("--out-model-dir", type=Path, default=Path("model_kernels/stage1_pedigree_env_dth_v2"))
    parser.add_argument("--window-features", type=Path, default=Path("environment/dth_api_weather_windows.tsv"))
    parser.add_argument("--envdata", type=Path, default=Path("environment/envdata.tsv"))
    parser.add_argument("--locdata", type=Path, default=Path("environment/locdata.tsv"))
    args = parser.parse_args()

    order = read_order(args.base_model_dir / f"{args.prefix}_K_E_unique_order.tsv")
    env_ids = order["env_id"].astype(str).reset_index(drop=True)
    envdata = pd.read_csv(args.envdata, sep="\t", dtype=str, low_memory=False)
    locdata = pd.read_csv(args.locdata, sep="\t", dtype=str, low_memory=False)
    env_base = base_env_table(envdata, env_ids)

    feature_sets = {
        "geo": build_geo(env_base, locdata).reindex(env_ids),
        "observed_envdata": build_observed_envdata(envdata, env_ids),
        "api_sowing_windows": build_window_features(args.window_features, env_ids),
    }
    features = pd.concat(feature_sets.values(), axis=1)
    features.index = env_ids
    z, scaling = zscore_with_missing(features)
    K = kernel_from_features(z)

    if args.out_model_dir.exists():
        shutil.rmtree(args.out_model_dir)
    shutil.copytree(args.base_model_dir, args.out_model_dir)
    np.save(args.out_model_dir / f"{args.prefix}_K_E_unique.npy", K)
    z.reset_index(names="env_id").to_parquet(args.out_model_dir / f"{args.prefix}_DTH_env_features_v2.parquet", index=False)
    z.reset_index(names="env_id").to_csv(
        args.out_model_dir / f"{args.prefix}_DTH_env_features_v2.tsv.gz", sep="\t", index=False
    )
    scaling.to_csv(args.out_model_dir / f"{args.prefix}_DTH_env_features_v2_scaling.tsv", sep="\t", index=False)

    manifest_rows = []
    for group, df in feature_sets.items():
        for col in df.dropna(axis=1, how="all").columns:
            manifest_rows.append({"feature_group": group, "feature": col})
    pd.DataFrame(manifest_rows).to_csv(
        args.out_model_dir / f"{args.prefix}_DTH_env_features_v2_manifest.tsv", sep="\t", index=False
    )
    qc = pd.DataFrame(
        [
            {"metric": "env_count", "value": len(env_ids)},
            {"metric": "feature_count_after_missing_indicators", "value": z.shape[1]},
            {"metric": "K_E_shape", "value": f"{K.shape[0]}x{K.shape[1]}"},
            {"metric": "K_E_mean_diag", "value": float(np.diag(K).mean())},
            {"metric": "K_E_min_diag", "value": float(np.diag(K).min())},
            {"metric": "K_E_max_diag", "value": float(np.diag(K).max())},
            {"metric": "K_E_symmetry_max_abs", "value": float(np.max(np.abs(K - K.T)))},
        ]
    )
    qc.to_csv(args.out_model_dir / f"{args.prefix}_DTH_env_features_v2_qc.tsv", sep="\t", index=False)
    print(qc.to_string(index=False))
    print(f"Wrote {args.out_model_dir}")


if __name__ == "__main__":
    main()
