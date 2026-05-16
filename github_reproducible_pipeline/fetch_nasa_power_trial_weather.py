from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
OUT = BASE / "environment"
MANIFEST = OUT / "trial_weather_fetch_manifest.tsv"
REQUEST_OUT = OUT / "trial_weather_request_features_nasa_power.tsv"
FEATURE_OUT = OUT / "trial_weather_features_nasa_power.tsv"
QC_OUT = OUT / "trial_weather_fetch_nasa_power_qc.tsv"
FAIL_OUT = OUT / "trial_weather_fetch_nasa_power_failures.tsv"
NASA_START = pd.Timestamp("1981-01-01")


PARAMETERS = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN",
    "WS2M",
]


def request_key(row: pd.Series) -> str:
    return "|".join(
        [
            f"{row.latitude:.5f}",
            f"{row.longitude:.5f}",
            str(row.nasa_start_date),
            str(row.nasa_end_date),
        ]
    )


def request_url(row: pd.Series) -> str:
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": f"{row.longitude:.5f}",
        "latitude": f"{row.latitude:.5f}",
        "start": str(row.nasa_start_date).replace("-", ""),
        "end": str(row.nasa_end_date).replace("-", ""),
        "format": "JSON",
    }
    return "https://power.larc.nasa.gov/api/temporal/daily/point?" + urllib.parse.urlencode(params)


def fetch_json(url: str, timeout: int = 90, retries: int = 4, sleep_seconds: float = 2.0) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "trial-weather-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in {429, 500, 502, 503, 504}:
                time.sleep(sleep_seconds * (attempt + 1))
                continue
            raise
        except Exception as exc:
            last_error = exc
            time.sleep(sleep_seconds * (attempt + 1))
    raise RuntimeError(f"Request failed after retries: {last_error}")


def to_series(params: dict, name: str) -> pd.Series:
    values = params.get(name, {})
    if not values:
        return pd.Series(dtype=float)
    s = pd.Series(values, dtype="float64")
    s.index = pd.to_datetime(s.index, format="%Y%m%d", errors="coerce")
    s = s.replace(-999, np.nan)
    return s.sort_index()


def vpd_kpa(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    return es * (1 - rh_pct / 100.0)


def safe_mean(s: pd.Series) -> float:
    return float(s.mean()) if s.notna().any() else np.nan


def safe_sum(s: pd.Series) -> float:
    return float(s.sum()) if s.notna().any() else np.nan


def aggregate_payload(payload: dict, req: pd.Series) -> dict[str, object]:
    params = payload.get("properties", {}).get("parameter", {})
    tmean = to_series(params, "T2M")
    tmax = to_series(params, "T2M_MAX")
    tmin = to_series(params, "T2M_MIN")
    rh = to_series(params, "RH2M")
    precip = to_series(params, "PRECTOTCORR")
    radiation = to_series(params, "ALLSKY_SFC_SW_DWN")
    wind = to_series(params, "WS2M")

    daily = pd.DataFrame(
        {
            "temperature_mean_c": tmean,
            "temperature_max_c": tmax,
            "temperature_min_c": tmin,
            "relative_humidity_mean_pct": rh,
            "precipitation_mm": precip,
            "solar_radiation_mj_m2_day": radiation,
            "wind_speed_2m_m_s": wind,
        }
    )
    daily["vpd_mean_kpa"] = vpd_kpa(daily["temperature_mean_c"], daily["relative_humidity_mean_pct"])
    n_days = int(len(daily))

    out: dict[str, object] = {
        "weather_request_id": req.weather_request_id,
        "latitude": req.latitude,
        "longitude": req.longitude,
        "weather_start_date": req.weather_start_date,
        "weather_end_date": req.weather_end_date,
        "nasa_start_date": req.nasa_start_date,
        "nasa_end_date": req.nasa_end_date,
        "n_days_weather": n_days,
        "temperature_mean_c": safe_mean(daily["temperature_mean_c"]),
        "temperature_max_c": float(daily["temperature_max_c"].max()) if n_days else np.nan,
        "temperature_min_c": float(daily["temperature_min_c"].min()) if n_days else np.nan,
        "relative_humidity_mean_pct": safe_mean(daily["relative_humidity_mean_pct"]),
        "precipitation_total_mm": safe_sum(daily["precipitation_mm"]),
        "precipitation_mean_daily_mm": safe_mean(daily["precipitation_mm"]),
        "solar_radiation_total_mj_m2": safe_sum(daily["solar_radiation_mj_m2_day"]),
        "solar_radiation_mean_daily_mj_m2": safe_mean(daily["solar_radiation_mj_m2_day"]),
        "wind_speed_2m_mean_m_s": safe_mean(daily["wind_speed_2m_m_s"]),
        "vpd_mean_kpa": safe_mean(daily["vpd_mean_kpa"]),
        "vpd_max_kpa": float(daily["vpd_mean_kpa"].max()) if n_days else np.nan,
        "heat_days_tmax_ge_30": int((daily["temperature_max_c"] >= 30).sum()) if n_days else 0,
        "heat_days_tmax_ge_35": int((daily["temperature_max_c"] >= 35).sum()) if n_days else 0,
        "extreme_heat_days_tmax_ge_40": int((daily["temperature_max_c"] >= 40).sum()) if n_days else 0,
        "dry_days_precip_lt_1mm": int((daily["precipitation_mm"] < 1).sum()) if n_days else 0,
        "heavy_rain_days_precip_ge_20mm": int((daily["precipitation_mm"] >= 20).sum()) if n_days else 0,
        "high_vpd_days_gt_1_5": int((daily["vpd_mean_kpa"] > 1.5).sum()) if n_days else 0,
        "high_vpd_days_gt_2_0": int((daily["vpd_mean_kpa"] > 2.0).sum()) if n_days else 0,
    }
    out["drought_days_precip_lt_1mm_and_vpd_gt_1_5"] = (
        int(((daily["precipitation_mm"] < 1) & (daily["vpd_mean_kpa"] > 1.5)).sum()) if n_days else 0
    )
    out["radiation_vpd_stress_index"] = (
        out["solar_radiation_mean_daily_mj_m2"] * out["vpd_mean_kpa"]
        if pd.notna(out["solar_radiation_mean_daily_mj_m2"]) and pd.notna(out["vpd_mean_kpa"])
        else np.nan
    )
    return out


def append_tsv(row: dict[str, object], path: Path) -> None:
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False, mode="a", header=not path.exists(), lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-failures", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.1)
    args = parser.parse_args()

    if args.clear_failures and FAIL_OUT.exists():
        FAIL_OUT.unlink()

    manifest = pd.read_csv(MANIFEST, sep="\t", dtype=str)
    ready = manifest[manifest["ready_to_fetch"].astype(str).str.upper().eq("TRUE")].copy()
    for col in ["latitude", "longitude"]:
        ready[col] = pd.to_numeric(ready[col], errors="coerce")
    ready["weather_start"] = pd.to_datetime(ready["weather_start_date"], errors="coerce")
    ready["weather_end"] = pd.to_datetime(ready["weather_end_date"], errors="coerce")
    ready["nasa_start"] = ready["weather_start"].where(ready["weather_start"] >= NASA_START, NASA_START)
    ready = ready[ready["weather_end"] >= NASA_START].copy()
    ready["nasa_start_date"] = ready["nasa_start"].dt.strftime("%Y-%m-%d")
    ready["nasa_end_date"] = ready["weather_end"].dt.strftime("%Y-%m-%d")
    ready = ready[ready["latitude"].notna() & ready["longitude"].notna() & ready["nasa_start_date"].notna()].copy()
    ready["weather_request_id"] = ready.apply(request_key, axis=1)

    requests = (
        ready[
            [
                "weather_request_id",
                "latitude",
                "longitude",
                "weather_start_date",
                "weather_end_date",
                "nasa_start_date",
                "nasa_end_date",
            ]
        ]
        .drop_duplicates("weather_request_id")
        .sort_values("weather_request_id")
        .reset_index(drop=True)
    )

    if args.resume and REQUEST_OUT.exists():
        done = set(pd.read_csv(REQUEST_OUT, sep="\t", dtype=str, usecols=["weather_request_id"])["weather_request_id"])
        requests = requests[~requests["weather_request_id"].isin(done)].copy()
    if args.limit is not None:
        requests = requests.head(args.limit).copy()

    def fetch_one(req: pd.Series) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        url = request_url(req)
        try:
            payload = fetch_json(url)
            features = aggregate_payload(payload, req)
            features["fetch_status"] = "ok"
            return features, None
        except Exception as exc:
            return None, {
                "weather_request_id": req.weather_request_id,
                "latitude": req.latitude,
                "longitude": req.longitude,
                "nasa_start_date": req.nasa_start_date,
                "nasa_end_date": req.nasa_end_date,
                "error": str(exc),
                "url": url,
            }

    fetched = 0
    request_rows = [row for _, row in requests.iterrows()]
    if args.workers <= 1:
        for req in request_rows:
            features, fail = fetch_one(req)
            if features is not None:
                append_tsv(features, REQUEST_OUT)
                fetched += 1
                if fetched % 50 == 0:
                    print(f"Fetched {fetched} requests", flush=True)
            if fail is not None:
                append_tsv(fail, FAIL_OUT)
                print(f"Failed {fail['weather_request_id']}: {fail['error']}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_one, req): req.weather_request_id for req in request_rows}
            for future in as_completed(futures):
                features, fail = future.result()
                if features is not None:
                    append_tsv(features, REQUEST_OUT)
                    fetched += 1
                    if fetched % 50 == 0:
                        print(f"Fetched {fetched} requests", flush=True)
                if fail is not None:
                    append_tsv(fail, FAIL_OUT)
                    print(f"Failed {fail['weather_request_id']}: {fail['error']}", flush=True)
                if args.sleep:
                    time.sleep(args.sleep)

    if REQUEST_OUT.exists():
        req_features = pd.read_csv(REQUEST_OUT, sep="\t", dtype=str, low_memory=False)
        req_features = req_features.drop_duplicates("weather_request_id", keep="last")
        current_request_ids = set(ready["weather_request_id"])
        current_req_features = req_features[req_features["weather_request_id"].isin(current_request_ids)].copy()
        env_features = ready[["env_id", "weather_request_id"]].merge(current_req_features, on="weather_request_id", how="left")
        env_features = env_features.drop_duplicates("env_id", keep="first")
        env_features.to_csv(FEATURE_OUT, sep="\t", index=False, lineterminator="\n")
        qc = pd.DataFrame(
            [
                {"metric": "manifest_ready_rows_covered_by_nasa_date_range", "value": len(ready)},
                {"metric": "manifest_ready_unique_env_id_covered_by_nasa_date_range", "value": ready["env_id"].nunique()},
                {"metric": "unique_weather_requests_total", "value": ready["weather_request_id"].nunique()},
                {"metric": "unique_weather_requests_fetched_current_manifest", "value": current_req_features["weather_request_id"].nunique()},
                {"metric": "unique_weather_requests_cached_total", "value": req_features["weather_request_id"].nunique()},
                {"metric": "env_id_with_weather_features", "value": int(env_features["fetch_status"].eq("ok").sum())},
                {"metric": "failed_requests_logged", "value": len(pd.read_csv(FAIL_OUT, sep="\t")) if FAIL_OUT.exists() else 0},
            ]
        )
        qc.to_csv(QC_OUT, sep="\t", index=False, lineterminator="\n")
        print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
