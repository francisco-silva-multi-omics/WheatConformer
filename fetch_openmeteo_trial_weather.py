from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
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
FEATURE_OUT = OUT / "trial_weather_features_openmeteo.tsv"
REQUEST_OUT = OUT / "trial_weather_request_features_openmeteo.tsv"
QC_OUT = OUT / "trial_weather_fetch_openmeteo_qc.tsv"
FAIL_OUT = OUT / "trial_weather_fetch_openmeteo_failures.tsv"

HOURLY_VARS = [
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


def request_url(row: pd.Series) -> str:
    params = {
        "latitude": f"{row.latitude:.5f}",
        "longitude": f"{row.longitude:.5f}",
        "start_date": row.weather_start_date,
        "end_date": row.weather_end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "auto",
        "models": "era5",
    }
    return "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)


def request_key(row: pd.Series) -> str:
    return "|".join(
        [
            f"{row.latitude:.5f}",
            f"{row.longitude:.5f}",
            str(row.weather_start_date),
            str(row.weather_end_date),
        ]
    )


def safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if values.notna().any() else np.nan


def safe_sum(values: pd.Series) -> float:
    return float(values.sum()) if values.notna().any() else np.nan


def aggregate_hourly(payload: dict, req: pd.Series) -> dict[str, object]:
    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError("No hourly data returned")

    df = pd.DataFrame(hourly)
    df["date"] = pd.to_datetime(df["time"], errors="coerce").dt.date
    numeric_cols = [c for c in df.columns if c not in {"time", "date"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    daily = df.groupby("date", dropna=True).agg(
        temp_mean=("temperature_2m", "mean"),
        temp_max=("temperature_2m", "max"),
        temp_min=("temperature_2m", "min"),
        rh_mean=("relative_humidity_2m", "mean"),
        dewpoint_mean=("dew_point_2m", "mean"),
        precip_sum=("precipitation", "sum"),
        radiation_sum=("shortwave_radiation", "sum"),
        et0_sum=("et0_fao_evapotranspiration", "sum"),
        vpd_mean=("vapour_pressure_deficit", "mean"),
        vpd_max=("vapour_pressure_deficit", "max"),
        soil_moisture_0_7_mean=("soil_moisture_0_to_7cm", "mean"),
        soil_moisture_7_28_mean=("soil_moisture_7_to_28cm", "mean"),
    )

    n_days = int(len(daily))
    out: dict[str, object] = {
        "weather_request_id": req.weather_request_id,
        "latitude": req.latitude,
        "longitude": req.longitude,
        "weather_start_date": req.weather_start_date,
        "weather_end_date": req.weather_end_date,
        "n_days_weather": n_days,
        "openmeteo_timezone": payload.get("timezone", ""),
        "openmeteo_elevation": payload.get("elevation", np.nan),
        "temperature_mean_c": safe_mean(daily["temp_mean"]),
        "temperature_max_c": float(daily["temp_max"].max()) if n_days else np.nan,
        "temperature_min_c": float(daily["temp_min"].min()) if n_days else np.nan,
        "relative_humidity_mean_pct": safe_mean(daily["rh_mean"]),
        "dewpoint_mean_c": safe_mean(daily["dewpoint_mean"]),
        "precipitation_total_mm": safe_sum(daily["precip_sum"]),
        "precipitation_mean_daily_mm": safe_mean(daily["precip_sum"]),
        "shortwave_radiation_total_wm2h": safe_sum(daily["radiation_sum"]),
        "shortwave_radiation_mean_daily_wm2h": safe_mean(daily["radiation_sum"]),
        "et0_total_mm": safe_sum(daily["et0_sum"]),
        "et0_mean_daily_mm": safe_mean(daily["et0_sum"]),
        "vpd_mean_kpa": safe_mean(daily["vpd_mean"]),
        "vpd_max_kpa": float(daily["vpd_max"].max()) if n_days else np.nan,
        "soil_moisture_0_7_mean_m3m3": safe_mean(daily["soil_moisture_0_7_mean"]),
        "soil_moisture_7_28_mean_m3m3": safe_mean(daily["soil_moisture_7_28_mean"]),
    }

    out["heat_days_tmax_ge_30"] = int((daily["temp_max"] >= 30).sum()) if n_days else 0
    out["heat_days_tmax_ge_35"] = int((daily["temp_max"] >= 35).sum()) if n_days else 0
    out["extreme_heat_days_tmax_ge_40"] = int((daily["temp_max"] >= 40).sum()) if n_days else 0
    out["dry_days_precip_lt_1mm"] = int((daily["precip_sum"] < 1).sum()) if n_days else 0
    out["heavy_rain_days_precip_ge_20mm"] = int((daily["precip_sum"] >= 20).sum()) if n_days else 0
    out["water_balance_et0_minus_precip_mm"] = (
        out["et0_total_mm"] - out["precipitation_total_mm"]
        if pd.notna(out["et0_total_mm"]) and pd.notna(out["precipitation_total_mm"])
        else np.nan
    )
    out["drought_days_precip_lt_1mm_and_vpd_gt_1_5"] = (
        int(((daily["precip_sum"] < 1) & (daily["vpd_mean"] > 1.5)).sum()) if n_days else 0
    )
    out["high_vpd_days_gt_1_5"] = int((daily["vpd_mean"] > 1.5).sum()) if n_days else 0
    out["high_vpd_days_gt_2_0"] = int((daily["vpd_mean"] > 2.0).sum()) if n_days else 0
    return out


def fetch_json(url: str, timeout: int = 90, retries: int = 4, sleep_seconds: float = 2.0) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "trial-weather-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read()
            return json.loads(data.decode("utf-8"))
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


def append_tsv(row: dict[str, object], path: Path) -> None:
    df = pd.DataFrame([row])
    df.to_csv(path, sep="\t", index=False, mode="a", header=not path.exists(), lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Fetch only the first N unfetched unique requests.")
    parser.add_argument("--sleep", type=float, default=0.15, help="Seconds to sleep between successful requests.")
    parser.add_argument("--resume", action="store_true", help="Skip request IDs already present in request feature output.")
    parser.add_argument("--clear-failures", action="store_true", help="Clear the prior failure log before fetching.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel fetch workers.")
    args = parser.parse_args()

    if args.clear_failures and FAIL_OUT.exists():
        FAIL_OUT.unlink()

    manifest = pd.read_csv(MANIFEST, sep="\t", dtype=str)
    ready = manifest[manifest["ready_to_fetch"].astype(str).str.upper().eq("TRUE")].copy()
    for col in ["latitude", "longitude"]:
        ready[col] = pd.to_numeric(ready[col], errors="coerce")
    ready = ready[ready["latitude"].notna() & ready["longitude"].notna()].copy()
    ready["weather_request_id"] = ready.apply(request_key, axis=1)

    requests = (
        ready[["weather_request_id", "latitude", "longitude", "weather_start_date", "weather_end_date"]]
        .drop_duplicates("weather_request_id")
        .sort_values("weather_request_id")
        .reset_index(drop=True)
    )

    done_ids: set[str] = set()
    if args.resume and REQUEST_OUT.exists():
        done_ids = set(pd.read_csv(REQUEST_OUT, sep="\t", dtype=str, usecols=["weather_request_id"])["weather_request_id"])
    requests = requests[~requests["weather_request_id"].isin(done_ids)].copy()
    if args.limit is not None:
        requests = requests.head(args.limit).copy()

    def fetch_one(req: pd.Series) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        url = request_url(req)
        try:
            payload = fetch_json(url)
            features = aggregate_hourly(payload, req)
            features["fetch_status"] = "ok"
            return features, None
        except Exception as exc:
            fail = {
                "weather_request_id": req.weather_request_id,
                "latitude": req.latitude,
                "longitude": req.longitude,
                "weather_start_date": req.weather_start_date,
                "weather_end_date": req.weather_end_date,
                "error": str(exc),
                "url": url,
            }
            return None, fail

    fetched = 0
    failed = 0
    request_rows = [row for _, row in requests.iterrows()]
    if args.workers <= 1:
        iterator = (fetch_one(req) for req in request_rows)
        for features, fail in iterator:
            if features is not None:
                append_tsv(features, REQUEST_OUT)
                fetched += 1
                if fetched % 50 == 0:
                    print(f"Fetched {fetched} requests", flush=True)
                time.sleep(args.sleep)
            if fail is not None:
                failed += 1
                append_tsv(fail, FAIL_OUT)
                print(f"Failed {fail['weather_request_id']}: {fail['error']}", flush=True)
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
                    failed += 1
                    append_tsv(fail, FAIL_OUT)
                    print(f"Failed {fail['weather_request_id']}: {fail['error']}", flush=True)
                if args.sleep:
                    time.sleep(args.sleep)

    if REQUEST_OUT.exists():
        req_features = pd.read_csv(REQUEST_OUT, sep="\t", dtype=str, low_memory=False)
        req_features = req_features.drop_duplicates("weather_request_id", keep="last")
        env_features = ready[["env_id", "weather_request_id"]].merge(req_features, on="weather_request_id", how="left")
        env_features = env_features.drop_duplicates("env_id", keep="first")
        env_features.to_csv(FEATURE_OUT, sep="\t", index=False, lineterminator="\n")

        qc = pd.DataFrame(
            [
                {"metric": "manifest_ready_rows", "value": len(ready)},
                {"metric": "manifest_ready_unique_env_id", "value": ready["env_id"].nunique()},
                {"metric": "unique_weather_requests_total", "value": ready["weather_request_id"].nunique()},
                {"metric": "unique_weather_requests_fetched", "value": req_features["weather_request_id"].nunique()},
                {"metric": "env_id_with_weather_features", "value": env_features["fetch_status"].eq("ok").sum()},
                {"metric": "failed_requests_logged", "value": len(pd.read_csv(FAIL_OUT, sep="\t")) if FAIL_OUT.exists() else 0},
            ]
        )
        qc.to_csv(QC_OUT, sep="\t", index=False, lineterminator="\n")
        print(qc.to_string(index=False))
    else:
        print("No request features were written.")


if __name__ == "__main__":
    main()
