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
ENV = BASE / "environment"
NASA_START = pd.Timestamp("1981-01-01")
PARAMETERS = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "WS2M"]
WINDOWS = [(0, 30), (30, 60), (60, 90), (0, 90), (0, 120)]


def request_url(row: pd.Series) -> str:
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": f"{row.longitude:.5f}",
        "latitude": f"{row.latitude:.5f}",
        "start": str(row.window_start_date).replace("-", ""),
        "end": str(row.window_end_date).replace("-", ""),
        "format": "JSON",
    }
    return "https://power.larc.nasa.gov/api/temporal/daily/point?" + urllib.parse.urlencode(params)


def fetch_json(url: str, timeout: int = 90, retries: int = 4, sleep_seconds: float = 2.0) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dth-weather-window-fetch/1.0"})
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
    return s.replace(-999, np.nan).sort_index()


def vpd_kpa(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    return es * (1 - rh_pct / 100.0)


def safe_mean(s: pd.Series) -> float:
    return float(s.mean()) if s.notna().any() else np.nan


def safe_sum(s: pd.Series) -> float:
    return float(s.sum()) if s.notna().any() else np.nan


def aggregate_payload(payload: dict, req: pd.Series) -> dict[str, object]:
    params = payload.get("properties", {}).get("parameter", {})
    daily = pd.DataFrame(
        {
            "tmean": to_series(params, "T2M"),
            "tmax": to_series(params, "T2M_MAX"),
            "tmin": to_series(params, "T2M_MIN"),
            "rh": to_series(params, "RH2M"),
            "precip": to_series(params, "PRECTOTCORR"),
            "radiation": to_series(params, "ALLSKY_SFC_SW_DWN"),
            "wind": to_series(params, "WS2M"),
        }
    )
    daily["vpd"] = vpd_kpa(daily["tmean"], daily["rh"])
    daily["gdd_base0"] = daily["tmean"].clip(lower=0)
    daily["gdd_base5"] = (daily["tmean"] - 5).clip(lower=0)
    n_days = int(len(daily))
    out: dict[str, object] = {
        "env_id": req.env_id,
        "window_label": req.window_label,
        "window_start_date": req.window_start_date,
        "window_end_date": req.window_end_date,
        "weather_request_id": req.weather_request_id,
        "fetch_status": "ok",
        "n_days": n_days,
        "temperature_mean_c": safe_mean(daily["tmean"]),
        "temperature_max_c": float(daily["tmax"].max()) if n_days else np.nan,
        "temperature_min_c": float(daily["tmin"].min()) if n_days else np.nan,
        "gdd_base0_sum": safe_sum(daily["gdd_base0"]),
        "gdd_base5_sum": safe_sum(daily["gdd_base5"]),
        "cold_days_tmin_lt_0": int((daily["tmin"] < 0).sum()) if n_days else 0,
        "chill_days_tmean_0_10": int(((daily["tmean"] >= 0) & (daily["tmean"] <= 10)).sum()) if n_days else 0,
        "heat_days_tmax_ge_30": int((daily["tmax"] >= 30).sum()) if n_days else 0,
        "heat_days_tmax_ge_35": int((daily["tmax"] >= 35).sum()) if n_days else 0,
        "precipitation_total_mm": safe_sum(daily["precip"]),
        "dry_days_precip_lt_1mm": int((daily["precip"] < 1).sum()) if n_days else 0,
        "solar_radiation_total_mj_m2": safe_sum(daily["radiation"]),
        "solar_radiation_mean_daily_mj_m2": safe_mean(daily["radiation"]),
        "relative_humidity_mean_pct": safe_mean(daily["rh"]),
        "wind_speed_2m_mean_m_s": safe_mean(daily["wind"]),
        "vpd_mean_kpa": safe_mean(daily["vpd"]),
        "vpd_max_kpa": float(daily["vpd"].max()) if n_days else np.nan,
        "high_vpd_days_gt_1_5": int((daily["vpd"] > 1.5).sum()) if n_days else 0,
        "drought_days_precip_lt_1mm_and_vpd_gt_1_5": int(((daily["precip"] < 1) & (daily["vpd"] > 1.5)).sum()) if n_days else 0,
    }
    return out


def write_row(row: dict[str, object], path: Path) -> None:
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False, mode="a", header=not path.exists(), lineterminator="\n")


def build_window_manifest(fetch_manifest: pd.DataFrame, env_filter: set[str] | None = None) -> pd.DataFrame:
    ready = fetch_manifest[fetch_manifest["ready_to_fetch"].astype(str).str.upper().eq("TRUE")].copy()
    if env_filter is not None:
        ready = ready[ready["env_id"].astype(str).isin(env_filter)].copy()
    ready["sowing_date"] = pd.to_datetime(ready["sowing_date"], errors="coerce")
    ready = ready[ready["sowing_date"].notna()].copy()
    for col in ["latitude", "longitude"]:
        ready[col] = pd.to_numeric(ready[col], errors="coerce")
    ready = ready[ready["latitude"].notna() & ready["longitude"].notna()].copy()

    rows = []
    for _, row in ready.iterrows():
        for start, end in WINDOWS:
            ws = row["sowing_date"] + pd.Timedelta(days=start)
            we = row["sowing_date"] + pd.Timedelta(days=end - 1)
            if we < NASA_START:
                continue
            if ws < NASA_START:
                ws = NASA_START
            label = f"d{start}_{end}"
            rows.append(
                {
                    "env_id": row["env_id"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "sowing_date": row["sowing_date"].strftime("%Y-%m-%d"),
                    "window_label": label,
                    "window_start_date": ws.strftime("%Y-%m-%d"),
                    "window_end_date": we.strftime("%Y-%m-%d"),
                }
            )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return manifest
    manifest["weather_request_id"] = manifest.apply(
        lambda r: "|".join(
            [
                f"{float(r.latitude):.5f}",
                f"{float(r.longitude):.5f}",
                str(r.window_start_date),
                str(r.window_end_date),
            ]
        ),
        axis=1,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch DTH-specific fixed-window NASA POWER weather features.")
    parser.add_argument("--model-env-order", type=Path, default=None, help="Optional K_E order file limiting env_id values.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--out-prefix", default="dth_api_weather_windows")
    args = parser.parse_args()

    env_filter = None
    if args.model_env_order is not None and args.model_env_order.exists():
        order = pd.read_csv(args.model_env_order, sep="\t", dtype=str)
        id_col = "env_id" if "env_id" in order.columns else order.columns[0]
        env_filter = set(order[id_col].dropna().astype(str))

    fetch_manifest = pd.read_csv(ENV / "trial_weather_fetch_manifest.tsv", sep="\t", dtype=str)
    manifest = build_window_manifest(fetch_manifest, env_filter)
    manifest_path = ENV / f"{args.out_prefix}_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False, lineterminator="\n")

    request_path = ENV / f"{args.out_prefix}_request_features.tsv"
    feature_path = ENV / f"{args.out_prefix}.tsv"
    fail_path = ENV / f"{args.out_prefix}_failures.tsv"
    qc_path = ENV / f"{args.out_prefix}_qc.tsv"

    requests = manifest.drop_duplicates("weather_request_id").sort_values("weather_request_id").reset_index(drop=True)
    if args.resume and request_path.exists():
        done = set(pd.read_csv(request_path, sep="\t", dtype=str, usecols=["weather_request_id"])["weather_request_id"])
        requests = requests[~requests["weather_request_id"].isin(done)].copy()
    if args.limit is not None:
        requests = requests.head(args.limit).copy()

    def fetch_one(req: pd.Series) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        try:
            return aggregate_payload(fetch_json(request_url(req)), req), None
        except Exception as exc:
            return None, {
                "weather_request_id": req.weather_request_id,
                "env_id": req.env_id,
                "window_label": req.window_label,
                "error": str(exc),
                "url": request_url(req),
            }

    req_rows = [row for _, row in requests.iterrows()]
    if args.workers <= 1:
        for req in req_rows:
            result, fail = fetch_one(req)
            if result is not None:
                write_row(result, request_path)
            if fail is not None:
                write_row(fail, fail_path)
            if args.sleep:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_one, req) for req in req_rows]
            for future in as_completed(futures):
                result, fail = future.result()
                if result is not None:
                    write_row(result, request_path)
                if fail is not None:
                    write_row(fail, fail_path)
                if args.sleep:
                    time.sleep(args.sleep)

    if request_path.exists():
        req_features = pd.read_csv(request_path, sep="\t", dtype=str, low_memory=False)
        req_features = req_features.drop_duplicates("weather_request_id", keep="last")
        env_features = manifest[["env_id", "window_label", "weather_request_id"]].merge(
            req_features.drop(columns=["env_id", "window_label"], errors="ignore"), on="weather_request_id", how="left"
        )
        env_features.to_csv(feature_path, sep="\t", index=False, lineterminator="\n")
    qc = pd.DataFrame(
        [
            {"metric": "window_manifest_rows", "value": len(manifest)},
            {"metric": "window_manifest_env_id", "value": manifest["env_id"].nunique() if not manifest.empty else 0},
            {"metric": "requests_remaining_this_run", "value": len(requests)},
            {"metric": "cached_request_rows", "value": len(pd.read_csv(request_path, sep="\t")) if request_path.exists() else 0},
            {"metric": "failure_rows", "value": len(pd.read_csv(fail_path, sep="\t")) if fail_path.exists() else 0},
        ]
    )
    qc.to_csv(qc_path, sep="\t", index=False, lineterminator="\n")
    print(qc.to_string(index=False))
    print(f"Wrote {manifest_path}")
    print(f"Wrote {feature_path}")


if __name__ == "__main__":
    main()
