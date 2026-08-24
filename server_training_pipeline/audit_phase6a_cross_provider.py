from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from server_training_pipeline.fetch_cmip6_member_resolved import (
    atomic_json,
    atomic_tsv,
    resolve,
    sha256_file,
)


DEFAULT_PROTOCOL = Path(
    "server_training_pipeline/phase6a_cross_provider_audit_protocol_v1.json"
)
DEFAULT_OUTPUT = Path("audit/v2/phase6a_cross_provider_audit_v1")
CDS_INDEX = Path(
    "environment/v2/phase6a_cds_era5_land_daily_full_v1/cds_era5_land_fetch_index.tsv"
)
OPEN_INDEX = Path(
    "environment/v2/phase6a_openmeteo_era5_daily_full_v1/daily_request_fetch_index.tsv"
)
NORMALIZED_ROOT = Path("environment/v2/e_projection_core_v1_historical_daily")
OPEN_ROOT = Path("environment/v2/phase6a_openmeteo_era5_daily_full_v1")


def load_protocol(root: Path, path: Path) -> dict[str, Any]:
    resolved = resolve(root, path)
    protocol = json.loads(resolved.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "phase6a_cross_provider_audit_v1":
        raise ValueError("Cross-provider audit protocol mismatch")
    for relative, expected in protocol["parent_artifacts"].items():
        if sha256_file(resolve(root, Path(relative))) != expected:
            raise ValueError(f"Frozen cross-provider parent changed: {relative}")
    protocol["_sha256"] = sha256_file(resolved)
    return protocol


def paired_sufficient_statistics(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    x = left[mask]
    y = right[mask]
    difference = y - x
    return {
        "n": len(x),
        "sum_x": float(x.sum()),
        "sum_y": float(y.sum()),
        "sum_x2": float(np.dot(x, x)),
        "sum_y2": float(np.dot(y, y)),
        "sum_xy": float(np.dot(x, y)),
        "sum_difference": float(difference.sum()),
        "sum_squared_difference": float(np.dot(difference, difference)),
        "sum_absolute_difference": float(np.abs(difference).sum()),
        "missing_disagreement_count": int(np.logical_xor(np.isfinite(left), np.isfinite(right)).sum()),
        "row_count": len(left),
    }


def statistics_to_metrics(values: dict[str, float | int]) -> dict[str, float | int]:
    n = int(values["n"])
    if n == 0:
        return {"n": 0, "bias": np.nan, "rmse": np.nan, "mae": np.nan, "pearson": np.nan}
    sum_x = float(values["sum_x"])
    sum_y = float(values["sum_y"])
    variance_x = float(values["sum_x2"]) - sum_x * sum_x / n
    variance_y = float(values["sum_y2"]) - sum_y * sum_y / n
    covariance = float(values["sum_xy"]) - sum_x * sum_y / n
    denominator = math.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0))
    return {
        "n": n,
        "bias": float(values["sum_difference"]) / n,
        "rmse": math.sqrt(float(values["sum_squared_difference"]) / n),
        "mae": float(values["sum_absolute_difference"]) / n,
        "pearson": covariance / denominator if denominator > 0 else np.nan,
        "missing_disagreement_fraction": float(values["missing_disagreement_count"])
        / max(int(values["row_count"]), 1),
    }


def audit_record(root_value: str, record: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(root_value)
    cds_path = (
        root
        / NORMALIZED_ROOT
        / "cds_trial_windows"
        / record["cds_request_id"][:2]
        / f"{record['cds_request_id']}.parquet"
    )
    open_path = root / OPEN_ROOT / record["open_daily_path"]
    cds = pd.read_parquet(cds_path)
    opened = pd.read_parquet(open_path)
    cds["date"] = pd.to_datetime(cds.date)
    opened["date"] = pd.to_datetime(opened.date)
    merged = cds.merge(opened, on="date", how="outer", validate="one_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError(f"Provider dates disagree for {record['cds_request_id']}")
    rows = []
    for canonical, diagnostic in protocol["variables"].items():
        stats = paired_sufficient_statistics(merged[canonical], merged[diagnostic])
        metrics = statistics_to_metrics(stats)
        row = {
            "cds_request_id": record["cds_request_id"],
            "openmeteo_request_id": record["openmeteo_request_id"],
            "variable": canonical,
            **stats,
            **{f"metric_{key}": value for key, value in metrics.items()},
        }
        if canonical == "precipitation_mm_day":
            threshold = float(protocol["wet_day_threshold_mm"])
            left_wet = merged[canonical].to_numpy(dtype=float) >= threshold
            right_wet = merged[diagnostic].to_numpy(dtype=float) >= threshold
            row["wet_day_agreement_count"] = int((left_wet == right_wet).sum())
            row["wet_day_row_count"] = len(merged)
        else:
            row["wet_day_agreement_count"] = 0
            row["wet_day_row_count"] = 0
        rows.append(row)
    return rows


def audit_partition(root_value: str, records: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.extend(audit_record(root_value, record, protocol))
    return rows


def build_pairs(root: Path) -> pd.DataFrame:
    cds = pd.read_csv(root / CDS_INDEX, sep="\t", dtype=str)
    opened = pd.read_csv(root / OPEN_INDEX, sep="\t", dtype=str)
    for frame in (cds, opened):
        frame["latitude_5dp"] = pd.to_numeric(frame.latitude).round(5)
        frame["longitude_5dp"] = pd.to_numeric(frame.longitude).round(5)
    keys = ["latitude_5dp", "longitude_5dp", "request_start_date", "request_end_date"]
    cds_view = cds.rename(columns={"request_id": "cds_request_id"})
    open_view = opened.rename(
        columns={"request_id": "openmeteo_request_id", "daily_path": "open_daily_path"}
    )
    pairs = cds_view[keys + ["cds_request_id"]].merge(
        open_view[keys + ["openmeteo_request_id", "open_daily_path"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(pairs) != 7094 or not pairs._merge.eq("both").all():
        raise ValueError("CDS and Open-Meteo request identities are not one-to-one")
    return pairs.drop(columns="_merge").sort_values("cds_request_id").reset_index(drop=True)


def run(root: Path, protocol_path: Path, output: Path, workers: int) -> dict[str, Any]:
    protocol = load_protocol(root, protocol_path)
    if workers < 1 or workers > 8:
        raise ValueError("Cross-provider workers must be between 1 and 8")
    pairs = build_pairs(root)
    records = pairs.to_dict("records")
    partitions = [records[index::workers] for index in range(workers)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(audit_partition, str(root), partition, protocol): len(partition)
            for partition in partitions
            if partition
        }
        completed = 0
        for future in as_completed(futures):
            part = future.result()
            rows.extend(part)
            completed += len(part) // len(protocol["variables"])
            print(f"Cross-provider requests {completed}/{len(records)}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["variable", "cds_request_id"]).reset_index(drop=True)
    summary_rows = []
    for variable, group in frame.groupby("variable"):
        totals = {
            key: group[key].sum()
            for key in (
                "n",
                "sum_x",
                "sum_y",
                "sum_x2",
                "sum_y2",
                "sum_xy",
                "sum_difference",
                "sum_squared_difference",
                "sum_absolute_difference",
                "missing_disagreement_count",
                "row_count",
            )
        }
        metrics = statistics_to_metrics(totals)
        wet_n = int(group.wet_day_row_count.sum())
        wet_agreement = (
            float(group.wet_day_agreement_count.sum()) / wet_n if wet_n else np.nan
        )
        thresholds = protocol["diagnostic_thresholds"][variable]
        checks = {
            "correlation": float(metrics["pearson"]) >= float(thresholds["minimum_correlation"]),
            "absolute_bias": abs(float(metrics["bias"])) <= float(thresholds["maximum_absolute_bias"]),
            "rmse": float(metrics["rmse"]) <= float(thresholds["maximum_rmse"]),
        }
        if "minimum_wet_day_agreement" in thresholds:
            checks["wet_day_agreement"] = wet_agreement >= float(
                thresholds["minimum_wet_day_agreement"]
            )
        summary_rows.append(
            {
                "variable": variable,
                **metrics,
                "wet_day_agreement": wet_agreement,
                "diagnostic_failed_checks": ";".join(
                    key for key, value in checks.items() if not value
                ),
                "diagnostic_status": "PASS" if all(checks.values()) else "FLAG_REVIEW",
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("variable").reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    atomic_tsv(output / "cross_provider_request_metrics.tsv", frame)
    atomic_tsv(output / "cross_provider_summary.tsv", summary)
    result = {
        "status": "PASS_WITH_DIAGNOSTIC_FLAGS"
        if summary.diagnostic_status.ne("PASS").any()
        else "PASS",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["_sha256"],
        "matched_request_count": len(pairs),
        "metric_row_count": len(frame),
        "flagged_variable_count": int(summary.diagnostic_status.ne("PASS").sum()),
        "authoritative_provider_unchanged": True,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
        "artifacts": {
            "cross_provider_request_metrics.tsv": sha256_file(
                output / "cross_provider_request_metrics.tsv"
            ),
            "cross_provider_summary.tsv": sha256_file(output / "cross_provider_summary.tsv"),
        },
    }
    atomic_json(output / "cross_provider_provenance.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run(args.root.resolve(), args.protocol, resolve(args.root.resolve(), args.output), args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
