from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from server_training_pipeline.build_phase6a_projection_core_historical import (
    ENVIRONMENT_MAP,
    SOIL_SITES,
    STATIC_SITES,
    load_soil_lookup,
    site_key,
)
from server_training_pipeline.fetch_cmip6_member_resolved import (
    atomic_json,
    atomic_tsv,
    resolve,
    sha256_file,
)


DEFAULT_PROTOCOL = Path("server_training_pipeline/phase6b_future_covariate_protocol_v1.json")
DEFAULT_PARENT = Path("audit/v2/e_projection_core_v1_release_v2/E_PROJECTION_CORE_V1_RELEASE.json")
DEFAULT_PARENT_MANIFEST = Path(
    "audit/v2/e_projection_core_v1_release_v2/E_PROJECTION_CORE_V1_CLOSING_MANIFEST.tsv"
)
DEFAULT_RAW_MANIFEST = Path(
    "audit/v2/phase6a_projection_core_raw_archive_v1/cmip6_raw_archive_manifest.tsv"
)
DEFAULT_PARAMETERS = Path(
    "audit/v2/phase6a_bias_adjustment_v2/bias_adjustment_parameter_index.tsv"
)
DEFAULT_OUTPUT = Path("audit/v2/e_projection_core_v1_future_covariates_v1_freeze")


def stable_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def month_day_ordinal(value: str) -> int:
    parsed = pd.Timestamp(f"2000-{value}")
    return int(parsed.dayofyear)


def circular_distance(left: int, right: int) -> int:
    difference = abs(left - right)
    return min(difference, 366 - difference)


def select_sowing_medoid(values: pd.Series) -> tuple[str | None, int, float]:
    parsed = pd.to_datetime(values, errors="coerce").dropna()
    if parsed.empty:
        return None, 0, np.nan
    month_days = parsed.dt.strftime("%m-%d").tolist()
    counts = Counter(month_days)
    ordinals = [month_day_ordinal(value) for value in month_days]
    candidates = []
    for candidate, frequency in counts.items():
        ordinal = month_day_ordinal(candidate)
        distances = [circular_distance(ordinal, observed) for observed in ordinals]
        candidates.append((sum(distances), -frequency, candidate, float(np.median(distances))))
    _, _, selected, median_distance = min(candidates)
    return selected, len(month_days), median_distance


def verify_parent_release(root: Path, parent_path: Path, manifest_path: Path) -> dict[str, Any]:
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        parent.get("status") != "PASS_E_PROJECTION_CORE_V1_REMEDIATED_HISTORICAL_TRANSFER_CERTIFIED"
        or parent.get("future_covariate_generation_allowed") is not True
        or parent.get("future_prediction_allowed") is not False
    ):
        raise ValueError("Certified E_PROJECTION_CORE_V1 parent release is not generation-ready")
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    for row in manifest.itertuples(index=False):
        path = root / row.path
        if not path.is_file() or sha256_file(path) != row.sha256:
            raise ValueError(f"Parent closing-manifest artifact changed: {row.path}")
    if sha256_file(manifest_path) != parent["closing_manifest_sha256"]:
        raise ValueError("Parent closing manifest checksum differs from the release decision")
    return parent


def build_location_manifest(root: Path) -> pd.DataFrame:
    sites = pd.read_csv(root / SOIL_SITES, sep="\t", dtype=str)
    environments = pd.read_csv(root / ENVIRONMENT_MAP, sep="\t", dtype=str)
    static = pd.read_parquet(root / STATIC_SITES)[["environment_id", "elevation_m"]]
    environments = environments.merge(static, on="environment_id", how="left", validate="one_to_one")
    environments["coordinate_key"] = [
        site_key(row.latitude, row.longitude) for row in environments.itertuples(index=False)
    ]
    grouped = {key: value for key, value in environments.groupby("coordinate_key", sort=False)}
    soil = load_soil_lookup(root)
    rows = []
    for site in sites.itertuples(index=False):
        key = site_key(site.latitude, site.longitude)
        history = grouped.get(key, pd.DataFrame())
        sowing, support, circular_median = select_sowing_medoid(
            history.sowing_date if len(history) else pd.Series(dtype=str)
        )
        elevations = (
            pd.to_numeric(history.elevation_m, errors="coerce").dropna().to_numpy(dtype=float)
            if len(history)
            else np.asarray([], dtype=float)
        )
        soil_row = soil[site.site_id]
        rows.append(
            {
                "location_id": site.site_id,
                "latitude": float(site.latitude),
                "longitude": float(site.longitude),
                "historical_environment_count": len(history),
                "historical_sowing_support_count": support,
                "prospective_sowing_month_day": sowing or "",
                "sowing_anchor_status": "OBSERVED_SITE_MEDOID"
                if sowing
                else "UNAVAILABLE_NO_SOWING_METADATA",
                "sowing_circular_median_absolute_deviation_days": circular_median,
                "elevation_m": float(np.median(elevations)) if len(elevations) else np.nan,
                "elevation_support_count": len(elevations),
                "elevation_iqr_m": float(np.quantile(elevations, 0.75) - np.quantile(elevations, 0.25))
                if len(elevations)
                else np.nan,
                "elevation_status": "HISTORICAL_LOCATION_MEDIAN"
                if len(elevations)
                else "UNAVAILABLE",
                "soil_status": soil_row["soil_status"],
                "soil_source_class": soil_row["soil_source_class"],
                "soil_feature_eligible": soil_row["soil_feature_eligible"],
                "soil_missing_mask": soil_row["soil_missing_mask"],
                "available_water_capacity_mm": soil_row["available_water_capacity_mm"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("location_id").reset_index(drop=True)
    if len(frame) != 907 or frame.location_id.duplicated().any():
        raise ValueError("Future location axis must contain 907 unique locations")
    return frame


def build_plan(
    raw: pd.DataFrame, parameters: pd.DataFrame, protocol: dict[str, Any]
) -> pd.DataFrame:
    future = raw[raw.experiment_id.isin(protocol["ssp_experiments"])].copy()
    if len(future) != 364 or not future.status.eq("PASS").all():
        raise ValueError("Future raw archive must contain 364 certified assets")
    parameter_lookup = parameters.set_index(["source_id", "member_id"])
    rows = []
    required = set(protocol["required_variables"])
    for (source_id, member_id, experiment_id), group in future.groupby(
        ["source_id", "member_id", "experiment_id"], sort=True
    ):
        if set(group.variable) != required or len(group) != len(required):
            raise ValueError(f"Incomplete future variable group: {source_id}/{experiment_id}")
        if (source_id, member_id) not in parameter_lookup.index:
            raise ValueError(f"Missing historical-only bias parameters: {source_id}/{member_id}")
        calendars = group.calendar.unique()
        if len(calendars) != 1:
            raise ValueError(f"Future variables disagree on calendar: {source_id}/{experiment_id}")
        asset_hashes = dict(zip(group.variable, group.asset_sha256, strict=True))
        asset_paths = dict(zip(group.variable, group.asset_path, strict=True))
        for period, (start_year, end_year) in protocol["future_periods"].items():
            identity = {
                "source_id": source_id,
                "member_id": member_id,
                "experiment_id": experiment_id,
                "SSP": protocol["ssp_experiments"][experiment_id],
                "period": period,
            }
            rows.append(
                {
                    "matrix_id": stable_id(identity),
                    **identity,
                    "start_year": start_year,
                    "end_year": end_year,
                    "calendar": calendars[0],
                    "raw_asset_count": len(group),
                    "raw_asset_paths_json": json.dumps(asset_paths, sort_keys=True),
                    "raw_asset_sha256_json": json.dumps(asset_hashes, sort_keys=True),
                    "bias_parameter_path": parameter_lookup.loc[(source_id, member_id), "parameter_path"],
                    "bias_parameter_sha256": parameter_lookup.loc[
                        (source_id, member_id), "parameter_sha256"
                    ],
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["source_id", "experiment_id", "period"]
    ).reset_index(drop=True)
    if len(frame) != 104 or frame.matrix_id.duplicated().any():
        raise ValueError("Future generation plan must contain 104 unique matrices")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--parent-manifest", type=Path, default=DEFAULT_PARENT_MANIFEST)
    parser.add_argument("--raw-manifest", type=Path, default=DEFAULT_RAW_MANIFEST)
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = resolve(root, args.protocol)
    parent_path = resolve(root, args.parent)
    parent_manifest_path = resolve(root, args.parent_manifest)
    raw_path = resolve(root, args.raw_manifest)
    parameter_path = resolve(root, args.parameters)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "phase6b_member_resolved_future_covariates_v1":
        raise ValueError("Phase 6B future-covariate protocol identity mismatch")
    parent = verify_parent_release(root, parent_path, parent_manifest_path)
    raw = pd.read_csv(raw_path, sep="\t", dtype=str)
    parameters = pd.read_csv(parameter_path, sep="\t", dtype=str)
    locations = build_location_manifest(root)
    plan = build_plan(raw, parameters, protocol)
    output = resolve(root, args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Phase 6B freeze directory already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    locations_path = output / "future_projection_location_manifest.tsv"
    plan_path = output / "future_covariate_generation_plan.tsv"
    atomic_tsv(locations_path, locations)
    atomic_tsv(plan_path, plan)
    lock = {
        "status": "PASS_FUTURE_COVARIATE_GENERATION_LOCKED",
        "protocol_version": protocol["protocol_version"],
        "selection_data": protocol["selection_data"],
        "parent_release_id": parent["release_id"],
        "parent_release_sha256": sha256_file(parent_path),
        "parent_closing_manifest_sha256": sha256_file(parent_manifest_path),
        "protocol_sha256": sha256_file(protocol_path),
        "raw_manifest_sha256": sha256_file(raw_path),
        "bias_parameter_index_sha256": sha256_file(parameter_path),
        "location_manifest_sha256": sha256_file(locations_path),
        "generation_plan_sha256": sha256_file(plan_path),
        "location_count": len(locations),
        "location_with_sowing_anchor_count": int(
            locations.sowing_anchor_status.eq("OBSERVED_SITE_MEDOID").sum()
        ),
        "location_without_sowing_anchor_count": int(
            locations.sowing_anchor_status.ne("OBSERVED_SITE_MEDOID").sum()
        ),
        "matrix_count": len(plan),
        "future_climate_values_read": False,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
        "future_prediction_allowed": False,
    }
    atomic_json(output / "future_covariate_generation_lock.json", lock)
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
