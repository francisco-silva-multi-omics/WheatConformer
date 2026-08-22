from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from server_training_pipeline.fetch_cmip6_member_resolved import atomic_json, atomic_tsv, resolve, sha256_file


DEFAULT_PROTOCOL = Path("server_training_pipeline/phase6a_historical_transfer_contract_v2.json")
DEFAULT_ERA = Path(
    "environment/v2/e_projection_core_v1_historical_backcast/era5_land_historical_projection_core_features.parquet"
)
DEFAULT_CMIP_INDEX = Path("audit/v2/phase6a_projection_core_historical_backcast_v2/cmip6_historical_backcast_index.tsv")
DEFAULT_ROBUST = Path(
    "environment/v2/e_projection_core_v1_applicability_domain_reference/historical_robust_feature_reference.tsv"
)
DEFAULT_OUTPUT = Path("audit/v2/e_projection_core_v1_release_v2")
SOURCE_COLUMNS = {"source_id", "member_id", "calendar", "bias_parameter_complete_fraction"}


def logical_dtype(series: pd.Series, column: str) -> str:
    if column.endswith("_count"):
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if len(finite) and not np.allclose(finite, np.round(finite), atol=0.0, rtol=0.0):
            return "invalid_nonintegral_count"
        return "nullable_integer_count"
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    return "string"


def logical_schema(frame: pd.DataFrame) -> dict[str, str]:
    return {
        column: logical_dtype(frame[column], column)
        for column in frame.columns
        if column not in SOURCE_COLUMNS
    }


def model_metrics(
    era: pd.DataFrame,
    candidate: pd.DataFrame,
    robust: pd.DataFrame,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    common = era.merge(candidate, on="environment_id", suffixes=("__era", "__cmip"), validate="one_to_one")
    features = [
        feature
        for feature in robust.feature
        if f"{feature}__era" in common and f"{feature}__cmip" in common
    ]
    scale = robust.set_index("feature").loc[features, "robust_scale"].astype(float)
    rows = []
    for feature in features:
        left = pd.to_numeric(common[f"{feature}__era"], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(common[f"{feature}__cmip"], errors="coerce").to_numpy(dtype=float)
        left_finite = left[np.isfinite(left)]
        right_finite = right[np.isfinite(right)]
        shift = (
            abs(float(np.median(right_finite)) - float(np.median(left_finite)))
            / float(scale.loc[feature])
            if len(left_finite) and len(right_finite)
            else np.nan
        )
        rows.append(
            {
                "feature": feature,
                "feature_block": robust.set_index("feature").loc[feature, "feature_block"],
                "common_environment_count": len(common),
                "ERA_missing_fraction": float((~np.isfinite(left)).mean()),
                "CMIP6_missing_fraction": float((~np.isfinite(right)).mean()),
                "missing_fraction_increase": float(
                    (~np.isfinite(right)).mean() - (~np.isfinite(left)).mean()
                ),
                "absolute_robust_location_shift": shift,
            }
        )
    details = pd.DataFrame(rows)
    shifts = details.absolute_robust_location_shift.dropna().to_numpy(dtype=float)
    missing_increase = details.missing_fraction_increase.to_numpy(dtype=float)
    common_fraction = len(common) / len(era)
    eligible_column = "projection_core_climate_eligible__cmip"
    eligible_fraction = float(common[eligible_column].astype(bool).mean())
    summary = {
        "common_environment_count": len(common),
        "common_environment_fraction": common_fraction,
        "climate_eligible_fraction": eligible_fraction,
        "feature_count": len(features),
        "median_absolute_robust_location_shift": float(np.median(shifts)),
        "p95_absolute_robust_location_shift": float(np.quantile(shifts, 0.95)),
        "median_missing_fraction_increase": float(np.median(np.abs(missing_increase))),
    }
    checks = {
        "common_environment_fraction": common_fraction
        >= float(protocol["minimum_common_environment_fraction"]),
        "climate_eligible_fraction": eligible_fraction
        >= float(protocol["minimum_climate_eligible_fraction"]),
        "median_location_shift": summary["median_absolute_robust_location_shift"]
        <= float(protocol["maximum_median_absolute_robust_location_shift"]),
        "p95_location_shift": summary["p95_absolute_robust_location_shift"]
        <= float(protocol["maximum_95th_percentile_absolute_robust_location_shift"]),
        "median_missing_increase": summary["median_missing_fraction_increase"]
        <= float(protocol["maximum_median_absolute_missing_fraction_increase"]),
    }
    summary["failed_checks"] = ";".join(key for key, value in checks.items() if not value)
    summary["status"] = "PASS" if all(checks.values()) else "FAIL"
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--era", type=Path, default=DEFAULT_ERA)
    parser.add_argument("--cmip-index", type=Path, default=DEFAULT_CMIP_INDEX)
    parser.add_argument("--robust", type=Path, default=DEFAULT_ROBUST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = resolve(root, args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "phase6a_historical_transfer_v2":
        raise ValueError("Historical transfer protocol mismatch")
    for relative, expected in protocol["parent_artifacts"].items():
        if sha256_file(resolve(root, Path(relative))) != expected:
            raise ValueError(f"Frozen historical-transfer parent changed: {relative}")
    era_path = resolve(root, args.era)
    index_path = resolve(root, args.cmip_index)
    if not index_path.is_file():
        raise ValueError("Bias-adjusted historical CMIP6 backcast is incomplete")
    era = pd.read_parquet(era_path)
    era = era[pd.to_datetime(era.sowing_date).dt.year.between(1981, 2010)].copy()
    robust = pd.read_csv(resolve(root, args.robust), sep="\t", dtype=str)
    index = pd.read_csv(index_path, sep="\t", dtype=str)
    if len(index) != int(protocol["required_source_count"]):
        raise ValueError("Historical transfer source grid is incomplete")
    era_schema = logical_schema(era)
    era_storage_schema = {
        column: str(era[column].dtype) for column in era_schema
    }
    summary_rows = []
    detail_rows = []
    schema_rows = []
    for row in index.itertuples(index=False):
        candidate = pd.read_parquet(root / row.output_path)
        candidate_schema = logical_schema(candidate)
        candidate_storage_schema = {
            column: str(candidate[column].dtype) for column in candidate_schema
        }
        schema_exact = candidate_schema == era_schema
        storage_dtype_differences = sorted(
            column
            for column in set(era_storage_schema) & set(candidate_storage_schema)
            if era_storage_schema[column] != candidate_storage_schema[column]
        )
        summary, details = model_metrics(era, candidate, robust, protocol)
        summary.update(
            {
                "source_id": row.source_id,
                "member_id": row.member_id,
                "schema_exact": schema_exact,
            }
        )
        if not schema_exact:
            summary["status"] = "FAIL"
            summary["failed_checks"] = ";".join(
                filter(None, [summary["failed_checks"], "schema_exact"])
            )
        details.insert(0, "source_id", row.source_id)
        details.insert(1, "member_id", row.member_id)
        detail_rows.append(details)
        summary_rows.append(summary)
        schema_rows.append(
            {
                "source_id": row.source_id,
                "member_id": row.member_id,
                "ERA_feature_count": len(era_schema),
                "CMIP6_feature_count": len(candidate_schema),
                "schema_exact": schema_exact,
                "ERA_only_features": ";".join(sorted(set(era_schema) - set(candidate_schema))),
                "CMIP6_only_features": ";".join(sorted(set(candidate_schema) - set(era_schema))),
                "logical_dtype_mismatches": ";".join(
                    sorted(
                        column
                        for column in set(era_schema) & set(candidate_schema)
                        if era_schema[column] != candidate_schema[column]
                    )
                ),
                "physical_storage_dtype_differences": ";".join(
                    storage_dtype_differences
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("source_id")
    details = pd.concat(detail_rows, ignore_index=True)
    schemas = pd.DataFrame(schema_rows).sort_values("source_id")
    output = resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_tsv(output / "historical_transfer_summary.tsv", summary)
    atomic_tsv(output / "historical_transfer_feature_metrics.tsv", details)
    atomic_tsv(output / "feature_parity_matrix.tsv", schemas)
    feature_parity = {
        "status": "PASS" if schemas.schema_exact.all() else "FAIL",
        "protocol_version": protocol["protocol_version"],
        "source_count": len(schemas),
        "all_schemas_exact": bool(schemas.schema_exact.all()),
        "feature_parity_matrix_sha256": sha256_file(output / "feature_parity_matrix.tsv"),
        "future_SSP_values_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    transfer = {
        "status": "PASS" if summary.status.eq("PASS").all() else "FAIL",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(protocol_path),
        "source_count": len(summary),
        "all_sources_pass": bool(summary.status.eq("PASS").all()),
        "summary_sha256": sha256_file(output / "historical_transfer_summary.tsv"),
        "feature_metrics_sha256": sha256_file(output / "historical_transfer_feature_metrics.tsv"),
        "phenotype_values_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_SSP_values_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(output / "feature_parity_certification.json", feature_parity)
    atomic_json(output / "historical_transfer_certification.json", transfer)
    print(json.dumps(transfer, indent=2, sort_keys=True))
    if feature_parity["status"] != "PASS" or transfer["status"] != "PASS":
        raise SystemExit("Historical transfer certification failed")


if __name__ == "__main__":
    main()
