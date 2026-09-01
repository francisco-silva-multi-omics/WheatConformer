from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from server_training_pipeline.phase6a_environment_source_recovery import (
    DAILY_VARIABLES,
    OPEN_METEO_MODEL,
    build_cds_request_inventory,
    request_identity,
    stable_json_sha256,
)


PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_phenology_readiness_protocol_v1.json"
)
PLAN = Path(
    "server_training_pipeline/stage1_v2_phase6_post_hierarchy_screen_plan_v3.json"
)
FEATURE_CONTRACT = Path(
    "server_training_pipeline/phase6a_projection_core_feature_contract_v1.json"
)
FA_SUMMARY = Path(
    "model_kernels/stage1_v2_phase6_factor_analytic_optimization_amendment_v2/phase_1"
)
WEATHER_MANIFEST = Path("environment/trial_weather_fetch_manifest.tsv")
ENVIRONMENT_MAP = Path(
    "audit/v2/phase6a_environment_source_contract_v10/environment_daily_request_map.tsv"
)
PROJECTION_RELEASE = Path(
    "audit/v2/e_projection_core_v1_release_v2/E_PROJECTION_CORE_V1_RELEASE.json"
)
SPLIT_RELEASE = Path(
    "audit/v2/e_projection_core_v1_split_bound_historical_v1_release/"
    "SPLIT_BOUND_PROJECTION_INPUT_RELEASE_DECISION.json"
)
CERTIFIED_ENVIRONMENT_AXIS = Path(
    "environment/v2/e_projection_core_v1_split_bound_historical_v1/states/"
    "GNEW_EOBS__OUTER1__INNER1/environment_entities.tsv"
)
BIAS_REFERENCE_DAILY = Path(
    "environment/v2/e_projection_core_v1_historical_daily/cds_bias_reference"
)
CROSS_PROVIDER_PROVENANCE = Path(
    "audit/v2/phase6a_cross_provider_audit_v1/cross_provider_provenance.json"
)
FA_RELEASE = Path("audit/v2/stage1_v2_phase6_fa_terminal_no_advance_v1")
OUTPUT = Path("audit/v2/stage1_v2_phase6_phenology_readiness_v1")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def harvest_horizon_audit(
    weather: pd.DataFrame, policy: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, int | None]:
    def unique_dates(values: pd.Series) -> list[pd.Timestamp]:
        parsed = pd.to_datetime(values, errors="coerce").dropna().drop_duplicates()
        return sorted(parsed.tolist())

    rows: list[dict[str, Any]] = []
    for environment_id, group in weather.groupby("env_id", sort=True, dropna=False):
        sowing_dates = unique_dates(group["sowing_date"])
        finish_dates = unique_dates(group["harvest_finish_date"])
        start_dates = unique_dates(group["harvest_start_date"])
        source = "missing"
        sowing = sowing_dates[0] if len(sowing_dates) == 1 else pd.NaT
        anchor = pd.NaT
        if len(sowing_dates) == 0:
            status = "MISSING_SOWING"
        elif len(sowing_dates) > 1:
            status = "CONFLICTING_SOWING"
        elif len(finish_dates) > 1:
            status = "CONFLICTING_HARVEST_FINISH"
        elif len(finish_dates) == 1:
            anchor = finish_dates[0]
            source = "harvest_finish_date"
            status = "PENDING_RANGE_CHECK"
        elif len(start_dates) > 1:
            status = "CONFLICTING_HARVEST_START"
        elif len(start_dates) == 1:
            anchor = start_dates[0]
            source = "harvest_start_date"
            status = "PENDING_RANGE_CHECK"
        else:
            status = "MISSING_HARVEST"
        season_days = (anchor - sowing).days if pd.notna(anchor) else np.nan
        rows.append(
            {
                "environment_id": str(environment_id),
                "sowing_date": sowing.strftime("%Y-%m-%d") if pd.notna(sowing) else "",
                "harvest_anchor_date": (
                    anchor.strftime("%Y-%m-%d") if pd.notna(anchor) else ""
                ),
                "harvest_anchor_source": source,
                "season_days": season_days,
                "source_row_count": len(group),
                "distinct_sowing_date_count": len(sowing_dates),
                "distinct_harvest_finish_date_count": len(finish_dates),
                "distinct_harvest_start_date_count": len(start_dates),
                "status": status,
            }
        )

    audit = pd.DataFrame(rows)
    minimum = int(policy["minimum_valid_season_days"])
    maximum = int(policy["maximum_valid_season_days"])
    pending = audit["status"].eq("PENDING_RANGE_CHECK")
    in_range = audit["season_days"].between(minimum, maximum, inclusive="both")
    audit.loc[pending & in_range, "status"] = (
        "ELIGIBLE_NONPHENOTYPIC_HARVEST_ANCHOR"
    )
    audit.loc[pending & ~in_range, "status"] = "OUTSIDE_VALID_RANGE"
    eligible = audit.loc[
        audit["status"].eq("ELIGIBLE_NONPHENOTYPIC_HARVEST_ANCHOR"),
        "season_days",
    ].astype(int)
    endpoints = [int(value) for value in policy["candidate_inclusive_endpoint_days"]]
    coverage_rows = []
    for endpoint in endpoints:
        covered = int(eligible.le(endpoint).sum())
        coverage_rows.append(
            {
                "inclusive_endpoint_day": endpoint,
                "eligible_environment_count": len(eligible),
                "covered_environment_count": covered,
                "uncovered_environment_count": len(eligible) - covered,
                "coverage_fraction": covered / len(eligible) if len(eligible) else np.nan,
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    accepted = coverage.loc[
        coverage["coverage_fraction"].ge(
            float(policy["minimum_valid_nonphenotypic_harvest_coverage"])
        )
    ]
    endpoint = int(accepted.iloc[0]["inclusive_endpoint_day"]) if len(accepted) else None
    return audit, coverage, endpoint


def extension_inventories(
    environment_map: pd.DataFrame,
    endpoint: int,
    extension_start: int,
    reference_daily: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    local = environment_map.copy()
    local["latitude"] = pd.to_numeric(local["latitude"], errors="coerce")
    local["longitude"] = pd.to_numeric(local["longitude"], errors="coerce")
    local["_sowing"] = pd.to_datetime(local["sowing_date"], errors="coerce")
    local = local.loc[
        local["_sowing"].notna()
        & np.isfinite(local["latitude"])
        & np.isfinite(local["longitude"])
        & local["source_archive_status"].eq("READY_SOWING_RELATIVE_ARCHIVE")
    ].copy()
    local["request_start_date"] = (
        local["_sowing"] + pd.to_timedelta(extension_start, unit="D")
    ).dt.strftime("%Y-%m-%d")
    local["request_end_date"] = (
        local["_sowing"] + pd.to_timedelta(endpoint, unit="D")
    ).dt.strftime("%Y-%m-%d")
    local["request_id"] = local.apply(
        lambda row: stable_json_sha256(
            request_identity(
                row.latitude,
                row.longitude,
                row.request_start_date,
                row.request_end_date,
            )
        ),
        axis=1,
    )
    mapping = local[
        [
            "environment_id",
            "request_id",
            "latitude",
            "longitude",
            "sowing_date",
            "request_start_date",
            "request_end_date",
        ]
    ].sort_values("environment_id", kind="stable")
    counts = mapping.groupby("request_id").size().rename("mapped_environment_count")
    daily = (
        mapping.drop(columns=["environment_id", "sowing_date"])
        .drop_duplicates("request_id")
        .merge(counts, left_on="request_id", right_index=True)
        .sort_values("request_id", kind="stable")
        .reset_index(drop=True)
    )
    daily["provider"] = "open_meteo"
    daily["model"] = OPEN_METEO_MODEL
    daily["required_daily_variables"] = ";".join(DAILY_VARIABLES)
    daily["source_archive_status"] = "READY_PHENOLOGY_HORIZON_EXTENSION"
    daily["request_status"] = "READY_TO_FETCH"
    if reference_daily is None:
        daily["source_route"] = "FETCH_CDS_EXTENSION"
        reuse = daily.iloc[0:0].copy()
        masked = daily.iloc[0:0].copy()
    else:
        reference = reference_daily.copy()
        daily["_latitude_key"] = pd.to_numeric(daily["latitude"]).round(5)
        daily["_longitude_key"] = pd.to_numeric(daily["longitude"]).round(5)
        reference["_latitude_key"] = pd.to_numeric(reference["latitude"]).round(5)
        reference["_longitude_key"] = pd.to_numeric(reference["longitude"]).round(5)
        reference_columns = [
            "_latitude_key",
            "_longitude_key",
            "reference_site_id",
            "reference_request_id",
            "reference_start",
            "reference_end",
            "reference_daily_path",
            "reference_daily_sha256",
            "reference_metadata_path",
            "reference_metadata_sha256",
            "reference_required_climate_complete_days",
            "reference_required_climate_incomplete_days",
        ]
        daily = daily.merge(
            reference[reference_columns],
            on=["_latitude_key", "_longitude_key"],
            how="left",
            validate="many_to_one",
        )
        start = pd.to_datetime(daily["request_start_date"], errors="raise")
        end = pd.to_datetime(daily["request_end_date"], errors="raise")
        reference_start = pd.to_datetime(daily["reference_start"], errors="coerce")
        reference_end = pd.to_datetime(daily["reference_end"], errors="coerce")
        date_covered = start.ge(reference_start) & end.le(reference_end)
        reference_complete = (
            pd.to_numeric(
                daily["reference_required_climate_incomplete_days"], errors="coerce"
            ).eq(0)
            & pd.to_numeric(
                daily["reference_required_climate_complete_days"], errors="coerce"
            ).gt(0)
        )
        reusable = date_covered & reference_complete
        structurally_unavailable = date_covered & ~reference_complete
        daily["source_route"] = np.select(
            [reusable, structurally_unavailable],
            [
                "REUSE_CERTIFIED_CDS_1981_2010",
                "MASKED_NO_VALID_CDS_LAND_REFERENCE",
            ],
            default="FETCH_CDS_EXTENSION",
        )
        reuse = daily.loc[reusable].copy()
        masked = daily.loc[structurally_unavailable].copy()
        daily[reference_columns[2:]] = daily[reference_columns[2:]].astype(object)
        daily.loc[~date_covered, reference_columns[2:]] = ""
        daily = daily.drop(columns=["_latitude_key", "_longitude_key"])
        reuse = reuse.drop(columns=["_latitude_key", "_longitude_key"])
        masked = masked.drop(columns=["_latitude_key", "_longitude_key"])
    fetch_daily = daily.loc[daily["source_route"].eq("FETCH_CDS_EXTENSION")]
    if len(fetch_daily):
        cds = build_cds_request_inventory(fetch_daily)
    else:
        cds = pd.DataFrame(
            columns=[
                "request_id",
                "source_request_id",
                "dataset",
                "latitude",
                "longitude",
                "request_start_date",
                "request_end_date",
                "variable",
                "data_format",
                "mapped_environment_count",
                "request_payload_json",
                "request_status",
            ]
        )
    return mapping, daily, reuse, masked, cds


def load_bias_reference_daily(root: Path, directory: Path) -> pd.DataFrame:
    rows = []
    for metadata_path in sorted(directory.rglob("*.json")):
        record = read_json(metadata_path)
        if record.get("status") != "PASS":
            continue
        output_path = resolve(root, Path(str(record["output_path"])))
        if not output_path.is_file():
            raise FileNotFoundError(f"Certified CDS reference is absent: {output_path}")
        if sha256_file(output_path) != record["output_sha256"]:
            raise ValueError(f"Certified CDS reference checksum changed: {output_path}")
        rows.append(
            {
                "latitude": record["latitude"],
                "longitude": record["longitude"],
                "reference_site_id": record["site_id"],
                "reference_request_id": record["request_id"],
                "reference_start": record["internal_start"],
                "reference_end": record["internal_end"],
                "reference_daily_path": record["output_path"],
                "reference_daily_sha256": record["output_sha256"],
                "reference_metadata_path": metadata_path.relative_to(root).as_posix(),
                "reference_metadata_sha256": sha256_file(metadata_path),
                "reference_required_climate_complete_days": record[
                    "required_climate_complete_days"
                ],
                "reference_required_climate_incomplete_days": record[
                    "required_climate_incomplete_days"
                ],
            }
        )
    reference = pd.DataFrame(rows)
    if len(reference) != 907 or not reference["reference_site_id"].is_unique:
        raise ValueError("Continuous CDS reference does not contain 907 unique sites")
    if reference[["latitude", "longitude"]].duplicated().any():
        raise ValueError("Continuous CDS reference coordinates are not unique")
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze terminal FA disposition and Stage-1 v2 phenology readiness"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--fa-summary-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = (args.code_root or root).resolve()
    fa_summary = (
        args.fa_summary_dir.resolve()
        if args.fa_summary_dir
        else resolve(root, FA_SUMMARY)
    )
    paths = {
        "protocol": resolve(code_root, PROTOCOL),
        "plan": resolve(code_root, PLAN),
        "feature_contract": resolve(code_root, FEATURE_CONTRACT),
        "FA_decision": fa_summary / "FA_OPTIMIZATION_AMENDMENT_DECISION.json",
        "FA_decision_table": fa_summary / "fa_optimization_amendment_decision.tsv",
        "FA_activity": fa_summary / "fa_optimization_amendment_activity.tsv",
        "weather_manifest": resolve(root, WEATHER_MANIFEST),
        "environment_map": resolve(root, ENVIRONMENT_MAP),
        "projection_release": resolve(root, PROJECTION_RELEASE),
        "split_release": resolve(root, SPLIT_RELEASE),
        "certified_environment_axis": resolve(root, CERTIFIED_ENVIRONMENT_AXIS),
        "cross_provider_provenance": resolve(root, CROSS_PROVIDER_PROVENANCE),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phenology readiness inputs are missing: {missing}")

    protocol = read_json(paths["protocol"])
    plan = read_json(paths["plan"])
    feature_contract = read_json(paths["feature_contract"])
    fa = read_json(paths["FA_decision"])
    projection = read_json(paths["projection_release"])
    split = read_json(paths["split_release"])
    fa_table = pd.read_csv(paths["FA_decision_table"], sep="\t")
    activity = pd.read_csv(paths["FA_activity"], sep="\t")
    weather = pd.read_csv(paths["weather_manifest"], sep="\t", dtype=str)
    environment_map = pd.read_csv(paths["environment_map"], sep="\t", dtype=str)
    environment_axis = pd.read_csv(
        paths["certified_environment_axis"], sep="\t", dtype=str
    )
    axis_ids = environment_axis["environment_id"].astype(str)
    environment_map_ids = environment_map["environment_id"].astype(str)
    environment_map = environment_map.loc[environment_map_ids.isin(set(axis_ids))].copy()
    reference_daily = load_bias_reference_daily(root, resolve(root, BIAS_REFERENCE_DAILY))
    cross_provider = read_json(paths["cross_provider_provenance"])

    audit, coverage, endpoint = harvest_horizon_audit(
        weather, protocol["horizon_policy"]
    )
    current_endpoint = int(protocol["horizon_policy"]["current_inclusive_endpoint_day"])
    current_row = coverage.loc[
        coverage["inclusive_endpoint_day"].eq(current_endpoint)
    ].iloc[0]
    extension_required = endpoint is not None and endpoint > current_endpoint
    mapping = pd.DataFrame()
    daily = pd.DataFrame()
    cds = pd.DataFrame()
    reuse = pd.DataFrame()
    masked = pd.DataFrame()
    if endpoint is not None and extension_required:
        mapping, daily, reuse, masked, cds = extension_inventories(
            environment_map,
            endpoint,
            int(protocol["horizon_policy"]["extension_start_day"]),
            reference_daily,
        )

    candidate_rows = fa_table.loc[
        fa_table["candidate"].isin(
            ["normalized_direction_FA_rank2", "normalized_direction_FA_rank4"]
        )
    ]
    candidate_activity = activity.loc[
        activity["candidate"].isin(
            ["normalized_direction_FA_rank2", "normalized_direction_FA_rank4"]
        )
    ]
    terminal_checks = {
        "FA_status_terminal_no_advance": fa.get("status")
        == protocol["parent_FA_status"],
        "FA_selected_candidate_null": fa.get("selected_candidate") is None,
        "FA_confirmation_blocked": fa.get("full_confirmation_allowed") is False,
        "FA_outer_blocked": fa.get("outer_evaluation_allowed") is False,
        "FA_outer_and_final_unread": fa.get("outer_test_metrics_read") is False
        and fa.get("outer_test_outcomes_read") is False
        and fa.get("final_holdout_outcomes_read") is False,
        "two_active_FA_candidates_terminal": len(candidate_rows) == 2
        and candidate_rows["decision"].eq("active_component_do_not_advance").all(),
        "FA_activity_certified": len(candidate_activity) == 10
        and candidate_activity["FA_optimization_path_certified"].astype(bool).all()
        and candidate_activity["FA_final_component_active"].astype(bool).all(),
        "Huber_reference_retained": fa.get("reference_candidate")
        == protocol["retained_reference"],
    }
    readiness_checks = {
        "post_hierarchy_plan_v3": plan.get("protocol_version")
        == "stage1_v2_phase6_post_hierarchy_screen_plan_v3",
        "projection_core_v1_certified": projection.get("status")
        == "PASS_E_PROJECTION_CORE_V1_REMEDIATED_HISTORICAL_TRANSFER_CERTIFIED",
        "split_bound_150_states_certified": split.get("status")
        == "PASS_SPLIT_BOUND_HISTORICAL_PROJECTION_INPUTS_CERTIFIED"
        and int(split.get("state_count", 0)) == 150,
        "current_horizon_exact": feature_contract[
            "fixed_windows_days_relative_to_sowing"
        ]["w150_179"]
        == [150, 179],
        "horizon_selected_from_nonphenotypic_metadata": endpoint is not None,
        "selected_horizon_meets_coverage": endpoint is not None
        and float(
            coverage.loc[
                coverage["inclusive_endpoint_day"].eq(endpoint), "coverage_fraction"
            ].iloc[0]
        )
        >= float(
            protocol["horizon_policy"][
                "minimum_valid_nonphenotypic_harvest_coverage"
            ]
        ),
        "global_phenology_quantiles_forbidden": protocol["horizon_policy"][
            "global_DTH_or_DTM_quantiles_allowed"
        ]
        is False,
        "extension_inventory_nonempty": not extension_required or len(daily) > 0,
        "extension_request_ids_unique": not extension_required
        or daily["request_id"].is_unique,
        "weather_audit_environment_ids_unique": audit["environment_id"].is_unique,
        "certified_environment_axis_exact": len(environment_axis) == 11161
        and axis_ids.is_unique,
        "extension_source_map_matches_certified_axis": len(environment_map) == 11161
        and environment_map["environment_id"].is_unique
        and set(environment_map["environment_id"]) == set(axis_ids),
        "continuous_CDS_reference_907_sites": len(reference_daily) == 907,
        "extension_routes_exhaustive": not extension_required
        or len(reuse) + len(masked) + len(cds) == len(daily),
        "cross_provider_prior_certification_pass": cross_provider.get("status")
        == "PASS"
        and int(cross_provider.get("matched_request_count", 0)) == 7094
        and int(cross_provider.get("flagged_variable_count", -1)) == 0,
        "new_Open_Meteo_extension_fetch_not_required": protocol[
            "extension_contract"
        ]["cross_provider_policy"]["new_Open_Meteo_extension_fetch_required"]
        is False,
        "no_future_or_protected_outcome_access": all(
            protocol["protected_access"][key] is False
            for key in (
                "phenotype_values_read_for_horizon_selection",
                "inner_validation_metric_values_used_for_horizon_selection",
                "outer_test_metrics_read",
                "outer_test_outcomes_read",
                "final_holdout_outcomes_read",
                "future_SSP_values_read",
            )
        ),
    }
    terminal_checks = {key: bool(value) for key, value in terminal_checks.items()}
    readiness_checks = {key: bool(value) for key, value in readiness_checks.items()}
    failed = [
        key
        for key, value in {**terminal_checks, **readiness_checks}.items()
        if not value
    ]

    fa_output = resolve(root, FA_RELEASE)
    fa_output.mkdir(parents=True, exist_ok=True)
    fa_release = {
        "status": (
            "PASS_STAGE1_V2_PHASE6_FA_TERMINAL_NO_ADVANCE"
            if all(terminal_checks.values())
            else "FAIL_STAGE1_V2_PHASE6_FA_TERMINAL_FREEZE"
        ),
        "protocol_version": "stage1_v2_phase6_FA_terminal_no_advance_v1",
        "selected_candidate": None,
        "retained_reference": protocol["retained_reference"],
        "FA_confirmation_allowed": False,
        "outer_evaluation_allowed": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "checks": terminal_checks,
        "artifacts": {
            key: sha256_file(paths[key])
            for key in ("FA_decision", "FA_decision_table", "FA_activity")
        },
    }
    write_json(fa_output / "FA_TERMINAL_NO_ADVANCE_RELEASE.json", fa_release)

    output = resolve(root, OUTPUT)
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(
        output / "nonphenotypic_harvest_horizon_audit.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    coverage.to_csv(
        output / "daily_horizon_coverage.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    extension = output / "horizon_extension_contract"
    extension.mkdir(parents=True, exist_ok=True)
    if extension_required:
        mapping_path = extension / "environment_daily_extension_map.tsv"
        daily_path = extension / "daily_request_inventory.tsv"
        reuse_path = extension / "certified_CDS_reference_reuse_inventory.tsv"
        masked_path = extension / "masked_CDS_reference_inventory.tsv"
        cds_path = extension / "cds_era5_land_request_inventory.tsv"
        mapping.to_csv(
            mapping_path,
            sep="\t",
            index=False,
            lineterminator="\n",
        )
        daily.to_csv(
            daily_path,
            sep="\t",
            index=False,
            lineterminator="\n",
        )
        reuse.to_csv(
            reuse_path,
            sep="\t",
            index=False,
            lineterminator="\n",
        )
        masked.to_csv(
            masked_path,
            sep="\t",
            index=False,
            lineterminator="\n",
        )
        cds.to_csv(
            cds_path,
            sep="\t",
            index=False,
            lineterminator="\n",
        )
        extension_contract = {
            "status": "PASS",
            "protocol_version": "stage1_v2_phase6_phenology_daily_horizon_extension_v2_reuse_first",
            "selection_data": "stage1_v2_environment_identifiers_coordinates_and_sowing_metadata_only",
            "inclusive_extension_days": [
                int(protocol["horizon_policy"]["extension_start_day"]),
                endpoint,
            ],
            "environment_count": len(mapping),
            "request_count": len(daily),
            "certified_CDS_reference_reuse_request_count": len(reuse),
            "masked_no_valid_CDS_land_reference_request_count": len(masked),
            "new_CDS_fetch_request_count": len(cds),
            "new_Open_Meteo_fetch_request_count": 0,
            "cross_provider_certification_path": paths[
                "cross_provider_provenance"
            ].relative_to(root).as_posix(),
            "cross_provider_certification_sha256": sha256_file(
                paths["cross_provider_provenance"]
            ),
            "superseded_extension_cache_policy": "preserve_content_addressed_files_but_use_only_active_manifest_request_ids",
            "phenotype_values_read": False,
            "inner_validation_metrics_read": False,
            "outer_test_metrics_read": False,
            "outer_test_outcomes_read": False,
            "final_holdout_outcomes_read": False,
            "future_SSP_values_read": False,
            "future_predictions_generated": 0,
            "artifacts": {
                mapping_path.name: sha256_file(mapping_path),
                daily_path.name: sha256_file(daily_path),
                reuse_path.name: sha256_file(reuse_path),
                masked_path.name: sha256_file(masked_path),
                cds_path.name: sha256_file(cds_path),
            },
        }
        write_json(extension / "environment_source_contract.json", extension_contract)
    readiness = {
        "status": (
            "PASS_READY_FOR_PHENOLOGY_PHASE1"
            if not failed and not extension_required
            else (
                "BLOCKED_PHENOLOGY_DAILY_HORIZON_EXTENSION_REQUIRED"
                if not failed
                else "FAIL_STAGE1_V2_PHASE6_PHENOLOGY_READINESS"
            )
        ),
        "protocol_version": protocol["protocol_version"],
        "selection_data": protocol["selection_data"],
        "retained_reference": protocol["retained_reference"],
        "current_inclusive_endpoint_day": current_endpoint,
        "current_horizon_nonphenotypic_harvest_coverage": float(
            current_row["coverage_fraction"]
        ),
        "selected_inclusive_endpoint_day": endpoint,
        "selected_horizon_nonphenotypic_harvest_coverage": (
            float(
                coverage.loc[
                    coverage["inclusive_endpoint_day"].eq(endpoint),
                    "coverage_fraction",
                ].iloc[0]
            )
            if endpoint is not None
            else None
        ),
        "extension_required": extension_required,
        "extension_day_range": (
            [int(protocol["horizon_policy"]["extension_start_day"]), endpoint]
            if extension_required
            else None
        ),
        "extension_request_count": len(daily),
        "extension_reused_CDS_request_count": len(reuse),
        "extension_masked_no_valid_CDS_land_request_count": len(masked),
        "extension_new_CDS_fetch_request_count": len(cds),
        "extension_new_Open_Meteo_fetch_request_count": 0,
        "extension_environment_count": len(mapping),
        "phenology_phase1_state_count": int(
            protocol["prospective_phenology_contract"]["phase_1_state_count"]
        ),
        "phenology_training_allowed": not failed and not extension_required,
        "outer_evaluation_allowed": False,
        "future_predictions_allowed": False,
        "phenotype_values_read_for_horizon_selection": False,
        "inner_validation_metric_values_used_for_horizon_selection": False,
        "terminal_FA_decision_status_read": True,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "future_SSP_values_read": False,
        "future_predictions_generated": 0,
        "checks": readiness_checks,
        "failed_checks": failed,
        "artifacts": {
            "protocol": sha256_file(paths["protocol"]),
            "post_hierarchy_plan": sha256_file(paths["plan"]),
            "FA_terminal_release": sha256_file(
                fa_output / "FA_TERMINAL_NO_ADVANCE_RELEASE.json"
            ),
            "weather_manifest": sha256_file(paths["weather_manifest"]),
            "environment_map": sha256_file(paths["environment_map"]),
            "projection_release": sha256_file(paths["projection_release"]),
            "split_release": sha256_file(paths["split_release"]),
            "certified_environment_axis": sha256_file(
                paths["certified_environment_axis"]
            ),
            "cross_provider_provenance": sha256_file(
                paths["cross_provider_provenance"]
            ),
        },
    }
    write_json(output / "PHENOLOGY_READINESS_DECISION.json", readiness)
    print(json.dumps(fa_release, indent=2, sort_keys=True))
    print(json.dumps(readiness, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"Phenology readiness integrity failed: {failed}")


if __name__ == "__main__":
    main()
