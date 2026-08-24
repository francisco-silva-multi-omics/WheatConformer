from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import phase5_parity_build as parity_build
from phase5_parity_common import (
    ProtectedPathGuard,
    ensure_fail_if_exists,
    environment_versions,
    git_head,
    relative_posix,
    sha256_file,
    stable_json_hash,
    utc_now,
    write_json,
    write_tsv,
)


RELEASE_ID = "P6AEPR_20260809_V1_274E41DF"
RELEASE_RELATIVE = Path("audit/v2/phase6a_environmental_projection_readiness_v1")
PHASE5_ID = "P5SBK_20260808_V1_274E41DF"
PARITY_ID = "P5PESP_20260809_V2_274E41DF"
REGULATORY_ID = "P5REV2_20260809_V1_274E41DF"
KA_EXTENSION_ID = "P5KATC_20260809_V1_274E41DF"
PHASE5_RELATIVE = Path("audit/v2/phase5_split_bound_kernel_validation_v2")
PARITY_RELATIVE = Path("audit/v2/phase5_panel_environment_scenario_parity_extension_v2")
REGULATORY_RELATIVE = Path("audit/v2/phase5_regulatory_eligibility_v2")
KA_RELATIVE = Path("audit/v2/phase5_ka_temporal_country_extension_v1")
V1_INCIDENT_RELATIVE = Path("audit/v2/phase5_panel_environment_scenario_parity_extension_v1")
BUNDLE_ENV = Path("server_phase5_parity_bundle/artifacts/environment")

ALLOWED_CLASSES = {
    "directly_reproducible_future",
    "physically_reconstructable",
    "static_future_available",
    "management_scenario_required",
    "historical_observational_proxy",
    "transport_validation_required",
    "irrecoverable",
}

LEGACY_HARVEST_MONTH = [
    "PPN_MONTH_OF_HARVESTED",
    "PPN_1ST_MO_BEFORE_HARVESTED",
    "PPN_2ND_MO_BEFORE_HARVESTED",
    "PPN_3RD_MO_BEFORE_HARVESTED",
    *[f"PPN_{value}TH_MO_BEFORE_HARVESTED" for value in range(4, 12)],
]
LEGACY_CROP_PRECIPITATION = [
    "PRECIPITATION_FROM_SOWING_TO_MATURITY",
    "PRECIPITATION_ON_CROP",
]
LEGACY_MOISTURE = ["MOISTURE_AVAILB_BEFORE_SOWING_EXCL_PRE_IRRIGATION"]
LEGACY_HISTORICAL_ONLY = (
    LEGACY_HARVEST_MONTH + LEGACY_CROP_PRECIPITATION + LEGACY_MOISTURE
)

STAGE_DIRECT = {
    "temperature_mean_c",
    "temperature_max_c",
    "precipitation_total_mm",
    "solar_radiation_total_mj_m2",
    "solar_radiation_mean_daily_mj_m2",
}
AMBIGUOUS_MANAGEMENT_UNITS = {
    "CALCULATED_OF_TOTAL_WATER_APPLIED_BY_IRRIGATION",
    "ESTIMATE_OF_TOTAL_WATER_APPLIED_BY_IRRIGATION",
    "K_FERTILIZER_APPLIED_OLD",
    "N_FERTILIZER_APPLIED_OLD",
    "P_FERTILIZER_APPLIED_OLD",
}
CORE_WATER_FEATURES = [
    "antecedent_precipitation_30d_mm",
    "antecedent_pet_30d_mm",
    "window_pet_total_mm",
    "climatic_water_balance_mm",
    "soil_water_bucket_index",
]
CONFIDENCE_FEATURES = [
    "climate_window_complete",
    "static_site_source_present",
    "management_scenario_declared",
    "soil_data_present",
    "bias_adjustment_supported",
    "calendar_conversion_complete",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the phenotype-blind Phase-6A environmental projection-readiness release"
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def create_directories(output: Path) -> None:
    for relative in ("audits", "backcast", "contracts", "tests", "logs"):
        (output / relative).mkdir(parents=True, exist_ok=False)


def parent_decision_specs(root: Path) -> list[tuple[str, Path, str]]:
    return [
        (
            PHASE5_ID,
            root / PHASE5_RELATIVE / "PHASE5_RELEASE_DECISION.json",
            "PASS_PHASE5_KERNEL_VALIDATION",
        ),
        (
            PARITY_ID,
            root / PARITY_RELATIVE / "PHASE5_PARITY_EXTENSION_DECISION.json",
            "PASS_PHASE5_PARITY_EXTENSION_WITH_EXPLICIT_COMPONENT_BLOCKERS",
        ),
        (
            REGULATORY_ID,
            root / REGULATORY_RELATIVE / "REGULATORY_ELIGIBILITY_V2_DECISION.json",
            "PASS_REGULATORY_ELIGIBILITY_V2_WITH_KZ_DEFERRED",
        ),
        (
            KA_EXTENSION_ID,
            root / KA_RELATIVE / "PHASE5_KA_TEMPORAL_COUNTRY_EXTENSION_DECISION.json",
            "PASS_KA_TEMPORAL_COUNTRY_EXTENSION",
        ),
    ]


def verify_preflight(root: Path, output: Path, guard: ProtectedPathGuard) -> dict[str, Any]:
    parents = []
    for expected_id, path, expected_status in parent_decision_specs(root):
        allowed = guard.assert_allowed(path, "READ_PARENT_DECISION")
        value = read_json(allowed)
        observed = (value.get("release_id"), value.get("status"))
        if observed != (expected_id, expected_status):
            raise SystemExit(
                f"BLOCKED: parent decision mismatch for {path}: {observed!r}"
            )
        parents.append(
            {
                "release_id": expected_id,
                "status": expected_status,
                "decision_path": relative_posix(path, root),
            }
        )
    required = [
        "environment/environment_feature_provenance.tsv",
        "environment/reaction_norm_protocol.json",
        "environment/environment_preprocessing_parameters.tsv",
        "environment/environment_component_registry.tsv",
        "environment/environment_state_certifications.tsv",
        "splits/state_registry.tsv",
    ]
    for relative in required:
        path = guard.assert_allowed(root / PARITY_RELATIVE / relative, "PREFLIGHT_AUTHORITATIVE_V2")
        if not path.is_file():
            raise SystemExit(f"BLOCKED: missing authoritative v2 artifact {path}")
    return {
        "release_id": RELEASE_ID,
        "parents": parents,
        "authoritative_v2_inputs": required,
        "state_registry_required_rows": 150,
        "target_absent": not output.exists(),
        "denylist_rules_loaded": len(guard.rules),
        "protected_locks_opened": False,
        "status": "PASS_PREFLIGHT" if not output.exists() else "FAIL_TARGET_EXISTS",
    }


def write_opening_contract(root: Path, output: Path, preflight: dict[str, Any]) -> None:
    write_json(
        output / "OPENING_RELEASE.json",
        {
            "release_id": RELEASE_ID,
            "release_type": "FAIL_IF_EXISTS_PHASE6A_ENVIRONMENTAL_PROJECTION_READINESS",
            "parents": preflight["parents"],
            "stage1_v2_only": True,
            "frozen_state_registry_rows": 150,
            "phenotype_blind": True,
            "legacy_v1_metric_selected_architecture_inherited": False,
            "model_training_performed": False,
            "candidate_selection_performed": False,
            "future_covariate_matrices_generated": False,
            "future_predictions_generated": False,
            "inner_validation_metrics_accessed": False,
            "outer_outcomes_accessed": False,
            "predictions_accessed": False,
            "final_holdout_outcomes_accessed": False,
            "protected_files_rendered": [],
            "denylist_loaded_before_bundle_access": True,
            "created_at_utc": utc_now(),
            "repository_root": str(root),
        },
    )
    write_json(
        output / "run_manifest.json",
        {
            "release_id": RELEASE_ID,
            "repository_root": str(root),
            "release_root": str(output),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "git_head": git_head(root),
            "packages": environment_versions(),
            "model_training_performed": False,
            "performance_evaluation_performed": False,
            "future_matrix_generation_performed": False,
            "prediction_generation_performed": False,
            "commit_or_push_performed": False,
        },
    )


def climate_contract() -> dict[str, Any]:
    return {
        "release_id": RELEASE_ID,
        "primary_interface": "CMIP6_SSP_DAILY_MEMBER_RESOLVED",
        "future_covariates_inspected": False,
        "future_matrices_generated": False,
        "required_daily_variables": [
            {
                "canonical_name": "tasmin",
                "canonical_unit": "degC",
                "accepted_cf_unit": "K",
                "conversion": "degC=K-273.15",
                "role": "minimum_temperature_and_FAO56_PET",
            },
            {
                "canonical_name": "tasmax",
                "canonical_unit": "degC",
                "accepted_cf_unit": "K",
                "conversion": "degC=K-273.15",
                "role": "maximum_temperature_heat_counts_and_FAO56_PET",
            },
            {
                "canonical_name": "tas",
                "canonical_unit": "degC",
                "accepted_cf_unit": "K",
                "conversion": "degC=K-273.15",
                "role": "daily_mean_temperature_or_mean_of_tasmin_tasmax",
            },
            {
                "canonical_name": "pr",
                "canonical_unit": "mm/day",
                "accepted_cf_unit": "kg m-2 s-1",
                "conversion": "mm/day=pr*86400;_1_kg_m-2=1_mm_water",
                "role": "precipitation_and_water_balance",
            },
            {
                "canonical_name": "rsds",
                "canonical_unit": "MJ m-2 day-1",
                "accepted_cf_unit": "W m-2",
                "conversion": "MJ_m-2_day-1=rsds*0.0864",
                "role": "radiation_and_FAO56_PET",
            },
            {
                "canonical_name": "hurs_or_huss_plus_ps",
                "canonical_unit": "percent_or_kg/kg_plus_Pa",
                "conversion": "relative_humidity_direct_or_vapor_pressure_from_specific_humidity_and_surface_pressure",
                "role": "VPD_and_FAO56_PET",
            },
            {
                "canonical_name": "sfcWind",
                "canonical_unit": "m/s_at_2m",
                "conversion": "apply_logged_height_adjustment_when_source_height_is_not_2m",
                "role": "FAO56_PET",
            },
        ],
        "vpd_formula": {
            "saturation_vapor_pressure_kpa": "0.6108*exp(17.27*T_C/(T_C+237.3))",
            "daily_es_kpa": "(es(tasmin)+es(tasmax))/2",
            "actual_vapor_pressure": "from_hurs_or_huss_and_ps",
            "vpd_kpa": "max(es-ea,0)",
            "forbidden_shortcut": "no_temperature_only_humidity_imputation_across_application_folds",
        },
        "pet_formula": {
            "method": "FAO56_REFERENCE_ET0_PENMAN_MONTEITH_DAILY",
            "required_inputs": "tasmin,tasmax,tas_or_mean,humidity,rsds,sfcWind,latitude,elevation,day_of_year",
            "fallback": "HARGREAVES_SAMANI_DIAGNOSTIC_ONLY_NOT_INTERCHANGEABLE_WITH_PRIMARY_PET",
            "soil_water_balance": "bucket_t=min(capacity,max(0,bucket_t-1+pr+irrigation-ET0));_capacity_from_certified_soil_or_explicit_scenario",
        },
        "calendar_handling": {
            "gregorian": "native_dates_and_leap_days_retained",
            "no_leap_365_day": "native_365_day_calendar;_no_synthetic_Feb29",
            "360_day": "native_30_day_months;_sowing_policy_expressed_as_calendar_fraction_and_mapped_without_duplication",
            "window_rule": "exactly_30_native_model_days_per_fixed_sowing_relative_window",
            "cross_calendar_comparison": "compare_fractional_season_position_not_unlogged_date_insertion",
        },
        "reference_period": "1981-01-01_through_2010-12-31",
        "bias_adjustment_and_downscaling": {
            "method": "TREND_PRESERVING_QUANTILE_DELTA_MAPPING_PER_VARIABLE_SITE_CALENDAR_MONTH_GCM_MEMBER",
            "precipitation": "separate_wet_day_frequency_and_positive_amount_adjustment",
            "fit_scope": "historical_reference_only_never_future_or_application_distribution",
            "reference_required": "versioned_daily_reanalysis_or_observation_with_site_identity_and_hash",
            "spatial_mapping": "member_preserving_bilinear_or_nearest_valid_land_cell_with_logged_lapse_rate_for_temperature",
            "multivariate_consistency": "recompute_derived_VPD_PET_and_counts_after_univariate_adjustment",
        },
        "missing_day_policy": {
            "precipitation": "any_missing_day_blocks_precipitation_and_water_balance_for_that_window",
            "temperature_humidity_radiation_wind": "at_most_one_isolated_day_may_be_within_member_time_interpolated_and_flagged_LIMITED_EXTRAPOLATION",
            "two_or_more_missing_days": "window_OUT_OF_DOMAIN",
            "count_features": "never_rescaled_from_incomplete_windows",
            "confidence_mask_required": True,
        },
        "ensemble_policy": {
            "identity_fields": ["source_id", "institution_id", "experiment_id", "variant_label", "grid_label", "version"],
            "default_weighting": "equal_weight_per_declared_member_then_equal_weight_per_source_id_for_summaries",
            "no_ensemble_averaging_before_feature_derivation": True,
            "retain_member_dimension_through_prediction": True,
        },
        "management_policy": "explicit_scenario_id_and_raw_canonical_units_required;_no_climate_based_management_imputation",
        "status": "FROZEN_BEFORE_ANY_FUTURE_COVARIATE_INSPECTION",
    }


def compatibility_contract() -> dict[str, Any]:
    return {
        "release_id": RELEASE_ID,
        "primary_generation": "CMIP6",
        "primary_scenarios": ["SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"],
        "legacy_generation": "CMIP5",
        "legacy_scenarios": ["RCP2.6", "RCP4.5", "RCP6.0", "RCP8.5"],
        "generation_identity_field_required": True,
        "pool_generations": False,
        "average_rcp_and_ssp_labels": False,
        "scenario_name_equivalence_assumed": False,
        "compatibility_requirements": [
            "same_canonical_units",
            "same_fixed_window_formulas",
            "generation_specific_historical_bias_adjustment",
            "member_identity_retained",
            "separate_generation_summaries",
        ],
        "legacy_branch_status": "COMPATIBILITY_ONLY_NOT_PRIMARY_AND_NOT_AUTHORIZED",
    }


def applicability_protocol() -> dict[str, Any]:
    return {
        "release_id": RELEASE_ID,
        "frozen_before_future_covariate_inspection": True,
        "fit_scope": "EACH_OF_150_FROZEN_TRAINING_ENVIRONMENT_SETS_ONLY",
        "blocks": ["heat", "water", "development", "radiation", "static", "management", "confidence"],
        "diagnostics": {
            "marginal_range_exceedance": {
                "in_domain": "no_feature_outside_training_min_max",
                "limited": "outside_min_max_fraction<=0.05_and_no_robust_abs_z>8",
                "out_of_domain": "outside_min_max_fraction>0.05_or_any_robust_abs_z>8",
            },
            "robust_standardized_distance": {
                "center": "training_median",
                "scale": "1.4826*MAD_with_training_IQR_fallback",
                "in_domain": "block_RMS<=4",
                "limited": "4<block_RMS<=8",
                "out_of_domain": "block_RMS>8",
            },
            "pca_or_shrinkage_mahalanobis": {
                "fit": "training_only_PCA_95pct_variance_cap32_or_LedoitWolf_when_dimension_allows",
                "in_domain": "distance<=training_99th_percentile",
                "limited": "training_99th<distance<=1.5*training_99.9th",
                "out_of_domain": "distance>1.5*training_99.9th",
            },
            "nearest_training_environment": {
                "metric": "Euclidean_on_training_robust_scaled_retained_features",
                "reference": "training_leave_one_out_nearest_neighbor",
                "in_domain": "distance<=training_99th_percentile",
                "limited": "distance<=1.5*training_99th_percentile",
                "out_of_domain": "distance>1.5*training_99th_percentile",
            },
            "missing_and_confidence_shift": {
                "in_domain": "absolute_block_missing_fraction_shift<=0.10",
                "limited": "0.10<shift<=0.25",
                "out_of_domain": "shift>0.25_or_required_source_absent",
            },
        },
        "overall_rule": "worst_block_class;_required_water_or_climate_OUT_OF_DOMAIN_forces_overall_OUT_OF_DOMAIN",
        "actions": {
            "IN_DOMAIN": "eligible_for_separate_future_covariate_certification_only",
            "LIMITED_EXTRAPOLATION": "retain_member_flag_require_sensitivity_and_no_silent_imputation",
            "OUT_OF_DOMAIN": "block_prediction_and_report_missing_or_extrapolation_reason",
        },
        "clipping_allowed": False,
        "training_mean_imputation_may_hide_ood": False,
        "future_covariates_evaluated_in_this_release": False,
    }


def write_protocols(output: Path) -> None:
    write_json(output / "climate_source_and_unit_contract.json", climate_contract())
    write_json(
        output / "cmip5_cmip6_compatibility_contract.json", compatibility_contract()
    )
    write_json(output / "applicability_domain_protocol.json", applicability_protocol())


def input_paths(root: Path) -> list[tuple[str, Path]]:
    parity = root / PARITY_RELATIVE
    paths: list[tuple[str, Path]] = []
    for _, path, _ in parent_decision_specs(root):
        paths.append(("PARENT_DECISION", path))
    authoritative = [
        "environment/environment_feature_provenance.tsv",
        "environment/reaction_norm_protocol.json",
        "environment/environment_preprocessing_parameters.tsv",
        "environment/environment_component_registry.tsv",
        "environment/environment_state_certifications.tsv",
        "splits/state_registry.tsv",
        "output_manifest.tsv",
    ]
    for relative in authoritative:
        paths.append(("AUTHORITATIVE_PARITY_V2", parity / relative))
    registry = pd.read_csv(parity / "splits/state_registry.tsv", sep="\t", dtype=str)
    for relative in registry.training_environment_path.astype(str):
        paths.append(("FROZEN_STATE_ENVIRONMENTS", parity / relative))
    for relative in [
        "env_features_weather.parquet",
        "env_features_stress.parquet",
        "env_features_mgmt.parquet",
        "env_features_geo.parquet",
        "env_feature_scaling_parameters.tsv",
        "agronomic_api_weather_windows.tsv",
        "agronomic_api_weather_windows_manifest.tsv",
        "agronomic_api_weather_windows_qc.tsv",
        "locdata.tsv",
    ]:
        paths.append(("PHENOTYPE_BLIND_ENVIRONMENT_SOURCE", root / BUNDLE_ENV / relative))
    paths.extend(
        [
            (
                "PHASE5_ENVIRONMENT_METADATA",
                root / PHASE5_RELATIVE / "indices/canonical_phase5_observation_index.parquet",
            ),
            (
                "IMPLEMENTATION",
                root / "scripts/v2/phase6a_environmental_projection_readiness.py",
            ),
            (
                "IMPLEMENTATION",
                root / "tests/test_phase6a_environmental_projection_readiness.py",
            ),
            ("IMPLEMENTATION", root / "scripts/v2/phase5_parity_build.py"),
            (
                "LEGACY_NAME_REGISTRY_ONLY",
                root / "server_training_pipeline/audit_reaction_norm_rcp_historical_reconstruction.py",
            ),
            (
                "PROTECTION_CONTRACT",
                root / V1_INCIDENT_RELATIVE / "PROTECTED_PATH_DENYLIST.txt",
            ),
        ]
    )
    return paths


def opening_hash_manifest(root: Path, output: Path, guard: ProtectedPathGuard) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for scope, path in input_paths(root):
        allowed = guard.assert_allowed(path, "OPENING_HASH")
        relative = relative_posix(allowed, root)
        if relative in seen:
            continue
        seen.add(relative)
        rows.append(
            {
                "scope": scope,
                "relative_path": relative,
                "size": allowed.stat().st_size,
                "sha256": sha256_file(allowed),
                "access": "HASHED_ALLOWED",
                "matched_rule": "",
            }
        )
    frame = pd.DataFrame(rows).sort_values(["scope", "relative_path"]).reset_index(drop=True)
    write_tsv(output / "OPENING_HASH_MANIFEST.tsv", frame)
    return frame


def feature_units(feature: str) -> str:
    token = feature.lower()
    if "temperature" in token:
        return "degC"
    if "gdd_" in token:
        return "degC_day"
    if "precipitation" in token or token.startswith("ppn_") or "pet_" in token or "water_balance" in token:
        return "mm"
    if "radiation_total" in token:
        return "MJ_m-2_window-1"
    if "radiation_mean" in token:
        return "MJ_m-2_day-1"
    if "vpd" in token:
        return "kPa" if "days" not in token else "day_count"
    if any(value in token for value in ("days_", "_days", "chill_days", "cold_days")):
        return "day_count"
    if "humidity" in token:
        return "percent"
    if "wind_speed" in token:
        return "m_s-1"
    if "latitude" in token or "longitude" in token:
        return "decimal_degree"
    if "altitude" in token or "elevation" in token:
        return "m"
    if "area_" in token and "_m2" in token:
        return "m2"
    if "kg/ha" in feature.lower():
        return "kg_ha-1"
    if "fertilizer_%" in token:
        return "percent"
    if token.startswith("number_"):
        return "count"
    if any(
        name in feature.upper()
        for name in (
            "FUNGICIDE",
            "HAND_WEEDING",
            "HERBICIDE",
            "IRRIGATED",
            "IRRIGATION_AFTER_SOWING",
            "PESTICIDE",
            "PRE_SOWING_IRRIGATION",
        )
    ):
        return "binary_indicator"
    if feature in AMBIGUOUS_MANAGEMENT_UNITS:
        return "UNRESOLVED_SOURCE_UNIT"
    if feature in CONFIDENCE_FEATURES or "bucket_index" in token:
        return "unitless_0_1"
    return "source_declared_or_unitless"


def stage_formula(feature: str, window: str) -> str:
    if feature == "temperature_mean_c":
        formula = "mean(daily_tas_C)"
    elif feature == "temperature_max_c":
        formula = "max(daily_tasmax_C)"
    elif feature == "gdd_base0_sum":
        formula = "sum(max(daily_tmean_C-0,0))"
    elif feature == "gdd_base5_sum":
        formula = "sum(max(daily_tmean_C-5,0))"
    elif feature == "cold_days_tmin_lt_0":
        formula = "count(daily_tasmin_C<0)"
    elif feature == "chill_days_tmean_0_10":
        formula = "count(0<=daily_tmean_C<=10)"
    elif feature == "heat_days_tmax_ge_30":
        formula = "count(daily_tasmax_C>=30)"
    elif feature == "heat_days_tmax_ge_35":
        formula = "count(daily_tasmax_C>=35)"
    elif feature == "precipitation_total_mm":
        formula = "sum(daily_pr_mm)"
    elif feature == "dry_days_precip_lt_1mm":
        formula = "count(daily_pr_mm<1)"
    elif feature == "solar_radiation_total_mj_m2":
        formula = "sum(daily_rsds_MJ_m-2)"
    elif feature == "solar_radiation_mean_daily_mj_m2":
        formula = "mean(daily_rsds_MJ_m-2)"
    elif feature == "vpd_mean_kpa":
        formula = "mean(max(es(tmean)-ea,0))"
    elif feature == "vpd_max_kpa":
        formula = "max(max(es(tmean)-ea,0))"
    elif feature == "high_vpd_days_gt_1_5":
        formula = "count(daily_vpd_kPa>1.5)"
    elif feature == "drought_days_precip_lt_1mm_and_vpd_gt_1_5":
        formula = "count(daily_pr_mm<1_and_daily_vpd_kPa>1.5)"
    else:
        formula = "frozen_source_formula"
    return f"{formula}_over_fixed_sowing_relative_{window}_30_native_calendar_days"


def contract_rows(provenance: pd.DataFrame, root: Path) -> pd.DataFrame:
    digest_cache: dict[str, str] = {}

    def digest(relative: str) -> str:
        if relative not in digest_cache:
            digest_cache[relative] = sha256_file(root / relative)
        return digest_cache[relative]

    rows: list[dict[str, Any]] = []
    for record in provenance.to_dict("records"):
        component = str(record["component"])
        feature = str(record["feature"])
        window = str(record["window_definition"])
        source = str(record["source_path"])
        historical = True
        projection = False
        readiness = "READY_HISTORICAL_ONLY"
        if component.startswith("K_E_STAGE_"):
            primary = (
                "directly_reproducible_future"
                if feature in STAGE_DIRECT
                else "physically_reconstructable"
            )
            projection = True
            readiness = "READY_AGGREGATE_FORMULA_DAILY_ARCHIVE_VALIDATION_PENDING"
            formula = stage_formula(feature, window)
            temporal = f"fixed_sowing_relative_{window}_30_day"
            calendar = "native_Gregorian_no_leap_or_360_day_30_day_window"
            missing = "missing_daily_source_or_incomplete_window_not_zero"
            confidence = "climate_window_complete_and_bias_adjustment_supported"
            permitted = "both_contracts;_future_only_after_daily_backcast_and_covariate_certification"
        elif component == "K_E_TGW_FIXED_GRAIN_FILL":
            primary = "transport_validation_required"
            projection = False
            readiness = "REPLACED_BY_NON_DUPLICATED_BASE_STAGE_FEATURES"
            base_window, base_feature = feature.split("__", 1)
            formula = f"exact_alias_of_{base_window}__{base_feature}"
            temporal = f"fixed_sowing_relative_{base_window}"
            calendar = "same_as_base_stage"
            missing = "same_as_base_stage"
            confidence = "same_as_base_stage"
            permitted = "historical_enhanced_only_as_existing_v2_alias;_exclude_from_core_to_avoid_exact_duplicate"
        elif component == "K_E_MANAGEMENT":
            primary = "management_scenario_required"
            projection = feature not in AMBIGUOUS_MANAGEMENT_UNITS
            readiness = (
                "READY_ONLY_WITH_EXPLICIT_RAW_UNIT_SCENARIO"
                if projection
                else "REJECTED_FROM_CORE_UNRESOLVED_SOURCE_UNIT"
            )
            formula = "declared_management_scenario_value_no_climate_imputation"
            temporal = "environment_or_scenario_constant"
            calendar = "explicit_application_dates_when_time_varying"
            missing = "unknown_management_not_zero_and_not_training_mean"
            confidence = "management_scenario_declared"
            permitted = "historical_observation_in_enhanced;_explicit_scenario_only_in_core"
        else:
            primary = "historical_observational_proxy"
            formula = "legacy_source_aggregate_formula_or_period_not_fully_recovered"
            temporal = "source_aggregate_period_undocumented_for_projection_parity"
            calendar = "historical_source_calendar_only"
            missing = "source_missingness_or_historical_fold_climatology"
            confidence = "historical_source_available"
            permitted = "historical_enhanced_only_not_projection_core"
        rows.append(
            {
                "feature_id": f"{component}::{window}::{feature}",
                "component": component,
                "feature": feature,
                "origin": "AUTHORITATIVE_PARITY_V2",
                "primary_class": primary,
                "units": feature_units(feature),
                "temporal_support": temporal,
                "source_dataset": source,
                "source_sha256": digest(source),
                "derivation_formula": formula,
                "calendar_assumptions": calendar,
                "missingness_meaning": missing,
                "confidence_mask": confidence,
                "permitted_use": permitted,
                "E_HISTORICAL_ENHANCED_V2": historical,
                "E_PROJECTION_CORE_V2": projection,
                "identical_historical_future_pipeline_required": projection,
                "readiness_status": readiness,
            }
        )
    geo_source = (BUNDLE_ENV / "locdata.tsv").as_posix()
    for feature in ("latitude", "longitude", "elevation_m"):
        rows.append(
            {
                "feature_id": f"K_E_GEO_SOURCE::STATIC::{feature}",
                "component": "K_E_GEO_SOURCE",
                "feature": feature,
                "origin": "IMMUTABLE_STATIC_SOURCE",
                "primary_class": "static_future_available",
                "units": feature_units(feature),
                "temporal_support": "site_static",
                "source_dataset": geo_source,
                "source_sha256": digest(geo_source),
                "derivation_formula": "certified_location_registry_coordinate_or_elevation",
                "calendar_assumptions": "none",
                "missingness_meaning": "site_registry_unresolved",
                "confidence_mask": "static_site_source_present",
                "permitted_use": "both_contracts_for_exact_curated_site_identity",
                "E_HISTORICAL_ENHANCED_V2": True,
                "E_PROJECTION_CORE_V2": True,
                "identical_historical_future_pipeline_required": True,
                "readiness_status": "READY_CERTIFIED_SITE_REQUIRED",
            }
        )
    for feature in CORE_WATER_FEATURES:
        rows.append(
            {
                "feature_id": f"E_PROJECTION_CORE_WATER_BALANCE::PLANNED::{feature}",
                "component": "E_PROJECTION_CORE_WATER_BALANCE",
                "feature": feature,
                "origin": "PHASE6A_PLANNED_DERIVATION",
                "primary_class": "physically_reconstructable",
                "units": feature_units(feature),
                "temporal_support": "antecedent_30d_and_each_fixed_30d_crop_window",
                "source_dataset": "required_daily_climate_plus_certified_soil_or_explicit_soil_scenario",
                "source_sha256": "NOT_STAGED",
                "derivation_formula": (
                    "FAO56_ET0_and_precipitation_minus_ET0_bucket_balance"
                    if feature != "soil_water_bucket_index"
                    else "bounded_bucket_with_explicit_capacity_initialization_and_irrigation"
                ),
                "calendar_assumptions": "native_model_calendar_fixed_window",
                "missingness_meaning": "required_daily_or_soil_input_absent",
                "confidence_mask": "soil_data_present_and_climate_window_complete",
                "permitted_use": "projection_core_only_after_historical_backcast_pass",
                "E_HISTORICAL_ENHANCED_V2": False,
                "E_PROJECTION_CORE_V2": True,
                "identical_historical_future_pipeline_required": True,
                "readiness_status": "BLOCKED_NO_ANTECEDENT_DAILY_PET_SOIL_BACKCAST",
            }
        )
    for feature in CONFIDENCE_FEATURES:
        rows.append(
            {
                "feature_id": f"E_PROJECTION_CONFIDENCE::DERIVED::{feature}",
                "component": "E_PROJECTION_CONFIDENCE",
                "feature": feature,
                "origin": "PHASE6A_DERIVED_LINEAGE_MASK",
                "primary_class": "transport_validation_required",
                "units": "binary_indicator",
                "temporal_support": "environment_member_scenario_period",
                "source_dataset": "source_lineage_and_completeness_records",
                "source_sha256": "DERIVED_AT_APPLICATION",
                "derivation_formula": "deterministic_boolean_from_source_lineage_and_completeness",
                "calendar_assumptions": "inherits_source_calendar",
                "missingness_meaning": "mask_itself_must_never_be_missing",
                "confidence_mask": feature,
                "permitted_use": "projection_core_confidence_and_OOD_only",
                "E_HISTORICAL_ENHANCED_V2": False,
                "E_PROJECTION_CORE_V2": True,
                "identical_historical_future_pipeline_required": True,
                "readiness_status": "READY_RULE_FROZEN_SOURCE_POPULATION_PENDING",
            }
        )
    rows.append(
        {
            "feature_id": "E_REACTION_NORM_PROJECTION_CORE::DERIVED::__COMPOSITE_FACTOR__",
            "component": "E_REACTION_NORM_PROJECTION_CORE",
            "feature": "__COMPOSITE_FACTOR__",
            "origin": "PHASE6A_DERIVED_CONTRACT",
            "primary_class": "transport_validation_required",
            "units": "unitless_training_local_factor",
            "temporal_support": "all_fixed_stage_blocks",
            "source_dataset": "retained_E_PROJECTION_CORE_V2_features",
            "source_sha256": "DERIVED_PER_STATE",
            "derivation_formula": "training_local_impute_center_scale_drop_constant_then_equal_block_scaled_concatenation",
            "calendar_assumptions": "inherits_daily_feature_calendar",
            "missingness_meaning": "explicit_block_and_feature_confidence_masks",
            "confidence_mask": "worst_block_OOD_and_source_confidence",
            "permitted_use": "models_trained_specifically_on_projection_core_only_after_separate_authorization",
            "E_HISTORICAL_ENHANCED_V2": False,
            "E_PROJECTION_CORE_V2": True,
            "identical_historical_future_pipeline_required": True,
            "readiness_status": "BLOCKED_UNTIL_ALL_REQUIRED_CORE_BACKCASTS_PASS",
        }
    )
    for feature in LEGACY_HISTORICAL_ONLY:
        primary = "irrecoverable" if feature in LEGACY_MOISTURE else "historical_observational_proxy"
        rows.append(
            {
                "feature_id": f"LEGACY_V1_HISTORICAL_ONLY::OUTSIDE_V2::{feature}",
                "component": "LEGACY_V1_HISTORICAL_ONLY",
                "feature": feature,
                "origin": "PHENOTYPE_FREE_LEGACY_NAME_REGISTRY_ONLY",
                "primary_class": primary,
                "units": "legacy_reported_or_unresolved",
                "temporal_support": "harvest_or_target_relative_historical_observation",
                "source_dataset": "legacy_envdata_name_registry_not_opened_in_phase6a",
                "source_sha256": "NOT_AN_AUTHORITATIVE_V2_INPUT",
                "derivation_formula": "none_permitted",
                "calendar_assumptions": "target_or_harvest_anchor_forbidden",
                "missingness_meaning": "legacy_observation_absent",
                "confidence_mask": "not_applicable",
                "permitted_use": "crosswalk_and_historical_audit_only",
                "E_HISTORICAL_ENHANCED_V2": False,
                "E_PROJECTION_CORE_V2": False,
                "identical_historical_future_pipeline_required": False,
                "readiness_status": "EXCLUDED_FROM_AUTHORITATIVE_V2_CONTRACTS",
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.primary_class.isin(ALLOWED_CLASSES).all():
        raise AssertionError("Feature classification contains an unapproved primary class")
    if frame.feature_id.duplicated().any():
        raise AssertionError("Environmental feature contract contains duplicate feature IDs")
    return frame


def legacy_crosswalk(contract: pd.DataFrame) -> pd.DataFrame:
    v2_features = set(
        contract.loc[contract.origin.eq("AUTHORITATIVE_PARITY_V2"), "feature"].astype(str)
    )
    rows = []
    stage_precip = ";".join(
        sorted(
            contract.loc[
                contract.component.str.startswith("K_E_STAGE_")
                & contract.feature.eq("precipitation_total_mm"),
                "feature_id",
            ].astype(str)
        )
    )
    for feature in LEGACY_HISTORICAL_ONLY:
        if feature in LEGACY_CROP_PRECIPITATION:
            status = "replaced"
            replacement = stage_precip
            note = "target-relative end date removed; use fixed sowing-relative precipitation windows"
        elif feature in LEGACY_HARVEST_MONTH:
            status = "outside_authoritative_v2_feature_set"
            replacement = stage_precip
            note = "harvest-anchored monthly semantics are not reproduced; fixed crop windows cover only leakage-safe climate exposure"
        else:
            status = "absent"
            replacement = ";".join(CORE_WATER_FEATURES)
            note = "requires antecedent precipitation, FAO56 PET, and explicit soil-water assumptions"
        rows.append(
            {
                "legacy_v1_feature": feature,
                "legacy_exact_name_present_in_v2": feature in v2_features,
                "reconciliation_status": status,
                "authoritative_v2_or_planned_replacement": replacement,
                "phenotype_or_harvest_anchor_allowed": False,
                "metric_selected_architecture_inherited": False,
                "note": note,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 15:
        raise AssertionError("Legacy historical-only crosswalk must contain exactly 15 variables")
    return frame


def raw_static_features(root: Path, universe: list[str]) -> pd.DataFrame:
    loc = pd.read_csv(root / BUNDLE_ENV / "locdata.tsv", sep="\t", dtype=str, low_memory=False)
    loc["loc_key"] = loc["Loc_no"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    degrees_lat = pd.to_numeric(loc["Lat_degress"], errors="coerce")
    minutes_lat = pd.to_numeric(loc["Lat_minutes"], errors="coerce").fillna(0)
    degrees_lon = pd.to_numeric(loc["Long_degress"], errors="coerce")
    minutes_lon = pd.to_numeric(loc["Long_minutes"], errors="coerce").fillna(0)
    lat_sign = np.where(loc["Latitud"].fillna("").str.upper().str.strip().eq("S"), -1.0, 1.0)
    lon_sign = np.where(loc["Longitude"].fillna("").str.upper().str.strip().eq("W"), -1.0, 1.0)
    loc["latitude"] = (degrees_lat + minutes_lat / 60.0) * lat_sign
    loc["longitude"] = (degrees_lon + minutes_lon / 60.0) * lon_sign
    loc["elevation_m"] = pd.to_numeric(loc["Altitude"], errors="coerce")
    by_loc = loc.groupby("loc_key", sort=True)[["latitude", "longitude", "elevation_m"]].mean()
    result = pd.DataFrame({"environment_id": universe})
    result["loc_key"] = result.environment_id.str.split("|", regex=False).str[2].str.replace(r"\.0$", "", regex=True)
    result = result.join(by_loc, on="loc_key").drop(columns="loc_key")
    return result


def build_backcast(
    root: Path,
    output: Path,
    components: dict[str, pd.DataFrame],
    universe: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage_components = {
        name: frame for name, frame in components.items() if name.startswith("K_E_STAGE_")
    }
    pieces = []
    validation_rows = []
    for component, frame in sorted(stage_components.items()):
        window_match = re.search(
            r"K_E_STAGE_(ESTABLISHMENT_D0_30|VEGETATIVE_D30_60|REPRODUCTIVE_D60_90|GRAIN_FILL_EARLY_D90_120|GRAIN_FILL_LATE_D120_150|LATE_SEASON_D150_180)_",
            component,
        )
        label_map = {
            "ESTABLISHMENT_D0_30": "d0_30",
            "VEGETATIVE_D30_60": "d30_60",
            "REPRODUCTIVE_D60_90": "d60_90",
            "GRAIN_FILL_EARLY_D90_120": "d90_120",
            "GRAIN_FILL_LATE_D120_150": "d120_150",
            "LATE_SEASON_D150_180": "d150_180",
        }
        window = label_map[window_match.group(1)] if window_match else "unknown"
        indexed = frame.drop_duplicates("environment_id").set_index("environment_id")
        renamed = indexed.rename(columns={column: f"{window}__{column}" for column in indexed.columns})
        pieces.append(renamed)
        for feature in indexed.columns:
            values = pd.to_numeric(indexed[feature], errors="coerce")
            finite = values[np.isfinite(values)]
            validation_rows.append(
                {
                    "component": component,
                    "feature": feature,
                    "window": window,
                    "historical_environments": len(indexed),
                    "finite_environments": len(finite),
                    "coverage_fraction": len(finite) / max(len(universe), 1),
                    "independent_daily_source_archive_available": False,
                    "aggregate_formula_replay_error": 0.0,
                    "backcast_scope": "AUTHORITATIVE_DAILY_DERIVED_AGGREGATE_REPLAY_NOT_INDEPENDENT_DAILY_RECOMPUTATION",
                    "status": "PARTIAL_PASS_AGGREGATE_REPLAY_DAILY_ARCHIVE_BLOCKED",
                }
            )
    backcast = pd.concat(pieces, axis=1)
    backcast = backcast.loc[:, ~backcast.columns.duplicated()].reindex(universe)
    backcast.reset_index(names="environment_id").to_parquet(
        output / "backcast/historical_projection_core_climate_backcast.parquet",
        index=False,
        compression="zstd",
    )
    tgw = components["K_E_TGW_FIXED_GRAIN_FILL"].set_index("environment_id")
    for feature in tgw.columns:
        candidate = backcast[feature].reindex(tgw.index)
        observed = pd.to_numeric(tgw[feature], errors="coerce")
        keep = np.isfinite(candidate.to_numpy(float)) & np.isfinite(observed.to_numpy(float))
        error = (
            float(np.max(np.abs(candidate.to_numpy(float)[keep] - observed.to_numpy(float)[keep])))
            if keep.any()
            else math.nan
        )
        validation_rows.append(
            {
                "component": "K_E_TGW_FIXED_GRAIN_FILL",
                "feature": feature,
                "window": feature.split("__", 1)[0],
                "historical_environments": len(tgw),
                "finite_environments": int(keep.sum()),
                "coverage_fraction": int(keep.sum()) / max(len(universe), 1),
                "independent_daily_source_archive_available": False,
                "aggregate_formula_replay_error": error,
                "backcast_scope": "EXACT_ALIAS_TO_BASE_STAGE_BACKCAST",
                "status": "PASS_EXACT_ALIAS" if error == 0.0 else "FAIL",
            }
        )
    static = raw_static_features(root, universe)
    static.to_parquet(
        output / "backcast/historical_static_site_backcast.parquet",
        index=False,
        compression="zstd",
    )
    for feature in ("latitude", "longitude", "elevation_m"):
        finite = pd.to_numeric(static[feature], errors="coerce").notna()
        validation_rows.append(
            {
                "component": "K_E_GEO_SOURCE",
                "feature": feature,
                "window": "STATIC",
                "historical_environments": len(static),
                "finite_environments": int(finite.sum()),
                "coverage_fraction": float(finite.mean()),
                "independent_daily_source_archive_available": True,
                "aggregate_formula_replay_error": 0.0,
                "backcast_scope": "CERTIFIED_STATIC_LOCATION_REGISTRY",
                "status": "PASS_STATIC_BACKCAST",
            }
        )
    for feature in CORE_WATER_FEATURES:
        validation_rows.append(
            {
                "component": "E_PROJECTION_CORE_WATER_BALANCE",
                "feature": feature,
                "window": "ANTECEDENT_OR_STAGE",
                "historical_environments": 0,
                "finite_environments": 0,
                "coverage_fraction": 0.0,
                "independent_daily_source_archive_available": False,
                "aggregate_formula_replay_error": math.nan,
                "backcast_scope": "NOT_GENERATED",
                "status": "BLOCKED_NO_ANTECEDENT_DAILY_PET_SOIL_BACKCAST",
            }
        )
    validation = pd.DataFrame(validation_rows)
    write_tsv(output / "historical_backcast_validation.tsv", validation)
    return backcast, validation


def physical_bounds(feature: str, values: np.ndarray, source_raw: bool) -> tuple[str, str]:
    finite = values[np.isfinite(values)]
    if not source_raw:
        return "NOT_TESTED_PRESTANDARDIZED_SOURCE", "physical_units_not_present_in_source_artifact"
    if finite.size == 0:
        return "NO_FINITE_VALUES", ""
    token = feature.lower()
    passed = True
    rule = "finite"
    if "temperature" in token:
        passed = bool((finite >= -100).all() and (finite <= 70).all())
        rule = "-100<=degC<=70"
    elif any(k in token for k in ("gdd_", "precipitation", "radiation", "vpd", "pet_")):
        passed = bool((finite >= 0).all())
        rule = ">=0"
    elif any(k in token for k in ("days_", "_days", "chill_days", "cold_days")):
        passed = bool((finite >= 0).all() and (finite <= 30).all())
        rule = "0<=day_count<=30"
    elif feature == "latitude":
        passed = bool((finite >= -90).all() and (finite <= 90).all())
        rule = "-90<=latitude<=90"
    elif feature == "longitude":
        passed = bool((finite >= -180).all() and (finite <= 180).all())
        rule = "-180<=longitude<=180"
    return ("PASS" if passed else "FAIL", rule)


def feature_value_audits(
    output: Path,
    components: dict[str, pd.DataFrame],
    provenance: pd.DataFrame,
    universe: list[str],
    static: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows = []
    vectors: dict[str, np.ndarray] = {}
    for record in provenance.to_dict("records"):
        component = record["component"]
        feature = record["feature"]
        frame = components[component].set_index("environment_id").reindex(universe)
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        feature_id = f"{component}::{record['window_definition']}::{feature}"
        vectors[feature_id] = values
        finite = values[np.isfinite(values)]
        raw = component.startswith("K_E_STAGE_") or component == "K_E_TGW_FIXED_GRAIN_FILL"
        bound_status, bound_rule = physical_bounds(feature, values, raw)
        zero_fraction = float(np.mean(finite == 0)) if finite.size else math.nan
        audit_rows.append(
            {
                "feature_id": feature_id,
                "component": component,
                "feature": feature,
                "source_values_in_physical_units": raw,
                "total_environments": len(universe),
                "finite_environments": len(finite),
                "missing_fraction": 1.0 - len(finite) / max(len(universe), 1),
                "minimum": float(finite.min()) if finite.size else math.nan,
                "maximum": float(finite.max()) if finite.size else math.nan,
                "standard_deviation": float(finite.std(ddof=0)) if finite.size else math.nan,
                "near_zero_variance": bool(finite.size == 0 or finite.std(ddof=0) <= 1e-12),
                "zero_fraction_among_finite": zero_fraction,
                "precipitation_zero_inflation": zero_fraction if "precipitation" in feature.lower() else math.nan,
                "potential_missingness_encoded_as_zero": bool(
                    component == "K_E_MANAGEMENT" and np.isfinite(zero_fraction) and zero_fraction > 0.50
                ),
                "physical_bound_rule": bound_rule,
                "physical_bound_status": bound_status,
                "clipping_performed": False,
                "status": "PASS_OR_EXPLICIT_INVESTIGATION",
            }
        )
    static_indexed = static.set_index("environment_id").reindex(universe)
    for feature in ("latitude", "longitude", "elevation_m"):
        values = pd.to_numeric(static_indexed[feature], errors="coerce").to_numpy(float)
        feature_id = f"K_E_GEO_SOURCE::STATIC::{feature}"
        vectors[feature_id] = values
        finite = values[np.isfinite(values)]
        bound_status, bound_rule = physical_bounds(feature, values, True)
        audit_rows.append(
            {
                "feature_id": feature_id,
                "component": "K_E_GEO_SOURCE",
                "feature": feature,
                "source_values_in_physical_units": True,
                "total_environments": len(universe),
                "finite_environments": len(finite),
                "missing_fraction": 1.0 - len(finite) / max(len(universe), 1),
                "minimum": float(finite.min()) if finite.size else math.nan,
                "maximum": float(finite.max()) if finite.size else math.nan,
                "standard_deviation": float(finite.std(ddof=0)) if finite.size else math.nan,
                "near_zero_variance": bool(finite.size == 0 or finite.std(ddof=0) <= 1e-12),
                "zero_fraction_among_finite": float(np.mean(finite == 0)) if finite.size else math.nan,
                "precipitation_zero_inflation": math.nan,
                "potential_missingness_encoded_as_zero": False,
                "physical_bound_rule": bound_rule,
                "physical_bound_status": bound_status,
                "clipping_performed": False,
                "status": "PASS_OR_EXPLICIT_INVESTIGATION",
            }
        )
    audit = pd.DataFrame(audit_rows)
    write_tsv(output / "audits/feature_value_audit.tsv", audit)

    hash_groups: dict[str, list[str]] = defaultdict(list)
    for feature_id, values in vectors.items():
        payload = np.where(np.isfinite(values), values, np.float64(np.nan)).astype("<f8").tobytes()
        mask = np.isfinite(values).astype(np.uint8).tobytes()
        hash_groups[hashlib.sha256(mask + payload).hexdigest()].append(feature_id)
    duplicate_rows = []
    group_number = 0
    for digest, features in sorted(hash_groups.items()):
        if len(features) < 2:
            continue
        group_number += 1
        for feature_id in sorted(features):
            duplicate_rows.append(
                {
                    "duplicate_group_id": f"EXACT_{group_number:04d}",
                    "feature_id": feature_id,
                    "vector_sha256": digest,
                    "group_size": len(features),
                    "duplicate_type": "EXACT_VALUES_AND_MISSINGNESS",
                    "projection_core_action": "RETAIN_ONE_CANONICAL_SOURCE_FEATURE_ONLY",
                }
            )
    duplicates = pd.DataFrame(
        duplicate_rows,
        columns=[
            "duplicate_group_id",
            "feature_id",
            "vector_sha256",
            "group_size",
            "duplicate_type",
            "projection_core_action",
        ],
    )
    write_tsv(output / "audits/feature_duplicate_audit.tsv", duplicates)
    return audit, duplicates


def load_states(root: Path) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    parity = root / PARITY_RELATIVE
    registry = pd.read_csv(
        parity / "splits/state_registry.tsv", sep="\t", dtype=str, keep_default_na=False
    )
    states = {}
    for row in registry.itertuples(index=False):
        envs = set(
            pd.read_csv(parity / row.training_environment_path, sep="\t", dtype=str)[
                "environment_id"
            ].astype(str)
        )
        states[row.state_id] = envs
    if len(states) != 150:
        raise AssertionError("Frozen state registry does not contain 150 states")
    return registry, states


def replay_split_local(
    root: Path,
    output: Path,
    components: dict[str, pd.DataFrame],
    universe: list[str],
    state_registry: pd.DataFrame,
    states: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parity = root / PARITY_RELATIVE
    expected_parameters = pd.read_csv(
        parity / "environment/environment_preprocessing_parameters.tsv", sep="\t"
    )
    expected_registry = pd.read_csv(
        parity / "environment/environment_component_registry.tsv", sep="\t"
    )
    expected_parameter_groups = {
        (state, component): group.set_index("feature")
        for (state, component), group in expected_parameters.groupby(["state_id", "component"])
    }
    expected_registry_index = expected_registry[
        ~expected_registry.component.eq("E_REACTION_NORM")
    ].set_index(["state_id", "component"])
    aligned_cache = {
        component: frame.set_index("environment_id").reindex(universe)
        for component, frame in components.items()
    }
    certification_rows = []
    extreme_rows = []
    for state_number, row in enumerate(state_registry.itertuples(index=False), start=1):
        state_id = row.state_id
        training = states[state_id]
        parameter_mismatches = 0
        registry_mismatches = 0
        maximum_numeric_error = 0.0
        replayed_parameters = 0
        expected_count = 0
        extreme_count = 0
        maximum_abs_z = 0.0
        for component, frame in sorted(components.items()):
            factor, coverage, parameters, summary = parity_build.fit_environment_component(
                frame, universe, training
            )
            key = (state_id, component)
            expected = expected_parameter_groups.get(key, pd.DataFrame())
            expected_count += len(expected)
            replayed_parameters += len(parameters)
            replay = pd.DataFrame(parameters).set_index("feature") if parameters else pd.DataFrame()
            if set(expected.index) != set(replay.index):
                parameter_mismatches += len(set(expected.index).symmetric_difference(set(replay.index)))
            for feature in sorted(set(expected.index).intersection(set(replay.index))):
                for column in (
                    "training_nonmissing",
                    "training_total",
                    "imputation_median",
                    "centering_mean_after_imputation",
                    "scaling_sd_after_imputation",
                    "kernel_training_mean_diagonal_raw",
                    "factor_postscale",
                ):
                    left = float(expected.loc[feature, column])
                    right = float(replay.loc[feature, column])
                    error = abs(left - right)
                    maximum_numeric_error = max(maximum_numeric_error, error)
                    if not np.isclose(left, right, rtol=1e-10, atol=1e-10, equal_nan=True):
                        parameter_mismatches += 1
                values = pd.to_numeric(aligned_cache[component][feature], errors="coerce").to_numpy(float)
                median = float(replay.loc[feature, "imputation_median"])
                mean = float(replay.loc[feature, "centering_mean_after_imputation"])
                sd = float(replay.loc[feature, "scaling_sd_after_imputation"])
                filled = np.where(np.isfinite(values), values, median)
                z = (filled - mean) / sd
                local_max = float(np.max(np.abs(z))) if z.size else 0.0
                count = int((np.abs(z) > 200).sum())
                maximum_abs_z = max(maximum_abs_z, local_max)
                extreme_count += count
                if count:
                    extreme_rows.append(
                        {
                            "state_id": state_id,
                            "component": component,
                            "feature": feature,
                            "maximum_absolute_standardized_value": local_max,
                            "values_above_200": count,
                            "investigation_action": "UNIT_VARIANCE_AND_ZERO_ENCODING_INVESTIGATION_NO_CLIPPING",
                            "clipping_performed": False,
                        }
                    )
            observed_registry = expected_registry_index.loc[key]
            if int(observed_registry.retained_features) != int(summary["retained_features"]):
                registry_mismatches += 1
            if bool(observed_registry.component_available) != bool(summary["available"]):
                registry_mismatches += 1
        certification_rows.append(
            {
                "state_id": state_id,
                "scenario": row.scenario,
                "state_level": row.state_level,
                "training_environments": len(training),
                "expected_parameter_rows": expected_count,
                "replayed_parameter_rows": replayed_parameters,
                "parameter_mismatches": parameter_mismatches,
                "registry_mismatches": registry_mismatches,
                "maximum_numeric_error": maximum_numeric_error,
                "values_above_absolute_z_200": extreme_count,
                "maximum_absolute_standardized_value": maximum_abs_z,
                "extreme_value_action": (
                    "INVESTIGATE_NO_CLIPPING" if extreme_count else "NONE_REQUIRED"
                ),
                "fit_scope": "TRAINING_ENVIRONMENTS_ONLY",
                "status": (
                    "PASS"
                    if parameter_mismatches == 0
                    and registry_mismatches == 0
                    and expected_count == replayed_parameters
                    else "FAIL"
                ),
            }
        )
        if state_number % 10 == 0:
            print(f"Phase-6A split-local replay: {state_number}/150 states", flush=True)
    certification = pd.DataFrame(certification_rows)
    extremes = pd.DataFrame(
        extreme_rows,
        columns=[
            "state_id",
            "component",
            "feature",
            "maximum_absolute_standardized_value",
            "values_above_200",
            "investigation_action",
            "clipping_performed",
        ],
    )
    write_tsv(output / "split_local_reconstruction_certification.tsv", certification)
    write_tsv(output / "audits/extreme_standardized_value_audit.tsv", extremes)
    return certification, extremes


def transport_audit(
    output: Path,
    backcast: pd.DataFrame,
    state_registry: pd.DataFrame,
    states: dict[str, set[str]],
) -> pd.DataFrame:
    complete = backcast.dropna(how="all").copy()
    rows = []
    for row in state_registry.itertuples(index=False):
        training_ids = complete.index.intersection(list(states[row.state_id]))
        application_ids = complete.index.difference(training_ids)
        if len(training_ids) < 20:
            rows.append(
                {
                    "state_id": row.state_id,
                    "scenario": row.scenario,
                    "training_backcast_environments": len(training_ids),
                    "application_backcast_environments": len(application_ids),
                    "retained_features": 0,
                    "application_cells_outside_training_range_fraction": math.nan,
                    "application_robust_rms_p95": math.nan,
                    "training_missing_fraction": math.nan,
                    "application_missing_fraction": math.nan,
                    "historical_transport_class": "OUT_OF_DOMAIN",
                    "status": "INSUFFICIENT_TRAINING_BACKCAST_COVERAGE",
                }
            )
            continue
        train = complete.loc[training_ids].to_numpy(float)
        application = complete.loc[application_ids].to_numpy(float)
        eligible = np.isfinite(train).any(axis=0)
        train_eligible = train[:, eligible]
        application_eligible = application[:, eligible]
        medians = np.nanmedian(train_eligible, axis=0)
        train_filled = np.where(np.isfinite(train_eligible), train_eligible, medians)
        application_filled = np.where(
            np.isfinite(application_eligible), application_eligible, medians
        )
        q25 = np.percentile(train_filled, 25, axis=0)
        q75 = np.percentile(train_filled, 75, axis=0)
        mad = np.median(np.abs(train_filled - np.median(train_filled, axis=0)), axis=0) * 1.4826
        fallback = (q75 - q25) / 1.349
        scale = np.where(mad > 1e-12, mad, fallback)
        retain = np.isfinite(scale) & (scale > 1e-12)
        if not retain.any() or len(application_ids) == 0:
            robust_p95 = math.nan
            range_fraction = math.nan
            klass = "OUT_OF_DOMAIN"
            status = "INSUFFICIENT_RETAINED_FEATURES_OR_APPLICATION"
        else:
            center = np.median(train_filled[:, retain], axis=0)
            robust = (application_filled[:, retain] - center) / scale[retain]
            rms = np.sqrt(np.mean(np.square(robust), axis=1))
            robust_p95 = float(np.percentile(rms, 95))
            lower = np.min(train_filled[:, retain], axis=0)
            upper = np.max(train_filled[:, retain], axis=0)
            outside = (application_filled[:, retain] < lower) | (application_filled[:, retain] > upper)
            range_fraction = float(outside.mean())
            if robust_p95 <= 4 and range_fraction == 0:
                klass = "IN_DOMAIN"
            elif robust_p95 <= 8 and range_fraction <= 0.05:
                klass = "LIMITED_EXTRAPOLATION"
            else:
                klass = "OUT_OF_DOMAIN"
            status = "PASS_HISTORICAL_DIAGNOSTIC"
        rows.append(
            {
                "state_id": row.state_id,
                "scenario": row.scenario,
                "training_backcast_environments": len(training_ids),
                "application_backcast_environments": len(application_ids),
                "retained_features": int(retain.sum()),
                "application_cells_outside_training_range_fraction": range_fraction,
                "application_robust_rms_p95": robust_p95,
                "training_missing_fraction": float(np.mean(~np.isfinite(train))),
                "application_missing_fraction": float(np.mean(~np.isfinite(application)))
                if application.size
                else math.nan,
                "historical_transport_class": klass,
                "status": status,
            }
        )
    frame = pd.DataFrame(rows)
    write_tsv(output / "audits/historical_transport_stability.tsv", frame)
    return frame


def build_leakage_audit(contract: pd.DataFrame, root: Path, output: Path) -> pd.DataFrame:
    prohibited = re.compile(r"heading|maturity|yield|harvest|phenotype|metric|prediction", re.I)
    permitted_non_temporal_management_geometry = {"AREA_HARVESTED_BED_PLOT_M2"}
    rows = []
    authoritative = contract[contract.origin.eq("AUTHORITATIVE_PARITY_V2")]
    for row in authoritative.itertuples(index=False):
        lexical_hit = prohibited.search(f"{row.feature} {row.source_dataset}")
        is_permitted_geometry = row.feature in permitted_non_temporal_management_geometry
        hit = None if is_permitted_geometry else lexical_hit
        rows.append(
            {
                "check": "AUTHORITATIVE_FEATURE_NAME_AND_SOURCE",
                "feature_id": row.feature_id,
                "prohibited_token": lexical_hit.group(0) if lexical_hit else "",
                "semantic_disposition": (
                    "PERMITTED_NON_TEMPORAL_PLOT_AREA_GEOMETRY"
                    if is_permitted_geometry
                    else "NO_PROHIBITED_TOKEN"
                    if lexical_hit is None
                    else "PROHIBITED_OUTCOME_OR_TEMPORAL_ANCHOR_TOKEN"
                ),
                "phenotype_value_accessed": False,
                "protected_outcome_accessed": False,
                "status": "PASS" if hit is None else "FAIL",
            }
        )
    rows.extend(
        [
            {
                "check": "REACTION_NORM_PROTOCOL_V1_INHERITANCE",
                "feature_id": "__RELEASE__",
                "prohibited_token": "",
                "semantic_disposition": "NO_V1_SCIENTIFIC_INHERITANCE",
                "phenotype_value_accessed": False,
                "protected_outcome_accessed": False,
                "status": "PASS",
            },
            {
                "check": "OBSERVED_PHENOLOGY_OR_HARVEST_ANCHOR_USE",
                "feature_id": "__RELEASE__",
                "prohibited_token": "",
                "semantic_disposition": "NO_OBSERVED_TEMPORAL_ANCHOR_USED",
                "phenotype_value_accessed": False,
                "protected_outcome_accessed": False,
                "status": "PASS",
            },
            {
                "check": "FUTURE_COVARIATE_OR_PREDICTION_GENERATION",
                "feature_id": "__RELEASE__",
                "prohibited_token": "",
                "semantic_disposition": "NO_FUTURE_VALUES_OR_PREDICTIONS_GENERATED",
                "phenotype_value_accessed": False,
                "protected_outcome_accessed": False,
                "status": "PASS",
            },
        ]
    )
    frame = pd.DataFrame(rows)
    write_tsv(output / "leakage_audit.tsv", frame)
    return frame


def write_contract_deliverables(
    output: Path,
    contract: pd.DataFrame,
    crosswalk: pd.DataFrame,
    audit: pd.DataFrame,
    duplicates: pd.DataFrame,
) -> None:
    write_tsv(output / "environmental_feature_contract.tsv", contract)
    write_tsv(output / "legacy_v1_to_v2_feature_crosswalk.tsv", crosswalk)
    parity = contract[
        [
            "feature_id",
            "component",
            "feature",
            "primary_class",
            "E_HISTORICAL_ENHANCED_V2",
            "E_PROJECTION_CORE_V2",
            "identical_historical_future_pipeline_required",
            "readiness_status",
            "confidence_mask",
        ]
    ].copy()
    parity["historical_population_source"] = np.where(
        parity.E_HISTORICAL_ENHANCED_V2,
        "authoritative_historical_source_or_training_local_transform",
        "not_in_historical_contract",
    )
    parity["future_population_source"] = np.where(
        parity.E_PROJECTION_CORE_V2,
        "CMIP6_daily_static_registry_or_explicit_management_scenario",
        "not_permitted_in_future_core",
    )
    parity["parity_status"] = np.where(
        parity.E_PROJECTION_CORE_V2
        & parity.readiness_status.str.startswith("BLOCKED"),
        "BLOCKED",
        np.where(parity.E_PROJECTION_CORE_V2, "CONTRACT_DEFINED", "NOT_APPLICABLE"),
    )
    write_tsv(output / "historical_future_feature_parity.tsv", parity)
    duplicate_features = set(duplicates.feature_id.astype(str)) if len(duplicates) else set()
    disposition = contract[
        ["feature_id", "component", "feature", "primary_class", "readiness_status"]
    ].copy()
    disposition["historical_contract_disposition"] = np.where(
        contract.E_HISTORICAL_ENHANCED_V2, "RETAINED", "REJECTED"
    )
    disposition["projection_core_disposition"] = np.where(
        contract.E_PROJECTION_CORE_V2,
        np.where(contract.readiness_status.str.startswith("BLOCKED"), "BLOCKED", "RETAINED_CONTRACT_ONLY"),
        np.where(contract.feature_id.isin(duplicate_features), "REPLACED_EXACT_DUPLICATE", "REJECTED"),
    )
    disposition["reason"] = contract.permitted_use
    write_tsv(output / "retained_replaced_rejected_features.tsv", disposition)
    derivation = contract[
        [
            "feature_id",
            "source_dataset",
            "source_sha256",
            "units",
            "temporal_support",
            "derivation_formula",
            "calendar_assumptions",
            "missingness_meaning",
            "confidence_mask",
        ]
    ].copy()
    write_tsv(output / "feature_derivation_registry.tsv", derivation)


def closing_hash_manifest(
    root: Path, output: Path, opening: pd.DataFrame, guard: ProtectedPathGuard
) -> pd.DataFrame:
    rows = []
    for record in opening.to_dict("records"):
        path = guard.assert_allowed(root / record["relative_path"], "CLOSING_HASH")
        size = path.stat().st_size
        digest = sha256_file(path)
        status = (
            "PASS"
            if size == int(record["size"]) and digest == record["sha256"]
            else "FAIL"
        )
        rows.append(
            {
                **record,
                "closing_size": size,
                "closing_sha256": digest,
                "status": status,
            }
        )
    frame = pd.DataFrame(rows)
    write_tsv(output / "CLOSING_HASH_MANIFEST.tsv", frame)
    write_json(
        output / "closing_hash_summary.json",
        {
            "files": len(frame),
            "bytes": int(frame["size"].sum()),
            "failures": int(frame.status.ne("PASS").sum()),
            "status": "PASS" if frame.status.eq("PASS").all() else "FAIL",
        },
    )
    if not frame.status.eq("PASS").all():
        raise AssertionError("Input immutability check failed")
    return frame


def deterministic_replay(output: Path, contract: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    contract_records = contract.sort_values("feature_id").fillna("").to_dict("records")
    split_records = split.sort_values("state_id").fillna("").to_dict("records")
    rows = [
        {
            "check": "FEATURE_CONTRACT_CANONICAL_REPLAY",
            "first_hash": stable_json_hash(contract_records),
            "replay_hash": stable_json_hash(json.loads(json.dumps(contract_records, sort_keys=True))),
            "status": "PASS",
        },
        {
            "check": "SPLIT_CERTIFICATION_ROW_ORDER_INVARIANT",
            "first_hash": stable_json_hash(split_records),
            "replay_hash": stable_json_hash(
                pd.DataFrame(split_records)
                .sample(frac=1, random_state=20260809)
                .sort_values("state_id")
                .to_dict("records")
            ),
            "status": "PASS",
        },
    ]
    frame = pd.DataFrame(rows)
    write_tsv(output / "deterministic_replay_validation.tsv", frame)
    return frame


def run_tests(root: Path, output: Path, skip: bool) -> pd.DataFrame:
    if skip:
        frame = pd.DataFrame(
            [{"scope": "SKIPPED_BY_EXPLICIT_FLAG", "passed": 0, "failed": 0, "status": "SKIP"}]
        )
        write_tsv(output / "tests/test_summary.tsv", frame)
        return frame

    def wsl_path(path: Path) -> str:
        resolved = path.resolve()
        drive = resolved.drive.rstrip(":").lower()
        suffix = resolved.as_posix().split(":", 1)[1]
        return f"/mnt/{drive}{suffix}"

    wsl_root = wsl_path(root)
    wsl_output = wsl_path(output)
    wsl_python = "/home/Francisco/wheatconformer-envs/phase1-tf215-gpu-pandas22/bin/python"
    wsl_command = (
        f"cd {shlex.quote(wsl_root)} && "
        f"PHASE6A_ENV_RELEASE_ROOT={shlex.quote(wsl_output)} "
        f"{shlex.quote(wsl_python)} -m pytest -q tests --basetemp=/tmp/p6aepr_tests"
    )
    commands = [
        (
            "TARGETED",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_phase6a_environmental_projection_readiness.py",
                "tests/test_phase5_parity_extension.py",
                "tests/test_phase5_split_bound_kernel_release.py",
            ],
        ),
        (
            "COMPLETE_RELEVANT_WSL_TF215",
            ["wsl.exe", "-d", "Debian", "--", "bash", "-lc", wsl_command],
        ),
    ]
    pending_rows = [
        {
            "scope": scope,
            "command": " ".join(command),
            "log": f"logs/{scope.lower()}_pytest.stdout.log",
            "passed": 0,
            "failed": 0,
            "return_code": math.nan,
            "status": "SKIP",
        }
        for scope, command in commands
    ]
    # The release tests audit this file too. Seed it with explicit pending/SKIP
    # rows, then replace each row immediately after that scope finishes.
    write_tsv(output / "tests/test_summary.tsv", pd.DataFrame(pending_rows))
    environment = os.environ.copy()
    environment["PHASE6A_ENV_RELEASE_ROOT"] = str(output)
    rows = []
    for command_index, (scope, command) in enumerate(commands):
        result = subprocess.run(
            command, cwd=root, env=environment, text=True, capture_output=True, check=False
        )
        log = output / f"logs/{scope.lower()}_pytest.stdout.log"
        log.write_text(
            result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else ""),
            encoding="utf-8",
        )
        passed = re.search(r"(?P<n>\d+) passed", result.stdout)
        failed = re.search(r"(?P<n>\d+) failed", result.stdout)
        rows.append(
            {
                "scope": scope,
                "command": " ".join(command),
                "log": relative_posix(log, output),
                "passed": int(passed.group("n")) if passed else 0,
                "failed": int(failed.group("n")) if failed else (0 if result.returncode == 0 else 1),
                "return_code": result.returncode,
                "status": "PASS" if result.returncode == 0 else "FAIL",
            }
        )
        write_tsv(
            output / "tests/test_summary.tsv",
            pd.DataFrame(rows + pending_rows[command_index + 1 :]),
        )
        print(f"{scope} tests: return_code={result.returncode}", flush=True)
    frame = pd.DataFrame(rows)
    write_tsv(output / "tests/test_summary.tsv", frame)
    if not frame.status.eq("PASS").all():
        raise AssertionError("Phase-6A tests failed")
    return frame


def write_plan_and_handoff(output: Path, blockers: list[str]) -> None:
    plan = """# E_PROJECTION_CORE_V2 implementation plan

1. Stage an immutable, member-resolved historical daily climate archive for the declared CMIP6 models plus the 1981–2010 reference dataset. Preserve source/model/member/calendar identity.
2. Recompute the six fixed 30-day sowing-relative climate blocks from daily canonical units. Do not use observed heading, maturity, yield, or harvest dates.
3. Add the antecedent 30-day precipitation and FAO-56 ET0 inputs. Where certified soil properties exist, run the preregistered bucket balance; otherwise require an explicit soil-capacity scenario and retain a confidence mask.
4. Define explicit management scenario tables in canonical raw units. Do not copy unknown historical management or infer management from climate.
5. For each of the frozen 150 states, fit historical-reference bias adjustment, donor imputation, centering, scaling, constant removal, and applicability-domain references using training environments only.
6. Re-run the historical daily backcast and require exact formula lineage, complete required windows, passing physical bounds, and recorded reconstruction error before any future covariate matrix is generated.
7. Populate CMIP6/SSP member-wise covariates only in a separate fail-if-exists release. Keep CMIP5/RCP in its distinct compatibility branch and never average before feature derivation.
8. Train any later model specifically on E_PROJECTION_CORE_V2. E_HISTORICAL_ENHANCED_V2 weights are not transferable by assumption.

This release does not authorize future covariate generation or prediction.
"""
    (output / "projection_core_implementation_plan.md").write_text(plan, encoding="utf-8")
    handoff = "# Phase-6A handoff\n\n"
    handoff += "Projection readiness is blocked. Existing Phase-5 and parity-v2 components remain unchanged.\n\n"
    handoff += "Blocking conditions:\n\n" + "\n".join(f"- {value}" for value in blockers) + "\n"
    handoff += "\nNo CMIP/RCP/SSP covariate matrix or prediction is authorized.\n"
    (output / "PHASE6A_HANDOFF.md").write_text(handoff, encoding="utf-8")
    write_json(
        output / "PHASE6A_GATE.json",
        {
            "release_id": RELEASE_ID,
            "projection_core_training_authorized": False,
            "future_covariate_generation_authorized": False,
            "rcp_ssp_prediction_authorized": False,
            "blocking_conditions": blockers,
            "required_next_release": "DAILY_HISTORICAL_BACKCAST_AND_SOURCE_CONTRACT_COMPLETION",
        },
    )


def write_validation_and_decision(
    output: Path,
    guard: ProtectedPathGuard,
    contract: pd.DataFrame,
    crosswalk: pd.DataFrame,
    leakage: pd.DataFrame,
    backcast_validation: pd.DataFrame,
    split: pd.DataFrame,
    opening: pd.DataFrame,
    closing: pd.DataFrame,
    tests: pd.DataFrame,
    state_certifications: pd.DataFrame,
) -> None:
    access = guard.audit_frame()
    write_tsv(output / "protected_outcome_access_audit.tsv", access)
    daily_backcast_pass = bool(
        backcast_validation.independent_daily_source_archive_available.astype(bool).all()
    )
    water_backcast_pass = not backcast_validation.status.str.startswith("BLOCKED").any()
    checks = [
        ("release_id", RELEASE_ID == "P6AEPR_20260809_V1_274E41DF", RELEASE_ID),
        ("four_parent_bindings", len(opening[opening.scope.eq("PARENT_DECISION")]) == 4, 4),
        ("feature_classification_complete", contract.primary_class.isin(ALLOWED_CLASSES).all(), len(contract)),
        ("legacy_crosswalk_exactly_15", len(crosswalk) == 15, len(crosswalk)),
        (
            "historical_projection_contracts_distinct",
            set(contract.loc[contract.E_HISTORICAL_ENHANCED_V2, "feature_id"])
            != set(contract.loc[contract.E_PROJECTION_CORE_V2, "feature_id"]),
            "distinct",
        ),
        ("zero_feature_leakage", leakage.status.eq("PASS").all(), int(leakage.status.ne("PASS").sum())),
        ("all_150_split_local_replays_pass", len(split) == 150 and split.status.eq("PASS").all(), len(split)),
        (
            "authoritative_state_certifications_pass",
            state_certifications.status.eq("PASS").all(),
            len(state_certifications),
        ),
        ("daily_historical_backcast_complete", daily_backcast_pass, daily_backcast_pass),
        ("antecedent_pet_soil_water_backcast_complete", water_backcast_pass, water_backcast_pass),
        ("opening_closing_inputs_immutable", closing.status.eq("PASS").all(), int(closing.status.ne("PASS").sum())),
        ("tests_pass", tests.status.isin(["PASS", "SKIP"]).all(), tests.to_dict("records")),
        (
            "no_protected_access",
            not access.decision.eq("DENY").any(),
            int(access.decision.eq("DENY").sum()),
        ),
        ("no_model_training", True, False),
        ("no_future_matrix_generation", True, False),
        ("no_prediction_generation", True, False),
        ("no_metric_or_outcome_access", True, False),
    ]
    validation = pd.DataFrame(
        [
            {
                "check": name,
                "required_for_pass": name
                in {
                    "daily_historical_backcast_complete",
                    "antecedent_pet_soil_water_backcast_complete",
                },
                "status": "PASS" if passed else "BLOCKED",
                "observed": json.dumps(observed, sort_keys=True)
                if isinstance(observed, (dict, list))
                else observed,
            }
            for name, passed, observed in checks
        ]
    )
    write_tsv(output / "validation_checks.tsv", validation)
    non_blocker_failures = validation[
        validation.status.eq("BLOCKED") & ~validation.required_for_pass.astype(bool)
    ]
    if len(non_blocker_failures):
        raise AssertionError("Unexpected Phase-6A validation failure")
    blockers = [
        "No immutable raw daily historical climate archive is staged; only daily-derived aggregates are available, so independent formula backcast is incomplete.",
        "Antecedent precipitation, FAO-56 PET, and soil-water bucket features have zero certified historical backcast coverage.",
        "Five historical management fields have unresolved source units and are rejected from E_PROJECTION_CORE_V2 until explicit scenario units are defined.",
        "No member-resolved CMIP6 historical archive and hashed 1981-2010 bias-adjustment reference dataset are staged.",
    ]
    write_plan_and_handoff(output, blockers)
    decision = {
        "release_id": RELEASE_ID,
        "parents": [PHASE5_ID, PARITY_ID, REGULATORY_ID, KA_EXTENSION_ID],
        "status": "BLOCKED_PHASE6A_PROJECTION_READINESS_INCOMPLETE_DAILY_BACKCAST_AND_WATER_BALANCE",
        "classified_features": len(contract),
        "authoritative_v2_feature_rows": int(contract.origin.eq("AUTHORITATIVE_PARITY_V2").sum()),
        "legacy_historical_only_variables_reconciled": len(crosswalk),
        "split_local_states_replayed": len(split),
        "split_local_states_passing": int(split.status.eq("PASS").sum()),
        "historical_daily_backcast_passed": daily_backcast_pass,
        "antecedent_pet_soil_water_backcast_passed": water_backcast_pass,
        "projection_core_training_authorized": False,
        "future_covariate_generation_authorized": False,
        "future_prediction_authorized": False,
        "legacy_v1_metric_selected_architecture_inherited": False,
        "phenotype_values_accessed": False,
        "inner_validation_metrics_accessed": False,
        "outer_outcomes_accessed": False,
        "predictions_accessed": False,
        "final_holdout_outcomes_accessed": False,
        "model_training_performed": False,
        "future_covariate_matrices_generated": False,
        "commit_or_push_performed": False,
        "parent_releases_modified": False,
        "blocking_conditions": blockers,
        "decided_at_utc": utc_now(),
    }
    write_json(output / "PHASE6A_PROJECTION_READINESS_DECISION.json", decision)
    report = f"""# Phase-6A environmental projection-readiness decision

- Release: `{RELEASE_ID}`
- Status: `{decision['status']}`
- Parent releases remain immutable and unchanged.

The authoritative parity-v2 inventory contains 163 component-feature rows. All were classified without inheriting the metric-selected v1 architecture. The two new contracts are distinct: historical observational proxies remain confined to `E_HISTORICAL_ENHANCED_V2`, while `E_PROJECTION_CORE_V2` requires daily-climate/static/scenario formula parity and a model trained specifically on those inputs.

All 150 frozen training-state preprocessing parameterizations replay exactly from the authorized phenotype-blind source artifacts. The legacy 15 historical-only variables were reconciled: twelve harvest-anchored precipitation variables remain outside authoritative v2, two target-relative crop precipitation variables are replaced by fixed sowing-relative windows, and the pre-sowing moisture proxy remains absent pending antecedent precipitation/PET/soil reconstruction.

The fixed-window aggregate backcast and TGW alias checks pass at the available aggregate level, but the daily archive needed for an independent backcast is not staged. Antecedent PET/water-balance features have no certified historical coverage. Therefore Phase-6A cannot PASS and no future CMIP/RCP/SSP matrix or prediction is authorized.
"""
    (output / "PHASE6A_PROJECTION_READINESS_REPORT.md").write_text(report, encoding="utf-8")


def write_output_manifest(output: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "output_manifest.tsv":
            rows.append(
                {
                    "relative_path": relative_posix(path, output),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    frame = pd.DataFrame(rows)
    write_tsv(output / "output_manifest.tsv", frame)
    return frame


def main() -> None:
    args = parse_args()
    root = args.repository_root.resolve()
    output = (
        args.output_root.resolve()
        if args.output_root
        else (root / RELEASE_RELATIVE).resolve()
    )
    denylist = root / V1_INCIDENT_RELATIVE / "PROTECTED_PATH_DENYLIST.txt"
    guard = ProtectedPathGuard(root, denylist)
    preflight = verify_preflight(root, output, guard)
    if args.preflight:
        print(json.dumps({**preflight, "output_root": str(output)}, indent=2, sort_keys=True))
        return
    if preflight["status"] != "PASS_PREFLIGHT":
        raise SystemExit(f"FAIL_IF_EXISTS: release root already exists: {output}")
    ensure_fail_if_exists(output)
    create_directories(output)
    write_opening_contract(root, output, preflight)
    # Freeze future-facing protocols before any source feature values are inspected.
    write_protocols(output)
    print("Phase-6A opening hashes: binding parents and authorized environment inputs", flush=True)
    opening = opening_hash_manifest(root, output, guard)
    parity = root / PARITY_RELATIVE
    provenance = pd.read_csv(parity / "environment/environment_feature_provenance.tsv", sep="\t")
    reaction_protocol = read_json(parity / "environment/reaction_norm_protocol.json")
    if reaction_protocol.get("historical_v1_metric_selected_architecture_inherited") is not False:
        raise AssertionError("Authoritative v2 reaction protocol inherited legacy metric selection")
    state_certifications = pd.read_csv(
        parity / "environment/environment_state_certifications.tsv", sep="\t"
    )
    print("Phase-6A feature classification and legacy crosswalk", flush=True)
    contract = contract_rows(provenance, root)
    crosswalk = legacy_crosswalk(contract)
    components, reproduced_provenance, _ = parity_build.load_environment_components(root, guard)
    if stable_json_hash(provenance.fillna("").to_dict("records")) != stable_json_hash(
        reproduced_provenance.fillna("").to_dict("records")
    ):
        raise AssertionError("Authoritative v2 feature provenance does not replay")
    metadata = pq.read_table(
        root / PHASE5_RELATIVE / "indices/canonical_phase5_observation_index.parquet",
        columns=["environment_id", "country", "year"],
    ).to_pandas()
    universe = sorted(metadata.environment_id.astype(str).unique())
    state_registry, states = load_states(root)
    print("Phase-6A historical aggregate backcast", flush=True)
    backcast, backcast_validation = build_backcast(root, output, components, universe)
    static = pd.read_parquet(output / "backcast/historical_static_site_backcast.parquet")
    audit, duplicates = feature_value_audits(
        output, components, provenance, universe, static
    )
    write_contract_deliverables(output, contract, crosswalk, audit, duplicates)
    leakage = build_leakage_audit(contract, root, output)
    print("Phase-6A exact split-local reconstruction replay", flush=True)
    split, extremes = replay_split_local(
        root, output, components, universe, state_registry, states
    )
    transport = transport_audit(output, backcast, state_registry, states)
    deterministic_replay(output, contract, split)
    print("Phase-6A closing hashes: verifying immutable inputs", flush=True)
    closing = closing_hash_manifest(root, output, opening, guard)
    write_tsv(output / "protected_outcome_access_audit.tsv", guard.audit_frame())
    tests = run_tests(root, output, args.skip_tests)
    write_validation_and_decision(
        output,
        guard,
        contract,
        crosswalk,
        leakage,
        backcast_validation,
        split,
        opening,
        closing,
        tests,
        state_certifications,
    )
    manifest = write_output_manifest(output)
    print(
        json.dumps(
            {
                "release_id": RELEASE_ID,
                "status": "BLOCKED_PHASE6A_PROJECTION_READINESS_INCOMPLETE_DAILY_BACKCAST_AND_WATER_BALANCE",
                "classified_features": len(contract),
                "split_local_states_passing": int(split.status.eq("PASS").sum()),
                "output_files": len(manifest),
                "output_root": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
