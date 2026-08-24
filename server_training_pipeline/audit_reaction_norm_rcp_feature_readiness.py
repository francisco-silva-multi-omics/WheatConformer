from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .final_evaluation_contract import file_sha256


FORBIDDEN_OUTCOME_COLUMNS = {
    "phenotype_value",
    "target",
    "y",
    "y_true",
    "y_pred",
    "test_rmse",
    "test_pearson",
    "final_holdout_outcome",
}
MANIFEST_IDENTITY_COLUMNS = [
    "feature",
    "source_feature",
    "source_artifact",
    "feature_block",
    "eligible_traits",
    "regulatory_treatment",
    "is_missingness_indicator",
    "phenotype_derived",
    "fit_partition",
]
SOURCE_FEATURE_COLUMNS = [
    "scenario",
    "outer_fold",
    "feature",
    "feature_block",
    "source_feature",
    "source_artifact",
    "regulatory_treatment",
    "training_nonmissing",
    "test_nonmissing",
    "test_missing",
    "training_min",
    "training_q01",
    "training_q99",
    "training_max",
    "test_below_training_min_fraction",
    "test_above_training_max_fraction",
    "test_outside_training_range_fraction",
    "test_outside_robust_range_fraction",
    "test_abs_z_gt_moderate_fraction",
    "test_abs_z_gt_extreme_fraction",
    "test_max_abs_z",
]


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def stable_group_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def normalize_source_token(value: object) -> str:
    text = str(value).strip().upper()
    for prefix in ("OBSERVED__", "GENERIC__"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = re.sub(r"[^A-Z0-9%]+", "_", text).strip("_")
    return text


def source_artifact_family(value: object) -> str:
    text = str(value).strip().lower()
    if "envdata" in text:
        return "envdata"
    if "window" in text:
        return "fixed_weather_windows"
    if "coverage" in text:
        return "coverage_lineage"
    if "weather" in text or "climatology" in text:
        return "weather_or_climatology"
    if "locdata" in text:
        return "site_registry"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "unknown"


def canonical_source_identity(row: pd.Series) -> tuple[str, str]:
    token = normalize_source_token(row["source_feature"])
    exact = f"{source_artifact_family(row['source_artifact'])}::{token}"
    semantic = token.removeprefix("WEATHER_API_")
    return exact, semantic


def is_climate_text(text: str) -> bool:
    return any(
        token in text
        for token in (
            "TEMP",
            "TMIN",
            "TMAX",
            "HEAT",
            "GDD",
            "FROST",
            "VERNAL",
            "CHILL",
            "PRECIP",
            "RAIN",
            "DRY",
            "DROUGHT",
            "VPD",
            "HUMID",
            "SOLAR",
            "RADIATION",
            "WIND",
        )
    )


def is_binary_management(text: str) -> bool:
    return any(
        token in text
        for token in (
            "_APPLIED",
            "IRRIGATED",
            "IRRIGATION_AFTER_SOWING",
            "PRE_SOWING_IRRIGATION",
            "HERBICIDE_DAMAGE",
            "HAND_WEEDING",
        )
    ) and not any(token in text for token in ("NUMBER_", "AMOUNT", "KG_HA", "%"))


def is_irrigation_management_source(source_feature: object) -> bool:
    token = normalize_source_token(source_feature)
    return token in {
        "IRRIGATED",
        "NUMBER_POST_SOWING_IRRIGATIONS",
        "NUMBER_PRE_SOWING_IRRIGATIONS",
        "IRRIGATION_AFTER_SOWING",
        "PRE_SOWING_IRRIGATION",
        "CALCULATED_OF_TOTAL_WATER_APPLIED_BY_IRRIGATION",
        "ESTIMATE_OF_TOTAL_WATER_APPLIED_BY_IRRIGATION",
    }


def classify_feature(row: pd.Series) -> dict[str, object]:
    feature = str(row["feature"])
    source = str(row["source_feature"])
    block = str(row["feature_block"]).lower()
    artifact = str(row["source_artifact"]).lower()
    text = f"{feature} {source}".upper()
    is_missing = str(row["is_missingness_indicator"]).lower() in {
        "true",
        "1",
        "yes",
    }
    if is_missing:
        return {
            "projectability_class": "derived_missingness_indicator",
            "future_input_kind": "derived_from_source_presence",
            "range_rule_class": "binary_or_confidence_indicator",
            "population_contract_status": "READY_DERIVED_AFTER_SOURCE_POPULATION",
            "population_requirement": feature.removesuffix("__missing"),
            "classification_note": "Apply each fold's frozen missingness prevalence after populating the source feature.",
        }
    if block == "geo":
        if "PHOTOPERIOD" in text:
            return {
                "projectability_class": "derived_site_and_sowing",
                "future_input_kind": "certified_site_plus_sowing_policy",
                "range_rule_class": "cyclic_or_calendar",
                "population_contract_status": "READY_FORMULA_REUSE_REQUIRED",
                "population_requirement": "latitude,sowing_day_of_year",
                "classification_note": "Reuse the frozen astronomical daylength implementation.",
            }
        return {
            "projectability_class": "static_site",
            "future_input_kind": "certified_site_registry",
            "range_rule_class": "static_site_geometry",
            "population_contract_status": "READY_CERTIFIED_SITE_REQUIRED",
            "population_requirement": "latitude,longitude,elevation_m",
            "classification_note": "Coordinates must come from a certified historical or curated future site identity.",
        }
    if block == "management":
        binary = is_binary_management(text)
        return {
            "projectability_class": "explicit_management_scenario",
            "future_input_kind": "management_scenario_table",
            "range_rule_class": (
                "binary_or_confidence_indicator" if binary else "sparse_management"
            ),
            "population_contract_status": "REQUIRES_EXPLICIT_MANAGEMENT_POLICY",
            "population_requirement": "fixed_historical_management_or_explicit_future_scenario",
            "classification_note": "Do not use fold-standardized z as the hard gate for sparse or zero-inflated management fields.",
        }
    if is_irrigation_management_source(source) and "envdata" in artifact:
        binary = is_binary_management(text)
        return {
            "projectability_class": "explicit_management_scenario",
            "future_input_kind": "management_scenario_table",
            "range_rule_class": (
                "binary_or_confidence_indicator" if binary else "sparse_management"
            ),
            "population_contract_status": "REQUIRES_EXPLICIT_MANAGEMENT_POLICY",
            "population_requirement": "fixed_historical_management_or_explicit_future_scenario",
            "classification_note": "This envdata irrigation field is duplicated outside the management block and must use the same declared scenario value.",
        }
    if block == "confidence":
        return {
            "projectability_class": "derived_projection_confidence",
            "future_input_kind": "projection_lineage",
            "range_rule_class": "binary_or_confidence_indicator",
            "population_contract_status": "READY_LINEAGE_RULE_REQUIRED",
            "population_requirement": "climate,sowing,coordinate,and_management_source_lineage",
            "classification_note": "Projected climate is available but is not observed API weather or historical climatology.",
        }
    source_token = normalize_source_token(source)
    if block == "development" and source_token in {
        "SOWING_DAYOFYEAR",
        "SOWING_DAYOFYEAR_SIN",
        "SOWING_DAYOFYEAR_COS",
        "HAS_SOWING_DATE",
    }:
        return {
            "projectability_class": "explicit_sowing_policy",
            "future_input_kind": "sowing_policy",
            "range_rule_class": "cyclic_or_calendar",
            "population_contract_status": "READY_SOWING_POLICY_REQUIRED",
            "population_requirement": "fixed_explicit_or_rule_based_sowing_day",
            "classification_note": "Observed target heading and maturity dates remain forbidden.",
        }

    historical_target_relative = any(
        token in text
        for token in (
            "FROM_SOWING_TO_MATURITY",
            "PRECIPITATION_ON_CROP",
            "MONTH_OF_HARVESTED",
            "MO_BEFORE_HARVESTED",
        )
    )
    soil_state_proxy = any(
        token in text
        for token in (
            "MOISTURE_AVAILB",
            "MOISTURE_AVAILABLE_IN_FULL_ROOT_ZONE",
            "PRECIPITATION_AVAILABLE_TO_CROP",
        )
    )
    if historical_target_relative or soil_state_proxy:
        reason = (
            "target-relative historical window cannot use held-out phenology"
            if historical_target_relative
            else "historical soil/moisture proxy has no frozen daily-climate formula"
        )
        return {
            "projectability_class": "historical_only_unprojectable",
            "future_input_kind": "none_until_replacement_is_certified",
            "range_rule_class": "historical_only",
            "population_contract_status": "BLOCKED_REPLACEMENT_FORMULA_REQUIRED",
            "population_requirement": "phenotype_blind_replacement_formula_with_provenance",
            "classification_note": reason,
        }
    if "TOTAL_PRECIPIT_IN_12_MONTHS" in text:
        return {
            "projectability_class": "climate_aggregate_requires_period_contract",
            "future_input_kind": "bias_corrected_daily_precipitation",
            "range_rule_class": "continuous_climate",
            "population_contract_status": "REQUIRES_EXPLICIT_12_MONTH_PERIOD",
            "population_requirement": "pr,period_start,period_end",
            "classification_note": "The 12-month reference period must be fixed explicitly before population.",
        }
    if source.startswith("observed__") and "envdata" in artifact:
        return {
            "projectability_class": "historical_only_unprojectable",
            "future_input_kind": "none_until_replacement_is_certified",
            "range_rule_class": "historical_only",
            "population_contract_status": "BLOCKED_REPLACEMENT_FORMULA_REQUIRED",
            "population_requirement": "phenotype_blind_replacement_formula_with_provenance",
            "classification_note": "This legacy observed field has no frozen daily-climate derivation formula for future scenarios.",
        }
    if source.startswith("window__") or "API_D" in text:
        return {
            "projectability_class": "fixed_window_climate",
            "future_input_kind": "bias_corrected_daily_climate",
            "range_rule_class": "continuous_climate",
            "population_contract_status": "READY_FORMULA_REUSE_REQUIRED",
            "population_requirement": "daily_climate,sowing_policy,frozen_window_formula",
            "classification_note": "Recompute the same fixed sowing-relative window and metric.",
        }
    if is_climate_text(text) or "weather" in artifact or "climatology" in artifact:
        return {
            "projectability_class": "climate_aggregate",
            "future_input_kind": "bias_corrected_daily_climate",
            "range_rule_class": "continuous_climate",
            "population_contract_status": "READY_FORMULA_REUSE_REQUIRED",
            "population_requirement": "daily_climate_and_frozen_aggregation_formula",
            "classification_note": "Population requires exact historical formula parity, not name-based approximation.",
        }
    return {
        "projectability_class": "unmapped_source_feature",
        "future_input_kind": "unresolved",
        "range_rule_class": "historical_only",
        "population_contract_status": "BLOCKED_SOURCE_LINEAGE_UNRESOLVED",
        "population_requirement": "manual_source_and_formula_adjudication",
        "classification_note": "No defensible phenotype-blind future population rule was identified.",
    }


def load_current_manifests(
    outer_dir: Path, outer_protocol: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    manifests: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    references: list[dict[str, object]] = []
    for scenario, fold_count in dict(outer_protocol["scenarios"]).items():
        for outer_fold in range(int(fold_count)):
            directory = (
                outer_dir
                / "folds"
                / str(scenario)
                / f"outer_{outer_fold}"
                / "E_REACTION_NORM_V1"
            )
            manifest_path = directory / "E_REACTION_NORM_V1_feature_manifest.tsv"
            certification_path = directory / "E_REACTION_NORM_V1_certification.json"
            scaling_path = directory / "E_REACTION_NORM_V1_scaling.tsv"
            raw_path = directory / "E_REACTION_NORM_V1_raw.parquet"
            certification = read_json(certification_path)
            if certification.get("status") != "PASS":
                raise SystemExit(
                    f"Uncertified E_REACTION_NORM_V1 reference: {scenario} outer={outer_fold}"
                )
            manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
            missing_columns = sorted(set(MANIFEST_IDENTITY_COLUMNS) - set(manifest.columns))
            if missing_columns:
                raise SystemExit(f"Manifest {manifest_path} misses {missing_columns}")
            manifest["scenario"] = str(scenario)
            manifest["outer_fold"] = outer_fold
            manifests.append(manifest)
            features = manifest["feature"].astype(str).tolist()
            source_count = int((~truthy(manifest["is_missingness_indicator"])).sum())
            folds.append(
                {
                    "scenario": scenario,
                    "outer_fold": outer_fold,
                    "feature_count": len(features),
                    "source_feature_count": source_count,
                    "feature_set_sha256": hashlib.sha256(
                        "\n".join(sorted(features)).encode("utf-8")
                    ).hexdigest(),
                    "manifest_sha256": file_sha256(manifest_path),
                    "scaling_sha256": file_sha256(scaling_path),
                    "certification_sha256": file_sha256(certification_path),
                    "status": "PASS",
                }
            )
            references.append(
                {
                    "scenario": str(scenario),
                    "outer_fold": outer_fold,
                    "directory": directory,
                    "manifest_path": manifest_path,
                    "raw_path": raw_path,
                }
            )
    combined = pd.concat(manifests, ignore_index=True)
    union = set(combined["feature"].astype(str))
    for row, manifest in zip(folds, manifests):
        missing = sorted(union - set(manifest["feature"].astype(str)))
        row["missing_from_current_union_count"] = len(missing)
        row["missing_from_current_union"] = ";".join(missing)
    return combined, pd.DataFrame(folds), references


def annotate_duplicate_groups(lineage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = lineage.copy()
    local["exact_source_key"] = ""
    local["semantic_source_key"] = ""
    source_mask = ~truthy(local["is_missingness_indicator"])
    identities = local.loc[source_mask].apply(
        canonical_source_identity, axis=1, result_type="expand"
    )
    identities.columns = ["exact_source_key", "semantic_source_key"]
    local.loc[source_mask, identities.columns] = identities.to_numpy()

    exact_counts = local.loc[source_mask].groupby("exact_source_key")["feature"].nunique()
    semantic_counts = local.loc[source_mask].groupby("semantic_source_key")["feature"].nunique()
    exact_keys = set(exact_counts[exact_counts.gt(1)].index)
    semantic_keys = set(semantic_counts[semantic_counts.gt(1)].index)
    local["duplicate_group_type"] = "none"
    local["duplicate_group_id"] = ""
    local["duplicate_member_count"] = 1
    for index, row in local.loc[source_mask].iterrows():
        exact = str(row["exact_source_key"])
        semantic = str(row["semantic_source_key"])
        if exact in exact_keys:
            local.at[index, "duplicate_group_type"] = "exact_source_duplicate"
            local.at[index, "duplicate_group_id"] = stable_group_id("exact", exact)
            local.at[index, "duplicate_member_count"] = int(exact_counts[exact])
        elif semantic in semantic_keys:
            local.at[index, "duplicate_group_type"] = "semantic_duplicate_candidate"
            local.at[index, "duplicate_group_id"] = stable_group_id("semantic", semantic)
            local.at[index, "duplicate_member_count"] = int(semantic_counts[semantic])

    source_lookup = local.loc[source_mask].set_index("feature")
    for index, row in local.loc[~source_mask].iterrows():
        source_feature = str(row["feature"]).removesuffix("__missing")
        if source_feature not in source_lookup.index:
            continue
        source_row = source_lookup.loc[source_feature]
        for column in (
            "exact_source_key",
            "semantic_source_key",
            "duplicate_group_type",
            "duplicate_group_id",
            "duplicate_member_count",
        ):
            local.at[index, column] = source_row[column]

    duplicate_rows: list[dict[str, object]] = []
    members = local[source_mask & local["duplicate_group_type"].ne("none")]
    for group_id, frame in members.groupby("duplicate_group_id", sort=True):
        duplicate_rows.append(
            {
                "duplicate_group_id": group_id,
                "duplicate_group_type": frame["duplicate_group_type"].iloc[0],
                "member_count": frame["feature"].nunique(),
                "features": ";".join(sorted(frame["feature"].astype(str).unique())),
                "source_features": ";".join(
                    sorted(frame["source_feature"].astype(str).unique())
                ),
                "source_artifacts": ";".join(
                    sorted(frame["source_artifact"].astype(str).unique())
                ),
                "future_policy": (
                    "populate_once_and_copy_consistently_to_all_frozen_columns"
                    if frame["duplicate_group_type"].iloc[0]
                    == "exact_source_duplicate"
                    else "manual_semantic_and_unit_review_before_population"
                ),
                "projectability_classes": ";".join(
                    sorted(frame["projectability_class"].astype(str).unique())
                ),
                "range_rule_classes": ";".join(
                    sorted(frame["range_rule_class"].astype(str).unique())
                ),
                "classification_consistent": frame["projectability_class"].nunique()
                == 1
                and frame["range_rule_class"].nunique() == 1
                and frame["population_contract_status"].nunique() == 1,
            }
        )
    return local, pd.DataFrame(duplicate_rows)


def audit_exact_duplicate_values(
    lineage: pd.DataFrame,
    references: list[dict[str, object]],
    tolerance: float,
) -> pd.DataFrame:
    exact = lineage[
        (~truthy(lineage["is_missingness_indicator"]))
        & lineage["duplicate_group_type"].eq("exact_source_duplicate")
    ]
    rows: list[dict[str, object]] = []
    for reference in references:
        manifest = pd.read_csv(reference["manifest_path"], sep="\t", dtype=str)
        available = set(manifest["feature"].astype(str))
        for group_id, group in exact.groupby("duplicate_group_id", sort=True):
            features = sorted(set(group["feature"].astype(str)).intersection(available))
            if len(features) < 2:
                continue
            raw = pd.read_parquet(reference["raw_path"], columns=["env_id", *features])
            for feature_a, feature_b in itertools.combinations(features, 2):
                a = pd.to_numeric(raw[feature_a], errors="coerce").to_numpy(dtype=float)
                b = pd.to_numeric(raw[feature_b], errors="coerce").to_numpy(dtype=float)
                finite_a = np.isfinite(a)
                finite_b = np.isfinite(b)
                shared = finite_a & finite_b
                missingness_mismatch = int(np.count_nonzero(finite_a != finite_b))
                max_delta = (
                    float(np.max(np.abs(a[shared] - b[shared])))
                    if shared.any()
                    else float("nan")
                )
                passed = missingness_mismatch == 0 and (
                    not shared.any() or max_delta <= tolerance
                )
                rows.append(
                    {
                        "scenario": reference["scenario"],
                        "outer_fold": reference["outer_fold"],
                        "duplicate_group_id": group_id,
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "rows": len(raw),
                        "shared_nonmissing_rows": int(shared.sum()),
                        "missingness_mismatch_rows": missingness_mismatch,
                        "maximum_absolute_delta": max_delta,
                        "absolute_tolerance": tolerance,
                        "status": "PASS" if passed else "FAIL",
                    }
                )
    columns = [
        "scenario",
        "outer_fold",
        "duplicate_group_id",
        "feature_a",
        "feature_b",
        "rows",
        "shared_nonmissing_rows",
        "missingness_mismatch_rows",
        "maximum_absolute_delta",
        "absolute_tolerance",
        "status",
    ]
    return pd.DataFrame(rows, columns=columns)


def aggregate_historical_ranges(
    diagnostics: pd.DataFrame, lineage: pd.DataFrame, hard_z: float, extreme_z: float
) -> pd.DataFrame:
    source_lineage = lineage[~truthy(lineage["is_missingness_indicator"])].drop_duplicates(
        "feature"
    )
    metadata_columns = [
        "feature",
        "feature_block",
        "projectability_class",
        "range_rule_class",
        "duplicate_group_type",
        "duplicate_group_id",
    ]
    rows: list[dict[str, object]] = []
    for feature, frame in diagnostics.groupby("feature", sort=True):
        weights = pd.to_numeric(frame["test_nonmissing"], errors="coerce").fillna(0.0)
        denominator = float(weights.sum())

        def weighted(column: str) -> float:
            values = pd.to_numeric(frame[column], errors="coerce")
            return float((values * weights).sum() / denominator) if denominator > 0 else float("nan")

        rows.append(
            {
                "feature": feature,
                "historical_scenarios": ";".join(sorted(frame["scenario"].astype(str).unique())),
                "historical_fold_rows": len(frame),
                "historical_training_nonmissing_min": int(
                    pd.to_numeric(frame["training_nonmissing"], errors="coerce").min()
                ),
                "historical_test_nonmissing_total": int(denominator),
                "historical_training_min": float(
                    pd.to_numeric(frame["training_min"], errors="coerce").min()
                ),
                "historical_training_max": float(
                    pd.to_numeric(frame["training_max"], errors="coerce").max()
                ),
                "historical_outside_training_range_fraction_weighted": weighted(
                    "test_outside_training_range_fraction"
                ),
                "historical_abs_z_gt_extreme_fraction_weighted": weighted(
                    "test_abs_z_gt_extreme_fraction"
                ),
                "historical_max_abs_z": float(
                    pd.to_numeric(frame["test_max_abs_z"], errors="coerce").max()
                ),
            }
        )
    result = pd.DataFrame(rows)
    result = source_lineage[metadata_columns].merge(result, on="feature", how="left")
    result["historical_range_diagnostics_available"] = result[
        "historical_fold_rows"
    ].notna()

    def historical_status(row: pd.Series) -> str:
        if not bool(row["historical_range_diagnostics_available"]):
            return "NOT_IN_TEMPORAL_OR_COUNTRY_DIAGNOSTIC"
        rule = str(row["range_rule_class"])
        maximum_z = float(row["historical_max_abs_z"])
        if rule == "continuous_climate":
            if maximum_z > hard_z:
                return "HISTORICAL_BASELINE_EXCEEDS_GLOBAL_HARD_Z"
            if maximum_z > extreme_z:
                return "HISTORICAL_BASELINE_EXCEEDS_EXTREME_Z"
            return "PASS_CONTINUOUS_RANGE_DIAGNOSTIC"
        if rule == "sparse_management":
            return "Z_DIAGNOSTIC_ONLY_USE_DOMAIN_SUPPORT_PREVALENCE"
        if rule == "binary_or_confidence_indicator":
            minimum = float(row["historical_training_min"])
            maximum = float(row["historical_training_max"])
            return (
                "PASS_BINARY_DOMAIN"
                if minimum >= 0.0 and maximum <= 1.0
                else "FAIL_BINARY_DOMAIN_REVIEW"
            )
        if rule == "static_site_geometry":
            return "USE_SITE_DOMAIN_AND_DISTANCE_NOT_GLOBAL_Z"
        if rule == "cyclic_or_calendar":
            return "USE_CALENDAR_OR_CIRCULAR_DOMAIN_NOT_GLOBAL_Z"
        return "BLOCKED_PENDING_REPLACEMENT_FORMULA"

    result["historical_range_rule_status"] = result.apply(historical_status, axis=1)
    result["global_z_hard_gate_applicable"] = result["range_rule_class"].eq(
        "continuous_climate"
    )
    return result


def legacy_reconciliation(
    roots: Iterable[Path], current_features: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for root in roots:
        paths = sorted(root.rglob("E_REACTION_NORM_V1_feature_manifest.tsv"))
        if not paths:
            summaries.append(
                {
                    "reference_root": str(root.resolve()),
                    "status": "NO_MANIFESTS_FOUND",
                    "manifest_count": 0,
                }
            )
            continue
        sets: list[set[str]] = []
        source_counts: list[int] = []
        feature_support: dict[str, int] = {}
        for path in paths:
            frame = pd.read_csv(path, sep="\t", dtype=str)
            features = set(frame["feature"].astype(str))
            sets.append(features)
            source_counts.append(int((~truthy(frame["is_missingness_indicator"])).sum()))
            for feature in features:
                feature_support[feature] = feature_support.get(feature, 0) + 1
        union = set().union(*sets)
        intersection = set.intersection(*sets)
        current_only = sorted(current_features - union)
        legacy_only = sorted(union - current_features)
        summaries.append(
            {
                "reference_root": str(root.resolve()),
                "status": "PASS",
                "manifest_count": len(paths),
                "feature_count_min": min(map(len, sets)),
                "feature_count_max": max(map(len, sets)),
                "source_feature_count_min": min(source_counts),
                "source_feature_count_max": max(source_counts),
                "feature_union_count": len(union),
                "feature_intersection_count": len(intersection),
                "current_only_feature_count": len(current_only),
                "legacy_only_feature_count": len(legacy_only),
                "current_only_features": ";".join(current_only),
                "legacy_only_features": ";".join(legacy_only),
            }
        )
        for feature in sorted(current_features | union):
            details.append(
                {
                    "reference_root": str(root.resolve()),
                    "feature": feature,
                    "comparison_status": (
                        "shared"
                        if feature in current_features and feature in union
                        else "current_only"
                        if feature in current_features
                        else "legacy_only"
                    ),
                    "legacy_manifest_support": feature_support.get(feature, 0),
                    "legacy_manifest_count": len(paths),
                }
            )
    summary_columns = [
        "reference_root",
        "status",
        "manifest_count",
        "feature_count_min",
        "feature_count_max",
        "source_feature_count_min",
        "source_feature_count_max",
        "feature_union_count",
        "feature_intersection_count",
        "current_only_feature_count",
        "legacy_only_feature_count",
        "current_only_features",
        "legacy_only_features",
    ]
    detail_columns = [
        "reference_root",
        "feature",
        "comparison_status",
        "legacy_manifest_support",
        "legacy_manifest_count",
    ]
    return (
        pd.DataFrame(summaries, columns=summary_columns),
        pd.DataFrame(details, columns=detail_columns),
    )


def range_rule_table(protocol: dict[str, object]) -> pd.DataFrame:
    rows = []
    for rule, payload in dict(protocol["historical_range_rules"]).items():
        local = dict(payload)
        rows.append(
            {
                "range_rule_class": rule,
                "primary_checks": local.pop("primary_checks"),
                "fold_standardized_z_is_hard_gate": local.pop(
                    "fold_standardized_z_is_hard_gate", rule == "continuous_climate"
                ),
                "parameters": json.dumps(local, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit phenotype-blind readiness of the frozen reaction-norm feature contract "
            "for future RCP population without creating future matrices or predictions."
        )
    )
    parser.add_argument("--outer-dir", type=Path, required=True)
    parser.add_argument("--outer-protocol", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--projection-protocol", type=Path, required=True)
    parser.add_argument("--projection-plan", type=Path, required=True)
    parser.add_argument("--reporting-feature-diagnostics", type=Path, required=True)
    parser.add_argument("--readiness-protocol", type=Path, required=True)
    parser.add_argument("--legacy-reference-root", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    outer = read_json(args.outer_protocol)
    environment = read_json(args.environment_protocol)
    projection = read_json(args.projection_protocol)
    plan = read_json(args.projection_plan)
    readiness = read_json(args.readiness_protocol)
    checks: dict[str, bool] = {
        "outer_protocol_frozen": outer.get("status")
        == "frozen_after_inner_validation_before_outer_test",
        "environment_protocol_frozen": environment.get("status")
        == "frozen_before_inner_validation",
        "projection_protocol_is_planning_only": projection.get("status")
        == "planning_only_projection_blocked_pending_covariate_certification",
        "projection_plan_pass_and_blocked": plan.get("status") == "PASS"
        and plan.get("projection_allowed") is False,
        "readiness_protocol_is_audit_only": readiness.get("status")
        == "phenotype_blind_readiness_audit_only",
        "phenotypes_forbidden": readiness.get("phenotype_values_allowed") is False,
        "outer_test_environment_identifiers_allowed": readiness.get(
            "outer_test_environment_identifiers_allowed"
        )
        is True,
        "outer_outcomes_forbidden": readiness.get("outer_test_outcomes_allowed") is False,
        "outer_metrics_forbidden": readiness.get("outer_test_metrics_allowed") is False,
        "final_holdout_forbidden": readiness.get("final_holdout_outcomes_allowed") is False,
        "future_matrices_forbidden": readiness.get("future_covariate_matrices_allowed")
        is False,
        "rcp_predictions_forbidden": readiness.get("rcp_predictions_allowed") is False,
    }

    manifests, fold_reconciliation, references = load_current_manifests(
        args.outer_dir, outer
    )
    expected_folds = sum(map(int, dict(outer["scenarios"]).values()))
    checks["all_current_fold_manifests_present"] = len(fold_reconciliation) == expected_folds
    identity_conflicts = (
        manifests.groupby("feature")[MANIFEST_IDENTITY_COLUMNS[1:]]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    checks["feature_identity_consistent_across_folds"] = not identity_conflicts.any()
    checks["all_features_phenotype_blind"] = manifests["phenotype_derived"].astype(
        str
    ).str.lower().isin({"false", "0"}).all()

    lineage = manifests[MANIFEST_IDENTITY_COLUMNS].drop_duplicates().copy()
    fold_support = manifests.groupby("feature").agg(
        fold_reference_count=("feature", "size"),
        scenario_reference_count=("scenario", "nunique"),
    )
    lineage = lineage.merge(fold_support, left_on="feature", right_index=True, how="left")
    lineage["present_in_every_fold"] = lineage["fold_reference_count"].eq(expected_folds)
    classifications = lineage.apply(classify_feature, axis=1, result_type="expand")
    lineage = pd.concat(
        [lineage.reset_index(drop=True), classifications.reset_index(drop=True)], axis=1
    )
    lineage, duplicate_groups = annotate_duplicate_groups(lineage)
    checks["every_feature_classified"] = lineage["projectability_class"].ne(
        "unmapped_source_feature"
    ).all()

    plan_feature_path = (
        args.projection_plan.parent
        / "E_REACTION_NORM_RCP_V1_feature_population_plan.tsv"
    )
    plan_features = set(
        pd.read_csv(plan_feature_path, sep="\t", dtype=str)["feature"].astype(str)
    )
    current_features = set(lineage["feature"].astype(str))
    checks["projection_plan_feature_union_matches_current"] = plan_features == current_features

    duplicate_policy = dict(readiness["duplicate_policy"])
    duplicate_consistency = audit_exact_duplicate_values(
        lineage,
        references,
        tolerance=float(duplicate_policy["raw_numeric_absolute_tolerance"]),
    )
    exact_group_count = int(
        duplicate_groups.get("duplicate_group_type", pd.Series(dtype=str))
        .eq("exact_source_duplicate")
        .sum()
    )
    checks["exact_duplicate_groups_audited"] = exact_group_count == 0 or (
        not duplicate_consistency.empty
        and duplicate_consistency["duplicate_group_id"].nunique() == exact_group_count
    )
    checks["exact_duplicate_values_consistent"] = duplicate_consistency.empty or duplicate_consistency[
        "status"
    ].eq("PASS").all()
    checks["exact_duplicate_classification_consistent"] = duplicate_groups.empty or (
        duplicate_groups.loc[
            duplicate_groups["duplicate_group_type"].eq("exact_source_duplicate"),
            "classification_consistent",
        ]
        .fillna(False)
        .all()
    )
    if not duplicate_groups.empty:
        pair_summary = (
            duplicate_consistency.groupby("duplicate_group_id", sort=True)
            .agg(
                consistency_pair_checks=("status", "size"),
                consistency_failed_checks=("status", lambda values: int((values != "PASS").sum())),
                maximum_absolute_delta=("maximum_absolute_delta", "max"),
                missingness_mismatch_rows=("missingness_mismatch_rows", "sum"),
            )
            .reset_index()
            if not duplicate_consistency.empty
            else pd.DataFrame(columns=["duplicate_group_id"])
        )
        duplicate_groups = duplicate_groups.merge(
            pair_summary, on="duplicate_group_id", how="left"
        )
        duplicate_groups["consistency_status"] = np.where(
            duplicate_groups["duplicate_group_type"].eq("semantic_duplicate_candidate"),
            "REQUIRES_SEMANTIC_REVIEW",
            np.where(
                duplicate_groups["consistency_failed_checks"].fillna(0).eq(0),
                "PASS",
                "FAIL",
            ),
        )

    historical = pd.read_csv(args.reporting_feature_diagnostics, sep="\t")
    forbidden = sorted(FORBIDDEN_OUTCOME_COLUMNS.intersection(historical.columns))
    checks["historical_diagnostics_have_no_outcome_columns"] = not forbidden
    missing_diagnostic_columns = sorted(set(SOURCE_FEATURE_COLUMNS) - set(historical.columns))
    checks["historical_diagnostic_schema_complete"] = not missing_diagnostic_columns
    if forbidden or missing_diagnostic_columns:
        failed_schema = {
            "forbidden": forbidden,
            "missing": missing_diagnostic_columns,
        }
        raise SystemExit(f"Unsafe or incomplete historical range diagnostics: {failed_schema}")
    climate_policy = dict(readiness["historical_range_rules"])["continuous_climate"]
    historical_ranges = aggregate_historical_ranges(
        historical,
        lineage,
        hard_z=float(climate_policy["hard_absolute_z"]),
        extreme_z=float(climate_policy["extreme_absolute_z"]),
    )

    legacy_summary, legacy_detail = legacy_reconciliation(
        args.legacy_reference_root, current_features
    )
    source_lineage = lineage[~truthy(lineage["is_missingness_indicator"])]
    source_status_summary = (
        source_lineage.groupby(
            ["projectability_class", "range_rule_class", "population_contract_status"],
            dropna=False,
            sort=True,
        )
        .agg(
            source_features=("feature", "nunique"),
            minimum_fold_reference_count=("fold_reference_count", "min"),
            maximum_fold_reference_count=("fold_reference_count", "max"),
        )
        .reset_index()
    )
    block_summary = (
        source_lineage.groupby("feature_block", sort=True)
        .agg(
            source_features=("feature", "nunique"),
            fully_fold_stable_features=("present_in_every_fold", "sum"),
            historical_only_features=(
                "projectability_class",
                lambda values: int((values == "historical_only_unprojectable").sum()),
            ),
            exact_duplicate_members=(
                "duplicate_group_type",
                lambda values: int((values == "exact_source_duplicate").sum()),
            ),
            semantic_duplicate_candidate_members=(
                "duplicate_group_type",
                lambda values: int((values == "semantic_duplicate_candidate").sum()),
            ),
        )
        .reset_index()
    )

    historical_only_count = int(
        source_lineage["projectability_class"].eq("historical_only_unprojectable").sum()
    )
    unmapped_count = int(
        source_lineage["projectability_class"].eq("unmapped_source_feature").sum()
    )
    semantic_duplicate_count = int(
        duplicate_groups.get("duplicate_group_type", pd.Series(dtype=str))
        .eq("semantic_duplicate_candidate")
        .sum()
    )
    historical_global_z_conflicts = int(
        historical_ranges["historical_range_rule_status"]
        .eq("HISTORICAL_BASELINE_EXCEEDS_GLOBAL_HARD_Z")
        .sum()
    )
    feature_contract_ready = (
        historical_only_count == 0
        and unmapped_count == 0
        and semantic_duplicate_count == 0
        and checks["exact_duplicate_values_consistent"]
    )
    checks = {name: bool(value) for name, value in checks.items()}
    failed_checks = sorted(name for name, value in checks.items() if not value)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_frames = {
        "RCP_feature_readiness_lineage.tsv": lineage,
        "RCP_fold_feature_reconciliation.tsv": fold_reconciliation,
        "RCP_duplicate_source_groups.tsv": duplicate_groups,
        "RCP_duplicate_source_consistency.tsv": duplicate_consistency,
        "RCP_historical_range_rule_audit.tsv": historical_ranges,
        "RCP_range_rule_contract.tsv": range_rule_table(readiness),
        "RCP_feature_readiness_summary.tsv": source_status_summary,
        "RCP_feature_block_summary.tsv": block_summary,
        "RCP_legacy_feature_reconciliation.tsv": legacy_summary,
        "RCP_legacy_feature_difference.tsv": legacy_detail,
    }
    output_paths: list[Path] = []
    for filename, frame in output_frames.items():
        path = args.out_dir / filename
        write_tsv(frame, path)
        output_paths.append(path)

    matrix_outputs = sorted(
        path
        for path in args.out_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".parquet", ".npy", ".npz"}
    )
    checks["no_future_matrices_generated"] = not matrix_outputs
    if matrix_outputs:
        failed_checks.append("no_future_matrices_generated")

    blocking_reasons = [
        "future climate, sowing, site, and management inputs have not been populated",
        "future fold-local covariate certification has not been run",
    ]
    if historical_only_count:
        blocking_reasons.append(
            f"{historical_only_count} historical-only source features require replacement formulas"
        )
    if semantic_duplicate_count:
        blocking_reasons.append(
            f"{semantic_duplicate_count} semantic duplicate groups require adjudication"
        )
    if historical_global_z_conflicts:
        blocking_reasons.append(
            "the existing global z hard gate conflicts with observed historical transfer for "
            f"{historical_global_z_conflicts} continuous-climate features"
        )
    result = {
        "status": "PASS" if not failed_checks else "FAIL",
        "protocol_version": readiness["protocol_version"],
        "selection_data": readiness["selection_data"],
        "audit_complete": not failed_checks,
        "feature_contract_ready_for_population": feature_contract_ready,
        "future_covariate_population_allowed": False,
        "rcp_predictions_allowed": False,
        "phenotype_values_read": False,
        "outer_test_environment_identifiers_read": True,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_matrix_count_generated": len(matrix_outputs),
        "historical_fold_reference_count": len(fold_reconciliation),
        "feature_union_count": len(lineage),
        "source_feature_union_count": len(source_lineage),
        "minimum_fold_feature_count": int(fold_reconciliation["feature_count"].min()),
        "maximum_fold_feature_count": int(fold_reconciliation["feature_count"].max()),
        "minimum_fold_source_feature_count": int(
            fold_reconciliation["source_feature_count"].min()
        ),
        "maximum_fold_source_feature_count": int(
            fold_reconciliation["source_feature_count"].max()
        ),
        "historical_only_source_feature_count": historical_only_count,
        "unmapped_source_feature_count": unmapped_count,
        "exact_duplicate_group_count": exact_group_count,
        "semantic_duplicate_group_count": semantic_duplicate_count,
        "historical_continuous_features_exceeding_current_global_hard_z": historical_global_z_conflicts,
        "projection_block_reasons": blocking_reasons,
        "checks": checks,
        "failed_checks": sorted(set(failed_checks)),
        "inputs": {
            "outer_protocol": file_sha256(args.outer_protocol),
            "environment_protocol": file_sha256(args.environment_protocol),
            "projection_protocol": file_sha256(args.projection_protocol),
            "projection_plan": file_sha256(args.projection_plan),
            "projection_feature_plan": file_sha256(plan_feature_path),
            "historical_feature_diagnostics": file_sha256(
                args.reporting_feature_diagnostics
            ),
            "readiness_protocol": file_sha256(args.readiness_protocol),
        },
        "artifacts": {path.name: file_sha256(path) for path in output_paths},
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    certification_path = args.out_dir / "RCP_feature_readiness_certification.json"
    certification_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    if failed_checks:
        raise SystemExit("RCP feature-readiness audit failed")


if __name__ == "__main__":
    main()
