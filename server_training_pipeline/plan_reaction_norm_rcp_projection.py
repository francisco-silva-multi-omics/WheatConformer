from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .final_evaluation_contract import file_sha256


def classify_population(row: pd.Series) -> tuple[str, str, str]:
    feature = str(row["feature"])
    source = str(row["source_feature"])
    block = str(row["feature_block"])
    text = f"{feature} {source}".upper()
    if feature.endswith("__missing"):
        return (
            "derived_missingness_indicator",
            feature.removesuffix("__missing"),
            "source feature missingness plus fold-local historical missingness scaling",
        )
    if block == "geo":
        if "PHOTOPERIOD" in text:
            return (
                "derive_from_site_and_sowing_policy",
                "latitude,sowing_day_of_year",
                "astronomical daylength using the historical implementation",
            )
        return (
            "site_registry_static",
            "latitude,longitude,elevation_m",
            "copy only from a certified site identity and coordinate registry",
        )
    if block == "management":
        return (
            "explicit_management_policy",
            "management_scenario_table",
            "fixed historical management or an explicit future management scenario; never silent copy",
        )
    if block == "confidence":
        return (
            "derive_from_projection_lineage",
            "climate_source,sowing_policy,coordinate_source,management_source",
            "future projected weather is available but is neither observed API weather nor climatology",
        )
    if block == "development" and "SOWING" in text:
        return (
            "derive_from_sowing_policy",
            "sowing_policy,sowing_day_of_year",
            "fixed, explicit, or rule-based sowing date; target phenology is forbidden",
        )
    drivers: list[str] = []
    if any(token in text for token in ("TEMP", "TMIN", "TMAX", "HEAT", "GDD", "FROST", "VERNAL")):
        drivers.extend(["tasmin", "tasmax"])
    if any(token in text for token in ("PRECIP", "RAIN", "DRY", "WATER", "FLOOD", "MOIST")):
        drivers.append("pr")
    if any(token in text for token in ("VPD", "HUMID")):
        drivers.extend(["tasmin", "tasmax", "relative_humidity_or_huss_plus_ps"])
    if any(token in text for token in ("SOLAR", "RADIATION", "RSDS")):
        drivers.append("rsds")
    if not drivers:
        drivers = ["projected_daily_climate_and_fixed_window_aggregator"]
    return (
        "recompute_from_bias_corrected_daily_climate",
        ",".join(dict.fromkeys(drivers)),
        "apply the same fixed sowing-relative window and feature formula as E_REACTION_NORM_V1",
    )


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a phenotype-blind population plan for E_REACTION_NORM_RCP_V1."
    )
    parser.add_argument("--outer-dir", type=Path, required=True)
    parser.add_argument("--outer-protocol", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--projection-protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    outer = json.loads(args.outer_protocol.read_text(encoding="utf-8"))
    environment = json.loads(args.environment_protocol.read_text(encoding="utf-8"))
    projection = json.loads(args.projection_protocol.read_text(encoding="utf-8"))
    checks = {
        "outer_protocol_frozen": outer.get("status")
        == "frozen_after_inner_validation_before_outer_test",
        "environment_protocol_frozen": environment.get("status")
        == "frozen_before_inner_validation",
        "projection_protocol_planning_only": projection.get("status")
        == "planning_only_projection_blocked_pending_covariate_certification",
        "phenotypes_forbidden": projection.get("phenotype_values_allowed") is False,
        "outer_outcomes_forbidden": projection.get("outer_test_outcomes_allowed")
        is False,
        "final_holdout_forbidden": projection.get("final_holdout_outcomes_allowed")
        is False,
        "historical_architecture_matches": projection.get(
            "historical_environment_architecture"
        )
        == outer.get("selected_environment_architecture", "").removeprefix("explicit_"),
    }

    manifest_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    expected_fold_count = 0
    for scenario, fold_count in dict(outer["scenarios"]).items():
        for outer_fold in range(int(fold_count)):
            expected_fold_count += 1
            directory = (
                args.outer_dir
                / "folds"
                / str(scenario)
                / f"outer_{outer_fold}"
                / "E_REACTION_NORM_V1"
            )
            certification_path = directory / "E_REACTION_NORM_V1_certification.json"
            manifest_path = directory / "E_REACTION_NORM_V1_feature_manifest.tsv"
            scaling_path = directory / "E_REACTION_NORM_V1_scaling.tsv"
            certification = json.loads(certification_path.read_text(encoding="utf-8"))
            if certification.get("status") != "PASS":
                raise SystemExit(
                    f"Uncertified historical environment matrix: {scenario} outer={outer_fold}"
                )
            manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
            manifest["scenario"] = scenario
            manifest["outer_fold"] = outer_fold
            manifest_parts.append(manifest)
            fold_rows.append(
                {
                    "scenario": scenario,
                    "outer_fold": outer_fold,
                    "feature_count": len(manifest),
                    "source_feature_count": int(
                        (~manifest["is_missingness_indicator"].str.lower().isin({"true", "1"})).sum()
                    ),
                    "manifest_sha256": file_sha256(manifest_path),
                    "scaling_sha256": file_sha256(scaling_path),
                    "certification_sha256": file_sha256(certification_path),
                    "status": "PASS",
                }
            )
    manifests = pd.concat(manifest_parts, ignore_index=True)
    identity_columns = [
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
    conflicts = (
        manifests.groupby("feature")[identity_columns[1:]]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    checks["all_fold_references_present"] = len(fold_rows) == expected_fold_count
    checks["feature_identity_consistent_across_folds"] = not conflicts.any()
    checks["all_features_phenotype_blind"] = manifests["phenotype_derived"].astype(
        str
    ).str.lower().isin({"false", "0"}).all()
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SystemExit("RCP projection planning failed: " + ", ".join(failed))

    feature_plan = manifests[identity_columns].drop_duplicates().copy()
    fold_support = manifests.groupby("feature").size().rename("fold_reference_count")
    feature_plan["fold_reference_count"] = feature_plan["feature"].map(fold_support)
    classified = feature_plan.apply(classify_population, axis=1, result_type="expand")
    classified.columns = ["population_method", "required_future_input", "population_note"]
    feature_plan = pd.concat(
        [feature_plan.reset_index(drop=True), classified.reset_index(drop=True)], axis=1
    )
    feature_plan["population_required"] = True
    feature_plan["projection_status"] = "BLOCKED_PENDING_FUTURE_INPUT_AND_RANGE_CERTIFICATION"

    input_schema = pd.DataFrame(
        [
            ("future_env_id", "string", True, "unique projection environment identifier"),
            ("base_site_id", "string", True, "certified historical or curated future site"),
            ("latitude", "float", True, "decimal degrees"),
            ("longitude", "float", True, "decimal degrees"),
            ("elevation_m", "float", True, "metres above sea level"),
            ("climate_model", "string", True, "GCM or regional model identifier"),
            ("climate_realization", "string", True, "ensemble member identifier"),
            ("scenario", "string", True, "RCP or SSP label allowed by the protocol"),
            ("period_start", "date", True, "projection period start"),
            ("period_end", "date", True, "projection period end"),
            ("daily_climate_path", "path", True, "bias-corrected daily climate table"),
            ("bias_correction_method", "string", True, "recorded historical-only method"),
            ("sowing_policy", "string", True, "frozen, explicit, or rule-based policy"),
            ("sowing_day_of_year", "integer", False, "required for fixed or explicit policies"),
            ("management_policy", "string", True, "fixed historical or explicit scenario"),
            ("management_scenario_path", "path", True, "management covariates and lineage"),
        ],
        columns=["field", "dtype", "required", "description"],
    )
    projection_template = pd.DataFrame(columns=input_schema["field"].tolist())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.out_dir / "E_REACTION_NORM_RCP_V1_feature_population_plan.tsv"
    fold_path = args.out_dir / "E_REACTION_NORM_RCP_V1_historical_fold_references.tsv"
    schema_path = args.out_dir / "E_REACTION_NORM_RCP_V1_input_schema.tsv"
    template_path = args.out_dir / "E_REACTION_NORM_RCP_V1_projection_grid_template.tsv"
    write_tsv(feature_plan, feature_path)
    write_tsv(pd.DataFrame(fold_rows), fold_path)
    write_tsv(input_schema, schema_path)
    write_tsv(projection_template, template_path)
    artifacts = [feature_path, fold_path, schema_path, template_path]
    provenance = {
        "status": "PASS",
        "protocol_version": projection["protocol_version"],
        "planning_complete": True,
        "projection_allowed": False,
        "projection_block_reason": "Future climate and management inputs have not passed fold-local covariate-range certification",
        "phenotype_values_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "historical_fold_reference_count": len(fold_rows),
        "historical_feature_union_count": len(feature_plan),
        "checks": checks,
        "inputs": {
            "outer_protocol": file_sha256(args.outer_protocol),
            "environment_protocol": file_sha256(args.environment_protocol),
            "projection_protocol": file_sha256(args.projection_protocol),
        },
        "artifacts": {path.name: file_sha256(path) for path in artifacts},
        "planner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.out_dir / "E_REACTION_NORM_RCP_V1_plan.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
