from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from server_training_pipeline.fetch_cmip6_member_resolved import atomic_json, atomic_tsv, resolve, sha256_file


DEFAULT_OUTPUT = Path("audit/v2/e_projection_core_v1_readiness")
REFERENCE_CACHE = Path("environment/v2/phase6a_cds_era5_land_bias_reference_v1")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {
        "path": relative,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    required = {
        "raw_archive_certification": "audit/v2/phase6a_projection_core_raw_archive_v1/raw_archive_certification.json",
        "daily_normalization_contract": "server_training_pipeline/phase6a_daily_normalization_contract_v1.json",
        "daily_normalization_certification": "audit/v2/phase6a_daily_normalization_v1/daily_normalization_certification.json",
        "cross_provider_provenance": "audit/v2/phase6a_cross_provider_audit_v1/cross_provider_provenance.json",
        "soil_policy_certification": "audit/v2/phase6a_soilgrids_missing_resolution_v1/soilgrids_missing_resolution_certification.json",
        "bias_adjustment_contract": "server_training_pipeline/phase6a_bias_adjustment_contract_v2.json",
        "feature_contract": "server_training_pipeline/phase6a_projection_core_feature_contract_v1.json",
        "historical_ERA5_backcast": "audit/v2/phase6a_projection_core_historical_backcast_v1/historical_projection_core_backcast_provenance.json",
        "applicability_domain_contract": "server_training_pipeline/phase6a_applicability_domain_contract_v1.json",
    }
    optional_after_reference = {
        "bias_reference_normalization": "audit/v2/phase6a_daily_normalization_v1/cds_bias_reference/cds_bias_reference_daily_normalization_provenance.json",
        "bias_adjustment_parameters": "audit/v2/phase6a_bias_adjustment_v2/bias_adjustment_provenance.json",
        "historical_CMIP6_backcast": "audit/v2/phase6a_projection_core_historical_backcast_v2/cmip6_historical_backcast_provenance.json",
        "feature_parity_certification": "audit/v2/e_projection_core_v1_release_v2/feature_parity_certification.json",
        "historical_transfer_certification": "audit/v2/e_projection_core_v1_release_v2/historical_transfer_certification.json",
        "applicability_domain_reference": "audit/v2/e_projection_core_v1_release/applicability_domain_reference_provenance.json",
    }
    artifacts = {key: artifact(root, value) for key, value in {**required, **optional_after_reference}.items()}
    reference_inventory = pd.read_csv(
        root / "audit/v2/phase6a_cds_bias_reference_v1/cds_bias_reference_request_inventory.tsv",
        sep="\t",
        dtype=str,
    )
    reference_receipts = list((root / REFERENCE_CACHE / "requests").glob("*/*.json"))
    reference_complete = len(reference_receipts) == len(reference_inventory) == 907
    raw = read_json(root / required["raw_archive_certification"])
    daily = read_json(root / required["daily_normalization_certification"])
    cross = read_json(root / required["cross_provider_provenance"])
    soil = read_json(root / required["soil_policy_certification"])
    backcast = read_json(root / required["historical_ERA5_backcast"])
    bias_parameters = (
        read_json(root / optional_after_reference["bias_adjustment_parameters"])
        if artifacts["bias_adjustment_parameters"]["exists"]
        else {}
    )
    cmip_backcast = (
        read_json(root / optional_after_reference["historical_CMIP6_backcast"])
        if artifacts["historical_CMIP6_backcast"]["exists"]
        else {}
    )
    parity = (
        read_json(root / optional_after_reference["feature_parity_certification"])
        if artifacts["feature_parity_certification"]["exists"]
        else {}
    )
    transfer = (
        read_json(root / optional_after_reference["historical_transfer_certification"])
        if artifacts["historical_transfer_certification"]["exists"]
        else {}
    )
    checks = {
        "required_staging_artifacts_present": all(artifacts[key]["exists"] for key in required),
        "cmip6_raw_455_certified": raw["cmip6_assets"] == 455,
        "cmip6_historical_91_cover_reference": bool(
            raw["cmip6_historical_assets"] == 91
            and raw["checks"]["cmip6_historical_reference_coverage"]
        ),
        "trial_window_raw_archives_certified": bool(
            raw["provider_summary"]["cds_trial_window_checks_pass"]
            and raw["provider_summary"]["openmeteo_trial_window_checks_pass"]
        ),
        "daily_normalization_pass": daily["status"] == "PASS",
        "cross_provider_diagnostic_pass": cross["status"] == "PASS",
        "soil_policy_pass": soil["status"] == "PASS",
        "era5_historical_backcast_pass": backcast["status"] == "PASS",
        "continuous_CDS_1981_2010_reference_complete": reference_complete,
        "bias_reference_normalized": artifacts["bias_reference_normalization"]["exists"],
        "historical_only_bias_parameters_frozen": bool(
            bias_parameters.get("status") == "PASS"
            and bias_parameters.get("protocol_version") == "phase6a_bias_adjustment_v2"
        ),
        "bias_adjusted_historical_CMIP6_backcast_complete": bool(
            cmip_backcast.get("status") == "PASS"
            and cmip_backcast.get("protocol_version")
            == "phase6a_bias_adjusted_cmip6_historical_backcast_v2"
        ),
        "feature_parity_certified": parity.get("status") == "PASS",
        "historical_transfer_certified": transfer.get("status") == "PASS",
        "applicability_domain_reference_frozen": artifacts[
            "applicability_domain_reference"
        ]["exists"],
        "no_future_covariate_matrix": not (root / "environment/v2/e_projection_core_v1_future").exists(),
        "no_future_prediction": not (root / "trained_models/e_projection_core_v1_future").exists(),
    }
    blockers = [key for key, value in checks.items() if not value]
    ready = not blockers
    status = "PASS_READY_TO_GENERATE_MEMBER_RESOLVED_FUTURE_COVARIATES" if ready else "BLOCKED_E_PROJECTION_CORE_V1_INCOMPLETE"
    inventory = pd.DataFrame(
        [
            {"artifact": key, **value}
            for key, value in artifacts.items()
        ]
    )
    inventory_path = output / "e_projection_core_v1_artifact_inventory.tsv"
    atomic_tsv(inventory_path, inventory)
    result = {
        "status": status,
        "protocol_version": "e_projection_core_v1_readiness_v2",
        "selection_data": "historical_climate_environment_and_static_metadata_only",
        "reference_request_count": len(reference_inventory),
        "reference_completed_request_count": len(reference_receipts),
        "reference_pending_request_count": len(reference_inventory) - len(reference_receipts),
        "checks": checks,
        "blockers": blockers,
        "artifacts": {
            "e_projection_core_v1_artifact_inventory.tsv": sha256_file(inventory_path)
        },
        "future_covariate_generation_allowed": ready,
        "future_prediction_allowed": False,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(output / "E_PROJECTION_CORE_V1_READINESS.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
