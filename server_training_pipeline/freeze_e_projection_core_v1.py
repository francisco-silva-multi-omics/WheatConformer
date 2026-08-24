from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from server_training_pipeline.fetch_cmip6_member_resolved import atomic_json, atomic_tsv, resolve, sha256_file


DEFAULT_READINESS = Path("audit/v2/e_projection_core_v1_readiness/E_PROJECTION_CORE_V1_READINESS.json")
DEFAULT_OUTPUT = Path("audit/v2/e_projection_core_v1_release_v2")


REQUIRED_ARTIFACTS = (
    "audit/v2/phase6a_projection_core_raw_archive_v1/raw_archive_certification.json",
    "audit/v2/phase6a_daily_normalization_v1/daily_normalization_certification.json",
    "audit/v2/phase6a_daily_normalization_v1/cds_bias_reference/cds_bias_reference_daily_normalization_provenance.json",
    "audit/v2/phase6a_cross_provider_audit_v1/cross_provider_provenance.json",
    "audit/v2/phase6a_soilgrids_missing_resolution_v1/soilgrids_missing_resolution_certification.json",
    "server_training_pipeline/phase6a_daily_normalization_contract_v1.json",
    "server_training_pipeline/phase6a_bias_adjustment_contract_v2.json",
    "server_training_pipeline/phase6a_projection_core_feature_contract_v1.json",
    "server_training_pipeline/phase6a_historical_transfer_contract_v2.json",
    "server_training_pipeline/phase6a_applicability_domain_contract_v1.json",
    "audit/v2/phase6a_bias_adjustment_v2/bias_adjustment_provenance.json",
    "audit/v2/phase6a_projection_core_historical_backcast_v1/historical_projection_core_backcast_provenance.json",
    "audit/v2/phase6a_projection_core_historical_backcast_v2/cmip6_historical_backcast_provenance.json",
    "audit/v2/e_projection_core_v1_release_v2/feature_parity_certification.json",
    "audit/v2/e_projection_core_v1_release_v2/historical_transfer_certification.json",
    "audit/v2/e_projection_core_v1_release/applicability_domain_reference_provenance.json",
    "audit/v2/phase6a_cmip6_metadata_inventory_v1/cmip6_selected_asset_manifest.tsv",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    readiness_path = resolve(root, args.readiness)
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "PASS_READY_TO_GENERATE_MEMBER_RESOLVED_FUTURE_COVARIATES"
        or readiness.get("future_covariate_generation_allowed") is not True
        or readiness.get("blockers")
    ):
        raise ValueError("E_PROJECTION_CORE_V1 readiness gate has not passed")
    rows = []
    for relative in REQUIRED_ARTIFACTS:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Missing E_PROJECTION_CORE_V1 release artifact: {relative}")
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    output = resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    manifest_path = output / "E_PROJECTION_CORE_V1_CLOSING_MANIFEST.tsv"
    atomic_tsv(manifest_path, manifest)
    result = {
        "status": "PASS_E_PROJECTION_CORE_V1_REMEDIATED_HISTORICAL_TRANSFER_CERTIFIED",
        "release_id": "E_PROJECTION_CORE_V1_REMEDIATED_V2",
        "selection_data": "historical_climate_environment_and_static_metadata_only",
        "readiness_sha256": sha256_file(readiness_path),
        "closing_manifest_sha256": sha256_file(manifest_path),
        "artifact_count": len(manifest),
        "future_feature_identity": "source_id_x_member_id_x_SSP_x_location_id_x_period",
        "member_dimension_must_remain_resolved": True,
        "future_covariate_generation_allowed": True,
        "future_prediction_allowed": False,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(output / "E_PROJECTION_CORE_V1_RELEASE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
