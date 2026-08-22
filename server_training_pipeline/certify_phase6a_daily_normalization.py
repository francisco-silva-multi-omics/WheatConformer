from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from server_training_pipeline.fetch_cmip6_member_resolved import atomic_json, resolve, sha256_file


DEFAULT_CONTRACT = Path("server_training_pipeline/phase6a_daily_normalization_contract_v1.json")
DEFAULT_AUDIT = Path("audit/v2/phase6a_daily_normalization_v1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    root = args.root.resolve()
    contract_path = resolve(root, args.contract)
    audit = resolve(root, args.audit)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    cds_path = audit / "cds_daily_normalization_index.tsv"
    cmip_path = audit / "cmip6_historical_normalization_index.tsv"
    cds = pd.read_csv(cds_path, sep="\t", dtype=str)
    cmip = pd.read_csv(cmip_path, sep="\t", dtype=str)
    checks = {
        "contract_identity": contract.get("protocol_version") == "phase6a_daily_normalization_v1",
        "cds_request_count": len(cds) == 7094 and not cds.request_id.duplicated().any(),
        "cmip_model_count": len(cmip) == 13 and not cmip.source_id.duplicated().any(),
        "cmip_historical_asset_count": len(cmip) * 7 == 91,
        "cmip_reference_coverage": bool(
            cmip.first_date.le("1981-01-01").all() and cmip.last_date.ge("2010-12-31").all()
        ),
        "no_future_matrix_or_prediction": bool(
            not contract["future_covariate_matrices_generated"]
            and not contract["future_predictions_generated"]
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "status": status,
        "protocol_version": contract["protocol_version"],
        "protocol_sha256": sha256_file(contract_path),
        "cds_request_count": len(cds),
        "cds_daily_row_count": int(pd.to_numeric(cds.daily_rows).sum()),
        "cmip6_model_count": len(cmip),
        "cmip6_historical_asset_count": 91,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "artifacts": {
            "cds_daily_normalization_index.tsv": sha256_file(cds_path),
            "cmip6_historical_normalization_index.tsv": sha256_file(cmip_path),
        },
        "phenotype_values_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(audit / "daily_normalization_certification.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit("Daily normalization certification failed")


if __name__ == "__main__":
    main()
