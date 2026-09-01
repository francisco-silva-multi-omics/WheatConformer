from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONTRACT_DIR = Path(
    "audit/v2/stage1_v2_phase6_phenology_readiness_v1/horizon_extension_contract"
)
NORMALIZATION_CONTRACT = Path(
    "server_training_pipeline/phase6a_daily_normalization_contract_v1.json"
)
OUTPUT = Path(
    "audit/v2/stage1_v2_phase6_phenology_readiness_v1/reference_reuse_certification"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def certify_reuse_inventory(
    root: Path,
    reuse: pd.DataFrame,
    physical_domains: dict[str, list[float]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    variables = list(physical_domains)
    required_columns = [
        "date",
        "site_id",
        "latitude",
        "longitude",
        "required_climate_complete",
        *variables,
        *(f"{variable}_available" for variable in variables),
    ]
    rows: list[dict[str, Any]] = []
    grouped = list(reuse.groupby("reference_daily_path", sort=True))
    for file_number, (relative_path, requests) in enumerate(grouped, start=1):
        path = root / str(relative_path)
        expected_sha = requests["reference_daily_sha256"].drop_duplicates().tolist()
        file_hash_exact = (
            path.is_file()
            and len(expected_sha) == 1
            and sha256_file(path) == expected_sha[0]
        )
        if not file_hash_exact:
            for request in requests.itertuples(index=False):
                rows.append(
                    {
                        "request_id": request.request_id,
                        "reference_site_id": request.reference_site_id,
                        "status": "FAIL",
                        "detail": "reference file absent or checksum mismatch",
                    }
                )
            continue
        frame = pd.read_parquet(path, columns=required_columns)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        full_dates_valid = frame["date"].is_unique and frame["date"].is_monotonic_increasing
        frame = frame.set_index("date", drop=False)
        for request in requests.itertuples(index=False):
            start = pd.Timestamp(request.request_start_date)
            end = pd.Timestamp(request.request_end_date)
            expected_dates = pd.date_range(start, end, freq="D")
            selected = frame.loc[frame.index.intersection(expected_dates)].copy()
            date_axis_exact = len(selected) == len(expected_dates) and np.array_equal(
                selected.index.values.astype("datetime64[D]"),
                expected_dates.values.astype("datetime64[D]"),
            )
            site_axis_exact = (
                selected["site_id"].astype(str).eq(str(request.reference_site_id)).all()
                and np.isclose(
                    selected["latitude"].astype(float), float(request.latitude), atol=5e-6
                ).all()
                and np.isclose(
                    selected["longitude"].astype(float),
                    float(request.longitude),
                    atol=5e-6,
                ).all()
            )
            complete = bool(selected["required_climate_complete"].astype(bool).all())
            finite = bool(np.isfinite(selected[variables].to_numpy(dtype=float)).all())
            availability = bool(
                selected[[f"{variable}_available" for variable in variables]]
                .astype(bool)
                .all()
                .all()
            )
            domains = True
            for variable, bounds in physical_domains.items():
                values = selected[variable].astype(float)
                domains = domains and bool(values.between(bounds[0], bounds[1]).all())
            passed = all(
                [
                    full_dates_valid,
                    date_axis_exact,
                    site_axis_exact,
                    complete,
                    finite,
                    availability,
                    domains,
                ]
            )
            rows.append(
                {
                    "request_id": request.request_id,
                    "reference_site_id": request.reference_site_id,
                    "request_start_date": request.request_start_date,
                    "request_end_date": request.request_end_date,
                    "expected_daily_rows": len(expected_dates),
                    "observed_daily_rows": len(selected),
                    "reference_file_sha256_exact": file_hash_exact,
                    "reference_full_date_axis_unique_monotonic": full_dates_valid,
                    "requested_date_axis_exact": date_axis_exact,
                    "site_axis_exact": site_axis_exact,
                    "required_climate_complete": complete,
                    "required_availability_flags_complete": availability,
                    "required_values_finite": finite,
                    "physical_domains_pass": domains,
                    "status": "PASS" if passed else "FAIL",
                    "detail": "" if passed else "one or more reuse checks failed",
                }
            )
        if file_number % 50 == 0 or file_number == len(grouped):
            print(
                f"CERTIFY reference reuse {file_number}/{len(grouped)} files",
                flush=True,
            )
    checks = pd.DataFrame(rows).sort_values("request_id", kind="stable")
    summary = {
        "request_count": len(checks),
        "unique_reference_file_count": reuse["reference_daily_path"].nunique(),
        "expected_daily_row_count": int(checks["expected_daily_rows"].sum()),
        "observed_daily_row_count": int(checks["observed_daily_rows"].sum()),
        "pass_count": int(checks["status"].eq("PASS").sum()),
        "fail_count": int(checks["status"].eq("FAIL").sum()),
    }
    return checks, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Certify reuse of continuous CDS daily data for phenology"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = (args.code_root or root).resolve()
    contract_dir = root / CONTRACT_DIR
    contract_path = contract_dir / "environment_source_contract.json"
    reuse_path = contract_dir / "certified_CDS_reference_reuse_inventory.tsv"
    normalization_path = code_root / NORMALIZATION_CONTRACT
    contract = read_json(contract_path)
    normalization = read_json(normalization_path)
    expected_reuse_hash = contract["artifacts"][reuse_path.name]
    if sha256_file(reuse_path) != expected_reuse_hash:
        raise ValueError("Certified CDS reuse inventory checksum changed")
    reuse = pd.read_csv(reuse_path, sep="\t", dtype=str)
    checks, summary = certify_reuse_inventory(
        root, reuse, normalization["physical_domains"]
    )
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    checks_path = output / "cds_reference_reuse_request_checks.tsv"
    checks.to_csv(checks_path, sep="\t", index=False, lineterminator="\n")
    decision_checks = {
        "contract_protocol_reuse_first": contract.get("protocol_version")
        == "stage1_v2_phase6_phenology_daily_horizon_extension_v2_reuse_first",
        "request_count_exact_3804": summary["request_count"] == 3804,
        "all_requested_slices_pass": summary["fail_count"] == 0,
        "all_requested_slices_are_90_days": checks["expected_daily_rows"].eq(90).all(),
        "all_reference_checksums_exact": checks[
            "reference_file_sha256_exact"
        ].all(),
        "all_date_axes_exact": checks["requested_date_axis_exact"].all(),
        "all_site_axes_exact": checks["site_axis_exact"].all(),
        "all_required_values_finite": checks["required_values_finite"].all(),
        "all_physical_domains_pass": checks["physical_domains_pass"].all(),
    }
    decision_checks = {key: bool(value) for key, value in decision_checks.items()}
    failed = [key for key, value in decision_checks.items() if not value]
    decision = {
        "status": (
            "PASS_CDS_REFERENCE_REUSE_3804_REQUESTS_CERTIFIED"
            if not failed
            else "FAIL_CDS_REFERENCE_REUSE_CERTIFICATION"
        ),
        "protocol_version": "stage1_v2_phase6_phenology_CDS_reference_reuse_v1",
        "selection_data": "historical_climate_identifiers_dates_and_values_only",
        **summary,
        "checks": decision_checks,
        "failed_checks": failed,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "future_SSP_values_read": False,
        "future_predictions_generated": 0,
        "artifacts": {
            reuse_path.name: sha256_file(reuse_path),
            normalization_path.name: sha256_file(normalization_path),
            checks_path.name: sha256_file(checks_path),
        },
    }
    write_json(output / "CDS_REFERENCE_REUSE_CERTIFICATION.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"CDS reference reuse certification failed: {failed}")


if __name__ == "__main__":
    main()
