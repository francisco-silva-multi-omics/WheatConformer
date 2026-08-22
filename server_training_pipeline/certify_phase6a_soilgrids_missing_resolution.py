from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from server_training_pipeline.phase6a_environment_source_recovery import (
    SOILGRIDS_DEPTHS,
    SOILGRIDS_PROPERTIES,
    read_json,
    sha256_file,
    write_json_atomic,
)
from server_training_pipeline.resolve_phase6a_soilgrids_missing import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_PROTOCOL,
    DEFAULT_SOURCE_CACHE,
    PROTOCOL_VERSION,
    TERMINAL_STATUSES,
    load_frozen_inputs,
    resolve,
)


def python_bool(value: Any) -> bool:
    return bool(value)


def certify(
    root: Path,
    protocol_path: Path,
    source_cache: Path,
    audit_dir: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    protocol, freeze, inventory = load_frozen_inputs(
        protocol_path, source_cache, audit_dir
    )
    index_path = cache_dir / "soilgrids_missing_resolution_index.tsv"
    provenance_path = cache_dir / "soilgrids_missing_resolution_provenance.json"
    if not index_path.is_file() or not provenance_path.is_file():
        raise SystemExit("SoilGrids missing-resolution outputs are absent")
    index = pd.read_csv(index_path, sep="\t", dtype=str)
    provenance = read_json(provenance_path)
    statuses = index["status"].astype(str).str.removeprefix("CACHED_")
    accepted = index[statuses.eq("RECOVERED_NEAREST_VALID_SOIL_CELL")].copy()
    review = index[statuses.eq("CANDIDATE_DISTANCE_REVIEW")].copy()
    masked = index[statuses.eq("MASKED_NO_VALID_SOIL_CELL_WITHIN_RADIUS")].copy()

    value_checks: list[bool] = []
    value_hash_checks: list[bool] = []
    for frame_row in pd.concat([accepted, review]).itertuples(index=False):
        values_path = cache_dir / str(frame_row.values_path)
        values = pd.read_parquet(values_path)
        value_hash_checks.append(
            values_path.is_file() and sha256_file(values_path) == str(frame_row.values_sha256)
        )
        finite = np.isfinite(values["canonical_value"].to_numpy(float)).all()
        complete = len(values) == len(SOILGRIDS_PROPERTIES) * len(SOILGRIDS_DEPTHS)
        unique_layers = not values.duplicated(["property", "depth"]).any()
        pivot = values.pivot(index="depth", columns="property", values="canonical_value")
        expected_columns = set(SOILGRIDS_PROPERTIES)
        expected_depths = set(SOILGRIDS_DEPTHS)
        physical = (
            set(pivot.columns) == expected_columns
            and set(pivot.index) == expected_depths
            and (pivot["wv0033"] > pivot["wv1500"]).all()
            and pivot["wv0033"].between(0, 100, inclusive="right").all()
            and pivot["wv1500"].between(0, 100, inclusive="right").all()
            and pivot["cfvo"].between(0, 100, inclusive="both").all()
            and pivot["bdod"].between(0, 3, inclusive="right").all()
        )
        value_checks.append(python_bool(finite and complete and unique_layers and physical))

    accepted_distances = pd.to_numeric(accepted["soil_cell_distance_m"], errors="coerce")
    review_distances = pd.to_numeric(review["soil_cell_distance_m"], errors="coerce")
    nonaccepted_mask = (
        index.loc[~statuses.eq("RECOVERED_NEAREST_VALID_SOIL_CELL"), "explicit_soil_missing_mask"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    fallback_mask = (
        pd.concat([accepted, review])["explicit_soil_fallback_mask"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    index_sha = sha256_file(index_path)
    checks = {
        "frozen_resolver_identity_preserved": sha256_file(
            Path(__file__).with_name("resolve_phase6a_soilgrids_missing.py")
        )
        == freeze["resolver_sha256"],
        "source_site_set_exact": set(index["site_id"]) == set(inventory["site_id"]),
        "one_row_per_source_site": len(index) == len(inventory)
        and index["site_id"].is_unique,
        "all_sites_terminal": statuses.isin(TERMINAL_STATUSES).all(),
        "provenance_complete": provenance.get("run_status") == "COMPLETE",
        "provenance_index_identity": provenance.get("resolution_index_sha256")
        == index_sha,
        "accepted_distance_bound": accepted_distances.notna().all()
        and accepted_distances.le(float(protocol["automatic_acceptance_distance_m"])).all(),
        "review_distance_band": review_distances.notna().all()
        and review_distances.gt(float(protocol["automatic_acceptance_distance_m"])).all()
        and review_distances.le(float(protocol["maximum_search_radius_m"])).all(),
        "recovered_value_files_preserved": all(value_hash_checks),
        "recovered_values_complete_finite_physical": all(value_checks),
        "fallback_mask_present": fallback_mask.all(),
        "explicit_mask_for_nonaccepted": nonaccepted_mask.all(),
        "no_observation_exclusion": provenance.get("observations_excluded") == 0,
        "no_protected_or_prediction_access": all(
            provenance.get(key) in {False, 0}
            for key in (
                "phenotype_values_read",
                "inner_validation_metrics_read",
                "outer_test_outcomes_read",
                "outer_test_metrics_read",
                "final_holdout_outcomes_read",
                "future_climate_values_read",
                "future_covariate_matrices_generated",
                "future_predictions_generated",
            )
        ),
    }
    checks = {key: python_bool(value) for key, value in checks.items()}
    status_environment_counts = (
        index.assign(
            terminal_status=statuses,
            mapped_environment_count_numeric=pd.to_numeric(
                index["mapped_environment_count"], errors="raise"
            ),
        )
        .groupby("terminal_status", sort=True)["mapped_environment_count_numeric"]
        .sum()
        .astype(int)
        .to_dict()
    )
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol_version": PROTOCOL_VERSION,
        "selection_data": protocol["selection_data"],
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_climate_values_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
        "observations_excluded": 0,
        "source_missing_sites": len(index),
        "accepted_nearest_cell_sites": len(accepted),
        "distance_review_sites": len(review),
        "no_valid_local_cell_sites": len(masked),
        "explicit_mask_sites": len(review) + len(masked),
        "affected_environment_count": int(
            pd.to_numeric(index["mapped_environment_count"], errors="raise").sum()
        ),
        "affected_environment_count_by_status": {
            str(key): int(value) for key, value in status_environment_counts.items()
        },
        "accepted_distance_m": {
            "minimum": float(accepted_distances.min()),
            "median": float(accepted_distances.median()),
            "maximum": float(accepted_distances.max()),
        },
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "certifier_sha256": sha256_file(Path(__file__)),
        "artifacts": {
            "soilgrids_missing_resolution_freeze.json": sha256_file(
                audit_dir / "soilgrids_missing_resolution_freeze.json"
            ),
            "soilgrids_missing_site_inventory.tsv": sha256_file(
                audit_dir / "soilgrids_missing_site_inventory.tsv"
            ),
            "soilgrids_missing_resolution_index.tsv": index_sha,
            "soilgrids_missing_resolution_provenance.json": sha256_file(
                provenance_path
            ),
        },
    }
    output_path = audit_dir / "soilgrids_missing_resolution_certification.json"
    write_json_atomic(output_path, result)
    if result["status"] != "PASS":
        raise SystemExit("SoilGrids missing-resolution certification failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    root = args.root.resolve()
    result = certify(
        root,
        resolve(root, args.protocol),
        resolve(root, args.source_cache),
        resolve(root, args.audit_dir),
        resolve(root, args.cache_dir),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
