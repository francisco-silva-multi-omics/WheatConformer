from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from server_training_pipeline.fetch_cmip6_member_resolved import (
    atomic_json,
    atomic_tsv,
    resolve,
    sha256_file,
)
from server_training_pipeline.fit_phase6a_applicability_domain_reference import feature_block


DEFAULT_PROTOCOL = Path("server_training_pipeline/phase6b_future_covariate_protocol_v1.json")
DEFAULT_LOCK = Path(
    "audit/v2/e_projection_core_v1_future_covariates_v1_freeze/"
    "future_covariate_generation_lock.json"
)
DEFAULT_LOCATIONS = Path(
    "audit/v2/e_projection_core_v1_future_covariates_v1_freeze/"
    "future_projection_location_manifest.tsv"
)
DEFAULT_PLAN = Path(
    "audit/v2/e_projection_core_v1_future_covariates_v1_freeze/"
    "future_covariate_generation_plan.tsv"
)
DEFAULT_INDEX = Path(
    "audit/v2/e_projection_core_v1_future_covariates_v1/future_covariate_matrix_index.tsv"
)
DEFAULT_GENERATION = Path(
    "audit/v2/e_projection_core_v1_future_covariates_v1/"
    "future_covariate_generation_provenance.json"
)
DEFAULT_REFERENCE = Path("environment/v2/e_projection_core_v1_applicability_domain_reference")
DEFAULT_OUTPUT = Path("audit/v2/e_projection_core_v1_future_covariates_v1_certification")
DEFAULT_RELEASE = Path("audit/v2/e_projection_core_v1_future_covariates_v1_release")


CLASS_ORDER = {"SUPPORTED": 0, "EXTRAPOLATIVE": 1, "UNSUPPORTED": 2}


def classify(
    required_missing: np.ndarray,
    range_fraction: np.ndarray,
    robust_rms: np.ndarray,
    mahalanobis: np.ndarray | None,
    thresholds: dict[str, float] | None,
) -> np.ndarray:
    result = np.full(len(required_missing), "UNSUPPORTED", dtype=object)
    supported = (~required_missing) & (range_fraction <= 0.05) & (robust_rms <= 4.0)
    extrapolative = (~required_missing) & (range_fraction <= 0.20) & (robust_rms <= 8.0)
    if mahalanobis is not None and thresholds is not None:
        supported &= mahalanobis <= float(thresholds["mahalanobis_99"])
        extrapolative &= mahalanobis <= 1.5 * float(thresholds["mahalanobis_999"])
    result[extrapolative] = "EXTRAPOLATIVE"
    result[supported] = "SUPPORTED"
    return result


def feature_diagnostics(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    reference_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = reference.feature.tolist()
    ordered = reference.set_index("feature").loc[features]
    matrix = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    medians = ordered["median"].to_numpy(dtype=float)
    scales = ordered["robust_scale"].to_numpy(dtype=float)
    minimum = ordered["minimum"].to_numpy(dtype=float)
    maximum = ordered["maximum"].to_numpy(dtype=float)
    finite = np.isfinite(matrix)
    range_exceed = finite & ((matrix < minimum) | (matrix > maximum))
    range_fraction = np.divide(
        range_exceed.sum(axis=1),
        finite.sum(axis=1),
        out=np.ones(len(frame), dtype=float),
        where=finite.sum(axis=1) > 0,
    )
    robust = (matrix - medians) / scales
    robust_rms = np.sqrt(
        np.divide(
            np.where(finite, robust * robust, 0.0).sum(axis=1),
            finite.sum(axis=1),
            out=np.full(len(frame), np.inf),
            where=finite.sum(axis=1) > 0,
        )
    )
    filled = matrix.copy()
    missing = ~np.isfinite(filled)
    filled[missing] = np.broadcast_to(medians, filled.shape)[missing]
    standardized = (filled - medians) / scales
    components = np.load(reference_root / "pca_components.npy")
    pca_center = np.load(reference_root / "pca_center.npy")
    location = np.load(reference_root / "mahalanobis_location.npy")
    precision = np.load(reference_root / "mahalanobis_precision.npy")
    scores = (standardized - pca_center) @ components.T
    centered = scores - location
    mahalanobis = np.sqrt(np.einsum("ij,jk,ik->i", centered, precision, centered))
    thresholds = json.loads(
        (reference_root / "historical_distance_thresholds.json").read_text(encoding="utf-8")
    )
    required_missing = ~frame.projection_core_climate_eligible.astype(bool).to_numpy()
    overall_class = classify(
        required_missing, range_fraction, robust_rms, mahalanobis, thresholds
    )
    overall = frame[
        ["matrix_id", "source_id", "member_id", "SSP", "period", "location_id"]
    ].copy()
    overall["feature_block"] = "overall"
    overall["feature_count"] = len(features)
    overall["missing_feature_fraction"] = missing.mean(axis=1)
    overall["univariate_range_exceedance_fraction"] = range_fraction
    overall["robust_standardized_rms"] = robust_rms
    overall["mahalanobis_distance"] = mahalanobis
    overall["historical_mahalanobis_99"] = thresholds["mahalanobis_99"]
    overall["historical_mahalanobis_999"] = thresholds["mahalanobis_999"]
    overall["daily_extreme_fraction"] = frame.daily_extreme_fraction.to_numpy(dtype=float)
    overall["required_input_missing"] = required_missing
    overall["support_class"] = overall_class
    blocks = []
    block_labels = ordered.feature_block.to_numpy(dtype=str)
    for block in sorted(set(block_labels)):
        selected = block_labels == block
        block_finite = finite[:, selected]
        block_range = range_exceed[:, selected]
        block_robust = robust[:, selected]
        block_range_fraction = np.divide(
            block_range.sum(axis=1),
            block_finite.sum(axis=1),
            out=np.ones(len(frame), dtype=float),
            where=block_finite.sum(axis=1) > 0,
        )
        block_rms = np.sqrt(
            np.divide(
                np.where(block_finite, block_robust * block_robust, 0.0).sum(axis=1),
                block_finite.sum(axis=1),
                out=np.full(len(frame), np.inf),
                where=block_finite.sum(axis=1) > 0,
            )
        )
        block_frame = overall.copy()
        block_frame["feature_block"] = block
        block_frame["feature_count"] = int(selected.sum())
        block_frame["missing_feature_fraction"] = (~block_finite).mean(axis=1)
        block_frame["univariate_range_exceedance_fraction"] = block_range_fraction
        block_frame["robust_standardized_rms"] = block_rms
        block_frame["mahalanobis_distance"] = np.nan
        block_frame["historical_mahalanobis_99"] = np.nan
        block_frame["historical_mahalanobis_999"] = np.nan
        block_frame["support_class"] = classify(
            required_missing,
            block_range_fraction,
            block_rms,
            mahalanobis=None,
            thresholds=None,
        )
        blocks.append(block_frame)
    return overall, pd.concat([overall, *blocks], ignore_index=True)


def physical_checks(frame: pd.DataFrame) -> dict[str, bool]:
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    humidity = frame.filter(regex="__relative_humidity_mean_percent$").to_numpy(dtype=float)
    nonnegative = frame.filter(
        regex=(
            "__(precipitation_sum_mm|wet_day_count|dry_day_count|radiation_sum_mj_m2|"
            "radiation_mean_mj_m2_day|gdd_base[0-9]+_sum|frost_day_count|heat_day_30_count|"
            "extreme_heat_day_35_count|high_vpd_day_2kpa_count|pet_sum_mm)$"
        )
    ).to_numpy(dtype=float)
    counts = frame.filter(regex="__(wet_day_count|dry_day_count|frost_day_count|heat_day_30_count|extreme_heat_day_35_count|high_vpd_day_2kpa_count)$").to_numpy(dtype=float)
    checks = {
        "no_infinite_numeric_values": not np.isinf(numeric).any(),
        "humidity_in_physical_domain": bool(
            np.all((humidity[np.isfinite(humidity)] >= 0.0) & (humidity[np.isfinite(humidity)] <= 100.0))
        ),
        "nonnegative_positive_domain_features": bool(
            np.all(nonnegative[np.isfinite(nonnegative)] >= -1e-8)
        ),
        "thirty_day_event_counts_bounded": bool(
            np.all((counts[np.isfinite(counts)] >= -1e-8) & (counts[np.isfinite(counts)] <= 30.0 + 1e-8))
        ),
    }
    return checks


def build_gcm_agreement(
    frame: pd.DataFrame, reference: pd.DataFrame, output: Path
) -> tuple[Path, int]:
    features = reference.feature.tolist()
    historical_medians = reference.set_index("feature").loc[features, "median"].to_numpy(dtype=float)
    identity = ["SSP", "period", "location_id"]
    ordered = frame.sort_values(identity + ["source_id"]).reset_index(drop=True)
    group_sizes = ordered.groupby(identity, sort=True).size()
    if not group_sizes.eq(13).all():
        raise ValueError("GCM agreement requires exactly 13 member-resolved sources per identity")
    keys = ordered[identity].drop_duplicates().reset_index(drop=True)
    values = (
        ordered[features]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
        .to_numpy()
        .reshape(len(keys), 13, len(features))
    )
    finite = np.isfinite(values)
    source_count = finite.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        quantiles = np.nanquantile(values, [0.25, 0.5, 0.75], axis=1)
        minimum = np.nanmin(values, axis=1)
        maximum = np.nanmax(values, axis=1)
    delta = values - historical_medians[None, None, :]
    positive = ((delta > 0) & finite).sum(axis=1)
    negative = ((delta < 0) & finite).sum(axis=1)
    agreement_parts = []
    for feature_index, feature in enumerate(features):
        summary = keys.copy()
        summary["source_count"] = source_count[:, feature_index]
        summary["minimum"] = minimum[:, feature_index]
        summary["q25"] = quantiles[0, :, feature_index]
        summary["median"] = quantiles[1, :, feature_index]
        summary["q75"] = quantiles[2, :, feature_index]
        summary["maximum"] = maximum[:, feature_index]
        summary["positive"] = positive[:, feature_index]
        summary["negative"] = negative[:, feature_index]
        summary["finite"] = source_count[:, feature_index]
        summary["feature"] = feature
        summary["feature_block"] = feature_block(feature)
        summary["historical_reference_median"] = historical_medians[feature_index]
        summary["iqr"] = summary.q75 - summary.q25
        summary["range"] = summary.maximum - summary.minimum
        summary["sign_agreement_fraction"] = np.divide(
            np.maximum(summary.positive, summary.negative),
            summary.finite,
            out=np.full(len(summary), np.nan),
            where=summary.finite > 0,
        )
        summary["dominant_change_direction"] = np.select(
            [summary.positive > summary.negative, summary.negative > summary.positive],
            ["INCREASE", "DECREASE"],
            default="TIED_OR_NO_CHANGE",
        )
        agreement_parts.append(summary)
    agreement = pd.concat(agreement_parts, ignore_index=True)
    target = output / "future_gcm_agreement.parquet"
    agreement.to_parquet(target, index=False, compression="zstd")
    return target, len(agreement)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--locations", type=Path, default=DEFAULT_LOCATIONS)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--generation", type=Path, default=DEFAULT_GENERATION)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = resolve(root, args.protocol)
    lock_path = resolve(root, args.lock)
    location_path = resolve(root, args.locations)
    plan_path = resolve(root, args.plan)
    index_path = resolve(root, args.index)
    generation_path = resolve(root, args.generation)
    reference_root = resolve(root, args.reference)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    plan = pd.read_csv(plan_path, sep="\t", dtype=str)
    index = pd.read_csv(index_path, sep="\t", dtype=str)
    locations = pd.read_csv(location_path, sep="\t", dtype=str)
    reference = pd.read_csv(
        reference_root / "historical_robust_feature_reference.tsv", sep="\t"
    )
    checks: dict[str, bool] = {
        "generation_lock_pass": lock.get("status") == "PASS_FUTURE_COVARIATE_GENERATION_LOCKED",
        "protocol_matches_lock": sha256_file(protocol_path) == lock.get("protocol_sha256"),
        "location_manifest_matches_lock": sha256_file(location_path)
        == lock.get("location_manifest_sha256"),
        "generation_plan_matches_lock": sha256_file(plan_path)
        == lock.get("generation_plan_sha256"),
        "generation_complete": generation.get("status") == "PASS"
        and generation.get("run_status") == "COMPLETE",
        "generation_index_matches_provenance": sha256_file(index_path)
        == generation.get("matrix_index_sha256"),
        "prediction_remains_blocked": generation.get("future_prediction_allowed") is False
        and generation.get("future_predictions_generated") == 0,
        "matrix_count": len(index) == int(protocol["expected_matrix_count"]) == len(plan),
        "matrix_ids_unique": not index.matrix_id.duplicated().any(),
        "matrix_plan_exact": set(index.matrix_id) == set(plan.matrix_id),
    }
    frames = []
    matrix_check_rows = []
    expected_locations = locations.location_id.to_numpy(dtype=str)
    reference_features = reference.feature.tolist()
    for row in index.itertuples(index=False):
        path = root / row.output_path
        checksum_pass = path.is_file() and sha256_file(path) == row.output_sha256
        if not checksum_pass:
            matrix_check_rows.append(
                {"matrix_id": row.matrix_id, "status": "FAIL", "detail": "checksum_or_file"}
            )
            continue
        frame = pd.read_parquet(path)
        identity_pass = all(
            frame[column].astype(str).eq(str(getattr(row, column))).all()
            for column in ("source_id", "member_id", "SSP", "period")
        )
        location_pass = np.array_equal(
            frame.sort_values("location_id").location_id.astype(str).to_numpy(), expected_locations
        )
        schema_pass = not set(reference_features).difference(frame.columns)
        physical = physical_checks(frame)
        passed = (
            len(frame) == int(protocol["expected_location_count"])
            and identity_pass
            and location_pass
            and schema_pass
            and all(physical.values())
        )
        frame["matrix_id"] = row.matrix_id
        frames.append(frame)
        matrix_check_rows.append(
            {
                "matrix_id": row.matrix_id,
                "status": "PASS" if passed else "FAIL",
                "row_count": len(frame),
                "identity_pass": identity_pass,
                "location_axis_pass": location_pass,
                "schema_pass": schema_pass,
                **physical,
            }
        )
    matrix_checks = pd.DataFrame(matrix_check_rows)
    checks["all_matrix_checks_pass"] = len(matrix_checks) == len(index) and matrix_checks.status.eq("PASS").all()
    combined = pd.concat(frames, ignore_index=True)
    overall, diagnostics = feature_diagnostics(combined, reference, reference_root)
    output = resolve(root, args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Future covariate certification directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    matrix_checks_path = output / "future_covariate_matrix_validation.tsv"
    diagnostics_path = output / "future_applicability_diagnostics.parquet"
    overall_path = output / "future_applicability_overall.tsv"
    atomic_tsv(matrix_checks_path, matrix_checks)
    diagnostics.to_parquet(diagnostics_path, index=False, compression="zstd")
    atomic_tsv(overall_path, overall)
    agreement_path, agreement_rows = build_gcm_agreement(combined, reference, output)
    availability = (
        overall.groupby(["SSP", "period", "support_class"], sort=True)
        .agg(location_source_rows=("location_id", "size"), unique_locations=("location_id", "nunique"))
        .reset_index()
    )
    availability_path = output / "future_applicability_summary.tsv"
    atomic_tsv(availability_path, availability)
    checks.update(
        {
            "future_identity_rows": len(combined)
            == int(protocol["expected_matrix_count"]) * int(protocol["expected_location_count"]),
            "applicability_terminal": len(overall) == len(combined)
            and overall.support_class.isin(CLASS_ORDER).all(),
            "member_dimension_retained": combined.groupby(
                ["SSP", "period", "location_id"]
            ).source_id.nunique().eq(int(protocol["expected_source_count"])).all(),
            "gcm_agreement_complete": agreement_rows
            == int(protocol["expected_ssp_count"])
            * int(protocol["expected_period_count"])
            * int(protocol["expected_location_count"])
            * len(reference_features),
            "no_phenotype_or_outcome_access": True,
            "no_predictions_generated": True,
        }
    )
    failed = [name for name, passed in checks.items() if not passed]
    checks = {name: bool(passed) for name, passed in checks.items()}
    certification = {
        "status": "PASS_MEMBER_RESOLVED_FUTURE_COVARIATES_CERTIFIED"
        if not failed
        else "FAIL_MEMBER_RESOLVED_FUTURE_COVARIATE_CERTIFICATION",
        "protocol_version": "phase6b_member_resolved_future_covariate_certification_v1",
        "selection_data": "future_climate_covariates_and_historical_applicability_references_only",
        "checks": checks,
        "failed_checks": failed,
        "matrix_count": len(index),
        "future_identity_row_count": len(combined),
        "feature_count": len(reference_features),
        "applicability_status_counts": overall.support_class.value_counts().sort_index().to_dict(),
        "generation_lock_sha256": sha256_file(lock_path),
        "generation_provenance_sha256": sha256_file(generation_path),
        "matrix_index_sha256": sha256_file(index_path),
        "matrix_validation_sha256": sha256_file(matrix_checks_path),
        "applicability_diagnostics_sha256": sha256_file(diagnostics_path),
        "applicability_overall_sha256": sha256_file(overall_path),
        "applicability_summary_sha256": sha256_file(availability_path),
        "gcm_agreement_sha256": sha256_file(agreement_path),
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": len(index),
        "future_prediction_allowed": False,
        "future_predictions_generated": 0,
    }
    certification_path = output / "future_covariate_certification.json"
    atomic_json(certification_path, certification)
    release = resolve(root, args.release)
    if release.exists() and any(release.iterdir()):
        raise ValueError(f"Future covariate release directory already exists: {release}")
    release.mkdir(parents=True, exist_ok=True)
    support_counts = overall.support_class.value_counts().sort_index().to_dict()
    report_path = release / "PHASE6B_FUTURE_COVARIATE_REPORT.md"
    report_path.write_text(
        "# Phase 6B member-resolved future covariates\n\n"
        f"Status: `{certification['status']}`\n\n"
        f"- Matrices: {len(index)}\n"
        f"- Location rows: {len(combined)}\n"
        f"- Projection-core features audited: {len(reference_features)}\n"
        f"- Supported rows: {support_counts.get('SUPPORTED', 0)}\n"
        f"- Extrapolative rows: {support_counts.get('EXTRAPOLATIVE', 0)}\n"
        f"- Unsupported rows: {support_counts.get('UNSUPPORTED', 0)}\n"
        f"- GCM-agreement rows: {agreement_rows}\n\n"
        "All GCM/member/SSP/location/period identities remain resolved. No phenotype, "
        "evaluation outcome, or model metric was read. No future prediction was generated.\n\n"
        "A separately frozen model specification and prediction protocol is required before "
        "these covariates can be passed to a trained model.\n",
        encoding="utf-8",
    )
    handoff_path = release / "PHASE6B_HANDOFF.md"
    handoff_path.write_text(
        "# Phase 6B handoff\n\n"
        "1. Review applicability classes and GCM disagreement without model outcomes.\n"
        "2. Freeze the exact projection model, calibration policy, reporting periods, and "
        "unsupported-row behavior.\n"
        "3. Keep member-resolved predictions through reporting; apply equal source weighting "
        "only in the reporting layer.\n"
        "4. Do not generate predictions until that new protocol passes.\n",
        encoding="utf-8",
    )
    manifest_rows = []
    artifact_paths = [
        protocol_path,
        root / "server_training_pipeline/phase6a_projection_core_feature_contract_v1.json",
        root / "server_training_pipeline/phase6a_bias_adjustment_contract_v2.json",
        root / "server_training_pipeline/phase6a_applicability_domain_contract_v1.json",
        root / "server_training_pipeline/freeze_phase6b_future_covariates.py",
        root / "server_training_pipeline/build_phase6b_daily_extreme_reference.py",
        root / "server_training_pipeline/build_phase6b_member_resolved_future_covariates.py",
        lock_path,
        location_path,
        plan_path,
        root
        / "audit/v2/e_projection_core_v1_future_covariates_v1_reference/"
        "historical_daily_extreme_reference_provenance.json",
        root
        / "environment/v2/e_projection_core_v1_future_covariates_v1_reference/"
        "historical_daily_extreme_reference.nc",
        generation_path,
        index_path,
        matrix_checks_path,
        diagnostics_path,
        overall_path,
        availability_path,
        agreement_path,
        certification_path,
        report_path,
        handoff_path,
        Path(__file__),
    ] + [root / value for value in index.output_path]
    for path in artifact_paths:
        manifest_rows.append(
            {
                "path": path.resolve().relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    closing_path = release / "FUTURE_COVARIATE_CLOSING_MANIFEST.tsv"
    atomic_tsv(closing_path, pd.DataFrame(manifest_rows))
    decision = {
        **certification,
        "release_id": "E_PROJECTION_CORE_V1_FUTURE_COVARIATES_V1",
        "closing_manifest_sha256": sha256_file(closing_path),
        "closing_artifact_count": len(manifest_rows),
        "future_feature_identity": "source_id_x_member_id_x_SSP_x_location_id_x_period",
        "member_dimension_retained": True,
        "next_gate": "MODEL_SPECIFICATION_AND_PREDICTION_PROTOCOL_MUST_BE_FROZEN_SEPARATELY",
    }
    atomic_json(release / "FUTURE_COVARIATE_RELEASE_DECISION.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    if failed:
        raise SystemExit("Member-resolved future covariate certification failed")


if __name__ == "__main__":
    main()
