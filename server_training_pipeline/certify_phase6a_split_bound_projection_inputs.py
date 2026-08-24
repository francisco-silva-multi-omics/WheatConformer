from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from server_training_pipeline.fetch_cmip6_member_resolved import (
    atomic_json,
    atomic_tsv,
    resolve,
    sha256_file,
)


DEFAULT_PROTOCOL = Path(
    "server_training_pipeline/phase6a_split_bound_projection_inputs_protocol_v1.json"
)
DEFAULT_FREEZE = Path(
    "audit/v2/e_projection_core_v1_split_bound_historical_v1_freeze"
)
DEFAULT_INPUT = Path(
    "environment/v2/e_projection_core_v1_split_bound_historical_v1"
)
DEFAULT_AUDIT = Path(
    "audit/v2/e_projection_core_v1_split_bound_historical_v1"
)
DEFAULT_RELEASE = Path(
    "audit/v2/e_projection_core_v1_split_bound_historical_v1_release"
)


def add_check(
    rows: list[dict[str, object]], check: str, passed: bool, detail: str
) -> None:
    rows.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        }
    )


def deterministic_indices(state_id: str, candidate: np.ndarray, count: int) -> np.ndarray:
    if len(candidate) <= count:
        return candidate
    seed = int.from_bytes(state_id.encode("utf-8")[:8].ljust(8, b"\0"), "little")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(candidate, size=count, replace=False))


def manifest_inventory(root: Path, paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in sorted(set(path.resolve() for path in paths)):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = resolve(root, args.protocol)
    freeze = resolve(root, args.freeze)
    inputs = resolve(root, args.input)
    audit = resolve(root, args.audit)
    release = resolve(root, args.release)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock_path = freeze / "split_bound_projection_input_freeze.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    build_path = audit / "split_bound_projection_input_build_provenance.json"
    build = json.loads(build_path.read_text(encoding="utf-8"))
    schema_path = freeze / "feature_schema.tsv"
    states_path = freeze / "state_manifest.tsv"
    axis_path = freeze / "environment_axis.tsv"
    registry_path = inputs / "split_bound_projection_input_registry.tsv"
    schema = pd.read_csv(schema_path, sep="\t")
    states = pd.read_csv(states_path, sep="\t", dtype=str)
    axis = pd.read_csv(axis_path, sep="\t", dtype={"environment_id": str})
    registry = pd.read_csv(registry_path, sep="\t")
    historical_path = root / lock["historical_feature_path"]
    historical = pd.read_parquet(historical_path)
    aligned = historical.set_index("environment_id").reindex(axis.environment_id)
    features = schema.feature.tolist()
    raw = aligned[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    source_available = aligned.index.to_series().isin(historical.environment_id).to_numpy()
    climate_eligible = aligned["projection_core_climate_eligible"].eq(True).to_numpy(dtype=bool)
    active = source_available & climate_eligible
    raw_missing = ~np.isfinite(raw)
    raw_missing[~source_available, :] = True

    checks: list[dict[str, object]] = []
    add_check(
        checks,
        "protocol_identity",
        protocol.get("protocol_version") == "phase6a_split_bound_historical_projection_inputs_v1"
        and sha256_file(protocol_path) == lock.get("protocol_sha256"),
        protocol.get("protocol_version", "missing"),
    )
    add_check(
        checks,
        "certified_historical_source",
        sha256_file(historical_path) == lock.get("historical_feature_sha256"),
        lock.get("historical_feature_sha256", "missing"),
    )
    add_check(checks, "exact_feature_schema", len(features) == 153, f"features={len(features)}")
    add_check(
        checks,
        "complete_state_grid",
        len(states) == 150 and len(registry) == 150 and set(states.state_id) == set(registry.state_id),
        f"frozen={len(states)}; built={len(registry)}",
    )
    add_check(
        checks,
        "exact_environment_axis",
        len(axis) == 11161 and not axis.environment_id.duplicated().any(),
        f"environments={len(axis)}",
    )
    add_check(
        checks,
        "build_provenance",
        build.get("status") == "PASS"
        and build.get("state_count") == 150
        and build.get("feature_count") == 153
        and build.get("builder_sha256")
        == sha256_file(
            root / "server_training_pipeline/build_phase6a_split_bound_projection_inputs.py"
        ),
        build.get("status", "missing"),
    )

    state_rows: list[dict[str, object]] = []
    inventory_paths = [
        protocol_path,
        root / "server_training_pipeline/freeze_phase6a_split_bound_projection_inputs.py",
        root / "server_training_pipeline/build_phase6a_split_bound_projection_inputs.py",
        Path(__file__).resolve(),
        root / "scripts/v2/run_phase6a_split_bound_projection_inputs.ps1",
        lock_path,
        build_path,
        schema_path,
        states_path,
        axis_path,
        registry_path,
    ]
    max_parameter_delta = 0.0
    max_transform_delta = 0.0
    max_projection_delta = 0.0
    max_mask_mismatch = 0
    failed_states = 0
    for ordinal, state in enumerate(states.itertuples(index=False), start=1):
        state_dir = inputs / "states" / state.state_id
        metadata_path = state_dir / "state_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        artifact_ok = True
        for name, digest in metadata["artifact_sha256"].items():
            path = state_dir / name
            inventory_paths.append(path)
            artifact_ok &= path.is_file() and sha256_file(path) == digest
        inventory_paths.append(metadata_path)
        entities_path = state_dir / "environment_entities.tsv"
        params_path = state_dir / "feature_parameters.tsv"
        standardized_path = state_dir / "standardized_features_float32.npy"
        packed_path = state_dir / "feature_missing_mask_packbits.npy"
        projection_path = state_dir / "kernel_projection_float32.npy"
        factor_path = state_dir / "kernel_factor_float32.npy"
        singular_path = state_dir / "training_singular_values_float64.npy"
        entities = pd.read_csv(entities_path, sep="\t", dtype={"environment_id": str})
        params = pd.read_csv(params_path, sep="\t")
        standardized = np.load(standardized_path, mmap_mode="r")
        packed = np.load(packed_path, mmap_mode="r")
        projection = np.load(projection_path)
        factor = np.load(factor_path, mmap_mode="r")
        singular = np.load(singular_path)

        manifest_path = root / state.manifest_path
        manifest_ok = sha256_file(manifest_path) == state.manifest_sha256
        training_ids = set(pd.read_csv(manifest_path, sep="\t", dtype=str).environment_id)
        training_mask = axis.environment_id.isin(training_ids).to_numpy()
        fit_mask = training_mask & active
        partition_ok = np.array_equal(
            entities.partition.eq("TRAINING").to_numpy(), training_mask
        )
        shape_ok = (
            standardized.shape == (len(axis), len(features))
            and packed.shape == (len(axis), math.ceil(len(features) / 8))
            and projection.shape == (len(features), factor.shape[1])
            and factor.shape[0] == len(axis)
            and len(singular) == factor.shape[1]
            and len(params) == len(features)
        )
        schema_ok = params.feature.tolist() == features

        unpacked = np.unpackbits(packed, axis=1, count=len(features), bitorder="little").astype(bool)
        mask_mismatch = int(np.count_nonzero(unpacked != raw_missing))
        max_mask_mismatch = max(max_mask_mismatch, mask_mismatch)

        retained = params.feature_status.eq("RETAINED").to_numpy()
        training = raw[fit_mask]
        expected_nonmissing = np.isfinite(training).sum(axis=0)
        parameter_delta = 0.0
        for column in range(len(features)):
            finite = training[np.isfinite(training[:, column]), column]
            if retained[column]:
                expected_median = float(np.median(finite))
                imputed = np.where(
                    np.isfinite(training[:, column]), training[:, column], expected_median
                )
                expected_mean = float(imputed.mean())
                expected_scale = float(imputed.std(ddof=0))
                observed = params.iloc[column]
                parameter_delta = max(
                    parameter_delta,
                    abs(float(observed.imputation_median) - expected_median),
                    abs(float(observed.centering_mean_after_imputation) - expected_mean),
                    abs(float(observed.scaling_sd_after_imputation) - expected_scale),
                )
        parameter_nonmissing_ok = np.array_equal(
            params.training_nonmissing.to_numpy(dtype=int), expected_nonmissing
        )
        max_parameter_delta = max(max_parameter_delta, parameter_delta)

        training_indices = deterministic_indices(
            state.state_id, np.flatnonzero(training_mask), 48
        )
        application_indices = deterministic_indices(
            state.state_id + "application", np.flatnonzero(~training_mask), 48
        )
        inactive_indices = deterministic_indices(
            state.state_id + "inactive", np.flatnonzero(~active), 48
        )
        sample = np.unique(
            np.concatenate([training_indices, application_indices, inactive_indices])
        )
        expected_standardized = np.zeros((len(sample), len(features)), dtype=np.float64)
        if retained.any():
            selected = raw[np.ix_(sample, retained)].copy()
            medians = params.loc[retained, "imputation_median"].to_numpy(dtype=float)
            means = params.loc[retained, "centering_mean_after_imputation"].to_numpy(dtype=float)
            scales = params.loc[retained, "scaling_sd_after_imputation"].to_numpy(dtype=float)
            selected[~np.isfinite(selected)] = np.broadcast_to(
                medians, selected.shape
            )[~np.isfinite(selected)]
            expected_standardized[:, retained] = (selected - means) / scales
        expected_standardized[~active[sample], :] = 0.0
        transform_delta = float(
            np.max(
                np.abs(
                    np.asarray(standardized[sample], dtype=np.float64)
                    - expected_standardized
                )
            )
        )
        projection_delta = float(
            np.max(
                np.abs(
                    np.asarray(factor[sample], dtype=np.float64)
                    - expected_standardized @ projection.astype(np.float64)
                )
            )
        )
        max_transform_delta = max(max_transform_delta, transform_delta)
        max_projection_delta = max(max_projection_delta, projection_delta)
        inactive_zero = bool(
            np.all(np.asarray(standardized[~active], dtype=np.float32) == 0)
            and np.all(np.asarray(factor[~active], dtype=np.float32) == 0)
        )
        finite_ok = bool(
            np.isfinite(standardized).all()
            and np.isfinite(projection).all()
            and np.isfinite(factor).all()
            and np.isfinite(singular).all()
        )
        factor_rank_ok = (
            1 <= factor.shape[1] <= protocol["factorization"]["maximum_rank"]
            and np.all(singular > 0)
            and np.all(singular[:-1] >= singular[1:])
        )
        state_ok = all(
            [
                artifact_ok,
                manifest_ok,
                partition_ok,
                shape_ok,
                schema_ok,
                mask_mismatch == 0,
                parameter_nonmissing_ok,
                parameter_delta <= 1e-10,
                transform_delta <= 2e-5,
                projection_delta <= 2e-5,
                inactive_zero,
                finite_ok,
                factor_rank_ok,
            ]
        )
        failed_states += int(not state_ok)
        state_rows.append(
            {
                "state_id": state.state_id,
                "scenario": state.scenario,
                "state_level": state.state_level,
                "training_environments": int(training_mask.sum()),
                "fit_environments": int(fit_mask.sum()),
                "application_environments": int((~training_mask).sum()),
                "active_application_environments": int((~training_mask & active).sum()),
                "retained_features": int(retained.sum()),
                "factor_rank": factor.shape[1],
                "maximum_parameter_delta": parameter_delta,
                "maximum_transform_delta": transform_delta,
                "maximum_projection_delta": projection_delta,
                "mask_mismatch_count": mask_mismatch,
                "status": "PASS" if state_ok else "FAIL",
            }
        )
        print(
            f"[{ordinal:03d}/{len(states)}] {'PASS' if state_ok else 'FAIL'} {state.state_id}",
            flush=True,
        )

    state_certification = pd.DataFrame(state_rows)
    state_certification_path = audit / "split_bound_projection_state_certification.tsv"
    audit.mkdir(parents=True, exist_ok=True)
    atomic_tsv(state_certification_path, state_certification)
    inventory_paths.append(state_certification_path)
    add_check(
        checks,
        "all_state_artifacts_certified",
        failed_states == 0,
        f"passed={len(states) - failed_states}; failed={failed_states}",
    )
    add_check(
        checks,
        "training_only_parameter_reconstruction",
        max_parameter_delta <= 1e-10,
        f"max_abs_delta={max_parameter_delta:.12g}",
    )
    add_check(
        checks,
        "held_out_transformation_uses_frozen_parameters",
        max_transform_delta <= 2e-5,
        f"max_abs_delta={max_transform_delta:.12g}",
    )
    add_check(
        checks,
        "held_out_factor_uses_frozen_training_projection",
        max_projection_delta <= 2e-5,
        f"max_abs_delta={max_projection_delta:.12g}",
    )
    add_check(
        checks,
        "explicit_missing_masks_exact",
        max_mask_mismatch == 0,
        f"maximum_mismatches={max_mask_mismatch}",
    )
    add_check(
        checks,
        "no_outcome_or_future_SSP_access",
        all(
            lock.get(key) in (False, 0)
            for key in (
                "phenotype_values_read",
                "inner_validation_metrics_read",
                "outer_test_outcomes_read",
                "outer_test_metrics_read",
                "final_holdout_outcomes_read",
                "future_SSP_values_read",
            )
        ),
        "all protected data-access flags false",
    )
    checks_frame = pd.DataFrame(checks)
    checks_path = audit / "split_bound_projection_input_validation.tsv"
    atomic_tsv(checks_path, checks_frame)
    inventory_paths.append(checks_path)
    failed_checks = checks_frame.loc[checks_frame.status.eq("FAIL"), "check"].tolist()

    release.mkdir(parents=True, exist_ok=True)
    manifest_path = release / "SPLIT_BOUND_PROJECTION_INPUT_CLOSING_MANIFEST.tsv"
    inventory = manifest_inventory(root, inventory_paths)
    atomic_tsv(manifest_path, inventory)
    decision = {
        "status": "PASS_SPLIT_BOUND_HISTORICAL_PROJECTION_INPUTS_CERTIFIED"
        if not failed_checks
        else "FAIL_SPLIT_BOUND_HISTORICAL_PROJECTION_INPUTS",
        "release_id": "E_PROJECTION_CORE_V1_SPLIT_BOUND_HISTORICAL_V1",
        "protocol_version": protocol["protocol_version"],
        "selection_data": protocol["selection_data"],
        "state_count": len(states),
        "environment_count": len(axis),
        "feature_count": len(features),
        "state_certification_pass_count": int(state_certification.status.eq("PASS").sum()),
        "state_certification_fail_count": failed_states,
        "factor_rank_min": int(state_certification.factor_rank.min()),
        "factor_rank_max": int(state_certification.factor_rank.max()),
        "fit_environment_count_min": int(state_certification.fit_environments.min()),
        "fit_environment_count_max": int(state_certification.fit_environments.max()),
        "active_historical_environment_count": int(active.sum()),
        "inactive_historical_environment_count": int((~active).sum()),
        "checks": dict(zip(checks_frame.check, checks_frame.status.eq("PASS"), strict=True)),
        "failed_checks": failed_checks,
        "closing_manifest_sha256": sha256_file(manifest_path),
        "closing_manifest_entries": len(inventory),
        "closing_manifest_bytes": int(inventory.bytes.sum()),
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_SSP_values_read": False,
        "future_covariate_matrices_used_for_training": 0,
        "future_predictions_generated": 0,
        "phase6_model_selection_inputs_ready": not failed_checks,
    }
    decision_path = release / "SPLIT_BOUND_PROJECTION_INPUT_RELEASE_DECISION.json"
    atomic_json(decision_path, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    if failed_checks:
        raise SystemExit("Split-bound historical projection-input certification failed")


if __name__ == "__main__":
    main()
