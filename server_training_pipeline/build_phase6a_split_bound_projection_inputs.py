from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

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
DEFAULT_OUTPUT = Path(
    "environment/v2/e_projection_core_v1_split_bound_historical_v1"
)
DEFAULT_AUDIT = Path(
    "audit/v2/e_projection_core_v1_split_bound_historical_v1"
)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def fit_preprocessing(
    raw: np.ndarray,
    fit_mask: np.ndarray,
    minimum_fraction: float,
    minimum_count: int,
    constant_tolerance: float,
) -> dict[str, np.ndarray]:
    training = raw[fit_mask]
    if len(training) < minimum_count:
        raise ValueError("Insufficient climate-eligible training environments")
    nonmissing = np.isfinite(training).sum(axis=0)
    required = max(minimum_count, math.ceil(minimum_fraction * len(training)))
    medians = np.zeros(raw.shape[1], dtype=np.float64)
    means = np.zeros(raw.shape[1], dtype=np.float64)
    scales = np.ones(raw.shape[1], dtype=np.float64)
    retained = np.zeros(raw.shape[1], dtype=bool)
    for column in range(raw.shape[1]):
        finite = training[np.isfinite(training[:, column]), column]
        if nonmissing[column] < required or not finite.size:
            continue
        median = float(np.median(finite))
        imputed = np.where(np.isfinite(training[:, column]), training[:, column], median)
        mean = float(imputed.mean())
        scale = float(imputed.std(ddof=0))
        medians[column] = median
        means[column] = mean
        if np.isfinite(scale) and scale > constant_tolerance:
            scales[column] = scale
            retained[column] = True
    if not retained.any():
        raise ValueError("No nonconstant projection-core feature survives training-only preprocessing")
    return {
        "nonmissing": nonmissing.astype(np.int64),
        "required": np.asarray([required], dtype=np.int64),
        "medians": medians,
        "means": means,
        "scales": scales,
        "retained": retained,
    }


def transform_features(
    raw: np.ndarray,
    active_mask: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> np.ndarray:
    transformed = np.zeros(raw.shape, dtype=np.float64)
    retained = parameters["retained"]
    selected = raw[:, retained].copy()
    missing = ~np.isfinite(selected)
    medians = parameters["medians"][retained]
    selected[missing] = np.broadcast_to(medians, selected.shape)[missing]
    transformed[:, retained] = (
        selected - parameters["means"][retained]
    ) / parameters["scales"][retained]
    transformed[~active_mask, :] = 0.0
    return transformed


def canonicalize_svd(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.copy()
    for row in range(len(vectors)):
        pivot = int(np.argmax(np.abs(vectors[row])))
        if vectors[row, pivot] < 0:
            vectors[row] *= -1.0
    return vectors


def expected_state_files(state_dir: Path) -> list[Path]:
    return [
        state_dir / "environment_entities.tsv",
        state_dir / "feature_parameters.tsv",
        state_dir / "standardized_features_float32.npy",
        state_dir / "feature_missing_mask_packbits.npy",
        state_dir / "kernel_projection_float32.npy",
        state_dir / "kernel_factor_float32.npy",
        state_dir / "training_singular_values_float64.npy",
        state_dir / "state_metadata.json",
    ]


def completed_state(state_dir: Path) -> dict[str, Any] | None:
    metadata_path = state_dir / "state_metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for name, digest in metadata.get("artifact_sha256", {}).items():
        path = state_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            return None
    if any(not path.is_file() for path in expected_state_files(state_dir)):
        return None
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = resolve(root, args.protocol)
    freeze = resolve(root, args.freeze)
    output = resolve(root, args.output)
    audit = resolve(root, args.audit)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock_path = freeze / "split_bound_projection_input_freeze.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "PASS_SPLIT_BOUND_HISTORICAL_INPUTS_FROZEN":
        raise ValueError("Split-bound projection-input freeze is not valid")
    if sha256_file(protocol_path) != lock["protocol_sha256"]:
        raise ValueError("Projection-input protocol changed after freeze")
    for name, key in (
        ("feature_schema.tsv", "feature_schema_sha256"),
        ("state_manifest.tsv", "state_manifest_sha256"),
        ("environment_axis.tsv", "environment_axis_sha256"),
    ):
        if sha256_file(freeze / name) != lock[key]:
            raise ValueError(f"Frozen input changed: {name}")

    historical_path = root / lock["historical_feature_path"]
    if sha256_file(historical_path) != lock["historical_feature_sha256"]:
        raise ValueError("Certified historical feature matrix changed after freeze")
    schema = pd.read_csv(freeze / "feature_schema.tsv", sep="\t")
    states = pd.read_csv(freeze / "state_manifest.tsv", sep="\t", dtype=str)
    axis = pd.read_csv(freeze / "environment_axis.tsv", sep="\t", dtype={"environment_id": str})
    features = schema.feature.tolist()
    historical = pd.read_parquet(historical_path)
    if historical.environment_id.duplicated().any():
        raise ValueError("Historical projection-core environment IDs are not unique")
    aligned = historical.set_index("environment_id").reindex(axis.environment_id)
    raw = aligned[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    source_available = aligned.index.to_series().isin(historical.environment_id).to_numpy()
    climate_eligible = aligned["projection_core_climate_eligible"].eq(True).to_numpy(dtype=bool)
    active = source_available & climate_eligible
    missing_mask = ~np.isfinite(raw)
    missing_mask[~source_available, :] = True
    packed_missing = np.packbits(missing_mask, axis=1, bitorder="little")

    output.mkdir(parents=True, exist_ok=True)
    (output / "states").mkdir(parents=True, exist_ok=True)
    registry_rows: list[dict[str, Any]] = []
    preprocess = protocol["preprocessing"]
    factorization = protocol["factorization"]
    for ordinal, state in enumerate(states.itertuples(index=False), start=1):
        state_dir = output / "states" / state.state_id
        existing = completed_state(state_dir)
        if existing is not None:
            registry_rows.append(existing["registry_record"])
            print(f"[{ordinal:03d}/{len(states)}] CACHED {state.state_id}", flush=True)
            continue
        state_dir.mkdir(parents=True, exist_ok=True)
        training_ids_path = root / state.manifest_path
        if sha256_file(training_ids_path) != state.manifest_sha256:
            raise ValueError(f"Training-environment manifest changed: {state.state_id}")
        training_ids = set(pd.read_csv(training_ids_path, sep="\t", dtype=str).environment_id)
        training_mask = axis.environment_id.isin(training_ids).to_numpy()
        fit_mask = training_mask & active
        parameters = fit_preprocessing(
            raw,
            fit_mask,
            float(preprocess["minimum_training_nonmissing_fraction"]),
            int(preprocess["minimum_training_nonmissing_count"]),
            float(preprocess["constant_tolerance"]),
        )
        standardized = transform_features(raw, active, parameters)
        retained = parameters["retained"]
        retained_count = int(retained.sum())
        exact_factor = standardized[:, retained] / math.sqrt(retained_count)
        raw_diagonal = np.einsum(
            "ij,ij->i", exact_factor[fit_mask], exact_factor[fit_mask]
        )
        diagonal_scale = float(raw_diagonal.mean())
        if not np.isfinite(diagonal_scale) or diagonal_scale <= 0:
            raise ValueError(f"Invalid training kernel diagonal scale: {state.state_id}")
        exact_factor /= math.sqrt(diagonal_scale)
        maximum_rank = min(
            int(factorization["maximum_rank"]),
            exact_factor.shape[1],
            int(fit_mask.sum()) - 1,
        )
        _, singular_values_all, right_vectors_all = np.linalg.svd(
            exact_factor[fit_mask], full_matrices=False
        )
        singular_values = singular_values_all[:maximum_rank]
        right_vectors = canonicalize_svd(right_vectors_all[:maximum_rank])
        tolerance = max(exact_factor[fit_mask].shape) * np.finfo(float).eps * singular_values[0]
        rank = int((singular_values > tolerance).sum())
        if rank < 1:
            raise ValueError(f"Projection-core factorization has zero rank: {state.state_id}")
        singular_values = singular_values[:rank]
        right_vectors = right_vectors[:rank]
        projection = np.zeros((len(features), rank), dtype=np.float64)
        projection[retained, :] = (
            right_vectors.T / math.sqrt(retained_count * diagonal_scale)
        )
        factor = standardized @ projection
        factor[~active, :] = 0.0
        if not np.isfinite(standardized).all() or not np.isfinite(factor).all():
            raise ValueError(f"Nonfinite split-bound projection input: {state.state_id}")

        entities = pd.DataFrame(
            {
                "environment_index": axis.environment_index.astype(int),
                "environment_id": axis.environment_id,
                "partition": np.where(training_mask, "TRAINING", "APPLICATION"),
                "historical_source_available": source_available,
                "projection_core_climate_eligible": climate_eligible,
                "component_active": active,
                "observed_feature_count": (~missing_mask).sum(axis=1),
                "missing_feature_count": missing_mask.sum(axis=1),
            }
        )
        parameter_rows = []
        required = int(parameters["required"][0])
        for index, feature in enumerate(features):
            if parameters["retained"][index]:
                status = "RETAINED"
                reason = ""
            elif parameters["nonmissing"][index] < required:
                status = "DROPPED"
                reason = "INSUFFICIENT_TRAINING_NONMISSING"
            else:
                status = "DROPPED"
                reason = "TRAINING_CONSTANT_OR_NONFINITE_SCALE"
            parameter_rows.append(
                {
                    "feature_index": index,
                    "feature": feature,
                    "feature_block": schema.iloc[index].feature_block,
                    "training_fit_environments": int(fit_mask.sum()),
                    "training_nonmissing": int(parameters["nonmissing"][index]),
                    "minimum_training_nonmissing": required,
                    "imputation_median": parameters["medians"][index],
                    "centering_mean_after_imputation": parameters["means"][index],
                    "scaling_sd_after_imputation": parameters["scales"][index],
                    "feature_status": status,
                    "drop_reason": reason,
                    "kernel_training_mean_diagonal_raw": diagonal_scale,
                    "kernel_factor_postscale": 1.0 / math.sqrt(diagonal_scale),
                    "fit_scope": "CLIMATE_ELIGIBLE_TRAINING_ENVIRONMENTS_ONLY",
                }
            )
        parameters_frame = pd.DataFrame(parameter_rows)
        atomic_tsv(state_dir / "environment_entities.tsv", entities)
        atomic_tsv(state_dir / "feature_parameters.tsv", parameters_frame)
        atomic_npy(
            state_dir / "standardized_features_float32.npy", standardized.astype(np.float32)
        )
        atomic_npy(
            state_dir / "feature_missing_mask_packbits.npy", packed_missing.astype(np.uint8)
        )
        atomic_npy(state_dir / "kernel_projection_float32.npy", projection.astype(np.float32))
        atomic_npy(state_dir / "kernel_factor_float32.npy", factor.astype(np.float32))
        atomic_npy(
            state_dir / "training_singular_values_float64.npy", singular_values.astype(np.float64)
        )
        artifacts = {
            path.name: sha256_file(path)
            for path in expected_state_files(state_dir)
            if path.name != "state_metadata.json"
        }
        training_factor_diagonal = np.einsum(
            "ij,ij->i", factor[fit_mask], factor[fit_mask]
        )
        record = {
            "state_id": state.state_id,
            "scenario": state.scenario,
            "state_level": state.state_level,
            "training_environments": int(training_mask.sum()),
            "fit_environments": int(fit_mask.sum()),
            "application_environments": int((~training_mask).sum()),
            "active_application_environments": int((~training_mask & active).sum()),
            "inactive_environments": int((~active).sum()),
            "feature_count": len(features),
            "retained_feature_count": retained_count,
            "factor_rank": rank,
            "exact_kernel_training_mean_diagonal": 1.0,
            "truncated_factor_training_mean_diagonal": float(training_factor_diagonal.mean()),
            "minimum_training_singular_value": float(singular_values[-1]),
            "maximum_training_singular_value": float(singular_values[0]),
            "status": "PASS",
        }
        metadata = {
            "status": "PASS",
            "protocol_version": protocol["protocol_version"],
            "state_id": state.state_id,
            "training_manifest_sha256": state.manifest_sha256,
            "feature_schema_sha256": lock["feature_schema_sha256"],
            "environment_axis_sha256": lock["environment_axis_sha256"],
            "historical_feature_sha256": lock["historical_feature_sha256"],
            "factorization_method": factorization["method"],
            "artifact_sha256": artifacts,
            "registry_record": record,
            "phenotype_values_read": False,
            "inner_validation_metrics_read": False,
            "outer_test_outcomes_read": False,
            "outer_test_metrics_read": False,
            "final_holdout_outcomes_read": False,
            "future_SSP_values_read": False,
        }
        atomic_json(state_dir / "state_metadata.json", metadata)
        registry_rows.append(record)
        print(
            f"[{ordinal:03d}/{len(states)}] BUILT {state.state_id} "
            f"fit={fit_mask.sum()} retained={retained_count} rank={rank}",
            flush=True,
        )

    registry = pd.DataFrame(registry_rows).sort_values("state_id").reset_index(drop=True)
    registry_path = output / "split_bound_projection_input_registry.tsv"
    atomic_tsv(registry_path, registry)
    result = {
        "status": "PASS",
        "protocol_version": protocol["protocol_version"],
        "selection_data": protocol["selection_data"],
        "state_count": len(registry),
        "environment_count": len(axis),
        "feature_count": len(features),
        "factor_rank_min": int(registry.factor_rank.min()),
        "factor_rank_max": int(registry.factor_rank.max()),
        "fit_environment_count_min": int(registry.fit_environments.min()),
        "fit_environment_count_max": int(registry.fit_environments.max()),
        "inactive_environment_count": int((~active).sum()),
        "registry_sha256": sha256_file(registry_path),
        "freeze_sha256": sha256_file(lock_path),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_SSP_values_read": False,
        "future_predictions_generated": 0,
    }
    audit.mkdir(parents=True, exist_ok=True)
    atomic_json(audit / "split_bound_projection_input_build_provenance.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
