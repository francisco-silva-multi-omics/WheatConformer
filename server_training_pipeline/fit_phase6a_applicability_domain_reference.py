from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from server_training_pipeline.fetch_cmip6_member_resolved import atomic_json, atomic_tsv, resolve, sha256_file


DEFAULT_PROTOCOL = Path("server_training_pipeline/phase6a_applicability_domain_contract_v1.json")
DEFAULT_FEATURES = Path(
    "environment/v2/e_projection_core_v1_historical_backcast/era5_land_historical_projection_core_features.parquet"
)
DEFAULT_SPLITS = Path(
    "audit/v2/phase5_panel_environment_scenario_parity_extension_v2/splits/state_entities"
)
DEFAULT_OUTPUT = Path("environment/v2/e_projection_core_v1_applicability_domain_reference")
DEFAULT_AUDIT = Path("audit/v2/e_projection_core_v1_release")


def feature_block(name: str) -> str:
    lower = name.lower()
    if name in {"latitude", "longitude"}:
        return "geography"
    if "available_water_capacity" in lower or any(
        token in lower for token in ("precip", "wet_day", "dry_day", "pet_", "water_balance")
    ):
        return "water"
    if "radiation" in lower:
        return "radiation"
    if any(token in lower for token in ("gdd", "frost", "heat", "vpd", "tasmin", "tasmean", "tasmax")):
        return "heat"
    if any(token in lower for token in ("complete", "eligible", "missing", "observed", "mask")):
        return "confidence"
    return "development"


def select_features(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "environment_id",
        "daily_request_id",
        "sowing_date",
        "soil_status",
        "soil_source_class",
        "water_balance_disabled_reason",
    }
    selected = []
    for column in frame.columns:
        if column in excluded or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        if frame[column].nunique(dropna=True) <= 1:
            continue
        selected.append(column)
    return selected


def robust_scale_parameters(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        q25, q75 = np.quantile(finite, [0.25, 0.75])
        iqr = float(q75 - q25)
        scale = 1.4826 * mad
        scale_source = "MAD"
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = iqr / 1.349
            scale_source = "IQR"
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(finite))
            scale_source = "SD"
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
            scale_source = "UNIT_FALLBACK"
        rows.append(
            {
                "feature": column,
                "feature_block": feature_block(column),
                "observed_count": len(finite),
                "missing_count": len(values) - len(finite),
                "minimum": float(finite.min()),
                "maximum": float(finite.max()),
                "median": median,
                "mad": mad,
                "q25": float(q25),
                "q75": float(q75),
                "iqr": iqr,
                "robust_scale": scale,
                "scale_source": scale_source,
            }
        )
    return pd.DataFrame(rows)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = resolve(root, args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "phase6a_applicability_domain_v1":
        raise ValueError("Applicability-domain protocol identity mismatch")
    feature_path = resolve(root, args.features)
    frame = pd.read_parquet(feature_path)
    frame = frame[frame.projection_core_climate_eligible.astype(bool)].copy()
    if frame.environment_id.duplicated().any() or len(frame) < 10000:
        raise ValueError("Historical applicability-domain environment population is invalid")
    columns = select_features(frame)
    statistics = robust_scale_parameters(frame, columns)
    ordered = statistics.set_index("feature").loc[columns]
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    medians = ordered["median"].to_numpy(dtype=float)
    scales = ordered["robust_scale"].to_numpy(dtype=float)
    missing = ~np.isfinite(matrix)
    matrix[missing] = np.broadcast_to(medians, matrix.shape)[missing]
    standardized = (matrix - medians) / scales
    maximum_components = min(32, standardized.shape[1], standardized.shape[0] - 1)
    pca_full = PCA(n_components=maximum_components, svd_solver="randomized", random_state=20260819)
    scores_full = pca_full.fit_transform(standardized)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    retained = min(maximum_components, int(np.searchsorted(cumulative, 0.95) + 1))
    scores = scores_full[:, :retained]
    covariance = LedoitWolf().fit(scores)
    centered = scores - covariance.location_
    mahalanobis = np.sqrt(np.einsum("ij,jk,ik->i", centered, covariance.precision_, centered))
    neighbors = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(scores)
    nearest = neighbors.kneighbors(scores, return_distance=True)[0][:, 1]
    thresholds = {
        "training_environment_count": len(frame),
        "feature_count": len(columns),
        "pca_component_count": retained,
        "pca_explained_variance_fraction": float(cumulative[retained - 1]),
        "mahalanobis_99": float(np.quantile(mahalanobis, 0.99)),
        "mahalanobis_999": float(np.quantile(mahalanobis, 0.999)),
        "nearest_neighbor_99": float(np.quantile(nearest, 0.99)),
        "nearest_neighbor_999": float(np.quantile(nearest, 0.999)),
    }
    output = resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_tsv(output / "historical_robust_feature_reference.tsv", statistics)
    atomic_npy(output / "pca_components.npy", pca_full.components_[:retained].astype(np.float32))
    atomic_npy(output / "pca_center.npy", pca_full.mean_.astype(np.float32))
    atomic_npy(output / "pca_explained_variance_ratio.npy", pca_full.explained_variance_ratio_[:retained].astype(np.float32))
    atomic_npy(output / "mahalanobis_location.npy", covariance.location_.astype(np.float32))
    atomic_npy(output / "mahalanobis_precision.npy", covariance.precision_.astype(np.float32))
    atomic_json(output / "historical_distance_thresholds.json", thresholds)
    splits = resolve(root, args.splits)
    split_rows = []
    for path in sorted(splits.glob("*__training_environments.tsv")):
        ids = pd.read_csv(path, sep="\t", dtype=str)
        available = ids.environment_id.isin(frame.environment_id)
        split_rows.append(
            {
                "state": path.stem.replace("__training_environments", ""),
                "manifest_path": path.relative_to(root).as_posix(),
                "manifest_sha256": sha256_file(path),
                "training_environment_count": len(ids),
                "projection_core_available_count": int(available.sum()),
                "projection_core_missing_count": int((~available).sum()),
            }
        )
    split_manifest = pd.DataFrame(split_rows)
    if len(split_manifest) != 150:
        raise ValueError("Applicability-domain split manifest must contain 150 frozen states")
    atomic_tsv(output / "split_training_environment_manifest.tsv", split_manifest)
    inventory_rows = []
    for path in sorted(output.iterdir()):
        if path.is_file():
            inventory_rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    inventory = pd.DataFrame(inventory_rows)
    atomic_tsv(output / "applicability_domain_artifact_inventory.tsv", inventory)
    audit = resolve(root, args.audit)
    audit.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "PASS",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(protocol_path),
        "historical_feature_sha256": sha256_file(feature_path),
        "historical_environment_count": len(frame),
        "feature_count": len(columns),
        "split_state_count": len(split_manifest),
        "distance_thresholds": thresholds,
        "artifact_inventory_sha256": sha256_file(
            output / "applicability_domain_artifact_inventory.tsv"
        ),
        "future_covariates_evaluated": False,
        "phenotype_values_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
    }
    atomic_json(audit / "applicability_domain_reference_provenance.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
