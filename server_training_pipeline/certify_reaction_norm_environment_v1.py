from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .final_evaluation_contract import file_sha256


REQUIRED_ARTIFACTS = {
    "matrix": "E_REACTION_NORM_V1.parquet",
    "raw_matrix": "E_REACTION_NORM_V1_raw.parquet",
    "order": "E_REACTION_NORM_V1_order.tsv",
    "feature_manifest": "E_REACTION_NORM_V1_feature_manifest.tsv",
    "scaling": "E_REACTION_NORM_V1_scaling.tsv",
    "kernel": "K_E_REACTION_NORM_V1.npy",
    "kernel_order": "K_E_REACTION_NORM_V1_order.tsv",
    "kernel_manifest": "reaction_norm_environment_kernel_manifest.tsv",
    "provenance": "E_REACTION_NORM_V1_provenance.json",
}


def identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "sha256": file_sha256(path),
    }


def current(source: dict[str, object]) -> bool:
    path = Path(str(source.get("path", "")))
    return path.is_file() and file_sha256(path) == source.get("sha256")


def add_check(
    rows: list[dict[str, object]], check: str, passed: bool, detail: str
) -> None:
    rows.append(
        {"check": check, "status": "PASS" if passed else "FAIL", "detail": detail}
    )


def recompute_builder_kernel(
    values: np.ndarray, fit_positions: np.ndarray
) -> np.ndarray:
    """Reproduce the builder's float32 kernel arithmetic exactly."""
    matrix = np.asarray(values, dtype=np.float32)
    raw = ((matrix @ matrix.T) / max(matrix.shape[1], 1)).astype(np.float32)
    raw = ((raw + raw.T) * np.float32(0.5)).astype(np.float32)
    diagonal = np.diag(raw)[np.asarray(fit_positions, dtype=int)]
    mean_diagonal = float(np.mean(diagonal)) if diagonal.size else 0.0
    if not np.isfinite(mean_diagonal) or mean_diagonal <= 0:
        return raw.copy()
    return (raw / mean_diagonal).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Certify a fold-local E_REACTION_NORM_V1 artifact bundle."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    artifact_dir = args.artifact_dir.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    paths = {name: artifact_dir / filename for name, filename in REQUIRED_ARTIFACTS.items()}
    for name, path in paths.items():
        add_check(rows, f"{name}_present", path.is_file() and path.stat().st_size > 0, str(path))
    if any(row["status"] == "FAIL" for row in rows):
        pd.DataFrame(rows).to_csv(
            artifact_dir / "E_REACTION_NORM_V1_validation.tsv", sep="\t", index=False
        )
        raise SystemExit("E_REACTION_NORM_V1 is missing required artifacts")

    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    matrix = pd.read_parquet(paths["matrix"])
    raw = pd.read_parquet(paths["raw_matrix"])
    order = pd.read_csv(paths["order"], sep="\t", dtype={"env_id": str})
    kernel_order = pd.read_csv(paths["kernel_order"], sep="\t", dtype={"env_id": str})
    manifest = pd.read_csv(paths["feature_manifest"], sep="\t", dtype=str)
    kernel_manifest = pd.read_csv(paths["kernel_manifest"], sep="\t", dtype=str)
    scaling = pd.read_csv(paths["scaling"], sep="\t")
    kernel = np.load(paths["kernel"], mmap_mode="r")

    feature_columns = [column for column in matrix.columns if column != "env_id"]
    raw_columns = [column for column in raw.columns if column != "env_id"]
    add_check(
        rows,
        "protocol_frozen",
        protocol.get("status") == "frozen_before_inner_validation",
        str(protocol.get("status")),
    )
    add_check(
        rows,
        "protocol_identity",
        provenance.get("sources", {}).get("protocol", {}).get("sha256")
        == file_sha256(protocol_path),
        f"protocol_version={provenance.get('protocol_version')}",
    )
    add_check(
        rows,
        "phenotype_blind",
        provenance.get("phenotype_values_read") is False
        and provenance.get("actual_heading_or_maturity_dates_used") is False,
        "phenotype_values_read=false; actual target phenology dates forbidden",
    )
    add_check(
        rows,
        "protected_outcomes_unread",
        provenance.get("outer_test_metrics_read") is False
        and provenance.get("final_holdout_outcomes_read") is False,
        "outer_test_metrics_read=false; final_holdout_outcomes_read=false",
    )
    source_status = all(current(value) for value in provenance.get("sources", {}).values())
    add_check(
        rows,
        "source_identities_current",
        source_status,
        f"source_count={len(provenance.get('sources', {}))}",
    )

    order_valid = (
        "env_id" in order
        and "compact_kernel_index" in order
        and not order["env_id"].isna().any()
        and not order["env_id"].duplicated().any()
        and np.array_equal(
            pd.to_numeric(order["compact_kernel_index"], errors="coerce").to_numpy(),
            np.arange(len(order)),
        )
    )
    add_check(rows, "environment_order", order_valid, f"rows={len(order)}")
    add_check(
        rows,
        "matrix_order_alignment",
        matrix["env_id"].astype(str).tolist() == order["env_id"].astype(str).tolist()
        and raw["env_id"].astype(str).tolist() == order["env_id"].astype(str).tolist()
        and kernel_order["env_id"].astype(str).tolist() == order["env_id"].astype(str).tolist(),
        f"matrix_rows={len(matrix)}; raw_rows={len(raw)}; order_rows={len(order)}",
    )
    values = matrix[feature_columns].to_numpy(dtype=np.float64)
    add_check(
        rows,
        "matrix_finite",
        values.size > 0 and np.isfinite(values).all(),
        f"shape={values.shape}",
    )
    manifest_required = {
        "feature",
        "source_feature",
        "source_artifact",
        "feature_block",
        "eligible_traits",
        "regulatory_treatment",
        "is_missingness_indicator",
        "phenotype_derived",
        "fit_partition",
    }
    add_check(
        rows,
        "feature_manifest_columns",
        manifest_required.issubset(manifest.columns),
        f"missing={sorted(manifest_required.difference(manifest.columns))}",
    )
    add_check(
        rows,
        "feature_manifest_alignment",
        manifest["feature"].tolist() == feature_columns,
        f"manifest_features={len(manifest)}; matrix_features={len(feature_columns)}",
    )
    add_check(
        rows,
        "kernel_manifest_contract",
        len(kernel_manifest) == 1
        and kernel_manifest.iloc[0].get("kernel") == "K_E_REACTION_NORM_V1"
        and kernel_manifest.iloc[0].get("eligible_traits") == "*"
        and str(kernel_manifest.iloc[0].get("enabled_default", "")).lower() == "false",
        f"rows={len(kernel_manifest)}",
    )
    add_check(
        rows,
        "feature_provenance",
        manifest["phenotype_derived"].str.lower().eq("false").all()
        and manifest["fit_partition"].eq("outer_training_environments_only").all()
        and manifest["source_artifact"].fillna("").ne("").all(),
        f"blocks={sorted(manifest['feature_block'].unique().tolist())}",
    )
    expected_blocks = {"geo", "development", "heat", "water", "radiation", "management", "confidence"}
    observed_blocks = set(manifest["feature_block"])
    raw_blocks = {str(column).split("__", 1)[0] for column in raw_columns}
    add_check(
        rows,
        "required_feature_blocks",
        expected_blocks.issubset(raw_blocks),
        f"missing_raw={sorted(expected_blocks-raw_blocks)}; "
        f"dropped_from_standardized={sorted(expected_blocks-observed_blocks)}",
    )
    valid_traits = set(protocol.get("trait_axis_policy", {}))
    bad_eligibility = []
    for row in manifest.itertuples(index=False):
        eligible = {value for value in str(row.eligible_traits).split(",") if value}
        if not eligible or not eligible.issubset(valid_traits):
            bad_eligibility.append(str(row.feature))
    add_check(
        rows,
        "trait_eligibility",
        not bad_eligibility,
        f"invalid_features={len(bad_eligibility)}",
    )

    fit_source = Path(str(provenance["sources"]["fit_environment_ids"]["path"]))
    fit_frame = pd.read_csv(fit_source, sep="\t", dtype=str)
    fit_col = "env_id" if "env_id" in fit_frame else fit_frame.columns[0]
    fit_ids = set(fit_frame[fit_col].fillna("").astype(str).str.strip())
    fit_mask = matrix["env_id"].astype(str).isin(fit_ids).to_numpy()
    fit_values = values[fit_mask]
    fit_means = np.mean(fit_values, axis=0)
    fit_stds = np.std(fit_values, axis=0)
    add_check(
        rows,
        "outer_training_scaling",
        fit_values.shape[0] == provenance.get("fit_environment_count")
        and np.max(np.abs(fit_means)) < 2e-5
        and np.max(np.abs(fit_stds - 1.0)) < 2e-5,
        f"fit_rows={fit_values.shape[0]}; max_abs_mean={np.max(np.abs(fit_means)):.8g}; "
        f"max_abs_sd_delta={np.max(np.abs(fit_stds-1.0)):.8g}",
    )
    add_check(
        rows,
        "scaling_scope",
        scaling["fit_environment_count"].eq(fit_values.shape[0]).all()
        and scaling["imputation"].eq("outer_training_median").all(),
        f"scaling_rows={len(scaling)}",
    )

    kernel_shape = kernel.ndim == 2 and kernel.shape == (len(order), len(order))
    add_check(rows, "kernel_square", kernel_shape, f"shape={kernel.shape}")
    if kernel_shape:
        kernel_values = np.asarray(kernel, dtype=np.float64)
        symmetry = float(np.max(np.abs(kernel_values - kernel_values.T)))
        fit_positions = np.flatnonzero(fit_mask)
        diagonal = np.diag(kernel_values)
        mean_diag = float(diagonal[fit_positions].mean())
        recomputed = recompute_builder_kernel(values, fit_positions)
        max_delta = float(
            np.max(np.abs(recomputed.astype(np.float64) - kernel_values))
        )
        sampled = np.linspace(0, len(order) - 1, min(len(order), 512), dtype=int)
        min_eigenvalue = float(
            np.linalg.eigvalsh(kernel_values[np.ix_(sampled, sampled)]).min()
        )
        add_check(
            rows,
            "kernel_finite_symmetric",
            np.isfinite(kernel_values).all() and symmetry <= 1e-5,
            f"symmetry_max_abs={symmetry:.8g}",
        )
        add_check(
            rows,
            "kernel_training_mean_diagonal",
            np.isfinite(diagonal).all()
            and np.all(diagonal > 0)
            and abs(mean_diag - 1.0) <= 1e-5,
            f"fit_mean_diagonal={mean_diag:.8g}; min_diagonal={diagonal.min():.8g}; "
            f"full_mean_diagonal={diagonal.mean():.8g}",
        )
        add_check(
            rows,
            "kernel_reproducible_from_matrix",
            max_delta <= 2e-5,
            f"max_abs_delta={max_delta:.8g}",
        )
        add_check(
            rows,
            "kernel_sampled_psd",
            min_eigenvalue >= -1e-4,
            f"sampled_n={len(sampled)}; min_eigenvalue={min_eigenvalue:.8g}",
        )

    validation = pd.DataFrame(rows)
    validation_path = artifact_dir / "E_REACTION_NORM_V1_validation.tsv"
    validation.to_csv(validation_path, sep="\t", index=False)
    failed = validation[validation["status"].eq("FAIL")]
    artifacts = {name: identity(path) for name, path in paths.items()}
    artifacts["validation"] = identity(validation_path)
    certification = {
        "status": "PASS" if failed.empty else "FAIL",
        "protocol_version": protocol.get("protocol_version"),
        "selection_data": "environment_identifiers_and_outer_training_preprocessing_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "check_count": len(validation),
        "failed_check_count": len(failed),
        "environment_count": len(order),
        "fit_environment_count": int(fit_mask.sum()),
        "feature_count": len(feature_columns),
        "builder_sha256": provenance.get("builder_sha256"),
        "certifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifact_identities": artifacts,
    }
    certification_path = artifact_dir / "E_REACTION_NORM_V1_certification.json"
    certification_path.write_text(json.dumps(certification, indent=2), encoding="utf-8")
    print(json.dumps(certification, indent=2), flush=True)
    if not failed.empty:
        print(failed.to_string(index=False), flush=True)
        raise SystemExit("E_REACTION_NORM_V1 certification failed")


if __name__ == "__main__":
    main()
