"""Create post-canonical development folds and fold-local Stage-1 weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb


VIEW_VERSION = "stage1_v2_model_view_2026_07_30_v1"
FOLD_NAMESPACE = "stage1_v2_reconstruction_fold_v1"
SELECTED_TRAITS = {
    "1000_GRAIN_WEIGHT", "ABOVE_GROUND_BIOMASS", "DAYS_TO_HEADING", "DAYS_TO_MATURITY",
    "GRAIN_YIELD", "PLANT_HEIGHT", "TEST_WEIGHT",
}
SCENARIOS = {
    "heldout_genotype": ("genotype_fold", "resolved_gid"),
    "heldout_environment": ("environment_fold", "canonical_environment_id"),
    "heldout_genotype_environment_pair": ("pair_fold", "genotype_environment_pair_key"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fold_for(kind: str, value: str, folds: int = 5) -> int:
    token = f"{FOLD_NAMESPACE}|{kind}|{value}"
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16) % folds


class Writer:
    def __init__(self, path: Path):
        self.path = path
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None

    def write(self, frame: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        elif table.schema != self.schema:
            table = table.cast(self.schema, safe=False)
        self.writer.write_table(table, row_group_size=100_000)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def load_axis(path: Path, column: str) -> set[str]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if column not in frame:
        raise KeyError(f"{path} lacks required column {column}")
    return set(frame[column].astype(str))


def fit_weight_parameters(
    training_variance: pd.Series,
    variance_floor_quantile: float = 0.01,
    missing_variance_quantile: float = 0.75,
    precision_clip_quantile: float = 0.99,
) -> dict[str, float | int]:
    """Fit variance stabilization using training rows only."""
    train_var = pd.to_numeric(training_variance, errors="coerce")
    finite_positive = train_var[np.isfinite(train_var) & train_var.gt(0)]
    if finite_positive.empty:
        variance_floor = missing_variance = 1.0
    else:
        variance_floor = float(finite_positive.quantile(variance_floor_quantile))
        missing_variance = float(finite_positive.quantile(missing_variance_quantile))
        if not np.isfinite(variance_floor) or variance_floor <= 0:
            variance_floor = float(finite_positive.min())
        if not np.isfinite(missing_variance) or missing_variance <= 0:
            missing_variance = float(finite_positive.median())
    train_stabilized = train_var.where(
        np.isfinite(train_var) & train_var.gt(0), missing_variance
    ).clip(lower=variance_floor)
    train_precision = 1.0 / train_stabilized
    precision_clip = float(train_precision.quantile(precision_clip_quantile)) if len(train_precision) else 1.0
    if not np.isfinite(precision_clip) or precision_clip <= 0:
        precision_clip = float(train_precision.max()) if len(train_precision) else 1.0
    train_clipped = train_precision.clip(upper=precision_clip)
    normalization_mean = float(train_clipped.mean()) if len(train_clipped) else 1.0
    if not np.isfinite(normalization_mean) or normalization_mean <= 0:
        normalization_mean = 1.0
    return {
        "finite_positive_training_variances": len(finite_positive),
        "variance_floor": variance_floor,
        "missing_variance_replacement": missing_variance,
        "precision_clip": precision_clip,
        "training_precision_mean_after_clip": normalization_mean,
    }


def apply_weight_parameters(
    variance: pd.Series, parameters: dict[str, float | int]
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Apply frozen training-fold parameters without fitting on target rows."""
    raw_var = pd.to_numeric(variance, errors="coerce")
    missing_flag = ~(np.isfinite(raw_var) & raw_var.gt(0))
    variance_floor = float(parameters["variance_floor"])
    missing_variance = float(parameters["missing_variance_replacement"])
    precision_clip = float(parameters["precision_clip"])
    normalization_mean = float(parameters["training_precision_mean_after_clip"])
    stabilized = raw_var.where(~missing_flag, missing_variance).clip(lower=variance_floor)
    raw_precision = 1.0 / stabilized
    precision = raw_precision.clip(upper=precision_clip)
    weight = precision / normalization_mean
    return missing_flag, stabilized, weight, (~missing_flag & raw_var.lt(variance_floor)), raw_precision.gt(precision_clip)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--legacy-genotype-order", type=Path, required=True)
    parser.add_argument("--legacy-environment-order", type=Path, required=True)
    parser.add_argument("--pedigree-order", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--variance-floor-quantile", type=float, default=0.01)
    parser.add_argument("--missing-variance-quantile", type=float, default=0.75)
    parser.add_argument("--precision-clip-quantile", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)

    genotype_axis = load_axis(args.legacy_genotype_order.resolve(), "sample_id")
    environment_axis = load_axis(args.legacy_environment_order.resolve(), "env_id")
    pedigree_axis = load_axis(args.pedigree_order.resolve(), "sample_id")
    model_path = result_dir / "selected_trait_model_view_v2.parquet"
    eligibility_cols = [
        "stage1_v2_row_id", "canonical_germplasm_key", "canonical_environment_id", "accepted_canonical_trait",
        "legacy_genotype_kernel_available", "legacy_environment_kernel_available", "pedigree_kernel_available",
        "legacy_joint_kernel_eligible", "model_eligibility_reason", "genotype_fold", "environment_fold", "pair_fold",
        "protected_outer_or_final_membership_used", "model_view_version",
    ]
    eligibility_path = result_dir / "stage1_to_model_eligibility_ledger_v2.parquet"
    model_writer = Writer(model_path)
    eligibility_writer = Writer(eligibility_path)
    selected_rows = 0
    stage1_reader = pq.ParquetFile(args.stage1.resolve())
    try:
        for batch in stage1_reader.iter_batches(batch_size=args.batch_size):
            model = batch.to_pandas()
            model = model[model["accepted_canonical_trait"].isin(SELECTED_TRAITS)].copy()
            if model.empty:
                continue
            model["genotype_environment_pair_key"] = (
                model["resolved_gid"].astype(str) + "|" + model["canonical_environment_id"].astype(str)
            )
            model["genotype_fold"] = model["resolved_gid"].astype(str).map(
                lambda value: fold_for("genotype", value, args.folds)
            ).astype("int8")
            model["environment_fold"] = model["canonical_environment_id"].astype(str).map(
                lambda value: fold_for("environment", value, args.folds)
            ).astype("int8")
            model["pair_fold"] = model["genotype_environment_pair_key"].map(
                lambda value: fold_for("pair", value, args.folds)
            ).astype("int8")
            model["legacy_genotype_kernel_available"] = model["canonical_germplasm_key"].isin(genotype_axis)
            model["legacy_environment_kernel_available"] = model["canonical_environment_id"].isin(environment_axis)
            model["pedigree_kernel_available"] = model["canonical_germplasm_key"].isin(pedigree_axis)
            model["legacy_joint_kernel_eligible"] = (
                model["legacy_genotype_kernel_available"] & model["legacy_environment_kernel_available"]
            )
            model["model_eligibility_reason"] = np.select(
                [
                    model["legacy_joint_kernel_eligible"],
                    ~model["legacy_genotype_kernel_available"] & ~model["legacy_environment_kernel_available"],
                    ~model["legacy_genotype_kernel_available"],
                    ~model["legacy_environment_kernel_available"],
                ],
                [
                    "ELIGIBLE_LEGACY_JOINT_KERNEL_AXES", "RETAINED_MISSING_BOTH_LEGACY_KERNEL_AXES",
                    "RETAINED_MISSING_LEGACY_GENOTYPE_AXIS", "RETAINED_MISSING_LEGACY_ENVIRONMENT_AXIS",
                ],
                default="RETAINED_MODEL_SPECIFIC_REVIEW",
            )
            model["fold_assignment_timing"] = "AFTER_CANONICALIZATION_AND_STAGE1"
            model["protected_outer_or_final_membership_used"] = False
            model["model_view_version"] = VIEW_VERSION
            model_writer.write(model)
            eligibility_writer.write(model[eligibility_cols].copy())
            selected_rows += len(model)
    finally:
        model_writer.close()
        eligibility_writer.close()
    if selected_rows == 0:
        raise RuntimeError("No selected-trait Stage-1 rows were found")

    con = duckdb.connect(str(result_dir / "model_views.duckdb"))
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA memory_limit='2GB'")
    model_sql = str(model_path).replace("'", "''")
    model_stats = con.execute(f"""
        SELECT count(*), count(DISTINCT stage1_v2_row_id), count(DISTINCT resolved_gid),
               count(DISTINCT canonical_environment_id), count(DISTINCT accepted_canonical_trait),
               sum(legacy_joint_kernel_eligible::BIGINT)
        FROM read_parquet('{model_sql}')
    """).fetchone()
    if model_stats[0] != selected_rows or model_stats[1] != selected_rows:
        raise RuntimeError("Selected model-view row conservation or ID uniqueness failed")
    selected_trait_values = {
        row[0] for row in con.execute(
            f"SELECT DISTINCT accepted_canonical_trait FROM read_parquet('{model_sql}')"
        ).fetchall()
    }
    if selected_trait_values != SELECTED_TRAITS:
        raise RuntimeError(f"Selected trait set mismatch: {sorted(selected_trait_values)}")

    weight_path = result_dir / "fold_local_weights_v2.parquet"
    writer = Writer(weight_path)
    parameter_rows = []
    parameter_lookup: dict[tuple[str, int, str], dict[str, float | int]] = {}
    weight_rows = 0
    for scenario, (fold_column, _) in SCENARIOS.items():
        parameter_frame = pq.read_table(
            model_path, columns=["accepted_canonical_trait", "var_g_e", fold_column]
        ).to_pandas()
        for fold in range(args.folds):
            for trait, trait_frame in parameter_frame.groupby("accepted_canonical_trait", sort=True):
                is_validation = trait_frame[fold_column].eq(fold)
                train_var = pd.to_numeric(trait_frame.loc[~is_validation, "var_g_e"], errors="coerce")
                fitted = fit_weight_parameters(
                    train_var,
                    args.variance_floor_quantile,
                    args.missing_variance_quantile,
                    args.precision_clip_quantile,
                )
                parameter_rows.append({
                    "scenario": scenario, "fold": fold, "accepted_canonical_trait": trait,
                    "training_rows": int((~is_validation).sum()), "validation_rows": int(is_validation.sum()),
                    **fitted,
                    "variance_floor_quantile": args.variance_floor_quantile,
                    "missing_variance_quantile": args.missing_variance_quantile,
                    "precision_clip_quantile": args.precision_clip_quantile,
                    "fit_scope": "INNER_TRAINING_ROWS_ONLY", "model_view_version": VIEW_VERSION,
                })
                parameter_lookup[(scenario, fold, str(trait))] = fitted
        del parameter_frame

    weight_columns = [
        "stage1_v2_row_id", "accepted_canonical_trait", "resolved_gid",
        "canonical_environment_id", "var_g_e",
    ]
    try:
        for scenario, (fold_column, _) in SCENARIOS.items():
            for fold in range(args.folds):
                reader = pq.ParquetFile(model_path)
                for batch in reader.iter_batches(
                    columns=weight_columns + [fold_column], batch_size=args.batch_size
                ):
                    subset = batch.to_pandas()
                    traits = subset["accepted_canonical_trait"].astype(str)
                    floor = traits.map(
                        lambda trait: float(parameter_lookup[(scenario, fold, trait)]["variance_floor"])
                    ).to_numpy(dtype="float64")
                    missing_variance = traits.map(
                        lambda trait: float(parameter_lookup[(scenario, fold, trait)]["missing_variance_replacement"])
                    ).to_numpy(dtype="float64")
                    precision_clip = traits.map(
                        lambda trait: float(parameter_lookup[(scenario, fold, trait)]["precision_clip"])
                    ).to_numpy(dtype="float64")
                    normalization = traits.map(
                        lambda trait: float(parameter_lookup[(scenario, fold, trait)]["training_precision_mean_after_clip"])
                    ).to_numpy(dtype="float64")
                    raw_var = pd.to_numeric(subset["var_g_e"], errors="coerce").to_numpy(dtype="float64")
                    missing_flag = ~(np.isfinite(raw_var) & (raw_var > 0))
                    stabilized = np.where(missing_flag, missing_variance, raw_var)
                    stabilized = np.maximum(stabilized, floor)
                    raw_precision = 1.0 / stabilized
                    fold_weight = np.minimum(raw_precision, precision_clip) / normalization
                    floored = (~missing_flag) & (raw_var < floor)
                    clipped = raw_precision > precision_clip
                    membership = np.where(
                        subset[fold_column].to_numpy() == fold, "VALIDATION", "TRAINING"
                    )
                    subset = subset[weight_columns].copy()
                    subset["scenario"] = scenario
                    subset["fold"] = np.int8(fold)
                    subset["membership"] = membership
                    subset["source_variance"] = raw_var
                    subset["fold_local_stabilized_variance"] = stabilized
                    subset["fold_local_weight"] = fold_weight
                    subset["variance_imputed_from_training"] = missing_flag
                    subset["variance_floored_from_training"] = floored
                    subset["precision_clipped_from_training"] = clipped
                    subset["weight_parameter_scope"] = "INNER_TRAINING_FOLD"
                    subset["model_view_version"] = VIEW_VERSION
                    writer.write(subset)
                    weight_rows += len(subset)
    finally:
        writer.close()
    parameters = pd.DataFrame(parameter_rows)
    parameters.to_csv(result_dir / "fold_local_weight_parameters_v2.tsv", sep="\t", index=False)
    expected_weight_rows = selected_rows * len(SCENARIOS) * args.folds
    if weight_rows != expected_weight_rows:
        raise RuntimeError(f"Fold-local weight completeness failed: {weight_rows} != {expected_weight_rows}")
    fold_counts = []
    for scenario, (column, key_column) in SCENARIOS.items():
        for fold in range(args.folds):
            counts = con.execute(f"""
                SELECT sum(({column}!={fold})::BIGINT), sum(({column}={fold})::BIGINT),
                       count(DISTINCT CASE WHEN {column}={fold} THEN {key_column} END)
                FROM read_parquet('{model_sql}')
            """).fetchone()
            fold_counts.append({
                "scenario": scenario, "fold": fold,
                "training_rows": int(counts[0]),
                "validation_rows": int(counts[1]),
                "validation_unique_units": int(counts[2]),
                "protected_membership_used": False,
            })
    pd.DataFrame(fold_counts).to_csv(result_dir / "development_fold_summary_v2.tsv", sep="\t", index=False)
    con.close()
    summary = {
        "status": "PASS_MODEL_VIEWS_AND_FOLD_LOCAL_WEIGHTS",
        "model_view_version": VIEW_VERSION,
        "selected_stage1_rows": selected_rows,
        "selected_traits": sorted(selected_trait_values),
        "unique_genotypes": int(model_stats[2]),
        "unique_environments": int(model_stats[3]),
        "legacy_joint_kernel_eligible_rows": int(model_stats[5]),
        "retained_not_legacy_joint_kernel_eligible_rows": int(selected_rows - model_stats[5]),
        "fold_scenarios": list(SCENARIOS), "folds_per_scenario": args.folds,
        "fold_local_weight_rows": weight_rows,
        "fold_local_weight_parameter_rows": len(parameters),
        "outer_test_content_read": False, "final_holdout_content_read": False,
        "files": {
            model_path.name: {"bytes": model_path.stat().st_size, "sha256": file_sha256(model_path)},
            weight_path.name: {"bytes": weight_path.stat().st_size, "sha256": file_sha256(weight_path)},
        },
    }
    (result_dir / "model_view_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
