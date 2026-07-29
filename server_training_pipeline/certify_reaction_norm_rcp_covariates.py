from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .final_evaluation_contract import file_sha256
from .report_reaction_norm_routed_diagnostics import resolve_provenance_source


FORBIDDEN_OUTCOME_COLUMNS = {
    "phenotype_value",
    "target",
    "y",
    "y_true",
    "y_pred",
    "final_holdout_outcome",
}


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Certify future reaction-norm covariates against every fold-local historical range."
    )
    parser.add_argument("--future-raw-matrix", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--projection-plan", type=Path, required=True)
    parser.add_argument("--outer-dir", type=Path, required=True)
    parser.add_argument("--outer-protocol", type=Path, required=True)
    parser.add_argument("--projection-protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    projection_plan = json.loads(args.projection_plan.read_text(encoding="utf-8"))
    outer = json.loads(args.outer_protocol.read_text(encoding="utf-8"))
    protocol = json.loads(args.projection_protocol.read_text(encoding="utf-8"))
    range_policy = dict(protocol["range_certification"])
    if projection_plan.get("status") != "PASS" or projection_plan.get(
        "projection_allowed"
    ) is not False:
        raise SystemExit("RCP population plan is absent or has already bypassed certification")
    if protocol.get("status") != "planning_only_projection_blocked_pending_covariate_certification":
        raise SystemExit("RCP projection protocol is not in its frozen pre-certification state")
    future = pd.read_parquet(args.future_raw_matrix)
    forbidden = sorted(FORBIDDEN_OUTCOME_COLUMNS.intersection(future.columns))
    if forbidden:
        raise SystemExit(f"Future covariate matrix contains outcome columns: {forbidden}")
    if "future_env_id" not in future:
        raise SystemExit("Future covariate matrix requires future_env_id")
    future["future_env_id"] = future["future_env_id"].fillna("").astype(str).str.strip()
    if future["future_env_id"].eq("").any() or future["future_env_id"].duplicated().any():
        raise SystemExit("future_env_id values must be unique and nonempty")

    plan_dir = args.projection_plan.parent
    feature_plan = pd.read_csv(
        plan_dir / "E_REACTION_NORM_RCP_V1_feature_population_plan.tsv",
        sep="\t",
        dtype=str,
    )
    source_features = feature_plan[
        ~feature_plan["is_missingness_indicator"].str.lower().isin({"true", "1", "yes"})
    ]["feature"].drop_duplicates()
    missing_features = sorted(set(source_features) - set(future.columns))
    numeric = future.reindex(columns=source_features).apply(pd.to_numeric, errors="coerce")
    finite_population_fraction = float(np.isfinite(numeric.to_numpy(dtype=float)).mean())
    nonfinite_environment_fraction = float(
        (~np.isfinite(numeric.to_numpy(dtype=float))).any(axis=1).mean()
    )

    feature_rows: list[dict[str, object]] = []
    environment_rows: list[dict[str, object]] = []
    fold_reference_rows: list[dict[str, object]] = []
    generated_matrices: list[Path] = []
    hard_limit = float(range_policy["hard_absolute_z_limit"])
    extreme_z = float(range_policy["extreme_absolute_z"])
    moderate_z = float(range_policy["moderate_absolute_z"])
    q_low = 0.01
    q_high = 0.99

    for scenario, fold_count in dict(outer["scenarios"]).items():
        for outer_fold in range(int(fold_count)):
            reference = (
                args.outer_dir
                / "folds"
                / str(scenario)
                / f"outer_{outer_fold}"
                / "E_REACTION_NORM_V1"
            )
            certification_path = reference / "E_REACTION_NORM_V1_certification.json"
            certification = json.loads(certification_path.read_text(encoding="utf-8"))
            if certification.get("status") != "PASS":
                raise SystemExit(
                    f"Historical reference is uncertified: {scenario} outer={outer_fold}"
                )
            raw = pd.read_parquet(reference / "E_REACTION_NORM_V1_raw.parquet").set_index(
                "env_id"
            )
            standardized_reference = pd.read_parquet(
                reference / "E_REACTION_NORM_V1.parquet"
            )
            manifest = pd.read_csv(
                reference / "E_REACTION_NORM_V1_feature_manifest.tsv",
                sep="\t",
                dtype=str,
            )
            scaling = pd.read_csv(
                reference / "E_REACTION_NORM_V1_scaling.tsv", sep="\t"
            ).set_index("feature")
            missingness_path = reference / "E_REACTION_NORM_V1_missingness_indicators.tsv"
            missingness = (
                pd.read_csv(missingness_path, sep="\t").set_index("feature")
                if missingness_path.exists() and missingness_path.stat().st_size > 0
                else pd.DataFrame()
            )
            reference_provenance = json.loads(
                (reference / "E_REACTION_NORM_V1_provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            fit_source = resolve_provenance_source(
                reference_provenance["sources"]["fit_environment_ids"],
                data_root=args.root.resolve(),
            )
            fit_frame = pd.read_csv(fit_source, sep="\t", dtype=str)
            fit_column = "env_id" if "env_id" in fit_frame else fit_frame.columns[0]
            fit_ids = pd.Index(fit_frame[fit_column].astype(str)).intersection(raw.index)
            projected = pd.DataFrame(index=future["future_env_id"])
            source_z = pd.DataFrame(index=future["future_env_id"])
            fold_missing: list[str] = []
            for manifest_row in manifest.itertuples(index=False):
                feature = str(manifest_row.feature)
                is_missing = str(manifest_row.is_missingness_indicator).lower() in {
                    "true",
                    "1",
                    "yes",
                }
                source = feature.removesuffix("__missing")
                if source not in future:
                    fold_missing.append(source)
                    projected[feature] = np.nan
                    continue
                values = pd.to_numeric(future[source], errors="coerce")
                if is_missing:
                    if missingness.empty or feature not in missingness.index:
                        raise SystemExit(
                            f"Missing historical missingness scaler for {feature}"
                        )
                    p = float(missingness.loc[feature, "fit_missing_fraction"])
                    std = math.sqrt(p * (1.0 - p))
                    projected[feature] = (
                        (values.isna().astype(float) - p) / std
                    ).to_numpy(dtype=float)
                    continue
                if source not in scaling.index:
                    raise SystemExit(f"Missing historical scaler for {source}")
                scaler = scaling.loc[source]
                mean = float(scaler["mean"])
                std = float(scaler["std"])
                z = (values - mean) / std
                projected[feature] = z.to_numpy(dtype=float)
                source_z[source] = z.to_numpy(dtype=float)
                fit_values = pd.to_numeric(raw.loc[fit_ids, source], errors="coerce")
                finite_fit = fit_values[np.isfinite(fit_values)]
                finite_future = values[np.isfinite(values)]
                minimum = float(finite_fit.min())
                maximum = float(finite_fit.max())
                low = float(finite_fit.quantile(q_low))
                high = float(finite_fit.quantile(q_high))
                denominator = max(len(finite_future), 1)
                feature_rows.append(
                    {
                        "scenario": scenario,
                        "outer_fold": outer_fold,
                        "feature": source,
                        "feature_block": manifest_row.feature_block,
                        "future_rows": len(values),
                        "future_nonfinite_rows": int((~np.isfinite(values)).sum()),
                        "historical_training_min": minimum,
                        "historical_training_q01": low,
                        "historical_training_q99": high,
                        "historical_training_max": maximum,
                        "future_below_training_min_fraction": float(
                            values.lt(minimum).sum() / denominator
                        ),
                        "future_above_training_max_fraction": float(
                            values.gt(maximum).sum() / denominator
                        ),
                        "future_outside_robust_range_fraction": float(
                            (values.lt(low) | values.gt(high)).sum() / denominator
                        ),
                        "future_abs_z_gt_moderate_fraction": float(
                            z.abs().gt(moderate_z).sum() / denominator
                        ),
                        "future_abs_z_gt_extreme_fraction": float(
                            z.abs().gt(extreme_z).sum() / denominator
                        ),
                        "future_max_abs_z": float(z.abs().max()),
                    }
                )
            projected = projected.reindex(columns=manifest["feature"].tolist())
            projected.insert(0, "future_env_id", future["future_env_id"].to_numpy())
            matrix_path = (
                args.out_dir
                / "folds"
                / str(scenario)
                / f"outer_{outer_fold}"
                / "E_REACTION_NORM_RCP_V1.parquet"
            )
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            projected.to_parquet(matrix_path, index=False)
            generated_matrices.append(matrix_path)
            abs_z = source_z.abs()
            environment = pd.DataFrame(
                {
                    "scenario": scenario,
                    "outer_fold": outer_fold,
                    "future_env_id": future["future_env_id"].to_numpy(),
                    "source_feature_count": source_z.shape[1],
                    "nonfinite_source_features": (~np.isfinite(source_z))
                    .sum(axis=1)
                    .to_numpy(),
                    "abs_z_gt_moderate_features": abs_z.gt(moderate_z)
                    .sum(axis=1)
                    .to_numpy(),
                    "abs_z_gt_extreme_features": abs_z.gt(extreme_z)
                    .sum(axis=1)
                    .to_numpy(),
                    "maximum_absolute_z": abs_z.max(axis=1).to_numpy(),
                }
            )
            environment["abs_z_gt_extreme_feature_fraction"] = (
                environment["abs_z_gt_extreme_features"]
                / environment["source_feature_count"]
            )
            environment_rows.extend(environment.to_dict("records"))
            fold_reference_rows.append(
                {
                    "scenario": scenario,
                    "outer_fold": outer_fold,
                    "historical_feature_count": len(manifest),
                    "future_missing_feature_count": len(set(fold_missing)),
                    "future_maximum_absolute_z": float(abs_z.max().max()),
                    "historical_certification_sha256": file_sha256(
                        certification_path
                    ),
                    "historical_standardized_matrix_columns": len(
                        standardized_reference.columns
                    )
                    - 1,
                }
            )

    feature_diagnostics = pd.DataFrame(feature_rows)
    environment_diagnostics = pd.DataFrame(environment_rows)
    fold_references = pd.DataFrame(fold_reference_rows)
    expected_folds = sum(map(int, dict(outer["scenarios"]).values()))
    checks = {
        "projection_plan_pass": projection_plan.get("status") == "PASS",
        "phenotype_columns_absent": not forbidden,
        "future_ids_unique_nonempty": future["future_env_id"].ne("").all()
        and not future["future_env_id"].duplicated().any(),
        "all_planned_source_features_present": not missing_features,
        "minimum_feature_population_fraction": finite_population_fraction
        >= float(range_policy["minimum_feature_population_fraction"]),
        "nonfinite_environment_fraction": nonfinite_environment_fraction
        <= float(range_policy["maximum_environment_fraction_with_any_nonfinite_feature"]),
        "every_fold_reference_checked": len(fold_references) == expected_folds,
        "hard_absolute_z_limit": environment_diagnostics[
            "maximum_absolute_z"
        ].max()
        <= hard_limit,
        "extreme_z_feature_fraction": environment_diagnostics[
            "abs_z_gt_extreme_feature_fraction"
        ].max()
        <= float(
            range_policy[
                "maximum_source_feature_fraction_above_extreme_z_per_environment"
            ]
        ),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = sorted(name for name, passed in checks.items() if not passed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.out_dir / "E_REACTION_NORM_RCP_V1_range_by_feature.tsv"
    environment_path = (
        args.out_dir / "E_REACTION_NORM_RCP_V1_range_by_environment.tsv"
    )
    fold_path = args.out_dir / "E_REACTION_NORM_RCP_V1_range_by_fold.tsv"
    write_tsv(feature_diagnostics, feature_path)
    write_tsv(environment_diagnostics, environment_path)
    write_tsv(fold_references, fold_path)
    artifacts = [feature_path, environment_path, fold_path, *generated_matrices]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "protocol_version": protocol["protocol_version"],
        "projection_allowed": not failed,
        "selection_data": "phenotype_blind_future_environment_covariates_only",
        "phenotype_values_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "future_environment_count": len(future),
        "historical_fold_reference_count": len(fold_references),
        "feature_population_fraction": finite_population_fraction,
        "nonfinite_environment_fraction": nonfinite_environment_fraction,
        "checks": checks,
        "failed_checks": failed,
        "inputs": {
            "future_raw_matrix": file_sha256(args.future_raw_matrix),
            "projection_plan": file_sha256(args.projection_plan),
            "outer_protocol": file_sha256(args.outer_protocol),
            "projection_protocol": file_sha256(args.projection_protocol),
        },
        "artifacts": {
            str(path.relative_to(args.out_dir)): file_sha256(path) for path in artifacts
        },
        "certifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    certification_path = (
        args.out_dir / "E_REACTION_NORM_RCP_V1_covariate_certification.json"
    )
    certification_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    if failed:
        raise SystemExit("RCP covariate-range certification failed")


if __name__ == "__main__":
    main()
