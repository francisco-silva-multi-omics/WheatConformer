from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .final_evaluation_contract import file_sha256
from .summarize_nested_evaluation import calibration_parameters, run_record


REQUIRED_PREDICTION_COLUMNS = {
    "canonical_observation_id",
    "split",
    "trait_name_canonical",
    "panel_sample_id",
    "env_kernel_id",
    "phenotype_value",
    "y_pred",
    "y_pred_train_mean",
}


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_ids(path: Path, preferred: str = "env_id") -> pd.Index:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    column = preferred if preferred in frame else frame.columns[0]
    values = frame[column].fillna("").astype(str).str.strip()
    return pd.Index(values[values.ne("")].drop_duplicates(), dtype="object")


def finite_pair(frame: pd.DataFrame, prediction_column: str) -> tuple[np.ndarray, np.ndarray]:
    y = pd.to_numeric(frame["phenotype_value"], errors="coerce").to_numpy(dtype=float)
    prediction = pd.to_numeric(frame[prediction_column], errors="coerce").to_numpy(
        dtype=float
    )
    keep = np.isfinite(y) & np.isfinite(prediction)
    return y[keep], prediction[keep]


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - prediction)))) if len(y) else float("nan")


def mae(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(y - prediction))) if len(y) else float("nan")


def correlation(y: np.ndarray, prediction: np.ndarray, *, rank: bool = False) -> float:
    if len(y) < 2:
        return float("nan")
    if rank:
        y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
        prediction = pd.Series(prediction).rank(method="average").to_numpy(dtype=float)
    if np.std(y) <= 0 or np.std(prediction) <= 0:
        return float("nan")
    return float(np.corrcoef(y, prediction)[0, 1])


def prediction_sd_ratio(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.std(prediction) / np.std(y)) if len(y) > 1 and np.std(y) > 0 else float("nan")


def metric_record(y: np.ndarray, prediction: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_rmse": rmse(y, prediction),
        f"{prefix}_mae": mae(y, prediction),
        f"{prefix}_pearson": correlation(y, prediction),
        f"{prefix}_spearman": correlation(y, prediction, rank=True),
        f"{prefix}_prediction_sd_ratio": prediction_sd_ratio(y, prediction),
    }


def top_k_regret(
    group: pd.DataFrame,
    *,
    fraction: float,
    minimum_top_k: int,
    maximum_fraction: float,
) -> dict[str, float | int]:
    y, prediction = finite_pair(group, "y_pred")
    n = len(y)
    if n < 2:
        return {
            "rows": n,
            "k": 0,
            "upper_tail_regret": float("nan"),
            "lower_tail_regret": float("nan"),
            "upper_tail_regret_sd": float("nan"),
            "lower_tail_regret_sd": float("nan"),
        }
    k = max(minimum_top_k, int(math.ceil(fraction * n)))
    k = min(k, max(1, int(math.floor(maximum_fraction * n))))
    true_upper = np.argpartition(y, n - k)[n - k :]
    predicted_upper = np.argpartition(prediction, n - k)[n - k :]
    true_lower = np.argpartition(y, k - 1)[:k]
    predicted_lower = np.argpartition(prediction, k - 1)[:k]
    upper = max(0.0, float(np.mean(y[true_upper]) - np.mean(y[predicted_upper])))
    lower = max(0.0, float(np.mean(y[predicted_lower]) - np.mean(y[true_lower])))
    scale = float(np.std(y))
    return {
        "rows": n,
        "k": k,
        "upper_tail_regret": upper,
        "lower_tail_regret": lower,
        "upper_tail_regret_sd": upper / scale if scale > 0 else float("nan"),
        "lower_tail_regret_sd": lower / scale if scale > 0 else float("nan"),
    }


def within_environment_diagnostics(
    predictions: pd.DataFrame, protocol: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = dict(protocol["within_environment"])
    environment_column = str(policy["environment_id_column"])
    minimum_rows = int(policy["minimum_rows_per_environment_trait"])
    maximize_traits = set(map(str, policy["maximize_traits"]))
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    keys = ["scenario", "outer_fold", "model_label", "trait_name_canonical"]
    test = predictions[predictions["split"].astype(str).eq("test")].copy()
    for key, frame in test.groupby(keys, sort=True, dropna=False):
        eligible_groups: list[pd.DataFrame] = []
        for environment_id, group in frame.groupby(environment_column, sort=True):
            y_values = pd.to_numeric(group["phenotype_value"], errors="coerce")
            pred_values = pd.to_numeric(group["y_pred"], errors="coerce")
            finite = np.isfinite(y_values) & np.isfinite(pred_values)
            local = group.loc[finite].copy()
            y, pred = finite_pair(local, "y_pred")
            if len(y) < minimum_rows:
                continue
            local["_centered_y"] = y - float(np.mean(y))
            local["_centered_prediction"] = pred - float(np.mean(pred))
            eligible_groups.append(local)
            regret = top_k_regret(
                group,
                fraction=float(policy["top_k_fraction"]),
                minimum_top_k=int(policy["minimum_top_k"]),
                maximum_fraction=float(policy["maximum_top_k_fraction"]),
            )
            trait = str(key[-1])
            detail_rows.append(
                {
                    **dict(zip(keys, key)),
                    "env_kernel_id": environment_id,
                    **regret,
                    "directional_top_k_policy": (
                        "maximize" if trait in maximize_traits else "report_both_tails"
                    ),
                    "directional_top_k_regret": (
                        regret["upper_tail_regret"]
                        if trait in maximize_traits
                        else float("nan")
                    ),
                    "directional_top_k_regret_sd": (
                        regret["upper_tail_regret_sd"]
                        if trait in maximize_traits
                        else float("nan")
                    ),
                }
            )
        if eligible_groups:
            centered = pd.concat(eligible_groups, ignore_index=True)
            y = centered["_centered_y"].to_numpy(dtype=float)
            pred = centered["_centered_prediction"].to_numpy(dtype=float)
            trait_detail = pd.DataFrame(detail_rows)
            selector = np.ones(len(trait_detail), dtype=bool)
            for column, value in zip(keys, key):
                selector &= trait_detail[column].astype(str).eq(str(value)).to_numpy()
            selected_detail = trait_detail.loc[selector]
            summary_rows.append(
                {
                    **dict(zip(keys, key)),
                    "test_rows_total": len(frame),
                    "test_environments_total": frame[environment_column].nunique(),
                    "centered_rows": len(centered),
                    "centered_environments": centered[environment_column].nunique(),
                    "centered_rmse": rmse(y, pred),
                    "centered_pearson": correlation(y, pred),
                    "centered_spearman": correlation(y, pred, rank=True),
                    "upper_tail_regret_mean": selected_detail[
                        "upper_tail_regret"
                    ].mean(),
                    "upper_tail_regret_sd_mean": selected_detail[
                        "upper_tail_regret_sd"
                    ].mean(),
                    "lower_tail_regret_mean": selected_detail[
                        "lower_tail_regret"
                    ].mean(),
                    "lower_tail_regret_sd_mean": selected_detail[
                        "lower_tail_regret_sd"
                    ].mean(),
                    "directional_top_k_policy": selected_detail[
                        "directional_top_k_policy"
                    ].iloc[0],
                    "directional_top_k_regret_mean": selected_detail[
                        "directional_top_k_regret"
                    ].mean(),
                    "directional_top_k_regret_sd_mean": selected_detail[
                        "directional_top_k_regret_sd"
                    ].mean(),
                }
            )
        else:
            summary_rows.append(
                {
                    **dict(zip(keys, key)),
                    "test_rows_total": len(frame),
                    "test_environments_total": frame[environment_column].nunique(),
                    "centered_rows": 0,
                    "centered_environments": 0,
                    "centered_rmse": float("nan"),
                    "centered_pearson": float("nan"),
                    "centered_spearman": float("nan"),
                    "directional_top_k_policy": (
                        "maximize" if str(key[-1]) in maximize_traits else "report_both_tails"
                    ),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def calibration_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["scenario", "outer_fold", "model_label", "trait_name_canonical"]
    for key, frame in predictions.groupby(keys, sort=True, dropna=False):
        val = frame[frame["split"].astype(str).eq("val")]
        test = frame[frame["split"].astype(str).eq("test")]
        if val.empty or test.empty:
            continue
        intercept, slope = calibration_parameters(val)
        y, raw = finite_pair(test, "y_pred")
        calibrated = intercept + slope * raw
        if len(val) < 2 or np.var(pd.to_numeric(val["y_pred"], errors="coerce")) <= 0:
            status = "FLAG_UNIDENTIFIABLE_VALIDATION_SLOPE"
        elif slope < 0:
            status = "FLAG_NEGATIVE_VALIDATION_SLOPE"
        else:
            status = "POSITIVE_VALIDATION_SLOPE"
        rows.append(
            {
                **dict(zip(keys, key)),
                "validation_rows": len(val),
                "test_rows": len(test),
                "calibration_intercept_from_validation": intercept,
                "calibration_slope_from_validation": slope,
                "calibration_status": status,
                "negative_calibration_slope": slope < 0,
                **metric_record(y, raw, "raw"),
                **metric_record(y, calibrated, "calibrated"),
            }
        )
    return pd.DataFrame(rows)


def availability_tables(
    predictions: pd.DataFrame,
    outer_protocol: dict[str, object],
    reporting_protocol: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    traits = list(map(str, outer_protocol["traits"]))
    reporting_policy = dict(reporting_protocol["trait_reporting_policy"])
    for scenario, fold_count in dict(outer_protocol["scenarios"]).items():
        route = dict(outer_protocol["scenario_routes"])[scenario]
        for outer_fold in range(int(fold_count)):
            fold = predictions[
                predictions["scenario"].astype(str).eq(str(scenario))
                & pd.to_numeric(predictions["outer_fold"], errors="coerce").eq(outer_fold)
            ]
            for trait in traits:
                local = fold[fold["trait_name_canonical"].astype(str).eq(trait)]
                val = local[local["split"].astype(str).eq("val")]
                test = local[local["split"].astype(str).eq("test")]
                status = (
                    "AVAILABLE"
                    if not val.empty and not test.empty
                    else "TEST_ONLY_NO_CALIBRATION"
                    if not test.empty
                    else "VALIDATION_ONLY_NO_OUTER_METRIC"
                    if not val.empty
                    else "STRUCTURALLY_UNAVAILABLE"
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "outer_fold": outer_fold,
                        "trait_name_canonical": trait,
                        "trait_reporting_policy": reporting_policy.get(
                            trait, reporting_policy["default"]
                        ),
                        "trial_hierarchy_candidate": route[
                            "trial_hierarchy_candidate"
                        ],
                        "future_environment_compatible": route[
                            "future_environment_compatible"
                        ],
                        "availability_status": status,
                        "validation_rows": len(val),
                        "test_rows": len(test),
                        "validation_genotypes": val["panel_sample_id"].nunique(),
                        "test_genotypes": test["panel_sample_id"].nunique(),
                        "validation_environments": val["env_kernel_id"].nunique(),
                        "test_environments": test["env_kernel_id"].nunique(),
                        "minimum_test_ensemble_members": (
                            pd.to_numeric(
                                test.get("ensemble_member_count"), errors="coerce"
                            ).min()
                            if not test.empty
                            else float("nan")
                        ),
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(
            [
                "scenario",
                "trait_name_canonical",
                "trait_reporting_policy",
                "trial_hierarchy_candidate",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            requested_folds=("outer_fold", "size"),
            available_folds=(
                "availability_status",
                lambda x: int(pd.Series(x).eq("AVAILABLE").sum()),
            ),
            test_rows=("test_rows", "sum"),
            test_environments_mean=("test_environments", "mean"),
            test_genotypes_mean=("test_genotypes", "mean"),
        )
        .reset_index()
    )
    summary["availability_fraction"] = (
        summary["available_folds"] / summary["requested_folds"]
    )
    return detail, summary


def environment_range_diagnostics(
    raw: pd.DataFrame,
    standardized: pd.DataFrame,
    manifest: pd.DataFrame,
    fit_ids: Iterable[str],
    test_ids: Iterable[str],
    *,
    scenario: str,
    outer_fold: int,
    q_low: float,
    q_high: float,
    moderate_z: float,
    extreme_z: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = raw.copy().set_index("env_id")
    standardized = standardized.copy().set_index("env_id")
    fit = pd.Index(list(map(str, fit_ids))).intersection(raw.index)
    test = pd.Index(list(map(str, test_ids))).intersection(raw.index)
    source_manifest = manifest[
        ~manifest["is_missingness_indicator"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
    ].drop_duplicates("feature")
    feature_rows: list[dict[str, object]] = []
    environment_parts: list[pd.DataFrame] = []
    for row in source_manifest.itertuples(index=False):
        feature = str(row.feature)
        if feature not in raw or feature not in standardized:
            continue
        fit_values = pd.to_numeric(raw.loc[fit, feature], errors="coerce")
        test_values = pd.to_numeric(raw.loc[test, feature], errors="coerce")
        test_z = pd.to_numeric(standardized.loc[test, feature], errors="coerce")
        finite_fit = fit_values[np.isfinite(fit_values)]
        finite_test = test_values[np.isfinite(test_values)]
        if finite_fit.empty:
            continue
        minimum = float(finite_fit.min())
        maximum = float(finite_fit.max())
        robust_low = float(finite_fit.quantile(q_low))
        robust_high = float(finite_fit.quantile(q_high))
        below = test_values.lt(minimum)
        above = test_values.gt(maximum)
        outside_robust = test_values.lt(robust_low) | test_values.gt(robust_high)
        abs_z = test_z.abs()
        denominator = max(int(finite_test.size), 1)
        feature_rows.append(
            {
                "scenario": scenario,
                "outer_fold": outer_fold,
                "feature": feature,
                "feature_block": row.feature_block,
                "source_feature": row.source_feature,
                "source_artifact": row.source_artifact,
                "regulatory_treatment": row.regulatory_treatment,
                "training_nonmissing": len(finite_fit),
                "test_nonmissing": len(finite_test),
                "test_missing": int(test_values.isna().sum()),
                "training_min": minimum,
                "training_q01": robust_low,
                "training_q99": robust_high,
                "training_max": maximum,
                "test_below_training_min_fraction": float(below.sum() / denominator),
                "test_above_training_max_fraction": float(above.sum() / denominator),
                "test_outside_training_range_fraction": float(
                    (below | above).sum() / denominator
                ),
                "test_outside_robust_range_fraction": float(
                    outside_robust.sum() / denominator
                ),
                "test_abs_z_gt_moderate_fraction": float(
                    abs_z.gt(moderate_z).sum() / denominator
                ),
                "test_abs_z_gt_extreme_fraction": float(
                    abs_z.gt(extreme_z).sum() / denominator
                ),
                "test_max_abs_z": float(abs_z.max()) if abs_z.notna().any() else float("nan"),
            }
        )
        environment_parts.append(
            pd.DataFrame(
                {
                    "env_id": test,
                    "feature": feature,
                    "missing": test_values.isna().to_numpy(),
                    "outside_training_range": (below | above).to_numpy(),
                    "outside_robust_range": outside_robust.to_numpy(),
                    "abs_z": abs_z.to_numpy(dtype=float),
                }
            )
        )
    detail = pd.DataFrame(feature_rows)
    if not environment_parts:
        return detail, pd.DataFrame()
    long = pd.concat(environment_parts, ignore_index=True)
    env = (
        long.groupby("env_id", sort=True)
        .agg(
            source_feature_count=("feature", "size"),
            missing_source_features=("missing", "sum"),
            outside_training_range_features=("outside_training_range", "sum"),
            outside_robust_range_features=("outside_robust_range", "sum"),
            abs_z_gt_moderate_features=("abs_z", lambda x: int((x > moderate_z).sum())),
            abs_z_gt_extreme_features=("abs_z", lambda x: int((x > extreme_z).sum())),
            maximum_absolute_z=("abs_z", "max"),
        )
        .reset_index()
    )
    for count in [
        "missing_source_features",
        "outside_training_range_features",
        "outside_robust_range_features",
        "abs_z_gt_moderate_features",
        "abs_z_gt_extreme_features",
    ]:
        env[f"{count}_fraction"] = env[count] / env["source_feature_count"]
    env.insert(0, "outer_fold", outer_fold)
    env.insert(0, "scenario", scenario)
    return detail, env


def validate_reporting_inputs(
    outer_protocol: dict[str, object],
    selection_lock: dict[str, object],
    environment_lock: dict[str, object],
    outer_provenance: dict[str, object],
    support_provenance: dict[str, object],
) -> dict[str, bool]:
    checks = {
        "outer_protocol_frozen": outer_protocol.get("status")
        == "frozen_after_inner_validation_before_outer_test",
        "selection_lock_pass": selection_lock.get("status") == "PASS",
        "selection_lock_outer_unread_at_freeze": selection_lock.get(
            "outer_test_metrics_read"
        )
        is False,
        "selection_lock_final_holdout_unread": selection_lock.get(
            "final_holdout_outcomes_read"
        )
        is False,
        "environment_lock_pass": environment_lock.get("status") == "PASS",
        "environment_lock_final_holdout_unread": environment_lock.get(
            "final_holdout_outcomes_read"
        )
        is False,
        "outer_provenance_pass": outer_provenance.get("status") == "PASS",
        "outer_selection_unchanged": outer_provenance.get(
            "further_hyperparameter_selection_performed"
        )
        is False,
        "outer_metrics_unused_for_selection": outer_provenance.get(
            "outer_test_metrics_used_for_selection"
        )
        is False,
        "outer_final_holdout_unread": outer_provenance.get(
            "final_holdout_outcomes_read"
        )
        is False,
        "support_amendment_pass": support_provenance.get("status") == "PASS",
        "support_outcomes_unused": support_provenance.get(
            "outer_test_outcome_values_read"
        )
        is False,
        "support_metrics_unused": support_provenance.get(
            "outer_test_metrics_used_for_selection"
        )
        is False,
        "support_final_holdout_unread": support_provenance.get(
            "final_holdout_outcomes_read"
        )
        is False,
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("Reporting-only contract failed: " + ", ".join(failed))
    return checks


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce reporting-only diagnostics for the frozen routed reaction norm."
    )
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--run-glob", default="final_nested_reaction_norm_*")
    parser.add_argument("--outer-dir", type=Path, required=True)
    parser.add_argument("--outer-protocol", type=Path, required=True)
    parser.add_argument("--reporting-protocol", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--environment-selection-lock", type=Path, required=True)
    parser.add_argument("--outer-provenance", type=Path, required=True)
    parser.add_argument("--support-amendment-provenance", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--factorization-implementation", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    outer_protocol = read_json(args.outer_protocol)
    reporting_protocol = read_json(args.reporting_protocol)
    if reporting_protocol.get("status") != "reporting_only_after_frozen_outer_evaluation":
        raise SystemExit("Reporting protocol is not frozen as reporting-only")
    if reporting_protocol.get("model_selection_allowed") is not False:
        raise SystemExit("Reporting protocol permits model selection")
    checks = validate_reporting_inputs(
        outer_protocol,
        read_json(args.selection_lock),
        read_json(args.environment_selection_lock),
        read_json(args.outer_provenance),
        read_json(args.support_amendment_provenance),
    )

    trainer_sha = file_sha256(args.trainer)
    factorization_sha = file_sha256(args.factorization_implementation)
    frames: list[pd.DataFrame] = []
    lineage: list[dict[str, object]] = []
    for run_dir in sorted(args.models_root.glob(args.run_glob)):
        record = run_record(run_dir, trainer_sha, factorization_sha)
        if record is None:
            continue
        metadata = record[1]
        if metadata.get("external_split", {}).get("inner_fold") != "ensemble":
            continue
        frames.append(record[0])
        lineage.append(record[2])
    if not frames:
        raise SystemExit("No frozen routed outer ensembles were found")
    predictions = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns)
    if missing:
        raise SystemExit(f"Routed predictions are missing columns: {sorted(missing)}")
    observation_key = [
        "scenario",
        "outer_fold",
        "split",
        "canonical_observation_id",
    ]
    if predictions.duplicated(observation_key).any():
        raise SystemExit("Routed outer ensemble observations are not unique within fold")
    if predictions["split"].astype(str).eq("final_holdout").any():
        raise SystemExit("Final-holdout outcomes entered reporting diagnostics")
    expected_ensembles = sum(map(int, dict(outer_protocol["scenarios"]).values()))
    if len(frames) != expected_ensembles:
        raise SystemExit(
            f"Expected {expected_ensembles} routed ensembles; found {len(frames)}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    centered, top_k = within_environment_diagnostics(predictions, reporting_protocol)
    calibration = calibration_diagnostics(predictions)
    availability, availability_summary = availability_tables(
        predictions, outer_protocol, reporting_protocol
    )
    write_tsv(centered, args.out_dir / "within_environment_centered_metrics.tsv")
    write_tsv(top_k, args.out_dir / "within_environment_top_k_regret.tsv")
    write_tsv(calibration, args.out_dir / "raw_calibrated_metrics.tsv")
    write_tsv(availability, args.out_dir / "trait_scenario_availability.tsv")
    write_tsv(
        availability_summary,
        args.out_dir / "trait_scenario_availability_summary.tsv",
    )
    write_tsv(
        pd.DataFrame(lineage), args.out_dir / "reporting_input_prediction_lineage.tsv"
    )

    range_feature_parts: list[pd.DataFrame] = []
    range_environment_parts: list[pd.DataFrame] = []
    range_policy = dict(reporting_protocol["environment_range"])
    for scenario in map(str, range_policy["scenarios"]):
        for outer_fold in range(int(dict(outer_protocol["scenarios"])[scenario])):
            fold = predictions[
                predictions["scenario"].astype(str).eq(scenario)
                & pd.to_numeric(predictions["outer_fold"], errors="coerce").eq(
                    outer_fold
                )
                & predictions["split"].astype(str).eq("test")
            ]
            fold_dir = args.outer_dir / "folds" / scenario / f"outer_{outer_fold}"
            environment_dir = fold_dir / "E_REACTION_NORM_V1"
            certification = read_json(
                environment_dir / "E_REACTION_NORM_V1_certification.json"
            )
            if certification.get("status") != "PASS":
                raise SystemExit(
                    f"Uncertified E_REACTION_NORM_V1 for {scenario} outer={outer_fold}"
                )
            test_ids = fold["env_kernel_id"].astype(str).drop_duplicates()
            matrix_ids = pd.read_parquet(
                environment_dir / "E_REACTION_NORM_V1_raw.parquet",
                columns=["env_id"],
            )["env_id"].astype(str)
            missing_test_ids = sorted(set(test_ids) - set(matrix_ids))
            if missing_test_ids:
                raise SystemExit(
                    f"Environment design misses {len(missing_test_ids)} test IDs for "
                    f"{scenario} outer={outer_fold}"
                )
            feature, environment = environment_range_diagnostics(
                pd.read_parquet(environment_dir / "E_REACTION_NORM_V1_raw.parquet"),
                pd.read_parquet(environment_dir / "E_REACTION_NORM_V1.parquet"),
                pd.read_csv(
                    environment_dir / "E_REACTION_NORM_V1_feature_manifest.tsv",
                    sep="\t",
                    dtype=str,
                ),
                read_ids(fold_dir / "ids" / "outer_training_environment_ids.tsv"),
                test_ids,
                scenario=scenario,
                outer_fold=outer_fold,
                q_low=float(range_policy["robust_quantile_low"]),
                q_high=float(range_policy["robust_quantile_high"]),
                moderate_z=float(range_policy["moderate_absolute_z"]),
                extreme_z=float(range_policy["extreme_absolute_z"]),
            )
            range_feature_parts.append(feature)
            range_environment_parts.append(environment)
    range_features = pd.concat(range_feature_parts, ignore_index=True)
    range_environments = pd.concat(range_environment_parts, ignore_index=True)
    write_tsv(
        range_features, args.out_dir / "environment_extrapolation_by_feature.tsv"
    )
    write_tsv(
        range_environments,
        args.out_dir / "environment_extrapolation_by_environment.tsv",
    )

    output_paths = sorted(
        path
        for path in args.out_dir.iterdir()
        if path.is_file() and path.name != "reaction_norm_reporting_provenance.json"
    )
    provenance = {
        "status": "PASS",
        "protocol_version": reporting_protocol["protocol_version"],
        "reporting_only": True,
        "phenotype_values_read_for_reporting": True,
        "outer_test_metrics_read_for_locked_reporting": True,
        "outer_test_metrics_used_for_selection": False,
        "further_model_selection_performed": False,
        "final_holdout_outcomes_read": False,
        "ensemble_count": len(frames),
        "availability_grid_rows": len(availability),
        "centered_metric_rows": len(centered),
        "top_k_environment_rows": len(top_k),
        "negative_calibration_slope_count": int(
            calibration["negative_calibration_slope"].sum()
        ),
        "environment_range_feature_rows": len(range_features),
        "environment_range_environment_rows": len(range_environments),
        "checks": checks,
        "inputs": {
            "outer_protocol": file_sha256(args.outer_protocol),
            "reporting_protocol": file_sha256(args.reporting_protocol),
            "selection_lock": file_sha256(args.selection_lock),
            "environment_selection_lock": file_sha256(
                args.environment_selection_lock
            ),
            "outer_provenance": file_sha256(args.outer_provenance),
            "support_amendment_provenance": file_sha256(
                args.support_amendment_provenance
            ),
            "trainer": trainer_sha,
            "factorization": factorization_sha,
        },
        "artifacts": {path.name: file_sha256(path) for path in output_paths},
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.out_dir / "reaction_norm_reporting_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
