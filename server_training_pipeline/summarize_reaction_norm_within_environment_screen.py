from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .final_evaluation_contract import file_sha256
from .report_reaction_norm_routed_diagnostics import correlation
from .summarize_reaction_norm_loss_balance_screen import assert_prediction_identity, unique_file


REFERENCE = "current_routed_reference"


def _bounded_pairwise_accuracy(
    y: np.ndarray,
    prediction: np.ndarray,
    tolerance: float,
    seed_label: str,
    maximum_pairs: int = 5000,
) -> tuple[float, int]:
    n = len(y)
    if n < 2:
        return float("nan"), 0
    total = n * (n - 1) // 2
    if total <= maximum_pairs:
        left, right = np.triu_indices(n, k=1)
    else:
        seed = int.from_bytes(hashlib.sha256(seed_label.encode("utf-8")).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        pairs: set[tuple[int, int]] = set()
        attempts = 0
        while len(pairs) < maximum_pairs and attempts < maximum_pairs * 20:
            values = rng.integers(0, n, size=2)
            attempts += 1
            if values[0] == values[1]:
                continue
            pairs.add(tuple(sorted((int(values[0]), int(values[1])))))
        if not pairs:
            return float("nan"), 0
        pair_array = np.asarray(sorted(pairs), dtype=int)
        left, right = pair_array[:, 0], pair_array[:, 1]
    observed = y[left] - y[right]
    predicted = prediction[left] - prediction[right]
    eligible = np.isfinite(observed) & np.isfinite(predicted) & (np.abs(observed) > tolerance)
    if not eligible.any():
        return float("nan"), 0
    observed_sign = np.sign(observed[eligible])
    predicted_sign = np.sign(predicted[eligible])
    score = np.where(predicted_sign == 0, 0.5, (predicted_sign == observed_sign).astype(float))
    return float(np.mean(score)), int(len(score))


def _tail_regret_sd(y: np.ndarray, prediction: np.ndarray, policy: dict[str, object]) -> float:
    n = len(y)
    if n < 2 or np.std(y) <= 0:
        return float("nan")
    k = max(int(policy["minimum_top_k"]), int(math.ceil(float(policy["top_k_fraction"]) * n)))
    k = min(k, max(1, int(math.floor(float(policy["maximum_top_k_fraction"]) * n))))
    true_upper = np.argpartition(y, n - k)[n - k :]
    predicted_upper = np.argpartition(prediction, n - k)[n - k :]
    true_lower = np.argpartition(y, k - 1)[:k]
    predicted_lower = np.argpartition(prediction, k - 1)[:k]
    upper = max(0.0, float(np.mean(y[true_upper]) - np.mean(y[predicted_upper])))
    lower = max(0.0, float(np.mean(y[predicted_lower]) - np.mean(y[true_lower])))
    return float((upper + lower) / (2.0 * np.std(y)))


def validation_within_environment_metrics(
    predictions: pd.DataFrame, policy: dict[str, object]
) -> pd.DataFrame:
    environment_column = str(policy["environment_id_column"])
    minimum_rows = int(policy["minimum_rows_per_environment_trait"])
    minimum_pairs = int(policy["minimum_comparable_pairs_per_environment_trait"])
    tolerance = float(policy["pairwise_tie_tolerance_standardized"])
    rows: list[dict[str, object]] = []
    for trait, trait_frame in predictions.groupby("trait_name_canonical", sort=True):
        centered_y: list[np.ndarray] = []
        centered_prediction: list[np.ndarray] = []
        pair_correct = 0.0
        pair_count = 0
        regret: list[float] = []
        environment_count = 0
        for environment, group in trait_frame.groupby(environment_column, sort=True):
            y = pd.to_numeric(group["y_scaled"], errors="coerce").to_numpy(float)
            prediction = pd.to_numeric(group["y_pred_scaled"], errors="coerce").to_numpy(float)
            finite = np.isfinite(y) & np.isfinite(prediction)
            y, prediction = y[finite], prediction[finite]
            if len(y) < minimum_rows:
                continue
            environment_count += 1
            centered_y.append(y - np.mean(y))
            centered_prediction.append(prediction - np.mean(prediction))
            accuracy, pairs = _bounded_pairwise_accuracy(
                y,
                prediction,
                tolerance,
                f"{trait}|{environment}",
            )
            if pairs >= minimum_pairs and np.isfinite(accuracy):
                pair_correct += accuracy * pairs
                pair_count += pairs
            value = _tail_regret_sd(y, prediction, policy)
            if np.isfinite(value):
                regret.append(value)
        if centered_y:
            y_all = np.concatenate(centered_y)
            prediction_all = np.concatenate(centered_prediction)
            centered_rmse = float(np.sqrt(np.mean(np.square(y_all - prediction_all))))
            centered_pearson = correlation(y_all, prediction_all)
            centered_spearman = correlation(y_all, prediction_all, rank=True)
            centered_rows = len(y_all)
        else:
            centered_rmse = centered_pearson = centered_spearman = float("nan")
            centered_rows = 0
        rows.append(
            {
                "trait_name_canonical": trait,
                "centered_rows": centered_rows,
                "centered_environments": environment_count,
                "centered_standardized_rmse": centered_rmse,
                "centered_pearson": centered_pearson,
                "centered_spearman": centered_spearman,
                "pairwise_ordering_accuracy": pair_correct / pair_count
                if pair_count
                else float("nan"),
                "comparable_pairs": pair_count,
                "tail_regret_sd": float(np.mean(regret)) if regret else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def load_run(run_dir: Path, policy: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    metadata = json.loads(unique_file(run_dir, "*_run_metadata.json").read_text(encoding="utf-8"))
    macro = pd.read_csv(unique_file(run_dir, "*_macro_metrics.tsv"), sep="\t")
    traits = pd.read_csv(unique_file(run_dir, "*_trait_metrics.tsv"), sep="\t")
    predictions = pd.read_parquet(unique_file(run_dir, "*_predictions.parquet"))
    objective = metadata.get("within_environment_objective", {})
    if metadata.get("status") != "PASS" or metadata.get("evaluation_stage") != "inner_selection":
        raise ValueError(f"Within-environment run is not a PASS inner run: {run_dir}")
    if metadata.get("outer_test_metrics_read") is not False:
        raise ValueError(f"Within-environment run read outer metrics: {run_dir}")
    if metadata.get("final_holdout_outcomes_read") is not False:
        raise ValueError(f"Within-environment run read final holdout: {run_dir}")
    if not predictions["split"].astype(str).eq("val").all():
        raise ValueError(f"Within-environment predictions expose non-validation rows: {run_dir}")
    label = str(metadata["model_label"])
    selected_macro = macro[
        macro["split"].astype(str).eq("val") & macro["model"].astype(str).eq(label)
    ]
    selected_traits = traits[
        traits["split"].astype(str).eq("val")
        & traits["model"].astype(str).eq(label)
        & traits["coverage_group"].astype(str).eq("all")
    ].copy()
    if len(selected_macro) != 1 or selected_traits.empty:
        raise ValueError(f"Within-environment validation metrics are incomplete: {run_dir}")
    within = validation_within_environment_metrics(predictions, policy)
    finite_within = within[
        within[["centered_spearman", "pairwise_ordering_accuracy", "tail_regret_sd"]]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .all(axis=1)
    ]
    if finite_within.empty:
        raise ValueError(f"Within-environment validation support is empty: {run_dir}")
    calibration_error = float(
        np.abs(pd.to_numeric(selected_traits["prediction_sd_ratio"], errors="coerce") - 1.0).mean()
    )
    external = metadata["external_split"]
    row = {
        "run_dir": str(run_dir.resolve()),
        "scenario": str(external["scenario"]),
        "outer_fold": int(external["outer_fold"]),
        "inner_fold": int(external["inner_fold"]),
        "candidate": str(objective.get("candidate", "")),
        "seed": int(metadata["seed"]),
        "manifest_sha256": str(external.get("manifest_sha256", "")),
        "training_configuration": json.dumps(metadata.get("training_configuration", {}), sort_keys=True),
        "active_kernels": json.dumps(sorted(metadata.get("active_kernels", []))),
        "training_input_identities": json.dumps(
            metadata.get("training_input_identities", {}), sort_keys=True
        ),
        "val_normalized_rmse": float(selected_macro.iloc[0]["macro_normalized_rmse"]),
        "val_pearson": float(selected_macro.iloc[0]["macro_pearson"]),
        "val_calibration_error": calibration_error,
        "val_centered_spearman": float(finite_within["centered_spearman"].mean()),
        "val_pairwise_accuracy": float(finite_within["pairwise_ordering_accuracy"].mean()),
        "val_tail_regret_sd": float(finite_within["tail_regret_sd"].mean()),
    }
    return row, within, predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize matched inner-only within-environment objective candidates."
    )
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=["phase_1", "confirmation"], required=True)
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    known = {str(value["name"]) for value in protocol["candidates"]}
    candidates = set(args.candidate or known)
    candidates.add(REFERENCE)
    if not candidates.issubset(known):
        raise ValueError(f"Unknown within-environment candidates: {sorted(candidates-known)}")
    fold_spec = protocol[args.phase]["outer_folds_by_scenario"]
    inner_folds = int(protocol[args.phase]["inner_folds"])
    policy = protocol["within_environment_metrics"]

    rows: list[dict[str, object]] = []
    within_lookup: dict[tuple[str, int, int, str], pd.DataFrame] = {}
    prediction_lookup: dict[tuple[str, int, int, str], pd.DataFrame] = {}
    for run_dir in sorted(args.models_dir.glob("within_environment_inner_*")):
        metadata_paths = list(run_dir.glob("*_run_metadata.json"))
        if len(metadata_paths) != 1:
            continue
        metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
        external = metadata.get("external_split", {})
        scenario = str(external.get("scenario", ""))
        outer = int(external.get("outer_fold", -1))
        candidate = str(metadata.get("within_environment_objective", {}).get("candidate", ""))
        if scenario not in fold_spec or outer not in fold_spec[scenario] or candidate not in candidates:
            continue
        row, within, predictions = load_run(run_dir, policy)
        key = (scenario, outer, int(row["inner_fold"]), candidate)
        if key in within_lookup:
            raise ValueError(f"Duplicate within-environment run: {key}")
        rows.append(row)
        within_lookup[key] = within
        prediction_lookup[key] = predictions
    runs = pd.DataFrame(rows)
    expected = {
        (scenario, int(outer), inner, candidate)
        for scenario, outer_folds in fold_spec.items()
        for outer in outer_folds
        for inner in range(inner_folds)
        for candidate in candidates
    }
    if set(within_lookup) != expected:
        raise ValueError(
            "Within-environment run grid is incomplete: "
            f"missing={sorted(expected-set(within_lookup))[:20]}; "
            f"extra={sorted(set(within_lookup)-expected)[:20]}"
        )

    index = runs.set_index(["scenario", "outer_fold", "inner_fold", "candidate"])
    paired_rows: list[dict[str, object]] = []
    trait_rows: list[pd.DataFrame] = []
    for scenario, outer_folds in fold_spec.items():
        for outer in outer_folds:
            for inner in range(inner_folds):
                reference_key = (scenario, int(outer), inner, REFERENCE)
                reference = index.loc[reference_key]
                for candidate in sorted(candidates - {REFERENCE}):
                    key = (scenario, int(outer), inner, candidate)
                    current = index.loc[key]
                    for field in (
                        "seed",
                        "manifest_sha256",
                        "training_configuration",
                        "active_kernels",
                        "training_input_identities",
                    ):
                        if current[field] != reference[field]:
                            raise ValueError(f"Matched objective candidates disagree on {field}: {key}")
                    assert_prediction_identity(
                        prediction_lookup[key], prediction_lookup[reference_key]
                    )
                    paired_rows.append(
                        {
                            "scenario": scenario,
                            "outer_fold": int(outer),
                            "inner_fold": inner,
                            "candidate": candidate,
                            "seed": int(current["seed"]),
                            "relative_normalized_rmse_gain": (
                                reference["val_normalized_rmse"] - current["val_normalized_rmse"]
                            )
                            / reference["val_normalized_rmse"],
                            "pearson_gain": current["val_pearson"] - reference["val_pearson"],
                            "calibration_error_delta": current["val_calibration_error"]
                            - reference["val_calibration_error"],
                            "centered_spearman_gain": current["val_centered_spearman"]
                            - reference["val_centered_spearman"],
                            "pairwise_accuracy_gain": current["val_pairwise_accuracy"]
                            - reference["val_pairwise_accuracy"],
                            "tail_regret_sd_delta": current["val_tail_regret_sd"]
                            - reference["val_tail_regret_sd"],
                        }
                    )
                    left = within_lookup[key].rename(
                        columns=lambda value: f"{value}_candidate"
                        if value != "trait_name_canonical"
                        else value
                    )
                    right = within_lookup[reference_key].rename(
                        columns=lambda value: f"{value}_reference"
                        if value != "trait_name_canonical"
                        else value
                    )
                    trait_pair = left.merge(right, on="trait_name_canonical", validate="one_to_one")
                    trait_pair.insert(0, "candidate", candidate)
                    trait_pair.insert(0, "inner_fold", inner)
                    trait_pair.insert(0, "outer_fold", int(outer))
                    trait_pair.insert(0, "scenario", scenario)
                    trait_pair["centered_spearman_gain"] = (
                        trait_pair["centered_spearman_candidate"]
                        - trait_pair["centered_spearman_reference"]
                    )
                    trait_pair["pairwise_accuracy_gain"] = (
                        trait_pair["pairwise_ordering_accuracy_candidate"]
                        - trait_pair["pairwise_ordering_accuracy_reference"]
                    )
                    trait_pair["tail_regret_sd_delta"] = (
                        trait_pair["tail_regret_sd_candidate"]
                        - trait_pair["tail_regret_sd_reference"]
                    )
                    trait_rows.append(trait_pair)

    paired = pd.DataFrame(paired_rows)
    trait_paired = pd.concat(trait_rows, ignore_index=True)
    summary = (
        paired.groupby("candidate", sort=True)
        .agg(
            paired_inner_folds=("inner_fold", "size"),
            centered_spearman_gain_mean=("centered_spearman_gain", "mean"),
            centered_spearman_win_rate=(
                "centered_spearman_gain",
                lambda values: float((values > 0).mean()),
            ),
            pairwise_accuracy_gain_mean=("pairwise_accuracy_gain", "mean"),
            tail_regret_sd_delta_mean=("tail_regret_sd_delta", "mean"),
            relative_normalized_rmse_gain_mean=("relative_normalized_rmse_gain", "mean"),
            pearson_gain_mean=("pearson_gain", "mean"),
            calibration_error_delta_mean=("calibration_error_delta", "mean"),
        )
        .reset_index()
    )
    trait_summary = (
        trait_paired.groupby(["candidate", "trait_name_canonical"], sort=True)
        .agg(
            paired_trait_folds=("centered_spearman_gain", "count"),
            centered_spearman_gain_mean=("centered_spearman_gain", "mean"),
            pairwise_accuracy_gain_mean=("pairwise_accuracy_gain", "mean"),
            tail_regret_sd_delta_mean=("tail_regret_sd_delta", "mean"),
        )
        .reset_index()
    )
    acceptance = protocol["acceptance"]
    decisions: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        primary = trait_summary[
            trait_summary["candidate"].eq(row.candidate)
            & trait_summary["trait_name_canonical"].isin(acceptance["primary_guard_traits"])
        ]
        guards = {
            "centered_spearman": row.centered_spearman_gain_mean
            >= float(acceptance["minimum_centered_spearman_gain"]),
            "centered_spearman_win_rate": row.centered_spearman_win_rate
            >= float(acceptance["minimum_centered_spearman_fold_win_rate"]),
            "pairwise_accuracy": row.pairwise_accuracy_gain_mean
            >= float(acceptance["minimum_pairwise_accuracy_gain"]),
            "tail_regret": row.tail_regret_sd_delta_mean
            <= float(acceptance["maximum_tail_regret_increase_sd"]),
            "global_nrmse": row.relative_normalized_rmse_gain_mean
            >= -float(acceptance["maximum_global_relative_normalized_rmse_loss"]),
            "global_pearson": row.pearson_gain_mean
            >= -float(acceptance["maximum_global_pearson_drop"]),
            "calibration": row.calibration_error_delta_mean
            <= float(acceptance["maximum_calibration_error_increase"]),
            "primary_traits": len(primary) == len(acceptance["primary_guard_traits"])
            and primary["centered_spearman_gain_mean"].ge(
                -float(acceptance["maximum_primary_trait_centered_spearman_loss"])
            ).all(),
        }
        decisions.append(
            {
                "candidate": row.candidate,
                **{f"guard_{name}": bool(value) for name, value in guards.items()},
                "accepted": all(guards.values()),
            }
        )
    decision = summary.merge(pd.DataFrame(decisions), on="candidate")
    accepted = decision[decision["accepted"]]
    selected = (
        accepted.sort_values(
            ["centered_spearman_gain_mean", "pairwise_accuracy_gain_mean", "relative_normalized_rmse_gain_mean"],
            ascending=False,
        ).iloc[0]["candidate"]
        if not accepted.empty
        else REFERENCE
    )
    decision["decision"] = np.where(
        decision["candidate"].eq(selected) & decision["accepted"],
        "advance_to_full_inner_confirmation"
        if args.phase == "phase_1"
        else "freeze_for_cross_scenario_inner_guard",
        "do_not_advance",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "runs": args.out_dir / "within_environment_inner_screen_runs.tsv",
        "paired": args.out_dir / "within_environment_inner_screen_paired_metrics.tsv",
        "traits": args.out_dir / "within_environment_inner_screen_trait_metrics.tsv",
        "summary": args.out_dir / "within_environment_inner_screen_summary.tsv",
        "trait_summary": args.out_dir / "within_environment_inner_screen_trait_summary.tsv",
        "decision": args.out_dir / "within_environment_inner_screen_decision.tsv",
    }
    for frame, key in (
        (runs, "runs"),
        (paired, "paired"),
        (trait_paired, "traits"),
        (summary, "summary"),
        (trait_summary, "trait_summary"),
        (decision, "decision"),
    ):
        frame.to_csv(artifacts[key], sep="\t", index=False)
    provenance = {
        "status": "PASS",
        "protocol_version": "reaction_norm_within_environment_inner_screen_v1",
        "phase": args.phase,
        "selection_data": "inner_validation_only",
        "inner_validation_phenotype_values_read": True,
        "prior_routed_outer_metrics_known": True,
        "prior_routed_outer_metrics_used_as_screen_inputs": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "selected_candidate": selected,
        "candidate_count": len(candidates),
        "run_count": len(runs),
        "paired_inner_fold_count": len(paired) // max(len(candidates) - 1, 1),
        "matched_seed_status": "pass",
        "matched_validation_observation_status": "pass",
        "matched_training_configuration_status": "pass",
        "matched_kernel_identity_status": "pass",
        "protocol_sha256": file_sha256(args.protocol),
        "trainer_sha256": file_sha256(args.trainer),
        "acceptance": acceptance,
        "artifacts": {name: file_sha256(path) for name, path in artifacts.items()},
    }
    provenance_path = args.out_dir / "within_environment_inner_screen_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, allow_nan=False))
    print("\n=== WITHIN-ENVIRONMENT OBJECTIVE DECISION ===")
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
