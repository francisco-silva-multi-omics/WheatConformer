from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .final_evaluation_contract import file_sha256


REFERENCE = "current_trait_row_balanced"


def unique_file(run_dir: Path, pattern: str) -> Path:
    paths = list(run_dir.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} in {run_dir}; found {len(paths)}")
    return paths[0]


def read_predictions(run_dir: Path) -> pd.DataFrame:
    paths = list(run_dir.glob("*_predictions.parquet"))
    if len(paths) != 1:
        raise ValueError(f"Expected one prediction ledger in {run_dir}")
    frame = pd.read_parquet(paths[0])
    if not frame["split"].astype(str).eq("val").all():
        raise ValueError(f"Inner loss-balance run exposes non-validation rows: {run_dir}")
    return frame


def load_run(run_dir: Path) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    metadata = json.loads(
        unique_file(run_dir, "*_run_metadata.json").read_text(encoding="utf-8")
    )
    macro = pd.read_csv(unique_file(run_dir, "*_macro_metrics.tsv"), sep="\t")
    traits = pd.read_csv(unique_file(run_dir, "*_trait_metrics.tsv"), sep="\t")
    if metadata.get("status") != "PASS":
        raise ValueError(f"Run is not PASS: {run_dir}")
    if metadata.get("evaluation_stage") != "inner_selection":
        raise ValueError(f"Run is not inner selection: {run_dir}")
    if metadata.get("outer_test_metrics_read") is not False:
        raise ValueError(f"Run read outer-test metrics: {run_dir}")
    if metadata.get("final_holdout_outcomes_read") is not False:
        raise ValueError(f"Run read final-holdout outcomes: {run_dir}")
    if macro["split"].astype(str).eq("test").any():
        raise ValueError(f"Macro table exposes test metrics: {run_dir}")
    model_label = str(metadata["model_label"])
    model_macro = macro[
        macro["split"].astype(str).eq("val")
        & macro["model"].astype(str).eq(model_label)
    ]
    model_traits = traits[
        traits["split"].astype(str).eq("val")
        & traits["model"].astype(str).eq(model_label)
        & traits["coverage_group"].astype(str).eq("all")
    ].copy()
    if len(model_macro) != 1 or model_traits.empty:
        raise ValueError(f"Validation metrics are incomplete: {run_dir}")
    core = model_traits[["normalized_rmse", "pearson", "prediction_sd_ratio"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(core.to_numpy(float)).all():
        raise ValueError(f"Validation metrics are non-finite: {run_dir}")
    external = metadata["external_split"]
    loss = metadata.get("loss_balance", {})
    row = {
        "run_dir": str(run_dir.resolve()),
        "scenario": str(external["scenario"]),
        "outer_fold": int(external["outer_fold"]),
        "inner_fold": int(external["inner_fold"]),
        "candidate": str(loss.get("candidate", "")),
        "seed": int(metadata["seed"]),
        "manifest_sha256": str(external.get("manifest_sha256", "")),
        "evaluation_protocol_sha256": str(
            metadata.get("evaluation_protocol", {}).get("protocol_sha256", "")
        ),
        "loss_protocol_sha256": str(loss.get("protocol_sha256", "")),
        "training_configuration": json.dumps(
            metadata.get("training_configuration", {}), sort_keys=True
        ),
        "active_kernels": json.dumps(sorted(metadata.get("active_kernels", []))),
        "training_input_identities": json.dumps(
            metadata.get("training_input_identities", {}), sort_keys=True
        ),
        "val_normalized_rmse": float(model_macro.iloc[0]["macro_normalized_rmse"]),
        "val_pearson": float(model_macro.iloc[0]["macro_pearson"]),
        "val_calibration_error": float(
            np.abs(core["prediction_sd_ratio"].to_numpy(float) - 1.0).mean()
        ),
    }
    return row, model_traits, read_predictions(run_dir)


def assert_prediction_identity(left: pd.DataFrame, right: pd.DataFrame) -> None:
    columns = [
        "canonical_observation_id",
        "genotype_id",
        "environment_id",
        "trait_name_canonical",
        "phenotype_value",
    ]
    if len(left) != len(right):
        raise ValueError("Matched loss candidates predict different validation row counts")
    for column in columns:
        if column not in left or column not in right:
            raise ValueError(f"Prediction identity column is absent: {column}")
        a = left[column].to_numpy()
        b = right[column].to_numpy()
        equal = (
            np.array_equal(a.astype(np.float64), b.astype(np.float64))
            if column == "phenotype_value"
            else np.array_equal(a.astype(str), b.astype(str))
        )
        if not equal:
            raise ValueError(f"Matched validation predictions disagree on {column}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize matched inner-only reaction-norm loss-balance candidates."
    )
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--loss-balance-protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=["phase_1", "confirmation"], required=True)
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.loss_balance_protocol.read_text(encoding="utf-8"))
    policy_names = {str(value["name"]) for value in protocol["candidates"]}
    candidates = set(args.candidate or policy_names)
    candidates.add(REFERENCE)
    if not candidates.issubset(policy_names):
        raise ValueError(f"Unknown loss candidates: {sorted(candidates-policy_names)}")
    fold_spec = protocol[args.phase]["outer_folds_by_scenario"]
    expected_inner = int(protocol[args.phase]["inner_folds"])

    rows: list[dict[str, object]] = []
    trait_lookup: dict[tuple[str, int, int, str], pd.DataFrame] = {}
    prediction_lookup: dict[tuple[str, int, int, str], pd.DataFrame] = {}
    for run_dir in sorted(args.models_dir.glob("loss_balance_inner_*")):
        metadata_paths = list(run_dir.glob("*_run_metadata.json"))
        if len(metadata_paths) != 1:
            continue
        metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
        external = metadata.get("external_split", {})
        scenario = str(external.get("scenario", ""))
        outer_fold = int(external.get("outer_fold", -1))
        candidate = str(metadata.get("loss_balance", {}).get("candidate", ""))
        if scenario not in fold_spec or outer_fold not in fold_spec[scenario]:
            continue
        if candidate not in candidates:
            continue
        row, traits, predictions = load_run(run_dir)
        key = (scenario, outer_fold, int(row["inner_fold"]), candidate)
        if key in trait_lookup:
            raise ValueError(f"Duplicate loss-balance run: {key}")
        rows.append(row)
        trait_lookup[key] = traits
        prediction_lookup[key] = predictions
    runs = pd.DataFrame(rows)
    expected_keys = {
        (scenario, int(outer), inner, candidate)
        for scenario, outer_folds in fold_spec.items()
        for outer in outer_folds
        for inner in range(expected_inner)
        for candidate in candidates
    }
    observed_keys = set(trait_lookup)
    if observed_keys != expected_keys:
        raise ValueError(
            "Loss-balance run grid is incomplete: "
            f"missing={sorted(expected_keys-observed_keys)[:20]}; "
            f"extra={sorted(observed_keys-expected_keys)[:20]}"
        )

    paired_rows: list[dict[str, object]] = []
    trait_rows: list[pd.DataFrame] = []
    index = runs.set_index(["scenario", "outer_fold", "inner_fold", "candidate"])
    for scenario, outer_folds in fold_spec.items():
        for outer_fold in outer_folds:
            for inner_fold in range(expected_inner):
                reference_key = (scenario, int(outer_fold), inner_fold, REFERENCE)
                reference = index.loc[reference_key]
                reference_predictions = prediction_lookup[reference_key]
                for candidate in sorted(candidates - {REFERENCE}):
                    key = (scenario, int(outer_fold), inner_fold, candidate)
                    current = index.loc[key]
                    for field in (
                        "seed",
                        "manifest_sha256",
                        "evaluation_protocol_sha256",
                        "loss_protocol_sha256",
                        "training_configuration",
                        "active_kernels",
                        "training_input_identities",
                    ):
                        if current[field] != reference[field]:
                            raise ValueError(f"Matched loss candidates disagree on {field}: {key}")
                    assert_prediction_identity(prediction_lookup[key], reference_predictions)
                    gain = reference["val_normalized_rmse"] - current[
                        "val_normalized_rmse"
                    ]
                    paired_rows.append(
                        {
                            "scenario": scenario,
                            "outer_fold": int(outer_fold),
                            "inner_fold": inner_fold,
                            "candidate": candidate,
                            "seed": int(current["seed"]),
                            "reference_val_normalized_rmse": reference[
                                "val_normalized_rmse"
                            ],
                            "candidate_val_normalized_rmse": current[
                                "val_normalized_rmse"
                            ],
                            "relative_normalized_rmse_gain": gain
                            / reference["val_normalized_rmse"],
                            "pearson_gain": current["val_pearson"]
                            - reference["val_pearson"],
                            "calibration_error_delta": current[
                                "val_calibration_error"
                            ]
                            - reference["val_calibration_error"],
                        }
                    )
                    left = trait_lookup[key][
                        [
                            "trait_name_canonical",
                            "normalized_rmse",
                            "pearson",
                            "prediction_sd_ratio",
                        ]
                    ]
                    right = trait_lookup[reference_key][
                        [
                            "trait_name_canonical",
                            "normalized_rmse",
                            "pearson",
                            "prediction_sd_ratio",
                        ]
                    ]
                    trait_pair = left.merge(
                        right,
                        on="trait_name_canonical",
                        suffixes=("_candidate", "_reference"),
                        validate="one_to_one",
                    )
                    trait_pair.insert(0, "candidate", candidate)
                    trait_pair.insert(0, "inner_fold", inner_fold)
                    trait_pair.insert(0, "outer_fold", int(outer_fold))
                    trait_pair.insert(0, "scenario", scenario)
                    trait_pair["relative_normalized_rmse_gain"] = (
                        trait_pair["normalized_rmse_reference"]
                        - trait_pair["normalized_rmse_candidate"]
                    ) / trait_pair["normalized_rmse_reference"]
                    trait_pair["pearson_gain"] = (
                        trait_pair["pearson_candidate"]
                        - trait_pair["pearson_reference"]
                    )
                    trait_pair["calibration_error_delta"] = (
                        np.abs(trait_pair["prediction_sd_ratio_candidate"] - 1.0)
                        - np.abs(trait_pair["prediction_sd_ratio_reference"] - 1.0)
                    )
                    trait_rows.append(trait_pair)

    paired = pd.DataFrame(paired_rows)
    trait_paired = pd.concat(trait_rows, ignore_index=True)
    scenario_summary = (
        paired.groupby(["candidate", "scenario"], sort=True)
        .agg(
            paired_inner_folds=("inner_fold", "size"),
            relative_normalized_rmse_gain_mean=(
                "relative_normalized_rmse_gain",
                "mean",
            ),
            normalized_rmse_win_rate=(
                "relative_normalized_rmse_gain",
                lambda values: float((values > 0).mean()),
            ),
            pearson_gain_mean=("pearson_gain", "mean"),
            calibration_error_delta_mean=("calibration_error_delta", "mean"),
        )
        .reset_index()
    )
    trait_summary = (
        trait_paired.groupby(["candidate", "trait_name_canonical"], sort=True)
        .agg(
            paired_trait_folds=("relative_normalized_rmse_gain", "size"),
            relative_normalized_rmse_gain_mean=(
                "relative_normalized_rmse_gain",
                "mean",
            ),
            pearson_gain_mean=("pearson_gain", "mean"),
            calibration_error_delta_mean=("calibration_error_delta", "mean"),
        )
        .reset_index()
    )
    acceptance = protocol["acceptance"]
    overall = (
        paired.groupby("candidate", sort=True)
        .agg(
            paired_inner_folds=("inner_fold", "size"),
            relative_normalized_rmse_gain_mean=(
                "relative_normalized_rmse_gain",
                "mean",
            ),
            normalized_rmse_win_rate=(
                "relative_normalized_rmse_gain",
                lambda values: float((values > 0).mean()),
            ),
            pearson_gain_mean=("pearson_gain", "mean"),
            calibration_error_delta_mean=("calibration_error_delta", "mean"),
        )
        .reset_index()
    )
    decisions = []
    for row in overall.itertuples(index=False):
        unseen = scenario_summary[
            scenario_summary["candidate"].eq(row.candidate)
            & scenario_summary["scenario"].eq("unseen_genotypes")
        ]
        primary = trait_summary[
            trait_summary["candidate"].eq(row.candidate)
            & trait_summary["trait_name_canonical"].isin(
                acceptance["primary_guard_traits"]
            )
        ]
        guards = {
            "overall_gain": row.relative_normalized_rmse_gain_mean
            >= float(acceptance["minimum_relative_normalized_rmse_gain"]),
            "fold_win_rate": row.normalized_rmse_win_rate
            >= float(acceptance["minimum_paired_inner_fold_win_rate"]),
            "pearson": row.pearson_gain_mean
            >= -float(acceptance["maximum_mean_pearson_drop"]),
            "calibration": row.calibration_error_delta_mean
            <= float(acceptance["maximum_mean_calibration_error_increase"]),
            "unseen_genotype": not unseen.empty
            and float(unseen.iloc[0]["relative_normalized_rmse_gain_mean"]) > 0,
            "primary_traits": len(primary)
            == len(acceptance["primary_guard_traits"])
            and primary["relative_normalized_rmse_gain_mean"].ge(
                -float(acceptance["maximum_primary_trait_relative_nrmse_loss"])
            ).all(),
        }
        decisions.append(
            {
                "candidate": row.candidate,
                **{f"guard_{name}": bool(value) for name, value in guards.items()},
                "accepted": all(guards.values()),
            }
        )
    decision_frame = overall.merge(pd.DataFrame(decisions), on="candidate")
    accepted = decision_frame[decision_frame["accepted"]]
    selected = (
        accepted.sort_values(
            ["relative_normalized_rmse_gain_mean", "pearson_gain_mean"],
            ascending=[False, False],
        ).iloc[0]["candidate"]
        if not accepted.empty
        else REFERENCE
    )
    decision_frame["decision"] = np.where(
        decision_frame["candidate"].eq(selected) & decision_frame["accepted"],
        "advance_to_full_inner_confirmation"
        if args.phase == "phase_1"
        else "freeze_for_new_v5_outer_protocol",
        "do_not_advance",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "runs": args.out_dir / "loss_balance_inner_screen_runs.tsv",
        "paired": args.out_dir / "loss_balance_inner_screen_paired_metrics.tsv",
        "traits": args.out_dir / "loss_balance_inner_screen_trait_metrics.tsv",
        "scenario": args.out_dir / "loss_balance_inner_screen_scenario_summary.tsv",
        "trait_summary": args.out_dir / "loss_balance_inner_screen_trait_summary.tsv",
        "decision": args.out_dir / "loss_balance_inner_screen_decision.tsv",
    }
    runs.to_csv(artifacts["runs"], sep="\t", index=False)
    paired.to_csv(artifacts["paired"], sep="\t", index=False)
    trait_paired.to_csv(artifacts["traits"], sep="\t", index=False)
    scenario_summary.to_csv(artifacts["scenario"], sep="\t", index=False)
    trait_summary.to_csv(artifacts["trait_summary"], sep="\t", index=False)
    decision_frame.to_csv(artifacts["decision"], sep="\t", index=False)
    provenance = {
        "status": "PASS",
        "protocol_version": "reaction_norm_loss_balance_inner_screen_v1",
        "phase": args.phase,
        "selection_data": "inner_validation_only",
        "inner_validation_phenotype_values_read": True,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "selected_candidate": selected,
        "candidate_count": len(candidates),
        "run_count": len(runs),
        "paired_inner_fold_count": int(
            paired[["scenario", "outer_fold", "inner_fold"]]
            .drop_duplicates()
            .shape[0]
        ),
        "matched_seed_status": "pass",
        "matched_validation_observation_status": "pass",
        "matched_training_configuration_status": "pass",
        "matched_kernel_identity_status": "pass",
        "loss_balance_protocol_sha256": file_sha256(args.loss_balance_protocol),
        "trainer_sha256": file_sha256(args.trainer),
        "acceptance": acceptance,
        "artifacts": {
            name: file_sha256(path) for name, path in artifacts.items()
        },
    }
    provenance_path = args.out_dir / "loss_balance_inner_screen_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, allow_nan=False))
    print("\n=== LOSS-BALANCE DECISION ===")
    print(decision_frame.to_string(index=False))


if __name__ == "__main__":
    main()
