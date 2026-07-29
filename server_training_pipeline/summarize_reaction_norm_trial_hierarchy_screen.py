from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .final_evaluation_contract import file_sha256
from .summarize_reaction_norm_loss_balance_screen import (
    assert_prediction_identity,
    unique_file,
)
from .train_multitrait_multikernel_tf import regression_metrics


REFERENCE = "current_reaction_norm"


def scenario_noninferiority_pass(
    scenario_summary: pd.DataFrame,
    candidate: str,
    acceptance: dict[str, object],
) -> bool:
    required = set(map(str, acceptance.get("required_scenarios", [])))
    if not required:
        return True
    selected = scenario_summary[
        scenario_summary["candidate"].eq(candidate)
        & scenario_summary["scenario"].isin(required)
    ]
    return bool(
        set(selected["scenario"]) == required
        and selected["relative_normalized_rmse_gain_mean"].ge(
            -float(acceptance["maximum_scenario_relative_nrmse_loss"])
        ).all()
        and selected["normalized_rmse_win_rate"].ge(
            float(acceptance["minimum_scenario_fold_win_rate"])
        ).all()
        and selected["pearson_gain_mean"].ge(
            -float(acceptance["maximum_scenario_pearson_drop"])
        ).all()
        and selected["calibration_error_delta_mean"].le(
            float(acceptance["maximum_scenario_calibration_error_increase"])
        ).all()
    )


def load_run(run_dir: Path) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    metadata = json.loads(
        unique_file(run_dir, "*_run_metadata.json").read_text(encoding="utf-8")
    )
    macro = pd.read_csv(unique_file(run_dir, "*_macro_metrics.tsv"), sep="\t")
    traits = pd.read_csv(unique_file(run_dir, "*_trait_metrics.tsv"), sep="\t")
    predictions = pd.read_parquet(unique_file(run_dir, "*_predictions.parquet"))
    if metadata.get("status") != "PASS" or metadata.get("evaluation_stage") != "inner_selection":
        raise ValueError(f"Hierarchy run is not a PASS inner-selection run: {run_dir}")
    if metadata.get("outer_test_metrics_read") is not False:
        raise ValueError(f"Hierarchy run read outer-test metrics: {run_dir}")
    if metadata.get("final_holdout_outcomes_read") is not False:
        raise ValueError(f"Hierarchy run read final-holdout outcomes: {run_dir}")
    if not predictions["split"].astype(str).eq("val").all():
        raise ValueError(f"Hierarchy predictions expose non-validation rows: {run_dir}")
    label = str(metadata["model_label"])
    model_macro = macro[
        macro["split"].astype(str).eq("val") & macro["model"].astype(str).eq(label)
    ]
    model_traits = traits[
        traits["split"].astype(str).eq("val")
        & traits["model"].astype(str).eq(label)
        & traits["coverage_group"].astype(str).eq("all")
    ].copy()
    if len(model_macro) != 1 or model_traits.empty:
        raise ValueError(f"Hierarchy validation metrics are incomplete: {run_dir}")
    core = model_traits[["normalized_rmse", "pearson", "prediction_sd_ratio"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(core.to_numpy(float)).all():
        raise ValueError(f"Hierarchy validation metrics are non-finite: {run_dir}")
    external = metadata["external_split"]
    hierarchy = metadata.get("trial_hierarchy", {})
    row = {
        "run_dir": str(run_dir.resolve()),
        "scenario": str(external["scenario"]),
        "outer_fold": int(external["outer_fold"]),
        "inner_fold": int(external["inner_fold"]),
        "candidate": str(hierarchy.get("candidate", "")),
        "seed": int(metadata["seed"]),
        "manifest_sha256": str(external.get("manifest_sha256", "")),
        "evaluation_protocol_sha256": str(
            metadata.get("evaluation_protocol", {}).get("protocol_sha256", "")
        ),
        "hierarchy_protocol_sha256": str(hierarchy.get("protocol_sha256", "")),
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
    return row, model_traits, predictions


def subset_macro(frame: pd.DataFrame, prediction: str, minimum_rows: int = 20) -> dict[str, float]:
    rows = []
    for _, group in frame.groupby("trait_name_canonical", sort=True):
        if len(group) < minimum_rows:
            continue
        rows.append(
            regression_metrics(
                group["phenotype_value"].to_numpy(float),
                group[prediction].to_numpy(float),
                np.ones(len(group), dtype=float),
            )
        )
    if not rows:
        return {"normalized_rmse": float("nan"), "pearson": float("nan")}
    return {
        "normalized_rmse": float(np.mean([row["normalized_rmse"] for row in rows])),
        "pearson": float(np.mean([row["pearson"] for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize matched inner-only trial-hierarchy candidates."
    )
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--hierarchy-protocol", type=Path, required=True)
    parser.add_argument("--readiness-ledger", type=Path, required=True)
    parser.add_argument("--phase", choices=["phase_1", "confirmation"], required=True)
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.hierarchy_protocol.read_text(encoding="utf-8"))
    policy_names = {str(value["name"]) for value in protocol["candidates"]}
    candidates = set(args.candidate or policy_names)
    candidates.add(REFERENCE)
    if not candidates.issubset(policy_names):
        raise ValueError(f"Unknown hierarchy candidates: {sorted(candidates-policy_names)}")
    fold_spec = protocol[args.phase]["outer_folds_by_scenario"]
    expected_inner = int(protocol[args.phase]["inner_folds"])
    readiness = pd.read_parquet(
        args.readiness_ledger,
        columns=["canonical_observation_id", "recovery_readiness"],
    )
    if readiness["canonical_observation_id"].duplicated().any():
        raise ValueError("Recovery-readiness IDs are duplicated")

    rows = []
    trait_lookup = {}
    prediction_lookup = {}
    for run_dir in sorted(args.models_dir.glob("trial_hierarchy_inner_*")):
        metadata_paths = list(run_dir.glob("*_run_metadata.json"))
        if len(metadata_paths) != 1:
            continue
        metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
        external = metadata.get("external_split", {})
        scenario = str(external.get("scenario", ""))
        outer = int(external.get("outer_fold", -1))
        candidate = str(metadata.get("trial_hierarchy", {}).get("candidate", ""))
        if scenario not in fold_spec or outer not in fold_spec[scenario] or candidate not in candidates:
            continue
        row, traits, predictions = load_run(run_dir)
        key = (scenario, outer, int(row["inner_fold"]), candidate)
        if key in trait_lookup:
            raise ValueError(f"Duplicate hierarchy run: {key}")
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
    if set(trait_lookup) != expected_keys:
        raise ValueError(
            "Hierarchy run grid is incomplete: "
            f"missing={sorted(expected_keys-set(trait_lookup))[:20]}; "
            f"extra={sorted(set(trait_lookup)-expected_keys)[:20]}"
        )

    index = runs.set_index(["scenario", "outer_fold", "inner_fold", "candidate"])
    paired_rows = []
    trait_rows = []
    subset_rows = []
    for scenario, outer_folds in fold_spec.items():
        for outer in outer_folds:
            for inner in range(expected_inner):
                reference_key = (scenario, int(outer), inner, REFERENCE)
                reference = index.loc[reference_key]
                reference_predictions = prediction_lookup[reference_key]
                for candidate in sorted(candidates - {REFERENCE}):
                    key = (scenario, int(outer), inner, candidate)
                    current = index.loc[key]
                    for field in (
                        "seed",
                        "manifest_sha256",
                        "evaluation_protocol_sha256",
                        "hierarchy_protocol_sha256",
                        "training_configuration",
                        "active_kernels",
                        "training_input_identities",
                    ):
                        if current[field] != reference[field]:
                            raise ValueError(f"Matched hierarchy candidates disagree on {field}: {key}")
                    current_predictions = prediction_lookup[key]
                    assert_prediction_identity(current_predictions, reference_predictions)
                    gain = reference["val_normalized_rmse"] - current["val_normalized_rmse"]
                    paired_rows.append(
                        {
                            "scenario": scenario,
                            "outer_fold": int(outer),
                            "inner_fold": inner,
                            "candidate": candidate,
                            "seed": int(current["seed"]),
                            "reference_val_normalized_rmse": reference["val_normalized_rmse"],
                            "candidate_val_normalized_rmse": current["val_normalized_rmse"],
                            "relative_normalized_rmse_gain": gain / reference["val_normalized_rmse"],
                            "pearson_gain": current["val_pearson"] - reference["val_pearson"],
                            "calibration_error_delta": current["val_calibration_error"]
                            - reference["val_calibration_error"],
                        }
                    )
                    left = trait_lookup[key][
                        ["trait_name_canonical", "normalized_rmse", "pearson", "prediction_sd_ratio"]
                    ]
                    right = trait_lookup[reference_key][
                        ["trait_name_canonical", "normalized_rmse", "pearson", "prediction_sd_ratio"]
                    ]
                    trait_pair = left.merge(
                        right,
                        on="trait_name_canonical",
                        suffixes=("_candidate", "_reference"),
                        validate="one_to_one",
                    )
                    trait_pair.insert(0, "candidate", candidate)
                    trait_pair.insert(0, "inner_fold", inner)
                    trait_pair.insert(0, "outer_fold", int(outer))
                    trait_pair.insert(0, "scenario", scenario)
                    trait_pair["relative_normalized_rmse_gain"] = (
                        trait_pair["normalized_rmse_reference"]
                        - trait_pair["normalized_rmse_candidate"]
                    ) / trait_pair["normalized_rmse_reference"]
                    trait_pair["pearson_gain"] = (
                        trait_pair["pearson_candidate"] - trait_pair["pearson_reference"]
                    )
                    trait_rows.append(trait_pair)

                    joined = current_predictions[
                        ["canonical_observation_id", "trait_name_canonical", "phenotype_value", "y_pred"]
                    ].merge(
                        reference_predictions[["canonical_observation_id", "y_pred"]].rename(
                            columns={"y_pred": "y_pred_reference"}
                        ),
                        on="canonical_observation_id",
                        validate="one_to_one",
                    ).merge(
                        readiness,
                        on="canonical_observation_id",
                        how="left",
                        validate="one_to_one",
                    )
                    if joined["recovery_readiness"].isna().any():
                        raise ValueError("Hierarchy validation rows lack recovery provenance")
                    joined["recovery_subset"] = np.where(
                        joined["recovery_readiness"].eq("RETAINED_REFERENCE"),
                        "retained_reference",
                        "recovered",
                    )
                    for subset, group in joined.groupby("recovery_subset", sort=True):
                        candidate_metric = subset_macro(group, "y_pred")
                        reference_metric = subset_macro(group, "y_pred_reference")
                        if not np.isfinite(candidate_metric["normalized_rmse"]):
                            continue
                        subset_rows.append(
                            {
                                "scenario": scenario,
                                "outer_fold": int(outer),
                                "inner_fold": inner,
                                "candidate": candidate,
                                "recovery_subset": subset,
                                "rows": len(group),
                                "relative_normalized_rmse_gain": (
                                    reference_metric["normalized_rmse"]
                                    - candidate_metric["normalized_rmse"]
                                )
                                / reference_metric["normalized_rmse"],
                                "pearson_gain": candidate_metric["pearson"]
                                - reference_metric["pearson"],
                            }
                        )

    paired = pd.DataFrame(paired_rows)
    trait_paired = pd.concat(trait_rows, ignore_index=True)
    subsets = pd.DataFrame(subset_rows)
    summary = (
        paired.groupby("candidate", sort=True)
        .agg(
            paired_inner_folds=("inner_fold", "size"),
            relative_normalized_rmse_gain_mean=("relative_normalized_rmse_gain", "mean"),
            normalized_rmse_win_rate=(
                "relative_normalized_rmse_gain",
                lambda values: float((values > 0).mean()),
            ),
            pearson_gain_mean=("pearson_gain", "mean"),
            calibration_error_delta_mean=("calibration_error_delta", "mean"),
        )
        .reset_index()
    )
    scenario_summary = (
        paired.groupby(["candidate", "scenario"], sort=True)
        .agg(
            paired_scenario_folds=("inner_fold", "size"),
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
            relative_normalized_rmse_gain_mean=("relative_normalized_rmse_gain", "mean"),
            pearson_gain_mean=("pearson_gain", "mean"),
        )
        .reset_index()
    )
    subset_summary = (
        subsets.groupby(["candidate", "recovery_subset"], sort=True)
        .agg(
            paired_subset_folds=("relative_normalized_rmse_gain", "size"),
            rows_mean=("rows", "mean"),
            relative_normalized_rmse_gain_mean=("relative_normalized_rmse_gain", "mean"),
            pearson_gain_mean=("pearson_gain", "mean"),
        )
        .reset_index()
        if not subsets.empty
        else pd.DataFrame()
    )
    acceptance = protocol["acceptance"]
    decisions = []
    for row in summary.itertuples(index=False):
        primary = trait_summary[
            trait_summary["candidate"].eq(row.candidate)
            & trait_summary["trait_name_canonical"].isin(acceptance["primary_guard_traits"])
        ]
        retained = subset_summary[
            subset_summary["candidate"].eq(row.candidate)
            & subset_summary["recovery_subset"].eq("retained_reference")
        ]
        required_scenarios = set(acceptance.get("required_scenarios", []))
        guards = {
            "overall_gain": row.relative_normalized_rmse_gain_mean
            >= float(acceptance["minimum_relative_normalized_rmse_gain"]),
            "fold_win_rate": row.normalized_rmse_win_rate
            >= float(acceptance["minimum_paired_inner_fold_win_rate"]),
            "pearson": row.pearson_gain_mean >= -float(acceptance["maximum_mean_pearson_drop"]),
            "calibration": row.calibration_error_delta_mean
            <= float(acceptance["maximum_mean_calibration_error_increase"]),
            "retained_reference": len(retained) == 1
            and float(retained.iloc[0]["relative_normalized_rmse_gain_mean"])
            >= -float(acceptance["maximum_retained_reference_relative_nrmse_loss"]),
            "primary_traits": len(primary) == len(acceptance["primary_guard_traits"])
            and primary["relative_normalized_rmse_gain_mean"].ge(
                -float(acceptance["maximum_primary_trait_relative_nrmse_loss"])
            ).all(),
        }
        if required_scenarios:
            guards["scenario_noninferiority"] = scenario_noninferiority_pass(
                scenario_summary, str(row.candidate), acceptance
            )
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
            ["relative_normalized_rmse_gain_mean", "pearson_gain_mean"],
            ascending=[False, False],
        ).iloc[0]["candidate"]
        if not accepted.empty
        else REFERENCE
    )
    decision["decision"] = np.where(
        decision["candidate"].eq(selected) & decision["accepted"],
        "advance_to_full_inner_confirmation"
        if args.phase == "phase_1"
        else "freeze_for_new_outer_protocol",
        "do_not_advance",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "runs": args.out_dir / "trial_hierarchy_inner_screen_runs.tsv",
        "paired": args.out_dir / "trial_hierarchy_inner_screen_paired_metrics.tsv",
        "traits": args.out_dir / "trial_hierarchy_inner_screen_trait_metrics.tsv",
        "subsets": args.out_dir / "trial_hierarchy_inner_screen_subset_metrics.tsv",
        "trait_summary": args.out_dir / "trial_hierarchy_inner_screen_trait_summary.tsv",
        "subset_summary": args.out_dir / "trial_hierarchy_inner_screen_subset_summary.tsv",
        "scenario_summary": args.out_dir
        / "trial_hierarchy_inner_screen_scenario_summary.tsv",
        "decision": args.out_dir / "trial_hierarchy_inner_screen_decision.tsv",
    }
    for frame, key in (
        (runs, "runs"),
        (paired, "paired"),
        (trait_paired, "traits"),
        (subsets, "subsets"),
        (trait_summary, "trait_summary"),
        (subset_summary, "subset_summary"),
        (scenario_summary, "scenario_summary"),
        (decision, "decision"),
    ):
        frame.to_csv(artifacts[key], sep="\t", index=False)
    provenance = {
        "status": "PASS",
        "protocol_version": "reaction_norm_trial_hierarchy_inner_screen_v1",
        "phase": args.phase,
        "selection_data": "inner_validation_only",
        "inner_validation_phenotype_values_read": True,
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
        "hierarchy_protocol_sha256": file_sha256(args.hierarchy_protocol),
        "trainer_sha256": file_sha256(args.trainer),
        "readiness_ledger_sha256": file_sha256(args.readiness_ledger),
        "acceptance": acceptance,
        "artifacts": {name: file_sha256(path) for name, path in artifacts.items()},
    }
    provenance_path = args.out_dir / "trial_hierarchy_inner_screen_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, allow_nan=False))
    print("\n=== TRIAL-HIERARCHY DECISION ===")
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
