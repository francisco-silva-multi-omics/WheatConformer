from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL = Path(
    "server_training_pipeline/"
    "stage1_v2_phase6_hierarchy_calibration_adjudication_correction_protocol_v1.json"
)
SOURCE = Path("model_kernels/stage1_v2_phase6_hierarchy_calibration_amendment_v2")
OUTPUT = Path(
    "audit/v2/"
    "stage1_v2_phase6_hierarchy_calibration_adjudication_correction_v1"
)
REFERENCE = "historical_reaction_reference"
INFORMATION_SUBSETS = {
    "PEDIGREE_ONLY",
    "MARKER_SUPPORTED",
    "PEDIGREE_AND_MARKER",
    "NEITHER_PRODUCTION_PEDIGREE_NOR_DENSE_MARKERS",
    "RECOVERED_IDENTITY_OR_COMPONENT",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean())


def _minimum(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.min()) if numeric.notna().any() else float("nan")


def corrected_summary(
    protocol: dict[str, Any],
    paired: pd.DataFrame,
    paired_traits: pd.DataFrame,
    paired_guards: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    acceptance = protocol["acceptance"]
    primary = set(protocol["primary_traits"])
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for (scenario, candidate), local in paired.groupby(
        ["scenario", "candidate"], sort=False
    ):
        local_traits = paired_traits.loc[
            paired_traits["scenario"].eq(scenario)
            & paired_traits["candidate"].eq(candidate)
        ].copy()
        local_guards = paired_guards.loc[
            paired_guards["scenario"].eq(scenario)
            & paired_guards["candidate"].eq(candidate)
            & paired_guards["rows"].ge(int(acceptance["minimum_rows_for_guard"]))
        ].copy()
        primary_traits = local_traits.loc[
            local_traits["trait_name_canonical"].isin(primary)
        ]
        information = local_guards.loc[
            local_guards["subset"].isin(INFORMATION_SUBSETS)
        ]
        projection_inactive = local_guards.loc[
            local_guards["subset"].eq(
                "PROJECTION_CORE_INACTIVE_814_ENVIRONMENTS"
            )
        ]
        worst_index = pd.to_numeric(
            local_traits["calibration_error"], errors="coerce"
        ).idxmax()
        worst = local_traits.loc[worst_index]
        macro_max = float(
            pd.to_numeric(
                local["validation_macro_calibration_error"], errors="coerce"
            ).max()
        )
        primary_calibration = float(
            pd.to_numeric(
                primary_traits["calibration_error"], errors="coerce"
            ).max()
        )
        negative_slopes = int(
            pd.to_numeric(local_traits["calibration_slope"], errors="coerce").lt(0).sum()
        )
        gain = _mean(local["relative_nrmse_gain"])
        win_rate = _mean(local["nrmse_win"])
        pearson_gain = _mean(local["pearson_gain"])
        spearman_gain = _mean(local["centered_spearman_gain"])
        pairwise_gain = _mean(local["pairwise_accuracy_gain"])
        primary_gain = _minimum(primary_traits["relative_nrmse_gain"])
        information_gain = _minimum(information["relative_nrmse_gain"])
        inactive_gain = _mean(projection_inactive["relative_nrmse_gain"])
        guards = {
            "overall_gain": gain >= float(acceptance["minimum_relative_nrmse_gain"]),
            "fold_win_rate": win_rate
            >= float(acceptance["minimum_paired_inner_fold_win_rate"]),
            "pearson": pearson_gain >= -float(acceptance["maximum_mean_pearson_drop"]),
            "macro_calibration": macro_max
            <= float(acceptance["maximum_absolute_macro_calibration_error"]),
            "primary_calibration": primary_calibration
            <= float(acceptance["maximum_primary_trait_absolute_calibration_error"]),
            "negative_slopes": negative_slopes == 0,
            "centered_spearman": spearman_gain
            >= -float(
                acceptance["within_environment_centered_spearman_maximum_drop"]
            ),
            "pairwise_accuracy": pairwise_gain
            >= -float(
                acceptance["within_environment_pairwise_accuracy_maximum_drop"]
            ),
            "primary_traits": np.isnan(primary_gain)
            or primary_gain
            >= -float(acceptance["primary_trait_maximum_relative_nrmse_loss"]),
            "information_subsets": np.isnan(information_gain)
            or information_gain
            >= -float(acceptance["information_class_maximum_relative_nrmse_loss"]),
            "projection_inactive": np.isnan(inactive_gain)
            or inactive_gain
            >= -float(
                acceptance[
                    "projection_inactive_environment_maximum_relative_nrmse_loss"
                ]
            ),
        }
        eligible = candidate == REFERENCE or all(guards.values())
        rows.append(
            {
                "scenario": scenario,
                "candidate": candidate,
                "paired_inner_folds": int(local["state_id"].nunique()),
                "validation_normalized_rmse_mean": _mean(
                    local["validation_macro_normalized_rmse"]
                ),
                "validation_pearson_mean": _mean(local["validation_macro_pearson"]),
                "relative_normalized_rmse_gain_mean": gain,
                "normalized_rmse_win_rate": win_rate,
                "pearson_gain_mean": pearson_gain,
                "centered_spearman_gain_mean": spearman_gain,
                "pairwise_accuracy_gain_mean": pairwise_gain,
                "absolute_macro_calibration_error_mean": _mean(
                    local["validation_macro_calibration_error"]
                ),
                "absolute_macro_calibration_error_max": macro_max,
                "worst_trait_fold_calibration_error_max": float(
                    worst["calibration_error"]
                ),
                "primary_trait_calibration_error_max": primary_calibration,
                "negative_trait_calibration_slopes": negative_slopes,
                "primary_trait_relative_nrmse_gain_min": primary_gain,
                "information_subset_relative_nrmse_gain_min": information_gain,
                "projection_inactive_relative_nrmse_gain_mean": inactive_gain,
                **{f"guard_{name}": bool(value) for name, value in guards.items()},
                "eligible_for_route_freeze": bool(eligible),
            }
        )
        diagnostics.append(
            {
                "scenario": scenario,
                "candidate": candidate,
                "state_id": worst["state_id"],
                "trait_name_canonical": worst["trait_name_canonical"],
                "rows": int(worst["rows"]),
                "normalized_rmse": float(worst["normalized_rmse"]),
                "pearson": float(worst["pearson"]),
                "calibration_slope": float(worst["calibration_slope"]),
                "calibration_error": float(worst["calibration_error"]),
                "selection_role": "diagnostic_only",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def select_candidate(decision: pd.DataFrame, protocol: dict[str, Any]) -> str | None:
    candidates = decision.loc[
        decision["candidate"].isin(protocol["candidate_order"])
        & decision["eligible_for_route_freeze"].eq(True)
    ].copy()
    if candidates.empty:
        return None
    order = {name: index for index, name in enumerate(protocol["candidate_order"])}
    candidates["candidate_order"] = candidates["candidate"].map(order)
    candidates = candidates.sort_values(
        [
            "validation_normalized_rmse_mean",
            "validation_pearson_mean",
            "candidate_order",
        ],
        ascending=[True, False, True],
    )
    return str(candidates.iloc[0]["candidate"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correct the Stage-1 v2 hierarchy macro-calibration adjudication"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--code-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = (args.code_root or root).resolve()
    protocol_path = code_root / PROTOCOL
    source = root / SOURCE
    required = {
        "source_decision": source / "CALIBRATION_AMENDMENT_DECISION.json",
        "source_decision_table": source / "calibration_amendment_decision.tsv",
        "runs": source / "calibration_amendment_runs.tsv",
        "paired": source / "calibration_amendment_paired_metrics.tsv",
        "traits": source / "calibration_amendment_paired_trait_metrics.tsv",
        "guards": source / "calibration_amendment_paired_guard_metrics.tsv",
        "protocol": protocol_path,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Adjudication correction inputs are missing: {missing}")
    protocol = read_json(protocol_path)
    source_decision = read_json(required["source_decision"])
    source_table = pd.read_csv(required["source_decision_table"], sep="\t")
    runs = pd.read_csv(required["runs"], sep="\t")
    paired = pd.read_csv(required["paired"], sep="\t")
    traits = pd.read_csv(required["traits"], sep="\t")
    guards = pd.read_csv(required["guards"], sep="\t")

    corrected, diagnostics = corrected_summary(protocol, paired, traits, guards)
    selected = select_candidate(corrected, protocol)
    corrected["decision"] = np.where(
        corrected["candidate"].eq(REFERENCE),
        "stable_reference",
        np.where(
            corrected["candidate"].eq(selected),
            "freeze_for_trait_balance_reference_and_125_route_lock",
            np.where(
                corrected["eligible_for_route_freeze"],
                "eligible_not_selected",
                "do_not_advance",
            ),
        ),
    )

    trait_macro = (
        traits.groupby(["state_id", "candidate"], as_index=False)["calibration_error"]
        .mean()
        .rename(columns={"calibration_error": "trait_mean_calibration_error"})
    )
    trace = paired[["state_id", "candidate", "validation_macro_calibration_error"]].merge(
        trait_macro, on=["state_id", "candidate"], how="left", validate="one_to_one"
    )
    trace["absolute_delta"] = (
        trace["validation_macro_calibration_error"]
        - trace["trait_mean_calibration_error"]
    ).abs()
    source_artifacts = source_decision.get("artifact_sha256", {})
    artifact_checks = {
        name: sha256_file(source / name) == expected
        for name, expected in source_artifacts.items()
    }
    source_macro = source_table.set_index("candidate")[
        "absolute_macro_calibration_error_max"
    ]
    corrected_macro = corrected.set_index("candidate")[
        "absolute_macro_calibration_error_max"
    ]
    diagnostic_max = diagnostics.set_index("candidate")["calibration_error"]
    checks = {
        "protocol_identity": protocol.get("protocol_version")
        == "stage1_v2_phase6_hierarchy_calibration_adjudication_correction_v1",
        "source_terminal_no_advance_preserved": source_decision.get("status")
        == protocol["source_terminal_status"]
        and source_decision.get("selected_candidate") is None,
        "source_integrity_checks_pass": not source_decision.get("failed_checks")
        and all(bool(value) for value in source_decision.get("checks", {}).values()),
        "source_artifacts_match": bool(artifact_checks) and all(artifact_checks.values()),
        "outer_and_final_unread": runs["outer_test_metrics_read"].eq(False).all()
        and runs["outer_test_outcomes_read"].eq(False).all()
        and runs["final_holdout_outcomes_read"].eq(False).all(),
        "macro_metric_reconstructs_from_trait_mean": float(trace["absolute_delta"].max())
        <= 1e-12,
        "source_field_reproduces_worst_trait_fold": all(
            abs(float(source_macro.loc[name]) - float(diagnostic_max.loc[name])) <= 1e-12
            for name in source_macro.index
        ),
        "source_field_differs_from_frozen_macro_semantics": any(
            abs(float(source_macro.loc[name]) - float(corrected_macro.loc[name])) > 1e-6
            for name in source_macro.index
        ),
        "candidate_state_grid_complete": len(paired) == 100
        and paired["state_id"].nunique() == 25,
        "selected_candidate_exists": selected in protocol["candidate_order"],
        "selected_candidate_all_guards_pass": selected is not None
        and bool(
            corrected.loc[
                corrected["candidate"].eq(selected)
            ].filter(regex=r"^guard_").iloc[0].astype(bool).all()
        ),
        "no_new_training_or_predictions": protocol["new_model_training_allowed"] is False
        and protocol["new_prediction_generation_allowed"] is False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(
        output / "corrected_calibration_decision.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    diagnostics.to_csv(
        output / "worst_trait_fold_calibration_diagnostic.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    trace.to_csv(
        output / "macro_calibration_semantic_trace.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(
        [{"check": name, "status": "PASS" if passed else "FAIL"} for name, passed in checks.items()]
    ).to_csv(
        output / "validation_checks.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    incident = {
        "status": "CONFIRMED_IMPLEMENTATION_ADJUDICATION_INCIDENT",
        "source_release_preserved": True,
        "source_status": source_decision["status"],
        "source_decision_sha256": sha256_file(required["source_decision"]),
        "source_decision_table_sha256": sha256_file(required["source_decision_table"]),
        "incident": "worst_trait_fold_error_was_labeled_and_gated_as_macro_calibration",
        "threshold_changed": False,
        "model_training_repeated": False,
        "outer_or_final_outcomes_read": False,
    }
    write_json(output / "ADJUDICATION_IMPLEMENTATION_INCIDENT.json", incident)
    artifacts = [
        "corrected_calibration_decision.tsv",
        "worst_trait_fold_calibration_diagnostic.tsv",
        "macro_calibration_semantic_trace.tsv",
        "validation_checks.tsv",
        "ADJUDICATION_IMPLEMENTATION_INCIDENT.json",
    ]
    result = {
        "status": (
            "PASS_STAGE1_V2_PHASE6_HIERARCHY_CALIBRATION_ADJUDICATION_CORRECTED"
            if not failed
            else "FAIL_STAGE1_V2_PHASE6_HIERARCHY_CALIBRATION_ADJUDICATION_CORRECTION"
        ),
        "protocol_version": protocol["protocol_version"],
        "selection_data": "previously_frozen_nested_inner_validation_metrics_only",
        "source_release_preserved": True,
        "selected_candidate": selected,
        "route_freeze_allowed": selected is not None and not failed,
        "trait_balance_screen_allowed": selected is not None and not failed,
        "outer_evaluation_allowed": False,
        "new_model_fit_count": 0,
        "new_prediction_count": 0,
        "outer_test_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "checks": checks,
        "failed_checks": failed,
        "source_artifact_checks": artifact_checks,
        "artifacts": {name: sha256_file(output / name) for name in artifacts},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "CALIBRATION_ADJUDICATION_CORRECTION.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"Calibration adjudication correction failed: {failed}")


if __name__ == "__main__":
    main()
