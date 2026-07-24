from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_nested_factorization_provenance import VALID_STATUSES, classify_metadata
from .final_evaluation_contract import file_sha256


def prediction_file(run_dir: Path) -> Path:
    parquet = sorted(run_dir.glob("*_predictions.parquet"))
    tsv = sorted(run_dir.glob("*_predictions.tsv.gz"))
    if parquet:
        return parquet[0] if len(parquet) == 1 else Path()
    return tsv[0] if len(tsv) == 1 else Path()


def read_table(path: Path) -> pd.DataFrame:
    return (
        pd.read_parquet(path)
        if path.suffix == ".parquet"
        else pd.read_csv(path, sep="\t", low_memory=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the locked reaction-norm outer-evaluation grid and provenance."
    )
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--reaction-protocol", type=Path, required=True)
    parser.add_argument("--outer-protocol", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--environment-selection-lock", type=Path, required=True)
    parser.add_argument("--support-policy", type=Path, required=True)
    parser.add_argument("--final-holdout-environments", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--factorization-implementation", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reaction = json.loads(args.reaction_protocol.read_text(encoding="utf-8"))
    outer = json.loads(args.outer_protocol.read_text(encoding="utf-8"))
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    environment_protocol = json.loads(
        args.environment_protocol.read_text(encoding="utf-8")
    )
    environment_lock = json.loads(
        args.environment_selection_lock.read_text(encoding="utf-8")
    )
    support_policy = json.loads(args.support_policy.read_text(encoding="utf-8"))
    required_kernels = set(outer["required_kernels"])
    selected_candidate = str(outer["selected_candidate"])
    selected_model_label = str(outer["selected_model_label"])
    expected_members = int(outer["outer_member_policy"]["member_count"])
    expected_grid = {
        (scenario, fold)
        for scenario, fold_count in outer["scenarios"].items()
        for fold in range(int(fold_count))
    }
    trainer_sha256 = file_sha256(args.trainer)
    factorization_sha256 = file_sha256(args.factorization_implementation)
    outer_sha256 = file_sha256(args.outer_protocol)
    lock_sha256 = file_sha256(args.selection_lock)
    environment_protocol_sha256 = file_sha256(args.environment_protocol)
    environment_lock_sha256 = file_sha256(args.environment_selection_lock)
    support_sha256 = file_sha256(args.support_policy)

    final_holdout = pd.read_csv(args.final_holdout_environments, sep="\t", dtype=str)
    holdout_column = "env_id" if "env_id" in final_holdout else "environment_id"
    final_holdout_ids = set(final_holdout[holdout_column].dropna().astype(str))
    if not final_holdout_ids:
        raise SystemExit("Final-holdout environment manifest is empty")

    checks = {
        "outer_protocol_frozen": outer.get("status")
        == "frozen_after_inner_validation_before_outer_test",
        "selection_lock_pass": lock.get("status") == "PASS",
        "selection_lock_candidate": lock.get("selected_candidate")
        == selected_candidate,
        "selection_lock_protocol": lock.get("outer_evaluation_protocol_sha256")
        == outer_sha256,
        "selection_lock_outer_unread_at_freeze": lock.get("outer_test_metrics_read")
        is False,
        "selection_lock_final_holdout_unread": lock.get(
            "final_holdout_outcomes_read"
        )
        is False,
        "environment_selection_lock_pass": environment_lock.get("status") == "PASS",
        "environment_selection_lock_candidate": environment_lock.get(
            "selected_environment_architecture"
        )
        == outer.get("selected_environment_architecture"),
        "environment_selection_lock_protocol": environment_lock.get(
            "outer_evaluation_protocol_sha256"
        )
        == outer_sha256,
        "environment_selection_lock_outer_unread_at_freeze": environment_lock.get(
            "outer_test_metrics_read"
        )
        is False,
        "environment_selection_lock_final_holdout_unread": environment_lock.get(
            "final_holdout_outcomes_read"
        )
        is False,
        "environment_protocol_identity": outer.get(
            "environment_architecture_protocol_sha256"
        )
        == environment_protocol_sha256,
        "environment_protocol_frozen": environment_protocol.get("status")
        == "frozen_before_inner_validation",
        "inner_protocol_identity": outer.get("inner_reaction_protocol_sha256")
        == file_sha256(args.reaction_protocol),
        "support_policy_identity": outer.get("outer_member_policy", {}).get(
            "support_policy_sha256"
        )
        == support_sha256,
        "support_policy_frozen": support_policy.get("status") == "frozen",
        "no_further_selection": outer.get("model_contract", {}).get(
            "no_further_hyperparameter_selection"
        )
        is True,
        "final_holdout_unavailable": outer.get("model_contract", {}).get(
            "final_holdout_available"
        )
        is False,
    }

    member_rows: list[dict[str, object]] = []
    member_grid: dict[tuple[str, int], set[int]] = {}
    for run_dir in sorted(
        args.models_dir.glob("nested_outer_member_reaction_norm_*_outer*_inner*")
    ):
        metadata_paths = sorted(run_dir.glob("*_run_metadata.json"))
        prediction_path = prediction_file(run_dir)
        if len(metadata_paths) != 1 or not prediction_path.is_file():
            continue
        metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
        external = metadata.get("external_split", {})
        scenario = str(external.get("scenario", ""))
        outer_fold = int(external.get("outer_fold", -1))
        inner_fold = int(external.get("inner_fold", -1))
        provenance_status, provenance_detail = classify_metadata(
            metadata, trainer_sha256, factorization_sha256
        )
        local_checks = {
            "known_grid": (scenario, outer_fold) in expected_grid,
            "inner_fold": 0 <= inner_fold < expected_members,
            "stage": metadata.get("evaluation_stage") == "outer_evaluation",
            "candidate": metadata.get("hyperparameter_label") == selected_candidate,
            "model_label": metadata.get("model_label") == selected_model_label,
            "kernels": set(metadata.get("active_kernels", [])) == required_kernels,
            "outer_protocol": metadata.get("outer_evaluation_protocol", {}).get(
                "sha256"
            )
            == outer_sha256,
            "selection_lock": metadata.get("reaction_selection_lock", {}).get(
                "sha256"
            )
            == lock_sha256,
            "environment_selection_lock": metadata.get(
                "environment_selection_lock", {}
            ).get("sha256")
            == environment_lock_sha256,
            "environment_architecture": metadata.get("environment_architecture")
            == outer.get("selected_environment_architecture"),
            "environment_protocol": metadata.get(
                "environment_architecture_protocol", {}
            ).get("sha256")
            == environment_protocol_sha256,
            "environment_design_certified": metadata.get("environment_design", {}).get(
                "certification_status"
            )
            == "PASS",
            "outer_metrics_read": metadata.get("outer_test_metrics_read") is True,
            "final_holdout_unread": metadata.get("final_holdout_outcomes_read")
            is False,
            "provenance": provenance_status in VALID_STATUSES,
        }
        predictions = read_table(prediction_path)
        split_values = set(predictions["split"].astype(str))
        local_checks["prediction_splits"] = split_values == {"val", "test"}
        local_checks["no_final_holdout_split"] = "final_holdout" not in split_values
        local_checks["no_final_holdout_environment"] = not bool(
            set(predictions["environment_id"].astype(str)) & final_holdout_ids
        )
        local_checks["finite_predictions"] = bool(
            np.isfinite(pd.to_numeric(predictions["y_pred"], errors="coerce")).all()
        )
        failed = sorted(name for name, passed in local_checks.items() if not passed)
        if failed:
            raise SystemExit(f"Outer member failed {failed}: {run_dir}")
        member_grid.setdefault((scenario, outer_fold), set()).add(inner_fold)
        member_rows.append(
            {
                "run_dir": str(run_dir),
                "scenario": scenario,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "seed": int(metadata["seed"]),
                "prediction_rows": len(predictions),
                "test_rows": int(predictions["split"].astype(str).eq("test").sum()),
                "provenance_status": provenance_status,
                "provenance_detail": provenance_detail,
                "metadata_sha256": file_sha256(metadata_paths[0]),
                "prediction_sha256": file_sha256(prediction_path),
            }
        )

    checks["member_grid"] = set(member_grid) == expected_grid and all(
        folds == set(range(expected_members)) for folds in member_grid.values()
    )
    checks["member_count"] = len(member_rows) == len(expected_grid) * expected_members

    ensemble_rows: list[dict[str, object]] = []
    ensemble_grid: set[tuple[str, int]] = set()
    availability_rows: list[dict[str, object]] = []
    for run_dir in sorted(
        args.models_dir.glob("final_nested_reaction_norm_*_outer*")
    ):
        metadata_paths = sorted(run_dir.glob("*_run_metadata.json"))
        prediction_path = prediction_file(run_dir)
        if len(metadata_paths) != 1 or not prediction_path.is_file():
            continue
        metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
        external = metadata.get("external_split", {})
        ensemble = metadata.get("ensemble", {})
        scenario = str(external.get("scenario", ""))
        outer_fold = int(external.get("outer_fold", -1))
        key = (scenario, outer_fold)
        predictions = read_table(prediction_path)
        test = predictions[predictions["split"].astype(str).eq("test")].copy()
        local_checks = {
            "known_grid": key in expected_grid,
            "unique_grid": key not in ensemble_grid,
            "stage": metadata.get("evaluation_stage") == "outer_evaluation",
            "candidate": metadata.get("hyperparameter_label") == selected_candidate,
            "model_label": metadata.get("model_label") == selected_model_label,
            "ensemble_members": int(ensemble.get("member_count", -1))
            == expected_members,
            "outer_test_not_selected": ensemble.get("outer_test_used_for_selection")
            is False,
            "support_policy": ensemble.get("support_policy", {}).get("sha256")
            == support_sha256,
            "prediction_splits": set(predictions["split"].astype(str))
            == {"val", "test"},
            "test_ids_unique": not test["canonical_observation_id"].duplicated().any(),
            "no_final_holdout_environment": not bool(
                set(predictions["environment_id"].astype(str)) & final_holdout_ids
            ),
            "minimum_member_support": bool(
                pd.to_numeric(test["ensemble_member_count"], errors="coerce")
                .ge(int(support_policy["minimum_test_members"]))
                .all()
            ),
        }
        failed = sorted(name for name, passed in local_checks.items() if not passed)
        if failed:
            raise SystemExit(f"Outer ensemble failed {failed}: {run_dir}")
        ensemble_grid.add(key)
        ensemble_rows.append(
            {
                "run_dir": str(run_dir),
                "scenario": scenario,
                "outer_fold": outer_fold,
                "prediction_rows": len(predictions),
                "test_rows": len(test),
                "test_traits": test["trait_name_canonical"].nunique(),
                "partial_member_test_observations": int(
                    ensemble.get("partial_member_test_observations", -1)
                ),
                "metadata_sha256": file_sha256(metadata_paths[0]),
                "prediction_sha256": file_sha256(prediction_path),
            }
        )
        for trait, group in test.groupby("trait_name_canonical", sort=True):
            availability_rows.append(
                {
                    "scenario": scenario,
                    "outer_fold": outer_fold,
                    "trait_name_canonical": trait,
                    "test_rows": len(group),
                    "minimum_ensemble_members": int(group["ensemble_member_count"].min()),
                    "complete_member_fraction": float(
                        group["ensemble_member_count"].eq(expected_members).mean()
                    ),
                }
            )
    checks["ensemble_grid"] = ensemble_grid == expected_grid
    checks["ensemble_count"] = len(ensemble_rows) == len(expected_grid)

    fold_metrics_path = args.summary_dir / "nested_outer_fold_metrics.tsv"
    fold_summary_path = args.summary_dir / "nested_outer_fold_summary.tsv"
    lineage_path = args.summary_dir / "nested_summary_input_provenance.tsv"
    for name, path in (
        ("fold_metrics", fold_metrics_path),
        ("fold_summary", fold_summary_path),
        ("lineage", lineage_path),
    ):
        checks[f"summary_{name}"] = path.is_file() and path.stat().st_size > 0
    if fold_metrics_path.is_file():
        fold_metrics = pd.read_csv(fold_metrics_path, sep="\t")
        observed_metric_grid = set(
            zip(
                fold_metrics["scenario"].astype(str),
                pd.to_numeric(fold_metrics["outer_fold"]).astype(int),
            )
        )
        checks["summary_grid"] = observed_metric_grid == expected_grid
        checks["summary_no_duplicate_trait_keys"] = not fold_metrics.duplicated(
            ["scenario", "outer_fold", "trait_name_canonical"]
        ).any()
        core = fold_metrics[
            ["test_rmse", "test_pearson", "train_mean_improvement"]
        ].apply(pd.to_numeric, errors="coerce")
        checks["summary_core_metrics_finite"] = bool(np.isfinite(core).all().all())

    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SystemExit(f"Reaction-norm outer-evaluation verification failed: {failed}")

    pd.DataFrame(member_rows).to_csv(
        args.out_dir / "reaction_norm_outer_member_inventory.tsv", sep="\t", index=False
    )
    pd.DataFrame(ensemble_rows).to_csv(
        args.out_dir / "reaction_norm_outer_ensemble_inventory.tsv", sep="\t", index=False
    )
    pd.DataFrame(availability_rows).to_csv(
        args.out_dir / "reaction_norm_outer_trait_availability.tsv", sep="\t", index=False
    )
    provenance = {
        "status": "PASS",
        "protocol_version": outer["protocol_version"],
        "selection_data": "frozen_inner_validation_decision",
        "further_hyperparameter_selection_performed": False,
        "outer_test_metrics_read_for_locked_reporting": True,
        "outer_test_metrics_used_for_selection": False,
        "final_holdout_outcomes_read": False,
        "selected_candidate": selected_candidate,
        "scenario_count": len(outer["scenarios"]),
        "outer_fold_count": len(expected_grid),
        "outer_member_count": len(member_rows),
        "outer_ensemble_count": len(ensemble_rows),
        "required_kernels": sorted(required_kernels),
        "trainer_sha256": trainer_sha256,
        "factorization_sha256": factorization_sha256,
        "outer_protocol_sha256": outer_sha256,
        "selection_lock_sha256": lock_sha256,
        "environment_protocol_sha256": environment_protocol_sha256,
        "environment_selection_lock_sha256": environment_lock_sha256,
        "checks": checks,
    }
    (args.out_dir / "reaction_norm_outer_evaluation_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
