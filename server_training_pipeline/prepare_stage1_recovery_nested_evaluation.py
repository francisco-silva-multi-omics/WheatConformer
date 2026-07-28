from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .final_evaluation_contract import file_sha256, load_protocol


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def checksum_manifest(root: Path, paths: list[Path], destination: Path) -> None:
    destination.write_text(
        "\n".join(
            f"{file_sha256(path)}  {relative(root, path)}" for path in paths
        )
        + "\n",
        encoding="utf-8",
    )


def read_identifier_columns(path: Path) -> pd.DataFrame:
    columns = [
        "canonical_observation_id",
        "panel_sample_id",
        "genotype_id",
        "env_kernel_id",
        "cycle",
        "country",
        "trait_name_canonical",
    ]
    if "".join(path.suffixes).lower().endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, sep="\t", usecols=columns, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a new nested-evaluation contract around phenotype-blind Stage-1 "
            "row recovery while preserving the selected reaction-norm architecture."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--recovery-ledger", type=Path, required=True)
    parser.add_argument("--base-ledger", type=Path, required=True)
    parser.add_argument("--recovery-validation", type=Path, required=True)
    parser.add_argument("--base-evaluation-protocol", type=Path, required=True)
    parser.add_argument("--base-evaluation-contract", type=Path, required=True)
    parser.add_argument("--base-outer-protocol", type=Path, required=True)
    parser.add_argument("--base-selection-lock", type=Path, required=True)
    parser.add_argument("--base-environment-selection-lock", type=Path, required=True)
    parser.add_argument("--frozen-final-holdout-environments", type=Path, required=True)
    parser.add_argument(
        "--trait-environment-extension-implementation",
        type=Path,
        default=Path(__file__).with_name("extend_trait_environment_kernel.py"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    paths = {
        name: (value if value.is_absolute() else root / value).resolve()
        for name, value in {
            "recovery_ledger": args.recovery_ledger,
            "base_ledger": args.base_ledger,
            "recovery_validation": args.recovery_validation,
            "base_evaluation_protocol": args.base_evaluation_protocol,
            "base_evaluation_contract": args.base_evaluation_contract,
            "base_outer_protocol": args.base_outer_protocol,
            "base_selection_lock": args.base_selection_lock,
            "base_environment_selection_lock": args.base_environment_selection_lock,
            "frozen_final_holdout_environments": args.frozen_final_holdout_environments,
            "trait_environment_extension_implementation": (
                args.trait_environment_extension_implementation
            ),
        }.items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    out_dir = (
        args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "evaluation_protocol": out_dir
        / "stage1_recovery_nested_evaluation_protocol.json",
        "outer_protocol": out_dir
        / "stage1_recovery_reaction_norm_outer_protocol.json",
        "selection_lock": out_dir / "reaction_norm_selection_lock.json",
        "environment_selection_lock": out_dir
        / "reaction_norm_environment_selection_lock.json",
        "freeze": out_dir / "stage1_recovery_nested_freeze.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.force:
        raise SystemExit(
            "Immutable Stage-1 recovery nested artifacts already exist; refusing to "
            f"overwrite: {[str(path) for path in existing]}"
        )

    recovery = json.loads(paths["recovery_validation"].read_text(encoding="utf-8"))
    if recovery.get("status") != "PASS":
        raise ValueError("Stage-1 weight recovery validation did not pass")
    recovery_checks = recovery.get("checks", {})
    required_recovery_checks = {
        "registry_accepts_exactly_all_P3_rows",
        "model_ids_equal_retained_environment_and_weight_recovery",
        "ledger_ids_equal_model_ids",
        "uniform_ledger_weights_equal_one",
        "nested_split_genotype_id_matches_certified_kernel_id",
        "P3_source_weights_preserved_as_invalid_in_ledger",
        "phenotype_values_unread",
        "outer_test_metrics_unread",
        "final_holdout_outcomes_unread",
    }
    failed_recovery_checks = sorted(
        name for name in required_recovery_checks if recovery_checks.get(name) is not True
    )
    if failed_recovery_checks:
        raise ValueError(
            "Stage-1 recovery validation is incomplete: "
            f"{failed_recovery_checks}"
        )
    ledger_identity = recovery.get("inputs", {}).get("multitrait_ledger", {})
    if ledger_identity.get("sha256") != file_sha256(paths["recovery_ledger"]):
        raise ValueError("Recovery ledger does not match the certified validation input")

    base_identifiers = read_identifier_columns(paths["base_ledger"])
    recovery_identifiers = read_identifier_columns(paths["recovery_ledger"])
    for frame in (base_identifiers, recovery_identifiers):
        for column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str).str.strip()
    for label, frame in (
        ("base", base_identifiers),
        ("recovery", recovery_identifiers),
    ):
        if frame["canonical_observation_id"].eq("").any() or frame[
            "canonical_observation_id"
        ].duplicated().any():
            raise ValueError(f"{label} ledger observation IDs are empty or duplicated")
    base_by_id = base_identifiers.set_index("canonical_observation_id").sort_index()
    recovery_by_id = recovery_identifiers.set_index(
        "canonical_observation_id"
    ).sort_index()
    missing_base_ids = base_by_id.index.difference(recovery_by_id.index)
    if len(missing_base_ids):
        raise ValueError(
            "Recovered ledger lost certified baseline observations: "
            f"{list(missing_base_ids[:5])}"
        )
    common_recovery = recovery_by_id.loc[base_by_id.index]
    split_identity_columns = [
        "panel_sample_id",
        "genotype_id",
        "env_kernel_id",
        "cycle",
        "country",
        "trait_name_canonical",
    ]
    identity_mismatches = {
        column: int((~base_by_id[column].eq(common_recovery[column])).sum())
        for column in split_identity_columns
    }
    changed_identity_columns = {
        column: count for column, count in identity_mismatches.items() if count
    }
    if changed_identity_columns:
        raise ValueError(
            "Common observation split identities changed in the recovered ledger: "
            f"{changed_identity_columns}"
        )
    expected_new_rows = int(recovery["counts"]["recovered_environment_rows"]) + int(
        recovery["counts"]["recovered_weight_rows"]
    )
    observed_new_rows = len(recovery_by_id) - len(base_by_id)
    if observed_new_rows != expected_new_rows:
        raise ValueError(
            "Recovered ledger row delta does not match certified recovery counts: "
            f"observed={observed_new_rows} expected={expected_new_rows}"
        )

    base_evaluation = load_protocol(paths["base_evaluation_protocol"])
    base_contract = json.loads(
        paths["base_evaluation_contract"].read_text(encoding="utf-8")
    )
    base_outer = json.loads(paths["base_outer_protocol"].read_text(encoding="utf-8"))
    base_selection = json.loads(paths["base_selection_lock"].read_text(encoding="utf-8"))
    base_environment_selection = json.loads(
        paths["base_environment_selection_lock"].read_text(encoding="utf-8")
    )
    if base_outer.get("status") != "frozen_after_inner_validation_before_outer_test":
        raise ValueError("Base reaction-norm outer protocol is not frozen")
    if base_contract.get("status") != "frozen":
        raise ValueError("Base nested-evaluation contract is not frozen")
    base_identity_checks = {
        "base_contract_protocol": base_contract.get("protocol_sha256")
        == file_sha256(paths["base_evaluation_protocol"]),
        "base_contract_holdout": base_contract.get(
            "final_holdout_environment_ids_sha256"
        )
        == file_sha256(paths["frozen_final_holdout_environments"]),
        "base_outer_evaluation_protocol": base_outer.get(
            "evaluation_protocol_sha256"
        )
        == file_sha256(paths["base_evaluation_protocol"]),
        "base_selection_outer_protocol": base_selection.get(
            "outer_evaluation_protocol_sha256"
        )
        == file_sha256(paths["base_outer_protocol"]),
        "base_environment_selection_outer_protocol": base_environment_selection.get(
            "outer_evaluation_protocol_sha256"
        )
        == file_sha256(paths["base_outer_protocol"]),
        "base_selection_candidate": base_selection.get("selected_candidate")
        == base_outer.get("selected_candidate"),
        "base_environment_selection_candidate": base_environment_selection.get(
            "selected_environment_architecture"
        )
        == base_outer.get("selected_environment_architecture"),
    }
    failed_base_identities = sorted(
        name for name, passed in base_identity_checks.items() if not passed
    )
    if failed_base_identities:
        raise ValueError(
            "Base nested/reaction provenance is inconsistent: "
            f"{failed_base_identities}"
        )
    if base_selection.get("status") != "PASS" or base_environment_selection.get(
        "status"
    ) != "PASS":
        raise ValueError("Base reaction-norm selection locks did not pass")
    if base_selection.get("outer_test_metrics_read") is not False or base_environment_selection.get(
        "outer_test_metrics_read"
    ) is not False:
        raise ValueError("Base architecture locks were not frozen before outer testing")

    evaluation = {
        key: value
        for key, value in base_evaluation.items()
        if key not in {"protocol_path", "protocol_sha256"}
    }
    evaluation.update(
        {
            "protocol_version": "multitrait_stage1_recovery_nested_v3",
            "status": "frozen",
            "freeze_kind": "phenotype_blind_stage1_recovery_before_recovery_outer_test",
            "selection_data": "identifiers_uncertainty_metadata_and_kernel_orders_only",
            "phenotype_values_used_for_recovery_decision": False,
            "outer_test_metrics_read_at_freeze": False,
            "final_holdout_outcomes_read": False,
            "architecture_selection": {
                "source_outer_protocol": relative(
                    root, paths["base_outer_protocol"]
                ),
                "source_outer_protocol_sha256": file_sha256(
                    paths["base_outer_protocol"]
                ),
                "selected_candidate": base_outer["selected_candidate"],
                "selected_environment_architecture": base_outer[
                    "selected_environment_architecture"
                ],
                "further_selection_performed": False,
            },
            "data_recovery": {
                "validation": relative(root, paths["recovery_validation"]),
                "validation_sha256": file_sha256(paths["recovery_validation"]),
                "ledger": relative(root, paths["recovery_ledger"]),
                "ledger_sha256": file_sha256(paths["recovery_ledger"]),
                "base_ledger": relative(root, paths["base_ledger"]),
                "base_ledger_sha256": file_sha256(paths["base_ledger"]),
                "baseline_rows": len(base_by_id),
                "recovered_rows": len(recovery_by_id),
                "new_rows": observed_new_rows,
                "common_split_identities_preserved": True,
                "recovered_environment_rows": int(
                    recovery["counts"]["recovered_environment_rows"]
                ),
                "recovered_weight_rows": int(
                    recovery["counts"]["recovered_weight_rows"]
                ),
                "weight_power": 0.0,
                "fold_local_weight_statistics": True,
            },
            "frozen_final_holdout_reuse": {
                "source": relative(
                    root, paths["frozen_final_holdout_environments"]
                ),
                "source_sha256": file_sha256(
                    paths["frozen_final_holdout_environments"]
                ),
                "reselect_holdout": False,
            },
        }
    )
    write_json(output_paths["evaluation_protocol"], evaluation)

    outer = dict(base_outer)
    outer.update(
        {
            "protocol_version": "multitrait_reaction_norm_stage1_recovery_nested_v3",
            "selection_data": "frozen_architecture_plus_phenotype_blind_stage1_recovery",
            "outer_test_metrics_read_at_freeze": False,
            "final_holdout_outcomes_read": False,
            "evaluation_protocol_version": evaluation["protocol_version"],
            "evaluation_protocol_sha256": file_sha256(
                output_paths["evaluation_protocol"]
            ),
            "selected_model_label": "multitrait_reaction_norm_stage1_recovered_v1_frozen",
            "data_recovery_contract": {
                "recovery_validation_sha256": file_sha256(
                    paths["recovery_validation"]
                ),
                "recovery_ledger_sha256": file_sha256(paths["recovery_ledger"]),
                "frozen_final_holdout_sha256": file_sha256(
                    paths["frozen_final_holdout_environments"]
                ),
                "recovery_selected_using_outer_metrics": False,
                "further_hyperparameter_selection": False,
                "common_support_comparison_required": True,
            },
            "trait_environment_recovery_contract": {
                "kernel": "K_E_TGW_V2",
                "policy": "frozen_feature_projection_preserve_original_block",
                "implementation_sha256": file_sha256(
                    paths["trait_environment_extension_implementation"]
                ),
                "original_block_max_abs_tolerance": 5e-6,
                "refit_feature_columns": False,
                "refit_feature_scaling": False,
                "phenotype_values_read": False,
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
            },
        }
    )
    write_json(output_paths["outer_protocol"], outer)
    outer_sha256 = file_sha256(output_paths["outer_protocol"])

    selection = dict(base_selection)
    selection.update(
        {
            "freeze_kind": "reaction_norm_architecture_reused_for_stage1_recovery",
            "selection_data": "previously_frozen_inner_validation_architecture",
            "outer_test_metrics_read": False,
            "final_holdout_outcomes_read": False,
            "selected_model_label": outer["selected_model_label"],
            "outer_evaluation_protocol_sha256": outer_sha256,
            "parent_selection_lock": {
                "path": relative(root, paths["base_selection_lock"]),
                "sha256": file_sha256(paths["base_selection_lock"]),
            },
            "stage1_recovery_validation_sha256": file_sha256(
                paths["recovery_validation"]
            ),
            "further_hyperparameter_selection": False,
        }
    )
    write_json(output_paths["selection_lock"], selection)
    environment_selection = dict(base_environment_selection)
    environment_selection.update(
        {
            "freeze_kind": "reaction_norm_environment_reused_for_stage1_recovery",
            "selection_data": "previously_frozen_inner_validation_architecture",
            "outer_test_metrics_read": False,
            "final_holdout_outcomes_read": False,
            "outer_evaluation_protocol_sha256": outer_sha256,
            "parent_environment_selection_lock": {
                "path": relative(
                    root, paths["base_environment_selection_lock"]
                ),
                "sha256": file_sha256(
                    paths["base_environment_selection_lock"]
                ),
            },
            "stage1_recovery_validation_sha256": file_sha256(
                paths["recovery_validation"]
            ),
            "further_environment_architecture_selection": False,
        }
    )
    write_json(
        output_paths["environment_selection_lock"], environment_selection
    )

    checks = {
        "recovery_validation_pass": recovery.get("status") == "PASS",
        "recovery_phenotype_values_unread": recovery.get("phenotype_values_read")
        is False,
        "recovery_outer_metrics_unread": recovery.get("outer_test_metrics_read")
        is False,
        "recovery_final_holdout_unread": recovery.get("final_holdout_outcomes_read")
        is False,
        "ledger_identity_matches_recovery_validation": ledger_identity.get("sha256")
        == file_sha256(paths["recovery_ledger"]),
        "baseline_observations_are_subset_of_recovery": len(missing_base_ids) == 0,
        "common_split_identities_preserved": not changed_identity_columns,
        "certified_recovery_row_delta": observed_new_rows == expected_new_rows,
        "scenario_assignment_preserved": evaluation["scenario_assignment_id"]
        == base_evaluation["scenario_assignment_id"],
        "final_holdout_assignment_preserved": evaluation[
            "final_holdout_assignment_id"
        ]
        == base_evaluation["final_holdout_assignment_id"],
        "architecture_candidate_preserved": outer["selected_candidate"]
        == base_outer["selected_candidate"],
        "environment_architecture_preserved": outer[
            "selected_environment_architecture"
        ]
        == base_outer["selected_environment_architecture"],
        "training_configuration_preserved": outer["selected_configuration"]
        == base_outer["selected_configuration"],
        "trait_environment_extension_implementation_frozen": outer[
            "trait_environment_recovery_contract"
        ]["implementation_sha256"]
        == file_sha256(paths["trait_environment_extension_implementation"]),
        "final_holdout_reselection_disabled": evaluation[
            "frozen_final_holdout_reuse"
        ]["reselect_holdout"]
        is False,
        **base_identity_checks,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    freeze = {
        "status": status,
        "protocol_version": "stage1_recovery_nested_freeze_v3",
        "selection_data": "phenotype_blind_recovery_plus_previously_frozen_architecture",
        "phenotype_values_read_for_recovery": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "further_hyperparameter_selection": False,
        "checks": checks,
        "artifacts": {
            name: {"path": relative(root, path), "sha256": file_sha256(path)}
            for name, path in output_paths.items()
            if name != "freeze"
        },
        "inputs": {
            name: {"path": relative(root, path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
    }
    write_json(output_paths["freeze"], freeze)
    selection_checksums = out_dir / "reaction_norm_selection_artifacts.sha256"
    environment_checksums = (
        out_dir / "reaction_norm_environment_selection_artifacts.sha256"
    )
    checksum_manifest(
        root,
        [
            paths["base_selection_lock"],
            paths["trait_environment_extension_implementation"],
            paths["recovery_validation"],
            output_paths["evaluation_protocol"],
            output_paths["outer_protocol"],
            output_paths["selection_lock"],
            output_paths["freeze"],
        ],
        selection_checksums,
    )
    checksum_manifest(
        root,
        [
            paths["base_environment_selection_lock"],
            paths["trait_environment_extension_implementation"],
            paths["recovery_validation"],
            output_paths["evaluation_protocol"],
            output_paths["outer_protocol"],
            output_paths["environment_selection_lock"],
            output_paths["freeze"],
        ],
        environment_checksums,
    )
    print(json.dumps(freeze, indent=2, allow_nan=False))
    if status != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"Stage-1 recovery nested freeze failed: {failed}")


if __name__ == "__main__":
    main()
