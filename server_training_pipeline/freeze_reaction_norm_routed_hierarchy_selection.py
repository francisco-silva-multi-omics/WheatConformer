from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .final_evaluation_contract import file_sha256, load_protocol
from .nested_evaluation import verify_manifest_contract


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def write_immutable(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == value:
            return
        if existing.get("status") == "PASS":
            raise SystemExit(f"Existing PASS routed lock disagrees: {path}")
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def source_decision_checks(
    provenance_path: Path,
    specification: dict[str, object],
) -> tuple[dict[str, bool], list[Path]]:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    artifact_paths = [
        provenance_path.parent / filename
        for filename in specification["required_artifact_sha256"]
    ]
    checks = {
        "status": provenance.get("status") == "PASS",
        "phase": provenance.get("phase") == specification.get("phase"),
        "selected_candidate": provenance.get("selected_candidate")
        == specification.get("selected_candidate"),
        "paired_inner_fold_count": int(
            provenance.get("paired_inner_fold_count", -1)
        )
        == int(specification.get("paired_inner_fold_count", -2)),
        "protocol_sha256": provenance.get("hierarchy_protocol_sha256")
        == specification.get("protocol_sha256"),
        "outer_unread": provenance.get("outer_test_metrics_read") is False,
        "final_holdout_unread": provenance.get("final_holdout_outcomes_read")
        is False,
        "artifact_hashes": all(
            path.is_file()
            and file_sha256(path)
            == specification["required_artifact_sha256"][path.name]
            for path in artifact_paths
        ),
    }
    return checks, [provenance_path, *artifact_paths]


def checksum_lines(root: Path, paths: list[Path]) -> str:
    unique = sorted(set(path.resolve() for path in paths), key=str)
    lines = []
    for path in unique:
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = str(path)
        lines.append(f"{file_sha256(path)}  {label}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze identifier-routed hierarchy selection before outer metrics."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-contract", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--reaction-protocol", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--known-hierarchy-protocol", type=Path, required=True)
    parser.add_argument("--transfer-guard-protocol", type=Path, required=True)
    parser.add_argument("--known-confirmation-provenance", type=Path, required=True)
    parser.add_argument("--transfer-guard-provenance", type=Path, required=True)
    parser.add_argument("--outer-protocol", type=Path, required=True)
    parser.add_argument("--hierarchy-trainer", type=Path, required=True)
    parser.add_argument("--base-trainer", type=Path, required=True)
    parser.add_argument("--factorization-implementation", type=Path, required=True)
    parser.add_argument("--run-verifier", type=Path, required=True)
    parser.add_argument("--outer-verifier", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    names = [
        "ledger",
        "split_manifest",
        "split_contract",
        "evaluation_protocol",
        "reaction_protocol",
        "environment_protocol",
        "known_hierarchy_protocol",
        "transfer_guard_protocol",
        "known_confirmation_provenance",
        "transfer_guard_provenance",
        "outer_protocol",
        "hierarchy_trainer",
        "base_trainer",
        "factorization_implementation",
        "run_verifier",
        "outer_verifier",
    ]
    paths = {name: resolve(root, getattr(args, name)) for name in names}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Routed freeze inputs are missing: {missing}")

    contract = verify_manifest_contract(paths["split_manifest"], paths["split_contract"])
    evaluation = load_protocol(paths["evaluation_protocol"])
    reaction = json.loads(paths["reaction_protocol"].read_text(encoding="utf-8"))
    environment = json.loads(
        paths["environment_protocol"].read_text(encoding="utf-8")
    )
    outer = json.loads(paths["outer_protocol"].read_text(encoding="utf-8"))
    source = outer["source_decisions"]
    known_checks, known_artifacts = source_decision_checks(
        paths["known_confirmation_provenance"], source["known_environment_confirmation"]
    )
    transfer_checks, transfer_artifacts = source_decision_checks(
        paths["transfer_guard_provenance"], source["environment_transfer_guard"]
    )

    expected_routes = {
        scenario: (
            "trial_and_environment_intercepts"
            if scenario == "unseen_genotypes"
            else "current_reaction_norm"
        )
        for scenario in outer["scenarios"]
    }
    observed_routes = {
        scenario: route.get("trial_hierarchy_candidate")
        for scenario, route in outer.get("scenario_routes", {}).items()
    }
    implementation_paths = {
        "hierarchy_trainer_sha256": paths["hierarchy_trainer"],
        "base_trainer_sha256": paths["base_trainer"],
        "factorization_sha256": paths["factorization_implementation"],
        "run_verifier_sha256": paths["run_verifier"],
        "outer_verifier_sha256": paths["outer_verifier"],
    }
    checks = {
        "contract_frozen": contract.get("status") == "frozen",
        "ledger_matches_contract": contract.get("ledger_sha256")
        == file_sha256(paths["ledger"]),
        "manifest_matches_contract": contract.get("entity_manifest_sha256")
        == file_sha256(paths["split_manifest"]),
        "evaluation_matches_contract": contract.get("protocol_sha256")
        == evaluation.get("protocol_sha256"),
        "outer_protocol_frozen": outer.get("status")
        == "frozen_after_inner_validation_before_outer_test",
        "outer_unread_at_freeze": outer.get("outer_test_metrics_read_at_freeze")
        is False,
        "outer_unused_for_routing": outer.get("outer_test_metrics_used_for_routing")
        is False,
        "final_holdout_unread": outer.get("final_holdout_outcomes_read") is False,
        "reaction_protocol": outer.get("inner_reaction_protocol_sha256")
        == file_sha256(paths["reaction_protocol"]),
        "environment_protocol": outer.get(
            "environment_architecture_protocol_sha256"
        )
        == file_sha256(paths["environment_protocol"]),
        "evaluation_protocol": outer.get("evaluation_protocol_sha256")
        == file_sha256(paths["evaluation_protocol"]),
        "known_hierarchy_protocol": source["known_environment_confirmation"][
            "protocol_sha256"
        ]
        == file_sha256(paths["known_hierarchy_protocol"]),
        "transfer_guard_protocol": source["environment_transfer_guard"][
            "protocol_sha256"
        ]
        == file_sha256(paths["transfer_guard_protocol"]),
        "reaction_candidate_fixed": outer.get("selected_candidate")
        == "reaction_norm_identity_covariance",
        "environment_architecture_fixed": outer.get(
            "selected_environment_architecture"
        )
        == "explicit_E_REACTION_NORM_V1",
        "scenario_grid_matches_routes": set(outer["scenarios"])
        == set(outer.get("scenario_routes", {})),
        "identifier_route_exact": observed_routes == expected_routes,
        "future_environment_route": outer.get("model_contract", {}).get(
            "future_environment_route"
        )
        == "current_reaction_norm",
        "no_further_selection": all(
            outer.get("model_contract", {}).get(key) is True
            for key in (
                "no_further_hyperparameter_selection",
                "no_further_environment_architecture_selection",
                "no_further_hierarchy_selection",
            )
        ),
        "implementation_frozen": all(
            outer.get("implementation", {}).get(label) == file_sha256(path)
            for label, path in implementation_paths.items()
        ),
        "known_decision": all(known_checks.values()),
        "transfer_decision": all(transfer_checks.values()),
        "protocols_frozen": reaction.get("status") == "frozen_before_inner_validation"
        and environment.get("status") == "frozen_before_inner_validation",
    }

    transfer_decision = pd.read_csv(
        paths["transfer_guard_provenance"].parent
        / "trial_hierarchy_inner_screen_decision.tsv",
        sep="\t",
    )
    candidate_row = transfer_decision[
        transfer_decision["candidate"].astype(str).eq(
            "trial_and_environment_intercepts"
        )
    ]
    accepted_value = (
        str(candidate_row.iloc[0]["accepted"]).strip().lower()
        if len(candidate_row) == 1
        else ""
    )
    checks["transfer_candidate_rejected"] = bool(
        len(candidate_row) == 1
        and accepted_value in {"false", "0", "no"}
        and candidate_row.iloc[0]["decision"] == "do_not_advance"
    )
    failed = sorted(name for name, passed in checks.items() if not passed)

    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze = {
        "status": "PASS" if not failed else "FAIL",
        "protocol_version": "reaction_norm_routed_hierarchy_selection_lock_v1",
        "selection_data": "inner_validation_only",
        "outer_test_metrics_read": False,
        "outer_test_metrics_used_for_routing": False,
        "final_holdout_outcomes_read": False,
        "outer_evaluation_allowed": not failed,
        "selected_candidate": outer["selected_candidate"],
        "selected_model_label": outer["selected_model_label"],
        "scenario_routes": outer["scenario_routes"],
        "outer_evaluation_protocol_sha256": file_sha256(paths["outer_protocol"]),
        "checks": checks,
        "failed_checks": failed,
    }
    environment_lock = {
        "status": freeze["status"],
        "protocol_version": "reaction_norm_routed_environment_selection_lock_v1",
        "selection_data": "inner_validation_only",
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "outer_evaluation_allowed": not failed,
        "selected_environment_architecture": outer[
            "selected_environment_architecture"
        ],
        "environment_architecture_protocol_sha256": file_sha256(
            paths["environment_protocol"]
        ),
        "outer_evaluation_protocol_sha256": file_sha256(paths["outer_protocol"]),
        "scenario_routes": outer["scenario_routes"],
    }
    freeze_path = out_dir / "routed_hierarchy_selection_freeze.json"
    selection_path = out_dir / "reaction_norm_selection_lock.json"
    environment_path = out_dir / "reaction_norm_environment_selection_lock.json"
    write_immutable(freeze_path, freeze)
    write_immutable(selection_path, freeze)
    write_immutable(environment_path, environment_lock)

    common = [
        *paths.values(),
        *known_artifacts,
        *transfer_artifacts,
        freeze_path,
        selection_path,
        environment_path,
    ]
    checksums = checksum_lines(root, common)
    (out_dir / "reaction_norm_selection_artifacts.sha256").write_text(
        checksums, encoding="utf-8"
    )
    (out_dir / "reaction_norm_environment_selection_artifacts.sha256").write_text(
        checksums, encoding="utf-8"
    )
    print(json.dumps(freeze, indent=2, allow_nan=False))
    if failed:
        raise SystemExit("Routed hierarchy selection freeze failed")


if __name__ == "__main__":
    main()
