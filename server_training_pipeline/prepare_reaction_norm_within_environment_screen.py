from __future__ import annotations

import argparse
import json
from pathlib import Path

from .final_evaluation_contract import file_sha256, load_protocol
from .nested_evaluation import verify_manifest_contract
from .train_multitrait_multikernel_tf import file_identity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the within-environment objective screen before validation metrics."
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-contract", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--reaction-protocol", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--hierarchy-protocol", type=Path, required=True)
    parser.add_argument("--within-environment-protocol", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--objective-implementation", type=Path, required=True)
    parser.add_argument("--final-holdout-environments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    evaluation = load_protocol(args.evaluation_protocol)
    contract = verify_manifest_contract(args.split_manifest, args.split_contract)
    reaction = json.loads(args.reaction_protocol.read_text(encoding="utf-8"))
    environment = json.loads(args.environment_protocol.read_text(encoding="utf-8"))
    hierarchy = json.loads(args.hierarchy_protocol.read_text(encoding="utf-8"))
    objective = json.loads(args.within_environment_protocol.read_text(encoding="utf-8"))
    hierarchy_candidates = {
        str(value["name"]): value for value in hierarchy.get("candidates", [])
    }
    environment_candidates = {
        str(value["name"]): value for value in environment.get("candidates", [])
    }
    objective_candidates = {
        str(value["name"]): value for value in objective.get("candidates", [])
    }
    route_hierarchies = {
        str(value.get("trial_hierarchy_candidate", ""))
        for value in objective.get("scenario_routes", {}).values()
    }
    checks = {
        "evaluation_contract_frozen": contract.get("status") == "frozen",
        "ledger_matches_contract": file_sha256(args.ledger)
        == contract.get("ledger_sha256"),
        "manifest_matches_contract": file_sha256(args.split_manifest)
        == contract.get("entity_manifest_sha256"),
        "evaluation_matches_contract": evaluation.get("protocol_sha256")
        == contract.get("protocol_sha256"),
        "fresh_scenario_assignment": evaluation.get("scenario_assignment_id")
        == "reaction_norm_within_environment_development_v1",
        "sealed_holdout_assignment_reused": evaluation.get(
            "final_holdout_assignment_id"
        )
        == "multitrait_quantitative_final_v4",
        "reaction_protocol_frozen": reaction.get("status")
        == "frozen_before_inner_validation",
        "environment_protocol_frozen": environment.get("status")
        == "frozen_before_inner_validation",
        "objective_protocol_frozen": objective.get("status")
        == "frozen_before_inner_validation",
        "objective_inner_only": objective.get("selection_data")
        == "inner_validation_only",
        "objective_prior_outer_known": objective.get("prior_routed_outer_metrics_known")
        is True,
        "objective_prior_outer_unused": objective.get(
            "prior_routed_outer_metrics_used_as_screen_inputs"
        )
        is False,
        "objective_outer_unread": objective.get("outer_test_metrics_read") is False,
        "objective_final_holdout_unread": objective.get("final_holdout_outcomes_read")
        is False,
        "reaction_candidate_fixed": objective.get("selected_reaction_candidate")
        in {str(value["name"]) for value in reaction.get("candidates", [])},
        "environment_architecture_fixed": objective.get(
            "selected_environment_architecture"
        )
        in environment_candidates,
        "hierarchy_routes_known": route_hierarchies.issubset(hierarchy_candidates),
        "reference_present": "current_routed_reference" in objective_candidates,
        "candidate_count_bounded": 2 <= len(objective_candidates) <= 4,
        "trainer_isolated": args.trainer.name
        == "train_multitrait_reaction_norm_within_environment_tf.py",
        "objective_implementation_present": args.objective_implementation.name
        == "within_environment_objective.py",
        "final_holdout_manifest_present": args.final_holdout_environments.is_file(),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    status = "PASS" if not failed else "FAIL"
    artifacts = {
        path.name: file_identity(path)
        for path in (
            args.ledger,
            args.split_manifest,
            args.split_contract,
            args.evaluation_protocol,
            args.reaction_protocol,
            args.environment_protocol,
            args.hierarchy_protocol,
            args.within_environment_protocol,
            args.trainer,
            args.objective_implementation,
            args.final_holdout_environments,
        )
    }
    payload = {
        "status": status,
        "protocol_version": "reaction_norm_within_environment_screen_freeze_v1",
        "selection_data": "future_inner_validation_only",
        "prior_routed_outer_metrics_known": True,
        "prior_routed_outer_metrics_used_as_screen_inputs": False,
        "phenotype_values_read_at_freeze": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "new_outer_protocol_required_after_selection": True,
        "checks": checks,
        "failed_checks": failed,
        "artifacts": artifacts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, allow_nan=False))
    if failed:
        raise SystemExit("Within-environment screen freeze failed")


if __name__ == "__main__":
    main()
