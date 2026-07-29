from __future__ import annotations

import argparse
import json
from pathlib import Path

from .final_evaluation_contract import file_sha256, load_protocol
from .nested_evaluation import verify_manifest_contract


def identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def write_freeze(path: Path, freeze: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == freeze:
            return
        if existing.get("status") == "PASS":
            raise SystemExit(
                "Existing PASS trial-hierarchy freeze disagrees with current inputs; "
                "use a new screen directory"
            )
    path.write_text(
        json.dumps(freeze, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a fresh trial-hierarchy inner screen before metrics."
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-contract", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--reaction-protocol", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--hierarchy-protocol", type=Path, required=True)
    parser.add_argument("--hierarchy-confirmation-provenance", type=Path)
    parser.add_argument("--loss-balance-provenance", type=Path, required=True)
    parser.add_argument("--readiness-ledger", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        args.ledger,
        args.split_manifest,
        args.split_contract,
        args.evaluation_protocol,
        args.reaction_protocol,
        args.environment_protocol,
        args.hierarchy_protocol,
        args.loss_balance_provenance,
        args.readiness_ledger,
        args.trainer,
    ]
    if args.hierarchy_confirmation_provenance is not None:
        paths.append(args.hierarchy_confirmation_provenance)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = verify_manifest_contract(args.split_manifest, args.split_contract)
    evaluation = load_protocol(args.evaluation_protocol)
    reaction = json.loads(args.reaction_protocol.read_text(encoding="utf-8"))
    environment = json.loads(args.environment_protocol.read_text(encoding="utf-8"))
    hierarchy = json.loads(args.hierarchy_protocol.read_text(encoding="utf-8"))
    loss_balance = json.loads(
        args.loss_balance_provenance.read_text(encoding="utf-8")
    )
    hierarchy_scenarios = {
        str(scenario)
        for phase in ("phase_1", "confirmation")
        for scenario in hierarchy.get(phase, {}).get("outer_folds_by_scenario", {})
    }
    allowed_hierarchy_scenarios = set(
        map(
            str,
            hierarchy.get(
                "selection_scenarios", [hierarchy.get("selection_scenario", "")]
            ),
        )
    )
    confirmation = (
        json.loads(
            args.hierarchy_confirmation_provenance.read_text(encoding="utf-8")
        )
        if args.hierarchy_confirmation_provenance is not None
        else None
    )
    source_confirmation = hierarchy.get("source_confirmation")
    confirmation_artifacts_pass = True
    if source_confirmation is not None and confirmation is not None:
        confirmation_dir = args.hierarchy_confirmation_provenance.parent
        confirmation_artifacts_pass = all(
            (confirmation_dir / name).is_file()
            and file_sha256(confirmation_dir / name) == expected
            for name, expected in source_confirmation[
                "required_artifact_sha256"
            ].items()
        )
    checks = {
        "evaluation_contract_frozen": contract.get("status") == "frozen",
        "ledger_matches_contract": contract.get("ledger_sha256")
        == file_sha256(args.ledger),
        "manifest_matches_contract": contract.get("entity_manifest_sha256")
        == file_sha256(args.split_manifest),
        "evaluation_protocol_matches_contract": contract.get("protocol_sha256")
        == evaluation.get("protocol_sha256"),
        "fresh_scenario_assignment": contract.get("scenario_assignment_id")
        == "reaction_norm_trial_hierarchy_development_v1",
        "sealed_final_holdout_reused": contract.get("final_holdout_assignment_id")
        == "multitrait_quantitative_final_v4"
        and contract.get("frozen_final_holdout_source", {}).get("reused_exactly")
        is True,
        "reaction_protocol_frozen": reaction.get("status")
        == "frozen_before_inner_validation",
        "environment_protocol_frozen": environment.get("status")
        == "frozen_before_inner_validation",
        "hierarchy_protocol_frozen": hierarchy.get("status")
        == "frozen_before_inner_validation",
        "hierarchy_scenario_contract": hierarchy_scenarios
        == allowed_hierarchy_scenarios,
        "reaction_candidate_fixed": hierarchy.get("selected_reaction_candidate")
        == "reaction_norm_identity_covariance",
        "environment_architecture_fixed": hierarchy.get(
            "selected_environment_architecture"
        )
        == "explicit_E_REACTION_NORM_V1",
        "current_loss_fixed": hierarchy.get("selected_loss")
        == loss_balance.get("selected_candidate")
        == "current_trait_row_balanced",
        "loss_balance_decision_pass": loss_balance.get("status") == "PASS",
        "loss_balance_selection_inner_only": loss_balance.get("selection_data")
        == "inner_validation_only",
        "loss_balance_outer_unread": loss_balance.get("outer_test_metrics_read")
        is False,
        "loss_balance_final_holdout_unread": loss_balance.get(
            "final_holdout_outcomes_read"
        )
        is False,
        "hierarchy_outer_unread": hierarchy.get("outer_test_metrics_read") is False,
        "hierarchy_final_holdout_unread": hierarchy.get(
            "final_holdout_outcomes_read"
        )
        is False,
        "confirmation_binding_presence": (source_confirmation is None)
        == (confirmation is None),
        "confirmation_binding_status": source_confirmation is None
        or (
            confirmation is not None
            and confirmation.get("status") == "PASS"
            and confirmation.get("phase") == source_confirmation.get("phase")
            and confirmation.get("selected_candidate")
            == source_confirmation.get("selected_candidate")
            and int(confirmation.get("paired_inner_fold_count", -1))
            == int(source_confirmation.get("paired_inner_fold_count", -2))
            and confirmation.get("hierarchy_protocol_sha256")
            == source_confirmation.get("hierarchy_protocol_sha256")
            and confirmation.get("outer_test_metrics_read") is False
            and confirmation.get("final_holdout_outcomes_read") is False
            and confirmation_artifacts_pass
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    freeze = {
        "status": "PASS" if not failed else "FAIL",
        "protocol_version": "reaction_norm_trial_hierarchy_screen_freeze_v1",
        "selection_data": "future_inner_validation_only",
        "prior_v4_outer_metrics_known": True,
        "prior_v4_outer_metrics_used_as_screen_inputs": False,
        "phenotype_values_read_at_freeze": False,
        "outer_test_metrics_read_by_screen": False,
        "final_holdout_outcomes_read": False,
        "outer_evaluation_allowed": False,
        "new_outer_protocol_required_after_selection": True,
        "checks": checks,
        "failed_checks": failed,
        "artifacts": {path.name: identity(path) for path in paths},
    }
    write_freeze(args.out, freeze)
    print(json.dumps(freeze, indent=2, allow_nan=False))
    if failed:
        raise SystemExit("Trial-hierarchy screen freeze failed")


if __name__ == "__main__":
    main()
