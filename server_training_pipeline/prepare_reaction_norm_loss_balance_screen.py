from __future__ import annotations

import argparse
import json
from pathlib import Path

from .final_evaluation_contract import file_sha256


def identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the recovered-data balanced-loss screen before inner metrics."
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-contract", type=Path, required=True)
    parser.add_argument("--recovery-outer-protocol", type=Path, required=True)
    parser.add_argument("--reaction-protocol", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--loss-balance-protocol", type=Path, required=True)
    parser.add_argument("--leverage-provenance", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        args.ledger,
        args.split_manifest,
        args.split_contract,
        args.recovery_outer_protocol,
        args.reaction_protocol,
        args.environment_protocol,
        args.loss_balance_protocol,
        args.leverage_provenance,
        args.trainer,
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = json.loads(args.split_contract.read_text(encoding="utf-8"))
    outer = json.loads(args.recovery_outer_protocol.read_text(encoding="utf-8"))
    reaction = json.loads(args.reaction_protocol.read_text(encoding="utf-8"))
    environment = json.loads(args.environment_protocol.read_text(encoding="utf-8"))
    loss = json.loads(args.loss_balance_protocol.read_text(encoding="utf-8"))
    leverage = json.loads(args.leverage_provenance.read_text(encoding="utf-8"))
    checks = {
        "split_contract_frozen": contract.get("status") == "frozen",
        "ledger_matches_contract": contract.get("ledger_sha256")
        == file_sha256(args.ledger),
        "manifest_matches_contract": contract.get("entity_manifest_sha256")
        == file_sha256(args.split_manifest),
        "recovery_outer_protocol_frozen": outer.get("status")
        == "frozen_after_inner_validation_before_outer_test",
        "reaction_candidate_fixed": outer.get("selected_candidate")
        == loss.get("selected_reaction_candidate")
        == "reaction_norm_identity_covariance",
        "environment_architecture_fixed": outer.get(
            "selected_environment_architecture"
        )
        == loss.get("selected_environment_architecture")
        == "explicit_E_REACTION_NORM_V1",
        "reaction_protocol_frozen": reaction.get("status")
        == "frozen_before_inner_validation",
        "environment_protocol_frozen": environment.get("status")
        == "frozen_before_inner_validation",
        "loss_protocol_frozen": loss.get("status")
        == "frozen_before_inner_validation",
        "loss_protocol_outer_unread": loss.get("outer_test_metrics_read") is False,
        "loss_protocol_final_holdout_unread": loss.get(
            "final_holdout_outcomes_read"
        )
        is False,
        "leverage_audit_pass": leverage.get("status") == "PASS",
        "leverage_audit_phenotype_blind": leverage.get("phenotype_values_read")
        is False,
        "leverage_audit_outer_unread": leverage.get("outer_test_metrics_read")
        is False,
        "leverage_audit_final_holdout_unread": leverage.get(
            "final_holdout_outcomes_read"
        )
        is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    freeze = {
        "status": "PASS" if not failed else "FAIL",
        "protocol_version": "reaction_norm_loss_balance_screen_freeze_v1",
        "selection_data": "future_inner_validation_only",
        "prior_v4_outer_metrics_known": True,
        "prior_v4_outer_metrics_used_as_screen_inputs": False,
        "phenotype_values_read_at_freeze": False,
        "outer_test_metrics_read_by_screen": False,
        "final_holdout_outcomes_read": False,
        "outer_evaluation_allowed": False,
        "new_v5_outer_assignment_required_after_selection": True,
        "checks": checks,
        "failed_checks": failed,
        "artifacts": {path.name: identity(path) for path in paths},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(freeze, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(freeze, indent=2, allow_nan=False))
    if failed:
        raise SystemExit("Balanced-loss screen freeze failed")


if __name__ == "__main__":
    main()
