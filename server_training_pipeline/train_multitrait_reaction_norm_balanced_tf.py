from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import train_multitrait_reaction_norm_tf as base
from .loss_balance import (
    fold_local_balanced_loss_weights,
    loss_weight_diagnostics,
    validate_loss_balance_policy,
)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def option_value(arguments: list[str], option: str, default: str | None = None) -> str:
    if option not in arguments:
        if default is None:
            raise SystemExit(f"Balanced reaction trainer requires {option}")
        return default
    position = arguments.index(option)
    if position + 1 >= len(arguments):
        raise SystemExit(f"Missing value for {option}")
    return arguments[position + 1]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--loss-balance-protocol", type=Path, required=True)
    parser.add_argument("--loss-balance-candidate", required=True)
    custom, remaining = parser.parse_known_args()

    protocol_path = custom.loss_balance_protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_inner_validation":
        raise SystemExit("Loss-balance protocol was not frozen before inner validation")
    if protocol.get("outer_test_metrics_read") is not False:
        raise SystemExit("Loss-balance protocol has read outer-test metrics")
    if protocol.get("final_holdout_outcomes_read") is not False:
        raise SystemExit("Loss-balance protocol has read final-holdout outcomes")
    policies = {str(value["name"]): value for value in protocol.get("candidates", [])}
    if custom.loss_balance_candidate not in policies:
        raise SystemExit("Loss-balance candidate is absent from the frozen protocol")
    policy = policies[custom.loss_balance_candidate]
    validate_loss_balance_policy(policy)

    stage = option_value(remaining, "--evaluation-stage")
    if stage != "inner_selection":
        raise SystemExit(
            "The v1 balanced-loss trainer is restricted to inner selection; "
            "outer evaluation requires a separately frozen v5 contract"
        )
    reaction_candidate = option_value(remaining, "--reaction-candidate")
    environment_architecture = option_value(
        remaining, "--environment-architecture"
    )
    if reaction_candidate != protocol.get("selected_reaction_candidate"):
        raise SystemExit("Balanced-loss protocol and reaction candidate disagree")
    if environment_architecture != protocol.get("selected_environment_architecture"):
        raise SystemExit("Balanced-loss protocol and environment architecture disagree")
    if float(protocol.get("precision_weight_power", float("nan"))) != 0.0:
        raise SystemExit("Balanced-loss v1 requires the frozen uniform precision policy")
    if float(option_value(remaining, "--weight-power")) != 0.0:
        raise SystemExit("Balanced-loss command must use the frozen weight_power=0")

    out_dir = Path(option_value(remaining, "--out-dir"))
    prefix = option_value(remaining, "--prefix", "multitrait_reaction_norm")
    captured: dict[str, object] = {}
    original_make_dataset = base.make_dataset

    def balanced_make_dataset(frame, expert_columns, batch_size, shuffle, seed):
        local = frame
        if shuffle:
            local = frame.copy()
            weights = fold_local_balanced_loss_weights(local, policy)
            local["loss_weight"] = weights
            captured["diagnostics"] = loss_weight_diagnostics(
                local,
                weights,
                policy_name=custom.loss_balance_candidate,
            )
        return original_make_dataset(local, expert_columns, batch_size, shuffle, seed)

    base.make_dataset = balanced_make_dataset
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        base.main()
    finally:
        sys.argv = original_argv
        base.make_dataset = original_make_dataset

    diagnostics = captured.get("diagnostics")
    if diagnostics is None:
        raise RuntimeError("Balanced-loss training did not expose its training partition")
    diagnostics_path = out_dir / f"{prefix}_loss_weight_diagnostics.tsv"
    diagnostics.to_csv(diagnostics_path, sep="\t", index=False)

    metadata_path = out_dir / f"{prefix}_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["base_trainer_sha256"] = metadata.get("trainer_sha256")
    metadata["trainer_sha256"] = file_sha256(Path(__file__).resolve())
    metadata["hyperparameter_label"] = custom.loss_balance_candidate
    metadata["model_family"] = (
        "penalized_multitrait_reaction_norm_mixed_model_fold_local_balanced_loss"
    )
    metadata["loss_balance"] = {
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "protocol_version": protocol["protocol_version"],
        "candidate": custom.loss_balance_candidate,
        "policy": policy,
        "count_fit_partition": "inner_training_only",
        "recovery_status_used_for_weighting": False,
        "phenotype_values_used_for_weight_counts": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "diagnostics_path": str(diagnostics_path.resolve()),
        "diagnostics_sha256": file_sha256(diagnostics_path),
    }
    metadata["phenotype_preprocessing"]["loss_balance_count_fit_partition"] = (
        "inner_training_only"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
