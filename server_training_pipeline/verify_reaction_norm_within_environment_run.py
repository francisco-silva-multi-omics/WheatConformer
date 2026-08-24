from __future__ import annotations

import argparse
import json
from pathlib import Path

from .final_evaluation_contract import file_sha256, load_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an inner-only ranking-objective run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--inner-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--environment-certification", type=Path, required=True)
    parser.add_argument("--kernel-certification", type=Path, required=True)
    args = parser.parse_args()

    metadata_path = args.run_dir / f"{args.prefix}_run_metadata.json"
    history_path = args.run_dir / f"{args.prefix}_history.tsv"
    predictions_path = args.run_dir / f"{args.prefix}_predictions.parquet"
    target_path = args.run_dir / f"{args.prefix}_within_environment_target_summary.tsv"
    pair_path = args.run_dir / f"{args.prefix}_within_environment_pair_support.tsv"
    if not all(
        path.is_file()
        for path in (metadata_path, history_path, predictions_path, target_path, pair_path)
    ):
        raise SystemExit(1)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    candidates = {str(value["name"]): value for value in protocol.get("candidates", [])}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    objective = metadata.get("within_environment_objective", {})
    external = metadata.get("external_split", {})
    environment = json.loads(args.environment_certification.read_text(encoding="utf-8"))
    kernels = json.loads(args.kernel_certification.read_text(encoding="utf-8"))
    checks = {
        "status": metadata.get("status") == "PASS",
        "inner_only": metadata.get("evaluation_stage") == "inner_selection",
        "outer_unread": metadata.get("outer_test_metrics_read") is False,
        "final_holdout_unread": metadata.get("final_holdout_outcomes_read") is False,
        "candidate_known": args.candidate in candidates,
        "candidate": objective.get("candidate") == args.candidate,
        "candidate_contract": objective.get("candidate_contract")
        == candidates.get(args.candidate),
        "protocol": objective.get("protocol_sha256") == file_sha256(args.protocol),
        "trainer": metadata.get("trainer_sha256") == file_sha256(args.trainer),
        "scenario": external.get("scenario") == args.scenario,
        "outer_fold": int(external.get("outer_fold", -1)) == args.outer_fold,
        "inner_fold": int(external.get("inner_fold", -1)) == args.inner_fold,
        "seed": int(metadata.get("seed", -1)) == args.seed,
        "manifest": external.get("manifest_sha256") == file_sha256(args.split_manifest),
        "evaluation": metadata.get("evaluation_protocol", {}).get("protocol_sha256")
        == load_protocol(args.evaluation_protocol).get("protocol_sha256"),
        "environment_certified": environment.get("status") == "PASS",
        "environment_identity": metadata.get("environment_design", {}).get(
            "certification_sha256"
        )
        == file_sha256(args.environment_certification),
        "kernels_certified": kernels.get("status") == "PASS",
        "kernel_certification_identity": metadata.get("certification_summary_sha256")
        == file_sha256(args.kernel_certification),
        "loo_means": objective.get("environment_mean_method")
        == "weighted_leave_one_genotype_out",
        "pair_partition": objective.get("pair_sampling_partition")
        == "inner_training_rows_only",
        "target_summary": objective.get("target_summary_sha256")
        == file_sha256(target_path),
        "pair_support": objective.get("pair_support_sha256") == file_sha256(pair_path),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SystemExit("Within-environment run verification failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
