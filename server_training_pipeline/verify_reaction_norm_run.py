from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .final_evaluation_contract import file_sha256, load_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a reaction-norm inner run before resume.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--stage",
        choices=["inner_selection", "outer_evaluation"],
        default="inner_selection",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--inner-fold", type=int, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--reaction-protocol", type=Path, required=True)
    parser.add_argument("--outer-evaluation-protocol", type=Path)
    parser.add_argument("--reaction-selection-lock", type=Path)
    parser.add_argument("--certification-summary", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--factorization-implementation", type=Path, required=True)
    args = parser.parse_args()

    metadata_path = args.run_dir / f"{args.prefix}_run_metadata.json"
    if not metadata_path.is_file():
        raise SystemExit("metadata absent")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reaction = json.loads(args.reaction_protocol.read_text(encoding="utf-8"))
    evaluation = load_protocol(args.evaluation_protocol)
    candidates = {str(value["name"]): value for value in reaction["candidates"]}
    if args.candidate not in candidates:
        raise SystemExit(f"candidate absent from frozen protocol: {args.candidate}")
    candidate = candidates[args.candidate]
    training = reaction["training"]
    expected_configuration = {
        "max_rank_genotype": int(training["max_rank_genotype"]),
        "max_rank_environment": int(training["max_rank_environment"]),
        "reaction_rank": int(candidate["reaction_rank"]),
        "trait_covariance_shrinkage": float(candidate["trait_covariance_shrinkage"]),
        "trait_covariance_minimum_pairs": int(training["trait_covariance_minimum_pairs"]),
        "ridge_penalty": float(candidate["ridge_penalty"]),
        "residual_scale_floor": float(training["residual_scale_floor"]),
        "epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "patience": int(training["patience"]),
        "intra_op_threads": int(training["intra_op_threads"]),
        "inter_op_threads": int(training["inter_op_threads"]),
    }
    external = metadata.get("external_split", {})
    preprocessing = metadata.get("phenotype_preprocessing", {})
    checks = {
        "status": metadata.get("status") == "PASS",
        "stage": metadata.get("evaluation_stage") == args.stage,
        "scenario": external.get("scenario") == args.scenario,
        "outer_fold": int(external.get("outer_fold", -1)) == args.outer_fold,
        "inner_fold": int(external.get("inner_fold", -1)) == args.inner_fold,
        "seed": int(metadata.get("seed", -1)) == args.seed,
        "candidate": metadata.get("hyperparameter_label") == args.candidate,
        "manifest": external.get("manifest_sha256") == file_sha256(args.split_manifest),
        "evaluation_protocol": metadata.get("evaluation_protocol", {}).get(
            "protocol_sha256"
        )
        == evaluation["protocol_sha256"],
        "reaction_protocol": metadata.get("reaction_protocol", {}).get("sha256")
        == file_sha256(args.reaction_protocol),
        "certification": metadata.get("certification_summary_sha256")
        == file_sha256(args.certification_summary),
        "trainer": metadata.get("trainer_sha256") == file_sha256(args.trainer),
        "factorization": metadata.get("kernel_factorization_sha256")
        == file_sha256(args.factorization_implementation),
        "configuration": metadata.get("training_configuration") == expected_configuration,
        "active_kernels": set(metadata.get("active_kernels", []))
        == set(reaction["required_kernels"]),
        "factorization_mode": metadata.get("requested_factorization_mode")
        == training["factorization_mode"]
        and metadata.get("effective_factorization_mode") == training["factorization_mode"],
        "kernel_centering": metadata.get("kernel_centered")
        is bool(training["kernel_centering"]),
        "fold_local_weights": preprocessing.get("fold_local_weights") is True,
        "outer_test_outcomes_unused": preprocessing.get("outer_test_outcomes_used") is False,
        "final_holdout_unread": metadata.get("final_holdout_outcomes_read") is False,
    }
    protected_rows = int(
        preprocessing.get("protected_outcome_rows_cleared_before_preprocessing", 0)
    )
    if args.stage == "inner_selection":
        checks["protected_outcomes_cleared"] = protected_rows > 0
        checks["outer_test_metrics_unread"] = (
            metadata.get("outer_test_metrics_read") is False
        )
        checks["outer_authorization_absent"] = not metadata.get(
            "outer_evaluation_protocol"
        ) and not metadata.get("reaction_selection_lock")
    else:
        if args.outer_evaluation_protocol is None or args.reaction_selection_lock is None:
            raise SystemExit(
                "Outer run verification requires the outer protocol and selection lock"
            )
        outer = json.loads(
            args.outer_evaluation_protocol.read_text(encoding="utf-8")
        )
        lock = json.loads(args.reaction_selection_lock.read_text(encoding="utf-8"))
        checks.update(
            {
                "outer_test_metrics_read": metadata.get("outer_test_metrics_read")
                is True,
                "outer_protocol": metadata.get("outer_evaluation_protocol", {}).get(
                    "sha256"
                )
                == file_sha256(args.outer_evaluation_protocol),
                "selection_lock": metadata.get("reaction_selection_lock", {}).get(
                    "sha256"
                )
                == file_sha256(args.reaction_selection_lock),
                "selected_candidate": args.candidate
                == outer.get("selected_candidate")
                == lock.get("selected_candidate"),
                "selection_lock_pass": lock.get("status") == "PASS"
                and lock.get("outer_evaluation_allowed") is True,
                "protected_outcomes_not_mutated": protected_rows == 0,
            }
        )
    required_suffixes = [
        "trait_metrics.tsv",
        "macro_metrics.tsv",
        "trait_covariance.tsv",
        "trait_residual_scales.tsv",
        "component_variance_proxies.tsv",
        "fold_expert_support.tsv",
        "split_leakage_qc.tsv",
    ]
    for suffix in required_suffixes:
        checks[f"output_{suffix}"] = (args.run_dir / f"{args.prefix}_{suffix}").is_file()
    checks["predictions"] = any(
        path.is_file()
        for path in (
            args.run_dir / f"{args.prefix}_predictions.parquet",
            args.run_dir / f"{args.prefix}_predictions.tsv.gz",
        )
    )
    macro_path = args.run_dir / f"{args.prefix}_macro_metrics.tsv"
    if macro_path.is_file():
        macro = pd.read_csv(macro_path, sep="\t")
        has_test = macro["split"].astype(str).eq("test").any()
        checks["test_metric_contract"] = (
            not has_test if args.stage == "inner_selection" else has_test
        )
    leakage_path = args.run_dir / f"{args.prefix}_split_leakage_qc.tsv"
    if leakage_path.is_file():
        leakage = pd.read_csv(leakage_path, sep="\t")
        checks["leakage"] = (
            not leakage.empty
            and leakage["leakage_status"].astype(str).str.lower().eq("pass").all()
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("stale or incomplete: " + ",".join(failed))
    print("PASS")


if __name__ == "__main__":
    main()
