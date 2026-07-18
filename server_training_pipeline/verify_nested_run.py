from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .final_evaluation_contract import file_sha256, load_protocol


def prediction_exists(run_dir: Path, prefix: str) -> bool:
    return any(
        path.exists() and path.stat().st_size > 0
        for path in [
            run_dir / f"{prefix}_predictions.parquet",
            run_dir / f"{prefix}_predictions.tsv.gz",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that a nested run is safe to resume.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--stage", choices=["inner_selection", "outer_evaluation"], required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--inner-fold", type=int, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--mode", choices=["env", "additive", "full"], required=True)
    parser.add_argument("--rank-genotype", type=int, required=True)
    parser.add_argument("--rank-environment", type=int, required=True)
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--intra-op-threads", type=int, required=True)
    parser.add_argument("--inter-op-threads", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--certification-summary", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--factorization-implementation", type=Path, required=True)
    args = parser.parse_args()

    metadata_path = args.run_dir / f"{args.prefix}_run_metadata.json"
    if not metadata_path.exists():
        raise SystemExit("metadata absent")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    protocol = load_protocol(args.protocol)
    expected_configuration = {
        "max_rank_genotype": args.rank_genotype,
        "max_rank_environment": args.rank_environment,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "intra_op_threads": args.intra_op_threads,
        "inter_op_threads": args.inter_op_threads,
    }
    expected_effects = {
        "env": (False, True, False),
        "additive": (True, True, False),
        "full": (True, True, True),
    }[args.mode]
    external = metadata.get("external_split", {})
    preprocessing = metadata.get("phenotype_preprocessing", {})
    checks = {
        "evaluation_stage": metadata.get("evaluation_stage") == args.stage,
        "scenario": external.get("scenario") == args.scenario,
        "outer_fold": int(external.get("outer_fold", -1)) == args.outer_fold,
        "inner_fold": int(external.get("inner_fold", -1)) == args.inner_fold,
        "manifest_sha256": external.get("manifest_sha256") == file_sha256(args.manifest),
        "protocol_sha256": metadata.get("evaluation_protocol", {}).get("protocol_sha256")
        == protocol["protocol_sha256"],
        "certification_sha256": metadata.get("certification_summary_sha256")
        == file_sha256(args.certification_summary),
        "trainer_sha256": metadata.get("trainer_sha256") == file_sha256(args.trainer),
        "kernel_factorization_sha256": metadata.get("kernel_factorization_sha256")
        == file_sha256(args.factorization_implementation),
        "candidate": metadata.get("hyperparameter_label") == args.candidate,
        "seed": int(metadata.get("seed", -1)) == args.seed,
        "model_label": metadata.get("model_label") == args.model_label,
        "training_configuration": metadata.get("training_configuration")
        == expected_configuration,
        "genotype_main": bool(metadata.get("include_genotype_main")) == expected_effects[0],
        "environment_main": bool(metadata.get("include_environment_main")) == expected_effects[1],
        "interaction": bool(metadata.get("include_interaction")) == expected_effects[2],
        "fold_local_weights": preprocessing.get("fold_local_weights") is True,
        "stage1_policy": preprocessing.get("stage1_policy")
        in {
            "environment_isolated_stage1_adjustment",
            "genotype_environment_raw_mean_and_sampling_variance",
        },
        "predictions": prediction_exists(args.run_dir, args.prefix),
    }
    macro_path = args.run_dir / f"{args.prefix}_macro_metrics.tsv"
    if args.stage == "inner_selection" and macro_path.exists():
        macro = pd.read_csv(macro_path, sep="\t")
        checks["inner_has_no_test_metrics"] = not macro["split"].eq("test").any()
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("stale or incomplete: " + ",".join(failed))
    print("PASS")


if __name__ == "__main__":
    main()
