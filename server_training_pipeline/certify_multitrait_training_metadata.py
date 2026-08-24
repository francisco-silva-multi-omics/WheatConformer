from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

from .compare_multitrait_variants import csv_values, load_run, run_directory


def git_commit() -> str:
    repository = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def configuration_difference(
    observed: dict[str, object], expected: dict[str, object]
) -> dict[str, dict[str, object]]:
    return {
        key: {"observed": observed.get(key, "<missing>"), "expected": expected.get(key)}
        for key in sorted(set(observed) | set(expected))
        if observed.get(key) != expected.get(key)
    }


def certify_run(
    root: Path,
    variant: str,
    mode: str,
    seed: int,
    expected: dict[str, object],
    allow_backfill_missing: bool,
) -> dict[str, object]:
    models_root = root / "trained_models"
    run = load_run(root, models_root, variant, mode, seed)
    run_dir = run_directory(models_root, variant, mode, seed)
    metadata_paths = sorted(run_dir.glob("*_run_metadata.json"))
    if len(metadata_paths) != 1:
        raise ValueError(
            f"Expected one run metadata file in {run_dir}; found {len(metadata_paths)}"
        )
    metadata_path = metadata_paths[0]
    metadata = run["metadata"]
    observed = metadata.get("training_configuration")
    if observed == expected:
        status = "PASS_EXISTING"
    elif observed in [None, {}]:
        if not allow_backfill_missing:
            raise ValueError(
                f"Training configuration is missing from {run_dir}; explicit "
                "--allow-backfill-missing is required"
            )
        configuration_json = json.dumps(expected, sort_keys=True, separators=(",", ":"))
        configuration_sha256 = hashlib.sha256(configuration_json.encode()).hexdigest()
        original_text = metadata_path.read_text(encoding="utf-8")
        backup_path = metadata_path.with_name(
            f"{metadata_path.stem}.pre_config_certification.json"
        )
        if not backup_path.exists():
            backup_path.write_text(original_text, encoding="utf-8")
        metadata["training_configuration"] = expected
        metadata["training_configuration_certification"] = {
            "status": "BACKFILLED_MISSING_METADATA",
            "basis": "completed matched-run artifact plus explicit frozen runner contract",
            "certified_at_utc": datetime.now(timezone.utc).isoformat(),
            "certifier_git_commit": git_commit(),
            "configuration_sha256": configuration_sha256,
            "pre_certification_backup": str(backup_path),
            "variant": variant,
            "mode": mode,
            "seed": seed,
        }
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary.replace(metadata_path)
        status = "PASS_BACKFILLED"
    else:
        difference = configuration_difference(observed, expected)
        raise ValueError(
            f"Nonempty training configuration conflicts with the frozen contract in "
            f"{run_dir}: {json.dumps(difference, sort_keys=True)}"
        )
    return {
        "variant": variant,
        "mode": mode,
        "seed": seed,
        "run_dir": str(run_dir),
        "retained_traits": ";".join(run["metadata"].get("traits", [])),
        "evaluation_pair_count": len(run["prediction_metric_keys"]),
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Certify matched multitrait run training metadata without retraining."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--modes", default="env,additive,full")
    parser.add_argument("--seeds", default="2026,2027,2028,2029")
    parser.add_argument("--max-rank-genotype", type=int, default=128)
    parser.add_argument("--max-rank-environment", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--intra-op-threads", type=int, default=16)
    parser.add_argument("--inter-op-threads", type=int, default=2)
    parser.add_argument("--allow-backfill-missing", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    expected = {
        "max_rank_genotype": args.max_rank_genotype,
        "max_rank_environment": args.max_rank_environment,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "intra_op_threads": args.intra_op_threads,
        "inter_op_threads": args.inter_op_threads,
    }
    rows = [
        certify_run(
            root,
            variant,
            mode,
            seed,
            expected,
            args.allow_backfill_missing,
        )
        for variant in args.variant
        for mode in csv_values(args.modes)
        for seed in [int(value) for value in csv_values(args.seeds)]
    ]
    output = args.out if args.out.is_absolute() else root / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(output, sep="\t", index=False, lineterminator="\n")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
