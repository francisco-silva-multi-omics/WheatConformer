from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .final_evaluation_contract import file_sha256
from .nested_evaluation import assign_nested_split, verify_manifest_contract


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if "".join(path.suffixes).lower().endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, sep="\t", low_memory=False, usecols=columns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export immutable outer-training IDs for fold-local preprocessing."
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    contract = verify_manifest_contract(args.manifest, args.contract)
    if file_sha256(args.ledger) != contract.get("ledger_sha256"):
        raise SystemExit("Fold export ledger does not match the frozen manifest contract")
    ledger = read_table(
        args.ledger,
        columns=["panel_sample_id", "env_kernel_id", "cycle", "country"],
    )
    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str)
    # Inner fold zero is used only to recover the outer train union. Its train
    # and validation rows together are exactly the outer-training population.
    train, val, test, omitted, leakage = assign_nested_split(
        ledger,
        manifest,
        scenario=args.scenario,
        outer_fold=args.outer_fold,
        inner_fold=0,
    )
    outer_training = ledger.iloc[sorted(set(train) | set(val))].copy()
    outer_test = ledger.iloc[test].copy()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for column, filename, output_column in [
        ("env_kernel_id", "outer_training_environment_ids.tsv", "env_id"),
        ("panel_sample_id", "outer_training_genotype_ids.tsv", "sample_id"),
    ]:
        values = sorted(
            set(outer_training[column].fillna("").astype(str).str.strip()).difference({""})
        )
        pd.DataFrame({output_column: values}).to_csv(
            out_dir / filename, sep="\t", index=False, lineterminator="\n"
        )
    row_counts = pd.DataFrame(
        [
            {"partition": "outer_training", "rows": len(outer_training)},
            {"partition": "outer_test", "rows": len(outer_test)},
            {"partition": "omitted_or_final", "rows": len(omitted)},
        ]
    )
    row_counts.to_csv(out_dir / "outer_fold_row_counts.tsv", sep="\t", index=False)
    fold_contract = {
        "status": "frozen",
        "scenario": args.scenario,
        "outer_fold": args.outer_fold,
        "source_manifest_sha256": contract["entity_manifest_sha256"],
        "ledger_sha256": contract["ledger_sha256"],
        "outer_training_rows": len(outer_training),
        "outer_test_rows": len(outer_test),
        "omitted_or_final_rows": len(omitted),
        "outer_training_environment_ids_sha256": file_sha256(
            out_dir / "outer_training_environment_ids.tsv"
        ),
        "outer_training_genotype_ids_sha256": file_sha256(
            out_dir / "outer_training_genotype_ids.tsv"
        ),
        "leakage_status": leakage["leakage_status"],
    }
    (out_dir / "outer_fold_contract.json").write_text(
        json.dumps(fold_contract, indent=2), encoding="utf-8"
    )
    print(json.dumps(fold_contract, indent=2))


if __name__ == "__main__":
    main()
