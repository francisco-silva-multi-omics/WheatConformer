from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_common import file_identity, write_json
from .audit_information_attrition import (
    git_commit,
    normalized_text,
    order_ids,
    resolve,
)


REQUIRED_COLUMNS = (
    "canonical_observation_id",
    "canonical_germplasm_key",
    "env_kernel_id",
    "trait_name_canonical",
    "n_plot_records",
    "stage1_to_model_status",
)


def classify_readiness(frame: pd.DataFrame) -> pd.Series:
    retained = frame["stage1_to_model_status"].eq(
        "retained_in_stage1_model_observations"
    )
    invalid_weight = frame["stage1_to_model_status"].eq(
        "invalid_or_nonpositive_stage1_weight"
    )
    legacy = frame["legacy_pedigree_available"]
    canonical = frame["canonical_v3_pedigree_available"]
    environment = frame["global_environment_available"]

    status = pd.Series(
        "P1_MODEL_INTERSECTION_REBUILD_REVIEW", index=frame.index, dtype=object
    )
    status.loc[retained] = "RETAINED_REFERENCE"
    status.loc[~retained & invalid_weight] = "P3_REPAIR_WEIGHT_METADATA"
    eligible = ~retained & ~invalid_weight
    status.loc[eligible & ~canonical & ~environment] = (
        "P1_RECOVER_PEDIGREE_AND_ENVIRONMENT"
    )
    status.loc[eligible & ~canonical & environment] = "P1_RECOVER_PEDIGREE_IDENTITY"
    status.loc[eligible & canonical & ~environment] = "P1_RECOVER_ENVIRONMENT"
    status.loc[eligible & canonical & environment & ~legacy] = (
        "P1_CANONICAL_V3_MODEL_INPUT_REBUILD"
    )
    return status


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.copy()
    local["n_plot_records"] = pd.to_numeric(
        local["n_plot_records"], errors="coerce"
    ).fillna(0)
    return (
        local.groupby("recovery_readiness", sort=True)
        .agg(
            stage1_rows=("canonical_observation_id", "size"),
            unique_genotypes=("canonical_germplasm_key", "nunique"),
            unique_environments=("env_kernel_id", "nunique"),
            unique_traits=("trait_name_canonical", "nunique"),
            represented_raw_plot_records=("n_plot_records", "sum"),
        )
        .reset_index()
        .sort_values("stage1_rows", ascending=False)
        .reset_index(drop=True)
    )


def grouped_candidates(frame: pd.DataFrame, axis: str) -> pd.DataFrame:
    if axis == "genotype":
        key = "canonical_germplasm_key"
        counterpart = "env_kernel_id"
        counterpart_name = "unique_environments"
    else:
        key = "env_kernel_id"
        counterpart = "canonical_germplasm_key"
        counterpart_name = "unique_genotypes"
    candidates = frame[~frame["recovery_readiness"].eq("RETAINED_REFERENCE")].copy()
    candidates["n_plot_records"] = pd.to_numeric(
        candidates["n_plot_records"], errors="coerce"
    ).fillna(0)
    return (
        candidates.groupby(key, dropna=False, sort=True)
        .agg(
            stage1_rows=("canonical_observation_id", "size"),
            **{counterpart_name: (counterpart, "nunique")},
            unique_traits=("trait_name_canonical", "nunique"),
            represented_raw_plot_records=("n_plot_records", "sum"),
            legacy_pedigree_available=("legacy_pedigree_available", "all"),
            canonical_v3_pedigree_available=(
                "canonical_v3_pedigree_available",
                "all",
            ),
            global_environment_available=("global_environment_available", "all"),
            recovery_readiness=(
                "recovery_readiness",
                lambda values: ";".join(sorted(set(values))),
            ),
        )
        .reset_index()
        .sort_values(
            ["represented_raw_plot_records", "stage1_rows"], ascending=False
        )
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether excluded Stage-1 rows are recoverable with full kernel orders."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--attrition-ledger",
        type=Path,
        default=Path(
            "audit/information_attrition_v2/stage1_to_model_attrition_ledger.parquet"
        ),
    )
    parser.add_argument(
        "--legacy-pedigree-order",
        type=Path,
        default=Path("genotype_panels/pedigree/K_A_sample_order.tsv"),
    )
    parser.add_argument(
        "--canonical-v3-order",
        type=Path,
        default=Path(
            "genotype_panels/pedigree_canonical_v3/K_A_CANONICAL_V3_sample_order.tsv"
        ),
    )
    parser.add_argument(
        "--global-environment-order",
        type=Path,
        default=Path("environment/env_kernel_sample_order.tsv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("audit/stage1_signal_recovery_v1"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    paths = {
        "attrition_ledger": resolve(root, args.attrition_ledger),
        "legacy_pedigree_order": resolve(root, args.legacy_pedigree_order),
        "canonical_v3_order": resolve(root, args.canonical_v3_order),
        "global_environment_order": resolve(root, args.global_environment_order),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    frame = pd.read_parquet(paths["attrition_ledger"], columns=list(REQUIRED_COLUMNS))
    frame["canonical_observation_id"] = normalized_text(
        frame["canonical_observation_id"]
    )
    frame["canonical_germplasm_key"] = normalized_text(
        frame["canonical_germplasm_key"]
    )
    frame["env_kernel_id"] = normalized_text(frame["env_kernel_id"])
    frame["trait_name_canonical"] = normalized_text(
        frame["trait_name_canonical"], upper=True
    )

    legacy_order = order_ids(paths["legacy_pedigree_order"])
    canonical_order = order_ids(paths["canonical_v3_order"])
    environment_order = order_ids(paths["global_environment_order"])
    frame["legacy_pedigree_available"] = frame["canonical_germplasm_key"].isin(
        legacy_order
    )
    frame["canonical_v3_pedigree_available"] = frame[
        "canonical_germplasm_key"
    ].isin(canonical_order)
    frame["global_environment_available"] = frame["env_kernel_id"].isin(
        environment_order
    )
    frame["recovery_readiness"] = classify_readiness(frame)

    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(frame)
    summary.to_csv(
        out_dir / "stage1_recovery_readiness_summary.tsv", sep="\t", index=False
    )
    grouped_candidates(frame, "genotype").to_csv(
        out_dir / "stage1_recovery_genotypes.tsv", sep="\t", index=False
    )
    grouped_candidates(frame, "environment").to_csv(
        out_dir / "stage1_recovery_environments.tsv", sep="\t", index=False
    )
    output_columns = [
        *REQUIRED_COLUMNS,
        "legacy_pedigree_available",
        "canonical_v3_pedigree_available",
        "global_environment_available",
        "recovery_readiness",
    ]
    frame[output_columns].to_parquet(
        out_dir / "stage1_recovery_readiness_ledger.parquet", index=False
    )

    checks = {
        "observation_ids_unique": not frame["canonical_observation_id"].duplicated().any(),
        "all_rows_classified": frame["recovery_readiness"].ne("").all(),
        "classification_partitions_rows": int(summary["stage1_rows"].sum())
        == len(frame),
        "canonical_v3_not_smaller_than_legacy": len(canonical_order)
        >= len(legacy_order),
        "phenotype_values_unread": True,
        "outer_test_metrics_unread": True,
        "final_holdout_outcomes_unread": True,
        "kernels_unchanged": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    code_root = Path(__file__).resolve().parents[1]
    provenance = {
        "status": status,
        "protocol_version": "stage1_signal_recovery_readiness_v1",
        "selection_data": "identifiers_support_status_and_full_kernel_orders_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "kernels_modified": False,
        "checks": checks,
        "order_sizes": {
            "legacy_pedigree": len(legacy_order),
            "canonical_v3_pedigree": len(canonical_order),
            "global_environment": len(environment_order),
        },
        "code_root": str(code_root),
        "git_commit": git_commit(code_root),
        "inputs": {name: file_identity(path) for name, path in paths.items()},
    }
    write_json(out_dir / "stage1_recovery_readiness_provenance.json", provenance)
    print(
        json.dumps(
            {
                "status": status,
                "stage1_rows": len(frame),
                "excluded_rows": int(
                    (~frame["recovery_readiness"].eq("RETAINED_REFERENCE")).sum()
                ),
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )
    if status != "PASS":
        raise SystemExit("Stage-1 recovery-readiness audit failed")


if __name__ == "__main__":
    main()
