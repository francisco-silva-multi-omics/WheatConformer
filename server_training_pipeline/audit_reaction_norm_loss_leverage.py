from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .final_evaluation_contract import file_sha256
from .loss_balance import fold_local_balanced_loss_weights
from .nested_evaluation import assign_nested_split, verify_manifest_contract


LEDGER_COLUMNS = [
    "canonical_observation_id",
    "trait_name_canonical",
    "genotype_id",
    "environment_id",
    "panel_sample_id",
    "env_kernel_id",
    "cycle",
    "country",
    "weight_g_e",
]
READINESS_COLUMNS = [
    "canonical_observation_id",
    "canonical_germplasm_key",
    "env_kernel_id",
    "recovery_readiness",
]


def parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema.names)


def read_columns(path: Path, required: list[str]) -> pd.DataFrame:
    available = parquet_columns(path)
    missing = sorted(set(required).difference(available))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return pd.read_parquet(path, columns=required)


def family_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    required = {"sample_id", "parent1", "parent2"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Pedigree parent table is missing columns: {missing}")
    result: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        sample = str(getattr(row, "sample_id")).strip()
        parents = sorted(
            value
            for value in (
                str(getattr(row, "parent1")).strip(),
                str(getattr(row, "parent2")).strip(),
            )
            if value
        )
        if sample:
            result[sample] = "|".join(parents) if parents else f"UNRESOLVED:{sample}"
    return result


def maximum_entity_share(
    frame: pd.DataFrame, column: str, weight_column: str = "loss_weight"
) -> float:
    total = float(frame[weight_column].sum())
    shares = frame.groupby(column, dropna=False)[weight_column].sum() / total
    return float(shares.max())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit fold-local reaction-norm loss leverage using identifiers, "
            "recovery provenance, and training support only."
        )
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--readiness-ledger", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-contract", type=Path, required=True)
    parser.add_argument("--loss-balance-protocol", type=Path, required=True)
    parser.add_argument("--pedigree-parent-table", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.ledger,
        args.readiness_ledger,
        args.split_manifest,
        args.split_contract,
        args.loss_balance_protocol,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = verify_manifest_contract(args.split_manifest, args.split_contract)
    if file_sha256(args.ledger) != contract.get("ledger_sha256"):
        raise ValueError("Loss-leverage audit ledger disagrees with the split contract")
    protocol = json.loads(args.loss_balance_protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_inner_validation":
        raise ValueError("Loss-balance protocol is not frozen")
    if float(protocol.get("precision_weight_power", float("nan"))) != 0.0:
        raise ValueError("Loss-leverage audit currently requires weight_power=0")

    ledger = read_columns(args.ledger, LEDGER_COLUMNS)
    readiness = read_columns(args.readiness_ledger, READINESS_COLUMNS).rename(
        columns={
            "canonical_germplasm_key": "readiness_genotype_id",
            "env_kernel_id": "source_env_kernel_id",
        }
    )
    if ledger["canonical_observation_id"].duplicated().any():
        raise ValueError("Recovered ledger observation IDs are duplicated")
    if readiness["canonical_observation_id"].duplicated().any():
        raise ValueError("Readiness ledger observation IDs are duplicated")
    ledger = ledger.merge(
        readiness,
        on="canonical_observation_id",
        how="left",
        validate="one_to_one",
    )
    if ledger["recovery_readiness"].isna().any():
        raise ValueError("Some recovered-ledger rows lack readiness provenance")
    if not ledger["genotype_id"].astype(str).eq(
        ledger["readiness_genotype_id"].astype(str)
    ).all():
        raise ValueError("Readiness and model genotype identities disagree")
    ledger["source_trial"] = (
        ledger["source_env_kernel_id"].fillna("").astype(str).str.split("|").str[0]
    )
    retained = ledger["recovery_readiness"].eq("RETAINED_REFERENCE")
    ledger.loc[retained, "source_trial"] = "RETAINED_REFERENCE_ALL"
    families = family_map(args.pedigree_parent_table)
    ledger["family_id"] = ledger["genotype_id"].astype(str).map(families)
    ledger["family_id"] = ledger["family_id"].fillna(
        "UNRESOLVED:" + ledger["genotype_id"].astype(str)
    )

    manifest = pd.read_csv(args.split_manifest, sep="\t", dtype=str)
    policies = {str(value["name"]): value for value in protocol["candidates"]}
    fold_spec = protocol["confirmation"]["outer_folds_by_scenario"]
    expected_inner = int(protocol["confirmation"]["inner_folds"])
    fold_rows: list[dict[str, object]] = []
    recovery_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    for scenario, outer_folds in fold_spec.items():
        for outer_fold in outer_folds:
            for inner_fold in range(expected_inner):
                train_index, _, _, _, leakage = assign_nested_split(
                    ledger,
                    manifest,
                    scenario=scenario,
                    outer_fold=int(outer_fold),
                    inner_fold=inner_fold,
                )
                leakage_rows.append(leakage)
                training = ledger.iloc[train_index].copy().reset_index(drop=True)
                training["weight_g_e"] = 1.0
                for candidate, policy in policies.items():
                    training["loss_weight"] = fold_local_balanced_loss_weights(
                        training, policy
                    )
                    for trait, group in training.groupby(
                        "trait_name_canonical", sort=True
                    ):
                        total = float(group["loss_weight"].sum())
                        normalized = group["loss_weight"].to_numpy(float) / total
                        recovery_mass = float(
                            group.loc[
                                ~group["recovery_readiness"].eq("RETAINED_REFERENCE"),
                                "loss_weight",
                            ].sum()
                            / total
                        )
                        fold_rows.append(
                            {
                                "scenario": scenario,
                                "outer_fold": int(outer_fold),
                                "inner_fold": inner_fold,
                                "loss_balance_candidate": candidate,
                                "trait_name_canonical": trait,
                                "training_rows": int(len(group)),
                                "unique_training_genotypes": int(
                                    group["genotype_id"].nunique()
                                ),
                                "unique_training_environments": int(
                                    group["environment_id"].nunique()
                                ),
                                "unique_training_families": int(
                                    group["family_id"].nunique()
                                ),
                                "effective_observation_count": float(
                                    1.0 / np.square(normalized).sum()
                                ),
                                "recovered_loss_weight_share": recovery_mass,
                                "maximum_environment_weight_share": maximum_entity_share(
                                    group, "environment_id"
                                ),
                                "maximum_genotype_weight_share": maximum_entity_share(
                                    group, "genotype_id"
                                ),
                                "maximum_family_weight_share": maximum_entity_share(
                                    group, "family_id"
                                ),
                                "maximum_trial_weight_share": maximum_entity_share(
                                    group, "source_trial"
                                ),
                            }
                        )
                    grouped = (
                        training.groupby(
                            [
                                "trait_name_canonical",
                                "recovery_readiness",
                                "source_trial",
                            ],
                            dropna=False,
                            sort=True,
                        )
                        .agg(
                            rows=("canonical_observation_id", "size"),
                            unique_genotypes=("genotype_id", "nunique"),
                            unique_environments=("environment_id", "nunique"),
                            loss_weight_sum=("loss_weight", "sum"),
                        )
                        .reset_index()
                    )
                    trait_mass = training.groupby("trait_name_canonical")[
                        "loss_weight"
                    ].sum()
                    grouped["loss_weight_share_within_trait"] = [
                        value / trait_mass.loc[trait]
                        for trait, value in zip(
                            grouped["trait_name_canonical"], grouped["loss_weight_sum"]
                        )
                    ]
                    grouped.insert(0, "loss_balance_candidate", candidate)
                    grouped.insert(0, "inner_fold", inner_fold)
                    grouped.insert(0, "outer_fold", int(outer_fold))
                    grouped.insert(0, "scenario", scenario)
                    recovery_rows.extend(grouped.to_dict("records"))

    fold_summary = pd.DataFrame(fold_rows)
    recovery_summary = pd.DataFrame(recovery_rows)
    leakage = pd.DataFrame(leakage_rows)
    checks = {
        "ledger_ids_unique": bool(
            not ledger["canonical_observation_id"].duplicated().any()
        ),
        "readiness_complete": bool(ledger["recovery_readiness"].notna().all()),
        "split_leakage_zero": bool(leakage["leakage_status"].eq("pass").all()),
        "loss_weights_positive_finite": bool(
            np.isfinite(recovery_summary["loss_weight_sum"].to_numpy(float)).all()
            and recovery_summary["loss_weight_sum"].gt(0).all()
        ),
        "phenotype_values_not_read": True,
        "outer_test_metrics_not_read": True,
        "final_holdout_outcomes_not_read": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_path = args.out_dir / "reaction_norm_loss_leverage_fold_summary.tsv"
    recovery_path = args.out_dir / "reaction_norm_loss_leverage_recovery_summary.tsv"
    leakage_path = args.out_dir / "reaction_norm_loss_leverage_split_qc.tsv"
    fold_summary.to_csv(fold_path, sep="\t", index=False)
    recovery_summary.to_csv(recovery_path, sep="\t", index=False)
    leakage.to_csv(leakage_path, sep="\t", index=False)
    provenance = {
        "status": status,
        "protocol_version": "reaction_norm_loss_leverage_v1",
        "selection_data": "identifiers_uncertainty_metadata_and_training_support_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "recovery_status_used_for_weighting": False,
        "fold_count": int(
            fold_summary[["scenario", "outer_fold", "inner_fold"]]
            .drop_duplicates()
            .shape[0]
        ),
        "candidate_count": len(policies),
        "checks": checks,
        "inputs": {
            str(path.resolve()): file_sha256(path)
            for path in (
                args.ledger,
                args.readiness_ledger,
                args.split_manifest,
                args.split_contract,
                args.loss_balance_protocol,
            )
        },
        "artifacts": {
            str(path.resolve()): file_sha256(path)
            for path in (fold_path, recovery_path, leakage_path)
        },
    }
    if args.pedigree_parent_table is not None:
        provenance["inputs"][str(args.pedigree_parent_table.resolve())] = file_sha256(
            args.pedigree_parent_table
        )
    provenance_path = args.out_dir / "reaction_norm_loss_leverage_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, allow_nan=False))
    if status != "PASS":
        raise SystemExit("Reaction-norm loss-leverage audit failed")


if __name__ == "__main__":
    main()
