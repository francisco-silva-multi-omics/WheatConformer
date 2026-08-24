from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_common import file_identity, write_json
from .audit_information_attrition import git_commit, resolve


SELECTED_TRAITS = {
    "1000_GRAIN_WEIGHT",
    "ABOVE_GROUND_BIOMASS",
    "DAYS_TO_HEADING",
    "DAYS_TO_MATURITY",
    "GRAIN_YIELD",
    "PLANT_HEIGHT",
    "TEST_WEIGHT",
}

OPTIONAL_EVIDENCE_COLUMNS = (
    "weight_g_e",
    "var_g_e",
    "raw_var_g_e",
    "stabilized_var_g_e",
    "SE_g_e",
    "raw_weight_g_e",
    "source_weight_g_e",
    "weight_variance_imputed",
    "weight_variance_floored",
    "stage1_sigma2",
    "stage1_df_resid",
    "stage1_rank",
)


def normalized_text(values: pd.Series, *, upper: bool = False) -> pd.Series:
    result = values.fillna("").astype(str).str.strip()
    return result.str.upper() if upper else result


def normalized_environment(values: pd.Series) -> pd.Series:
    return normalized_text(values).str.replace(r"\s+", " ", regex=True)


def table_columns(path: Path) -> list[str]:
    if "".join(path.suffixes).lower().endswith(".parquet"):
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    return list(pd.read_csv(path, sep="\t", nrows=0).columns)


def read_table_columns(
    path: Path, requested: list[str], *, required: set[str]
) -> pd.DataFrame:
    available = set(table_columns(path))
    missing = sorted(required.difference(available))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    columns = [column for column in requested if column in available]
    if "".join(path.suffixes).lower().endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, sep="\t", usecols=columns, low_memory=False)


def numeric_status(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return pd.Series(
        np.select(
            [np.isfinite(numeric) & (numeric > 0), np.isfinite(numeric)],
            ["finite_positive", "finite_nonpositive"],
            default="missing_or_nonfinite",
        ),
        index=values.index,
        dtype=object,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Certify Stage-1 invalid-weight rows for leakage-safe inclusion without "
            "reading phenotype values or evaluation outcomes."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--readiness-ledger",
        type=Path,
        default=Path(
            "audit/stage1_signal_recovery_v1/stage1_recovery_readiness_ledger.parquet"
        ),
    )
    parser.add_argument(
        "--stage1-phenotypes",
        type=Path,
        default=Path("phenotypes/stage1_adjusted_phenotypes.parquet"),
    )
    parser.add_argument(
        "--alias-registry",
        type=Path,
        default=Path(
            "audit/stage1_environment_alias_recovery_v1/environment_alias_registry.tsv"
        ),
    )
    parser.add_argument(
        "--genotype-order",
        type=Path,
        default=Path(
            "genotype_panels/pedigree_canonical_v3/K_A_CANONICAL_V3_sample_order.tsv"
        ),
    )
    parser.add_argument("--genotype-order-column", default="sample_id")
    parser.add_argument(
        "--environment-order",
        type=Path,
        default=Path("environment/env_kernel_sample_order.tsv"),
    )
    parser.add_argument("--environment-order-column", default="env_id")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("audit/stage1_weight_recovery_v1")
    )
    args = parser.parse_args()

    root = args.root.resolve()
    paths = {
        "readiness_ledger": resolve(root, args.readiness_ledger),
        "stage1_phenotypes": resolve(root, args.stage1_phenotypes),
        "alias_registry": resolve(root, args.alias_registry),
        "genotype_order": resolve(root, args.genotype_order),
        "environment_order": resolve(root, args.environment_order),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    readiness = read_table_columns(
        paths["readiness_ledger"],
        [
            "canonical_observation_id",
            "canonical_germplasm_key",
            "env_kernel_id",
            "trait_name_canonical",
            "stage1_to_model_status",
            "recovery_readiness",
        ],
        required={
            "canonical_observation_id",
            "recovery_readiness",
        },
    )
    candidate_ids = set(
        normalized_text(
            readiness.loc[
                readiness["recovery_readiness"].eq("P3_REPAIR_WEIGHT_METADATA"),
                "canonical_observation_id",
            ]
        )
    ).difference({""})
    if not candidate_ids:
        raise ValueError("The readiness ledger has no P3 weight-recovery candidates")

    required_stage1 = {
        "canonical_observation_id",
        "canonical_germplasm_key",
        "env_kernel_id",
        "trait_name_canonical",
        "stage1_model_status",
        "weight_g_e",
        "var_g_e",
    }
    stage1 = read_table_columns(
        paths["stage1_phenotypes"],
        [
            "canonical_observation_id",
            "canonical_germplasm_key",
            "env_kernel_id",
            "trait_name_canonical",
            "stage1_model_status",
            "n_plot_records",
            *OPTIONAL_EVIDENCE_COLUMNS,
        ],
        required=required_stage1,
    )
    stage1["canonical_observation_id"] = normalized_text(
        stage1["canonical_observation_id"]
    )
    candidates = stage1[
        stage1["canonical_observation_id"].isin(candidate_ids)
    ].copy()
    candidates["canonical_germplasm_key"] = normalized_text(
        candidates["canonical_germplasm_key"]
    )
    candidates["env_kernel_id_source"] = normalized_text(candidates["env_kernel_id"])
    candidates["environment_alias_registry_source_id"] = normalized_environment(
        candidates["env_kernel_id"]
    )
    candidates["trait_name_canonical"] = normalized_text(
        candidates["trait_name_canonical"], upper=True
    )

    aliases = pd.read_csv(paths["alias_registry"], sep="\t", dtype=str)
    alias_required = {"source_env_id", "target_env_id", "mapping_status"}
    alias_missing = sorted(alias_required.difference(aliases.columns))
    if alias_missing:
        raise ValueError(f"Alias registry is missing columns: {alias_missing}")
    aliases["source_env_id"] = normalized_environment(aliases["source_env_id"])
    aliases["target_env_id"] = normalized_text(aliases["target_env_id"])
    if aliases["source_env_id"].duplicated().any():
        raise ValueError("Alias registry contains duplicate source environment IDs")
    if not aliases["mapping_status"].eq("ACCEPTED_ALIAS").all():
        raise ValueError("Alias registry contains non-accepted environment mappings")
    alias_lookup = dict(zip(aliases["source_env_id"], aliases["target_env_id"], strict=True))
    mapped_environment = candidates["environment_alias_registry_source_id"].map(
        alias_lookup
    )
    candidates["environment_alias_applied"] = mapped_environment.notna()
    candidates["env_kernel_id_resolved"] = mapped_environment.fillna(
        candidates["env_kernel_id_source"]
    )

    genotype_order = pd.read_csv(paths["genotype_order"], sep="\t", dtype=str)
    environment_order = pd.read_csv(paths["environment_order"], sep="\t", dtype=str)
    if args.genotype_order_column not in genotype_order:
        raise ValueError("Genotype order is missing the requested ID column")
    if args.environment_order_column not in environment_order:
        raise ValueError("Environment order is missing the requested ID column")
    genotype_ids = set(normalized_text(genotype_order[args.genotype_order_column])).difference(
        {""}
    )
    environment_ids = set(
        normalized_text(environment_order[args.environment_order_column])
    ).difference({""})
    candidates["genotype_order_available"] = candidates[
        "canonical_germplasm_key"
    ].isin(genotype_ids)
    candidates["environment_order_available"] = candidates[
        "env_kernel_id_resolved"
    ].isin(environment_ids)
    candidates["source_weight_status"] = numeric_status(candidates["weight_g_e"])
    candidates["source_variance_status"] = numeric_status(candidates["var_g_e"])
    candidates["weight_recovery_method"] = np.where(
        candidates["source_variance_status"].eq("finite_positive"),
        "fold_local_training_transform_from_source_variance",
        "fold_local_training_variance_imputation",
    )
    candidates["weight_recovery_decision"] = np.where(
        candidates["genotype_order_available"]
        & candidates["environment_order_available"]
        & candidates["trait_name_canonical"].isin(SELECTED_TRAITS)
        & ~candidates["source_weight_status"].eq("finite_positive"),
        "ACCEPT_FOLD_LOCAL_WEIGHT_RECOVERY",
        "REJECT_WEIGHT_RECOVERY",
    )
    candidates["input_weight_preserved_unmodified"] = True
    candidates["static_weight_imputation_allowed"] = False
    candidates["uniform_ledger_weight_required"] = True

    duplicate_candidate_rows = candidates["canonical_observation_id"].duplicated(
        keep=False
    )
    checks = {
        "readiness_candidate_ids_unique_nonempty": bool(
            candidate_ids
            and not normalized_text(
                readiness.loc[
                    readiness["recovery_readiness"].eq(
                        "P3_REPAIR_WEIGHT_METADATA"
                    ),
                    "canonical_observation_id",
                ]
            ).duplicated().any()
        ),
        "all_candidates_found_exactly_once_in_stage1": bool(
            set(candidates["canonical_observation_id"]) == candidate_ids
            and not duplicate_candidate_rows.any()
        ),
        "all_source_weights_invalid": bool(
            ~candidates["source_weight_status"].eq("finite_positive").any()
        ),
        "all_candidate_genotypes_in_canonical_order": bool(
            candidates["genotype_order_available"].all()
        ),
        "all_candidate_environments_resolve_to_global_order": bool(
            candidates["environment_order_available"].all()
        ),
        "all_candidates_are_frozen_traits": bool(
            candidates["trait_name_canonical"].isin(SELECTED_TRAITS).all()
        ),
        "all_candidates_accepted_by_frozen_metadata_rules": bool(
            candidates["weight_recovery_decision"]
            .eq("ACCEPT_FOLD_LOCAL_WEIGHT_RECOVERY")
            .all()
        ),
        "phenotype_values_unread": True,
        "outer_test_metrics_unread": True,
        "final_holdout_outcomes_unread": True,
        "kernels_unchanged": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates.sort_values("canonical_observation_id", kind="stable").to_csv(
        out_dir / "stage1_weight_recovery_registry.tsv", sep="\t", index=False
    )
    (
        candidates.groupby(
            [
                "trait_name_canonical",
                "source_weight_status",
                "source_variance_status",
                "weight_recovery_method",
                "weight_recovery_decision",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            stage1_rows=("canonical_observation_id", "size"),
            unique_genotypes=("canonical_germplasm_key", "nunique"),
            unique_environments=("env_kernel_id_resolved", "nunique"),
        )
        .reset_index()
        .to_csv(
            out_dir / "stage1_weight_recovery_summary.tsv", sep="\t", index=False
        )
    )
    pd.DataFrame(
        [
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "detail": "",
            }
            for name, passed in checks.items()
        ]
    ).to_csv(out_dir / "stage1_weight_recovery_checks.tsv", sep="\t", index=False)

    code_root = Path(__file__).resolve().parents[1]
    provenance = {
        "status": status,
        "protocol_version": "stage1_fold_local_weight_recovery_v1",
        "selection_data": "identifiers_uncertainty_metadata_and_kernel_orders_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "kernels_modified": False,
        "weight_policy": {
            "source_weight_preserved": True,
            "static_imputation_allowed": False,
            "ledger_weight_power": 0.0,
            "ledger_weight": "uniform",
            "training_weight_statistics": "inner_training_partition_only",
            "missing_or_nonpositive_variance": "training_partition_trait_quantile",
        },
        "counts": {
            "candidate_rows": len(candidates),
            "finite_positive_source_variance_rows": int(
                candidates["source_variance_status"].eq("finite_positive").sum()
            ),
            "training_only_variance_imputation_rows": int(
                (~candidates["source_variance_status"].eq("finite_positive")).sum()
            ),
            "accepted_rows": int(
                candidates["weight_recovery_decision"]
                .eq("ACCEPT_FOLD_LOCAL_WEIGHT_RECOVERY")
                .sum()
            ),
        },
        "checks": checks,
        "code_root": str(code_root),
        "git_commit": git_commit(code_root),
        "inputs": {name: file_identity(path) for name, path in paths.items()},
    }
    write_json(out_dir / "stage1_weight_recovery_provenance.json", provenance)
    print(json.dumps(provenance, indent=2))
    if status != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"Stage-1 weight recovery audit failed: {failed}")


if __name__ == "__main__":
    main()
