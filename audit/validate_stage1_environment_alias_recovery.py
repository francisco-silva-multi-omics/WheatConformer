from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_common import file_identity, write_json
from .audit_information_attrition import git_commit, resolve


MODEL_COLUMNS = (
    "canonical_observation_id",
    "canonical_germplasm_key",
    "env_kernel_id",
    "env_kernel_id_original",
    "environment_alias_applied",
    "environment_alias_mapping_status",
    "geno_kernel_index",
    "env_kernel_index",
)


def read_columns(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path, columns=list(columns))
    return pd.read_csv(path, sep="\t", usecols=list(columns), low_memory=False)


def normalized_ids(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def boolean_values(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    normalized = normalized_ids(values).str.lower()
    unexpected = sorted(set(normalized).difference({"true", "false", "1", "0"}))
    if unexpected:
        raise ValueError(f"Unexpected boolean values: {unexpected[:5]}")
    return normalized.isin({"true", "1"})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that an alias-aware Stage-1 model table recovers exactly the "
            "certified P1 environment rows without changing retained observations."
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
        "--alias-registry",
        type=Path,
        default=Path(
            "audit/stage1_environment_alias_recovery_v1/environment_alias_registry.tsv"
        ),
    )
    parser.add_argument("--model-observations", type=Path, required=True)
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
        "--out-dir",
        type=Path,
        default=Path("audit/stage1_environment_alias_recovery_v1/model_validation"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    paths = {
        "readiness_ledger": resolve(root, args.readiness_ledger),
        "alias_registry": resolve(root, args.alias_registry),
        "model_observations": resolve(root, args.model_observations),
        "genotype_order": resolve(root, args.genotype_order),
        "environment_order": resolve(root, args.environment_order),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    readiness = read_columns(
        paths["readiness_ledger"],
        (
            "canonical_observation_id",
            "env_kernel_id",
            "recovery_readiness",
        ),
    )
    model = read_columns(paths["model_observations"], MODEL_COLUMNS)
    aliases = pd.read_csv(paths["alias_registry"], sep="\t", dtype=str)
    genotype_order = pd.read_csv(paths["genotype_order"], sep="\t", dtype=str)
    environment_order = pd.read_csv(
        paths["environment_order"], sep="\t", dtype=str
    )
    for column, frame, label in (
        (args.genotype_order_column, genotype_order, "genotype order"),
        (args.environment_order_column, environment_order, "environment order"),
    ):
        if column not in frame.columns:
            raise ValueError(f"{label} is missing requested column: {column}")

    for column in ("canonical_observation_id", "env_kernel_id"):
        readiness[column] = normalized_ids(readiness[column])
    for column in (
        "canonical_observation_id",
        "canonical_germplasm_key",
        "env_kernel_id",
        "env_kernel_id_original",
        "environment_alias_mapping_status",
    ):
        model[column] = normalized_ids(model[column])
    model["environment_alias_applied"] = boolean_values(
        model["environment_alias_applied"]
    )
    aliases["source_env_id"] = normalized_ids(aliases["source_env_id"])
    aliases["target_env_id"] = normalized_ids(aliases["target_env_id"])
    if aliases["source_env_id"].duplicated().any():
        raise ValueError("Alias registry contains duplicate source environment IDs")
    if aliases["target_env_id"].duplicated().any():
        raise ValueError("Alias registry contains duplicate target environment IDs")
    if not aliases["mapping_status"].eq("ACCEPTED_ALIAS").all():
        raise ValueError("Alias registry contains a non-accepted mapping")

    retained_ids = set(
        readiness.loc[
            readiness["recovery_readiness"].eq("RETAINED_REFERENCE"),
            "canonical_observation_id",
        ]
    )
    recoverable_ids = set(
        readiness.loc[
            readiness["recovery_readiness"].eq("P1_RECOVER_ENVIRONMENT"),
            "canonical_observation_id",
        ]
    )
    excluded_ids = set(
        readiness.loc[
            ~readiness["recovery_readiness"].isin(
                {"RETAINED_REFERENCE", "P1_RECOVER_ENVIRONMENT"}
            ),
            "canonical_observation_id",
        ]
    )
    expected_ids = retained_ids | recoverable_ids
    model_ids = set(model["canonical_observation_id"])
    alias_rows = model[model["environment_alias_applied"]].copy()
    alias_model_ids = set(alias_rows["canonical_observation_id"])
    alias_mapping = dict(
        zip(aliases["source_env_id"], aliases["target_env_id"], strict=True)
    )
    expected_alias_targets = alias_rows["env_kernel_id_original"].map(alias_mapping)

    genotype_ids = normalized_ids(genotype_order[args.genotype_order_column])
    environment_ids = normalized_ids(
        environment_order[args.environment_order_column]
    )
    genotype_index = {value: index for index, value in enumerate(genotype_ids)}
    environment_index = {value: index for index, value in enumerate(environment_ids)}
    expected_genotype_indices = model["canonical_germplasm_key"].map(genotype_index)
    expected_environment_indices = model["env_kernel_id"].map(environment_index)
    observed_genotype_indices = pd.to_numeric(
        model["geno_kernel_index"], errors="coerce"
    )
    observed_environment_indices = pd.to_numeric(
        model["env_kernel_index"], errors="coerce"
    )

    checks = {
        "readiness_observation_ids_unique_nonempty": bool(
            readiness["canonical_observation_id"].ne("").all()
            and not readiness["canonical_observation_id"].duplicated().any()
        ),
        "model_observation_ids_unique_nonempty": bool(
            model["canonical_observation_id"].ne("").all()
            and not model["canonical_observation_id"].duplicated().any()
        ),
        "model_observations_equal_retained_plus_recoverable": model_ids
        == expected_ids,
        "all_retained_observations_preserved": retained_ids.issubset(model_ids),
        "all_P1_environment_observations_recovered": recoverable_ids.issubset(
            model_ids
        ),
        "nonrecoverable_observations_excluded": not bool(model_ids & excluded_ids),
        "alias_applied_exactly_to_P1_environment_observations": alias_model_ids
        == recoverable_ids,
        "alias_sources_are_certified": set(alias_rows["env_kernel_id_original"])
        == set(aliases["source_env_id"]),
        "alias_targets_match_registry": bool(
            expected_alias_targets.notna().all()
            and expected_alias_targets.eq(alias_rows["env_kernel_id"]).all()
        ),
        "unaliased_environment_ids_unchanged": bool(
            model.loc[~model["environment_alias_applied"], "env_kernel_id"]
            .eq(
                model.loc[
                    ~model["environment_alias_applied"], "env_kernel_id_original"
                ]
            )
            .all()
        ),
        "alias_status_consistent": bool(
            alias_rows["environment_alias_mapping_status"].eq("ACCEPTED_ALIAS").all()
            and model.loc[
                ~model["environment_alias_applied"],
                "environment_alias_mapping_status",
            ]
            .eq("NOT_APPLICABLE")
            .all()
        ),
        "genotype_order_mapping_complete": expected_genotype_indices.notna().all(),
        "environment_order_mapping_complete": expected_environment_indices.notna().all(),
        "genotype_indices_exact": bool(
            np.array_equal(
                observed_genotype_indices.to_numpy(),
                expected_genotype_indices.to_numpy(),
                equal_nan=True,
            )
        ),
        "environment_indices_exact": bool(
            np.array_equal(
                observed_environment_indices.to_numpy(),
                expected_environment_indices.to_numpy(),
                equal_nan=True,
            )
        ),
        "phenotype_values_unread": True,
        "outer_test_metrics_unread": True,
        "final_holdout_outcomes_unread": True,
        "kernels_unchanged": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    check_table = pd.DataFrame(
        [
            {
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "detail": "",
            }
            for check, passed in checks.items()
        ]
    )
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_table.to_csv(
        out_dir / "stage1_environment_alias_model_validation.tsv",
        sep="\t",
        index=False,
    )

    code_root = Path(__file__).resolve().parents[1]
    provenance = {
        "status": status,
        "protocol_version": "stage1_environment_alias_model_validation_v1",
        "selection_data": "observation_identifiers_alias_provenance_and_kernel_indices_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "kernels_modified": False,
        "checks": checks,
        "counts": {
            "retained_reference_rows": len(retained_ids),
            "recovered_environment_rows": len(recoverable_ids),
            "model_rows": len(model),
            "excluded_rows": len(excluded_ids),
            "alias_source_environments": aliases["source_env_id"].nunique(),
            "alias_target_environments": aliases["target_env_id"].nunique(),
        },
        "code_root": str(code_root),
        "git_commit": git_commit(code_root),
        "inputs": {name: file_identity(path) for name, path in paths.items()},
    }
    write_json(
        out_dir / "stage1_environment_alias_model_validation.json", provenance
    )
    print(
        json.dumps(
            {
                "status": status,
                **provenance["counts"],
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )
    if status != "PASS":
        failed = check_table.loc[check_table["status"].eq("FAIL"), "check"].tolist()
        raise SystemExit(f"Stage-1 environment alias model validation failed: {failed}")


if __name__ == "__main__":
    main()
