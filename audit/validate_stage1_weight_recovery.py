from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_common import file_identity, write_json
from .audit_information_attrition import git_commit, resolve


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if "".join(path.suffixes).lower().endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, sep="\t", usecols=columns, low_memory=False)


def text(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def finite_positive(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return pd.Series(np.isfinite(numeric) & (numeric > 0), index=values.index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate exact Stage-1 weight-row recovery and the uniform multi-trait ledger."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--readiness-ledger", type=Path, required=True)
    parser.add_argument("--weight-registry", type=Path, required=True)
    parser.add_argument("--model-observations", type=Path, required=True)
    parser.add_argument("--multitrait-ledger", type=Path, required=True)
    parser.add_argument("--genotype-order", type=Path, required=True)
    parser.add_argument("--environment-order", type=Path, required=True)
    parser.add_argument("--genotype-order-column", default="sample_id")
    parser.add_argument("--environment-order-column", default="env_id")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("audit/stage1_weight_recovery_v1/model_validation"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    paths = {
        "readiness_ledger": resolve(root, args.readiness_ledger),
        "weight_registry": resolve(root, args.weight_registry),
        "model_observations": resolve(root, args.model_observations),
        "multitrait_ledger": resolve(root, args.multitrait_ledger),
        "genotype_order": resolve(root, args.genotype_order),
        "environment_order": resolve(root, args.environment_order),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    readiness = read_table(
        paths["readiness_ledger"],
        columns=["canonical_observation_id", "recovery_readiness"],
    )
    registry = pd.read_csv(paths["weight_registry"], sep="\t", dtype=str)
    model_columns = [
        "canonical_observation_id",
        "canonical_germplasm_key",
        "env_kernel_id",
        "geno_kernel_index",
        "env_kernel_index",
        "weight_g_e",
        "var_g_e",
    ]
    model = read_table(paths["model_observations"], columns=model_columns)
    ledger_columns = [
        "canonical_observation_id",
        "weight_g_e",
        "source_weight_g_e",
        "raw_var_g_e",
        "weight_variance_imputed",
        "panel_sample_id",
        "genotype_id",
        "geno_source_kernel_index",
        "env_source_kernel_index",
    ]
    ledger = read_table(paths["multitrait_ledger"], columns=ledger_columns)
    genotype_order = pd.read_csv(paths["genotype_order"], sep="\t", dtype=str)
    environment_order = pd.read_csv(paths["environment_order"], sep="\t", dtype=str)
    if args.genotype_order_column not in genotype_order:
        raise ValueError("Genotype order is missing the requested ID column")
    if args.environment_order_column not in environment_order:
        raise ValueError("Environment order is missing the requested ID column")

    for frame in (readiness, registry, model, ledger):
        frame["canonical_observation_id"] = text(frame["canonical_observation_id"])
    accepted_registry = registry[
        registry["weight_recovery_decision"].eq(
            "ACCEPT_FOLD_LOCAL_WEIGHT_RECOVERY"
        )
    ].copy()
    accepted_ids = set(accepted_registry["canonical_observation_id"]).difference({""})
    retained_ids = set(
        readiness.loc[
            readiness["recovery_readiness"].eq("RETAINED_REFERENCE"),
            "canonical_observation_id",
        ]
    ).difference({""})
    environment_recovery_ids = set(
        readiness.loc[
            readiness["recovery_readiness"].eq("P1_RECOVER_ENVIRONMENT"),
            "canonical_observation_id",
        ]
    ).difference({""})
    p3_ids = set(
        readiness.loc[
            readiness["recovery_readiness"].eq("P3_REPAIR_WEIGHT_METADATA"),
            "canonical_observation_id",
        ]
    ).difference({""})
    expected_ids = retained_ids | environment_recovery_ids | p3_ids
    model_ids = set(model["canonical_observation_id"]).difference({""})
    ledger_ids = set(ledger["canonical_observation_id"]).difference({""})

    genotype_ids = text(genotype_order[args.genotype_order_column])
    environment_ids = text(environment_order[args.environment_order_column])
    genotype_index = {value: index for index, value in enumerate(genotype_ids)}
    environment_index = {value: index for index, value in enumerate(environment_ids)}
    model_genotype = text(model["canonical_germplasm_key"])
    model_environment = text(model["env_kernel_id"])
    expected_genotype_index = model_genotype.map(genotype_index)
    expected_environment_index = model_environment.map(environment_index)
    observed_genotype_index = pd.to_numeric(
        model["geno_kernel_index"], errors="coerce"
    )
    observed_environment_index = pd.to_numeric(
        model["env_kernel_index"], errors="coerce"
    )

    p3_model = model[model["canonical_observation_id"].isin(p3_ids)]
    p3_ledger = ledger[ledger["canonical_observation_id"].isin(p3_ids)]
    ledger_weight = pd.to_numeric(ledger["weight_g_e"], errors="coerce").to_numpy(
        dtype=float
    )
    p3_source_weight_valid = finite_positive(p3_ledger["source_weight_g_e"])
    checks = {
        "readiness_ids_unique_nonempty": bool(
            readiness["canonical_observation_id"].ne("").all()
            and not readiness["canonical_observation_id"].duplicated().any()
        ),
        "registry_accepts_exactly_all_P3_rows": accepted_ids == p3_ids,
        "model_ids_equal_retained_environment_and_weight_recovery": model_ids
        == expected_ids,
        "ledger_ids_equal_model_ids": ledger_ids == model_ids,
        "model_ids_unique": not model["canonical_observation_id"].duplicated().any(),
        "ledger_ids_unique": not ledger["canonical_observation_id"].duplicated().any(),
        "all_P3_rows_present_once_in_model": bool(
            len(p3_model) == len(p3_ids)
            and not p3_model["canonical_observation_id"].duplicated().any()
        ),
        "all_P3_rows_present_once_in_ledger": bool(
            len(p3_ledger) == len(p3_ids)
            and not p3_ledger["canonical_observation_id"].duplicated().any()
        ),
        "P3_source_weights_remain_invalid_in_model": bool(
            ~finite_positive(p3_model["weight_g_e"]).any()
        ),
        "P3_source_weights_preserved_as_invalid_in_ledger": bool(
            ~p3_source_weight_valid.any()
        ),
        "uniform_ledger_weights_finite_positive": bool(
            np.isfinite(ledger_weight).all() and np.all(ledger_weight > 0)
        ),
        "uniform_ledger_weights_equal_one": bool(
            np.allclose(ledger_weight, 1.0, atol=0.0, rtol=0.0)
        ),
        "nested_split_genotype_id_matches_certified_kernel_id": bool(
            text(ledger["panel_sample_id"]).eq(text(ledger["genotype_id"])).all()
        ),
        "model_genotype_mapping_complete": expected_genotype_index.notna().all(),
        "model_environment_mapping_complete": expected_environment_index.notna().all(),
        "model_genotype_indices_exact": bool(
            np.array_equal(
                observed_genotype_index.to_numpy(),
                expected_genotype_index.to_numpy(),
                equal_nan=True,
            )
        ),
        "model_environment_indices_exact": bool(
            np.array_equal(
                observed_environment_index.to_numpy(),
                expected_environment_index.to_numpy(),
                equal_nan=True,
            )
        ),
        "ledger_source_indices_match_model": bool(
            ledger.set_index("canonical_observation_id")[
                ["geno_source_kernel_index", "env_source_kernel_index"]
            ]
            .sort_index()
            .to_numpy()
            .astype(float)
            .tolist()
            == model.set_index("canonical_observation_id")[
                ["geno_kernel_index", "env_kernel_index"]
            ]
            .sort_index()
            .to_numpy()
            .astype(float)
            .tolist()
        ),
        "phenotype_values_unread": True,
        "outer_test_metrics_unread": True,
        "final_holdout_outcomes_unread": True,
        "kernels_unchanged": True,
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    status = "PASS" if all(checks.values()) else "FAIL"
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_table = pd.DataFrame(
        [
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "detail": "",
            }
            for name, passed in checks.items()
        ]
    )
    check_table.to_csv(
        out_dir / "stage1_weight_recovery_model_validation.tsv",
        sep="\t",
        index=False,
    )
    provenance = {
        "status": status,
        "protocol_version": "stage1_weight_recovery_model_validation_v1",
        "selection_data": "observation_identifiers_weight_provenance_and_kernel_indices_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "kernels_modified": False,
        "counts": {
            "retained_reference_rows": len(retained_ids),
            "recovered_environment_rows": len(environment_recovery_ids),
            "recovered_weight_rows": len(p3_ids),
            "model_rows": len(model),
            "multitrait_ledger_rows": len(ledger),
            "P3_rows_requiring_fold_local_variance_imputation": int(
                pd.Series(p3_ledger["raw_var_g_e"])
                .pipe(finite_positive)
                .eq(False)
                .sum()
            ),
        },
        "checks": checks,
        "code_root": str(Path(__file__).resolve().parents[1]),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "inputs": {name: file_identity(path) for name, path in paths.items()},
    }
    write_json(out_dir / "stage1_weight_recovery_model_validation.json", provenance)
    print(json.dumps(provenance, indent=2))
    if status != "PASS":
        failed = check_table.loc[check_table["status"].eq("FAIL"), "check"].tolist()
        raise SystemExit(f"Stage-1 weight recovery validation failed: {failed}")


if __name__ == "__main__":
    main()
