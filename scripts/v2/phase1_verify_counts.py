"""Verify Phase 1 headline counts from non-protected server artifacts.

Only identifier, trait, status, and mapping columns are loaded. Phenotype values,
locked outer-test outputs, and final-holdout artifacts are intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


SELECTED_TRAITS = {
    "1000_GRAIN_WEIGHT",
    "ABOVE_GROUND_BIOMASS",
    "DAYS_TO_HEADING",
    "DAYS_TO_MATURITY",
    "GRAIN_YIELD",
    "PLANT_HEIGHT",
    "TEST_WEIGHT",
}


def fail_if_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    fail_if_exists(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalized(series: pd.Series) -> pd.Series:
    return series.fillna("").astype("string").str.strip()


def first_present(columns: list[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"None of the expected columns {candidates} is present in {columns}")


def identifier_counts(path: Path, *, filter_selected_traits: bool) -> dict[str, object]:
    schema = pq.ParquetFile(path).schema_arrow
    columns = schema.names
    observation = first_present(columns, ("canonical_observation_id", "observation_id"))
    genotype = first_present(columns, ("canonical_germplasm_key", "genotype_id", "resolved_gid"))
    environment = first_present(columns, ("env_kernel_id", "environment_id", "env_id"))
    trait = first_present(columns, ("trait_name_canonical", "trait_id", "trait"))
    selected = [observation, genotype, environment, trait]
    if "environment_alias_applied" in columns:
        selected.append("environment_alias_applied")
    frame = pd.read_parquet(path, columns=selected)
    trait_values = normalized(frame[trait]).str.upper()
    if filter_selected_traits:
        frame = frame.loc[trait_values.isin(SELECTED_TRAITS)].copy()
        trait_values = normalized(frame[trait]).str.upper()
    observation_values = normalized(frame[observation])
    genotype_values = normalized(frame[genotype])
    environment_values = normalized(frame[environment])
    output: dict[str, object] = {
        "rows": len(frame),
        "unique_observations": observation_values.nunique(),
        "duplicate_observation_rows": int(observation_values.duplicated().sum()),
        "blank_observation_rows": int(observation_values.eq("").sum()),
        "unique_genotypes": genotype_values[genotype_values.ne("")].nunique(),
        "unique_environments": environment_values[environment_values.ne("")].nunique(),
        "unique_traits": trait_values[trait_values.ne("")].nunique(),
    }
    if "environment_alias_applied" in frame:
        alias = frame["environment_alias_applied"]
        if pd.api.types.is_bool_dtype(alias):
            alias_values = alias.fillna(False)
        else:
            alias_values = normalized(alias).str.upper().isin({"1", "TRUE", "YES", "Y"})
        output["environment_alias_applied_rows"] = int(alias_values.sum())
    return output


def schema_rows(name: str, path: Path) -> list[dict[str, object]]:
    parquet = pq.ParquetFile(path)
    return [
        {
            "artifact": name,
            "relative_path": path.as_posix(),
            "parquet_rows": parquet.metadata.num_rows,
            "row_groups": parquet.metadata.num_row_groups,
            "column_position": position,
            "column_name": field.name,
            "column_type": str(field.type),
        }
        for position, field in enumerate(parquet.schema_arrow)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle_root.resolve() / "artifacts"
    out_dir = args.out_dir.resolve()

    artifacts = {
        "selected_trait_attrition_ledger": bundle / "audit/information_attrition_v2/selected_trait_attrition_ledger.parquet",
        "stage1_adjusted_phenotypes": bundle / "phenotypes/stage1_adjusted_phenotypes.parquet",
        "stage1_alias_model_ready": bundle / "model_kernels/stage1_canonical_v3_environment_alias_v1/stage1_canonical_v3_environment_alias_v1_model_ready_stage1_observations.parquet",
        "stage1_alias_weight_model_ready": bundle / "model_kernels/stage1_canonical_v3_environment_alias_weight_v1/stage1_canonical_v3_environment_alias_weight_v1_model_ready_stage1_observations.parquet",
        "certified_multitrait_ledger": bundle / "model_kernels/multitrait_pedigree_env_uniform_tgw_certified/multitrait_pedigree_uniform_tgw_certified_observations.parquet",
        "recovered_multitrait_ledger": bundle / "model_kernels/multitrait_stage1_recovered_v1/multitrait_stage1_recovered_v1_observations.parquet",
    }
    for path in artifacts.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    schemas: list[dict[str, object]] = []
    direct_rows: list[dict[str, object]] = []
    counts: dict[str, dict[str, object]] = {}
    for name, path in artifacts.items():
        schemas.extend(schema_rows(name, path))
        result = identifier_counts(
            path,
            filter_selected_traits=name == "stage1_adjusted_phenotypes",
        )
        counts[name] = result
        direct_rows.append({"artifact": name, **result})

    weight_registry_path = bundle / "audit/stage1_weight_recovery_v1/stage1_weight_recovery_registry.tsv"
    weight_registry = pd.read_csv(weight_registry_path, sep="\t", low_memory=False)
    decision = normalized(weight_registry["weight_recovery_decision"])
    accepted_weight_rows = int(decision.eq("ACCEPT_FOLD_LOCAL_WEIGHT_RECOVERY").sum())

    alias_registry_path = bundle / "audit/stage1_environment_alias_recovery_v1/environment_alias_registry.tsv"
    alias_registry = pd.read_csv(alias_registry_path, sep="\t", dtype="string")

    observed = {
        "canonical_selected_trait_records": counts["selected_trait_attrition_ledger"]["rows"],
        "stage1_selected_trait_observations": counts["stage1_adjusted_phenotypes"]["rows"],
        "stage1_selected_trait_genotypes": counts["stage1_adjusted_phenotypes"]["unique_genotypes"],
        "stage1_selected_trait_environments": counts["stage1_adjusted_phenotypes"]["unique_environments"],
        "environment_alias_applied_rows": counts["stage1_alias_weight_model_ready"].get("environment_alias_applied_rows", -1),
        "fold_local_weight_recovery_rows": accepted_weight_rows,
    }
    expected = {
        "canonical_selected_trait_records": 2_022_291,
        "stage1_selected_trait_observations": 278_001,
        "stage1_selected_trait_genotypes": 5_253,
        "stage1_selected_trait_environments": 1_015,
        "environment_alias_applied_rows": 22_609,
        "fold_local_weight_recovery_rows": 59,
    }
    sources = {
        "canonical_selected_trait_records": "selected_trait_attrition_ledger.parquet metadata",
        "stage1_selected_trait_observations": "stage1_adjusted_phenotypes.parquet filtered to seven traits",
        "stage1_selected_trait_genotypes": "stage1_adjusted_phenotypes.parquet identifier columns",
        "stage1_selected_trait_environments": "stage1_adjusted_phenotypes.parquet identifier columns",
        "environment_alias_applied_rows": "environment_alias_weight model-ready observations alias flag",
        "fold_local_weight_recovery_rows": "stage1_weight_recovery_registry.tsv accepted decision",
    }
    expected_rows = [
        {
            "metric": metric,
            "expected": expected_value,
            "observed": observed[metric],
            "status": "PASS" if observed[metric] == expected_value else "FAIL",
            "direct_source": sources[metric],
        }
        for metric, expected_value in expected.items()
    ]

    write_tsv(
        out_dir / "artifact_schema_inventory.tsv",
        schemas,
        ["artifact", "relative_path", "parquet_rows", "row_groups", "column_position", "column_name", "column_type"],
    )
    direct_fields = [
        "artifact", "rows", "unique_observations", "duplicate_observation_rows",
        "blank_observation_rows", "unique_genotypes", "unique_environments",
        "unique_traits", "environment_alias_applied_rows",
    ]
    for row in direct_rows:
        row.setdefault("environment_alias_applied_rows", "")
    write_tsv(out_dir / "direct_artifact_counts.tsv", direct_rows, direct_fields)
    write_tsv(
        out_dir / "expected_vs_observed_counts.tsv",
        expected_rows,
        ["metric", "expected", "observed", "status", "direct_source"],
    )
    report = {
        "status": "PASS" if all(row["status"] == "PASS" for row in expected_rows) else "FAIL",
        "selected_traits": sorted(SELECTED_TRAITS),
        "expected": expected,
        "observed": observed,
        "alias_registry_rows": len(alias_registry),
        "weight_registry_rows": len(weight_registry),
        "protected_artifacts_read": False,
        "phenotype_value_columns_read": False,
    }
    report_path = out_dir / "count_verification.json"
    fail_if_exists(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit("One or more Phase 1 headline counts did not match")


if __name__ == "__main__":
    main()
