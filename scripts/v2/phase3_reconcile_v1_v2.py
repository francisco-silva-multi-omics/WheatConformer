"""Compare certified-v1 and reconstructed-v2 Stage-1 populations only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd


SELECTED_TRAITS = [
    "1000_GRAIN_WEIGHT", "ABOVE_GROUND_BIOMASS", "DAYS_TO_HEADING", "DAYS_TO_MATURITY",
    "GRAIN_YIELD", "PLANT_HEIGHT", "TEST_WEIGHT",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-stage1", type=Path, required=True)
    parser.add_argument("--v2-stage1", type=Path, required=True)
    parser.add_argument("--environment-aliases", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    con = duckdb.connect(str(result_dir / "reconciliation.duckdb"))
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA memory_limit='2GB'")
    v1 = str(args.v1_stage1.resolve()).replace("'", "''")
    v2 = str(args.v2_stage1.resolve()).replace("'", "''")
    aliases = str(args.environment_aliases.resolve()).replace("'", "''")
    trait_sql = ",".join(f"'{value}'" for value in SELECTED_TRAITS)
    con.execute(
        f"""
        CREATE TABLE aliases AS
        SELECT source_env_id, target_env_id
        FROM read_csv('{aliases}', delim='\\t', header=true, all_varchar=true)
        WHERE alias_decision='ACCEPT'
        """
    )
    duplicate_aliases = con.execute(
        "SELECT count(*) - count(DISTINCT source_env_id) FROM aliases"
    ).fetchone()[0]
    if duplicate_aliases:
        raise RuntimeError("Environment alias registry source key is not unique")
    con.execute(
        f"""
        CREATE TABLE v1_keys AS
        SELECT canonical_observation_id AS v1_stage1_id,
               coalesce(a.target_env_id, s.env_kernel_id) AS canonical_environment_id,
               s.resolved_gid, s.trait_name_canonical AS canonical_trait,
               s.trait_name_original AS original_trait, s.unit,
               sha256(coalesce(a.target_env_id, s.env_kernel_id) || '|' || s.resolved_gid || '|' ||
                      s.trait_name_canonical || '|' || s.trait_name_original || '|' || coalesce(s.unit,'')) AS population_key
        FROM read_parquet('{v1}') s LEFT JOIN aliases a ON s.env_kernel_id=a.source_env_id
        WHERE s.trait_name_canonical IN ({trait_sql})
        """
    )
    con.execute(
        f"""
        CREATE TABLE v2_keys AS
        SELECT stage1_v2_row_id AS v2_stage1_id, canonical_environment_id, resolved_gid,
               accepted_canonical_trait AS canonical_trait, trait_name_original AS original_trait,
               standardized_unit AS unit,
               sha256(canonical_environment_id || '|' || resolved_gid || '|' || accepted_canonical_trait || '|' ||
                      trait_name_original || '|' || coalesce(standardized_unit,'')) AS population_key
        FROM read_parquet('{v2}') WHERE accepted_canonical_trait IN ({trait_sql})
        """
    )
    for table in ["v1_keys", "v2_keys"]:
        duplicates = con.execute(f"SELECT count(*) - count(DISTINCT population_key) FROM {table}").fetchone()[0]
        if duplicates:
            raise RuntimeError(f"{table} has {duplicates} duplicate population keys")
    reconciliation = con.execute(
        """
        SELECT coalesce(v1.population_key, v2.population_key) AS population_key,
               v1.v1_stage1_id, v2.v2_stage1_id,
               coalesce(v2.canonical_environment_id, v1.canonical_environment_id) AS canonical_environment_id,
               coalesce(v2.resolved_gid, v1.resolved_gid) AS resolved_gid,
               coalesce(v2.canonical_trait, v1.canonical_trait) AS canonical_trait,
               coalesce(v2.original_trait, v1.original_trait) AS original_trait,
               coalesce(v2.unit, v1.unit) AS unit,
               CASE WHEN v1.population_key IS NOT NULL AND v2.population_key IS NOT NULL THEN 'MATCHED_V1_V2'
                    WHEN v1.population_key IS NOT NULL THEN 'V1_ONLY'
                    ELSE 'V2_ONLY' END AS population_status
        FROM v1_keys v1 FULL OUTER JOIN v2_keys v2 USING (population_key)
        ORDER BY canonical_trait, canonical_environment_id, resolved_gid, original_trait, unit
        """
    ).fetch_df()
    reconciliation.to_parquet(result_dir / "selected_stage1_population_key_reconciliation.parquet", index=False)
    by_trait = (
        reconciliation.groupby(["canonical_trait", "population_status"], dropna=False)
        .size().rename("rows").reset_index()
    )
    by_trait.to_csv(result_dir / "selected_stage1_before_after_by_trait.tsv", sep="\t", index=False)
    v1_counts = con.execute(
        "SELECT count(*), count(DISTINCT resolved_gid), count(DISTINCT canonical_environment_id), count(DISTINCT canonical_trait) FROM v1_keys"
    ).fetchone()
    v2_counts = con.execute(
        "SELECT count(*), count(DISTINCT resolved_gid), count(DISTINCT canonical_environment_id), count(DISTINCT canonical_trait) FROM v2_keys"
    ).fetchone()
    status_counts = reconciliation["population_status"].value_counts().to_dict()
    summary = {
        "status": "PASS_POPULATION_ONLY_RECONCILIATION",
        "comparison_scope": "Stage-1 population keys only; no outer-test or final-holdout outcomes",
        "v1_selected_rows": int(v1_counts[0]), "v1_selected_genotypes": int(v1_counts[1]),
        "v1_selected_environments": int(v1_counts[2]), "v1_selected_traits": int(v1_counts[3]),
        "v2_selected_rows": int(v2_counts[0]), "v2_selected_genotypes": int(v2_counts[1]),
        "v2_selected_environments": int(v2_counts[2]), "v2_selected_traits": int(v2_counts[3]),
        "matched_population_keys": int(status_counts.get("MATCHED_V1_V2", 0)),
        "v1_only_population_keys": int(status_counts.get("V1_ONLY", 0)),
        "v2_only_population_keys": int(status_counts.get("V2_ONLY", 0)),
        "v1_stage1_sha256": sha256(args.v1_stage1.resolve()),
        "v2_stage1_sha256": sha256(args.v2_stage1.resolve()),
        "outer_test_content_read": False, "final_holdout_content_read": False,
    }
    (result_dir / "stage1_v1_v2_population_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    con.close()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
