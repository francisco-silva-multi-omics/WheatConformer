"""Build immutable Stage-1 v2 raw/canonical layers and row dispositions.

This is a diagnostic reconstruction. It never writes to a certified-v1 path and
fails if its versioned output directory already exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq


LAYER_VERSION = "stage1_v2_layers_2026_07_30_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def scalar(con: duckdb.DuckDBPyConnection, query: str) -> int:
    return int(con.execute(query).fetchone()[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-ledger", type=Path, required=True)
    parser.add_argument("--registries", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_input = args.raw_ledger.resolve()
    registry_dir = args.registries.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    raw_output = result_dir / "raw_observations_v2.parquet"
    canonical_output = result_dir / "canonical_observations_v2.parquet"
    ledger_output = result_dir / "row_disposition_ledger_v2.parquet"

    genotype_path = registry_dir / "genotype_alias_registry_v2.tsv"
    trait_path = registry_dir / "trait_alias_registry_v2.tsv"
    unit_path = registry_dir / "trait_unit_rules_v2.tsv"
    environment_path = registry_dir / "environment_alias_registry_v2.tsv"
    required = [raw_input, genotype_path, trait_path, unit_path, environment_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    con = duckdb.connect(str(result_dir / "phase3_layers.duckdb"))
    (result_dir / "duckdb_tmp").mkdir()
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("PRAGMA temp_directory=?", [str((result_dir / "duckdb_tmp").resolve())])

    raw_p = sql_path(raw_input)
    genotype_p = sql_path(genotype_path)
    trait_p = sql_path(trait_path)
    unit_p = sql_path(unit_path)
    environment_p = sql_path(environment_path)
    raw_out = sql_path(raw_output)
    canonical_out = sql_path(canonical_output)
    ledger_out = sql_path(ledger_output)

    con.execute(f"CREATE VIEW raw_source AS SELECT * FROM read_parquet('{raw_p}')")
    con.execute(
        f"CREATE TABLE genotype_registry AS SELECT * FROM read_csv('{genotype_p}', delim='\\t', header=true, all_varchar=true)"
    )
    con.execute(
        f"CREATE TABLE trait_registry AS SELECT * FROM read_csv('{trait_p}', delim='\\t', header=true, all_varchar=true)"
    )
    con.execute(
        f"CREATE TABLE unit_registry AS SELECT * FROM read_csv('{unit_p}', delim='\\t', header=true, all_varchar=true)"
    )
    con.execute(
        f"CREATE TABLE environment_registry AS SELECT * FROM read_csv('{environment_p}', delim='\\t', header=true, all_varchar=true)"
    )

    raw_rows = scalar(con, "SELECT count(*) FROM raw_source")
    checks = [
        ("genotype_registry", "trial_key, cycle_norm, CID_norm, SID_norm"),
        ("trait_registry", "trait_key"),
        ("unit_registry", "trait_key, raw_unit"),
        ("environment_registry", "source_env_id"),
    ]
    cardinality_rows: list[dict[str, object]] = []
    for table, keys in checks:
        rows = scalar(con, f"SELECT count(*) FROM {table}")
        distinct_keys = scalar(con, f"SELECT count(*) FROM (SELECT DISTINCT {keys} FROM {table})")
        duplicate_keys = rows - distinct_keys
        cardinality_rows.append({
            "join_or_assertion": table,
            "expected_cardinality": "unique right-side key (m:1 join)",
            "left_rows_before": raw_rows,
            "right_rows": rows,
            "right_distinct_keys": distinct_keys,
            "duplicate_right_keys": duplicate_keys,
            "left_rows_after": "",
            "status": "PASS" if duplicate_keys == 0 else "FAIL",
        })
        if duplicate_keys:
            raise RuntimeError(f"{table} has {duplicate_keys} duplicate right-side keys")

    con.execute(
        f"""
        COPY (
          SELECT *, '{LAYER_VERSION}'::VARCHAR AS raw_layer_version,
                 '{sha256(raw_input)}'::VARCHAR AS immutable_input_sha256
          FROM raw_source
        ) TO '{raw_out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )

    con.execute(
        """
        CREATE TABLE joined AS
        SELECT
          r.*,
          'CAN2_' || substr(sha256(r.raw_source_row_id), 1, 24) AS canonical_row_id,
          CASE
            WHEN regexp_full_match(upper(trim(coalesce(r.raw_gid, ''))), '^(GID)?[0-9]+(\\.0)?$')
              THEN regexp_replace(regexp_replace(upper(trim(r.raw_gid)), '^GID', ''), '\\.0$', '')
            ELSE ''
          END AS raw_gid_normalized_v2,
          coalesce(g.accepted_gid, '') AS registry_accepted_gid,
          coalesce(g.registry_decision, 'UNRESOLVED_NO_REGISTRY_KEY') AS genotype_registry_decision,
          coalesce(g.registry_version, '') AS genotype_registry_version,
          coalesce(t.accepted_canonical_trait, '') AS accepted_canonical_trait,
          coalesce(t.accepted_standard_unit, '') AS accepted_standard_unit,
          coalesce(t.trait_alias_decision, 'UNRESOLVED_NO_TRAIT_KEY') AS trait_alias_decision_v2,
          coalesce(t.registry_version, '') AS trait_registry_version,
          try_cast(u.scale AS DOUBLE) AS unit_scale,
          try_cast(u.offset AS DOUBLE) AS unit_offset,
          coalesce(u.unit_rule, 'UNRESOLVED_NO_UNIT_RULE') AS unit_rule,
          coalesce(u.unit_rule_version, '') AS unit_rule_version,
          CASE WHEN e.alias_decision = 'ACCEPT' THEN e.target_env_id ELSE r.env_kernel_id END AS canonical_environment_id,
          CASE WHEN e.alias_decision = 'ACCEPT' THEN e.target_trial_name ELSE r.trial_name END AS canonical_trial_name,
          coalesce(e.alias_decision, 'NO_ALIAS_REQUIRED') AS environment_alias_decision,
          coalesce(e.registry_version, '') AS environment_registry_version
        FROM raw_source r
        LEFT JOIN genotype_registry g
          ON r.trial_key = g.trial_key
         AND r.cycle = g.cycle_norm
         AND r.CID_normalized = g.CID_norm
         AND r.SID_normalized = g.SID_norm
        LEFT JOIN trait_registry t ON r.trait_key = t.trait_key
        LEFT JOIN unit_registry u
          ON r.trait_key = u.trait_key
         AND coalesce(r.raw_unit, '') = coalesce(u.raw_unit, '')
        LEFT JOIN environment_registry e ON r.env_kernel_id = e.source_env_id
        """
    )
    joined_rows = scalar(con, "SELECT count(*) FROM joined")
    cardinality_rows.append({
        "join_or_assertion": "raw_left_join_all_registries",
        "expected_cardinality": "m:1; no row multiplication or loss",
        "left_rows_before": raw_rows,
        "right_rows": "",
        "right_distinct_keys": "",
        "duplicate_right_keys": "",
        "left_rows_after": joined_rows,
        "status": "PASS" if joined_rows == raw_rows else "FAIL",
    })
    if joined_rows != raw_rows:
        raise RuntimeError(f"Registry joins changed row count: {raw_rows} -> {joined_rows}")

    con.execute(
        """
        CREATE TABLE resolved AS
        SELECT *,
          CASE
            WHEN raw_gid_normalized_v2 != '' AND registry_accepted_gid != ''
                 AND raw_gid_normalized_v2 != registry_accepted_gid THEN ''
            WHEN raw_gid_normalized_v2 != '' THEN raw_gid_normalized_v2
            WHEN registry_accepted_gid != '' THEN registry_accepted_gid
            ELSE ''
          END AS resolved_gid_v2,
          CASE
            WHEN raw_gid_normalized_v2 != '' AND registry_accepted_gid != ''
                 AND raw_gid_normalized_v2 != registry_accepted_gid THEN 'AMBIGUOUS_RAW_GID_REGISTRY_CONFLICT'
            WHEN raw_gid_normalized_v2 != '' AND registry_accepted_gid = raw_gid_normalized_v2 THEN 'ACCEPT_RAW_GID_REGISTRY_CONCORDANT'
            WHEN raw_gid_normalized_v2 != '' THEN 'ACCEPT_EXACT_RAW_GID'
            WHEN registry_accepted_gid != '' THEN 'ACCEPT_VERSIONED_REGISTRY_GID'
            WHEN starts_with(genotype_registry_decision, 'AMBIGUOUS') THEN genotype_registry_decision
            ELSE 'UNRESOLVED_NO_ACCEPTED_GID'
          END AS genotype_resolution_status_v2,
          CASE WHEN unit_scale IS NOT NULL AND unit_offset IS NOT NULL AND numeric_value_finite
               THEN numeric_value * unit_scale + unit_offset ELSE NULL END AS value_standardized,
          CASE WHEN unit_scale IS NOT NULL AND unit_offset IS NOT NULL
               THEN accepted_standard_unit ELSE '' END AS standardized_unit,
          coalesce(canonical_environment_id, '') || '|' || coalesce(accepted_canonical_trait, '') || '|' ||
          coalesce(CASE
            WHEN raw_gid_normalized_v2 != '' AND registry_accepted_gid != '' AND raw_gid_normalized_v2 != registry_accepted_gid THEN ''
            WHEN raw_gid_normalized_v2 != '' THEN raw_gid_normalized_v2
            WHEN registry_accepted_gid != '' THEN registry_accepted_gid ELSE '' END, '') || '|' ||
          coalesce(rep, '') || '|' || coalesce(subblock, '') || '|' || coalesce(plot, '') AS semantic_plot_components
        FROM joined
        """
    )

    con.execute(
        """
        CREATE TABLE pre_duplicate AS
        SELECT *,
          CASE
            WHEN NOT numeric_parse_pass OR NOT numeric_value_finite THEN 'EXCLUDED_NUMERIC_PARSE_OR_NONFINITE'
            WHEN starts_with(genotype_resolution_status_v2, 'AMBIGUOUS') THEN 'EXCLUDED_AMBIGUOUS_GENOTYPE_IDENTITY'
            WHEN resolved_gid_v2 = '' THEN 'EXCLUDED_UNRESOLVED_GENOTYPE_IDENTITY'
            WHEN trait_alias_decision_v2 != 'ACCEPT_UNIQUE_TRAIT_UNIT' OR accepted_canonical_trait = '' THEN 'EXCLUDED_AMBIGUOUS_TRAIT_ALIAS'
            WHEN unit_scale IS NULL OR unit_offset IS NULL OR standardized_unit = '' THEN 'EXCLUDED_UNRESOLVED_UNIT_STANDARDIZATION'
            WHEN coalesce(canonical_trial_name, '') = '' OR coalesce(cycle, '') = ''
              OR coalesce(occ, '') = '' OR coalesce(loc_no, '') = '' THEN 'EXCLUDED_INCOMPLETE_ENVIRONMENT_KEY'
            ELSE 'PRE_DUPLICATE_ELIGIBLE'
          END AS pre_duplicate_disposition,
          row_number() OVER (
            PARTITION BY provisional_raw_source_row_id
            ORDER BY source_file, source_member, source_physical_row, raw_source_row_id
          ) AS source_copy_rank,
          count(*) OVER (PARTITION BY provisional_raw_source_row_id) AS source_copy_rows,
          'PLT2_' || substr(sha256(semantic_plot_components), 1, 24) AS semantic_plot_key_v2
        FROM resolved
        """
    )

    con.execute(
        """
        CREATE TABLE plot_stats AS
        SELECT semantic_plot_key_v2,
               count(*) AS representative_rows,
               count(DISTINCT value_standardized) AS distinct_standardized_values
        FROM pre_duplicate
        WHERE pre_duplicate_disposition = 'PRE_DUPLICATE_ELIGIBLE'
          AND source_copy_rank = 1
          AND coalesce(plot, '') != ''
        GROUP BY semantic_plot_key_v2
        """
    )
    con.execute(
        """
        CREATE TABLE adjudicated AS
        SELECT p.*,
          coalesce(s.representative_rows, 1) AS plot_representative_rows,
          coalesce(s.distinct_standardized_values, 1) AS plot_distinct_values,
          row_number() OVER (
            PARTITION BY p.semantic_plot_key_v2
            ORDER BY p.source_copy_rank, p.source_file, p.source_member, p.source_physical_row, p.raw_source_row_id
          ) AS plot_representative_rank
        FROM pre_duplicate p
        LEFT JOIN plot_stats s USING (semantic_plot_key_v2)
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            a.* EXCLUDE (semantic_plot_components, pre_duplicate_disposition),
            'GID' || resolved_gid_v2 AS canonical_germplasm_key,
            CASE
              WHEN pre_duplicate_disposition != 'PRE_DUPLICATE_ELIGIBLE' THEN pre_duplicate_disposition
              WHEN source_copy_rank > 1 THEN 'EXCLUDED_SOURCE_COPY_DUPLICATE'
              WHEN coalesce(plot, '') != '' AND plot_distinct_values > 1 THEN 'EXCLUDED_CONFLICTING_NONEMPTY_PLOT'
              WHEN coalesce(plot, '') != '' AND plot_representative_rows > 1 AND plot_representative_rank > 1
                THEN 'EXCLUDED_CONCORDANT_NONEMPTY_PLOT_DUPLICATE'
              ELSE 'ELIGIBLE_STAGE1_CONTRIBUTOR'
            END AS row_disposition_v2,
            CASE
              WHEN trim(both ';' from
                (CASE WHEN numeric_zero THEN 'RAW_VALUE_ZERO;' ELSE '' END) ||
                (CASE WHEN numeric_sentinel_candidate THEN 'NUMERIC_SENTINEL_CANDIDATE;' ELSE '' END) ||
                (CASE WHEN genotype_resolution_status_v2 = 'ACCEPT_VERSIONED_REGISTRY_GID' THEN 'GID_RECOVERED_FROM_VERSIONED_REGISTRY;' ELSE '' END) ||
                (CASE WHEN genotype_resolution_status_v2 = 'ACCEPT_RAW_GID_REGISTRY_CONCORDANT' THEN 'RAW_GID_REGISTRY_CONCORDANT;' ELSE '' END) ||
                (CASE WHEN environment_alias_decision = 'ACCEPT' THEN 'ENVIRONMENT_ALIAS_APPLIED;' ELSE '' END) ||
                (CASE WHEN unit_rule = 'ASSUME_TRAIT_STANDARD_UNIT_RAW_BLANK' THEN 'RAW_UNIT_BLANK_STANDARD_UNIT_ASSUMED;' ELSE '' END) ||
                (CASE WHEN unit_scale IS NOT NULL AND unit_scale != 1 THEN 'UNIT_NUMERICALLY_CONVERTED;' ELSE '' END) ||
                (CASE WHEN coalesce(plot, '') = '' AND pre_duplicate_disposition = 'PRE_DUPLICATE_ELIGIBLE' THEN 'PLOT_ID_MISSING_REPLICATE_PRESERVED;' ELSE '' END) ||
                (CASE WHEN source_copy_rows > 1 THEN 'BYTE_IDENTICAL_SOURCE_COPY_GROUP;' ELSE '' END) ||
                (CASE WHEN coalesce(plot, '') != '' AND plot_distinct_values > 1 THEN 'CONFLICTING_NONEMPTY_PLOT_VALUES;' ELSE '' END) ||
                (CASE WHEN identifier_normalization_changed THEN 'IDENTIFIER_NORMALIZATION_APPLIED;' ELSE '' END)
              ) = '' THEN 'NONE'
              ELSE trim(both ';' from
                (CASE WHEN numeric_zero THEN 'RAW_VALUE_ZERO;' ELSE '' END) ||
                (CASE WHEN numeric_sentinel_candidate THEN 'NUMERIC_SENTINEL_CANDIDATE;' ELSE '' END) ||
                (CASE WHEN genotype_resolution_status_v2 = 'ACCEPT_VERSIONED_REGISTRY_GID' THEN 'GID_RECOVERED_FROM_VERSIONED_REGISTRY;' ELSE '' END) ||
                (CASE WHEN genotype_resolution_status_v2 = 'ACCEPT_RAW_GID_REGISTRY_CONCORDANT' THEN 'RAW_GID_REGISTRY_CONCORDANT;' ELSE '' END) ||
                (CASE WHEN environment_alias_decision = 'ACCEPT' THEN 'ENVIRONMENT_ALIAS_APPLIED;' ELSE '' END) ||
                (CASE WHEN unit_rule = 'ASSUME_TRAIT_STANDARD_UNIT_RAW_BLANK' THEN 'RAW_UNIT_BLANK_STANDARD_UNIT_ASSUMED;' ELSE '' END) ||
                (CASE WHEN unit_scale IS NOT NULL AND unit_scale != 1 THEN 'UNIT_NUMERICALLY_CONVERTED;' ELSE '' END) ||
                (CASE WHEN coalesce(plot, '') = '' AND pre_duplicate_disposition = 'PRE_DUPLICATE_ELIGIBLE' THEN 'PLOT_ID_MISSING_REPLICATE_PRESERVED;' ELSE '' END) ||
                (CASE WHEN source_copy_rows > 1 THEN 'BYTE_IDENTICAL_SOURCE_COPY_GROUP;' ELSE '' END) ||
                (CASE WHEN coalesce(plot, '') != '' AND plot_distinct_values > 1 THEN 'CONFLICTING_NONEMPTY_PLOT_VALUES;' ELSE '' END) ||
                (CASE WHEN identifier_normalization_changed THEN 'IDENTIFIER_NORMALIZATION_APPLIED;' ELSE '' END)
              )
            END AS quality_flags_v2,
            '{LAYER_VERSION}'::VARCHAR AS canonical_layer_version
          FROM adjudicated a
        ) TO '{canonical_out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    con.execute(f"CREATE VIEW canonical AS SELECT * FROM read_parquet('{canonical_out}')")
    canonical_rows = scalar(con, "SELECT count(*) FROM canonical")
    canonical_ids = scalar(con, "SELECT count(DISTINCT canonical_row_id) FROM canonical")
    raw_ids = scalar(con, "SELECT count(DISTINCT raw_source_row_id) FROM canonical")
    if canonical_rows != raw_rows or canonical_ids != raw_rows or raw_ids != raw_rows:
        raise RuntimeError(
            f"Canonical conservation/ID uniqueness failed: raw={raw_rows}, canonical={canonical_rows}, "
            f"canonical_ids={canonical_ids}, raw_ids={raw_ids}"
        )
    cardinality_rows.append({
        "join_or_assertion": "canonical_row_conservation_and_ids",
        "expected_cardinality": "exactly one unique canonical ID per raw row",
        "left_rows_before": raw_rows,
        "right_rows": canonical_rows,
        "right_distinct_keys": canonical_ids,
        "duplicate_right_keys": canonical_rows - canonical_ids,
        "left_rows_after": canonical_rows,
        "status": "PASS",
    })

    con.execute(
        f"""
        COPY (
          SELECT raw_source_row_id, canonical_row_id, source_file, source_file_sha256,
                 source_member, source_physical_row, all_rawdata_row_number,
                 trial_name, canonical_trial_name, cycle, occ, loc_no,
                 CID_raw, SID_raw, raw_gid, raw_gid_normalized_v2,
                 registry_accepted_gid, resolved_gid_v2, genotype_resolution_status_v2,
                 trait_name_original, trait_key, accepted_canonical_trait,
                 raw_unit, standardized_unit, raw_value_token, numeric_value,
                 value_standardized, rep, subblock, plot, semantic_plot_key_v2,
                 row_disposition_v2, quality_flags_v2, canonical_layer_version
          FROM canonical
        ) TO '{ledger_out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )

    disposition = con.execute(
        """
        SELECT row_disposition_v2, count(*) AS rows,
               count(DISTINCT source_file) AS source_files,
               count(DISTINCT trial_name || '|' || cycle) AS trial_cycles,
               count(DISTINCT accepted_canonical_trait) FILTER (WHERE accepted_canonical_trait != '') AS canonical_traits,
               count(DISTINCT resolved_gid_v2) FILTER (WHERE resolved_gid_v2 != '') AS resolved_gids
        FROM canonical GROUP BY row_disposition_v2 ORDER BY row_disposition_v2
        """
    ).fetch_df()
    disposition.to_csv(result_dir / "row_disposition_summary.tsv", sep="\t", index=False)

    source_recon = con.execute(
        """
        SELECT source_file, source_file_sha256,
               count(*) AS raw_rows,
               count(DISTINCT raw_source_row_id) AS unique_raw_ids,
               count(DISTINCT canonical_row_id) AS unique_canonical_ids,
               count(*) FILTER (WHERE numeric_parse_pass AND numeric_value_finite) AS numeric_rows,
               count(*) FILTER (WHERE row_disposition_v2 = 'ELIGIBLE_STAGE1_CONTRIBUTOR') AS stage1_contributor_rows,
               count(*) FILTER (WHERE starts_with(row_disposition_v2, 'EXCLUDED')) AS excluded_rows,
               count(*) FILTER (WHERE resolved_gid_v2 != '') AS rows_with_resolved_gid_v2
        FROM canonical GROUP BY source_file, source_file_sha256 ORDER BY source_file
        """
    ).fetch_df()
    source_recon["reconciliation_status"] = "PASS"
    source_recon.to_csv(result_dir / "source_file_reconciliation.tsv", sep="\t", index=False)

    trial_coverage = con.execute(
        """
        SELECT trial_name, trial_key, cycle,
               count(*) AS raw_rows,
               count(*) FILTER (WHERE numeric_parse_pass AND numeric_value_finite) AS numeric_rows,
               count(*) FILTER (WHERE numeric_parse_pass AND numeric_value_finite AND resolved_gid_v2 != '') AS numeric_rows_with_gid,
               count(*) FILTER (WHERE numeric_parse_pass AND numeric_value_finite AND resolved_gid_v2 = '') AS numeric_rows_without_gid,
               count(*) FILTER (WHERE genotype_resolution_status_v2 = 'ACCEPT_VERSIONED_REGISTRY_GID') AS rows_gid_recovered_registry,
               count(DISTINCT resolved_gid_v2) FILTER (WHERE resolved_gid_v2 != '') AS distinct_resolved_gids,
               CASE
                 WHEN count(*) FILTER (WHERE numeric_parse_pass AND numeric_value_finite) = 0 THEN 'NOT_APPLICABLE_NO_NUMERIC_PHENOTYPES'
                 WHEN count(*) FILTER (WHERE numeric_parse_pass AND numeric_value_finite AND resolved_gid_v2 != '') = 0 THEN 'FAIL_TRIAL_HAS_NO_MATCHING_GID'
                 WHEN count(*) FILTER (WHERE numeric_parse_pass AND numeric_value_finite AND resolved_gid_v2 = '') = 0 THEN 'PASS_ALL_NUMERIC_ROWS_HAVE_GID'
                 ELSE 'PARTIAL_SOME_NUMERIC_ROWS_WITHOUT_GID'
               END AS gid_coverage_status
        FROM canonical GROUP BY trial_name, trial_key, cycle ORDER BY trial_name, cycle
        """
    ).fetch_df()
    trial_coverage.to_csv(result_dir / "trial_gid_coverage.tsv", sep="\t", index=False)

    attrition = con.execute(
        """
        SELECT 'RAW_ROWS' AS step, count(*) AS rows FROM canonical
        UNION ALL SELECT 'NUMERIC_FINITE', count(*) FROM canonical WHERE numeric_parse_pass AND numeric_value_finite
        UNION ALL SELECT 'RESOLVED_GID', count(*) FROM canonical WHERE numeric_parse_pass AND numeric_value_finite AND resolved_gid_v2 != ''
        UNION ALL SELECT 'RESOLVED_TRAIT', count(*) FROM canonical WHERE numeric_parse_pass AND numeric_value_finite AND resolved_gid_v2 != '' AND trait_alias_decision_v2 = 'ACCEPT_UNIQUE_TRAIT_UNIT'
        UNION ALL SELECT 'RESOLVED_UNIT', count(*) FROM canonical WHERE numeric_parse_pass AND numeric_value_finite AND resolved_gid_v2 != '' AND trait_alias_decision_v2 = 'ACCEPT_UNIQUE_TRAIT_UNIT' AND unit_scale IS NOT NULL
        UNION ALL SELECT 'COMPLETE_ENVIRONMENT', count(*) FROM canonical WHERE numeric_parse_pass AND numeric_value_finite AND resolved_gid_v2 != '' AND trait_alias_decision_v2 = 'ACCEPT_UNIQUE_TRAIT_UNIT' AND unit_scale IS NOT NULL AND canonical_trial_name != '' AND cycle != '' AND occ != '' AND loc_no != ''
        UNION ALL SELECT 'STAGE1_CONTRIBUTORS', count(*) FROM canonical WHERE row_disposition_v2 = 'ELIGIBLE_STAGE1_CONTRIBUTOR'
        """
    ).fetch_df()
    attrition.insert(1, "step_order", range(1, len(attrition) + 1))
    attrition.to_csv(result_dir / "attrition_waterfall_v2.tsv", sep="\t", index=False)

    duplicate_audit = con.execute(
        """
        SELECT semantic_plot_key_v2, canonical_environment_id, resolved_gid_v2,
               accepted_canonical_trait, standardized_unit, rep, subblock, plot,
               count(*) AS rows, count(DISTINCT value_standardized) AS distinct_values,
               string_agg(DISTINCT row_disposition_v2, ';' ORDER BY row_disposition_v2) AS dispositions
        FROM canonical
        WHERE source_copy_rows > 1 OR (plot != '' AND plot_representative_rows > 1)
        GROUP BY ALL ORDER BY rows DESC, semantic_plot_key_v2
        """
    ).fetch_df()
    duplicate_audit.to_csv(result_dir / "duplicate_and_biological_replicate_ledger.tsv", sep="\t", index=False)

    pd.DataFrame(cardinality_rows).to_csv(result_dir / "join_cardinality_report_v2.tsv", sep="\t", index=False)
    raw_parquet_rows = pq.ParquetFile(raw_output).metadata.num_rows
    canonical_parquet_rows = pq.ParquetFile(canonical_output).metadata.num_rows
    ledger_parquet_rows = pq.ParquetFile(ledger_output).metadata.num_rows
    fail_trials = int((trial_coverage["gid_coverage_status"] == "FAIL_TRIAL_HAS_NO_MATCHING_GID").sum())
    partial_trials = int(trial_coverage["gid_coverage_status"].str.startswith("PARTIAL").sum())
    summary = {
        "status": "PASS_LAYERS_BUILT",
        "layer_version": LAYER_VERSION,
        "raw_input_rows": raw_rows,
        "raw_layer_rows": raw_parquet_rows,
        "canonical_layer_rows": canonical_parquet_rows,
        "row_disposition_ledger_rows": ledger_parquet_rows,
        "unique_raw_source_row_ids": raw_ids,
        "unique_canonical_row_ids": canonical_ids,
        "stage1_contributor_rows": int(disposition.loc[disposition["row_disposition_v2"] == "ELIGIBLE_STAGE1_CONTRIBUTOR", "rows"].sum()),
        "trial_cycles": len(trial_coverage),
        "trial_cycles_with_no_matching_gid": fail_trials,
        "trial_cycles_with_partial_gid_coverage": partial_trials,
        "files": {},
    }
    for path in [raw_output, canonical_output, ledger_output,
                 result_dir / "source_file_reconciliation.tsv",
                 result_dir / "trial_gid_coverage.tsv",
                 result_dir / "join_cardinality_report_v2.tsv"]:
        summary["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (result_dir / "layer_build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    con.close()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
