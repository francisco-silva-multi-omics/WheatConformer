"""Inventory Phase-4 phenotype-design identifiability without fitting models."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


SELECTED_TRAITS = (
    "1000_GRAIN_WEIGHT",
    "ABOVE_GROUND_BIOMASS",
    "DAYS_TO_HEADING",
    "DAYS_TO_MATURITY",
    "GRAIN_YIELD",
    "PLANT_HEIGHT",
    "TEST_WEIGHT",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    canonical = args.canonical.resolve()
    if not canonical.is_file():
        raise FileNotFoundError(canonical)

    traits_sql = ",".join("?" for _ in SELECTED_TRAITS)
    con = duckdb.connect()
    group_sql = f"""
        WITH g AS (
          SELECT canonical_environment_id, canonical_trial_name, cycle, occ, loc_no,
                 country, loc_desc, accepted_canonical_trait, trait_name_original,
                 standardized_unit,
                 count(*) AS n_observations,
                 count(DISTINCT resolved_gid_v2) AS n_genotypes,
                 count(*) - count(DISTINCT resolved_gid_v2) AS repeat_surplus,
                 count(DISTINCT CASE WHEN trim(coalesce(rep, '')) <> '' THEN rep END) AS n_rep_levels,
                 count(DISTINCT CASE WHEN trim(coalesce(subblock, '')) <> '' THEN subblock END) AS n_block_levels,
                 count(DISTINCT CASE WHEN try_cast(plot AS DOUBLE) IS NOT NULL THEN try_cast(plot AS DOUBLE) END) AS n_numeric_plots,
                 sum(CASE WHEN trim(coalesce(rep, '')) <> '' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS rep_coverage,
                 sum(CASE WHEN trim(coalesce(subblock, '')) <> '' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS block_coverage,
                 sum(CASE WHEN trim(coalesce(plot, '')) <> '' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS plot_coverage,
                 sum(CASE WHEN try_cast(plot AS DOUBLE) IS NOT NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS numeric_plot_coverage,
                 count(DISTINCT CASE WHEN trim(coalesce(rep, '')) <> '' THEN resolved_gid_v2 || chr(31) || rep END)
                   - count(DISTINCT resolved_gid_v2) AS genotype_rep_surplus
          FROM read_parquet(?)
          WHERE row_disposition_v2 = 'ELIGIBLE_STAGE1_CONTRIBUTOR'
            AND accepted_canonical_trait IN ({traits_sql})
          GROUP BY ALL
        )
        SELECT *,
          (n_observations >= 6 AND n_genotypes >= 2) AS basic_model_eligible,
          (repeat_surplus > 0) AS any_repeat_available,
          (n_rep_levels >= 2 AND rep_coverage >= 0.8) AS rep_adjustment_identifiable,
          (n_block_levels >= 2 AND block_coverage >= 0.8) AS block_adjustment_identifiable,
          (n_numeric_plots >= 8 AND numeric_plot_coverage >= 0.8) AS plot_order_model_identifiable,
          FALSE AS independent_field_row_available,
          FALSE AS independent_field_column_available,
          FALSE AS ar1_by_ar1_identifiable,
          CASE WHEN n_numeric_plots >= 8 AND numeric_plot_coverage >= 0.8
               THEN 'PLOT_ORDER_1D_ONLY' ELSE 'NO_SPATIAL_COORDINATE_MODEL' END AS spatial_design_class,
          CASE WHEN repeat_surplus > 0 THEN 'REPLICATE_SPLIT_POSSIBLE_FOR_SUBSET'
               ELSE 'NO_WITHIN_ENTRY_REPLICATION' END AS ranking_ceiling_class
        FROM g
        ORDER BY canonical_trial_name, cycle, occ, loc_no, accepted_canonical_trait,
                 trait_name_original, standardized_unit
    """
    params = [str(canonical), *SELECTED_TRAITS]
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        escaped = str(output).replace("'", "''")
        con.execute(
            f"COPY ({group_sql}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)",
            params,
        )
        escaped_input = str(output).replace("'", "''")
        con.execute(f"CREATE TEMP VIEW phase4_inventory AS SELECT * FROM read_parquet('{escaped_input}')")
    else:
        con.execute(f"CREATE TEMP TABLE phase4_inventory AS {group_sql}", params)
    summary = con.execute(
        """
        SELECT count(*) AS groups,
               sum(n_observations) AS observations,
               sum(basic_model_eligible::INTEGER) AS basic_eligible,
               sum(any_repeat_available::INTEGER) AS any_repeat,
               sum(rep_adjustment_identifiable::INTEGER) AS rep_eligible,
               sum(block_adjustment_identifiable::INTEGER) AS block_eligible,
               sum(plot_order_model_identifiable::INTEGER) AS plot_order_eligible,
               median(n_observations) AS median_n,
               quantile_cont(n_observations, 0.9) AS p90_n,
               quantile_cont(n_observations, 0.99) AS p99_n,
               max(n_observations) AS max_n
        FROM phase4_inventory
        """
    ).fetchdf()
    print(summary.to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
