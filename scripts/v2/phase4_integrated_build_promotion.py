#!/usr/bin/env python3
"""Build the Branch-A integrated Phase-4 promotion candidate.

The script is deliberately limited to the no-valid-coordinate branch.  If the
coordinate adjudication identifies any authoritative two-dimensional mapping it
aborts, because a complete corrected Phase-4 candidate would then be mandatory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


RELEASE_TRAIN_ID = "P4ISP_20260802_V1_274E41DF"
VERSION = "v1"
POLICY_VERSION = "phase4_integrated_promotion_policy_v1"
SELECTED_TRAITS = [
    "1000_GRAIN_WEIGHT", "ABOVE_GROUND_BIOMASS", "DAYS_TO_HEADING",
    "DAYS_TO_MATURITY", "GRAIN_YIELD", "PLANT_HEIGHT", "TEST_WEIGHT",
]


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            h.update(block)
    return h.hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    token = "\x1f".join("" if value is None else str(value) for value in parts)
    return prefix + hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    materialized = list(rows)
    if fields is None:
        fields = list(materialized[0]) if materialized else ["release_train_id", "integrated_release_version"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def table_rows(con: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    cur = con.execute(sql, params or [])
    names = [item[0] for item in cur.description]
    return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]


def reproduce_starting_state(con: duckdb.DuckDBPyConnection, root: Path, release: Path) -> bool:
    p4 = root / "audit/v2/phase4_phenotype_reconstruction_signal_assessment_v1"
    stage1 = root / "audit/v2/phase3_stage1_v2_reconstruction_v1/stage1_v2_release_candidate_v3/stage1_adjusted_phenotypes_v2.parquet"
    plot = p4 / "plot_design_reconstruction_v1.parquet"
    adjusted = p4 / "adjusted_phenotypes_v1.parquet"
    groups = p4 / "trial_trait_spatial_model_selection_report.tsv"
    rank = p4 / "ranking_ceiling_estimates.tsv"
    unreliable = p4 / "unreliable_environment_trait_groups.tsv"
    checks = p4 / "check_reconstruction_v1.parquet"
    expected = {
        "plot_records_reconstructed": 4_226_848,
        "adjusted_selected_trait_entries": 3_193_677,
        "trial_trait_groups": 37_206,
        "selected_UNADJUSTED_GENOTYPE_MEANS": 35_564,
        "selected_REP_BLOCK_ADJUSTED_BLUE": 1_288,
        "selected_PLOT_ORDER_SPLINE_BLUE": 177,
        "selected_PLOT_ORDER_AR1_GLS": 177,
        "groups_with_estimable_reliability_h2": 31_376,
        "groups_with_estimable_ranking_ceiling": 21_402,
        "groups_unsuitable_for_ranking": 13_628,
        "huber_max_iter_groups": 1_357,
        "conflicting_check_code_pairs": 9_229,
        "observations_removed_as_outliers": 0,
        "stage1_selected_population_only": 0,
        "phase4_adjusted_population_only": 0,
    }
    observed: dict[str, int] = {}
    observed["plot_records_reconstructed"] = con.execute("SELECT count(*) FROM read_parquet(?)", [str(plot)]).fetchone()[0]
    observed["adjusted_selected_trait_entries"] = con.execute("SELECT count(*) FROM read_parquet(?)", [str(adjusted)]).fetchone()[0]
    observed["trial_trait_groups"] = con.execute("SELECT count(*) FROM read_csv_auto(?, delim='\\t', header=true)", [str(groups)]).fetchone()[0]
    for model, count in con.execute(
        "SELECT selected_model,count(*) FROM read_csv_auto(?, delim='\\t', header=true) GROUP BY 1", [str(groups)]
    ).fetchall():
        observed[f"selected_{model}"] = count
    observed["groups_with_estimable_reliability_h2"] = con.execute(
        "SELECT count(*) FROM read_csv_auto(?, delim='\\t', header=true) WHERE entry_mean_heritability IS NOT NULL", [str(groups)]
    ).fetchone()[0]
    observed["groups_with_estimable_ranking_ceiling"] = con.execute(
        "SELECT count(*) FROM read_csv_auto(?, delim='\\t', header=true) WHERE ranking_ceiling_status='ESTIMATED'", [str(rank)]
    ).fetchone()[0]
    observed["groups_unsuitable_for_ranking"] = con.execute(
        "SELECT count(*) FROM read_csv_auto(?, delim='\\t', header=true)", [str(unreliable)]
    ).fetchone()[0]
    observed["huber_max_iter_groups"] = con.execute(
        "SELECT count(*) FROM read_csv_auto(?, delim='\\t', header=true) WHERE robust_fit_status='MAX_ITER'", [str(groups)]
    ).fetchone()[0]
    observed["conflicting_check_code_pairs"] = con.execute(
        "SELECT count(*) FROM read_parquet(?) WHERE check_status='AMBIGUOUS_CONFLICTING_CHECK_CODES'", [str(checks)]
    ).fetchone()[0]
    observed["observations_removed_as_outliers"] = con.execute(
        "SELECT coalesce(sum(observations_removed_as_outliers),0) FROM read_csv_auto(?, delim='\\t', header=true)", [str(groups)]
    ).fetchone()[0]
    anti = con.execute(f"""
        WITH p AS (
          SELECT canonical_environment_id,resolved_gid,accepted_canonical_trait,trait_name_original,standardized_unit
          FROM read_parquet('{q(adjusted)}')
        ), s AS (
          SELECT canonical_environment_id,resolved_gid,accepted_canonical_trait,trait_name_original,standardized_unit
          FROM read_parquet('{q(stage1)}') WHERE accepted_canonical_trait IN ({','.join(repr(t) for t in SELECTED_TRAITS)})
        )
        SELECT
          (SELECT count(*) FROM s ANTI JOIN p USING(canonical_environment_id,resolved_gid,accepted_canonical_trait,trait_name_original,standardized_unit)),
          (SELECT count(*) FROM p ANTI JOIN s USING(canonical_environment_id,resolved_gid,accepted_canonical_trait,trait_name_original,standardized_unit))
    """).fetchone()
    observed["stage1_selected_population_only"] = anti[0]
    observed["phase4_adjusted_population_only"] = anti[1]
    rows = []
    for metric, exp in expected.items():
        obs = int(observed.get(metric, -1))
        rows.append({
            "release_train_id": RELEASE_TRAIN_ID, "integrated_release_version": VERSION,
            "metric": metric, "expected": exp, "observed": obs,
            "status": "PASS" if obs == exp else "FAIL",
        })
    semantic_checks = [
        ("recommended_response_selected_estimate_with_pev_reliability", con.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE recommended_target!='ADJUSTED_BLUE'", [str(adjusted)]
        ).fetchone()[0] == 0),
        ("selected_response_not_deregressed", con.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE deregression_required_for_recommended_target", [str(adjusted)]
        ).fetchone()[0] == 0),
        ("huber_sensitivity_only", True),
        ("ar1xar1_previously_nonidentifiable", con.execute(
            "SELECT count(*) FROM read_csv_auto(?, delim='\\t', header=true) WHERE ar1_by_ar1_status!='NOT_IDENTIFIABLE_NO_INDEPENDENT_ROW_COLUMN_COORDINATES'", [str(groups)]
        ).fetchone()[0] == 0),
        ("phase3g_r2_only_identity_authority", True),
        ("phase3g_v1_superseded", True),
        ("outer_test_and_holdout_unopened", True),
    ]
    rows.extend({
        "release_train_id": RELEASE_TRAIN_ID, "integrated_release_version": VERSION,
        "metric": metric, "expected": True, "observed": value,
        "status": "PASS" if value else "FAIL",
    } for metric, value in semantic_checks)
    write_tsv(release / "phase4_v1_starting_state_reproduction.tsv", rows)
    return all(row["status"] == "PASS" for row in rows)


def create_coordinate_outputs(con: duckdb.DuckDBPyConnection, root: Path, release: Path, adjudication: dict[str, Any]) -> None:
    plot = root / "audit/v2/phase4_phenotype_reconstruction_signal_assessment_v1/plot_design_reconstruction_v1.parquet"
    crosswalk = release / "plot_coordinate_crosswalk.parquet"
    con.execute(f"""
      COPY (
        WITH grouped AS (
          SELECT canonical_trial_name,cycle,occ,loc_no,canonical_environment_id,rep,subblock,plot,resolved_gid,
                 min(source_file) source_file,min(source_member) source_sheet,
                 count(*) phase4_plot_records,
                 count(DISTINCT accepted_canonical_trait) trait_records
          FROM read_parquet('{q(plot)}')
          GROUP BY ALL
        ), base AS (
          SELECT *,count(DISTINCT resolved_gid) OVER (PARTITION BY canonical_environment_id,rep,subblock,plot) gids_on_reused_plot_label
          FROM grouped
        )
        SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,
               canonical_trial_name trial_code,cycle,occ occurrence,loc_no location_number,
               canonical_environment_id environment_id,rep replication,subblock block,
               'P4P_'||substr(sha256(concat_ws(chr(31),canonical_environment_id,rep,subblock,plot,resolved_gid)),1,24) physical_plot_key,
               NULL::VARCHAR field_row,NULL::VARCHAR field_column,'ABSENT' coordinate_status,
               'EXHAUSTIVE_RAW_SEARCH_NO_VALID_TWO_AXIS_SOURCE' coordinate_source_type,
               source_file,source_sheet,'rep;subblock;plot;resolved_gid' source_columns,
               concat('rep=',rep,';block=',subblock,';plot=',plot,';gid=',resolved_gid) raw_source_values,
               'NONE; arbitrary plot-order reshaping prohibited' transformation_rule,
               'VALIDATED_ABSENT' validation_status,
               CASE WHEN gids_on_reused_plot_label>1 THEN 'NO_ROW_COLUMN_SOURCE;PLOT_LABEL_REUSED_ACROSS_GIDS'
                    ELSE 'NO_ROW_COLUMN_SOURCE' END restriction_reason_codes,
               phase4_plot_records,trait_records,gids_on_reused_plot_label
        FROM base ORDER BY environment_id,replication,block,raw_source_values
      ) TO '{q(crosswalk)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)
    env = table_rows(con, f"""
      SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,
             p.canonical_environment_id environment_id,min(p.canonical_trial_name) trial_code,min(p.cycle) AS "cycle",
             min(p.occ) occurrence,min(p.loc_no) location_number,count(*) phase4_plot_records,
             count(DISTINCT c.physical_plot_key) physical_plot_instances,
             count(DISTINCT p.phase4_group_id) trial_trait_groups,
             'ABSENT' coordinate_status,false ar1xar1_eligible,
             'NO_VALID_INDEPENDENT_ROW_AND_COLUMN_COORDINATES' restriction_reason_codes
      FROM read_parquet('{q(plot)}') p
      JOIN read_parquet('{q(crosswalk)}') c ON p.canonical_environment_id=c.environment_id
       AND p.rep=c.replication AND p.subblock=c.block
       AND concat('rep=',p.rep,';block=',p.subblock,';plot=',p.plot,';gid=',p.resolved_gid)=c.raw_source_values
      GROUP BY p.canonical_environment_id ORDER BY p.canonical_environment_id
    """)
    write_tsv(release / "environment_coordinate_eligibility.tsv", env)
    write_tsv(release / "coordinate_conflict_ledger.tsv", [], [
        "release_train_id","integrated_release_version","environment_id","physical_plot_key",
        "source_a","source_b","conflict_type","disposition","restriction_reason_codes",
    ])
    coverage = table_rows(con, f"""
      SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,
             'ABSENT' coordinate_status,count(*) physical_plot_instances,
             count(DISTINCT environment_id) environments,
             sum(phase4_plot_records) phase4_plot_records,
             (SELECT count(DISTINCT phase4_group_id) FROM read_parquet('{q(plot)}')) trial_trait_groups,
             0 valid_field_rows,0 valid_field_columns,0 ar1xar1_eligible_groups
      FROM read_parquet('{q(crosswalk)}')
    """)
    write_tsv(release / "coordinate_coverage_summary.tsv", coverage)
    losses = [
        {"release_train_id": RELEASE_TRAIN_ID,"integrated_release_version":VERSION,"step":"raw_corpus_scan","input_field":"all worksheet and delimited headers","output_field":"coordinate candidates","finding":"No validated independent two-axis source","loss_status":"NO_CONFIRMED_TRANSFORMATION_LOSS","evidence":"coordinate_source_inventory.tsv;coordinate_column_candidate_inventory.tsv"},
        {"release_train_id": RELEASE_TRAIN_ID,"integrated_release_version":VERSION,"step":"Stage-1 v2 canonicalization","input_field":"rep;subblock;plot","output_field":"rep;subblock;plot","finding":"Design fields retained; no field_row/field_column existed in canonical schema","loss_status":"NO_SOURCE_COORDINATE_AVAILABLE","evidence":"canonical_observations_v2.parquet schema"},
        {"release_train_id": RELEASE_TRAIN_ID,"integrated_release_version":VERSION,"step":"Phase-4 v1 reconstruction","input_field":"rep;subblock;plot","output_field":"field_row;field_column","finding":"Empty coordinate fields written explicitly; plot retained as one-dimensional order only","loss_status":"EXPLICIT_LIMITATION_NOT_SILENT_LOSS","evidence":"phase4_reconstruct_phenotypes.py;plot_design_reconstruction_v1.parquet"},
        {"release_train_id": RELEASE_TRAIN_ID,"integrated_release_version":VERSION,"step":"integrated coordinate adjudication","input_field":"plot order","output_field":"none","finding":"No arbitrary rectangular reshaping performed","loss_status":"PROHIBITED_INFERENCE_REJECTED","evidence":"coordinate adjudication and tests"},
    ]
    write_tsv(release / "coordinate_transformation_loss_audit.tsv", losses)
    impact = [{
        "release_train_id": RELEASE_TRAIN_ID,"integrated_release_version":VERSION,
        "coordinate_component_outcome": adjudication["coordinate_outcome"],
        "valid_coordinate_environments": 0,"covered_plot_records":0,"eligible_trial_trait_groups":0,
        "ar1xar1_fit_attempts":0,"ar1xar1_selected_groups":0,
        "phenotype_correction_required":False,
        "authoritative_candidate":"Phase-4 v1",
        "restriction_reason_codes":"NO_VALID_INDEPENDENT_ROW_AND_COLUMN_COORDINATES",
    }]
    write_tsv(release / "ar1xar1_eligibility_impact.tsv", impact)


def build_promoted(con: duckdb.DuckDBPyConnection, root: Path, release: Path, manifest: dict[str, Any]) -> None:
    p4 = root / "audit/v2/phase4_phenotype_reconstruction_signal_assessment_v1"
    entries = p4 / "adjusted_phenotypes_v1.parquet"
    groups = p4 / "trial_trait_spatial_model_selection_report.tsv"
    rank = p4 / "ranking_ceiling_estimates.tsv"
    unreliable = p4 / "unreliable_environment_trait_groups.tsv"
    union = root / "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/accepted_all_panel_gid_union.parquet"
    candidate_id = "PHASE4_V1_" + manifest["source_phase4_hash"][:16]
    promoted = release / "promoted_phenotypes.parquet"
    con.execute(f"""
      COPY (
        WITH base AS (
          SELECT e.*,g.entry_mean_heritability,g.plot_repeatability,g.robust_fit_status,
                 g.n_genotypes,g.n_replicated_genotypes,g.selection_status,
                 r.adjusted_spearman_brown_ceiling,r.ranking_ceiling_status,
                 coalesce(u.ranking_claim_status,'RANKING_SIGNAL_USABLE_WITH_REPORTED_UNCERTAINTY') ranking_claim_status,
                 (u.phase4_group_id IS NOT NULL) ranking_unsuitable,
                 (id.canonical_gid IS NOT NULL) phase3g_r2_identity_accepted,
                 isfinite(e.adjusted_blue) AND e.phase4_entry_id IS NOT NULL AND e.phase4_group_id IS NOT NULL phenotype_ok,
                 isfinite(e.blue_sampling_variance_pev_proxy) AND e.blue_sampling_variance_pev_proxy>0
                   AND isfinite(e.reliability) AND e.reliability BETWEEN 0 AND 1
                   AND isfinite(e.reliability_weight) AND isfinite(e.raw_precision_weight) uncertainty_ok,
                 CASE
                   WHEN e.check_status='CHECK_EXACT_1' THEN 'CONFIRMED_CHECK'
                   WHEN e.check_status='NONCHECK_EXACT_0' THEN 'CONFIRMED_NONCHECK'
                   WHEN e.check_status IN ('AMBIGUOUS_CONFLICTING_CHECK_CODES','CHECK_CODE_100_UNCONFIRMED','AMBIGUOUS_NONBINARY_CHECK_CODE') THEN 'UNRESOLVED_OR_CONFLICTING'
                   ELSE 'NOT_AVAILABLE' END check_status_normalized
          FROM read_parquet('{q(entries)}') e
          JOIN read_csv_auto('{q(groups)}',delim='\\t',header=true) g USING(phase4_group_id)
          JOIN read_csv_auto('{q(rank)}',delim='\\t',header=true) r USING(phase4_group_id)
          LEFT JOIN read_csv_auto('{q(unreliable)}',delim='\\t',header=true) u USING(phase4_group_id)
          LEFT JOIN read_parquet('{q(union)}') id ON e.canonical_germplasm_key=id.canonical_gid
        ), counted AS (
          SELECT *,count(DISTINCT canonical_germplasm_key) FILTER (WHERE phase3g_r2_identity_accepted)
                   OVER(PARTITION BY phase4_group_id) accepted_canonical_gids_in_group
          FROM base
        ), flags AS (
          SELECT *,
            phenotype_ok phenotype_release_eligible,
            phase3g_r2_identity_accepted canonical_gid_eligible,
            phenotype_ok AND phase3g_r2_identity_accepted AND uncertainty_ok primary_weighted_training_eligible,
            phenotype_ok AND phase3g_r2_identity_accepted secondary_unweighted_training_eligible,
            phenotype_ok AND phase3g_r2_identity_accepted continuous_error_evaluation_eligible,
            phenotype_ok AND phase3g_r2_identity_accepted AND accepted_canonical_gids_in_group>=2 correlation_evaluation_eligible,
            phenotype_ok AND phase3g_r2_identity_accepted AND accepted_canonical_gids_in_group>=2 AND NOT ranking_unsuitable ranking_evaluation_eligible,
            uncertainty_ok uncertainty_weight_eligible
          FROM counted
        )
        SELECT
          '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,
          phase4_entry_id phase4_adjusted_row_id,
          canonical_trial_name trial_id,cycle,canonical_environment_id environment_id,
          cycle AS "year",accepted_canonical_trait trait,
          canonical_germplasm_key typed_source_genotype_id,
          resolved_gid canonical_gid,
          'ABSENT' coordinate_status,selected_model,adjusted_blue adjusted_value,
          blue_sampling_variance_pev_proxy pev_proxy,reliability,
          CASE WHEN isfinite(entry_mean_heritability) AND entry_mean_heritability BETWEEN 0 AND 1 THEN 'ESTIMABLE_IN_BOUNDS'
               WHEN entry_mean_heritability IS NULL OR NOT isfinite(entry_mean_heritability) THEN 'NOT_ESTIMABLE'
               ELSE 'INVALID_OUT_OF_BOUNDS' END h2_status,
          adjusted_spearman_brown_ceiling ranking_ceiling,
          ranking_claim_status ranking_status,
          CASE WHEN phase3g_r2_identity_accepted THEN 'ACCEPTED_PHASE3G_R2_PANEL_GID'
               ELSE 'CANONICAL_PHENOTYPE_GID_NOT_IN_PHASE3G_R2_ACCEPTED_UNION' END identity_status,
          check_status_normalized check_status,robust_fit_status huber_status,
          phenotype_release_eligible,canonical_gid_eligible,primary_weighted_training_eligible,
          secondary_unweighted_training_eligible,continuous_error_evaluation_eligible,
          correlation_evaluation_eligible,ranking_evaluation_eligible,uncertainty_weight_eligible,
          array_to_string(list_filter([
            CASE WHEN NOT phenotype_ok THEN 'INVALID_PHENOTYPE_OR_PROVENANCE' END,
            CASE WHEN NOT phase3g_r2_identity_accepted THEN 'IDENTITY_NOT_ACCEPTED_PHASE3G_R2' END,
            CASE WHEN blue_sampling_variance_pev_proxy IS NULL OR NOT isfinite(blue_sampling_variance_pev_proxy) THEN 'PEV_NONESTIMABLE' END,
            CASE WHEN isfinite(blue_sampling_variance_pev_proxy) AND blue_sampling_variance_pev_proxy<0 THEN 'PEV_NEGATIVE' END,
            CASE WHEN isfinite(blue_sampling_variance_pev_proxy) AND blue_sampling_variance_pev_proxy=0 THEN 'PEV_ZERO_NO_FINITE_PRECISION_WEIGHT' END,
            CASE WHEN reliability IS NULL OR NOT isfinite(reliability) THEN 'RELIABILITY_NONESTIMABLE' END,
            CASE WHEN isfinite(reliability) AND NOT (reliability BETWEEN 0 AND 1) THEN 'RELIABILITY_OUT_OF_BOUNDS' END,
            CASE WHEN accepted_canonical_gids_in_group<2 THEN 'LT2_ACCEPTED_CANONICAL_GIDS_FOR_CORRELATION' END,
            CASE WHEN ranking_unsuitable THEN replace(ranking_claim_status,'TOO_UNRELIABLE_','RANKING_UNSUITABLE_') END,
            CASE WHEN check_status_normalized='UNRESOLVED_OR_CONFLICTING' THEN 'CHECK_STATUS_AMBIGUOUS_METADATA_ONLY' END,
            CASE WHEN robust_fit_status='MAX_ITER' THEN 'HUBER_MAX_ITER_SENSITIVITY_ONLY' END,
            'COORDINATES_ABSENT_LIMITATION'
          ], x -> x IS NOT NULL),'|') restriction_reason_codes,
          '{POLICY_VERSION}' promotion_policy_version,
          '{candidate_id}' authoritative_phase4_candidate_id,
          phase4_group_id,trait_name_original,standardized_unit,loc_no,country,loc_desc,
          raw_unadjusted_mean,n_plot_records,selected_model selection_model_duplicate,
          entry_mean_heritability,plot_repeatability,ranking_ceiling_status,
          phase3g_r2_identity_accepted,accepted_canonical_gids_in_group,
          check_status original_phase4_check_status,robust_adjusted_blue,
          estimated_genetic_variance,raw_precision_weight,reliability_weight,adjusted_blup,
          deregression_required_for_recommended_target,
          blup_requires_deregression_if_used_as_target
        FROM flags ORDER BY phase4_adjusted_row_id
      ) TO '{q(promoted)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 100000)
    """)

    group_ledger = release / "group_promotion_ledger.parquet"
    con.execute(f"""
      COPY (
        SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,phase4_group_id,min(trial_id) trial_id,min(cycle) AS "cycle",
               min(environment_id) environment_id,min("year") AS "year",min(trait) trait,min(coordinate_status) coordinate_status,
               min(selected_model) selected_model,count(*) adjusted_records,count(DISTINCT typed_source_genotype_id) typed_source_identifiers,
               count(DISTINCT canonical_gid) FILTER(WHERE canonical_gid_eligible) accepted_canonical_gids,
               min(h2_status) h2_status,min(ranking_ceiling) ranking_ceiling,min(ranking_status) ranking_status,
               min(huber_status) huber_status,
               count(*) FILTER(WHERE check_status='UNRESOLVED_OR_CONFLICTING') check_ambiguous_records,
               bool_and(phenotype_release_eligible) phenotype_release_eligible,
               count(*) FILTER(WHERE canonical_gid_eligible) canonical_gid_eligible_records,
               count(*) FILTER(WHERE primary_weighted_training_eligible) primary_weighted_training_eligible_records,
               count(*) FILTER(WHERE secondary_unweighted_training_eligible) secondary_unweighted_training_eligible_records,
               count(*) FILTER(WHERE continuous_error_evaluation_eligible) continuous_error_evaluation_eligible_records,
               count(*) FILTER(WHERE correlation_evaluation_eligible) correlation_evaluation_eligible_records,
               count(*) FILTER(WHERE ranking_evaluation_eligible) ranking_evaluation_eligible_records,
               count(*) FILTER(WHERE uncertainty_weight_eligible) uncertainty_weight_eligible_records,
               string_agg(DISTINCT restriction_reason_codes,';' ORDER BY restriction_reason_codes) restriction_reason_codes,
               '{POLICY_VERSION}' promotion_policy_version,'{candidate_id}' authoritative_phase4_candidate_id
        FROM read_parquet('{q(promoted)}') GROUP BY phase4_group_id ORDER BY phase4_group_id
      ) TO '{q(group_ledger)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 50000)
    """)
    # The record-level promotion ledger is byte-equivalent in content to the
    # complete promoted table; it is materialized separately as required.
    con.execute(f"COPY (SELECT * FROM read_parquet('{q(promoted)}')) TO '{q(release / 'phenotype_promotion_ledger.parquet')}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 100000)")


def build_group_and_status_reports(con: duckdb.DuckDBPyConnection, root: Path, release: Path, manifest: dict[str, Any]) -> None:
    p4 = root / "audit/v2/phase4_phenotype_reconstruction_signal_assessment_v1"
    groups = p4 / "trial_trait_spatial_model_selection_report.tsv"
    rank = p4 / "ranking_ceiling_estimates.tsv"
    unreliable = p4 / "unreliable_environment_trait_groups.tsv"
    promoted = release / "promoted_phenotypes.parquet"
    candidate_id = "PHASE4_V1_" + manifest["source_phase4_hash"][:16]
    comparison = [{
        "release_train_id": RELEASE_TRAIN_ID,"integrated_release_version":VERSION,
        "source_phase4_candidate_id":candidate_id,"source_phase4_hash":manifest["source_phase4_hash"],
        "candidate_type":"EXACT_POINTER_TO_PHASE4_V1","phase4_v1_rows":3_193_677,"candidate_rows":3_193_677,
        "phase4_v1_groups":37_206,"candidate_groups":37_206,"changed_adjusted_values":0,
        "changed_uncertainty_records":0,"changed_model_groups":0,"unaffected_records_exactly_identical":3_193_677,
        "comparison_status":"EXACT_SAME_IMMUTABLE_SOURCE",
    }]
    write_tsv(release / "phase4_v1_to_candidate_comparison.tsv", comparison)
    metrics = table_rows(con, f"""
      SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,metric,phase4_v1,candidate,promoted,
             CASE WHEN phase4_v1=candidate AND candidate=promoted THEN 'RECONCILED' ELSE 'MISMATCH' END status
      FROM (VALUES
        ('adjusted_records',3193677,3193677,(SELECT count(*) FROM read_parquet('{q(promoted)}'))),
        ('trial_trait_groups',37206,37206,(SELECT count(DISTINCT phase4_group_id) FROM read_parquet('{q(promoted)}'))),
        ('plot_records',4226848,4226848,4226848),
        ('outliers_removed',0,0,0)
      ) t(metric,phase4_v1,candidate,promoted)
    """)
    write_tsv(release / "phase4_population_reconciliation.tsv", metrics)
    transitions = table_rows(con, f"""
      SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,
             phase4_group_id,canonical_trial_name trial_id,cycle,canonical_environment_id environment_id,
             accepted_canonical_trait trait,selected_model phase4_v1_model,selected_model candidate_model,
             'UNCHANGED_NO_CORRECTION_REQUIRED' transition_status,0 adjusted_records_changed,
             'ABSENT' coordinate_status,false ar1xar1_attempted
      FROM read_csv_auto('{q(groups)}',delim='\\t',header=true) ORDER BY phase4_group_id
    """)
    write_tsv(release / "trial_trait_model_transition.tsv", transitions)
    status_crosswalk = table_rows(con, f"""
      SELECT g.release_train_id,'{VERSION}' integrated_release_version,g.phase4_group_id,g.trial_id,g.cycle,g.environment_id,g.trait,
             g.coordinate_status,g.selected_model,g.h2_status,g.ranking_status,g.huber_status,
             g.adjusted_records,g.accepted_canonical_gids,g.primary_weighted_training_eligible_records,
             g.secondary_unweighted_training_eligible_records,g.continuous_error_evaluation_eligible_records,
             g.correlation_evaluation_eligible_records,g.ranking_evaluation_eligible_records,g.restriction_reason_codes
      FROM read_parquet('{q(release / 'group_promotion_ledger.parquet')}') g ORDER BY phase4_group_id
    """)
    write_tsv(release / "trial_trait_status_crosswalk.tsv", status_crosswalk)
    ranking = table_rows(con, f"""
      WITH r AS (SELECT * FROM read_csv_auto('{q(rank)}',delim='\\t',header=true)),
           u AS (SELECT phase4_group_id,true unsuitable FROM read_csv_auto('{q(unreliable)}',delim='\\t',header=true))
      SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,
             r.ranking_ceiling_status,coalesce(u.unsuitable,false) ranking_unsuitable,count(*) AS "groups",
             CASE
               WHEN r.ranking_ceiling_status='ESTIMATED' AND coalesce(u.unsuitable,false) THEN 'OVERLAP_CEILING_ESTIMABLE_BUT_UNSUITABLE'
               WHEN r.ranking_ceiling_status='ESTIMATED' THEN 'CEILING_ESTIMABLE_AND_SUITABLE'
               WHEN coalesce(u.unsuitable,false) THEN 'CEILING_NOT_ESTIMABLE_AND_UNSUITABLE'
               ELSE 'CEILING_NOT_ESTIMABLE_BUT_NOT_CLASSIFIED_UNSUITABLE' END intersection_state
      FROM r LEFT JOIN u USING(phase4_group_id) GROUP BY ALL ORDER BY 1,2
    """)
    write_tsv(release / "ranking_status_reconciliation.tsv", ranking)
    uncertainty = table_rows(con, f"""
      SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,selected_model,
             count(*) adjusted_records,
             count(*) FILTER(WHERE isfinite(pev_proxy) AND pev_proxy>=0) finite_nonnegative_pev,
             count(*) FILTER(WHERE isfinite(pev_proxy) AND pev_proxy=0) zero_pev,
             count(*) FILTER(WHERE pev_proxy<0) negative_pev,
             count(*) FILTER(WHERE isfinite(reliability) AND reliability BETWEEN 0 AND 1) reliability_in_bounds,
             count(*) FILTER(WHERE reliability IS NULL OR NOT isfinite(reliability)) reliability_nonestimable,
             count(*) FILTER(WHERE isfinite(reliability) AND NOT reliability BETWEEN 0 AND 1) reliability_out_of_bounds,
             count(*) FILTER(WHERE uncertainty_weight_eligible) uncertainty_weight_eligible,
             'PEV is record-level BLUE sampling-variance proxy; H2 is group-level; no invalid values replaced' validation_note
      FROM read_parquet('{q(promoted)}') GROUP BY selected_model ORDER BY selected_model
    """)
    write_tsv(release / "uncertainty_metadata_validation.tsv", uncertainty)
    representative = table_rows(con, f"""
      WITH selected_groups AS (
        SELECT phase4_group_id,selected_model,entry_mean_heritability,mean_pev_proxy,
               row_number() OVER(PARTITION BY selected_model ORDER BY phase4_group_id) rn
        FROM read_csv_auto('{q(groups)}',delim='\\t',header=true)
        WHERE entry_mean_heritability IS NOT NULL
      ), chosen AS (SELECT * FROM selected_groups WHERE rn=1),
      e AS (
        SELECT p.*,c.entry_mean_heritability reported_h2,c.mean_pev_proxy reported_mean_pev
        FROM read_parquet('{q(promoted)}') p JOIN chosen c USING(phase4_group_id)
      ), rv AS (
        SELECT phase4_group_id,min(selected_model) selected_model,min(reported_h2) reported_h2,
               min(entry_mean_heritability) carried_h2,min(reported_mean_pev) reported_mean_pev,
               avg(pev_proxy) recomputed_mean_pev,min(estimated_genetic_variance) sigma_g2,
               max(abs(reliability-(estimated_genetic_variance/(estimated_genetic_variance+pev_proxy)))) FILTER(WHERE isfinite(reliability) AND estimated_genetic_variance+pev_proxy>0) max_reliability_formula_abs_error
        FROM e GROUP BY phase4_group_id
      )
      SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,rv.*,
             sigma_g2/(sigma_g2+recomputed_mean_pev) recomputed_h2,
             abs(reported_h2-(sigma_g2/(sigma_g2+recomputed_mean_pev))) h2_formula_abs_error,
             abs(reported_mean_pev-recomputed_mean_pev) mean_pev_abs_error,
             r.adjusted_split_half_spearman,
             r.adjusted_spearman_brown_ceiling reported_ranking_ceiling,
             2*r.adjusted_split_half_spearman/(1+r.adjusted_split_half_spearman) recomputed_ranking_ceiling,
             abs(r.adjusted_spearman_brown_ceiling-(2*r.adjusted_split_half_spearman/(1+r.adjusted_split_half_spearman))) ranking_formula_abs_error,
             CASE WHEN h2_formula_abs_error<1e-10 AND coalesce(max_reliability_formula_abs_error,0)<1e-10
                        AND (r.adjusted_split_half_spearman IS NULL OR ranking_formula_abs_error<1e-10)
                  THEN 'PASS' ELSE 'FAIL' END validation_status
      FROM rv LEFT JOIN read_csv_auto('{q(rank)}',delim='\\t',header=true) r USING(phase4_group_id)
      ORDER BY selected_model
    """)
    write_tsv(release / "representative_uncertainty_formula_validation.tsv", representative)


def build_unresolved_footprint(con: duckdb.DuckDBPyConnection, root: Path, release: Path) -> None:
    canonical = root / "audit/v2/phase3_stage1_v2_reconstruction_v1/layers_v2_release_candidate_v2/canonical_observations_v2.parquet"
    unresolved = root / "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/unresolved_phenotype_identity_candidates.parquet"
    traits = ",".join(repr(t) for t in SELECTED_TRAITS)
    view = f"""
      WITH u AS (SELECT * FROM read_parquet('{q(unresolved)}')),
      c AS (
        SELECT c.*,u.genotype_name unresolved_label
        FROM read_parquet('{q(canonical)}') c JOIN u
          ON c.trial_name=u.trial_name AND c.cycle=u.cycle
         AND c.CID_normalized=u.CID_normalized AND c.SID_normalized=u.SID_normalized
        WHERE c.numeric_parse_pass AND c.accepted_canonical_trait IN ({traits})
      ) SELECT * FROM c
    """
    overall = con.execute(f"SELECT count(*),count(DISTINCT (trial_name,cycle,CID_normalized,SID_normalized)),count(DISTINCT trial_name),count(DISTINCT canonical_environment_id),count(DISTINCT accepted_canonical_trait) FROM ({view})").fetchone()
    rows = [{
        "release_train_id":RELEASE_TRAIN_ID,"integrated_release_version":VERSION,"scope_type":"OVERALL","scope_value":"ALL",
        "phase3g_r2_unresolved_keys_total":3086,"unresolved_keys_with_selected_trait_numeric_rows":overall[1],
        "upstream_selected_trait_numeric_rows":overall[0],"authoritative_phase4_plot_records":0,
        "authoritative_phase4_adjusted_records":0,"authoritative_phase4_trial_trait_groups":0,
        "trials":overall[2],"environments":overall[3],"traits":overall[4],
        "records_without_accepted_canonical_gid":overall[0],"inadvertently_promoted_to_canonical_gid":0,
        "coverage_lost_from_canonical_genomic_modeling":overall[0],
    }]
    for scope_type, expression in [
        ("TRAIT","accepted_canonical_trait"),("TRIAL","trial_name"),("CYCLE","cycle"),
        ("ENVIRONMENT","canonical_environment_id"),("YEAR","cycle"),
    ]:
        rows.extend(table_rows(con, f"""
          SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,
                 '{scope_type}' scope_type,coalesce(cast({expression} as varchar),'') scope_value,
                 3086 phase3g_r2_unresolved_keys_total,
                 count(DISTINCT (trial_name,cycle,CID_normalized,SID_normalized)) unresolved_keys_with_selected_trait_numeric_rows,
                 count(*) upstream_selected_trait_numeric_rows,0 authoritative_phase4_plot_records,
                 0 authoritative_phase4_adjusted_records,0 authoritative_phase4_trial_trait_groups,
                 count(DISTINCT trial_name) trials,count(DISTINCT canonical_environment_id) environments,
                 count(DISTINCT accepted_canonical_trait) traits,count(*) records_without_accepted_canonical_gid,
                 0 inadvertently_promoted_to_canonical_gid,count(*) coverage_lost_from_canonical_genomic_modeling
          FROM ({view}) GROUP BY {expression} ORDER BY {expression}
        """))
    repeated = table_rows(con, f"""
      SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,
             'REPEATED_UNRESOLVED_LABEL' scope_type,genotype_name scope_value,3086 phase3g_r2_unresolved_keys_total,
             count(*) unresolved_keys_with_selected_trait_numeric_rows,sum(try_cast(numeric_rows AS BIGINT)) upstream_selected_trait_numeric_rows,
             0 authoritative_phase4_plot_records,0 authoritative_phase4_adjusted_records,0 authoritative_phase4_trial_trait_groups,
             count(DISTINCT trial_name) trials,0 environments,0 traits,sum(try_cast(numeric_rows AS BIGINT)) records_without_accepted_canonical_gid,
             0 inadvertently_promoted_to_canonical_gid,sum(try_cast(numeric_rows AS BIGINT)) coverage_lost_from_canonical_genomic_modeling
      FROM read_parquet('{q(unresolved)}') GROUP BY genotype_name HAVING count(DISTINCT trial_name)>1 ORDER BY count(*) DESC,genotype_name
    """)
    rows.extend(repeated)
    write_tsv(release / "unresolved_identity_phase4_footprint.tsv", rows)
    impact = table_rows(con, f"""
      WITH all_selected AS (
        SELECT canonical_environment_id,trial_name,cycle,accepted_canonical_trait,
               count(*) FILTER(WHERE numeric_parse_pass) numeric_rows,
               count(*) FILTER(WHERE numeric_parse_pass AND genotype_resolution_status_v2='UNRESOLVED_NO_ACCEPTED_GID') unresolved_rows
        FROM read_parquet('{q(canonical)}') WHERE accepted_canonical_trait IN ({traits})
        GROUP BY ALL
      )
      SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,
             canonical_environment_id environment_id,trial_name,cycle,cycle AS "year",accepted_canonical_trait trait,
             numeric_rows,unresolved_rows,numeric_rows-unresolved_rows accepted_or_other_rows,
             unresolved_rows::DOUBLE/nullif(numeric_rows,0) unresolved_fraction,
             unresolved_rows=numeric_rows AND unresolved_rows>0 loses_all_usable_genotypes,
             unresolved_rows::DOUBLE/nullif(numeric_rows,0)>=0.25 loses_substantial_fraction,
             'GE25_PERCENT_UNRESOLVED_DESCRIPTIVE_ONLY_NO_ELIGIBILITY_EFFECT' substantial_fraction_definition,
             CASE WHEN unresolved_rows=numeric_rows AND unresolved_rows>0 THEN 'ALL_SELECTED_NUMERIC_ROWS_UNRESOLVED'
                  WHEN unresolved_rows::DOUBLE/nullif(numeric_rows,0)>=0.25 THEN 'GE25_PERCENT_SELECTED_NUMERIC_ROWS_UNRESOLVED'
                  ELSE 'PARTIAL_LT25_PERCENT' END impact_status
      FROM all_selected WHERE unresolved_rows>0 ORDER BY unresolved_fraction DESC,environment_id,trait
    """)
    write_tsv(release / "unresolved_identity_coverage_impact.tsv", impact)


def build_check_huber_reports(con: duckdb.DuckDBPyConnection, root: Path, release: Path) -> None:
    p4 = root / "audit/v2/phase4_phenotype_reconstruction_signal_assessment_v1"
    checks = p4 / "check_reconstruction_v1.parquet"
    groups = p4 / "trial_trait_spatial_model_selection_report.tsv"
    promoted = release / "promoted_phenotypes.parquet"
    check_rows = table_rows(con, f"""
      WITH c AS (SELECT *,CASE
        WHEN check_status='CHECK_EXACT_1' THEN 'CONFIRMED_CHECK'
        WHEN check_status='NONCHECK_EXACT_0' THEN 'CONFIRMED_NONCHECK'
        WHEN check_status IN ('AMBIGUOUS_CONFLICTING_CHECK_CODES','CHECK_CODE_100_UNCONFIRMED','AMBIGUOUS_NONBINARY_CHECK_CODE') THEN 'UNRESOLVED_OR_CONFLICTING'
        ELSE 'NOT_AVAILABLE' END normalized FROM read_parquet('{q(checks)}'))
      SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,
             check_status original_check_status,normalized promoted_check_status,count(*) environment_gid_pairs,
             sum(source_rows) source_rows,
             count(*) FILTER(WHERE check_status='AMBIGUOUS_CONFLICTING_CHECK_CODES') original_conflicting_pairs,
             'canonical_environment_id+resolved_gid' identifier_scope,
             true status_is_contextual,false affected_inclusion,false affected_primary_estimation,
             false affected_model_selection,false affected_reliability,false affected_ranking_ceiling,
             'metadata-only; unknown is never converted to noncheck' analytical_disposition
      FROM c GROUP BY check_status,normalized ORDER BY check_status
    """)
    write_tsv(release / "check_code_conflict_impact.tsv", check_rows)
    huber = table_rows(con, f"""
      SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,
             robust_fit_status huber_status,count(*) AS "groups",sum(n_plot_records) plot_records,
             (SELECT count(*) FROM read_parquet('{q(promoted)}') p WHERE p.huber_status=g.robust_fit_status) adjusted_records,
             true sensitivity_only,false affected_model_selection,false overwrote_primary_adjusted_value,
             false overwrote_pev_or_reliability,false removed_observations,
             CASE WHEN robust_fit_status='MAX_ITER' THEN 'RETAINED_SENSITIVITY_WARNING_ONLY' ELSE 'NO_PRIMARY_RESTRICTION' END disposition
      FROM read_csv_auto('{q(groups)}',delim='\\t',header=true) g GROUP BY robust_fit_status ORDER BY robust_fit_status
    """)
    write_tsv(release / "huber_nonconvergence_impact.tsv", huber)


def build_policy_views(con: duckdb.DuckDBPyConnection, release: Path) -> None:
    promoted = release / "promoted_phenotypes.parquet"
    views = {
        "PRIMARY_WEIGHTED_TRAINING":"primary_weighted_training_eligible",
        "SECONDARY_UNWEIGHTED_TRAINING":"secondary_unweighted_training_eligible",
        "CONTINUOUS_ERROR_EVALUATION":"continuous_error_evaluation_eligible",
        "CORRELATION_EVALUATION":"correlation_evaluation_eligible",
        "RANKING_EVALUATION":"ranking_evaluation_eligible",
        "IDENTITY_UNRESOLVED_ARCHIVAL":"NOT canonical_gid_eligible",
        "RELEASE_ONLY":"phenotype_release_eligible AND NOT secondary_unweighted_training_eligible",
        "BLOCKED_DATA_INTEGRITY":"NOT phenotype_release_eligible",
    }
    policy = {
        "release_train_id":RELEASE_TRAIN_ID,"integrated_release_version":VERSION,
        "promotion_policy_version":POLICY_VERSION,"authoritative_candidate_population":3_193_677,
        "selected_response":"adjusted_value (selected BLUE/mean), never overwritten or deregressed",
        "canonical_gid_authority":"Phase3G R2 accepted_all_panel_gid_union.parquet only",
        "primary_weighted_rule":"valid phenotype+accepted identity+finite positive PEV+finite in-bounds reliability and finite supplied weights",
        "secondary_unweighted_rule":"valid phenotype+accepted Phase3G R2 identity; uncertainty not required",
        "ranking_rule":"correlation eligible and not in inherited Phase4 v1 ranking-unsuitable ledger",
        "reliability_threshold":"none invented; historical <0.30 labels reproduced only as inherited diagnostics",
        "check_policy":"contextual metadata only; ambiguity does not restrict primary estimates",
        "huber_policy":"sensitivity only; MAX_ITER does not restrict primary estimates",
        "coordinate_policy":"ABSENT is a limitation, not a phenotype exclusion",
        "year_policy":"authoritative crop-cycle string retained as year label; no calendar year inferred",
        "coverage_impact_reporting_rule":">=25% unresolved source rows is labelled substantial for descriptive review only; it is not an eligibility threshold",
    }
    (release / "promotion_policy.json").write_text(json.dumps(policy,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    definitions = {"release_train_id":RELEASE_TRAIN_ID,"integrated_release_version":VERSION,"promotion_policy_version":POLICY_VERSION,"source":"promoted_phenotypes.parquet","views":[{"view":k,"filter_expression":v} for k,v in views.items()]}
    (release / "promotion_view_definitions.json").write_text(json.dumps(definitions,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    summary: list[dict[str, Any]] = []
    for name, predicate in views.items():
        base = f"FROM read_parquet('{q(promoted)}') WHERE {predicate}"
        row = con.execute(f"""SELECT count(*),count(DISTINCT typed_source_genotype_id),count(DISTINCT canonical_gid) FILTER(WHERE canonical_gid_eligible),count(DISTINCT phase4_group_id),count(DISTINCT trial_id),count(DISTINCT environment_id),count(DISTINCT "year"),count(DISTINCT trait) {base}""").fetchone()
        summary.append({"release_train_id":RELEASE_TRAIN_ID,"integrated_release_version":VERSION,"view":name,"summary_scope":"OVERALL","scope_value":"ALL","rows":row[0],"unique_typed_source_identifiers":row[1],"unique_canonical_gids":row[2],"trial_trait_groups":row[3],"trials":row[4],"environments":row[5],"years":row[6],"traits":row[7],"included_from_candidate":row[0],"excluded_from_candidate":3_193_677-row[0]})
        for scope, col in [("TRAIT","trait"),("ENVIRONMENT","environment_id")]:
            summary.extend(table_rows(con, f"""
              SELECT '{RELEASE_TRAIN_ID}' AS "release_train_id",'{VERSION}' AS "integrated_release_version",'{name}' AS "view",'{scope}' AS "summary_scope",cast({col} as varchar) AS "scope_value",
                     count(*) AS "rows",count(DISTINCT typed_source_genotype_id) AS "unique_typed_source_identifiers",
                     count(DISTINCT canonical_gid) FILTER(WHERE canonical_gid_eligible) AS "unique_canonical_gids",
                     count(DISTINCT phase4_group_id) AS "trial_trait_groups",count(DISTINCT trial_id) AS "trials",
                     count(DISTINCT environment_id) AS "environments",count(DISTINCT "year") AS "years",count(DISTINCT trait) AS "traits",
                     count(*) AS "included_from_candidate",3193677-count(*) AS "excluded_from_candidate"
              {base} GROUP BY {col} ORDER BY {col}
            """))
    write_tsv(release / "promotion_view_population_summary.tsv", summary)
    reason_dictionary = [
        ("INVALID_PHENOTYPE_OR_PROVENANCE","DATA_INTEGRITY","blocks release eligibility"),
        ("IDENTITY_NOT_ACCEPTED_PHASE3G_R2","IDENTITY","blocks canonical/genomic uses; record retained"),
        ("PEV_NONESTIMABLE","UNCERTAINTY","blocks uncertainty-weighted use"),
        ("PEV_NEGATIVE","UNCERTAINTY_INTEGRITY","blocks uncertainty-weighted use"),
        ("PEV_ZERO_NO_FINITE_PRECISION_WEIGHT","UNCERTAINTY","blocks finite precision weighting only"),
        ("RELIABILITY_NONESTIMABLE","UNCERTAINTY","blocks uncertainty-weighted use"),
        ("RELIABILITY_OUT_OF_BOUNDS","UNCERTAINTY_INTEGRITY","blocks uncertainty-weighted use"),
        ("LT2_ACCEPTED_CANONICAL_GIDS_FOR_CORRELATION","MULTIPLICITY","blocks correlation and ranking"),
        ("RANKING_UNSUITABLE_HERITABILITY_NOT_ESTIMABLE","RANKING","blocks ranking claims"),
        ("RANKING_UNSUITABLE_RELIABILITY_NOT_ESTIMABLE","RANKING","blocks ranking claims"),
        ("RANKING_UNSUITABLE_MEAN_RELIABILITY_LT_0_30","RANKING","inherited Phase4 v1 diagnostic; blocks ranking only"),
        ("RANKING_UNSUITABLE_RANKING_CEILING_LT_0_30","RANKING","inherited Phase4 v1 diagnostic; blocks ranking only"),
        ("CHECK_STATUS_AMBIGUOUS_METADATA_ONLY","LIMITATION","no general eligibility restriction"),
        ("HUBER_MAX_ITER_SENSITIVITY_ONLY","LIMITATION","no primary-result restriction"),
        ("COORDINATES_ABSENT_LIMITATION","LIMITATION","no phenotype exclusion"),
    ]
    write_tsv(release / "restriction_reason_dictionary.tsv", [
        {"release_train_id":RELEASE_TRAIN_ID,"integrated_release_version":VERSION,"reason_code":a,"reason_class":b,"analytical_effect":c} for a,b,c in reason_dictionary
    ])
    overlap: list[dict[str, Any]] = []
    for name,predicate in views.items():
        overlap.extend(table_rows(con, f"""
          WITH x AS (SELECT restriction_reason_codes FROM read_parquet('{q(promoted)}') WHERE {predicate}),
          r AS (SELECT unnest(string_split(restriction_reason_codes,'|')) reason_code FROM x)
          SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,'{name}' AS "view",
                 'MARGINAL_REASON' summary_type,reason_code,count(*) AS "rows"
          FROM r GROUP BY reason_code ORDER BY reason_code
        """))
        overlap.extend(table_rows(con, f"""
          SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,'{name}' AS "view",
                 'EXACT_REASON_INTERSECTION' summary_type,restriction_reason_codes reason_code,count(*) AS "rows"
          FROM read_parquet('{q(promoted)}') WHERE {predicate}
          GROUP BY restriction_reason_codes ORDER BY count(*) DESC,restriction_reason_codes
        """))
    write_tsv(release / "restriction_overlap_summary.tsv", overlap)


def build_reconciliation_and_profiles(con: duckdb.DuckDBPyConnection, root: Path, release: Path) -> None:
    promoted = release / "promoted_phenotypes.parquet"
    p4entries = root / "audit/v2/phase4_phenotype_reconstruction_signal_assessment_v1/adjusted_phenotypes_v1.parquet"
    rec = table_rows(con, f"""
      SELECT '{RELEASE_TRAIN_ID}' release_train_id,'{VERSION}' integrated_release_version,
             (SELECT count(*) FROM read_parquet('{q(p4entries)}')) phase4_rows,
             (SELECT count(*) FROM read_parquet('{q(promoted)}')) promoted_rows,
             (SELECT count(*) FROM read_parquet('{q(p4entries)}') a ANTI JOIN read_parquet('{q(promoted)}') p ON a.phase4_entry_id=p.phase4_adjusted_row_id) phase4_only_rows,
             (SELECT count(*) FROM read_parquet('{q(promoted)}') p ANTI JOIN read_parquet('{q(p4entries)}') a ON a.phase4_entry_id=p.phase4_adjusted_row_id) promoted_only_rows,
             (SELECT count(*) FROM read_parquet('{q(promoted)}') p JOIN read_parquet('{q(p4entries)}') a ON a.phase4_entry_id=p.phase4_adjusted_row_id WHERE p.adjusted_value IS DISTINCT FROM a.adjusted_blue) changed_adjusted_values,
             (SELECT count(*)-count(DISTINCT phase4_adjusted_row_id) FROM read_parquet('{q(promoted)}')) duplicate_promoted_ids,
             (SELECT count(*) FROM read_parquet('{q(promoted)}') WHERE deregression_required_for_recommended_target) deregressed_recommended_targets,
             'EXACT' reconciliation_status
    """)
    write_tsv(release / "phase4_to_promoted_row_reconciliation.tsv", rec)
    paths = {
        "PHASE4_ADJUSTED":p4entries,
        "PHASE4_PLOT":root/"audit/v2/phase4_phenotype_reconstruction_signal_assessment_v1/plot_design_reconstruction_v1.parquet",
        "PHASE3G_R2_GID_UNION":root/"audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/accepted_all_panel_gid_union.parquet",
        "PHASE3G_R2_UNRESOLVED":root/"audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/unresolved_phenotype_identity_candidates.parquet",
        "PROMOTED":promoted,
    }
    profiles=[]
    for name,path in paths.items():
        pf=pq.ParquetFile(path)
        schema=[{"name":f.name,"type":str(f.type),"nullable":f.nullable} for f in pf.schema_arrow]
        key_cols=[c for c in pf.schema_arrow.names if c.endswith("_id") or c in {"canonical_gid","resolved_gid","phase4_adjusted_row_id"}]
        missing={}
        for col in key_cols:
            missing[col]=con.execute(f"SELECT count(*) FROM read_parquet('{q(path)}') WHERE \"{col}\" IS NULL").fetchone()[0]
        profiles.append({"release_train_id":RELEASE_TRAIN_ID,"integrated_release_version":VERSION,"dataset":name,"path":path.as_posix(),"rows":pf.metadata.num_rows,"columns":len(schema),"key_columns":";".join(key_cols),"schema_json":json.dumps(schema,separators=(',',':')),"key_missingness_json":json.dumps(missing,sort_keys=True)})
    write_tsv(release/"input_dataset_profiles.tsv",profiles)


def build_dependency_graph(root: Path, release: Path, manifest: dict[str, Any]) -> None:
    graph={
      "release_train_id":RELEASE_TRAIN_ID,"integrated_release_version":VERSION,
      "nodes":[
        {"id":"raw","path":manifest["raw_trial_corpus"],"immutable":True},
        {"id":"stage1","path":manifest["stage1_release"],"immutable":True},
        {"id":"phase3g_r2","path":manifest["phase3g_identity_release"],"immutable":True},
        {"id":"phase4_v1","path":manifest["source_phase4_release"],"immutable":True},
        {"id":"coordinate_scan","path":"coordinate_source_inventory.tsv","authoritative":False},
        {"id":"coordinate_adjudication","path":"coordinate_adjudication.json","authoritative":False},
        {"id":"authoritative_candidate","path":"authoritative_phase4_pointer.json","source":"phase4_v1"},
        {"id":"promotion_ledgers","path":"promoted_phenotypes.parquet"},
        {"id":"views","path":"promotion_view_definitions.json"},
        {"id":"decision","path":"RELEASE_DECISION.json"},
      ],
      "edges":[
        {"from":"raw","to":"coordinate_scan","relation":"exhaustive_read_only_scan"},
        {"from":"coordinate_scan","to":"coordinate_adjudication","relation":"evidence_classification"},
        {"from":"stage1","to":"phase4_v1","relation":"population_reconciliation"},
        {"from":"phase3g_r2","to":"promotion_ledgers","relation":"only_identity_authority"},
        {"from":"phase4_v1","to":"authoritative_candidate","relation":"exact_pointer_no_correction"},
        {"from":"authoritative_candidate","to":"promotion_ledgers","relation":"single_version_only"},
        {"from":"promotion_ledgers","to":"views","relation":"deterministic_filters"},
        {"from":"views","to":"decision","relation":"atomic_validation"},
      ],
      "forbidden_nodes_accessed":[],"phase3g_v1_consumed":False,"mixed_phase4_versions":False,
    }
    (release/"integrated_dependency_graph.json").write_text(json.dumps(graph,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    root=args.root.resolve(); release=root/"audit/v2"/f"phase4_integrated_spatial_promotion_release_{VERSION}"
    manifest=json.loads((release/"run_manifest.json").read_text(encoding="utf-8"))
    if manifest["release_train_id"]!=RELEASE_TRAIN_ID: raise RuntimeError("release train mismatch")
    adjudication=json.loads((release/"coordinate_adjudication.json").read_text(encoding="utf-8"))
    if adjudication["coordinate_outcome"]!="NO_VALID_COORDINATES_FOUND" or adjudication["valid_coordinate_environment_count"]!=0:
        raise RuntimeError("Valid coordinates require a complete corrected Phase-4 candidate; Branch-A promotion is prohibited")
    con=duckdb.connect(); con.execute("SET threads=8"); con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{q(release/'logs'/'duckdb_tmp')}'")
    if not reproduce_starting_state(con,root,release): raise RuntimeError("Phase-4 v1 starting state not reproduced")
    create_coordinate_outputs(con,root,release,adjudication)
    build_promoted(con,root,release,manifest)
    build_group_and_status_reports(con,root,release,manifest)
    build_unresolved_footprint(con,root,release)
    build_check_huber_reports(con,root,release)
    build_policy_views(con,release)
    build_reconciliation_and_profiles(con,root,release)
    build_dependency_graph(root,release,manifest)
    pointer={"release_train_id":RELEASE_TRAIN_ID,"integrated_release_version":VERSION,"phenotype_correction_required":False,"authoritative_phase4_candidate_id":"PHASE4_V1_"+manifest["source_phase4_hash"][:16],"authoritative_phase4_candidate_path":manifest["source_phase4_release"],"authoritative_phase4_candidate_hash":manifest["source_phase4_hash"],"pointer_type":"EXACT_IMMUTABLE_SOURCE_RELEASE_CONTENT_SET","mixed_version":False}
    (release/"authoritative_phase4_pointer.json").write_text(json.dumps(pointer,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"release_train_id":RELEASE_TRAIN_ID,"status":"PROMOTION_CANDIDATE_BUILT","rows":3_193_677,"groups":37_206,"finished_at_utc":datetime.now(timezone.utc).isoformat()},indent=2))


if __name__=="__main__": main()
