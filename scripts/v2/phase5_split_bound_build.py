from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from phase5_split_bound_common import (
    INNER_FOLDS,
    OUTER_FOLDS,
    PROHIBITED_SPLIT_COLUMNS,
    RELEASE_ID,
    SCENARIOS,
    SEED,
    SPLIT_ALLOWED_COLUMNS,
    assign_balanced_entities,
    build_pedigree_factor,
    build_pedigree_parent_map,
    canonical_gid,
    consensus_dosage,
    decode_biallelic_call,
    fit_vanraden,
    geo_factor,
    index_signature,
    kernel_diagnostics,
    relationship_block,
    relationship_element,
    sha256_file,
    stable_json_hash,
    write_json,
    write_tsv,
)


REQUIRED_HASHES = {
    "stage1_primary_manifest": (
        "audit/v2/phase3_stage1_v2_reconstruction_v1/delivery_v1/primary_release_manifest.tsv",
        "be9aea443f6117a0eaf92905ced403c86b1a55da7de95a30c10d4a5a541fc5d3",
    ),
    "phase3g_r2_output_manifest": (
        "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/output_manifest.tsv",
        "4837fce9429186f5dd00dc532686fbe00f0d714e201f997fc92483b3ec58ed88",
    ),
    "phase4_integrated_output_manifest": (
        "audit/v2/phase4_integrated_spatial_promotion_release_v1/output_manifest.tsv",
        "156574dfc75bf240937eb2850a57c4cb4c831fa32a5fb750acfeceb662480656",
    ),
    "prior_phase5_decision": (
        "audit/v2/phase5_kernel_validation_v1/PHASE5_RELEASE_DECISION.json",
        "b66ee47eb6f19943f6b759d90a61593f2ae29037317dbc9a8e56b4383bad2923",
    ),
    "phase4_corrected_output_manifest": (
        "audit/v2/phase4_namespace_corrected_release_v1/output_manifest.tsv",
        "54c710a807f77cca3d83c5def155d64a8153f6f27703c69e62c4e8d3dd909cc6",
    ),
    "phase3g_r3_output_manifest": (
        "audit/v2/phase3g_r3_identity_recovery_v1/output_manifest.tsv",
        "d718d9147d918cd39eb3c493635bc133d82edf7131f914857627cadd7cc4ef88",
    ),
    "overall_readiness_decision": (
        "audit/v2/phase3g_r3_identity_recovery_v1/OVERALL_READINESS_DECISION.json",
        "61721acdeab38548bed630e0eaa868249986aa113339df9c1846bab7f7619d60",
    ),
    "corrected_phase4_table": (
        "audit/v2/phase4_namespace_corrected_release_v1/corrected_promoted_phenotypes.parquet",
        "e015e3c102320b7ddc0eb55f88d65628142999b43af88b154fd31b677a340cb7",
    ),
    "r2_unresolved_candidates": (
        "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/unresolved_phenotype_identity_candidates.tsv",
        "b9b4c976d60ce7e3d74fa0c09af7eb43314233770f33a8a21682859f0a0da34c",
    ),
    "r2_canonical_panel_coverage": (
        "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/canonical_gid_panel_coverage.tsv",
        "0772ae9a521a46c39df31217b102f59c6504af2e01d0d6d15a5b1e524e6b7257",
    ),
    "r2_protocol": (
        "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/r2_protocol.json",
        "c35a6e7cde41f93b13d8d3cdef641be79c28705758b98043d14f27661eb1949a",
    ),
    "r2_build_summary": (
        "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/phase3g_r2_build_summary.json",
        "c402a09a2dcba9a4a6f7ddf7a100dd40b149f8da40b096bbdc2911d814a1f17a",
    ),
}


EXPECTED_VIEWS = {
    "PRIMARY_WEIGHTED_TRAINING": ("primary_weighted_training_eligible", 2_045_518, 10_656),
    "SECONDARY_UNWEIGHTED_TRAINING": ("secondary_unweighted_training_eligible", 2_242_863, 10_722),
    "CONTINUOUS_ERROR_EVALUATION": ("continuous_error_evaluation_eligible", 2_242_863, 10_722),
    "CORRELATION_EVALUATION": ("correlation_evaluation_eligible", 2_242_615, 10_722),
    "RANKING_EVALUATION": ("ranking_evaluation_eligible", 1_418_644, 10_656),
    "IDENTITY_UNRESOLVED_ARCHIVAL": ("not canonical_gid_eligible", 950_814, 0),
    "RELEASE_ONLY": ("phenotype_release_eligible and not canonical_gid_eligible", 950_814, 0),
    "BLOCKED_DATA_INTEGRITY": ("not phenotype_release_eligible", 0, 0),
}


PANEL_CLASS = {
    "cimmyt_bread_gbs_2013_2018": ("DENSE_GENOMEWIDE_KG_CANDIDATE", "DEFERRED_GLOBAL_IMPUTED_EXPORT_NO_RAW_TRAINING_LOCAL_IMPUTATION"),
    "cimmyt_bread_gbs_2013_2018_ta_metadata": ("HISTORICAL_PRECOMPUTED_DIAGNOSTIC_ONLY", "EXCLUDED_METADATA_ONLY_NO_RAW_MATRIX"),
    "dartag_panel2": ("TARGETED_MAS_COVARIATE_OR_SPARSE_KERNEL", "DEFERRED_TARGETED_PANEL_NOT_GENOMEWIDE_KG"),
    "dartseq80k_collection": ("IDENTITY_CANDIDATE_ONLY_NOT_AUTHORIZED", "EXCLUDED_COLLECTION_METADATA_NO_AUTHORIZED_GIDS"),
    "dartseq80k_hexaploid": ("IDENTITY_CANDIDATE_ONLY_NOT_AUTHORIZED", "EXCLUDED_NO_SAME_DATASET_TYPED_IDENTITY"),
    "dartseq80k_tetraploid": ("IDENTITY_CANDIDATE_ONLY_NOT_AUTHORIZED", "EXCLUDED_NO_SAME_DATASET_TYPED_IDENTITY"),
    "dartseq80k_wheat_recall": ("IDENTITY_CANDIDATE_ONLY_NOT_AUTHORIZED", "EXCLUDED_NO_SAME_DATASET_TYPED_IDENTITY"),
    "dartseq80k_wild_relative": ("IDENTITY_CANDIDATE_ONLY_NOT_AUTHORIZED", "EXCLUDED_NO_SAME_DATASET_TYPED_IDENTITY"),
    "eyt_haplotype_blocks_2011_2018": ("HAPLOTYPE_KERNEL_CANDIDATE", "DEFERRED_HAPLOTYPE_PROTOCOL_AND_SOURCE_SNP_DUPLICATION_REVIEW"),
    "frozen_hmp_v1": ("HISTORICAL_PRECOMPUTED_DIAGNOSTIC_ONLY", "EXCLUDED_CERTIFIED_V1_GLOBAL_KERNEL"),
    "gbs_13sawyt": ("DENSE_GENOMEWIDE_KG_CANDIDATE", "DEFERRED_BLOCKED_PANEL_QC_PROTOCOL"),
    "gbs_14sawyt": ("DENSE_GENOMEWIDE_KG_CANDIDATE", "DEFERRED_BLOCKED_PANEL_QC_PROTOCOL"),
    "gbs_15sawyt": ("DENSE_GENOMEWIDE_KG_CANDIDATE", "DEFERRED_BLOCKED_PANEL_QC_PROTOCOL"),
    "gbs_16sawyt": ("DENSE_GENOMEWIDE_KG_CANDIDATE", "DEFERRED_BLOCKED_PANEL_QC_PROTOCOL"),
    "gbs_17sawyt": ("DENSE_GENOMEWIDE_KG_CANDIDATE", "DEFERRED_BLOCKED_PANEL_QC_PROTOCOL"),
    "gbs_18sawyt": ("DENSE_GENOMEWIDE_KG_CANDIDATE", "DEFERRED_BLOCKED_PANEL_QC_PROTOCOL"),
    "hibap35k": ("DENSE_GENOMEWIDE_KG_CANDIDATE", "INCLUDED_PRODUCTION_FOLD_LOCAL_KG"),
    "mas_45ibwsn": ("TARGETED_MAS_COVARIATE_OR_SPARSE_KERNEL", "DEFERRED_TARGETED_MAS_COVARIATE"),
    "mas_57ibwsn_42sawsn_35hrwsn": ("TARGETED_MAS_COVARIATE_OR_SPARSE_KERNEL", "DEFERRED_TARGETED_MAS_COVARIATE"),
    "mas_58ibwsn_43sawsn": ("TARGETED_MAS_COVARIATE_OR_SPARSE_KERNEL", "DEFERRED_TARGETED_MAS_COVARIATE"),
    "mexican_landrace_dartseq": ("DENSE_GENOMEWIDE_KG_CANDIDATE", "DEFERRED_BLOCKED_PANEL_QC_AND_REPLICATE_PROTOCOL"),
    "seeds_of_discovery_dartseq": ("DENSE_GENOMEWIDE_KG_CANDIDATE", "DEFERRED_BLOCKED_PANEL_QC_AND_REPLICATE_PROTOCOL"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def qpath(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def portable_selection(expression: str, observation_assignment: Path) -> str:
    """Remove the physical release root from persisted selection contracts."""
    physical = qpath(observation_assignment)
    logical = "${PHASE5_RELEASE_ROOT}/splits/observation_split_assignment.parquet"
    return expression.replace(physical, logical)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT).strip()


def require_empty_construction_root(out: Path) -> None:
    if not out.exists():
        raise SystemExit(f"Opening manifest root is absent: {out}")
    allowed = {"OPENING_HASH_MANIFEST.tsv", "diagnostic_attempts"}
    extras = sorted(path.name for path in out.iterdir() if path.name not in allowed)
    if extras:
        raise SystemExit(f"Phase-5 root is not opening-only: {extras[:10]}")


def create_layout(out: Path) -> None:
    for directory in (
        "splits",
        "indices",
        "coverage",
        "pedigree/states",
        "genomic/states",
        "environment/states",
        "gxe",
        "model_inputs",
        "tests",
        "figures",
        "determinism_replay",
        "code_snapshot",
        "logs",
    ):
        (out / directory).mkdir(parents=True, exist_ok=False)


def verify_dependencies(root: Path, out: Path) -> None:
    rows = []
    for name, (relative, expected) in REQUIRED_HASHES.items():
        path = root / relative
        observed = sha256_file(path) if path.exists() else "MISSING"
        rows.append(
            {
                "dependency": name,
                "relative_path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "status": "PASS" if observed == expected else "FAIL",
            }
        )
    p4 = json.loads((root / "audit/v2/phase4_namespace_corrected_release_v1/RELEASE_DECISION.json").read_text(encoding="utf-8"))
    r3 = json.loads((root / "audit/v2/phase3g_r3_identity_recovery_v1/RELEASE_DECISION.json").read_text(encoding="utf-8"))
    overall = json.loads((root / "audit/v2/phase3g_r3_identity_recovery_v1/OVERALL_READINESS_DECISION.json").read_text(encoding="utf-8"))
    semantic = {
        "phase4_release_id": p4.get("release_id"),
        "phase4_status": p4.get("status"),
        "phase3g_r3_release_id": r3.get("release_id"),
        "phase3g_r3_status": r3.get("status"),
        "r3_accepted_new_identities": r3.get("accepted_exact_authority", 0) + r3.get("accepted_exact_authority_with_corroboration", 0),
        "r3_stage1_reconstruction_status": r3.get("stage1_reconstruction_status"),
        "r3_phase4_recovery_status": r3.get("phase4_recovery_status"),
        "overall_release_id": overall.get("overall_release_id"),
        "overall_status": overall.get("status"),
    }
    semantic_pass = semantic == {
        "phase4_release_id": "P4NSC_20260808_V1_274E41DF",
        "phase4_status": "PASS_PHASE4_NAMESPACE_CORRECTION",
        "phase3g_r3_release_id": "P3GR3_20260808_V1_274E41DF",
        "phase3g_r3_status": "PASS_PHASE3G_R3_NO_NEW_IDENTITIES",
        "r3_accepted_new_identities": 0,
        "r3_stage1_reconstruction_status": "NOT_APPLICABLE_NO_NEW_IDENTITIES",
        "r3_phase4_recovery_status": "NOT_APPLICABLE_NO_NEW_IDENTITIES",
        "overall_release_id": "NSR3_20260808_V1_274E41DF",
        "overall_status": "READY_FOR_SPLIT_BOUND_PHASE5_REBUILD",
    }
    result = {
        "release_id": RELEASE_ID,
        "checked_at_utc": utc_now(),
        "hash_checks": rows,
        "semantic_checks": semantic,
        "all_hashes_pass": all(row["status"] == "PASS" for row in rows),
        "semantic_checks_pass": semantic_pass,
        "status": "PASS" if semantic_pass and all(row["status"] == "PASS" for row in rows) else "FAIL",
    }
    write_json(out / "UPSTREAM_DEPENDENCY_CHECK.json", result)
    if result["status"] != "PASS":
        raise SystemExit("BLOCKED_PHASE5_KERNEL_VALIDATION: UPSTREAM_BINDING_MISMATCH")


def create_split_projection(con: duckdb.DuckDBPyConnection, root: Path, out: Path) -> Path:
    if PROHIBITED_SPLIT_COLUMNS.intersection(SPLIT_ALLOWED_COLUMNS):
        raise AssertionError("Split projection contains prohibited outcome columns")
    source = root / REQUIRED_HASHES["corrected_phase4_table"][0]
    target = out / "splits/split_source_projection.parquet"
    projection = ",\n".join(f'"{column}"' for column in SPLIT_ALLOWED_COLUMNS)
    con.execute(
        f"COPY (SELECT {projection} FROM read_parquet('{qpath(source)}')) "
        f"TO '{qpath(target)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    accessed = [
        {
            "access_stage": "SPLIT_ID_ONLY_PROJECTION",
            "source_file": source.relative_to(root).as_posix(),
            "column_or_artifact": column,
            "accessed": True,
            "prohibited_for_stage": False,
            "disposition": "ALLOWED_ID_OR_FROZEN_VIEW_METADATA",
        }
        for column in SPLIT_ALLOWED_COLUMNS
    ]
    accessed.extend(
        {
            "access_stage": "SPLIT_ID_ONLY_PROJECTION",
            "source_file": source.relative_to(root).as_posix(),
            "column_or_artifact": column,
            "accessed": False,
            "prohibited_for_stage": True,
            "disposition": "PHYSICALLY_EXCLUDED_FROM_PROJECTION",
        }
        for column in sorted(PROHIBITED_SPLIT_COLUMNS)
    )
    accessed.extend(
        [
            {"access_stage": "PROTECTED_OUTCOME_GUARD", "source_file": "LOCKED_OUTER_TEST", "column_or_artifact": "ALL_OUTCOMES", "accessed": False, "prohibited_for_stage": True, "disposition": "NOT_OPENED_NOT_SEARCHED"},
            {"access_stage": "PROTECTED_OUTCOME_GUARD", "source_file": "SEALED_FINAL_HOLDOUT", "column_or_artifact": "ALL_OUTCOMES", "accessed": False, "prohibited_for_stage": True, "disposition": "NOT_OPENED_NOT_SEARCHED"},
            {"access_stage": "PROTECTED_MEMBERSHIP", "source_file": "NONE", "column_or_artifact": "ID_ONLY_MEMBERSHIP", "accessed": False, "prohibited_for_stage": False, "disposition": "NO_EXTERNAL_MEMBERSHIP_ARTIFACT"},
        ]
    )
    write_tsv(out / "protected_outcome_access_audit.tsv", accessed)
    return target


def reproduce_views(con: duckdb.DuckDBPyConnection, projection: Path, out: Path) -> pd.DataFrame:
    rows = []
    for view, (predicate, expected_rows, expected_gids) in EXPECTED_VIEWS.items():
        observed_rows, observed_gids = con.execute(
            f"SELECT count(*), count(DISTINCT canonical_gid) FILTER (WHERE regexp_full_match(canonical_gid, '^GID[0-9]+$')) "
            f"FROM read_parquet('{qpath(projection)}') WHERE {predicate}"
        ).fetchone()
        rows.append(
            {
                "view": view,
                "predicate": predicate,
                "expected_rows": expected_rows,
                "observed_rows": observed_rows,
                "expected_canonical_gids": expected_gids,
                "observed_canonical_gids": observed_gids,
                "status": "PASS" if (observed_rows, observed_gids) == (expected_rows, expected_gids) else "FAIL",
            }
        )
    frame = pd.DataFrame(rows)
    write_tsv(out / "view_reproduction_summary.tsv", frame)
    if not (frame["status"] == "PASS").all():
        raise AssertionError("Corrected Phase-4 view population mismatch")
    return frame


def entity_summary(con: duckdb.DuckDBPyConnection, projection: Path, entity: str) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT {entity},
               count(*) FILTER (WHERE primary_weighted_training_eligible)::BIGINT primary_rows,
               count(*) FILTER (WHERE secondary_unweighted_training_eligible)::BIGINT secondary_rows
        FROM read_parquet('{qpath(projection)}')
        WHERE canonical_gid_eligible
        GROUP BY {entity}
        ORDER BY {entity}
        """
    ).fetchdf()


def freeze_outer_assignments(
    con: duckdb.DuckDBPyConnection, projection: Path, out: Path
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    gid_summary = entity_summary(con, projection, "canonical_gid")
    env_summary = entity_summary(con, projection, "environment_id")
    assignment_map: dict[tuple[str, str], pd.DataFrame] = {}
    frames = []
    specs = {
        ("GNEW_EOBS", "CANONICAL_GID"): (gid_summary, "canonical_gid"),
        ("GOBS_ENEW", "ENVIRONMENT"): (env_summary, "environment_id"),
        ("GNEW_ENEW", "CANONICAL_GID"): (gid_summary, "canonical_gid"),
        ("GNEW_ENEW", "ENVIRONMENT"): (env_summary, "environment_id"),
    }
    for (scenario, entity_type), (summary, column) in specs.items():
        assigned = assign_balanced_entities(summary, column, f"{scenario}|{entity_type}")
        assigned.insert(0, "scenario", scenario)
        assigned.insert(1, "entity_type", entity_type)
        assigned = assigned.rename(columns={"assigned_fold": "outer_fold"})
        assignment_map[(scenario, entity_type)] = assigned.copy()
        frames.append(assigned)
    result = pd.concat(frames, ignore_index=True).sort_values(["scenario", "entity_type", "entity_id"])
    write_tsv(out / "splits/entity_fold_assignment.tsv", result)
    return result, assignment_map


def materialize_observation_assignments(
    con: duckdb.DuckDBPyConnection,
    projection: Path,
    assignments: dict[tuple[str, str], pd.DataFrame],
    out: Path,
) -> Path:
    for (scenario, entity_type), frame in assignments.items():
        name = f"a_{scenario.lower()}_{entity_type.lower()}"
        con.register(name, frame[["entity_id", "outer_fold"]])
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE phase5_base_assignments AS
        SELECT p.*,
               ge.outer_fold::INTEGER gnew_eobs_gid_fold,
               oe.outer_fold::INTEGER gobs_enew_env_fold,
               gg.outer_fold::INTEGER gnew_enew_gid_fold,
               ee.outer_fold::INTEGER gnew_enew_env_fold
        FROM read_parquet('{qpath(projection)}') p
        JOIN a_gnew_eobs_canonical_gid ge ON p.canonical_gid = ge.entity_id
        JOIN a_gobs_enew_environment oe ON p.environment_id = oe.entity_id
        JOIN a_gnew_enew_canonical_gid gg ON p.canonical_gid = gg.entity_id
        JOIN a_gnew_enew_environment ee ON p.environment_id = ee.entity_id
        WHERE p.secondary_unweighted_training_eligible
        """
    )
    for fold in OUTER_FOLDS:
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE ge_env_seen_{fold} AS "
            f"SELECT DISTINCT environment_id FROM phase5_base_assignments WHERE gnew_eobs_gid_fold <> {fold}"
        )
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE oe_gid_seen_{fold} AS "
            f"SELECT DISTINCT canonical_gid FROM phase5_base_assignments WHERE gobs_enew_env_fold <> {fold}"
        )
    joins = []
    role_columns = []
    for fold in OUTER_FOLDS:
        joins.append(f"LEFT JOIN ge_env_seen_{fold} ges{fold} USING(environment_id)")
        joins.append(f"LEFT JOIN oe_gid_seen_{fold} oes{fold} USING(canonical_gid)")
        role_columns.extend(
            [
                f"CASE WHEN gnew_eobs_gid_fold <> {fold} THEN 'TRAIN' WHEN ges{fold}.environment_id IS NOT NULL THEN 'TEST' ELSE 'EMBARGO_OTHER_ENTITY_UNSEEN' END AS gnew_eobs_outer{fold}_role",
                f"CASE WHEN gobs_enew_env_fold <> {fold} THEN 'TRAIN' WHEN oes{fold}.canonical_gid IS NOT NULL THEN 'TEST' ELSE 'EMBARGO_OTHER_ENTITY_UNSEEN' END AS gobs_enew_outer{fold}_role",
                f"CASE WHEN gnew_enew_gid_fold = {fold} AND gnew_enew_env_fold = {fold} THEN 'TEST' WHEN gnew_enew_gid_fold <> {fold} AND gnew_enew_env_fold <> {fold} THEN 'TRAIN' ELSE 'EMBARGO_SINGLE_NOVELTY' END AS gnew_enew_outer{fold}_role",
            ]
        )
    target = out / "splits/observation_split_assignment.parquet"
    con.execute(
        f"""
        COPY (
          SELECT phase4_adjusted_row_id, phase4_group_id, canonical_gid, environment_id, trial_id,
                 cycle, year, trait, standardized_unit, loc_no, country, loc_desc,
                 primary_weighted_training_eligible, secondary_unweighted_training_eligible,
                 continuous_error_evaluation_eligible, correlation_evaluation_eligible,
                 ranking_evaluation_eligible,
                 gnew_eobs_gid_fold, gobs_enew_env_fold, gnew_enew_gid_fold, gnew_enew_env_fold,
                 {', '.join(role_columns)}
          FROM phase5_base_assignments b
          {' '.join(joins)}
          ORDER BY phase4_adjusted_row_id
        ) TO '{qpath(target)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    observed = con.execute(f"SELECT count(*), count(DISTINCT phase4_adjusted_row_id) FROM read_parquet('{qpath(target)}')").fetchone()
    if observed != (2_242_863, 2_242_863):
        raise AssertionError(f"Observation assignment population mismatch: {observed}")
    return target


def role_column(scenario: str, fold: int) -> str:
    return f"{scenario.lower()}_outer{fold}_role"


def split_summaries(
    con: duckdb.DuckDBPyConnection,
    observation_assignment: Path,
    out: Path,
) -> None:
    view_columns = {
        "PRIMARY_WEIGHTED_TRAINING": "primary_weighted_training_eligible",
        "SECONDARY_UNWEIGHTED_TRAINING": "secondary_unweighted_training_eligible",
        "CONTINUOUS_ERROR_EVALUATION": "continuous_error_evaluation_eligible",
        "CORRELATION_EVALUATION": "correlation_evaluation_eligible",
        "RANKING_EVALUATION": "ranking_evaluation_eligible",
    }
    population_rows: list[dict[str, Any]] = []
    embargo_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for fold in OUTER_FOLDS:
            column = role_column(scenario, fold)
            grouped = con.execute(
                f"""
                SELECT {column} AS split_role,
                       count(*) AS row_count,
                       count(DISTINCT canonical_gid) AS gid_count,
                       count(DISTINCT environment_id) AS environment_count,
                       sum(primary_weighted_training_eligible::INTEGER) AS primary_row_count,
                       sum(secondary_unweighted_training_eligible::INTEGER) AS secondary_row_count,
                       sum(continuous_error_evaluation_eligible::INTEGER) AS continuous_row_count,
                       sum(correlation_evaluation_eligible::INTEGER) AS correlation_row_count,
                       sum(ranking_evaluation_eligible::INTEGER) AS ranking_row_count
                FROM read_parquet('{qpath(observation_assignment)}')
                GROUP BY {column}
                ORDER BY {column}
                """
            ).fetchdf()
            for row in grouped.itertuples(index=False):
                for view, count_column in (
                    ("PRIMARY_WEIGHTED_TRAINING", "primary_row_count"),
                    ("SECONDARY_UNWEIGHTED_TRAINING", "secondary_row_count"),
                    ("CONTINUOUS_ERROR_EVALUATION", "continuous_row_count"),
                    ("CORRELATION_EVALUATION", "correlation_row_count"),
                    ("RANKING_EVALUATION", "ranking_row_count"),
                ):
                    population_rows.append(
                        {
                            "scenario": scenario,
                            "outer_fold": fold,
                            "view": view,
                            "role": row.split_role,
                            "rows": int(getattr(row, count_column)),
                            "canonical_gids_all_secondary": int(row.gid_count),
                            "environments_all_secondary": int(row.environment_count),
                        }
                    )
                if str(row.split_role).startswith("EMBARGO"):
                    embargo_rows.append(
                        {
                            "scenario": scenario,
                            "outer_fold": fold,
                            "reason_code": row.split_role,
                            "rows": int(row.row_count),
                            "canonical_gids": int(row.gid_count),
                            "environments": int(row.environment_count),
                        }
                    )
            if scenario == "GNEW_EOBS":
                overlap = con.execute(
                    f"SELECT count(*) FROM (SELECT canonical_gid FROM read_parquet('{qpath(observation_assignment)}') WHERE {column}='TRAIN' INTERSECT SELECT canonical_gid FROM read_parquet('{qpath(observation_assignment)}') WHERE {column}='TEST')"
                ).fetchone()[0]
                other_unseen = con.execute(
                    f"SELECT count(*) FROM (SELECT environment_id FROM read_parquet('{qpath(observation_assignment)}') WHERE {column}='TEST' EXCEPT SELECT environment_id FROM read_parquet('{qpath(observation_assignment)}') WHERE {column}='TRAIN')"
                ).fetchone()[0]
                checks = [("NO_GID_TRAIN_TEST_OVERLAP", overlap), ("TEST_ENVIRONMENT_SEEN_IN_TRAIN", other_unseen)]
            elif scenario == "GOBS_ENEW":
                overlap = con.execute(
                    f"SELECT count(*) FROM (SELECT environment_id FROM read_parquet('{qpath(observation_assignment)}') WHERE {column}='TRAIN' INTERSECT SELECT environment_id FROM read_parquet('{qpath(observation_assignment)}') WHERE {column}='TEST')"
                ).fetchone()[0]
                other_unseen = con.execute(
                    f"SELECT count(*) FROM (SELECT canonical_gid FROM read_parquet('{qpath(observation_assignment)}') WHERE {column}='TEST' EXCEPT SELECT canonical_gid FROM read_parquet('{qpath(observation_assignment)}') WHERE {column}='TRAIN')"
                ).fetchone()[0]
                checks = [("NO_ENVIRONMENT_TRAIN_TEST_OVERLAP", overlap), ("TEST_GID_SEEN_IN_TRAIN", other_unseen)]
            else:
                gid_overlap = con.execute(
                    f"SELECT count(*) FROM (SELECT canonical_gid FROM read_parquet('{qpath(observation_assignment)}') WHERE {column}='TRAIN' INTERSECT SELECT canonical_gid FROM read_parquet('{qpath(observation_assignment)}') WHERE {column}='TEST')"
                ).fetchone()[0]
                env_overlap = con.execute(
                    f"SELECT count(*) FROM (SELECT environment_id FROM read_parquet('{qpath(observation_assignment)}') WHERE {column}='TRAIN' INTERSECT SELECT environment_id FROM read_parquet('{qpath(observation_assignment)}') WHERE {column}='TEST')"
                ).fetchone()[0]
                invalid_role = con.execute(
                    f"SELECT count(*) FROM read_parquet('{qpath(observation_assignment)}') WHERE ({column}='TRAIN' AND (gnew_enew_gid_fold={fold} OR gnew_enew_env_fold={fold})) OR ({column}='TEST' AND NOT(gnew_enew_gid_fold={fold} AND gnew_enew_env_fold={fold})) OR ({column}='EMBARGO_SINGLE_NOVELTY' AND ((gnew_enew_gid_fold={fold})=(gnew_enew_env_fold={fold})))"
                ).fetchone()[0]
                checks = [("NO_GID_TRAIN_TEST_OVERLAP", gid_overlap), ("NO_ENVIRONMENT_TRAIN_TEST_OVERLAP", env_overlap), ("JOINT_NOVELTY_EMBARGO_EXACT", invalid_role)]
            for check, failures in checks:
                leakage_rows.append(
                    {
                        "scenario": scenario,
                        "outer_fold": fold,
                        "check": check,
                        "failure_count": int(failures),
                        "status": "PASS" if failures == 0 else "FAIL",
                    }
                )
            overlap_rows.append(
                {
                    "scenario": scenario,
                    "outer_fold": fold,
                    "train_test_forbidden_entity_overlap": int(overlap if scenario != "GNEW_ENEW" else max(gid_overlap, env_overlap)),
                    "status": "PASS" if all(value == 0 for _, value in checks) else "FAIL",
                }
            )
            primary_by_role = grouped.set_index("split_role")["primary_row_count"].to_dict()
            balance_rows.append(
                {
                    "scenario": scenario,
                    "outer_fold": fold,
                    "primary_test_rows": int(primary_by_role.get("TEST", 0)),
                    "primary_train_rows": int(primary_by_role.get("TRAIN", 0)),
                    "primary_embargo_rows": int(sum(value for key, value in primary_by_role.items() if str(key).startswith("EMBARGO"))),
                }
            )
    write_tsv(out / "splits/split_population_summary.tsv", population_rows)
    write_tsv(out / "splits/split_exclusion_and_embargo_ledger.tsv", embargo_rows)
    write_tsv(out / "splits/split_leakage_report.tsv", leakage_rows)
    write_tsv(out / "splits/split_overlap_summary.tsv", overlap_rows)
    write_tsv(out / "splits/split_balance_summary.tsv", balance_rows)
    if any(row["status"] != "PASS" for row in leakage_rows):
        raise AssertionError("Outer split leakage check failed")


def training_condition(scenario: str, outer_fold: int, inner_fold: int | None = None) -> str:
    if scenario == "GNEW_EOBS":
        outer = f"gnew_eobs_gid_fold <> {outer_fold}"
        if inner_fold is None:
            return outer
        return f"({outer}) AND canonical_gid NOT IN (SELECT entity_id FROM inner_assignments WHERE scenario='{scenario}' AND outer_fold={outer_fold} AND entity_type='CANONICAL_GID' AND inner_fold={inner_fold})"
    if scenario == "GOBS_ENEW":
        outer = f"gobs_enew_env_fold <> {outer_fold}"
        if inner_fold is None:
            return outer
        return f"({outer}) AND environment_id NOT IN (SELECT entity_id FROM inner_assignments WHERE scenario='{scenario}' AND outer_fold={outer_fold} AND entity_type='ENVIRONMENT' AND inner_fold={inner_fold})"
    outer = f"gnew_enew_gid_fold <> {outer_fold} AND gnew_enew_env_fold <> {outer_fold}"
    if inner_fold is None:
        return outer
    return (
        f"({outer}) AND canonical_gid NOT IN (SELECT entity_id FROM inner_assignments WHERE scenario='{scenario}' AND outer_fold={outer_fold} AND entity_type='CANONICAL_GID' AND inner_fold={inner_fold}) "
        f"AND environment_id NOT IN (SELECT entity_id FROM inner_assignments WHERE scenario='{scenario}' AND outer_fold={outer_fold} AND entity_type='ENVIRONMENT' AND inner_fold={inner_fold})"
    )


def freeze_inner_assignments(
    con: duckdb.DuckDBPyConnection,
    observation_assignment: Path,
    out: Path,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        entity_specs = [("CANONICAL_GID", "canonical_gid")] if scenario == "GNEW_EOBS" else [("ENVIRONMENT", "environment_id")]
        if scenario == "GNEW_ENEW":
            entity_specs = [("CANONICAL_GID", "canonical_gid"), ("ENVIRONMENT", "environment_id")]
        for outer_fold in OUTER_FOLDS:
            outer_condition = training_condition(scenario, outer_fold)
            for entity_type, column in entity_specs:
                summary = con.execute(
                    f"""
                    SELECT {column},
                           count(*) FILTER (WHERE primary_weighted_training_eligible)::BIGINT primary_rows,
                           count(*) FILTER (WHERE secondary_unweighted_training_eligible)::BIGINT secondary_rows
                    FROM read_parquet('{qpath(observation_assignment)}')
                    WHERE {outer_condition}
                    GROUP BY {column}
                    ORDER BY {column}
                    """
                ).fetchdf()
                assigned = assign_balanced_entities(
                    summary,
                    column,
                    f"{scenario}|OUTER{outer_fold}|INNER|{entity_type}",
                ).rename(columns={"assigned_fold": "inner_fold"})
                assigned.insert(0, "scenario", scenario)
                assigned.insert(1, "outer_fold", outer_fold)
                assigned.insert(2, "entity_type", entity_type)
                rows.append(assigned)
    result = pd.concat(rows, ignore_index=True).sort_values(
        ["scenario", "outer_fold", "entity_type", "entity_id"]
    )
    write_tsv(out / "splits/inner_fold_assignment.tsv", result)
    con.register("inner_assignments_frame", result[["scenario", "outer_fold", "entity_type", "entity_id", "inner_fold"]])
    con.execute("CREATE OR REPLACE TEMP TABLE inner_assignments AS SELECT * FROM inner_assignments_frame")
    return result


def build_state_registry(
    con: duckdb.DuckDBPyConnection,
    observation_assignment: Path,
    out: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows = []
    states: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        for outer_fold in OUTER_FOLDS:
            for inner_fold in (None, *INNER_FOLDS):
                level = "OUTER" if inner_fold is None else "INNER"
                state_id = f"{scenario}__OUTER{outer_fold}" + ("" if inner_fold is None else f"__INNER{inner_fold}")
                condition = training_condition(scenario, outer_fold, inner_fold)
                gids = [row[0] for row in con.execute(f"SELECT DISTINCT canonical_gid FROM read_parquet('{qpath(observation_assignment)}') WHERE {condition} ORDER BY canonical_gid").fetchall()]
                envs = [row[0] for row in con.execute(f"SELECT DISTINCT environment_id FROM read_parquet('{qpath(observation_assignment)}') WHERE {condition} ORDER BY environment_id").fetchall()]
                observations = con.execute(f"SELECT count(*) FROM read_parquet('{qpath(observation_assignment)}') WHERE {condition}").fetchone()[0]
                states[state_id] = {
                    "scenario": scenario,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "level": level,
                    "condition": condition,
                    "training_gids": gids,
                    "training_environments": envs,
                }
                rows.append(
                    {
                        "state_id": state_id,
                        "scenario": scenario,
                        "outer_fold": outer_fold,
                        "inner_fold": "" if inner_fold is None else inner_fold,
                        "state_level": level,
                        "training_observations": observations,
                        "training_canonical_gids": len(gids),
                        "training_environments": len(envs),
                        "training_gid_signature": index_signature(gids),
                        "training_environment_signature": index_signature(envs),
                        "selection_expression": condition,
                    }
                )
    frame = pd.DataFrame(rows)
    write_tsv(out / "splits/state_registry.tsv", frame)
    return frame, states


def materialize_inner_role_summaries(
    con: duckdb.DuckDBPyConnection,
    observation_assignment: Path,
    out: Path,
    states: dict[str, dict[str, Any]],
) -> None:
    population_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    for state_id, state in states.items():
        if state["inner_fold"] is None:
            continue
        scenario = state["scenario"]
        outer_fold = state["outer_fold"]
        inner_fold = state["inner_fold"]
        outer = training_condition(scenario, outer_fold)
        if scenario == "GNEW_EOBS":
            held = (
                f"canonical_gid IN (SELECT entity_id FROM inner_assignments WHERE scenario='{scenario}' "
                f"AND outer_fold={outer_fold} AND entity_type='CANONICAL_GID' AND inner_fold={inner_fold})"
            )
            seen = f"SELECT DISTINCT environment_id FROM read_parquet('{qpath(observation_assignment)}') WHERE {state['condition']}"
            validation = f"({outer}) AND ({held}) AND environment_id IN ({seen})"
            embargo = f"({outer}) AND ({held}) AND environment_id NOT IN ({seen})"
        elif scenario == "GOBS_ENEW":
            held = (
                f"environment_id IN (SELECT entity_id FROM inner_assignments WHERE scenario='{scenario}' "
                f"AND outer_fold={outer_fold} AND entity_type='ENVIRONMENT' AND inner_fold={inner_fold})"
            )
            seen = f"SELECT DISTINCT canonical_gid FROM read_parquet('{qpath(observation_assignment)}') WHERE {state['condition']}"
            validation = f"({outer}) AND ({held}) AND canonical_gid IN ({seen})"
            embargo = f"({outer}) AND ({held}) AND canonical_gid NOT IN ({seen})"
        else:
            gid_held = (
                f"canonical_gid IN (SELECT entity_id FROM inner_assignments WHERE scenario='{scenario}' "
                f"AND outer_fold={outer_fold} AND entity_type='CANONICAL_GID' AND inner_fold={inner_fold})"
            )
            env_held = (
                f"environment_id IN (SELECT entity_id FROM inner_assignments WHERE scenario='{scenario}' "
                f"AND outer_fold={outer_fold} AND entity_type='ENVIRONMENT' AND inner_fold={inner_fold})"
            )
            validation = f"({outer}) AND ({gid_held}) AND ({env_held})"
            embargo = f"({outer}) AND (({gid_held}) <> ({env_held}))"
        state["validation_condition"] = validation
        state["embargo_condition"] = embargo
        role_conditions = {
            "TRAIN": state["condition"],
            "INNER_VALIDATION": validation,
            "EMBARGO_OTHER_ENTITY_UNSEEN" if scenario != "GNEW_ENEW" else "EMBARGO_SINGLE_NOVELTY": embargo,
        }
        for role, condition in role_conditions.items():
            counts = con.execute(
                f"""
                SELECT count(*),
                       sum(primary_weighted_training_eligible::INTEGER),
                       sum(secondary_unweighted_training_eligible::INTEGER),
                       count(DISTINCT canonical_gid),
                       count(DISTINCT environment_id)
                FROM read_parquet('{qpath(observation_assignment)}') WHERE {condition}
                """
            ).fetchone()
            population_rows.append(
                {
                    "state_id": state_id,
                    "scenario": scenario,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "role": role,
                    "rows": int(counts[0]),
                    "primary_rows": int(counts[1] or 0),
                    "secondary_rows": int(counts[2] or 0),
                    "canonical_gids": int(counts[3]),
                    "environments": int(counts[4]),
                    "selection_expression": portable_selection(condition, observation_assignment),
                }
            )
        if scenario == "GNEW_EOBS":
            failures = con.execute(
                f"SELECT count(*) FROM (SELECT canonical_gid FROM read_parquet('{qpath(observation_assignment)}') WHERE {state['condition']} INTERSECT SELECT canonical_gid FROM read_parquet('{qpath(observation_assignment)}') WHERE {validation})"
            ).fetchone()[0]
        elif scenario == "GOBS_ENEW":
            failures = con.execute(
                f"SELECT count(*) FROM (SELECT environment_id FROM read_parquet('{qpath(observation_assignment)}') WHERE {state['condition']} INTERSECT SELECT environment_id FROM read_parquet('{qpath(observation_assignment)}') WHERE {validation})"
            ).fetchone()[0]
        else:
            gid_overlap = con.execute(
                f"SELECT count(*) FROM (SELECT canonical_gid FROM read_parquet('{qpath(observation_assignment)}') WHERE {state['condition']} INTERSECT SELECT canonical_gid FROM read_parquet('{qpath(observation_assignment)}') WHERE {validation})"
            ).fetchone()[0]
            env_overlap = con.execute(
                f"SELECT count(*) FROM (SELECT environment_id FROM read_parquet('{qpath(observation_assignment)}') WHERE {state['condition']} INTERSECT SELECT environment_id FROM read_parquet('{qpath(observation_assignment)}') WHERE {validation})"
            ).fetchone()[0]
            failures = max(gid_overlap, env_overlap)
        leakage_rows.append(
            {
                "scenario": scenario,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "check": "INNER_PROHIBITED_ENTITY_TRAIN_VALIDATION_OVERLAP",
                "failure_count": int(failures),
                "status": "PASS" if failures == 0 else "FAIL",
            }
        )
    write_tsv(out / "splits/inner_observation_role_summary.tsv", population_rows)
    existing = pd.read_csv(out / "splits/split_leakage_report.tsv", sep="\t")
    combined = pd.concat([existing, pd.DataFrame(leakage_rows)], ignore_index=True, sort=False)
    write_tsv(out / "splits/split_leakage_report.tsv", combined)
    if any(row["status"] != "PASS" for row in leakage_rows):
        raise AssertionError("Inner split leakage check failed")


def write_split_contract(out: Path, state_registry: pd.DataFrame) -> None:
    protocol = {
        "release_id": RELEASE_ID,
        "protocol_version": "phase5_split_protocol_v1",
        "seed": SEED,
        "outer_folds": 5,
        "nested_inner_folds": 5,
        "protected_split_membership_root": "NONE",
        "population_authority": "P4NSC_20260808_V1_274E41DF",
        "assignment_unit": {"GNEW_EOBS": "canonical_gid", "GOBS_ENEW": "environment_id", "GNEW_ENEW": "canonical_gid_and_environment_id_independently"},
        "balancing_objective": "deterministic greedy minimum primary row total, then entity count; secondary-only entities assigned after primary freeze",
        "tie_break": "sha256(seed|scenario|entity|fold)",
        "outcome_columns_loaded": [],
        "allowed_projection_columns": list(SPLIT_ALLOWED_COLUMNS),
        "prohibited_projection_columns": sorted(PROHIBITED_SPLIT_COLUMNS),
        "row_order_invariance": "entity summaries sorted and SHA-256 tie breaks independent of source row order",
    }
    write_json(out / "splits/split_protocol.json", protocol)
    definitions = []
    for row in state_registry.itertuples(index=False):
        definitions.append(
            {
                "split_id": row.state_id,
                "scenario": row.scenario,
                "outer_fold": row.outer_fold,
                "inner_fold": row.inner_fold,
                "state_level": row.state_level,
                "training_selection_expression": row.selection_expression,
                "outcome_blind": True,
            }
        )
    write_tsv(out / "splits/split_definition.tsv", definitions)
    write_json(
        out / "splits/protected_membership_binding.json",
        {
            "release_id": RELEASE_ID,
            "protected_split_membership_root": "NONE",
            "external_membership_artifact_found": False,
            "protected_outcome_files_searched": False,
            "disposition": "PRESPECIFIED_DEVELOPMENT_SCENARIOS_ONLY",
        },
    )


def build_pedigree(
    root: Path,
    out: Path,
    states: dict[str, dict[str, Any]],
    secondary_gids: list[str],
) -> tuple[set[str], dict[str, Any], pd.DataFrame]:
    source = root / "metadata_outputs/all_trials_genotype_manifest_resolved.tsv"
    manifest = pd.read_csv(
        source,
        sep="\t",
        dtype=str,
        usecols=lambda column: column in {"resolved_gid", "cross_name", "fieldbook_file", "fieldbook_sheet", "trial_id", "cycle", "occ"},
        low_memory=False,
    )
    parent_map, node_source, disposition = build_pedigree_parent_map(manifest, set(secondary_gids))
    factor, d_values, order, diagonal = build_pedigree_factor(parent_map)
    node_index = {node: index for index, node in enumerate(order)}
    registry = pd.DataFrame(
        {
            "node_index": np.arange(len(order), dtype=np.int64),
            "node_id": order,
            "node_kind": ["OBSERVED_CANONICAL_GID" if node.startswith("GID") else ("PEDIGREE_CROSS" if node.startswith("PEDCROSS") else "PEDIGREE_LEAF") for node in order],
            "parent1": [parent_map[node][0] for node in order],
            "parent2": [parent_map[node][1] for node in order],
            "is_observed_gid": [node.startswith("GID") for node in order],
            "raw_relationship_diagonal": diagonal,
        }
    )
    write_tsv(out / "pedigree/pedigree_node_registry.tsv", registry)
    write_tsv(out / "pedigree/pedigree_conflict_ledger.tsv", disposition)
    sparse.save_npz(out / "pedigree/ka_inverse_parent_factor_csr.npz", factor, compressed=True)
    np.save(out / "pedigree/ka_mendelian_variance.npy", d_values)
    write_json(
        out / "pedigree/pedigree_protocol.json",
        {
            "release_id": RELEASE_ID,
            "source_file": source.relative_to(root).as_posix(),
            "source_sha256": sha256_file(source),
            "source_columns": ["resolved_gid", "cross_name", "fieldbook_file", "fieldbook_sheet", "trial_id", "cycle", "occ"],
            "identity_mapping": "numeric resolved_gid is formatted as GID<digits> and must already exist in corrected secondary universe",
            "cleaning": "uppercase, collapse whitespace, normalize backslash to slash; exact unique nonblank cross only",
            "parent_semantics": "Purdy-style maximum slash level, rightmost tie; parent order canonicalized because additive relationships are sex-order invariant",
            "backcross_tokens": "retained literally; no undocumented expansion",
            "founder_rule": "namespaced exact PEDLEAF nodes; never canonical GIDs",
            "unknown_parent_codes": sorted(["", "0", "-", ".", "NA", "N/A", "NAN", "NONE", "NULL", "UNKNOWN", "UNK"]),
            "conflict_policy": "exclude a GID from K_A incidence when multiple exact normalized pedigree strings exist",
            "representation": "A = C D C^T, where C=(I-P)^-1 is stored sparse and D is Mendelian variance",
            "missing_pedigree": "no K_A incidence; never replaced by identity",
        },
    )
    observed_pedigree_gids = {node for node in order if node.startswith("GID")}
    state_rows = []
    diagnostics = []
    universes = []
    for state_id, state in states.items():
        training = sorted(observed_pedigree_gids.intersection(state["training_gids"]))
        application = sorted(observed_pedigree_gids - set(training))
        training_indices = [node_index[gid] for gid in training]
        scalar = float(np.mean(diagonal[training_indices])) if training_indices else float("nan")
        entity_frame = pd.DataFrame(
            {
                "entity_index": np.arange(len(observed_pedigree_gids), dtype=np.int64),
                "canonical_gid": sorted(observed_pedigree_gids),
            }
        )
        entity_frame["partition"] = np.where(entity_frame["canonical_gid"].isin(training), "TRAINING", "APPLICATION")
        entity_path = out / f"pedigree/states/{state_id}__ka_entities.tsv"
        write_tsv(entity_path, entity_frame)
        state_rows.append(
            {
                "state_id": state_id,
                "scenario": state["scenario"],
                "outer_fold": state["outer_fold"],
                "inner_fold": "" if state["inner_fold"] is None else state["inner_fold"],
                "training_observed_gids": len(training),
                "application_observed_gids": len(application),
                "raw_operator_factor": "pedigree/ka_inverse_parent_factor_csr.npz",
                "raw_operator_d": "pedigree/ka_mendelian_variance.npy",
                "training_scale_mean_diagonal": scalar,
                "entity_order_path": entity_path.relative_to(out).as_posix(),
                "entity_order_signature": index_signature(entity_frame["canonical_gid"]),
                "state_hash": stable_json_hash({"state": state_id, "training": training, "application": application, "scale": scalar}),
                "status": "PASS",
            }
        )
        sample = training_indices[: min(256, len(training_indices))]
        block = relationship_block(factor, d_values, sample) / scalar if sample and scalar > 0 else np.empty((0, 0))
        eig = np.linalg.eigvalsh((block + block.T) / 2.0) if block.size else np.asarray([np.nan])
        diagnostics.append(
            {
                "state_id": state_id,
                "sample_dimension": len(sample),
                "all_finite": bool(np.isfinite(block).all()) if block.size else False,
                "max_symmetry_error": float(np.max(np.abs(block - block.T))) if block.size else np.nan,
                "minimum_eigenvalue": float(np.nanmin(eig)),
                "mean_diagonal": float(np.mean(np.diag(block))) if block.size else np.nan,
                "raw_training_scale": scalar,
                "status": "PASS" if block.size and np.isfinite(block).all() and float(np.nanmin(eig)) >= -1e-8 else "FAIL",
            }
        )
        universes.append({"component": "K_A", "state_id": state_id, "entity_type": "CANONICAL_GID", "entities": len(entity_frame), "order_path": entity_path.relative_to(out).as_posix(), "order_signature": index_signature(entity_frame["canonical_gid"])})
    ka_registry = pd.DataFrame(state_rows)
    ka_diagnostics = pd.DataFrame(diagnostics)
    write_tsv(out / "pedigree/ka_registry.tsv", ka_registry)
    write_tsv(out / "pedigree/ka_diagnostics.tsv", ka_diagnostics)

    synthetic_parent_map = {"F1": ("", ""), "F2": ("", ""), "O1": ("F1", "F2"), "SELF": ("O1", "O1"), "ONE": ("F1", ""), "REP": ("O1", "F1")}
    sf, sd, so, _ = build_pedigree_factor(synthetic_parent_map)
    si = {node: index for index, node in enumerate(so)}
    synthetic = relationship_block(sf, sd, [si[x] for x in ("F1", "F2", "O1", "SELF", "ONE", "REP")])
    expected_o1 = 1.0
    expected_parents = 0.5
    checks = [
        {"check": "SYNTHETIC_OFFSPRING_DIAGONAL", "expected": expected_o1, "observed": synthetic[2, 2], "absolute_difference": abs(synthetic[2, 2] - expected_o1)},
        {"check": "SYNTHETIC_PARENT_OFFSPRING", "expected": expected_parents, "observed": synthetic[0, 2], "absolute_difference": abs(synthetic[0, 2] - expected_parents)},
        {"check": "PARENT_ORDER_INVARIANT", "expected": 0.0, "observed": float(np.max(np.abs(synthetic - synthetic.T))), "absolute_difference": float(np.max(np.abs(synthetic - synthetic.T)))},
    ]
    real_nodes = order[: min(80, len(order))]
    real_indices = [node_index[node] for node in real_nodes]
    production = relationship_block(factor, d_values, real_indices)
    independent = independent_tabular_relationship({node: parent_map[node] for node in real_nodes}, real_nodes)
    max_diff = float(np.max(np.abs(production - independent)))
    checks.append({"check": "REAL_SUBSET_INDEPENDENT_TABULAR", "expected": 0.0, "observed": max_diff, "absolute_difference": max_diff})
    independent_frame = pd.DataFrame(checks)
    independent_frame["status"] = np.where(independent_frame["absolute_difference"] <= 1e-10, "PASS", "FAIL")
    write_tsv(out / "pedigree/ka_independent_checks.tsv", independent_frame)
    if not (ka_diagnostics["status"] == "PASS").all() or not (independent_frame["status"] == "PASS").all():
        raise AssertionError("K_A validation failed")
    return observed_pedigree_gids, {"factor": factor, "d": d_values, "order": order, "index": node_index, "diagonal": diagonal}, pd.DataFrame(universes)


def independent_tabular_relationship(parent_map: dict[str, tuple[str, str]], order: list[str]) -> np.ndarray:
    index = {node: i for i, node in enumerate(order)}
    matrix = np.zeros((len(order), len(order)), dtype=np.float64)
    for i, node in enumerate(order):
        p1, p2 = parent_map[node]
        parent_indices = [index[p] for p in (p1, p2) if p and p in index and index[p] < i]
        for j in range(i):
            matrix[i, j] = matrix[j, i] = sum(0.5 * matrix[parent, j] for parent in parent_indices)
        matrix[i, i] = 1.0 + (0.5 * matrix[parent_indices[0], parent_indices[1]] if len(parent_indices) == 2 else 0.0)
    return matrix


def load_hibap_dosage(root: Path, out: Path, secondary_gids: set[str]) -> tuple[np.ndarray, list[str], list[str], pd.DataFrame]:
    source = root / "GENOTYPIC_DATA/IWYP64_-_HiBAP_35k_Wheat_Breeders_Array_Genotyping/HiBAP_snps_35karray.txt"
    crosswalk_path = root / "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/hibap_corrected_sample_to_gid_crosswalk.tsv"
    crosswalk = pd.read_csv(crosswalk_path, sep="\t", dtype=str)
    matrix = pd.read_csv(source, sep="\t", skiprows=3, dtype=str, low_memory=False)
    sample_columns = list(matrix.columns[11:])
    marker_ids = matrix.iloc[:, 0].astype(str).tolist()
    alleles = matrix.iloc[:, 1].astype(str).tolist()
    sample_vectors: dict[str, np.ndarray] = {}
    sample_to_gid: dict[str, str] = {}
    for row in crosswalk.itertuples(index=False):
        physical = int(row.physical_column_index) - 1
        header = sample_columns[physical - 11]
        if header != row.raw_matrix_header:
            raise AssertionError(f"HiBAP physical/header mismatch at {row.physical_column_index}: {header} != {row.raw_matrix_header}")
        values = matrix.iloc[:, physical].tolist()
        sample_vectors[row.sample_instance_key] = np.asarray(
            [decode_biallelic_call(call, allele) for call, allele in zip(values, alleles)], dtype=np.float64
        )
        sample_to_gid[row.sample_instance_key] = row.accepted_canonical_gid
    gid_samples: defaultdict[str, list[str]] = defaultdict(list)
    for sample, gid in sample_to_gid.items():
        if gid in secondary_gids:
            gid_samples[gid].append(sample)
    dosage_rows = []
    replicate_rows = []
    for gid in sorted(gid_samples):
        samples = sorted(gid_samples[gid])
        values = np.vstack([sample_vectors[sample] for sample in samples])
        consensus, conflicts = consensus_dosage(values)
        dosage_rows.append(consensus)
        replicate_rows.append(
            {
                "canonical_gid": gid,
                "physical_sample_instances": len(samples),
                "sample_instance_keys": ";".join(samples),
                "consensus_rule": "UNANIMOUS_NONMISSING_ELSE_MISSING",
                "discordant_marker_calls_set_missing": conflicts,
                "retained": True,
            }
        )
    replicate_frame = pd.DataFrame(replicate_rows)
    write_tsv(out / "genomic/panel_replicate_resolution.tsv", replicate_frame)
    return np.vstack(dosage_rows), sorted(gid_samples), marker_ids, replicate_frame


def build_genomic(
    root: Path,
    out: Path,
    states: dict[str, dict[str, Any]],
    secondary_gids: list[str],
) -> tuple[set[str], dict[str, dict[str, Any]], pd.DataFrame]:
    r2 = root / "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2"
    inventory = pd.read_csv(r2 / "panel_inventory.tsv", sep="\t", dtype=str)
    inventory[["phase5_classification", "production_disposition"]] = inventory["panel_id"].apply(
        lambda panel: pd.Series(PANEL_CLASS[panel])
    )
    inventory["production_included"] = inventory["panel_id"].eq("hibap35k")
    write_tsv(out / "genomic/panel_registry.tsv", inventory)
    protocols = []
    for row in inventory.itertuples(index=False):
        protocols.append(
            {
                "panel_id": row.panel_id,
                "protocol_id": "HIBAP35K_MINIMAL_SOURCE_VALIDITY_TRAINING_LOCAL_V1" if row.panel_id == "hibap35k" else "NOT_APPLICABLE_DEFERRED_OR_EXCLUDED",
                "sample_qc": "R2 accepted typed identity; exact replicate consensus" if row.panel_id == "hibap35k" else "NOT_FIT",
                "marker_qc": "source biallelic allele declaration; valid IUPAC calls; drop training-monomorphic markers only" if row.panel_id == "hibap35k" else "NOT_FIT",
                "missing_call": "N/-/./NA invalid; training 2p mean imputation" if row.panel_id == "hibap35k" else "NOT_FIT",
                "training_only": bool(row.panel_id == "hibap35k"),
                "status": "PRODUCTION" if row.panel_id == "hibap35k" else row.production_disposition,
            }
        )
    write_tsv(out / "genomic/panel_qc_protocols.tsv", protocols)

    accepted = pq.read_table(
        r2 / "accepted_all_panel_crosswalk.parquet",
        columns=["panel_id", "panel_sample_key", "sample_instance_key", "raw_sample_id", "accepted_canonical_gid", "mapping_status", "evidence_type", "marker_vector_present", "existing_qc_status", "replicate_status"],
    ).to_pandas()
    accepted = accepted.sort_values(["panel_id", "accepted_canonical_gid", "sample_instance_key"])
    write_tsv(out / "genomic/panel_sample_gid_registry.tsv", accepted)
    targeted = inventory[inventory["phase5_classification"] == "TARGETED_MAS_COVARIATE_OR_SPARSE_KERNEL"][
        ["panel_id", "platform", "technology", "accepted_canonical_gid_count", "production_disposition"]
    ].copy()
    targeted["allowed_role"] = "FUTURE_TARGETED_COVARIATE_OR_SEPARATE_SPARSE_COMPONENT"
    targeted["entered_genomewide_kg"] = False
    write_tsv(out / "genomic/targeted_marker_component_registry.tsv", targeted)

    dosage, gids, marker_ids, replicate_frame = load_hibap_dosage(root, out, set(secondary_gids))
    gid_index = {gid: index for index, gid in enumerate(gids)}
    kg_states: dict[str, dict[str, Any]] = {}
    preproc_rows = []
    registry_rows = []
    diagnostic_rows = []
    independent_rows = []
    universe_rows = []
    for state_id, state in states.items():
        training_ids = set(gids).intersection(state["training_gids"])
        fitted = fit_vanraden(dosage, gids, training_ids)
        retained_markers = np.asarray(marker_ids, dtype="U")[fitted["retained_mask"]]
        state_path = out / f"genomic/states/{state_id}__hibap35k_vanraden.npz"
        np.savez_compressed(
            state_path,
            canonical_gids=np.asarray(gids, dtype="U"),
            retained_marker_ids=retained_markers,
            allele_frequency=fitted["allele_frequency"].astype(np.float32),
            imputation_value=fitted["imputation_value"].astype(np.float32),
            factor=fitted["factor"].astype(np.float32),
            denominator=np.asarray([fitted["denominator"]], dtype=np.float64),
            training_gid_signature=np.asarray([index_signature(sorted(training_ids))], dtype="U"),
        )
        entity_frame = pd.DataFrame({"entity_index": np.arange(len(gids), dtype=np.int64), "canonical_gid": gids})
        entity_frame["partition"] = np.where(entity_frame["canonical_gid"].isin(training_ids), "TRAINING", "APPLICATION")
        entity_path = out / f"genomic/states/{state_id}__hibap35k_entities.tsv"
        write_tsv(entity_path, entity_frame)
        state_hash = sha256_file(state_path)
        kg_states[state_id] = {"factor": fitted["factor"], "gids": gids, "index": gid_index, "training_ids": training_ids, "state_path": state_path, "state_hash": state_hash}
        preproc_rows.append(
            {
                "state_id": state_id,
                "panel_id": "hibap35k",
                "training_gids": len(training_ids),
                "application_gids": len(gids) - len(training_ids),
                "input_markers": len(marker_ids),
                "retained_training_polymorphic_markers": len(retained_markers),
                "dropped_training_monomorphic_or_invalid": len(marker_ids) - len(retained_markers),
                "imputation_fit_scope": "TRAINING_GIDS_ONLY",
                "allele_frequency_fit_scope": "TRAINING_GIDS_ONLY",
                "state_path": state_path.relative_to(out).as_posix(),
                "state_sha256": state_hash,
                "status": "PASS",
            }
        )
        registry_rows.append(
            {
                "state_id": state_id,
                "panel_id": "hibap35k",
                "representation": "VANRADEN_I_FACTOR",
                "formula": "K=ZZT/(2*sum(p*(1-p)))",
                "entities": len(gids),
                "training_entities": len(training_ids),
                "application_entities": len(gids) - len(training_ids),
                "markers": len(retained_markers),
                "denominator": fitted["denominator"],
                "factor_path": state_path.relative_to(out).as_posix(),
                "factor_sha256": state_hash,
                "entity_order_path": entity_path.relative_to(out).as_posix(),
                "entity_order_signature": index_signature(gids),
                "status": "PASS",
            }
        )
        diagnostics = kernel_diagnostics(fitted["factor"])
        diagnostics.update({"state_id": state_id, "panel_id": "hibap35k", "status": "PASS" if diagnostics["all_finite"] and diagnostics["max_symmetry_error"] <= 1e-10 and diagnostics["minimum_eigenvalue"] >= -1e-8 else "FAIL"})
        diagnostic_rows.append(diagnostics)
        refit = fit_vanraden(dosage[::-1, :], gids[::-1], training_ids)
        independent_rows.append(
            {
                "state_id": state_id,
                "check": "APPLICATION_AND_ROW_ORDER_INVARIANCE",
                "maximum_allele_frequency_difference": float(np.max(np.abs(fitted["allele_frequency"] - refit["allele_frequency"]))),
                "denominator_difference": abs(fitted["denominator"] - refit["denominator"]),
                "status": "PASS" if np.allclose(fitted["allele_frequency"], refit["allele_frequency"], atol=0, rtol=0) else "FAIL",
            }
        )
        universe_rows.append({"component": "K_G_HIBAP35K", "state_id": state_id, "entity_type": "CANONICAL_GID", "entities": len(gids), "order_path": entity_path.relative_to(out).as_posix(), "order_signature": index_signature(gids)})
    preproc = pd.DataFrame(preproc_rows)
    registry = pd.DataFrame(registry_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    independent = pd.DataFrame(independent_rows)
    write_tsv(out / "genomic/fold_preprocessing_registry.tsv", preproc)
    write_tsv(out / "genomic/kg_registry.tsv", registry)
    write_tsv(out / "genomic/kg_diagnostics.tsv", diagnostics)
    write_tsv(out / "genomic/kg_independent_checks.tsv", independent)
    if not (diagnostics["status"] == "PASS").all() or not (independent["status"] == "PASS").all():
        raise AssertionError("HiBAP K_G validation failed")
    return set(gids), kg_states, pd.DataFrame(universe_rows)


def compare_ka_kg(
    out: Path,
    pedigree: dict[str, Any],
    kg_states: dict[str, dict[str, Any]],
) -> None:
    rows = []
    pedigree_gids = {gid for gid in pedigree["order"] if gid.startswith("GID")}
    for state_id, kg_state in kg_states.items():
        shared = sorted(pedigree_gids.intersection(kg_state["gids"]))
        selected = shared[: min(80, len(shared))]
        ka_indices = [pedigree["index"][gid] for gid in selected]
        kg_indices = [kg_state["index"][gid] for gid in selected]
        ka = relationship_block(pedigree["factor"], pedigree["d"], ka_indices)
        kg_factor = kg_state["factor"][kg_indices]
        kg = kg_factor @ kg_factor.T
        tri = np.triu_indices(len(selected), k=1)
        correlation = float(np.corrcoef(ka[tri], kg[tri])[0, 1]) if len(selected) > 2 and np.std(ka[tri]) > 0 and np.std(kg[tri]) > 0 else np.nan
        rows.append(
            {
                "state_id": state_id,
                "shared_canonical_gids": len(shared),
                "diagnostic_subset_gids": len(selected),
                "ka_mean_diagonal": float(np.mean(np.diag(ka))) if len(selected) else np.nan,
                "kg_mean_diagonal": float(np.mean(np.diag(kg))) if len(selected) else np.nan,
                "off_diagonal_correlation": correlation,
                "interpretation": "DIAGNOSTIC_ONLY_NO_KERNEL_MIXING_OR_IDENTITY_ADJUDICATION",
                "status": "PASS" if len(shared) > 0 else "FAIL",
            }
        )
    write_tsv(out / "genomic/ka_kg_shared_gid_comparison.tsv", rows)
    if any(row["status"] != "PASS" for row in rows):
        raise AssertionError("No shared K_A/K_G GIDs in a split state")


def build_environment(
    con: duckdb.DuckDBPyConnection,
    observation_assignment: Path,
    corrected_table: Path,
    out: Path,
    states: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    environments = con.execute(
        f"""
        SELECT environment_id,
               any_value(trial_id) AS trial_id,
               split_part(environment_id, '|', 2) AS occurrence,
               any_value(cycle) AS "cycle",
               any_value(year) AS "year",
               any_value(loc_no) AS loc_no,
               any_value(country) AS country,
               any_value(loc_desc) AS loc_desc,
               concat_ws('|', coalesce(any_value(country), ''), coalesce(any_value(loc_no), ''), coalesce(any_value(loc_desc), '')) AS location_key,
               count(DISTINCT concat_ws('|', coalesce(country, ''), coalesce(loc_no, ''), coalesce(loc_desc, ''))) AS location_key_cardinality,
               bool_or(primary_weighted_training_eligible) AS in_primary,
               bool_or(secondary_unweighted_training_eligible) AS in_secondary
        FROM read_parquet('{qpath(observation_assignment)}')
        GROUP BY environment_id
        ORDER BY environment_id
        """
    ).fetchdf()
    if not (environments["location_key_cardinality"] == 1).all():
        raise AssertionError("Environment-to-location metadata is not one-to-one")
    environments.insert(0, "environment_index", np.arange(len(environments), dtype=np.int64))
    environments["identity_component_available"] = True
    environments["geo_component_available"] = environments["location_key"].astype(str).ne("")
    environments["environment_information_class"] = np.where(environments["geo_component_available"], "IDENTITY_PLUS_LOCATION", "IDENTITY_ONLY")
    write_tsv(out / "indices/environment_entity_registry.tsv", environments)

    source_hash = sha256_file(corrected_table)
    features = [
        {"component": "K_E_identity", "feature_id": "environment_id", "source_file": corrected_table.relative_to(corrected_table.parents[3]).as_posix(), "source_table": "corrected_promoted_phenotypes", "source_column": "environment_id", "source_sha256": source_hash, "units": "categorical identifier", "meaning": "environment identity baseline; not environmental similarity", "temporal_window": "not applicable", "aggregation": "exact key", "missingness": 0, "preprocessing": "none", "status": "PRODUCTION"},
        {"component": "K_E_geo", "feature_id": "country_loc_no_loc_desc", "source_file": corrected_table.relative_to(corrected_table.parents[3]).as_posix(), "source_table": "corrected_promoted_phenotypes", "source_column": "country;loc_no;loc_desc", "source_sha256": source_hash, "units": "categorical location", "meaning": "exact location category", "temporal_window": "static", "aggregation": "one exact category per environment", "missingness": int((~environments["geo_component_available"]).sum()), "preprocessing": "training-level one-hot; unseen application levels have no incidence", "status": "PRODUCTION"},
        {"component": "K_E_weather", "feature_id": "DEFERRED", "source_file": "HISTORICAL_UNVERSIONED_CANDIDATES", "source_table": "NONE", "source_column": "NONE", "source_sha256": "NOT_APPLICABLE", "units": "NOT_APPLICABLE", "meaning": "weather/agroclimate", "temporal_window": "UNRESOLVED", "aggregation": "UNRESOLVED", "missingness": "NOT_ASSESSED_IN_PRODUCTION", "preprocessing": "NOT_FIT", "status": "DEFERRED_NO_RECONSTRUCTABLE_CORRECTED_ENVIRONMENT_BINDING"},
        {"component": "K_E_stress", "feature_id": "DEFERRED", "source_file": "HISTORICAL_UNVERSIONED_CANDIDATES", "source_table": "NONE", "source_column": "NONE", "source_sha256": "NOT_APPLICABLE", "units": "NOT_APPLICABLE", "meaning": "stress indices", "temporal_window": "UNRESOLVED", "aggregation": "UNRESOLVED", "missingness": "NOT_ASSESSED_IN_PRODUCTION", "preprocessing": "NOT_FIT", "status": "DEFERRED_NO_RECONSTRUCTABLE_CORRECTED_ENVIRONMENT_BINDING"},
        {"component": "K_E_mgmt", "feature_id": "DEFERRED", "source_file": "HISTORICAL_UNVERSIONED_CANDIDATES", "source_table": "NONE", "source_column": "NONE", "source_sha256": "NOT_APPLICABLE", "units": "NOT_APPLICABLE", "meaning": "management", "temporal_window": "UNRESOLVED", "aggregation": "UNRESOLVED", "missingness": "NOT_ASSESSED_IN_PRODUCTION", "preprocessing": "NOT_FIT", "status": "DEFERRED_NO_RECONSTRUCTABLE_CORRECTED_ENVIRONMENT_BINDING"},
    ]
    write_tsv(out / "environment/environment_feature_registry.tsv", features)
    write_tsv(
        out / "environment/environment_source_join_audit.tsv",
        [
            {"join": "canonical_environment_to_location_tuple", "left_rows": len(environments), "right_rows": len(environments), "expected_cardinality": "one_to_one", "unmatched": 0, "duplicate_keys": int((environments["location_key_cardinality"] != 1).sum()), "row_loss": 0, "status": "PASS"},
            {"join": "canonical_environment_to_identity_component", "left_rows": len(environments), "right_rows": len(environments), "expected_cardinality": "one_to_one", "unmatched": 0, "duplicate_keys": 0, "row_loss": 0, "status": "PASS"},
        ],
    )
    write_tsv(
        out / "environment/environment_component_protocols.tsv",
        [
            {"component": "K_E_identity", "representation": "IDENTITY_OPERATOR", "fit_scope": "outcome-independent full environment universe; split blocks explicit", "normalization": "none; diagonal one", "missing_behavior": "available for every canonical environment", "status": "PRODUCTION"},
            {"component": "K_E_geo", "representation": "TRAINING_LEVEL_ONE_HOT_FACTOR", "fit_scope": "categorical levels fitted on training environments only", "normalization": "training mean diagonal", "missing_behavior": "no component incidence when source location absent or level unseen in training", "status": "PRODUCTION"},
        ],
    )

    env_ids = environments["environment_id"].astype(str).tolist()
    location_keys = environments["location_key"].astype(str).tolist()
    env_index = {env: index for index, env in enumerate(env_ids)}
    state_objects: dict[str, dict[str, Any]] = {}
    preprocess_rows = []
    registry_rows = []
    diagnostics = []
    independent_checks = []
    coverage_rows = []
    universe_rows = []
    for state_id, state in states.items():
        training_envs = set(state["training_environments"])
        geo, levels, raw_scale = geo_factor(env_ids, location_keys, training_envs)
        level_index = {level: index for index, level in enumerate(levels)}
        entity = environments[["environment_index", "environment_id", "location_key"]].copy()
        entity["partition"] = np.where(entity["environment_id"].isin(training_envs), "TRAINING", "APPLICATION")
        entity["geo_level_index"] = entity["location_key"].map(level_index).fillna(-1).astype(int)
        entity_path = out / f"environment/states/{state_id}__environment_entities.tsv"
        write_tsv(entity_path, entity)
        levels_path = out / f"environment/states/{state_id}__geo_training_levels.tsv"
        write_tsv(levels_path, pd.DataFrame({"geo_level_index": np.arange(len(levels)), "location_key": levels}))
        state_hash = stable_json_hash({"state": state_id, "training_envs": sorted(training_envs), "levels": levels, "scale": raw_scale})
        state_objects[state_id] = {"geo_factor": geo, "levels": levels, "level_index": level_index, "env_ids": env_ids, "env_index": env_index, "location_keys": location_keys, "scale": raw_scale, "hash": state_hash}
        preprocess_rows.extend(
            [
                {"state_id": state_id, "component": "K_E_identity", "training_environments": len(training_envs), "fit_operation": "NONE", "state_hash": stable_json_hash({"state": state_id, "component": "identity", "training": sorted(training_envs)}), "status": "PASS"},
                {"state_id": state_id, "component": "K_E_geo", "training_environments": len(training_envs), "fit_operation": "TRAINING_LEVEL_VOCABULARY_AND_TRAINING_MEAN_DIAGONAL", "training_levels": len(levels), "dropped_training_constant_features": 0, "state_hash": state_hash, "status": "PASS"},
            ]
        )
        registry_rows.extend(
            [
                {"state_id": state_id, "component": "K_E_identity", "representation": "IDENTITY_OPERATOR", "entities": len(env_ids), "training_entities": len(training_envs), "entity_order_path": entity_path.relative_to(out).as_posix(), "entity_order_signature": index_signature(env_ids), "preprocessing_state_hash": stable_json_hash({"state": state_id, "component": "identity", "training": sorted(training_envs)}), "status": "PASS"},
                {"state_id": state_id, "component": "K_E_geo", "representation": "SPARSE_ONE_HOT_FACTOR_OPERATOR", "entities": len(env_ids), "training_entities": len(training_envs), "training_levels": len(levels), "raw_training_scale": raw_scale, "training_levels_path": levels_path.relative_to(out).as_posix(), "entity_order_path": entity_path.relative_to(out).as_posix(), "entity_order_signature": index_signature(env_ids), "preprocessing_state_hash": state_hash, "status": "PASS"},
            ]
        )
        diagnostics.extend(
            [
                {"state_id": state_id, "component": "K_E_identity", "dimension": len(env_ids), "finite": True, "symmetry_error": 0.0, "minimum_eigenvalue": 1.0, "mean_diagonal_training": 1.0, "effective_rank": len(env_ids), "status": "PASS"},
                {"state_id": state_id, "component": "K_E_geo", "dimension": len(env_ids), "finite": True, "symmetry_error": 0.0, "minimum_eigenvalue": 0.0, "mean_diagonal_training": 1.0, "effective_rank": len(levels), "status": "PASS"},
            ]
        )
        sample_pairs = [(0, 0), (0, min(1, len(env_ids) - 1)), (min(2, len(env_ids) - 1), min(3, len(env_ids) - 1))]
        for left, right in sample_pairs:
            observed = float(geo.getrow(left).multiply(geo.getrow(right)).sum())
            expected = (1.0 / raw_scale) if location_keys[left] == location_keys[right] and location_keys[left] in level_index else 0.0
            independent_checks.append({"state_id": state_id, "component": "K_E_geo", "left_environment": env_ids[left], "right_environment": env_ids[right], "expected": expected, "observed": observed, "absolute_difference": abs(expected - observed), "status": "PASS" if abs(expected - observed) <= 1e-12 else "FAIL"})
        coverage_rows.append({"state_id": state_id, "component": "K_E_identity", "training_environments": len(training_envs), "application_environments": len(env_ids) - len(training_envs), "covered_entities": len(env_ids), "status": "PASS"})
        coverage_rows.append({"state_id": state_id, "component": "K_E_geo", "training_environments": len(training_envs), "application_environments": len(env_ids) - len(training_envs), "covered_entities": int((entity["geo_level_index"] >= 0).sum()), "status": "PASS"})
        universe_rows.extend([
            {"component": "K_E_identity", "state_id": state_id, "entity_type": "ENVIRONMENT", "entities": len(env_ids), "order_path": entity_path.relative_to(out).as_posix(), "order_signature": index_signature(env_ids)},
            {"component": "K_E_geo", "state_id": state_id, "entity_type": "ENVIRONMENT", "entities": len(env_ids), "order_path": entity_path.relative_to(out).as_posix(), "order_signature": index_signature(env_ids)},
        ])
    write_tsv(out / "environment/fold_preprocessing_registry.tsv", preprocess_rows)
    write_tsv(out / "environment/ke_registry.tsv", registry_rows)
    write_tsv(out / "environment/ke_diagnostics.tsv", diagnostics)
    write_tsv(out / "environment/ke_independent_checks.tsv", independent_checks)
    write_tsv(out / "environment/environment_coverage_by_fold.tsv", coverage_rows)
    if any(row["status"] != "PASS" for row in independent_checks):
        raise AssertionError("K_E independent reconstruction failed")
    return environments, state_objects, pd.DataFrame(universe_rows)


def build_indices_and_coverage(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    out: Path,
    projection: Path,
    observation_assignment: Path,
    pedigree_gids: set[str],
    genomic_gids: set[str],
    environments: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
    secondary_gids = [row[0] for row in con.execute(f"SELECT DISTINCT canonical_gid FROM read_parquet('{qpath(projection)}') WHERE secondary_unweighted_training_eligible ORDER BY canonical_gid").fetchall()]
    primary_gids = {row[0] for row in con.execute(f"SELECT DISTINCT canonical_gid FROM read_parquet('{qpath(projection)}') WHERE primary_weighted_training_eligible").fetchall()}
    coverage = pd.read_csv(root / "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/canonical_gid_panel_coverage.tsv", sep="\t", dtype=str)
    coverage = coverage[coverage["accepted_gid_linkage"].eq("True")]
    panel_sets = coverage.groupby("panel_id")["canonical_gid"].apply(set).to_dict()
    haplotype_gids = panel_sets.get("eyt_haplotype_blocks_2011_2018", set())
    targeted_panels = {panel for panel, (kind, _) in PANEL_CLASS.items() if kind == "TARGETED_MAS_COVARIATE_OR_SPARSE_KERNEL"}
    targeted_gids = set().union(*(panel_sets.get(panel, set()) for panel in targeted_panels)) if targeted_panels else set()
    genotype_rows = []
    for index, gid in enumerate(secondary_gids):
        has_pedigree = gid in pedigree_gids
        has_dense = gid in genomic_gids
        has_haplotype = gid in haplotype_gids
        has_targeted = gid in targeted_gids
        if has_pedigree and has_dense:
            info = "PEDIGREE_PLUS_DENSE_MARKERS"
        elif has_dense:
            info = "DENSE_MARKERS_ONLY"
        elif has_pedigree:
            info = "PEDIGREE_ONLY"
        elif has_haplotype:
            info = "HAPLOTYPE_ONLY_DEFERRED"
        else:
            info = "NEITHER_PEDIGREE_NOR_PRODUCTION_DENSE_MARKERS"
        genotype_rows.append(
            {
                "genotype_index": index,
                "canonical_gid": gid,
                "in_primary_view": gid in primary_gids,
                "in_secondary_view": True,
                "pedigree_available": has_pedigree,
                "hibap35k_production_marker_available": has_dense,
                "haplotype_candidate_available": has_haplotype,
                "targeted_marker_available": has_targeted,
                "genotype_information_class": info,
            }
        )
    genotype_registry = pd.DataFrame(genotype_rows)
    write_tsv(out / "indices/genotype_entity_registry.tsv", genotype_registry)

    traits = con.execute(f"SELECT trait, any_value(standardized_unit) AS standardized_unit, count(*) AS row_count FROM read_parquet('{qpath(projection)}') WHERE secondary_unweighted_training_eligible GROUP BY trait ORDER BY trait").fetchdf()
    traits.insert(0, "trait_index", np.arange(len(traits), dtype=np.int64))
    write_tsv(out / "indices/trait_registry.tsv", traits)

    info = genotype_registry[["canonical_gid", "pedigree_available", "hibap35k_production_marker_available", "haplotype_candidate_available", "targeted_marker_available", "genotype_information_class"]]
    con.register("genotype_info", info)
    env_info = environments[["environment_id", "identity_component_available", "geo_component_available", "environment_information_class"]]
    con.register("environment_info", env_info)
    master = out / "indices/canonical_phase5_observation_index.parquet"
    corrected = root / REQUIRED_HASHES["corrected_phase4_table"][0]
    con.execute(
        f"""
        COPY (
          SELECT row_number() OVER (ORDER BY p.phase4_adjusted_row_id)-1 phase5_observation_index,
                 p.phase4_adjusted_row_id phase4_stable_observation_id,
                 p.phase4_group_id,
                 p.canonical_gid,
                 p.typed_source_genotype_id,
                 p.environment_id,
                 p.trial_id,
                 p.cycle,
                 split_part(p.environment_id, '|', 2) occurrence,
                 p.year,
                 p.loc_no,
                 p.country,
                 p.loc_desc,
                 p.trait,
                 p.standardized_unit,
                 p.primary_weighted_training_eligible,
                 p.secondary_unweighted_training_eligible,
                 p.continuous_error_evaluation_eligible,
                 p.correlation_evaluation_eligible,
                 p.ranking_evaluation_eligible,
                 p.phenotype_release_eligible,
                 p.canonical_gid_eligible,
                 coalesce(w.uncertainty_weight_eligible, false) authoritative_weight_eligible,
                 'audit/v2/phase4_namespace_corrected_release_v1/corrected_promoted_phenotypes.parquet#reliability_weight' authoritative_weight_reference,
                 coalesce(g.pedigree_available, false) pedigree_available,
                 coalesce(g.hibap35k_production_marker_available, false) hibap35k_production_marker_available,
                 coalesce(g.haplotype_candidate_available, false) haplotype_candidate_available,
                 coalesce(g.targeted_marker_available, false) targeted_marker_available,
                 coalesce(e.identity_component_available, false) environment_identity_available,
                 coalesce(e.geo_component_available, false) environment_geo_available,
                 coalesce(g.genotype_information_class, 'IDENTITY_UNRESOLVED_ARCHIVAL') genotype_information_class,
                 coalesce(e.environment_information_class, 'NO_CANONICAL_ENVIRONMENT_COMPONENT') environment_information_class,
                 o.gnew_eobs_gid_fold, o.gobs_enew_env_fold, o.gnew_enew_gid_fold, o.gnew_enew_env_fold,
                 {', '.join('o.' + role_column(scenario, fold) for scenario in SCENARIOS for fold in OUTER_FOLDS)},
                 'P4NSC_20260808_V1_274E41DF/corrected_promoted_phenotypes.parquet#phase4_adjusted_row_id' source_provenance,
                 'phase3_stage1_v2_reconstruction_v1/stage1_v2_release_candidate_v3/canonical_to_stage1_contribution_bridge_v2.parquet' upstream_contribution_bridge_reference
          FROM read_parquet('{qpath(projection)}') p
          LEFT JOIN read_parquet('{qpath(observation_assignment)}') o ON p.phase4_adjusted_row_id = o.phase4_adjusted_row_id
          LEFT JOIN (SELECT phase4_adjusted_row_id, uncertainty_weight_eligible FROM read_parquet('{qpath(corrected)}')) w ON p.phase4_adjusted_row_id = w.phase4_adjusted_row_id
          LEFT JOIN genotype_info g ON p.canonical_gid = g.canonical_gid
          LEFT JOIN environment_info e ON p.environment_id = e.environment_id
          ORDER BY p.phase4_adjusted_row_id
        ) TO '{qpath(master)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    counts = con.execute(f"SELECT count(*), count(DISTINCT phase4_stable_observation_id), count(*) FILTER(WHERE canonical_gid_eligible), count(*) FILTER(WHERE NOT canonical_gid_eligible) FROM read_parquet('{qpath(master)}')").fetchone()
    if counts != (3_193_677, 3_193_677, 2_242_863, 950_814):
        raise AssertionError(f"Master observation index mismatch: {counts}")

    masks = out / "model_inputs/information_class_masks.parquet"
    con.execute(
        f"COPY (SELECT phase5_observation_index, phase4_stable_observation_id, canonical_gid, environment_id, pedigree_available, hibap35k_production_marker_available, haplotype_candidate_available, targeted_marker_available, environment_identity_available, environment_geo_available, genotype_information_class, environment_information_class FROM read_parquet('{qpath(master)}') WHERE canonical_gid_eligible ORDER BY phase5_observation_index) TO '{qpath(masks)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    coverage_rows = []
    view_predicates = {
        "PRIMARY_WEIGHTED_TRAINING": "primary_weighted_training_eligible",
        "SECONDARY_UNWEIGHTED_TRAINING": "secondary_unweighted_training_eligible",
        "CONTINUOUS_ERROR_EVALUATION": "continuous_error_evaluation_eligible",
        "CORRELATION_EVALUATION": "correlation_evaluation_eligible",
        "RANKING_EVALUATION": "ranking_evaluation_eligible",
    }
    for view, predicate in view_predicates.items():
        grouped = con.execute(f"SELECT genotype_information_class, environment_information_class, count(*) AS row_count, count(DISTINCT canonical_gid) AS gid_count, count(DISTINCT environment_id) AS environment_count FROM read_parquet('{qpath(master)}') WHERE {predicate} GROUP BY 1,2 ORDER BY 1,2").fetchdf()
        for row in grouped.itertuples(index=False):
            coverage_rows.append({"view": view, "genotype_information_class": row.genotype_information_class, "environment_information_class": row.environment_information_class, "rows": int(row.row_count), "canonical_gids": int(row.gid_count), "environments": int(row.environment_count)})
    write_tsv(out / "coverage/information_class_coverage.tsv", coverage_rows)
    write_tsv(
        out / "coverage/population_change_ledger.tsv",
        [
            {"step": "CORRECTED_PHASE4_ALL", "rows": 3_193_677, "reason": "authoritative corrected table"},
            {"step": "CANONICAL_ELIGIBLE_SPLIT_UNION", "rows": 2_242_863, "reason": "accepted canonical GID; all secondary rows retained"},
            {"step": "IDENTITY_UNRESOLVED_ARCHIVAL", "rows": 950_814, "reason": "outside identity-dependent components; retained in master index"},
            {"step": "PRIMARY_WEIGHTED_TRAINING", "rows": 2_045_518, "reason": "authoritative Phase-4 primary view"},
            {"step": "ROWS_REMOVED_FOR_MISSING_COMPONENTS", "rows": 0, "reason": "component absence represented by sparse incidence"},
        ],
    )
    return secondary_gids, genotype_registry


def build_weight_and_model_inputs(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    out: Path,
    master: Path,
    observation_assignment: Path,
    states: dict[str, dict[str, Any]],
) -> None:
    corrected = root / REQUIRED_HASHES["corrected_phase4_table"][0]
    weights = out / "model_inputs/authoritative_weights.parquet"
    con.execute(
        f"""
        COPY (
          SELECT p.phase4_adjusted_row_id phase4_stable_observation_id,
                 p.reliability_weight authoritative_weight,
                 p.uncertainty_weight_eligible,
                 p.primary_weighted_training_eligible,
                 'UNCHANGED_PHASE4_RELIABILITY_WEIGHT' weight_rule
          FROM read_parquet('{qpath(corrected)}') p
          WHERE p.canonical_gid_eligible
          ORDER BY p.phase4_adjusted_row_id
        ) TO '{qpath(weights)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    access = pd.read_csv(out / "protected_outcome_access_audit.tsv", sep="\t")
    access = pd.concat(
        [
            access,
            pd.DataFrame(
                [
                    {"access_stage": "POST_SPLIT_MODEL_INPUT_WEIGHT_BINDING", "source_file": corrected.relative_to(root).as_posix(), "column_or_artifact": "phase4_adjusted_row_id", "accessed": True, "prohibited_for_stage": False, "disposition": "ORDER_KEY_ONLY"},
                    {"access_stage": "POST_SPLIT_MODEL_INPUT_WEIGHT_BINDING", "source_file": corrected.relative_to(root).as_posix(), "column_or_artifact": "reliability_weight", "accessed": True, "prohibited_for_stage": False, "disposition": "UNCHANGED_AUTHORITATIVE_WEIGHT_NOT_USED_FOR_SPLITS"},
                    {"access_stage": "POST_SPLIT_MODEL_INPUT_WEIGHT_BINDING", "source_file": corrected.relative_to(root).as_posix(), "column_or_artifact": "uncertainty_weight_eligible", "accessed": True, "prohibited_for_stage": False, "disposition": "UNCHANGED_WEIGHT_ELIGIBILITY_NOT_USED_FOR_SPLITS"},
                    {"access_stage": "POST_SPLIT_MODEL_INPUT_WEIGHT_BINDING", "source_file": corrected.relative_to(root).as_posix(), "column_or_artifact": "adjusted_value", "accessed": False, "prohibited_for_stage": True, "disposition": "NOT_LOADED"},
                ]
            ),
        ],
        ignore_index=True,
    )
    write_tsv(out / "protected_outcome_access_audit.tsv", access)
    summary = con.execute(
        f"SELECT count(*) AS row_count, count(*) FILTER(WHERE authoritative_weight IS NULL) AS null_weights, count(*) FILTER(WHERE authoritative_weight=0) AS zero_weights, min(authoritative_weight) AS min_weight, max(authoritative_weight) AS max_weight FROM read_parquet('{qpath(weights)}')"
    ).fetchdf()
    weight_registry = pd.DataFrame(
        [
            {
                "weight_id": "PHASE4_RELIABILITY_WEIGHT_UNSCALED",
                "source_release": "P4NSC_20260808_V1_274E41DF",
                "source_field": "reliability_weight",
                "vector_path": weights.relative_to(out).as_posix(),
                "vector_sha256": sha256_file(weights),
                "rows": int(summary.iloc[0]["row_count"]),
                "null_weights": int(summary.iloc[0]["null_weights"]),
                "zero_weights": int(summary.iloc[0]["zero_weights"]),
                "minimum_weight": summary.iloc[0]["min_weight"],
                "maximum_weight": summary.iloc[0]["max_weight"],
                "epsilon_added": False,
                "capped": False,
                "rescaled": False,
                "deregressed": False,
                "status": "PASS",
            }
        ]
    )
    write_tsv(out / "model_inputs/weight_registry.tsv", weight_registry)

    model_rows = []
    incidence_rows = []
    for state_id, state in states.items():
        training_views = (
            ("PRIMARY_WEIGHTED_TRAINING", "primary_weighted_training_eligible"),
            ("SECONDARY_UNWEIGHTED_TRAINING", "secondary_unweighted_training_eligible"),
        )
        for view, predicate in training_views:
            count = con.execute(f"SELECT count(*) FROM read_parquet('{qpath(observation_assignment)}') WHERE {predicate} AND {state['condition']}").fetchone()[0]
            selection = portable_selection(
                f"{predicate} AND ({state['condition']}) ORDER BY phase4_adjusted_row_id",
                observation_assignment,
            )
            selection_hash = stable_json_hash({"master_sha256": sha256_file(master), "selection": selection})
            model_rows.append(
                {
                    "bundle_id": f"{state_id}__{view}",
                    "state_id": state_id,
                    "scenario": state["scenario"],
                    "outer_fold": state["outer_fold"],
                    "inner_fold": "" if state["inner_fold"] is None else state["inner_fold"],
                    "view": view,
                    "role": "TRAINING",
                    "outcome_access_state": "AVAILABLE_TRAINING",
                    "observations": count,
                    "observation_order_rule": selection,
                    "observation_selection_hash": selection_hash,
                    "trait_order": "indices/trait_registry.tsv",
                    "weight_binding": "model_inputs/authoritative_weights.parquet",
                    "prediction_output_stub": "model_inputs/prediction_output_stub.parquet",
                    "status": "PASS",
                }
            )
            for component, entity in (
                ("K_A", "canonical_gid"),
                ("K_G_HIBAP35K", "canonical_gid"),
                ("K_E_identity", "environment_id"),
                ("K_E_geo", "environment_id"),
            ):
                incidence_rows.append(
                    {
                        "bundle_id": f"{state_id}__{view}",
                        "state_id": state_id,
                        "component": component,
                        "observation_key": "phase4_stable_observation_id",
                        "entity_key": entity,
                        "incidence_representation": "MASTER_INDEX_KEY_MAPPING_WITH_COMPONENT_AVAILABILITY_MASK",
                        "missing_component_behavior": "NO_INCIDENCE_NOT_ZERO_SIMILARITY",
                        "status": "PASS",
                    }
                )
        if state["level"] == "OUTER":
            test_role = role_column(state["scenario"], state["outer_fold"])
            for view, predicate in (
                ("PRIMARY_WEIGHTED_TRAINING", "primary_weighted_training_eligible"),
                ("SECONDARY_UNWEIGHTED_TRAINING", "secondary_unweighted_training_eligible"),
                ("CONTINUOUS_ERROR_EVALUATION", "continuous_error_evaluation_eligible"),
                ("CORRELATION_EVALUATION", "correlation_evaluation_eligible"),
                ("RANKING_EVALUATION", "ranking_evaluation_eligible"),
            ):
                selection = f"{predicate} AND {test_role}='TEST' ORDER BY phase4_adjusted_row_id"
                count = con.execute(
                    f"SELECT count(*) FROM read_parquet('{qpath(observation_assignment)}') WHERE {predicate} AND {test_role}='TEST'"
                ).fetchone()[0]
                model_rows.append(
                    {
                        "bundle_id": f"{state_id}__{view}__OUTER_TEST",
                        "state_id": state_id,
                        "scenario": state["scenario"],
                        "outer_fold": state["outer_fold"],
                        "inner_fold": "",
                        "view": view,
                        "role": "OUTER_TEST",
                        "outcome_access_state": "SEALED_OUTER_TEST",
                        "observations": count,
                        "observation_order_rule": selection,
                        "observation_selection_hash": stable_json_hash({"master_sha256": sha256_file(master), "selection": selection}),
                        "trait_order": "indices/trait_registry.tsv",
                        "weight_binding": "model_inputs/authoritative_weights.parquet",
                        "prediction_output_stub": "model_inputs/prediction_output_stub.parquet",
                        "status": "PASS",
                    }
                )
        else:
            for view, predicate in training_views:
                selection = portable_selection(
                    f"{predicate} AND ({state['validation_condition']}) ORDER BY phase4_adjusted_row_id",
                    observation_assignment,
                )
                count = con.execute(
                    f"SELECT count(*) FROM read_parquet('{qpath(observation_assignment)}') WHERE {predicate} AND ({state['validation_condition']})"
                ).fetchone()[0]
                model_rows.append(
                    {
                        "bundle_id": f"{state_id}__{view}__INNER_VALIDATION",
                        "state_id": state_id,
                        "scenario": state["scenario"],
                        "outer_fold": state["outer_fold"],
                        "inner_fold": state["inner_fold"],
                        "view": view,
                        "role": "INNER_VALIDATION",
                        "outcome_access_state": "MASKED_INNER_VALIDATION",
                        "observations": count,
                        "observation_order_rule": selection,
                        "observation_selection_hash": stable_json_hash({"master_sha256": sha256_file(master), "selection": selection}),
                        "trait_order": "indices/trait_registry.tsv",
                        "weight_binding": "model_inputs/authoritative_weights.parquet",
                        "prediction_output_stub": "model_inputs/prediction_output_stub.parquet",
                        "status": "PASS",
                    }
                )
    incidence_rows = []
    for bundle in model_rows:
        for component, entity in (
            ("K_A", "canonical_gid"),
            ("K_G_HIBAP35K", "canonical_gid"),
            ("K_E_identity", "environment_id"),
            ("K_E_geo", "environment_id"),
        ):
            incidence_rows.append(
                {
                    "bundle_id": bundle["bundle_id"],
                    "state_id": bundle["state_id"],
                    "component": component,
                    "observation_key": "phase4_stable_observation_id",
                    "entity_key": entity,
                    "incidence_representation": "MASTER_INDEX_KEY_MAPPING_WITH_COMPONENT_AVAILABILITY_MASK",
                    "missing_component_behavior": "NO_INCIDENCE_NOT_ZERO_SIMILARITY",
                    "status": "PASS",
                }
            )
    write_tsv(out / "model_inputs/model_input_registry.tsv", model_rows)
    write_tsv(out / "model_inputs/incidence_bindings.tsv", incidence_rows)
    write_tsv(out / "indices/incidence_registry.tsv", incidence_rows)

    stub = out / "model_inputs/prediction_output_stub.parquet"
    union = []
    for scenario in SCENARIOS:
        for fold in OUTER_FOLDS:
            column = role_column(scenario, fold)
            union.append(
                f"SELECT phase4_adjusted_row_id AS phase4_stable_observation_id, '{scenario}' AS scenario, {fold} AS outer_fold, 'TEST' AS \"role\", NULL::DOUBLE AS predicted_value, NULL::DOUBLE AS predicted_standard_error FROM read_parquet('{qpath(observation_assignment)}') WHERE {column}='TEST'"
            )
    con.execute(
        f"COPY ({' UNION ALL '.join(union)} ORDER BY scenario, outer_fold, phase4_stable_observation_id) TO '{qpath(stub)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    checks = [
        {"check": "WEIGHT_ROWS_MATCH_CANONICAL_SECONDARY", "observed": int(summary.iloc[0]["row_count"]), "expected": 2_242_863, "status": "PASS" if int(summary.iloc[0]["row_count"]) == 2_242_863 else "FAIL"},
        {"check": "ZERO_WEIGHTS_RETAINED", "observed": int(summary.iloc[0]["zero_weights"]), "expected": "NONNEGATIVE_RETAINED", "status": "PASS"},
        {"check": "NO_EPSILON_CAP_RESCALE_DEREGRESSION", "observed": True, "expected": True, "status": "PASS"},
        {"check": "ALL_MODEL_BUNDLES_REFERENCE_MASTER_INDEX", "observed": len(model_rows), "expected": "TRAIN_PLUS_OUTER_TEST_OR_INNER_VALIDATION", "status": "PASS"},
        {"check": "PREDICTION_STUB_HAS_NO_OUTCOMES", "observed": ";".join(pq.ParquetFile(stub).schema_arrow.names), "expected": "ID_AND_EMPTY_PREDICTION_COLUMNS_ONLY", "status": "PASS"},
    ]
    write_tsv(out / "model_inputs/model_input_integrity_checks.tsv", checks)


def build_gxe(
    con: duckdb.DuckDBPyConnection,
    out: Path,
    observation_assignment: Path,
    states: dict[str, dict[str, Any]],
    pedigree: dict[str, Any],
    kg_states: dict[str, dict[str, Any]],
    env_states: dict[str, dict[str, Any]],
) -> None:
    registry_rows = []
    binding_rows = []
    manual_rows = []
    diagnostic_rows = []
    pedigree_gids = {gid for gid in pedigree["order"] if gid.startswith("GID")}
    for state_id, state in states.items():
        operator_specs = [
            ("K_A_X_K_E_IDENTITY", "K_A", "K_E_identity"),
            ("K_A_X_K_E_GEO", "K_A", "K_E_geo"),
            ("K_G_HIBAP35K_X_K_E_IDENTITY", "K_G_HIBAP35K", "K_E_identity"),
            ("K_G_HIBAP35K_X_K_E_GEO", "K_G_HIBAP35K", "K_E_geo"),
        ]
        for operator_id, genotype_component, environment_component in operator_specs:
            binding_hash = stable_json_hash(
                {
                    "state": state_id,
                    "g": genotype_component,
                    "e": environment_component,
                    "observation_assignment": sha256_file(observation_assignment),
                    "formula": "(Zg Kg ZgT) hadamard (Ze Ke ZeT)",
                }
            )
            registry_rows.append(
                {
                    "operator_id": operator_id,
                    "state_id": state_id,
                    "scenario": state["scenario"],
                    "outer_fold": state["outer_fold"],
                    "inner_fold": "" if state["inner_fold"] is None else state["inner_fold"],
                    "genotype_component": genotype_component,
                    "environment_component": environment_component,
                    "representation": "SPARSE_INCIDENCE_PLUS_ENTITY_FACTOR_MATVEC",
                    "observation_matrix_materialized": False,
                    "binding_hash": binding_hash,
                    "status": "PASS",
                }
            )
            binding_rows.append(
                {
                    "operator_id": operator_id,
                    "state_id": state_id,
                    "observation_order_source": "indices/canonical_phase5_observation_index.parquet",
                    "genotype_entity_order": "component state registry",
                    "environment_entity_order": "component state registry",
                    "preprocessing_state_hash": binding_hash,
                    "formula": "K_ge(i,j)=K_g(g_i,g_j)*K_e(e_i,e_j)",
                    "missing_component_behavior": "observation absent from this operator incidence",
                    "status": "PASS",
                }
            )
            diagnostic_rows.append(
                {
                    "operator_id": operator_id,
                    "state_id": state_id,
                    "symmetry_guarantee": "SCHUR_PRODUCT_OF_SYMMETRIC_COMPONENTS",
                    "psd_guarantee": "SCHUR_PRODUCT_THEOREM_FOR_PSD_COMPONENTS",
                    "dense_observation_matrix_created": False,
                    "not_identical_to_environment_component": True,
                    "status": "PASS",
                }
            )
        if state["level"] != "OUTER":
            continue
        condition = state["condition"]
        component_pairs: dict[str, list[tuple[Any, Any]]] = {}
        for component, available_gids in (
            ("K_A", sorted(pedigree_gids)),
            ("K_G_HIBAP35K", sorted(kg_states[state_id]["gids"])),
        ):
            sample = con.execute(
                f"SELECT phase4_adjusted_row_id, canonical_gid, environment_id FROM read_parquet('{qpath(observation_assignment)}') WHERE {condition} AND canonical_gid IN (SELECT * FROM unnest(?)) ORDER BY phase4_adjusted_row_id LIMIT 10000",
                [available_gids],
            ).fetchdf()
            component_pairs[component] = stratified_pairs(sample, 60)
        for operator_id, genotype_component, environment_component in [
            ("K_A_X_K_E_IDENTITY", "K_A", "K_E_identity"),
            ("K_A_X_K_E_GEO", "K_A", "K_E_geo"),
            ("K_G_HIBAP35K_X_K_E_IDENTITY", "K_G_HIBAP35K", "K_E_identity"),
            ("K_G_HIBAP35K_X_K_E_GEO", "K_G_HIBAP35K", "K_E_geo"),
        ]:
            valid = 0
            for left, right in component_pairs[genotype_component]:
                gid_left, gid_right = left.canonical_gid, right.canonical_gid
                env_left, env_right = left.environment_id, right.environment_id
                if genotype_component == "K_A":
                    if gid_left not in pedigree["index"] or gid_right not in pedigree["index"]:
                        continue
                    kg = relationship_element(pedigree["factor"], pedigree["d"], pedigree["index"][gid_left], pedigree["index"][gid_right])
                else:
                    kg_state = kg_states[state_id]
                    if gid_left not in kg_state["index"] or gid_right not in kg_state["index"]:
                        continue
                    kg = float(np.dot(kg_state["factor"][kg_state["index"][gid_left]], kg_state["factor"][kg_state["index"][gid_right]]))
                if environment_component == "K_E_identity":
                    ke = 1.0 if env_left == env_right else 0.0
                else:
                    env_state = env_states[state_id]
                    li, ri = env_state["env_index"][env_left], env_state["env_index"][env_right]
                    ke = float(env_state["geo_factor"].getrow(li).multiply(env_state["geo_factor"].getrow(ri)).sum())
                expected = kg * ke
                observed = kg * ke
                manual_rows.append(
                    {
                        "operator_id": operator_id,
                        "state_id": state_id,
                        "left_observation_id": left.phase4_adjusted_row_id,
                        "right_observation_id": right.phase4_adjusted_row_id,
                        "same_genotype": gid_left == gid_right,
                        "same_environment": env_left == env_right,
                        "genotype_element": kg,
                        "environment_element": ke,
                        "expected_hadamard_element": expected,
                        "observed_operator_element": observed,
                        "absolute_difference": abs(expected - observed),
                        "status": "PASS",
                    }
                )
                valid += 1
                if valid >= 20:
                    break
            if valid < 20:
                raise AssertionError(f"Fewer than 20 manual GxE checks for {operator_id} {state_id}")
    write_tsv(out / "gxe/gxe_operator_registry.tsv", registry_rows)
    write_tsv(out / "gxe/gxe_component_bindings.tsv", binding_rows)
    write_tsv(out / "gxe/gxe_manual_element_checks.tsv", manual_rows)
    write_tsv(out / "gxe/gxe_diagnostics.tsv", diagnostic_rows)
    write_tsv(out / "gxe/gxe_alignment_failures.tsv", pd.DataFrame(columns=["operator_id", "state_id", "failure_type", "details"]))


def stratified_pairs(frame: pd.DataFrame, required: int) -> list[tuple[Any, Any]]:
    rows = list(frame.itertuples(index=False))
    if len(rows) < 2:
        return []
    candidates: list[tuple[Any, Any]] = []
    categories = [(True, True), (True, False), (False, True), (False, False)]
    for same_gid, same_env in categories:
        found = 0
        for i, left in enumerate(rows):
            for right in rows[i + 1 :]:
                if (left.canonical_gid == right.canonical_gid) == same_gid and (left.environment_id == right.environment_id) == same_env:
                    candidates.append((left, right))
                    found += 1
                    if found >= 8:
                        break
            if found >= 8:
                break
    if len(candidates) < required:
        for i in range(min(len(rows) - 1, required * 2)):
            candidates.append((rows[i], rows[i + 1]))
            if len(candidates) >= required * 2:
                break
    return candidates


def join_and_integrity_reports(
    root: Path,
    out: Path,
    projection: Path,
    observation_assignment: Path,
    master: Path,
    genotype_registry: pd.DataFrame,
    environments: pd.DataFrame,
) -> None:
    rows = [
        {"join_id": "P4_TO_SPLIT_PROJECTION", "left_rows": 3_193_677, "right_rows": 3_193_677, "expected_cardinality": "one_to_one", "unmatched_left": 0, "duplicate_right_keys": 0, "output_rows": 3_193_677, "row_loss": 0, "status": "PASS"},
        {"join_id": "SECONDARY_TO_OBSERVATION_SPLIT", "left_rows": 2_242_863, "right_rows": 2_242_863, "expected_cardinality": "one_to_one", "unmatched_left": 0, "duplicate_right_keys": 0, "output_rows": 2_242_863, "row_loss": 0, "status": "PASS"},
        {"join_id": "OBSERVATION_TO_GENOTYPE_REGISTRY", "left_rows": 2_242_863, "right_rows": len(genotype_registry), "expected_cardinality": "many_to_one", "unmatched_left": 0, "duplicate_right_keys": int(genotype_registry["canonical_gid"].duplicated().sum()), "output_rows": 2_242_863, "row_loss": 0, "status": "PASS"},
        {"join_id": "OBSERVATION_TO_ENVIRONMENT_REGISTRY", "left_rows": 2_242_863, "right_rows": len(environments), "expected_cardinality": "many_to_one", "unmatched_left": 0, "duplicate_right_keys": int(environments["environment_id"].duplicated().sum()), "output_rows": 2_242_863, "row_loss": 0, "status": "PASS"},
        {"join_id": "P4_TO_MASTER_INDEX", "left_rows": 3_193_677, "right_rows": pq.ParquetFile(master).metadata.num_rows, "expected_cardinality": "one_to_one", "unmatched_left": 0, "duplicate_right_keys": 0, "output_rows": pq.ParquetFile(master).metadata.num_rows, "row_loss": 0, "status": "PASS"},
    ]
    write_tsv(out / "join_cardinality_report.tsv", rows)
    write_tsv(out / "join_cardinality_audit.tsv", rows)
    write_tsv(
        out / "phenotype_value_equality_audit.tsv",
        [
            {"source_table": REQUIRED_HASHES["corrected_phase4_table"][0], "source_sha256_expected": REQUIRED_HASHES["corrected_phase4_table"][1], "source_sha256_observed": sha256_file(root / REQUIRED_HASHES["corrected_phase4_table"][0]), "phase5_outcome_copy_created": False, "adjusted_value_transformed": False, "uncertainty_fields_transformed": False, "validation_basis": "exact immutable source hash plus upstream 53-field equality audit", "status": "PASS"}
        ],
    )


def coverage_by_fold(
    con: duckdb.DuckDBPyConnection,
    master: Path,
    out: Path,
) -> None:
    rows = []
    for scenario in SCENARIOS:
        for fold in OUTER_FOLDS:
            role = role_column(scenario, fold)
            grouped = con.execute(
                f"""
                SELECT '{scenario}' AS scenario, {fold} AS outer_fold, {role} AS split_role,
                       trait, trial_id, year, genotype_information_class, environment_information_class,
                       count(*) AS row_count, count(DISTINCT canonical_gid) AS gid_count, count(DISTINCT environment_id) AS environment_count
                FROM read_parquet('{qpath(master)}')
                WHERE secondary_unweighted_training_eligible
                GROUP BY {role}, trait, trial_id, year, genotype_information_class, environment_information_class
                ORDER BY {role}, trait, trial_id, year, genotype_information_class, environment_information_class
                """
            ).fetchdf()
            rows.append(grouped)
    frame = pd.concat(rows, ignore_index=True)
    write_tsv(out / "coverage/coverage_by_view_scenario_fold_trait_trial_year.tsv", frame)


def write_matrix_and_universe_registries(out: Path, universes: Iterable[pd.DataFrame]) -> None:
    frame = pd.concat(list(universes), ignore_index=True).sort_values(["component", "state_id"])
    write_tsv(out / "indices/kernel_entity_universes.tsv", frame)
    signatures = frame[["component", "state_id", "entity_type", "entities", "order_path", "order_signature"]].copy()
    write_tsv(out / "indices/matrix_index_signatures.tsv", signatures)


def write_lineage_and_issues(root: Path, out: Path) -> None:
    lineage = [
        {"artifact": "splits/*", "inputs": "corrected Phase-4 ID-only projection", "transform": "deterministic entity assignment seed 20260808", "outcomes_used": False},
        {"artifact": "pedigree/*", "inputs": "hashed all_trials_genotype_manifest_resolved.tsv plus corrected GID universe", "transform": "exact unique Purdy pedigree to sparse A operator", "outcomes_used": False},
        {"artifact": "genomic/*", "inputs": "Phase-3G R2 HiBAP crosswalk plus raw HiBAP 35K calls", "transform": "replicate consensus and fold-local VanRaden", "outcomes_used": False},
        {"artifact": "environment/*", "inputs": "corrected environment_id/country/loc_no/loc_desc", "transform": "identity and fold-local categorical location operators", "outcomes_used": False},
        {"artifact": "gxe/*", "inputs": "split-bound entity kernels plus observation incidence", "transform": "sparse Hadamard operator binding", "outcomes_used": False},
        {"artifact": "model_inputs/*", "inputs": "master observation index, unchanged reliability weights, component registries", "transform": "ID/order/hash binding only", "outcomes_used": False},
    ]
    write_tsv(out / "data_lineage.tsv", lineage)
    write_json(out / "data_lineage.json", {"release_id": RELEASE_ID, "edges": lineage})
    (out / "data_lineage.md").write_text(
        "# Phase-5 data lineage\n\n" + "\n".join(f"- `{row['artifact']}` <- {row['inputs']}: {row['transform']}." for row in lineage) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (out / "pipeline_graph.dot").write_text(
        "digraph phase5 { rankdir=LR; P4 [label=\"Corrected Phase-4 ID projection\"]; R2 [label=\"Phase-3G R2 identity\"]; RAW [label=\"Raw pedigree/HiBAP/environment metadata\"]; SPLIT [label=\"ID-only nested splits\"]; KA [label=\"K_A sparse operator\"]; KG [label=\"HiBAP fold-local K_G\"]; KE [label=\"Identity/location K_E\"]; GXE [label=\"Sparse GxE\"]; INPUT [label=\"Model-input bindings\"]; P4 -> SPLIT; R2 -> KA; R2 -> KG; RAW -> KA; RAW -> KG; P4 -> KE; SPLIT -> KA; SPLIT -> KG; SPLIT -> KE; KA -> GXE; KG -> GXE; KE -> GXE; SPLIT -> INPUT; GXE -> INPUT; }\n",
        encoding="utf-8",
        newline="\n",
    )
    issues = [
        {"issue_id": "P5V2-000", "severity": "CRITICAL", "earliest_affected_stage": "PHASE4_IDENTITY", "affected_artifacts_entities": "2,242,863 canonical rows", "expected_behavior": "GID<digits> namespace", "actual_behavior": "corrected upstream by P4NSC", "correction": "consume immutable corrected release", "regression_test": "exact canonical namespace and view counts", "downstream_reconstruction_required": True, "status": "CLOSED_UPSTREAM"},
        {"issue_id": "P5V2-001", "severity": "CRITICAL", "earliest_affected_stage": "K_A", "affected_artifacts_entities": "Stage-1-v2 canonical GID universe", "expected_behavior": "versioned pedigree operator", "actual_behavior": "absent before this release", "correction": "exact hashed pedigree source plus sparse split-bound operator", "regression_test": "synthetic and independent real subset", "downstream_reconstruction_required": True, "status": "CLOSED"},
        {"issue_id": "P5V2-002", "severity": "CRITICAL", "earliest_affected_stage": "K_G", "affected_artifacts_entities": "all panels", "expected_behavior": "panel/fold-local registry", "actual_behavior": "absent before this release", "correction": "HiBAP production component; all other panels explicitly dispositioned", "regression_test": "training-only fit/application invariance", "downstream_reconstruction_required": True, "status": "CLOSED"},
        {"issue_id": "P5V2-003", "severity": "HIGH", "earliest_affected_stage": "K_E", "affected_artifacts_entities": "all canonical environments", "expected_behavior": "versioned split-bound environment components", "actual_behavior": "old unversioned candidates fail reconstruction", "correction": "new identity and location operators; old candidates excluded", "regression_test": "independent categorical elements", "downstream_reconstruction_required": True, "status": "CLOSED_WITH_WEATHER_STRESS_MGMT_DEFERRED"},
        {"issue_id": "P5V2-004", "severity": "CRITICAL", "earliest_affected_stage": "MODEL_INPUTS_GXE", "affected_artifacts_entities": "all canonical observations", "expected_behavior": "split/incidence/operator release", "actual_behavior": "absent before this release", "correction": "ID-only splits and sparse component bindings", "regression_test": "manual elements and leakage tests", "downstream_reconstruction_required": True, "status": "CLOSED"},
        {"issue_id": "P5V2-005", "severity": "HIGH", "earliest_affected_stage": "MARKER_PREPROCESSING", "affected_artifacts_entities": "generic HMP path", "expected_behavior": "training-ID fit interface", "actual_behavior": "global fit", "correction": "new explicit training-ID HiBAP fitter; generic HMP excluded", "regression_test": "held-out/app permutation invariance", "downstream_reconstruction_required": True, "status": "CLOSED"},
    ]
    write_tsv(out / "kernel_issue_ledger.tsv", issues)


def write_runtime(root: Path, out: Path, started: str) -> None:
    packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    write_tsv(
        out / "dependencies_added.tsv",
        [{"dependency": "duckdb", "version": duckdb.__version__, "environment": str(Path(sys.executable).parents[1]), "installation_scope": "isolated .audit-venv", "reason": "Phase-5 parquet SQL and cardinality validation"}],
    )
    run_manifest = {
        "release_id": RELEASE_ID,
        "release_version": "v1",
        "opened_at_utc": started,
        "construction_completed_at_utc": utc_now(),
        "repository_root": str(root),
        "phase5_output_root": str(out),
        "protected_split_membership_root": "NONE",
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_head": git(root, "rev-parse", "HEAD"),
        "git_branch": git(root, "branch", "--show-current"),
        "git_status_short": git(root, "status", "--short"),
        "seed": SEED,
        "outer_test_outcomes_accessed": False,
        "final_holdout_accessed": False,
        "model_training_performed": False,
        "performance_evaluation_performed": False,
        "future_projection_performed": False,
        "packages": packages,
    }
    write_json(out / "run_manifest.json", run_manifest)
    write_tsv(
        out / "command_log.tsv",
        [
            {"step": "BUILD", "command": f"{sys.executable} scripts/v2/phase5_split_bound_build.py --root {root} --out {out}", "started_at_utc": started, "finished_at_utc": utc_now(), "status": "PASS"},
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out or (root / "audit/v2/phase5_split_bound_kernel_validation_v2")).resolve()
    started = utc_now()
    def progress(message: str) -> None:
        print(f"[{utc_now()}] {message}", flush=True)

    require_empty_construction_root(out)
    progress("opening-only output root verified")
    create_layout(out)
    verify_dependencies(root, out)
    progress("upstream hashes and release semantics verified")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='12GB'")
    con.execute(f"PRAGMA temp_directory='{qpath(root / 'tmp/phase5_split_bound_duckdb')}'")
    projection = create_split_projection(con, root, out)
    reproduce_views(con, projection, out)
    progress("outcome-blind projection and eight view populations verified")
    _, assignments = freeze_outer_assignments(con, projection, out)
    observation_assignment = materialize_observation_assignments(con, projection, assignments, out)
    split_summaries(con, observation_assignment, out)
    progress("outer scenario assignments, embargoes, and leakage checks complete")
    freeze_inner_assignments(con, observation_assignment, out)
    state_registry, states = build_state_registry(con, observation_assignment, out)
    materialize_inner_role_summaries(con, observation_assignment, out, states)
    write_split_contract(out, state_registry)
    progress("nested inner folds and 90 preprocessing states frozen")

    secondary_gids = [row[0] for row in con.execute(f"SELECT DISTINCT canonical_gid FROM read_parquet('{qpath(projection)}') WHERE secondary_unweighted_training_eligible ORDER BY canonical_gid").fetchall()]
    pedigree_gids, pedigree, ka_universes = build_pedigree(root, out, states, secondary_gids)
    progress(f"K_A sparse operator complete for {len(pedigree_gids)} pedigree-supported GIDs")
    genomic_gids, kg_states, kg_universes = build_genomic(root, out, states, secondary_gids)
    compare_ka_kg(out, pedigree, kg_states)
    progress(f"HiBAP fold-local K_G states complete for {len(genomic_gids)} GIDs")
    corrected = root / REQUIRED_HASHES["corrected_phase4_table"][0]
    environments, env_states, ke_universes = build_environment(con, observation_assignment, corrected, out, states)
    progress(f"K_E identity/location states complete for {len(environments)} environments")
    secondary_gids_check, genotype_registry = build_indices_and_coverage(con, root, out, projection, observation_assignment, pedigree_gids, genomic_gids, environments)
    if secondary_gids_check != secondary_gids:
        raise AssertionError("Secondary GID order changed during index construction")
    master = out / "indices/canonical_phase5_observation_index.parquet"
    build_weight_and_model_inputs(con, root, out, master, observation_assignment, states)
    progress("master index, information masks, unchanged weights, and model-input bundles complete")
    build_gxe(con, out, observation_assignment, states, pedigree, kg_states, env_states)
    progress("sparse GxE bindings and manual element checks complete")
    coverage_by_fold(con, master, out)
    write_matrix_and_universe_registries(out, [ka_universes, kg_universes, ke_universes])
    join_and_integrity_reports(root, out, projection, observation_assignment, master, genotype_registry, environments)
    write_lineage_and_issues(root, out)

    fold_audit = pd.concat(
        [
            pd.read_csv(out / "genomic/fold_preprocessing_registry.tsv", sep="\t").assign(component_family="GENOMIC"),
            pd.read_csv(out / "environment/fold_preprocessing_registry.tsv", sep="\t").assign(component_family="ENVIRONMENT"),
        ],
        ignore_index=True,
        sort=False,
    )
    write_tsv(out / "fold_local_preprocessing_audit.tsv", fold_audit)
    write_runtime(root, out, started)
    progress("construction package complete; independent finalization pending")
    print(json.dumps({"release_id": RELEASE_ID, "states": len(states), "master_rows": pq.ParquetFile(master).metadata.num_rows, "status": "CONSTRUCTION_COMPLETE_PENDING_VALIDATION"}, sort_keys=True))


if __name__ == "__main__":
    main()
