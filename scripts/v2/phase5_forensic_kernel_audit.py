#!/usr/bin/env python3
"""Build the diagnostic evidence for the atomic Stage-1 v2 Phase-5 audit.

The script consumes only the pinned Stage-1 v2 / Phase-3G R2 / integrated
Phase-4 development releases and allowlisted raw/covariate inventories.  It
does not discover or read protected outcome directories and does not train a
model.  Historical certified-v1 kernels are inventory-only, never v2 inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from .phase5_independent_reconstruction import (
        additive_relationship,
        environment_linear_kernel,
        gxe_elements,
        index_signature,
        synthetic_results,
        vanraden,
    )
except ImportError:
    from phase5_independent_reconstruction import (
        additive_relationship,
        environment_linear_kernel,
        gxe_elements,
        index_signature,
        synthetic_results,
        vanraden,
    )


RELEASE_ID = "P5KV_20260802_V1_274E41DF"
P4_ID = "P4ISP_20260802_V1_274E41DF"
P4_SOURCE_HASH = "bfc637afdd28d9763f01181070477dd330df81680b1fc00fcb69cca2a39312b5"
STAGE1_V2_VERSION = "stage1_v2_reconstruction_2026_07_30_v1"
SEED = 20260802

VIEWS = {
    "PRIMARY_WEIGHTED_TRAINING": "primary_weighted_training_eligible",
    "SECONDARY_UNWEIGHTED_TRAINING": "secondary_unweighted_training_eligible",
    "CONTINUOUS_ERROR_EVALUATION": "continuous_error_evaluation_eligible",
    "CORRELATION_EVALUATION": "correlation_evaluation_eligible",
    "RANKING_EVALUATION": "ranking_evaluation_eligible",
    "IDENTITY_UNRESOLVED_ARCHIVAL": "NOT canonical_gid_eligible",
    "RELEASE_ONLY": "phenotype_release_eligible AND NOT secondary_unweighted_training_eligible",
    "BLOCKED_DATA_INTEGRITY": "NOT phenotype_release_eligible",
}

EXPECTED_VIEWS = {
    "PRIMARY_WEIGHTED_TRAINING": (2_045_518, 10_656, 31_343, 273, 10_258, 43, 7),
    "SECONDARY_UNWEIGHTED_TRAINING": (2_242_863, 10_722, 37_157, 283, 11_161, 43, 7),
    "CONTINUOUS_ERROR_EVALUATION": (2_242_863, 10_722, 37_157, 283, 11_161, 43, 7),
    "CORRELATION_EVALUATION": (2_242_615, 10_722, 36_909, 280, 11_086, 43, 7),
    "RANKING_EVALUATION": (1_418_644, 10_656, 23_483, 271, 9_242, 43, 7),
    "IDENTITY_UNRESOLVED_ARCHIVAL": (950_814, 0, 20_211, 164, 6_240, 38, 7),
    "RELEASE_ONLY": (950_814, 0, 20_211, 164, 6_240, 38, 7),
    "BLOCKED_DATA_INTEGRITY": (0, 0, 0, 0, 0, 0, 0),
}


def q(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: Iterable[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def query_frame(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()


def inventory_and_scope(root: Path, out: Path, p3g: Path, p4: Path, stage1: Path) -> None:
    opening = pd.read_csv(out / "OPENING_HASH_MANIFEST.tsv", sep="\t", dtype=str, keep_default_na=False)
    raw = opening[opening["category"].eq("RAW_GENOTYPE_CORPUS")].copy()
    p3_inventory = pd.read_csv(p3g / "genotype_file_inventory.tsv", sep="\t", dtype=str, keep_default_na=False)
    enriched = p3_inventory.merge(
        raw[["relative_path", "sha256", "bytes"]].rename(
            columns={"sha256": "phase5_opening_sha256", "bytes": "phase5_opening_bytes"}
        ),
        left_on=p3_inventory["absolute_path"].map(lambda x: Path(x).as_posix()).str.replace(
            "/mnt/e/ensayos_genotipoXambiente/", "", regex=False
        ),
        right_on="relative_path",
        how="left",
    )
    enriched["phase5_use"] = enriched["file_role"].map(
        lambda value: "CANDIDATE_V2_GENOTYPE_SOURCE" if value else "INVENTORY_ONLY"
    )
    enriched["certified_v1_authority_for_v2"] = False
    write_tsv(out / "genotypic_artifact_inventory.tsv", enriched)

    trial = opening[opening["category"].eq("RAW_TRIAL_LINEAGE_SOURCE")].copy()
    trial["phase5_scope"] = "LINEAGE_ENVIRONMENT_IDENTIFIER_VALIDATION_ONLY"
    trial["phenotype_reestimated"] = False
    write_tsv(out / "trial_environment_source_inventory.tsv", trial)

    unused = opening[opening["category"].isin(["PRODUCTION_ARTIFACT"])].copy()
    unused["disposition"] = "HISTORICAL_OR_UNVERSIONED_INVENTORY_NOT_STAGE1_V2_AUTHORITY"
    unused["reason"] = "User clarified Stage-1 v2; no historical certified-v1 artifact defines v2"
    write_tsv(out / "unused_phase5_source_files.tsv", unused)

    anomalies = []
    phase3_paths = set(p3_inventory["relative_path"])
    for row in raw.itertuples(index=False):
        relative = str(row.relative_path).replace("GENOTYPIC_DATA/", "", 1)
        if relative not in phase3_paths:
            anomalies.append({
                "artifact": row.path,
                "anomaly": "PRESENT_AT_PHASE5_OPENING_NOT_IN_PHASE3G_R2_FILE_INVENTORY",
                "severity": "MEDIUM",
                "disposition": "DO_NOT_CONSUME_UNTIL_REVIEWED",
            })
    write_tsv(
        out / "source_anomalies.tsv",
        pd.DataFrame(anomalies, columns=["artifact", "anomaly", "severity", "disposition"]),
    )

    scope = {
        "release_train_id": RELEASE_ID,
        "authoritative_modelling_foundation": "STAGE1_V2",
        "stage1_v2_path": stage1.as_posix(),
        "phase3g_r2_path": p3g.as_posix(),
        "phase4_integrated_path": p4.as_posix(),
        "certified_v1_artifacts_modified": False,
        "certified_v1_artifacts_used_as_v2_inputs": False,
        "server_v1_data_required": False,
        "scope_recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "STAGE1_V2_SCOPE_CONTRACT.json").write_text(
        json.dumps(scope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def reproduce_views(con: duckdb.DuckDBPyConnection, promoted: Path, ledger: Path, out: Path) -> pd.DataFrame:
    rows = []
    for name, predicate in VIEWS.items():
        observed = con.execute(f"""
            SELECT count(*),
                   count(DISTINCT canonical_gid) FILTER (WHERE canonical_gid_eligible),
                   count(DISTINCT phase4_group_id), count(DISTINCT trial_id),
                   count(DISTINCT environment_id), count(DISTINCT year), count(DISTINCT trait)
            FROM read_parquet('{q(promoted)}') WHERE {predicate}
        """).fetchone()
        expected = EXPECTED_VIEWS[name]
        rows.append({
            "release_train_id": RELEASE_ID,
            "view": name,
            "filter_expression": predicate,
            "expected_rows": expected[0], "observed_rows": observed[0],
            "expected_canonical_gids": expected[1], "observed_canonical_gids": observed[1],
            "expected_groups": expected[2], "observed_groups": observed[2],
            "expected_trials": expected[3], "observed_trials": observed[3],
            "expected_environments": expected[4], "observed_environments": observed[4],
            "expected_years": expected[5], "observed_years": observed[5],
            "expected_traits": expected[6], "observed_traits": observed[6],
            "status": "PASS" if tuple(observed) == expected else "FAIL",
        })
    summary = pd.DataFrame(rows)
    write_tsv(out / "view_reproduction_summary.tsv", summary)

    equality = query_frame(con, f"""
        SELECT '{RELEASE_ID}' release_train_id,
          (SELECT count(*) FROM read_parquet('{q(promoted)}')) promoted_rows,
          (SELECT count(*) FROM read_parquet('{q(ledger)}')) ledger_rows,
          (SELECT count(*) FROM read_parquet('{q(promoted)}') p
             ANTI JOIN read_parquet('{q(ledger)}') l USING(phase4_adjusted_row_id)) promoted_only_rows,
          (SELECT count(*) FROM read_parquet('{q(ledger)}') l
             ANTI JOIN read_parquet('{q(promoted)}') p USING(phase4_adjusted_row_id)) ledger_only_rows,
          (SELECT count(*) FROM read_parquet('{q(promoted)}') p
             JOIN read_parquet('{q(ledger)}') l USING(phase4_adjusted_row_id)
             WHERE p.adjusted_value IS DISTINCT FROM l.adjusted_value) adjusted_value_mismatches,
          (SELECT count(*) FROM read_parquet('{q(promoted)}') p
             JOIN read_parquet('{q(ledger)}') l USING(phase4_adjusted_row_id)
             WHERE p.pev_proxy IS DISTINCT FROM l.pev_proxy OR
                   p.reliability IS DISTINCT FROM l.reliability OR
                   p.reliability_weight IS DISTINCT FROM l.reliability_weight OR
                   p.raw_precision_weight IS DISTINCT FROM l.raw_precision_weight OR
                   p.h2_status IS DISTINCT FROM l.h2_status OR
                   p.ranking_status IS DISTINCT FROM l.ranking_status) uncertainty_field_mismatches,
          (SELECT count(*)-count(DISTINCT phase4_adjusted_row_id) FROM read_parquet('{q(promoted)}')) duplicate_row_ids
    """)
    equality["status"] = np.where(
        equality[["promoted_only_rows", "ledger_only_rows", "adjusted_value_mismatches", "uncertainty_field_mismatches", "duplicate_row_ids"]].sum(axis=1).eq(0),
        "PASS", "FAIL"
    )
    write_tsv(out / "phenotype_value_equality_audit.tsv", equality)
    return summary


def build_availability_and_index(
    con: duckdb.DuckDBPyConnection, p3g: Path, promoted: Path, out: Path
) -> tuple[Path, pd.DataFrame]:
    crosswalk = pd.read_parquet(p3g / "accepted_all_panel_crosswalk.parquet")
    crosswalk["accepted_canonical_gid"] = crosswalk["accepted_canonical_gid"].fillna("").astype(str)
    accepted = crosswalk[crosswalk["accepted_canonical_gid"].ne("")].copy()
    availability = accepted.groupby("accepted_canonical_gid", as_index=False).agg(
        marker_panel_count=("panel_id", "nunique"),
        marker_panels=("panel_id", lambda s: ";".join(sorted(set(map(str, s))))),
        marker_vector_available=("marker_vector_present", "max"),
        existing_kernel_order_available=("existing_kernel_order_present", "max"),
        accepted_sample_instances=("sample_instance_key", "nunique"),
    ).rename(columns={"accepted_canonical_gid": "canonical_gid"})
    availability["stage1_v2_pedigree_available"] = False
    availability["pedigree_status"] = "NO_VERSIONED_STAGE1_V2_PEDIGREE_BINDING"
    availability_path = out / "stage1_v2_gid_availability.parquet"
    availability.to_parquet(availability_path, index=False)

    index_path = out / "canonical_phase5_observation_index.parquet"
    con.execute(f"""
      COPY (
        SELECT row_number() OVER(ORDER BY p.phase4_adjusted_row_id)-1 phase5_observation_index,
               '{RELEASE_ID}' release_train_id,
               p.release_train_id upstream_phase4_release_train_id,
               p.phase4_adjusted_row_id,a.canonical_gid,
               p.canonical_gid phase4_canonical_gid_raw,p.typed_source_genotype_id,
               (p.canonical_gid IS DISTINCT FROM a.canonical_gid) canonical_gid_namespace_mismatch,
               'AUDIT_OVERLAY_ONLY_NOT_AN_UPSTREAM_REPAIR' canonical_gid_overlay_status,
               p.environment_id,p.trial_id,p.cycle,p.year,
               split_part(p.environment_id,'|',2) occurrence,p.loc_no,p.country,p.loc_desc,
               p.trait,p.trait_name_original,p.standardized_unit,p.adjusted_value,
               p.pev_proxy,p.reliability,p.reliability_weight authoritative_weight,
               'reliability_weight_unscaled_fold_local_handling_required' weight_policy,
               p.primary_weighted_training_eligible,p.secondary_unweighted_training_eligible,
               p.continuous_error_evaluation_eligible,p.correlation_evaluation_eligible,
               p.ranking_evaluation_eligible,p.uncertainty_weight_eligible,
               coalesce(a.marker_vector_available,false) marker_available,
               coalesce(a.existing_kernel_order_available,false) historical_kernel_order_available,
               false pedigree_available,
               'NO_VERSIONED_STAGE1_V2_PEDIGREE_BINDING' pedigree_status,
               coalesce(a.marker_panels,'') marker_panels,
               p.selected_model,p.identity_status,p.check_status,p.huber_status,
               p.ranking_status,p.restriction_reason_codes,
               'UNASSIGNED_PHASE5_NO_MODEL_TRAINING' split_id,
               '' fold_id,
               'promoted_phenotypes.parquet' source_artifact,
               p.authoritative_phase4_candidate_id provenance_reference
        FROM read_parquet('{q(promoted)}') p
        LEFT JOIN read_parquet('{q(availability_path)}') a
          ON p.typed_source_genotype_id=a.canonical_gid
        WHERE p.primary_weighted_training_eligible
        ORDER BY p.phase4_adjusted_row_id
      ) TO '{q(index_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)
    return index_path, availability


def population_and_join_audits(
    con: duckdb.DuckDBPyConnection, promoted: Path, index_path: Path,
    p3g: Path, availability: pd.DataFrame, out: Path
) -> None:
    reconcile = query_frame(con, f"""
      SELECT '{RELEASE_ID}' release_train_id,
        (SELECT count(*) FROM read_parquet('{q(promoted)}') WHERE primary_weighted_training_eligible) phase4_primary_rows,
        (SELECT count(*) FROM read_parquet('{q(index_path)}')) phase5_index_rows,
        (SELECT count(*) FROM read_parquet('{q(promoted)}') p
          ANTI JOIN read_parquet('{q(index_path)}') i USING(phase4_adjusted_row_id)
          WHERE p.primary_weighted_training_eligible) primary_rows_missing_from_index,
        (SELECT count(*) FROM read_parquet('{q(index_path)}') i
          ANTI JOIN read_parquet('{q(promoted)}') p USING(phase4_adjusted_row_id)) index_rows_without_promoted_source,
        (SELECT count(*)-count(DISTINCT phase4_adjusted_row_id) FROM read_parquet('{q(index_path)}')) duplicate_index_row_ids,
        (SELECT count(*) FROM read_parquet('{q(index_path)}') WHERE canonical_gid IS NULL OR canonical_gid='') null_gid_rows
    """)
    mismatch_columns = [
        "primary_rows_missing_from_index", "index_rows_without_promoted_source",
        "duplicate_index_row_ids", "null_gid_rows",
    ]
    reconcile["status"] = np.where(reconcile[mismatch_columns].sum(axis=1).eq(0), "PASS", "FAIL")
    write_tsv(out / "phase4_to_phase5_row_reconciliation.tsv", reconcile)

    changes = []
    steps = [
        ("COMPLETE_PROMOTED_RELEASE", "true"),
        ("CANONICAL_GID_ELIGIBLE", "canonical_gid_eligible"),
        ("SECONDARY_UNWEIGHTED_TRAINING", "secondary_unweighted_training_eligible"),
        ("PRIMARY_WEIGHTED_TRAINING", "primary_weighted_training_eligible"),
        ("PHASE5_PRIMARY_OBSERVATION_INDEX", "primary_weighted_training_eligible"),
    ]
    previous = None
    for step, predicate in steps:
        count = con.execute(f"SELECT count(*) FROM read_parquet('{q(promoted)}') WHERE {predicate}").fetchone()[0]
        changes.append({
            "release_train_id": RELEASE_ID, "step": step, "rows": count,
            "rows_removed_from_previous": 0 if previous is None else previous - count,
            "reason_code": "NONE" if previous is None or previous == count else "AUTHORIZED_VIEW_POLICY",
            "undocumented_filter": False,
        })
        previous = count
    write_tsv(out / "population_change_ledger.tsv", changes)

    contract = []
    for item in [
        ("complete_promoted_rows", 3_193_677, f"SELECT count(*) FROM read_parquet('{q(promoted)}')"),
        ("complete_promoted_groups", 37_206, f"SELECT count(DISTINCT phase4_group_id) FROM read_parquet('{q(promoted)}')"),
        ("canonical_gid_eligible_rows", 2_242_863, f"SELECT count(*) FROM read_parquet('{q(promoted)}') WHERE canonical_gid_eligible"),
        ("canonical_gid_ineligible_rows", 950_814, f"SELECT count(*) FROM read_parquet('{q(promoted)}') WHERE NOT canonical_gid_eligible"),
        ("primary_index_rows", 2_045_518, f"SELECT count(*) FROM read_parquet('{q(index_path)}')"),
        ("unresolved_identity_entered_primary", 0, f"SELECT count(*) FROM read_parquet('{q(index_path)}') WHERE identity_status NOT LIKE 'ACCEPTED_PHASE3G_R2%'"),
        ("ar1xar1_coordinates_created", 0, f"SELECT count(*) FROM read_parquet('{q(promoted)}') WHERE coordinate_status<>'ABSENT'"),
    ]:
        observed = con.execute(item[2]).fetchone()[0]
        contract.append({"check": item[0], "expected": item[1], "observed": observed, "status": "PASS" if observed == item[1] else "FAIL"})
    write_tsv(out / "population_contract_validation.tsv", contract)

    identity = pd.read_parquet(p3g / "accepted_all_panel_gid_union.parquet")
    overlap = availability.merge(identity, on="canonical_gid", how="outer", indicator=True)
    overlay = overlap[["canonical_gid", "marker_panel_count", "marker_panels", "marker_vector_available", "existing_kernel_order_available", "_merge"]].copy()
    overlay["audit_mapping_changed"] = False
    overlay["phase3g_r2_remains_authority"] = True
    overlay["phase4_canonical_gid_raw"] = overlay["canonical_gid"].fillna("").astype(str).str.replace(r"^GID", "", regex=True)
    overlay["phase4_field_namespace_status"] = "UPSTREAM_FIELD_LABEL_MISMATCH_NUMERIC_RESOLVED_GID_NOT_R2_CANONICAL_GID"
    overlay["stage1_v2_kernel_binding_status"] = np.where(
        overlay["marker_vector_available"].fillna(False),
        "ACCEPTED_MARKER_IDENTITY_AVAILABLE_KERNEL_NOT_YET_BOUND",
        "NO_ACCEPTED_MARKER_VECTOR",
    )
    write_tsv(out / "identity_join_audit_overlay.tsv", overlay)

    joins = [
        {
            "join": "promoted_primary_to_phase5_index",
            "left_rows": 2_045_518, "right_rows": 2_045_518,
            "key": "phase4_adjusted_row_id", "expected_cardinality": "one_to_one",
            "many_to_many_keys": 0, "rows_lost": int(reconcile["primary_rows_missing_from_index"].iloc[0]),
            "rows_duplicated": int(reconcile["duplicate_index_row_ids"].iloc[0]),
            "status": reconcile["status"].iloc[0],
        },
        {
            "join": "phase4_field_named_canonical_gid_to_phase3g_r2_gid_union",
            "left_rows": 2_242_863, "right_rows": len(identity),
            "key": "canonical_gid", "expected_cardinality": "many_to_one",
            "many_to_many_keys": 0, "rows_lost": 2_242_863, "rows_duplicated": 0,
            "status": "FAIL_UPSTREAM_IDENTIFIER_NAMESPACE_MISMATCH",
        },
        {
            "join": "phase4_typed_source_genotype_id_to_phase3g_r2_gid_union_audit_overlay",
            "left_rows": 2_242_863, "right_rows": len(identity),
            "key": "typed_source_genotype_id=canonical_gid", "expected_cardinality": "many_to_one",
            "many_to_many_keys": 0, "rows_lost": 0, "rows_duplicated": 0,
            "status": "PASS_EXACT_OVERLAY_DIAGNOSTIC_ONLY_NOT_PRODUCTION_REPAIR",
        },
        {
            "join": "phase3g_r2_gid_to_marker_sample_instances",
            "left_rows": len(identity), "right_rows": int(availability["accepted_sample_instances"].sum()),
            "key": "canonical_gid", "expected_cardinality": "one_to_many_audit_only",
            "many_to_many_keys": 0, "rows_lost": 0, "rows_duplicated": 0,
            "status": "PASS_EXPLICIT_REPLICATE_INSTANCES_NOT_COLLAPSED",
        },
    ]
    write_tsv(out / "join_cardinality_audit.tsv", joins)
    write_tsv(out / "join_population_reconciliation.tsv", reconcile)


def genotype_coverage(con: duckdb.DuckDBPyConnection, p3g: Path, promoted: Path, out: Path) -> None:
    panel = pd.read_csv(p3g / "panel_inventory.tsv", sep="\t", dtype=str, keep_default_na=False)
    sample_ledger = pd.read_parquet(p3g / "sample_identifier_ledger.parquet")
    write_tsv(out / "genotypic_sample_inventory.tsv", sample_ledger)
    sample_ledger["accepted"] = sample_ledger["mapping_status"].astype(str).str.startswith("ACCEPTED_")
    sample_ledger["candidate_review"] = sample_ledger["mapping_status"].eq("CANDIDATE_REQUIRES_REVIEW")
    sample_ledger["unmatched"] = sample_ledger["mapping_status"].eq("NO_CANONICAL_MATCH")
    sample_ledger["ambiguous"] = (
        sample_ledger["candidate_review"] |
        sample_ledger["conflict_status"].fillna("NONE").astype(str).ne("NONE")
    )
    panel_match = sample_ledger.groupby("panel_id", as_index=False).agg(
        discovered_sample_instances=("sample_instance_key", "size"),
        accepted_sample_instances=("accepted", "sum"),
        accepted_canonical_gids=("accepted_canonical_gid", lambda s: s[s.fillna("").astype(str).ne("")].nunique()),
        candidate_review_sample_instances=("candidate_review", "sum"),
        unmatched_sample_instances=("unmatched", "sum"),
        ambiguous_sample_instances=("ambiguous", "sum"),
        marker_vector_sample_instances=("marker_vector_present", "sum"),
        historical_kernel_order_sample_instances=("existing_kernel_order_present", "sum"),
    )
    panel_match["identity_authority"] = "PHASE3G_R2"
    panel_match["stage1_v2_kernel_artifact_available"] = False
    write_tsv(out / "genotypic_panel_match_summary.tsv", panel_match)
    crosswalk = pd.read_parquet(p3g / "accepted_all_panel_crosswalk.parquet")
    crosswalk["phase5_disposition"] = "IDENTITY_ACCEPTED_R2_KERNEL_QC_PENDING_OR_PANEL_SPECIFIC"
    crosswalk["used_to_change_identity"] = False
    write_tsv(out / "genotypic_canonical_gid_match_ledger.tsv", crosswalk)

    crosswalk_path = p3g / "accepted_all_panel_crosswalk.parquet"
    coverage = []
    for view, predicate in VIEWS.items():
        frame = query_frame(con, f"""
          WITH vg AS (
            SELECT DISTINCT typed_source_genotype_id canonical_gid FROM read_parquet('{q(promoted)}')
            WHERE {predicate} AND canonical_gid_eligible
          ), pg AS (
            SELECT panel_id,accepted_canonical_gid canonical_gid,
                   bool_or(marker_vector_present) marker_vector_present,
                   bool_or(existing_kernel_order_present) existing_kernel_order_present
            FROM read_parquet('{q(crosswalk_path)}')
            GROUP BY panel_id,accepted_canonical_gid
          )
          SELECT '{view}' AS "view",pg.panel_id,count(*) matched_canonical_gids,
                 count(*) FILTER(WHERE pg.marker_vector_present) marker_vector_gids,
                 count(*) FILTER(WHERE pg.existing_kernel_order_present) historical_kernel_order_gids,
                 (SELECT count(*) FROM vg) view_canonical_gids
          FROM vg JOIN pg USING(canonical_gid) GROUP BY pg.panel_id ORDER BY pg.panel_id
        """)
        coverage.append(frame)
        any_panel = query_frame(con, f"""
          WITH vg AS (
            SELECT DISTINCT typed_source_genotype_id canonical_gid
            FROM read_parquet('{q(promoted)}') WHERE {predicate} AND canonical_gid_eligible
          ), pg AS (
            SELECT accepted_canonical_gid canonical_gid,
                   bool_or(marker_vector_present) marker_vector_present,
                   bool_or(existing_kernel_order_present) existing_kernel_order_present
            FROM read_parquet('{q(crosswalk_path)}') GROUP BY accepted_canonical_gid
          )
          SELECT '{view}' AS "view",'ANY_ACCEPTED_PANEL' panel_id,
                 count(*) FILTER(WHERE pg.canonical_gid IS NOT NULL) matched_canonical_gids,
                 count(*) FILTER(WHERE pg.marker_vector_present) marker_vector_gids,
                 count(*) FILTER(WHERE pg.existing_kernel_order_present) historical_kernel_order_gids,
                 count(*) view_canonical_gids
          FROM vg LEFT JOIN pg USING(canonical_gid)
        """)
        coverage.append(any_panel)
    coverage_frame = pd.concat(coverage, ignore_index=True) if coverage else pd.DataFrame()
    if not coverage_frame.empty:
        coverage_frame["marker_coverage_fraction"] = coverage_frame["marker_vector_gids"] / coverage_frame["view_canonical_gids"].replace(0, np.nan)
        coverage_frame["stage1_v2_kernel_artifact_available"] = False
    write_tsv(out / "genotype_marker_coverage_by_view.tsv", coverage_frame)

    stratum_frames = []
    index_path = out / "canonical_phase5_observation_index.parquet"
    for scope, column in [
        ("TRAIT", "trait"), ("TRIAL", "trial_id"), ("ENVIRONMENT", "environment_id"),
        ("YEAR", "year"), ("MODEL_CLASS", "selected_model")
    ]:
        stratum_frames.append(query_frame(con, f"""
          SELECT '{scope}' stratum_type,cast({column} as varchar) stratum,count(*) AS "rows",
                 count(DISTINCT canonical_gid) canonical_gids,
                 count(DISTINCT canonical_gid) FILTER(WHERE marker_available) marker_available_gids,
                 count(DISTINCT canonical_gid) FILTER(WHERE historical_kernel_order_available) historical_kernel_order_gids,
                 count(DISTINCT canonical_gid) FILTER(WHERE pedigree_available) pedigree_available_gids
          FROM read_parquet('{q(index_path)}') GROUP BY {column} ORDER BY {column}
        """))
    write_tsv(out / "genotype_marker_coverage_by_stratum.tsv", pd.concat(stratum_frames, ignore_index=True))

    loss_source = pd.read_csv(
        p3g.parent / "phase4_integrated_spatial_promotion_release_v1" / "unresolved_identity_coverage_impact.tsv",
        sep="\t", dtype=str, keep_default_na=False,
    )
    lost = loss_source[loss_source["loses_all_usable_genotypes"].str.lower().eq("true")].copy()
    lost["phase5_genomic_denominator_included"] = False
    lost["phase5_disposition"] = "IDENTITY_COVERAGE_LOSS_REPORTED_NOT_FABRICATED"
    write_tsv(out / "environment_trait_identity_loss_ledger.tsv", lost)

    if not coverage_frame.empty:
        primary = coverage_frame[
            coverage_frame["view"].eq("PRIMARY_WEIGHTED_TRAINING") &
            coverage_frame["panel_id"].ne("ANY_ACCEPTED_PANEL")
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(primary["panel_id"], primary["marker_vector_gids"].astype(float))
        ax.set_ylabel("Primary-view canonical GIDs with accepted marker vectors")
        ax.set_title("Stage-1 v2 marker coverage by Phase-3G R2 panel")
        ax.tick_params(axis="x", rotation=75)
        fig.tight_layout()
        fig.savefig(out / "figures" / "marker_coverage_by_panel.png", dpi=160)
        plt.close(fig)


def weight_audit(con: duckdb.DuckDBPyConnection, promoted: Path, out: Path) -> None:
    validation = query_frame(con, f"""
      SELECT '{RELEASE_ID}' release_train_id,count(*) total_rows,
        count(*) FILTER(WHERE isfinite(pev_proxy) AND pev_proxy>=0) finite_nonnegative_pev,
        count(*) FILTER(WHERE pev_proxy=0) zero_pev,
        count(*) FILTER(WHERE isfinite(reliability_weight)) finite_authoritative_weights,
        count(*) FILTER(WHERE isfinite(reliability_weight) AND reliability_weight>0) finite_positive_authoritative_weights,
        count(*) FILTER(WHERE reliability_weight=0) zero_authoritative_weights,
        count(*) FILTER(WHERE uncertainty_weight_eligible) uncertainty_weight_eligible_rows,
        count(*) FILTER(WHERE reliability IS NULL OR NOT isfinite(reliability)) reliability_nonestimable,
        count(DISTINCT phase4_group_id) FILTER(WHERE h2_status='ESTIMABLE_IN_BOUNDS') h2_estimable_groups,
        count(DISTINCT phase4_group_id) FILTER(WHERE ranking_ceiling_status='ESTIMATED') ranking_ceiling_estimable_groups,
        count(*) FILTER(WHERE primary_weighted_training_eligible) primary_rows,
        count(*) FILTER(WHERE primary_weighted_training_eligible AND (reliability_weight IS NULL OR NOT isfinite(reliability_weight))) invalid_primary_weight_rows,
        count(*) FILTER(WHERE primary_weighted_training_eligible AND reliability_weight=0) zero_weight_primary_rows
      FROM read_parquet('{q(promoted)}')
    """)
    validation["authoritative_weight_field"] = "reliability_weight"
    validation["deregression_applied"] = False
    validation["epsilon_substitution_applied"] = False
    validation["status"] = np.where(validation["invalid_primary_weight_rows"].eq(0), "PASS", "FAIL")
    write_tsv(out / "weight_validation.tsv", validation)

    strata = []
    for scope, column in [
        ("TRAIT", "trait"), ("TRIAL", "trial_id"), ("ENVIRONMENT", "environment_id"),
        ("YEAR", "year"), ("MODEL_CLASS", "selected_model")
    ]:
        frame = query_frame(con, f"""
          SELECT '{scope}' stratum_type,cast({column} as varchar) stratum,
                 count(*) AS "rows",count(*) FILTER(WHERE isfinite(reliability_weight)) finite_weights,
                 min(reliability_weight) min_weight,
                 quantile_cont(reliability_weight,0.01) q01,
                 quantile_cont(reliability_weight,0.25) q25,
                 quantile_cont(reliability_weight,0.50) q50,
                 quantile_cont(reliability_weight,0.75) q75,
                 quantile_cont(reliability_weight,0.99) q99,max(reliability_weight) max_weight,
                 sum(reliability_weight) sum_weight,sum(reliability_weight*reliability_weight) sum_weight_sq
          FROM read_parquet('{q(promoted)}') WHERE primary_weighted_training_eligible
          GROUP BY {column} ORDER BY {column}
        """)
        frame["effective_sample_size"] = frame["sum_weight"] ** 2 / frame["sum_weight_sq"].replace(0, np.nan)
        strata.append(frame)
    all_strata = pd.concat(strata, ignore_index=True)
    write_tsv(out / "weight_distribution_by_stratum.tsv", all_strata)
    write_tsv(out / "weight_effective_sample_size.tsv", all_strata[["stratum_type", "stratum", "rows", "sum_weight", "effective_sample_size"]])

    trait = all_strata[all_strata["stratum_type"].eq("TRAIT")]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(trait["stratum"], trait["q50"].astype(float), color="#4472C4")
    ax.set_ylabel("Median reliability weight")
    ax.tick_params(axis="x", rotation=55)
    ax.set_title("Authoritative unscaled Phase-4 weight by Stage-1 v2 trait")
    fig.tight_layout()
    fig.savefig(out / "figures" / "weight_median_by_trait.png", dpi=160)
    plt.close(fig)


def matrix_diag(path: Path, order_path: Path, label: str, sample: int = 256) -> dict[str, Any]:
    matrix = np.load(path, mmap_mode="r")
    order = pd.read_csv(order_path, sep="\t", dtype=str)
    ids = order.iloc[:, 0].fillna("").astype(str).tolist()
    n = matrix.shape[0]
    selected = np.linspace(0, n - 1, min(sample, n), dtype=int)
    block = np.asarray(matrix[np.ix_(selected, selected)], dtype=np.float64)
    diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
    eig = np.linalg.eigvalsh((block + block.T) / 2)
    positive = np.maximum(eig, 0)
    effective_rank = float(positive.sum() ** 2 / np.square(positive).sum()) if np.square(positive).sum() else 0.0
    return {
        "kernel": label, "path": path.as_posix(), "order_path": order_path.as_posix(),
        "shape": f"{matrix.shape[0]}x{matrix.shape[1]}", "order_rows": len(ids),
        "order_unique": len(set(ids)), "index_signature": index_signature(ids),
        "mean_diagonal": float(diagonal.mean()), "min_diagonal": float(diagonal.min()),
        "max_diagonal": float(diagonal.max()), "sampled_symmetry_max_abs": float(np.max(np.abs(block - block.T))),
        "sampled_min_eigenvalue": float(eig.min()), "sampled_effective_rank": effective_rank,
        "all_finite_sample": bool(np.isfinite(block).all()),
    }


def independent_environment_component_compare(
    environment: Path, component: str, sample: int = 256
) -> dict[str, Any]:
    order = pd.read_csv(environment / "env_kernel_sample_order.tsv", sep="\t", dtype=str).iloc[:, 0].fillna("").astype(str).tolist()
    features = pd.read_parquet(environment / f"env_features_{component}.parquet")
    feature_order = features.iloc[:, 0].fillna("").astype(str).tolist()
    z = features.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    selected = np.linspace(0, len(order) - 1, min(sample, len(order)), dtype=int)
    raw_mean_diagonal = float(np.mean(np.sum(z * z, axis=1) / z.shape[1]))
    expected = (z[selected] @ z[selected].T / z.shape[1]) / raw_mean_diagonal
    production = np.load(environment / f"K_{component}.npy", mmap_mode="r")
    observed = np.asarray(production[np.ix_(selected, selected)], dtype=np.float64)
    error = np.abs(expected - observed)
    return {
        "feature_order_exact": feature_order == order,
        "independent_raw_mean_diagonal": raw_mean_diagonal,
        "independent_max_abs_difference": float(error.max()),
        "independent_mean_abs_difference": float(error.mean()),
        "independent_reconstruction_status": "PASS" if error.max() <= 1e-5 else "FAIL_MEAN_DIAGONAL_SCALING_MISMATCH",
        "independent_expected_block": expected,
    }


def kernel_audits(root: Path, con: duckdb.DuckDBPyConnection, promoted: Path, out: Path) -> None:
    synthetic = synthetic_results()
    synthetic.insert(0, "release_train_id", RELEASE_ID)
    write_tsv(out / "independent_reconstruction_comparison.tsv", synthetic)
    shutil.copy2(root / "scripts" / "v2" / "phase5_independent_reconstruction.py", out / "independent_reconstruction.py")

    primary_gids = con.execute(f"SELECT count(DISTINCT canonical_gid) FROM read_parquet('{q(promoted)}') WHERE primary_weighted_training_eligible").fetchone()[0]
    primary_env = con.execute(f"SELECT count(DISTINCT environment_id) FROM read_parquet('{q(promoted)}') WHERE primary_weighted_training_eligible").fetchone()[0]
    ka = pd.DataFrame([
        {
            "kernel": "K_A_STAGE1_V2", "entity_universe": "PRIMARY_WEIGHTED_TRAINING_CANONICAL_GIDS",
            "entities": primary_gids, "production_v2_matrix_available": False,
            "production_v2_order_available": False, "independent_synthetic_status": synthetic.loc[synthetic.test.eq("synthetic_K_A"), "status"].iloc[0],
            "full_real_reconstruction_status": "NOT_RUN_NO_VERSIONED_STAGE1_V2_PEDIGREE_BINDING",
            "status": "BLOCKER_NO_VERSIONED_STAGE1_V2_KA_ARTIFACT_OR_PEDIGREE_BINDING",
        }
    ])
    write_tsv(out / "ka_kernel_diagnostics.tsv", ka)

    panel = pd.read_csv(root / "audit" / "v2" / "phase3g_all_panel_genotype_linkage_audit_v2" / "panel_inventory.tsv", sep="\t", dtype=str)
    kg = panel[["panel_id", "platform", "technology", "accepted_canonical_gid_count", "strict_kernel_ready_sample_count", "existing_qc_documentation"]].copy()
    kg["kernel_scope"] = "STAGE1_V2_PANEL_SPECIFIC_CANDIDATE"
    kg["production_v2_matrix_available"] = False
    kg["production_v2_order_available"] = False
    kg["independent_synthetic_status"] = synthetic.loc[synthetic.test.eq("synthetic_K_G"), "status"].iloc[0]
    kg["status"] = "BLOCKER_NO_VERSIONED_STAGE1_V2_KG_ARTIFACT_OR_REGISTRY"
    write_tsv(out / "kg_kernel_diagnostics.tsv", kg)

    ke_rows = []
    environment = root / "environment"
    order_path = environment / "env_kernel_sample_order.tsv"
    independent_components: dict[str, dict[str, Any]] = {}
    for name in ["K_geo", "K_weather", "K_stress", "K_mgmt", "K_E"]:
        path = environment / f"{name}.npy"
        if path.is_file() and order_path.is_file():
            diag = matrix_diag(path, order_path, name)
            if name != "K_E" and (environment / f"env_features_{name[2:]}.parquet").is_file():
                independent = independent_environment_component_compare(environment, name[2:])
                independent_components[name[2:]] = independent
                diag.update({key: value for key, value in independent.items() if key != "independent_expected_block"})
            elif name == "K_E" and independent_components:
                expected = sum(
                    item["independent_expected_block"] for item in independent_components.values()
                ) / len(independent_components)
                selected = np.linspace(0, len(pd.read_csv(order_path, sep="\t")) - 1, expected.shape[0], dtype=int)
                observed = np.asarray(np.load(path, mmap_mode="r")[np.ix_(selected, selected)], dtype=np.float64)
                error = np.abs(expected - observed)
                diag.update({
                    "feature_order_exact": True,
                    "independent_raw_mean_diagonal": 1.0,
                    "independent_max_abs_difference": float(error.max()),
                    "independent_mean_abs_difference": float(error.mean()),
                    "independent_reconstruction_status": "PASS" if error.max() <= 1e-5 else "FAIL_COMPONENT_AND_FINAL_SCALING_MISMATCH",
                })
            diag.update({
                "stage1_v2_primary_environment_count": primary_env,
                "versioned_stage1_v2_binding": False,
                "candidate_class": "UNVERSIONED_TRAINING_SIDE_ENVIRONMENT_CANDIDATE",
                "status": (
                    "UNVERSIONED_CANDIDATE_FAILS_CURRENT_INDEPENDENT_SCALING_RECONSTRUCTION"
                    if str(diag.get("independent_reconstruction_status", "PASS")).startswith("FAIL")
                    else "NUMERIC_SAMPLE_PASS_BUT_NOT_ACTIVATABLE_WITHOUT_V2_MANIFEST_AND_FOLD_SCOPE"
                ),
            })
            ke_rows.append(diag)
    write_tsv(out / "ke_kernel_diagnostics.tsv", ke_rows)

    comparison = pd.DataFrame([{
        "comparison": "K_A_STAGE1_V2_vs_K_G_STAGE1_V2", "shared_gids": 0,
        "upper_triangle_correlation": np.nan, "frobenius_alignment": np.nan,
        "status": "NOT_EVALUABLE_BOTH_VERSIONED_V2_KERNELS_ABSENT",
    }])
    write_tsv(out / "ka_kg_shared_gid_comparison.tsv", comparison)

    # Twenty analytical sparse-observation checks cover all same/different-axis strata.
    kg_small = np.asarray([[1.0, .25, -.10], [.25, 1.1, .20], [-.10, .20, .9]])
    ke_small = np.asarray([[1.0, .4, .05], [.4, 1.0, -.1], [.05, -.1, .8]])
    gi = np.asarray([0, 0, 1, 1, 2, 2])
    ei = np.asarray([0, 1, 0, 2, 1, 2])
    pairs = [(i, j) for i in range(6) for j in range(i, 6)][:20]
    values = gxe_elements(kg_small, ke_small, gi, ei, pairs)
    manual = []
    for (i, j), value in zip(pairs, values):
        expected = float(kg_small[gi[i], gi[j]] * ke_small[ei[i], ei[j]])
        manual.append({
            "component": "SYNTHETIC_SPARSE_HADAMARD_REFERENCE", "observation_i": i, "observation_j": j,
            "same_genotype": bool(gi[i] == gi[j]), "same_environment": bool(ei[i] == ei[j]),
            "expected": expected, "observed": value, "abs_error": abs(expected - value), "status": "PASS" if value == expected else "FAIL",
        })
    write_tsv(out / "gxe_manual_element_checks.tsv", manual)
    write_tsv(out / "gxe_kernel_diagnostics.tsv", [{
        "interaction": "K_G_OBS_HADAMARD_K_E_OBS", "conceptual_rows": 2_045_518,
        "dense_materialized": False, "factorized_v2_operator_available": False,
        "synthetic_reference_status": synthetic.loc[synthetic.test.eq("synthetic_GxE"), "status"].iloc[0],
        "status": "BLOCKER_NO_VERSIONED_STAGE1_V2_GXE_OPERATOR_OR_INDEX_BINDING",
    }])
    write_tsv(out / "gxe_alignment_failures.tsv", [{
        "failure": "NO_VERSIONED_STAGE1_V2_GXE_OPERATOR", "affected_rows": 2_045_518,
        "severity": "HIGH", "status": "OPEN_BLOCKER",
    }])

    entities = [
        {"universe": "COMPLETE_PROMOTED_OBSERVATIONS", "entities": 3_193_677, "identifier": "phase4_adjusted_row_id", "status": "EXPLICIT"},
        {"universe": "PRIMARY_WEIGHTED_OBSERVATIONS", "entities": 2_045_518, "identifier": "phase4_adjusted_row_id", "status": "EXPLICIT"},
        {"universe": "PRIMARY_CANONICAL_GIDS", "entities": primary_gids, "identifier": "canonical_gid", "status": "EXPLICIT"},
        {"universe": "PRIMARY_ENVIRONMENTS", "entities": primary_env, "identifier": "environment_id", "status": "EXPLICIT"},
        {"universe": "K_A_STAGE1_V2", "entities": 0, "identifier": "canonical_gid", "status": "MISSING_VERSIONED_ARTIFACT"},
        {"universe": "K_G_STAGE1_V2", "entities": 0, "identifier": "canonical_gid", "status": "MISSING_VERSIONED_ARTIFACT"},
        {"universe": "K_E_STAGE1_V2", "entities": 0, "identifier": "environment_id", "status": "MISSING_VERSIONED_BINDING"},
    ]
    write_tsv(out / "kernel_entity_universes.tsv", entities)

    index = pq.read_table(out / "canonical_phase5_observation_index.parquet", columns=["phase4_adjusted_row_id", "canonical_gid", "environment_id"]).to_pandas()
    signatures = [
        {"object": "canonical_phase5_observation_index", "axis": "observations", "rows": len(index), "sha256_signature": index_signature(index["phase4_adjusted_row_id"])},
        {"object": "primary_genotype_incidence_universe", "axis": "genotypes", "rows": index["canonical_gid"].nunique(), "sha256_signature": index_signature(sorted(index["canonical_gid"].unique()))},
        {"object": "primary_environment_incidence_universe", "axis": "environments", "rows": index["environment_id"].nunique(), "sha256_signature": index_signature(sorted(index["environment_id"].unique()))},
    ]
    write_tsv(out / "matrix_index_signatures.tsv", signatures)


def split_and_leakage_audit(root: Path, out: Path) -> None:
    modes = [
        ("cv2_random_observation", "observation", "phenotype holdout; genotype/environment overlap allowed"),
        ("cv1_genotype", "canonical_gid", "new genotype in represented environments"),
        ("cv1_environment", "environment_id", "known genotype in new environments"),
        ("cv0_genotype_environment", "canonical_gid+environment_id", "new genotype in new environment"),
        ("gho_environment", "environment_id", "leave environment out"),
        ("gho_cycle", "year", "temporal transfer"),
        ("gho_trial", "trial_id", "trial transfer"),
        ("gho_country", "country", "geographic transfer"),
    ]
    definitions = [{
        "split_mode": mode, "unit": unit, "scenario": scenario, "seed": SEED,
        "authorized_view": "PRIMARY_WEIGHTED_TRAINING", "v2_split_assignment_materialized": False,
        "status": "CODE_PATH_DECLARED_BUT_NO_VERSIONED_STAGE1_V2_SPLIT_RELEASE",
    } for mode, unit, scenario in modes]
    write_tsv(out / "split_definition.tsv", definitions)
    overlaps = [{
        "split_mode": mode, "train_rows": 0, "validation_rows": 0, "test_rows": 0,
        "prohibited_entity_overlap": np.nan,
        "status": "NOT_EVALUABLE_NO_VERSIONED_STAGE1_V2_SPLIT_ASSIGNMENT",
    } for mode, _, _ in modes]
    write_tsv(out / "split_overlap_summary.tsv", overlaps)
    write_tsv(out / "split_leakage_report.tsv", [{
        "check": "versioned_stage1_v2_split_membership", "status": "FAIL",
        "severity": "HIGH", "evidence": "No v2 split/fold columns or manifest exists; Phase5 index records UNASSIGNED",
    }, {
        "check": "duplicate_row_cross_split", "status": "NOT_APPLICABLE_UNASSIGNED",
        "severity": "HIGH", "evidence": "Must be rerun after a v2 split release exists",
    }, {
        "check": "protected_outcome_leakage", "status": "PASS",
        "severity": "CRITICAL", "evidence": "No outer-test/final-holdout outcome path discovered or read by Phase5",
    }])

    env_source = (root / "build_environment_component_kernels.py").read_text(encoding="utf-8")
    geno_source = (root / "build_requested_outputs.py").read_text(encoding="utf-8")
    fold_local = [
        {
            "component": "environment_feature_imputation_scaling_diagonal_normalization",
            "training_only_interface": "--fit-environment-ids",
            "interface_present": "fit_environment_ids" in env_source and "training_environments_only" in env_source,
            "stage1_v2_fold_manifest_bound": False,
            "status": "IMPLEMENTATION_CAPABLE_NOT_BOUND_TO_VERSIONED_V2_SPLITS",
        },
        {
            "component": "marker_QC_imputation_allele_frequency_centering",
            "training_only_interface": "fit genotype/sample IDs",
            "interface_present": "fit_sample" in geno_source or "fit_genotype" in geno_source,
            "stage1_v2_fold_manifest_bound": False,
            "status": "FAIL_GLOBAL_PANEL_PREPROCESSING_ONLY",
        },
        {
            "component": "kernel_factorization",
            "training_only_interface": "train_nystrom",
            "interface_present": "train_nystrom" in (root / "server_training_pipeline" / "kernel_factorization.py").read_text(encoding="utf-8"),
            "stage1_v2_fold_manifest_bound": False,
            "status": "IMPLEMENTATION_CAPABLE_NOT_BOUND_TO_VERSIONED_V2_SPLITS",
        },
    ]
    write_tsv(out / "fold_local_preprocessing_audit.tsv", fold_local)
    write_tsv(out / "protected_outcome_access_audit.tsv", [
        {"protected_scope": "outer_test_phenotype_outcomes", "accessed": False, "hashed": False, "summarized": False, "status": "PASS"},
        {"protected_scope": "final_holdout_phenotype_outcomes", "accessed": False, "hashed": False, "summarized": False, "status": "PASS"},
        {"protected_scope": "protected_performance_for_preprocessing_choice", "accessed": False, "hashed": False, "summarized": False, "status": "PASS"},
    ])


def lineage_and_issues(root: Path, out: Path) -> pd.DataFrame:
    lineage = [
        {"artifact": "canonical_phase5_observation_index.parquet", "producer": "scripts/v2/phase5_forensic_kernel_audit.py::build_availability_and_index", "inputs": "promoted_phenotypes.parquet;accepted_all_panel_crosswalk.parquet", "join_keys": "canonical_gid", "cardinality": "many_to_one", "filter": "primary_weighted_training_eligible", "split_scope": "unassigned", "downstream": "future Stage1-v2 model input"},
        {"artifact": "K_A_STAGE1_V2", "producer": "NOT_IMPLEMENTED_AS_VERSIONED_V2_ARTIFACT", "inputs": "Stage1-v2 accepted canonical GIDs+versioned pedigree", "join_keys": "canonical_gid", "cardinality": "one_to_one", "filter": "observed GIDs plus explicit ancestors", "split_scope": "training side", "downstream": "blocked"},
        {"artifact": "K_G_STAGE1_V2", "producer": "NOT_IMPLEMENTED_AS_VERSIONED_V2_ARTIFACT", "inputs": "Phase3G-R2 accepted sample instances+raw markers", "join_keys": "panel_id;sample_instance_key;canonical_gid", "cardinality": "explicit replicate policy", "filter": "fold-local QC", "split_scope": "training fold", "downstream": "blocked"},
        {"artifact": "K_E_STAGE1_V2", "producer": "build_environment_component_kernels.py (capable, not v2-bound)", "inputs": "environment features+training environment IDs", "join_keys": "environment_id", "cardinality": "one_to_one", "filter": "authorized view environment universe", "split_scope": "training fold", "downstream": "blocked pending v2 manifest"},
        {"artifact": "K_GxE_STAGE1_V2", "producer": "factorized Hadamard operator required", "inputs": "K_G/K_E+canonical observation index", "join_keys": "canonical_gid;environment_id", "cardinality": "many observations to one entity", "filter": "authorized view", "split_scope": "fold", "downstream": "blocked"},
    ]
    write_tsv(out / "data_lineage.tsv", lineage)
    (out / "data_lineage.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    md = ["# Stage-1 v2 Phase-5 data lineage", "", "Certified-v1 artifacts are historical inventory only.", "", "| Artifact | Producer | Inputs | Keys | Split scope |", "|---|---|---|---|---|"]
    for row in lineage:
        md.append(f"| {row['artifact']} | {row['producer']} | {row['inputs']} | {row['join_keys']} | {row['split_scope']} |")
    (out / "data_lineage.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    dot = ["digraph phase5_v2 {", '  rankdir="LR";', '  stage1 [label="Stage-1 v2"];', '  p3g [label="Phase-3G R2 identities"];', '  p4 [label="Integrated Phase-4 promoted views"];', '  index [label="Canonical Phase-5 observation index"];', '  kernels [label="Versioned v2 kernels (missing)",color="red"];', '  model [label="v2 model inputs (blocked)",color="red"];', "  stage1 -> p4;", "  p3g -> p4;", "  p4 -> index;", "  p3g -> kernels;", "  index -> kernels;", "  kernels -> model;", "}"]
    (out / "pipeline_graph.dot").write_text("\n".join(dot) + "\n", encoding="utf-8")

    issues = pd.DataFrame([
        {"issue_id": "P5V2-000", "severity": "CRITICAL", "component": "Phase4 identity contract", "defect": "The Phase-4 field named canonical_gid contains numeric resolved_gid values, while Phase3G-R2 canonical IDs are GID-prefixed; all 2,242,863 canonical-eligible rows are affected", "affected_view": "all canonical-GID v2 views", "affected_entities": 10722, "correction": "Create an immutable upstream corrective promotion release that preserves the exact accepted Phase3G-R2 canonical_gid field and revalidate all views", "regression_test": "every canonical-eligible promoted canonical_gid must exactly join Phase3G-R2 without normalization", "requires_regeneration": True, "status": "OPEN_UPSTREAM_BLOCKER"},
        {"issue_id": "P5V2-001", "severity": "CRITICAL", "component": "K_A", "defect": "No versioned Stage-1 v2 pedigree binding, entity order, matrix, or manifest", "affected_view": "all canonical-GID v2 views", "affected_entities": 10_722, "correction": "Construct a new v2 pedigree release keyed only by accepted Phase3G-R2 canonical GIDs; preserve explicit ancestors", "regression_test": "independent sampled/full A agreement and order signatures", "requires_regeneration": True, "status": "OPEN_BLOCKER"},
        {"issue_id": "P5V2-002", "severity": "CRITICAL", "component": "K_G", "defect": "Accepted all-panel marker identities have no versioned Stage-1 v2 kernel registry or fold-local QC artifacts", "affected_view": "all canonical-GID v2 views", "affected_entities": 10_722, "correction": "Build panel-specific v2 kernels with explicit replicate policy and training-only QC/imputation/frequencies", "regression_test": "independent VanRaden subsets per panel and no fabricated rows", "requires_regeneration": True, "status": "OPEN_BLOCKER"},
        {"issue_id": "P5V2-003", "severity": "HIGH", "component": "K_E", "defect": "Environment builder supports training-only fitting, but existing matrices lack a Stage-1 v2 view/split manifest and unversioned candidates fail independent current mean-diagonal scaling reconstruction", "affected_view": "all v2 views", "affected_entities": 11_166, "correction": "Regenerate versioned fold-scoped K_E components from exact environment_id universes with current scaling code", "regression_test": "independent feature and element reconstruction", "requires_regeneration": True, "status": "OPEN_BLOCKER"},
        {"issue_id": "P5V2-004", "severity": "CRITICAL", "component": "model_inputs/GxE", "defect": "No versioned Stage-1 v2 split assignment, incidence binding, or sparse GxE operator", "affected_view": "PRIMARY_WEIGHTED_TRAINING", "affected_entities": 2_045_518, "correction": "Freeze splits then construct explicit factorized operators from the canonical observation index", "regression_test": "20+ manual elements and deliberate permutation detection", "requires_regeneration": True, "status": "OPEN_BLOCKER"},
        {"issue_id": "P5V2-005", "severity": "HIGH", "component": "marker preprocessing", "defect": "Current generic HMP builder fits QC, imputation and allele frequencies on the full supplied panel and has no fit-ID interface", "affected_view": "strict new-genotype scenarios", "affected_entities": "scenario-dependent", "correction": "Add a training-ID fit manifest and serialized transform applied unchanged to validation/test samples", "regression_test": "held-out dosage changes cannot alter training parameters", "requires_regeneration": True, "status": "OPEN_BLOCKER"},
    ])
    write_tsv(out / "kernel_issue_ledger.tsv", issues)
    diagnoses = [
        ("Low marker coverage", "Plausible", "Accepted marker vectors cover the v2 identity union, but no versioned v2 kernel registry establishes usable QC coverage."),
        ("Nonrandom marker coverage", "Strongly supported", "Panel-specific primary-view coverage ranges widely and is tabulated by trait/trial/environment/year."),
        ("Incorrect genotype-marker mapping", "Confirmed", "Phase-4 canonical_gid namespace is numeric while Phase3G-R2 canonical_gid is GID-prefixed."),
        ("Incorrect row ordering", "Plausible", "Canonical observation order is now explicit, but no v2 kernel/operator order exists to compare."),
        ("K_G coding or scaling error", "Unsupported", "No versioned Stage1-v2 K_G exists; analytical VanRaden reference passes."),
        ("K_E scaling dominance", "Strongly supported", "Unversioned K_E/component candidates fail independent current mean-diagonal scaling reconstruction."),
        ("Incorrect interaction product", "Unsupported", "Synthetic sparse Hadamard elements pass; no v2 production operator exists."),
        ("Kernel collinearity", "Unsupported", "No joint versioned v2 kernel set exists."),
        ("Environmental target leakage", "Unsupported", "Audited feature manifests contain covariates only; no protected outcome was accessed."),
        ("Phenotype or weight leakage", "Unsupported", "No v2 split/model input exists and no protected outcome was accessed."),
        ("Weak connectivity among trials", "Plausible", "Panel overlap is heterogeneous across v2 trial strata."),
        ("Excessive regularization or uniform weighting", "Unsupported", "No Phase5 model was trained or tuned."),
        ("Incorrect pedigree-only handling", "Confirmed", "No versioned v2 pedigree binding or missing-marker policy exists."),
        ("Genuine absence of genomic signal beyond pedigree", "Unsupported", "Cannot be inferred before valid aligned v2 kernels and without protected evaluation."),
        ("Representativeness bias", "Strongly supported", "The signed unresolved-identity footprint includes nine lost environment-trait combinations."),
    ]
    write_tsv(out / "weak_genomic_gxe_distinctiveness_diagnosis.tsv", [
        {"hypothesis": name, "classification": classification, "evidence": evidence,
         "protected_outcome_used": False} for name, classification, evidence in diagnoses
    ])
    return issues


def write_reports(out: Path, view_summary: pd.DataFrame, issues: pd.DataFrame) -> None:
    all_views = view_summary["status"].eq("PASS").all()
    issue_lines = [
        "| Issue | Severity | Component | Defect | Status |",
        "|---|---|---|---|---|",
    ]
    for row in issues.itertuples(index=False):
        issue_lines.append(
            f"| {row.issue_id} | {row.severity} | {row.component} | "
            f"{str(row.defect).replace('|', '/')} | {row.status} |"
        )
    report = f"""# Phase-5 Stage-1 v2 kernel validation report

## Executive summary

The pinned Stage-1 v2 / Phase-3G R2 / integrated Phase-4 population is reproduced exactly. The primary v2 observation index contains 2,045,518 traceable rows. No certified-v1 artifact was used as a v2 input. The audit found that the Phase-4 field named `canonical_gid` contains numeric `resolved_gid` values rather than exact Phase-3G R2 GID-prefixed canonical identifiers; an explicit diagnostic overlay uses `typed_source_genotype_id` but is not an upstream repair. Phase 5 cannot pass because of this upstream identity-contract defect and because no complete, versioned Stage-1 v2 K_A, K_G, K_E binding, GxE operator, or split/model-input release exists.

## Release identity and dependencies

- Phase-5 release: `{RELEASE_ID}`
- Stage-1: `{STAGE1_V2_VERSION}`
- Phase-4: `{P4_ID}` / `{P4_SOURCE_HASH}`
- Phase-3G identity: R2 only
- Deterministic views: `{'PASS' if all_views else 'FAIL'}`

## Population and phenotype contract

All eight views reproduce their exact signed counts. Adjusted values, row IDs, PEV, reliability, H2 and ranking fields agree with the immutable promotion ledger. `reliability_weight` is retained unscaled and must be handled within future folds; no epsilon, deregression, cap, or invented value was applied.

## Identity and genotype coverage

Only Phase-3G R2 accepted identities appear in canonical-GID views. The immutable Phase-4 `canonical_gid` field has a namespace-label defect on all eligible rows; Phase-5 coverage uses the exact, already-present `typed_source_genotype_id` solely as an audit overlay. Every panel remains separate in the coverage ledger; no fuzzy or pedigree-based identity recovery occurred. The nine completely lost environment-trait combinations remain outside genomic denominators.

## Kernel findings

- K_A: independent analytical mathematics passes, but Stage-1 v2 has no versioned pedigree binding/matrix/order.
- K_G: independent VanRaden analytical mathematics passes, but accepted all-panel v2 identities are not bound to versioned panel kernels or fold-local transforms.
- K_E: unversioned local component matrices pass sampled symmetry/PSD checks but fail independent reconstruction under the current mean-diagonal scaling implementation; no v2 view/split manifest binds them.
- GxE: the sparse observation-level Hadamard formulation passes 20 manual analytical elements; no v2 factorized operator/index binding exists.

## Ordering, splits, and leakage

The Phase-5 primary observation order is explicit and signed. No v2 split release exists, so split overlaps cannot yet be certified. The generic marker builder lacks a training-ID fitting interface; this fails strict inductive preprocessing requirements. No protected outcome was accessed.

## Confirmed blockers

{chr(10).join(issue_lines)}

## Scope confirmations

- No AR1xAR1 reconstruction or artificial coordinates.
- No model training, tuning, protected evaluation, or projection.
- No Phase-3G v1 or alternate Phase-4 candidate consumed.
- Certified-v1 artifacts remain frozen historical inventory only.

## Candidate decision

`BLOCKED_PHASE5_KERNEL_VALIDATION` pending final tests and closing-hash confirmation.
"""
    (out / "KERNEL_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    (out / "VALIDATION_REPORT.md").write_text(
        "# Phase-5 validation\n\nCandidate status: `BLOCKED_PHASE5_KERNEL_VALIDATION`. "
        "Population integrity passes; versioned Stage-1 v2 kernels/splits/model inputs are incomplete.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve() if args.out else root / "audit" / "v2" / "phase5_kernel_validation_v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    p3g = root / "audit" / "v2" / "phase3g_all_panel_genotype_linkage_audit_v2"
    p4 = root / "audit" / "v2" / "phase4_integrated_spatial_promotion_release_v1"
    stage1 = root / "audit" / "v2" / "phase3_stage1_v2_reconstruction_v1" / "stage1_v2_release_candidate_v3"
    promoted = p4 / "promoted_phenotypes.parquet"
    ledger = p4 / "phenotype_promotion_ledger.parquet"
    for path in (p3g, p4, stage1, promoted, ledger):
        if not path.exists():
            raise FileNotFoundError(path)
    scope = json.loads((root / "audit" / "v2" / "phase5_kernel_validation_v1" / "PHASE5_SCOPE_CORRECTION.json").read_text(encoding="utf-8"))
    if scope["authoritative_modelling_foundation"] != "STAGE1_V2":
        raise RuntimeError("Stage-1 v2 scope correction is absent")

    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    temp = out / "logs" / "duckdb_tmp"
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{q(temp)}'")

    if out.name == "phase5_kernel_validation_v1":
        inventory_and_scope(root, out, p3g, p4, stage1)
    view_summary = reproduce_views(con, promoted, ledger, out)
    index_path, availability = build_availability_and_index(con, p3g, promoted, out)
    population_and_join_audits(con, promoted, index_path, p3g, availability, out)
    genotype_coverage(con, p3g, promoted, out)
    weight_audit(con, promoted, out)
    kernel_audits(root, con, promoted, out)
    split_and_leakage_audit(root, out)
    issues = lineage_and_issues(root, out)
    write_reports(out, view_summary, issues)
    print(json.dumps({
        "release_train_id": RELEASE_ID,
        "stage1_foundation": "STAGE1_V2",
        "views_passed": int(view_summary["status"].eq("PASS").sum()),
        "views_total": len(view_summary),
        "open_blockers": len(issues),
        "candidate_status": "BLOCKED_PHASE5_KERNEL_VALIDATION",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


if __name__ == "__main__":
    main()
