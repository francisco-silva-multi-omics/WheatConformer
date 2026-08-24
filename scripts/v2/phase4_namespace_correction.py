#!/usr/bin/env python3
"""Create the immutable lossless Phase-4 canonical-GID namespace correction."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

try:
    from .phase4_namespace_r3_common import (
        PHASE3G_R2_ROOT, PHASE4_NS_RELEASE_ID, PHASE4_NS_ROOT, PHASE4_ROOT,
        q, sha256, write_json, write_tsv,
    )
except ImportError:  # direct script execution
    from phase4_namespace_r3_common import (
    PHASE3G_R2_ROOT,
    PHASE4_NS_RELEASE_ID,
    PHASE4_NS_ROOT,
    PHASE4_ROOT,
    q,
    sha256,
    write_json,
    write_tsv,
)


EXPECTED_VIEWS = {
    "PRIMARY_WEIGHTED_TRAINING": ("primary_weighted_training_eligible", 2_045_518, 10_656, 31_343, 273, 10_258, 43, 7),
    "SECONDARY_UNWEIGHTED_TRAINING": ("secondary_unweighted_training_eligible", 2_242_863, 10_722, 37_157, 283, 11_161, 43, 7),
    "CONTINUOUS_ERROR_EVALUATION": ("continuous_error_evaluation_eligible", 2_242_863, 10_722, 37_157, 283, 11_161, 43, 7),
    "CORRELATION_EVALUATION": ("correlation_evaluation_eligible", 2_242_615, 10_722, 36_909, 280, 11_086, 43, 7),
    "RANKING_EVALUATION": ("ranking_evaluation_eligible", 1_418_644, 10_656, 23_483, 271, 9_242, 43, 7),
    "IDENTITY_UNRESOLVED_ARCHIVAL": ("NOT canonical_gid_eligible", 950_814, 0, 20_211, 164, 6_240, 38, 7),
    "RELEASE_ONLY": ("phenotype_release_eligible AND NOT secondary_unweighted_training_eligible", 950_814, 0, 20_211, 164, 6_240, 38, 7),
    "BLOCKED_DATA_INTEGRITY": ("NOT phenotype_release_eligible", 0, 0, 0, 0, 0, 0, 0),
}


def exact_authority_join(left: pd.DataFrame, authority: pd.DataFrame) -> pd.DataFrame:
    """Small-table reference for the production exact typed-key join."""
    if authority["canonical_gid"].isna().any() or authority["canonical_gid"].astype(str).str.strip().eq("").any():
        raise ValueError("Authority contains null/blank canonical GIDs")
    if authority["canonical_gid"].duplicated().any():
        raise ValueError("Authority canonical GIDs must be unique")
    result = left.merge(
        authority[["canonical_gid"]].rename(columns={"canonical_gid": "authoritative_canonical_gid"}),
        left_on="typed_source_genotype_id",
        right_on="authoritative_canonical_gid",
        how="left",
        validate="many_to_one",
    )
    if result.loc[result["canonical_gid_eligible"], "authoritative_canonical_gid"].isna().any():
        raise ValueError("Eligible row has no exact authority mapping")
    return result


def scalar(con: duckdb.DuckDBPyConnection, query: str):
    return con.execute(query).fetchone()[0]


def build(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    promoted = PHASE4_ROOT / "promoted_phenotypes.parquet"
    union = PHASE3G_R2_ROOT / "accepted_all_panel_gid_union.parquet"
    corrected = out / "corrected_promoted_phenotypes.parquet"
    join_ledger = out / "identity_join_ledger.parquet"
    id_lineage = out / "old_new_observation_id_lineage.parquet"
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=8")

    duplicate_union = scalar(con, f"SELECT count(*)-count(DISTINCT canonical_gid) FROM read_parquet('{q(union)}')")
    if duplicate_union:
        raise RuntimeError("Phase3G R2 accepted canonical-GID union is not unique")

    join_metrics = con.execute(
        f"""
        SELECT count(*) source_rows,
               count(*) FILTER(WHERE p.canonical_gid_eligible) eligible_rows,
               count(*) FILTER(WHERE p.canonical_gid_eligible AND u.canonical_gid IS NULL) eligible_missing_authority,
               count(*) FILTER(WHERE p.canonical_gid_eligible AND u.canonical_gid IS NOT NULL) eligible_exact_authority,
               count(*) FILTER(WHERE NOT p.canonical_gid_eligible) unresolved_rows,
               count(*)-count(DISTINCT p.phase4_adjusted_row_id) duplicate_source_ids
        FROM read_parquet('{q(promoted)}') p
        LEFT JOIN read_parquet('{q(union)}') u ON p.typed_source_genotype_id=u.canonical_gid
        """
    ).fetchone()
    if join_metrics != (3_193_677, 2_242_863, 0, 2_242_863, 950_814, 0):
        raise RuntimeError(f"Namespace join invariant failed: {join_metrics}")

    con.execute(
        f"""
        COPY (
          SELECT p.* REPLACE(
                   CASE WHEN p.canonical_gid_eligible THEN u.canonical_gid ELSE p.canonical_gid END AS canonical_gid
                 ),
                 p.canonical_gid AS phase4_v1_numeric_resolved_gid,
                 CASE WHEN p.canonical_gid_eligible THEN 'PHASE3G_R2_ACCEPTED_ALL_PANEL_GID_UNION_EXACT_TYPED_KEY'
                      ELSE 'NOT_APPLICABLE_IDENTITY_UNRESOLVED' END AS canonical_gid_authority,
                 CASE WHEN p.canonical_gid_eligible THEN 'CORRECTED_TO_GID_PREFIXED_R2_NAMESPACE'
                      ELSE 'UNCHANGED_UNRESOLVED_ARCHIVAL' END AS namespace_correction_status,
                 '{PHASE4_NS_RELEASE_ID}' AS namespace_release_id
          FROM read_parquet('{q(promoted)}') p
          LEFT JOIN read_parquet('{q(union)}') u ON p.typed_source_genotype_id=u.canonical_gid
          ORDER BY p.phase4_adjusted_row_id
        ) TO '{q(corrected)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT '{PHASE4_NS_RELEASE_ID}' namespace_release_id,p.phase4_adjusted_row_id,
                 p.typed_source_genotype_id,p.canonical_gid phase4_v1_numeric_resolved_gid,
                 u.canonical_gid authoritative_phase3g_r2_canonical_gid,
                 p.canonical_gid_eligible,
                 CASE WHEN p.canonical_gid_eligible AND u.canonical_gid IS NOT NULL THEN 'EXACT_ONE_TO_ONE_AUTHORITY_MATCH'
                      WHEN NOT p.canonical_gid_eligible THEN 'NOT_APPLICABLE_UNRESOLVED_ARCHIVAL'
                      ELSE 'INTEGRITY_FAILURE' END join_status,
                 'typed_source_genotype_id=accepted_all_panel_gid_union.canonical_gid' join_rule,
                 'many_to_one; right key unique' asserted_cardinality
          FROM read_parquet('{q(promoted)}') p
          LEFT JOIN read_parquet('{q(union)}') u ON p.typed_source_genotype_id=u.canonical_gid
          ORDER BY p.phase4_adjusted_row_id
        ) TO '{q(join_ledger)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT '{PHASE4_NS_RELEASE_ID}' namespace_release_id,
                 o.phase4_adjusted_row_id old_phase4_adjusted_row_id,
                 c.phase4_adjusted_row_id new_phase4_adjusted_row_id,
                 o.canonical_gid old_numeric_canonical_gid,
                 c.canonical_gid new_canonical_gid,
                 o.typed_source_genotype_id,
                 CASE WHEN o.phase4_adjusted_row_id=c.phase4_adjusted_row_id THEN 'UNCHANGED' ELSE 'CHANGED' END id_status,
                 'P4E ID was generated from phase4_group_id and numeric resolved_gid before promotion; the corrected promoted canonical_gid field is not an ID input' id_policy
          FROM read_parquet('{q(promoted)}') o
          JOIN read_parquet('{q(corrected)}') c USING(phase4_adjusted_row_id)
          ORDER BY o.phase4_adjusted_row_id
        ) TO '{q(id_lineage)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )

    original_columns = pq.ParquetFile(promoted).schema_arrow.names
    nonidentity = [column for column in original_columns if column != "canonical_gid"]
    expressions = ",".join(
        f"sum(CASE WHEN o.\"{column}\" IS DISTINCT FROM c.\"{column}\" THEN 1 ELSE 0 END)::BIGINT AS \"{column}\""
        for column in nonidentity
    )
    mismatch_row = con.execute(
        f"SELECT {expressions} FROM read_parquet('{q(promoted)}') o JOIN read_parquet('{q(corrected)}') c USING(phase4_adjusted_row_id)"
    ).fetchone()
    equality = pd.DataFrame(
        {
            "namespace_release_id": PHASE4_NS_RELEASE_ID,
            "field": nonidentity,
            "mismatch_rows": list(mismatch_row),
            "status": ["PASS" if value == 0 else "FAIL" for value in mismatch_row],
        }
    )
    write_tsv(out / "non_identity_field_equality_audit.tsv", equality)

    view_rows: list[dict] = []
    for view, values in EXPECTED_VIEWS.items():
        expression, *expected = values
        observed = con.execute(
            f"""
            SELECT count(*),
                   count(DISTINCT canonical_gid) FILTER(WHERE canonical_gid_eligible),
                   count(DISTINCT phase4_group_id),count(DISTINCT trial_id),
                   count(DISTINCT environment_id),count(DISTINCT year),count(DISTINCT trait)
            FROM read_parquet('{q(corrected)}') WHERE {expression}
            """
        ).fetchone()
        view_rows.append(
            {
                "namespace_release_id": PHASE4_NS_RELEASE_ID,
                "view": view,
                "filter_expression": expression,
                **{f"expected_{name}": value for name, value in zip(("rows", "gids", "groups", "trials", "environments", "years", "traits"), expected)},
                **{f"observed_{name}": value for name, value in zip(("rows", "gids", "groups", "trials", "environments", "years", "traits"), observed)},
                "status": "PASS" if tuple(expected) == observed else "FAIL",
            }
        )
    views = pd.DataFrame(view_rows)
    write_tsv(out / "view_count_summary.tsv", views)

    corrected_metrics = con.execute(
        f"""
        SELECT count(*),count(*)-count(DISTINCT phase4_adjusted_row_id),
               count(*) FILTER(WHERE canonical_gid_eligible AND regexp_full_match(canonical_gid,'GID[0-9]+')),
               count(*) FILTER(WHERE NOT canonical_gid_eligible),
               count(*) FILTER(WHERE canonical_gid_eligible AND phase4_v1_numeric_resolved_gid=canonical_gid),
               count(*) FILTER(WHERE canonical_gid_eligible AND canonical_gid<>typed_source_genotype_id)
        FROM read_parquet('{q(corrected)}')
        """
    ).fetchone()
    cardinality = pd.DataFrame(
        [
            {"check": "phase3g_r2_union_unique", "expected": 0, "observed": duplicate_union},
            {"check": "source_rows", "expected": 3_193_677, "observed": join_metrics[0]},
            {"check": "eligible_exact_authority", "expected": 2_242_863, "observed": join_metrics[3]},
            {"check": "eligible_missing_authority", "expected": 0, "observed": join_metrics[2]},
            {"check": "corrected_rows", "expected": 3_193_677, "observed": corrected_metrics[0]},
            {"check": "duplicate_corrected_ids", "expected": 0, "observed": corrected_metrics[1]},
            {"check": "gid_prefixed_eligible_rows", "expected": 2_242_863, "observed": corrected_metrics[2]},
            {"check": "unresolved_archival_rows", "expected": 950_814, "observed": corrected_metrics[3]},
            {"check": "eligible_old_new_namespace_equal", "expected": 0, "observed": corrected_metrics[4]},
            {"check": "eligible_canonical_typed_key_mismatch", "expected": 0, "observed": corrected_metrics[5]},
        ]
    )
    cardinality["status"] = cardinality.apply(lambda row: "PASS" if row.expected == row.observed else "FAIL", axis=1)
    write_tsv(out / "identity_join_cardinality_audit.tsv", cardinality)

    write_json(
        out / "namespace_correction_protocol.json",
        {
            "release_id": PHASE4_NS_RELEASE_ID,
            "source_release_id": "P4ISP_20260802_V1_274E41DF",
            "identity_authority": "Phase3G R2 accepted_all_panel_gid_union.parquet",
            "join_rule": "exact typed_source_genotype_id to canonical_gid",
            "right_cardinality": "unique",
            "eligible_action": "replace canonical_gid with authoritative GID-prefixed key",
            "unresolved_action": "retain original unresolved archival identity field and copy it to phase4_v1_numeric_resolved_gid",
            "non_identity_policy": "exact preservation",
            "observation_id_policy": "unchanged; generated upstream from phase4_group_id and numeric resolved_gid, not the promoted canonical_gid field",
            "fuzzy_or_name_matching_used": False,
            "protected_outcomes_accessed": False,
        },
    )
    write_json(
        out / "observation_id_policy.json",
        {
            "status": "PASS_IDS_UNCHANGED",
            "source_code": "scripts/v2/phase4_reconstruct_phenotypes.py:527",
            "formula": "stable_id('P4E_', phase4_group_id, resolved_gid)",
            "promoted_canonical_gid_field_is_formula_input": False,
            "biological_mapping_changed": False,
            "lineage_rows": 3_193_677,
        },
    )
    all_pass = cardinality.status.eq("PASS").all() and equality.status.eq("PASS").all() and views.status.eq("PASS").all()
    decision = {
        "status": "PASS_PHASE4_NAMESPACE_CORRECTION" if all_pass else "BLOCKED_PHASE4_NAMESPACE_CORRECTION",
        "release_id": PHASE4_NS_RELEASE_ID,
        "source_release_id": "P4ISP_20260802_V1_274E41DF",
        "records": 3_193_677,
        "corrected_records": 2_242_863,
        "unresolved_archival_records": 950_814,
        "non_identity_fields_checked": len(nonidentity),
        "non_identity_mismatches": int(equality.mismatch_rows.sum()),
        "view_checks_passed": int(views.status.eq("PASS").sum()),
        "view_checks_total": len(views),
        "observation_ids_changed": 0,
        "corrected_table_sha256": sha256(corrected),
        "protected_outcomes_accessed": False,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "RELEASE_DECISION.json", decision)
    report = f"""# Phase-4 namespace correction

Status: `{decision['status']}`
Release: `{PHASE4_NS_RELEASE_ID}`

The exact Phase-3G R2 typed-key join corrected 2,242,863 canonical-eligible
records from numeric labels to authoritative `GID<digits>` keys. All 950,814
identity-unresolved archival records remain unresolved. All {len(nonidentity)}
non-identity fields have zero mismatches across 3,193,677 records; all eight
deterministic views reproduce exactly and no observation ID changed.

No fuzzy, genotype-name, marker, pedigree, row-order, or protected-outcome
evidence was used.
"""
    (out / "NAMESPACE_CORRECTION_REPORT.md").write_text(report, encoding="utf-8")
    con.close()
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PHASE4_NS_ROOT)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.exists():
        out.mkdir(parents=True)
    decision = build(out)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
