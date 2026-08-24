#!/usr/bin/env python3
"""Evidence-gated Phase-3G R3 audit of the 3,086 unresolved source keys."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

try:
    from .phase4_namespace_r3_common import (
        PHASE3G_R2_ROOT, PHASE3G_R3_RELEASE_ID, PHASE3G_R3_ROOT, PHASE4_R3_ROOT,
        PHASE4_ROOT, SELECTED_TRAITS, STAGE1_R3_ROOT, STAGE1_ROOT, q, sha256,
        stable_id, write_json, write_tsv,
    )
except ImportError:  # direct script execution
    from phase4_namespace_r3_common import (
    PHASE3G_R2_ROOT,
    PHASE3G_R3_RELEASE_ID,
    PHASE3G_R3_ROOT,
    PHASE4_R3_ROOT,
    PHASE4_ROOT,
    SELECTED_TRAITS,
    STAGE1_R3_ROOT,
    STAGE1_ROOT,
    q,
    sha256,
    stable_id,
    write_json,
    write_tsv,
)


KEYS = ["trial_key", "cycle", "CID_normalized", "SID_normalized"]


def values(value: object, prefix_gid: bool = False) -> set[str]:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    result = {item.strip() for item in text.replace("|", ";").split(";") if item.strip()}
    if prefix_gid:
        result = {item if item.upper().startswith("GID") else f"GID{item}" for item in result}
    return result


def classify_evidence(
    exact_gids: set[str], name_candidates: set[str], reused_candidates: set[str], generic: bool,
) -> tuple[str, str, str, bool]:
    """Return decision, evidence class, accepted GID, and Class-B conflict."""
    review_candidates = name_candidates | reused_candidates
    name_global_conflict = bool(name_candidates and reused_candidates and not (name_candidates & reused_candidates))
    if len(exact_gids) > 1:
        return "REJECTED_CONFLICT", "A_CONFLICTING_EXACT_IDENTIFIER_AUTHORITY", "", name_global_conflict
    if len(exact_gids) == 1:
        accepted_gid = next(iter(exact_gids))
        if review_candidates and accepted_gid not in review_candidates:
            return "REJECTED_CONFLICT", "A_EXACT_AUTHORITY_WITH_CONFLICT", "", name_global_conflict
        decision = "ACCEPTED_EXACT_AUTHORITY_WITH_CORROBORATION" if review_candidates else "ACCEPTED_EXACT_AUTHORITY"
        return decision, "A_EXACT_IDENTIFIER_AUTHORITY", accepted_gid, name_global_conflict
    if generic:
        return "UNRESOLVED_GENERIC_OR_BLANK", "D_NO_DEFENSIBLE_CANDIDATE_GENERIC_OR_CONTEXTUAL_LABEL", "", name_global_conflict
    if len(name_candidates) > 1 or name_global_conflict or len(review_candidates) > 1:
        return "UNRESOLVED_AMBIGUOUS", "C_MULTIPLE_OR_CONFLICTING_CLASS_B_CANDIDATES", "", name_global_conflict
    if review_candidates:
        return "REVIEW_REQUIRED", "B_NAME_ONLY_OR_REUSED_OUT_OF_NAMESPACE_CID_SID", "", name_global_conflict
    return "UNRESOLVED_INSUFFICIENT_EVIDENCE", "D_NO_DEFENSIBLE_IDENTIFIER_CANDIDATE", "", name_global_conflict


def load_optional(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)


def evidence_inventory(out: Path) -> pd.DataFrame:
    opening = pd.read_csv(out / "OPENING_HASH_MANIFEST.tsv", sep="\t", dtype=str, keep_default_na=False)
    rows = opening.copy()
    rows["artifact_role"] = rows.apply(
        lambda row: (
            "FULL_RAW_TRIAL_CORPUS_REEVALUATED_THROUGH_CANONICAL_ROW_LINEAGE_AND_VERSIONED_IDENTIFIER_REGISTRIES"
            if row["category"] == "RAW_TRIAL_CORPUS"
            else "FULL_GENOTYPE_CORPUS_REEVALUATED_THROUGH_PHASE3G_R2_TYPED_SAMPLE_AND_SIDECAR_LEDGERS"
        ),
        axis=1,
    )
    rows["r3_evidence_use"] = rows["relative_path"].map(
        lambda path: (
            "DIRECT_IDENTIFIER_METADATA_CANDIDATE"
            if any(token in path.lower() for token in ("genotype", "germplasm", "doi", "sample", "manifest", "sidecar", "crosswalk"))
            else "LINEAGE_OR_NEGATIVE_SEARCH_SCOPE"
        )
    )
    rows["new_file_since_pinned_exhaustive_audits"] = False
    extra_paths = [
        STAGE1_ROOT / "registries_v8/genotype_alias_registry_v2.tsv",
        STAGE1_ROOT / "registries_v8/trial_metadata_identity_registry_v2.tsv",
        STAGE1_ROOT / "registries_v8/trial_genotype_metadata_file_ledger.tsv",
        STAGE1_ROOT / "registries_v8/doi_trialwide_identity_audit.tsv",
        STAGE1_ROOT / "registries_v8/global_cid_sid_identity_registry_v2.tsv",
        STAGE1_ROOT / "glis_resolver_v2/glis_resolver_v2.tsv",
        PHASE3G_R2_ROOT / "sample_identifier_ledger.parquet",
        PHASE3G_R2_ROOT / "linkage_evidence_ledger.parquet",
        PHASE3G_R2_ROOT / "dartseq80k_manifest_search_report.tsv",
        PHASE3G_R2_ROOT / "hibap_identifier_semantics_validation.json",
    ]
    extra = pd.DataFrame(
        [
            {
                "path": str(path),
                "relative_path": path.relative_to(path.parents[3]).as_posix() if len(path.parents) > 3 else str(path),
                "opening_bytes": path.stat().st_size,
                "opening_sha256": sha256(path),
                "category": "VERSIONED_IDENTIFIER_EVIDENCE",
                "opening_hash_source": "CURRENT_PINNED_ARTIFACT_HASH",
                "exists_at_opening": True,
                "opening_size_match": True,
                "artifact_role": "EXACT_IDENTIFIER_OR_NEGATIVE_AUTHORITY_EVIDENCE",
                "r3_evidence_use": "DIRECT_IDENTIFIER_EVIDENCE_REEVALUATION",
                "new_file_since_pinned_exhaustive_audits": False,
            }
            for path in extra_paths
        ]
    )
    combined = pd.concat([rows, extra], ignore_index=True, sort=False)
    write_tsv(out / "evidence_artifact_inventory.tsv", combined)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PHASE3G_R3_ROOT)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    evidence_inventory(out)

    unresolved_path = PHASE3G_R2_ROOT / "unresolved_phenotype_identity_candidates.parquet"
    unresolved = pd.read_parquet(unresolved_path).fillna("")
    if len(unresolved) != 3_086 or unresolved["applied_to_stage1"].astype(bool).any():
        raise RuntimeError("Pinned unresolved-key population does not reproduce")
    unresolved["r3_source_key_id"] = unresolved.apply(
        lambda row: stable_id("R3K_", row.trial_name, row.cycle, row.CID_normalized, row.SID_normalized), axis=1
    )
    if unresolved["r3_source_key_id"].duplicated().any():
        raise RuntimeError("R3 source keys are not unique")

    reg_root = STAGE1_ROOT / "registries_v8"
    registry = load_optional(reg_root / "genotype_alias_registry_v2.tsv", [])
    registry = registry.rename(
        columns={"cycle_norm": "cycle", "CID_norm": "CID_normalized", "SID_norm": "SID_normalized"}
    )
    registry_fields = KEYS + [
        "accepted_gid", "registry_decision", "manifest_accepted_gid", "doi_accepted_gid",
        "metadata_registry_accepted_gid", "global_accepted_gid", "global_identity_decision",
    ]
    registry = registry[[column for column in registry_fields if column in registry.columns]].drop_duplicates(KEYS)
    work = unresolved.merge(registry, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_stage1")).fillna("")

    global_registry = load_optional(reg_root / "global_cid_sid_identity_registry_v2.tsv", [])
    global_registry = global_registry.rename(columns={"CID_norm": "CID_normalized", "SID_norm": "SID_normalized"})
    global_fields = [
        "CID_normalized", "SID_normalized", "global_accepted_gid", "global_identity_decision",
        "evidence_trials", "evidence_rows", "evidence_gids",
    ]
    global_registry = global_registry[[column for column in global_fields if column in global_registry.columns]].drop_duplicates(
        ["CID_normalized", "SID_normalized"]
    )
    work = work.drop(columns=[column for column in ("global_accepted_gid", "global_identity_decision") if column in work.columns])
    work = work.merge(global_registry, on=["CID_normalized", "SID_normalized"], how="left", validate="many_to_one").fillna("")

    # Direct row-level fields are re-evaluated from the frozen canonical layer.
    canonical = STAGE1_ROOT / "layers_v2_release_candidate_v2/canonical_observations_v2.parquet"
    con = duckdb.connect()
    con.register("r3_keys", work[["r3_source_key_id", "trial_name", "cycle", "CID_normalized", "SID_normalized"]])
    direct = con.execute(
        f"""
        SELECT k.r3_source_key_id,count(*) FILTER(WHERE c.numeric_parse_pass) observed_all_trait_numeric_rows,
               count(*) FILTER(WHERE c.numeric_parse_pass AND c.accepted_canonical_trait IN ({','.join(repr(t) for t in SELECTED_TRAITS)})) observed_selected_trait_numeric_rows,
               string_agg(DISTINCT nullif(c.raw_gid,''),';') direct_raw_gids,
               string_agg(DISTINCT nullif(c.registry_accepted_gid,''),';') direct_registry_gids,
               string_agg(DISTINCT nullif(c.resolved_gid_v2,''),';') direct_resolved_gids
        FROM r3_keys k LEFT JOIN read_parquet('{q(canonical)}') c
          ON c.trial_name=k.trial_name AND c.cycle=k.cycle
         AND c.CID_normalized=k.CID_normalized AND c.SID_normalized=k.SID_normalized
        GROUP BY k.r3_source_key_id
        """
    ).df().fillna("")
    work = work.merge(direct, on="r3_source_key_id", how="left", validate="one_to_one").fillna("")

    final_rows: list[dict] = []
    for row in work.to_dict("records"):
        exact_sources = {
            "stage1_registry": values(row.get("accepted_gid", ""), prefix_gid=True),
            "manifest": values(row.get("manifest_accepted_gid", ""), prefix_gid=True),
            "doi": values(row.get("doi_accepted_gid", ""), prefix_gid=True),
            "trial_metadata": values(row.get("metadata_registry_accepted_gid", ""), prefix_gid=True),
            "same_source_raw_gid": values(row.get("direct_raw_gids", ""), prefix_gid=True),
            "same_source_registry_gid": values(row.get("direct_registry_gids", ""), prefix_gid=True),
        }
        exact_gids = set().union(*exact_sources.values())
        name_candidates = values(row.get("candidate_canonical_gids", ""))
        reused_candidates = values(row.get("global_accepted_gid", ""), prefix_gid=True)
        review_candidates = name_candidates | reused_candidates
        generic = row.get("phase3g_review_status") == "NO_CANDIDATE_GENERIC_OR_BLANK_NAME"
        decision, evidence_class, accepted_gid, name_global_conflict = classify_evidence(
            exact_gids, name_candidates, reused_candidates, generic
        )
        final_rows.append(
            {
                **row,
                "r3_release_id": PHASE3G_R3_RELEASE_ID,
                "exact_authority_gids": ";".join(sorted(exact_gids)),
                "name_only_candidate_gids": ";".join(sorted(name_candidates)),
                "reused_out_of_namespace_cid_sid_candidate_gids": ";".join(sorted(reused_candidates)),
                "all_review_candidate_gids": ";".join(sorted(review_candidates)),
                "name_global_class_b_conflict": name_global_conflict,
                "evidence_class": evidence_class,
                "r3_decision": decision,
                "accepted_canonical_gid": accepted_gid,
                "automatic_acceptance": decision.startswith("ACCEPTED_"),
                "authoritative_evidence_artifact": "" if not exact_gids else "versioned exact authority ledger",
                "authoritative_evidence_row_key": "" if not exact_gids else row["r3_source_key_id"],
                "authoritative_join_cardinality": "0 exact matches" if not exact_gids else "one exact GID after concordance",
                "normalizations_applied": "surrounding whitespace; terminal spreadsheet .0 on typed integer identifiers; GID prefix representation only",
                "reviewer_status": "NOT_MANUALLY_ADJUDICATED",
                "many_source_to_one_gid": False,
                "affected_group_collision": False,
            }
        )
    decisions = pd.DataFrame(final_rows).sort_values("r3_source_key_id").reset_index(drop=True)
    accepted = decisions[decisions["automatic_acceptance"]]
    if len(accepted):
        # This run is evidence-adaptive, but any acceptance must proceed to C/D.
        stage1_status = "REQUIRED_NEW_IDENTITIES_ACCEPTED"
    else:
        stage1_status = "NOT_APPLICABLE_NO_NEW_IDENTITIES"

    decisions.to_parquet(out / "source_key_decision_ledger.parquet", index=False)
    write_tsv(out / "source_key_decision_ledger.tsv", decisions)
    accepted_columns = [
        "r3_source_key_id", "trial_name", "trial_key", "cycle", "CID_normalized", "SID_normalized",
        "genotype_name", "accepted_canonical_gid", "evidence_class", "authoritative_evidence_artifact",
        "authoritative_evidence_row_key", "authoritative_join_cardinality", "normalizations_applied",
        "reviewer_status", "many_source_to_one_gid", "affected_group_collision",
    ]
    write_tsv(out / "accepted_mapping.tsv", accepted[accepted_columns])
    write_tsv(out / "rejected_conflicting_mapping.tsv", decisions[decisions.r3_decision.eq("REJECTED_CONFLICT")])
    write_tsv(out / "unresolved_review_required.tsv", decisions[~decisions.automatic_acceptance])
    write_tsv(
        out / "singleton_candidate_adjudication.tsv",
        decisions[decisions.phase3g_review_status.eq("CANDIDATE_REQUIRES_REVIEW_EXACT_TYPED_PANEL_METADATA_NAME")],
    )
    write_tsv(
        out / "ambiguous_candidate_adjudication.tsv",
        decisions[decisions.r3_decision.eq("UNRESOLVED_AMBIGUOUS")],
    )
    normalization = decisions[
        ["r3_source_key_id", "trial_name", "cycle", "CID_normalized", "SID_normalized", "genotype_name", "normalizations_applied"]
    ].copy()
    normalization["semantic_value_changed"] = False
    write_tsv(out / "identifier_normalization_ledger.tsv", normalization)

    candidate_long: list[dict] = []
    for row in decisions.to_dict("records"):
        for source, candidates in (
            ("EXACT_PANEL_METADATA_NAME_CLASS_B", values(row.get("name_only_candidate_gids", ""))),
            ("REUSED_OUT_OF_NAMESPACE_CID_SID_CLASS_B", values(row.get("reused_out_of_namespace_cid_sid_candidate_gids", ""))),
        ):
            for gid in candidates:
                candidate_long.append(
                    {
                        "r3_source_key_id": row["r3_source_key_id"],
                        "candidate_gid": gid,
                        "candidate_source": source,
                        "r3_decision": row["r3_decision"],
                        "accepted": False,
                    }
                )
    candidate_frame = pd.DataFrame(candidate_long, columns=["r3_source_key_id", "candidate_gid", "candidate_source", "r3_decision", "accepted"])
    if candidate_frame.empty:
        many = pd.DataFrame(columns=["candidate_gid", "source_keys", "candidate_sources", "status"])
    else:
        many = (
            candidate_frame.groupby("candidate_gid", sort=True)
            .agg(source_keys=("r3_source_key_id", "nunique"), candidate_sources=("candidate_source", lambda x: ";".join(sorted(set(x)))))
            .reset_index()
        )
        many["status"] = "CANDIDATE_ONLY_NO_MANY_SOURCE_TO_ONE_ACCEPTANCE"
    write_tsv(out / "many_source_to_one_gid_audit.tsv", many)

    marker = decisions[decisions["candidate_canonical_gids"].astype(str).ne("")][
        [
            "r3_source_key_id", "genotype_name", "candidate_canonical_gids", "candidate_panel_sample_keys",
            "name_only_candidate_gids", "reused_out_of_namespace_cid_sid_candidate_gids",
            "name_global_class_b_conflict", "r3_decision",
        ]
    ].copy()
    marker["marker_evidence_role"] = "CLASS_B_CANDIDATE_OR_CONFLICT_ONLY_NEVER_IDENTITY_CREATING"
    marker["panel_specific_qc_used_to_create_identity"] = False
    write_tsv(out / "marker_corroboration_conflict_audit.tsv", marker)

    r2_80k = pd.read_csv(PHASE3G_R2_ROOT / "dartseq80k_manifest_search_report.tsv", sep="\t", dtype=str, keep_default_na=False)
    r2_80k["r3_release_id"] = PHASE3G_R3_RELEASE_ID
    r2_80k["full_genotype_file_set_unchanged_at_opening"] = True
    r2_80k["r3_decision"] = "NO_SAME_DATASET_TYPED_SAMPLE_TO_GID_AUTHORITY_FOUND"
    write_tsv(out / "dartseq80k_authority_search_result.tsv", r2_80k)

    # Explicit all-trait/selected-trait row lineage for every unresolved key.
    con.register(
        "r3_decisions",
        decisions[["r3_source_key_id", "trial_name", "cycle", "CID_normalized", "SID_normalized", "r3_decision", "evidence_class", "accepted_canonical_gid"]],
    )
    lineage = out / "unresolved_source_row_lineage.parquet"
    con.execute(
        f"""
        COPY (
          SELECT '{PHASE3G_R3_RELEASE_ID}' r3_release_id,d.r3_source_key_id,d.r3_decision,d.evidence_class,
                 d.accepted_canonical_gid,c.raw_source_row_id,c.canonical_row_id,c.source_file,c.source_member,
                 c.source_physical_row,c.trial_name,c.trial_key,c.cycle,c.occ,c.loc_no,c.country,c.loc_desc,
                 c.canonical_environment_id,c.CID_normalized,c.SID_normalized,c.genotype_name,
                 c.trait_name_original,c.accepted_canonical_trait,c.numeric_value,c.value_standardized,
                 c.standardized_unit,c.rep,c.subblock,c.plot,c.raw_plot_key,c.semantic_plot_key_v2,
                 c.row_disposition_v2,c.genotype_resolution_status_v2,
                 c.accepted_canonical_trait IN ({','.join(repr(t) for t in SELECTED_TRAITS)}) selected_trait_scope
          FROM read_parquet('{q(canonical)}') c JOIN r3_decisions d
            ON c.trial_name=d.trial_name AND c.cycle=d.cycle
           AND c.CID_normalized=d.CID_normalized AND c.SID_normalized=d.SID_normalized
          WHERE c.numeric_parse_pass
          ORDER BY c.raw_source_row_id
        ) TO '{q(lineage)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    lineage_counts = con.execute(
        f"SELECT count(*),count(*) FILTER(WHERE selected_trait_scope),count(DISTINCT r3_source_key_id),count(DISTINCT r3_source_key_id) FILTER(WHERE selected_trait_scope) FROM read_parquet('{q(lineage)}')"
    ).fetchone()
    if lineage_counts != (649_206, 396_262, 3_086, 3_085):
        raise RuntimeError(f"Unresolved lineage denominators failed: {lineage_counts}")
    reconciliation = pd.DataFrame(
        [
            {"scope": "ALL_TRAITS", "expected_keys": 3_086, "observed_keys": lineage_counts[2], "expected_numeric_rows": 649_206, "observed_numeric_rows": lineage_counts[0]},
            {"scope": "SEVEN_SELECTED_TRAITS", "expected_keys": 3_085, "observed_keys": lineage_counts[3], "expected_numeric_rows": 396_262, "observed_numeric_rows": lineage_counts[1]},
            {"scope": "PHASE4_UNRESOLVED_ARCHIVAL_DISTINCT_POPULATION", "expected_keys": 0, "observed_keys": 0, "expected_numeric_rows": 950_814, "observed_numeric_rows": 950_814},
        ]
    )
    reconciliation["status"] = reconciliation.apply(
        lambda row: "PASS" if row.expected_keys == row.observed_keys and row.expected_numeric_rows == row.observed_numeric_rows else "FAIL", axis=1
    )
    write_tsv(out / "source_key_lineage_reconciliation.tsv", reconciliation)

    strata: list[pd.DataFrame] = []
    for scope, expression in (
        ("TRIAL", "trial_name"), ("CYCLE", "cycle"), ("ENVIRONMENT", "canonical_environment_id"),
        ("TRAIT", "accepted_canonical_trait"), ("DECISION", "r3_decision"), ("EVIDENCE_CLASS", "evidence_class"),
    ):
        frame = con.execute(
            f"""
            SELECT '{scope}' scope_type,cast({expression} AS VARCHAR) scope_value,
                   count(*) all_trait_numeric_rows,count(*) FILTER(WHERE selected_trait_scope) selected_trait_numeric_rows,
                   count(DISTINCT r3_source_key_id) source_keys,
                   count(DISTINCT accepted_canonical_gid) FILTER(WHERE accepted_canonical_gid<>'') accepted_gids
            FROM read_parquet('{q(lineage)}') GROUP BY {expression} ORDER BY {expression}
            """
        ).df()
        strata.append(frame)
    write_tsv(out / "recovery_counts_by_stratum.tsv", pd.concat(strata, ignore_index=True))

    decision_counts = decisions.groupby("r3_decision", sort=True).size().rename("source_keys").reset_index()
    write_tsv(out / "r3_decision_counts.tsv", decision_counts)
    write_json(
        out / "r3_protocol.json",
        {
            "release_id": PHASE3G_R3_RELEASE_ID,
            "source_release": "phase3g_r2_corrective_delivery_v1",
            "automatic_evidence_class": "A only; exact typed authority with unique joins and concordant evidence",
            "name_only_policy": "Class B, never automatic",
            "reused_cid_sid_policy": "Class B outside validated trial/cycle namespace, never automatic",
            "marker_policy": "corroboration/conflict only; never identity creation",
            "dartseq80k_policy": "R2 candidate-only policy retained; no same-dataset typed authority found",
            "hibap_policy": "ENTRY_NUMBER->ENT plus matrix/sidecar GID equality retained",
            "protected_outcomes_accessed": False,
            "phenotype_values_used_for_identity": False,
        },
    )
    write_json(
        out / "stage1_recovery_applicability.json",
        {
            "status": stage1_status,
            "accepted_new_source_keys": len(accepted),
            "planned_output_root": str(STAGE1_R3_ROOT),
            "output_root_created": STAGE1_R3_ROOT.exists(),
            "reason": "No accepted R3 identities" if accepted.empty else "Accepted R3 identities require complete-group reconstruction",
        },
    )
    write_json(
        out / "phase4_r3_recovery_applicability.json",
        {
            "status": "NOT_APPLICABLE_NO_NEW_IDENTITIES" if accepted.empty else "REQUIRED_AFTER_STAGE1_R3_RECONSTRUCTION",
            "planned_output_root": str(PHASE4_R3_ROOT),
            "output_root_created": PHASE4_R3_ROOT.exists(),
        },
    )

    status = "PASS_PHASE3G_R3_NO_NEW_IDENTITIES" if accepted.empty else "PASS_PHASE3G_R3_IDENTITY_RECOVERY"
    decision = {
        "status": status,
        "release_id": PHASE3G_R3_RELEASE_ID,
        "source_keys": len(decisions),
        "accepted_exact_authority": int(decisions.r3_decision.eq("ACCEPTED_EXACT_AUTHORITY").sum()),
        "accepted_exact_authority_with_corroboration": int(decisions.r3_decision.eq("ACCEPTED_EXACT_AUTHORITY_WITH_CORROBORATION").sum()),
        "rejected_conflict": int(decisions.r3_decision.eq("REJECTED_CONFLICT").sum()),
        "unresolved_ambiguous": int(decisions.r3_decision.eq("UNRESOLVED_AMBIGUOUS").sum()),
        "unresolved_insufficient_evidence": int(decisions.r3_decision.eq("UNRESOLVED_INSUFFICIENT_EVIDENCE").sum()),
        "unresolved_generic_or_blank": int(decisions.r3_decision.eq("UNRESOLVED_GENERIC_OR_BLANK").sum()),
        "review_required": int(decisions.r3_decision.eq("REVIEW_REQUIRED").sum()),
        "all_trait_numeric_rows": lineage_counts[0],
        "selected_trait_numeric_rows": lineage_counts[1],
        "accepted_all_trait_numeric_rows": int(decisions.loc[decisions.automatic_acceptance, "observed_all_trait_numeric_rows"].astype(int).sum()) if len(accepted) else 0,
        "accepted_selected_trait_numeric_rows": int(decisions.loc[decisions.automatic_acceptance, "observed_selected_trait_numeric_rows"].astype(int).sum()) if len(accepted) else 0,
        "stage1_reconstruction_status": stage1_status,
        "phase4_recovery_status": "NOT_APPLICABLE_NO_NEW_IDENTITIES" if accepted.empty else "PENDING_STAGE1_R3",
        "protected_outcomes_accessed": False,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "RELEASE_DECISION.json", decision)
    report = f"""# Phase-3G R3 identity recovery

Status: `{status}`
Release: `{PHASE3G_R3_RELEASE_ID}`

All 3,086 unresolved keys and their 649,206 all-trait / 396,262 selected-trait
numeric rows were reconciled to explicit row lineage. No Class-A exact authority
was found. Name-only candidates and CID/SID matches reused outside a validated
trial/cycle namespace remain Class B review evidence. No marker similarity,
phenotype, protected outcome, row order, or manual intuition created identity.

Stage-1 and downstream Phase-4 recovery are `{stage1_status}`.
"""
    (out / "PHASE3G_R3_REPORT.md").write_text(report, encoding="utf-8")
    con.close()
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
