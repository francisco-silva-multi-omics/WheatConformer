from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


TERMINAL_CLASSES = (
    "direct_observed_ready",
    "direct_sparse_candidate",
    "pedigree_imputable",
    "candidate_unresolved",
    "ineligible",
)


def as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "pass", "retained"}
    )


def joined(values: Iterable[object]) -> str:
    return ";".join(sorted({str(value).strip() for value in values if str(value).strip()}))


def integer_or_zero(value: object) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    return 0 if pd.isna(numeric) else int(numeric)


def observation_counts(observations: pd.DataFrame) -> pd.DataFrame:
    required = {
        "canonical_gid",
        "primary_weighted_training_eligible",
        "secondary_unweighted_training_eligible",
    }
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"Stage-1 v2 observation index lacks columns: {missing}")
    local = observations[list(required)].copy()
    local["canonical_gid"] = local["canonical_gid"].fillna("").astype(str).str.strip()
    local = local[local["canonical_gid"].ne("")]
    local["primary"] = as_bool(local["primary_weighted_training_eligible"])
    local["secondary"] = as_bool(local["secondary_unweighted_training_eligible"])
    return (
        local.groupby("canonical_gid", sort=True)
        .agg(
            selected_trait_rows=("canonical_gid", "size"),
            primary_weighted_rows=("primary", "sum"),
            secondary_unweighted_rows=("secondary", "sum"),
        )
        .reset_index()
    )


def panel_readiness(
    source_axis: pd.DataFrame,
    overlap: pd.DataFrame,
    *,
    cimmyt_qc_gids: int,
    seeds_qc_gids: int,
) -> pd.DataFrame:
    required = {
        "panel_id",
        "raw_marker_availability",
        "accepted_canonical_gid_count",
        "v2_terminal_disposition",
        "allele_encoding",
        "identity_authority",
    }
    missing = sorted(required.difference(source_axis.columns))
    if missing:
        raise ValueError(f"Panel source-axis audit lacks columns: {missing}")
    overlap_counts = (
        overlap.set_index("panel_id")["primary_stage1_gids"]
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
        .to_dict()
    )
    rows: list[dict[str, object]] = []
    for source in source_axis.fillna("").to_dict("records"):
        panel = str(source["panel_id"])
        raw = str(source["raw_marker_availability"])
        disposition = str(source["v2_terminal_disposition"])
        stage1_gids = int(overlap_counts.get(panel, 0))
        raw_observed = raw.startswith("RAW_") and "IDENTITY_NOT_AUTHORIZED" not in raw
        raw_observed &= "HAPLOTYPE_CALLS" not in raw
        if panel == "cimmyt_bread_gbs_2013_2018":
            raw_observed = True
            observed_qc_gids = int(cimmyt_qc_gids)
            raw_status = "AVAILABLE_RETAINED_GLOBAL_UNIVERSE_WITH_MISSING_MASK"
            coordinate_status = "CHROMOSOME_POSITION_PRESENT_ASSEMBLY_UNVERIFIED"
            allele_status = "PASS_SOURCE_ALLELES"
        elif panel == "seeds_of_discovery_dartseq":
            observed_qc_gids = int(seeds_qc_gids)
            raw_status = "AVAILABLE_QC_CONSENSUS_OBSERVED_CALLS"
            coordinate_status = "NOT_CERTIFIED_MARKER_TAG_ONLY"
            allele_status = "PASS_DECLARED_REF_ALT"
        else:
            observed_qc_gids = stage1_gids if raw_observed else 0
            raw_status = "AVAILABLE_CERTIFIED_SOURCE_CALLS" if raw_observed else "NOT_DIRECT_OBSERVED_CALLS"
            coordinate_status = "NOT_CERTIFIED"
            allele_text = str(source.get("allele_encoding", ""))
            allele_status = (
                "SOURCE_DECLARED_NOT_REFERENCE_HARMONIZED"
                if raw_observed and allele_text not in {"", "NOT_APPLICABLE"}
                else "NOT_CERTIFIED"
            )
        identity_status = "ACCEPTED_SAME_DATASET" if stage1_gids else "NO_ACCEPTED_STAGE1_IDENTITY"
        if "IDENTITY_NOT_AUTHORIZED" in raw or "NO_SAME_DATASET_TYPED_IDENTITY" in disposition:
            identity_status = "CANDIDATE_ONLY_NOT_AUTHORIZED"
        graph_status = "NOT_BUILT_OR_CERTIFIED_V2"
        window_status = "NOT_EVALUATED_NO_GRAPH_PROJECTION"
        embedding_status = "NOT_BUILT_OR_CERTIFIED_V2"
        if raw_observed and observed_qc_gids > 0:
            kz_class = "direct_sparse_candidate"
            blocker = "REFERENCE_GRAPH_PROJECTION_AND_REGULATORY_WINDOW_OVERLAP_NOT_CERTIFIED"
        elif identity_status == "CANDIDATE_ONLY_NOT_AUTHORIZED" or stage1_gids > 0:
            kz_class = "candidate_unresolved"
            blocker = disposition or raw
        else:
            kz_class = "ineligible"
            blocker = disposition or "NO_STAGE1_V2_OVERLAP"
        rows.append(
            {
                "panel_id": panel,
                "accepted_panel_gids": integer_or_zero(
                    source.get("accepted_canonical_gid_count", 0)
                ),
                "stage1_v2_gids": stage1_gids,
                "observed_call_qc_gids": observed_qc_gids,
                "identity_status": identity_status,
                "raw_observed_call_status": raw_status,
                "marker_allele_status": allele_status,
                "marker_coordinate_status": coordinate_status,
                "graph_projection_status": graph_status,
                "regulatory_window_overlap_status": window_status,
                "direct_embedding_status": embedding_status,
                "panel_kz_class": kz_class,
                "panel_kz_blocker": blocker,
                "phase5_disposition": disposition,
                "identity_authority": str(source.get("identity_authority", "")),
            }
        )
    result = pd.DataFrame(rows).sort_values("panel_id").reset_index(drop=True)
    if result["panel_id"].duplicated().any():
        raise ValueError("Panel readiness contains duplicate panel IDs")
    return result


def gid_panel_evidence(
    genotype_ids: set[str],
    accepted: pd.DataFrame,
    readiness: pd.DataFrame,
    *,
    seeds_qc_ids: set[str],
    cimmyt_qc_ids: set[str],
    unresolved_80k: pd.DataFrame,
) -> pd.DataFrame:
    accepted_local = accepted[
        accepted["accepted_canonical_gid"].fillna("").astype(str).isin(genotype_ids)
    ].copy()
    accepted_local["canonical_gid"] = accepted_local["accepted_canonical_gid"].astype(str)
    accepted_agg = (
        accepted_local.groupby(["canonical_gid", "panel_id"], sort=True)
        .agg(
            accepted_sample_instances=("sample_instance_key", "nunique"),
            evidence_types=("evidence_type", joined),
            mapping_statuses=("mapping_status", joined),
        )
        .reset_index()
    )
    accepted_agg["accepted_identity"] = True
    accepted_agg["candidate_only"] = False

    unresolved = unresolved_80k[
        unresolved_80k["candidate_canonical_gid"].fillna("").astype(str).isin(genotype_ids)
    ].copy()
    unresolved["canonical_gid"] = unresolved["candidate_canonical_gid"].astype(str)
    unresolved_agg = (
        unresolved.groupby(["canonical_gid", "panel_id"], sort=True)
        .agg(
            accepted_sample_instances=("sample_instance_key", "nunique"),
            evidence_types=("evidence_type", joined),
            mapping_statuses=("mapping_status", joined),
        )
        .reset_index()
    )
    unresolved_agg["accepted_identity"] = False
    unresolved_agg["candidate_only"] = True

    evidence = pd.concat([accepted_agg, unresolved_agg], ignore_index=True)
    evidence = evidence.merge(readiness, on="panel_id", how="left", validate="many_to_one")
    if evidence["panel_kz_class"].isna().any():
        missing = sorted(evidence.loc[evidence["panel_kz_class"].isna(), "panel_id"].unique())
        raise ValueError(f"Missing panel-readiness rows: {missing}")
    evidence["passes_panel_sample_qc"] = evidence["accepted_identity"]
    seeds = evidence["panel_id"].eq("seeds_of_discovery_dartseq")
    cimmyt = evidence["panel_id"].eq("cimmyt_bread_gbs_2013_2018")
    evidence.loc[seeds, "passes_panel_sample_qc"] = evidence.loc[seeds, "canonical_gid"].isin(seeds_qc_ids)
    evidence.loc[cimmyt, "passes_panel_sample_qc"] = evidence.loc[cimmyt, "canonical_gid"].isin(cimmyt_qc_ids)
    evidence["direct_sparse_evidence"] = (
        evidence["accepted_identity"]
        & evidence["passes_panel_sample_qc"]
        & evidence["panel_kz_class"].eq("direct_sparse_candidate")
    )
    evidence["direct_observed_ready"] = False
    evidence["phase6_kz_eligible"] = False
    return evidence.sort_values(["canonical_gid", "panel_id", "candidate_only"]).reset_index(drop=True)


def terminal_class(row: pd.Series) -> str:
    if bool(row["direct_observed_ready"]):
        return "direct_observed_ready"
    if bool(row["direct_sparse_candidate"]):
        return "direct_sparse_candidate"
    if bool(row["pedigree_available"]):
        return "pedigree_imputable"
    if bool(row["has_unresolved_candidate"]):
        return "candidate_unresolved"
    return "ineligible"


def build_gid_manifest(
    genotypes: pd.DataFrame,
    counts: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    local = genotypes.copy()
    local["canonical_gid"] = local["canonical_gid"].astype(str)
    for column in ["in_primary_view", "in_secondary_view", "pedigree_available"]:
        local[column] = as_bool(local[column])
    local = local.merge(counts, on="canonical_gid", how="left", validate="one_to_one")
    for column in ["selected_trait_rows", "primary_weighted_rows", "secondary_unweighted_rows"]:
        local[column] = pd.to_numeric(local[column], errors="coerce").fillna(0).astype(int)

    aggregates: dict[str, dict[str, object]] = defaultdict(dict)
    for gid, group in evidence.groupby("canonical_gid", sort=True):
        accepted_group = group[group["accepted_identity"]]
        sparse_group = group[group["direct_sparse_evidence"]]
        unresolved_group = group[group["candidate_only"] | ~group["direct_sparse_evidence"]]
        aggregates[gid] = {
            "accepted_identity_panels": joined(accepted_group["panel_id"]),
            "raw_observed_call_panels": joined(sparse_group["panel_id"]),
            "coordinate_candidate_panels": joined(
                sparse_group.loc[
                    ~sparse_group["marker_coordinate_status"].eq("NOT_CERTIFIED"), "panel_id"
                ]
            ),
            "graph_projection_ready_panels": joined(
                group.loc[group["graph_projection_status"].eq("PASS"), "panel_id"]
            ),
            "direct_embedding_ready_panels": joined(
                group.loc[group["direct_embedding_status"].eq("PASS"), "panel_id"]
            ),
            "unresolved_candidate_panels": joined(unresolved_group["panel_id"]),
            "accepted_sample_instances": int(accepted_group["accepted_sample_instances"].sum()),
            "direct_sparse_candidate": bool(sparse_group.shape[0]),
            "direct_observed_ready": bool(group["direct_observed_ready"].any()),
            "has_unresolved_candidate": bool(unresolved_group.shape[0]),
        }
    aggregate_frame = pd.DataFrame.from_dict(aggregates, orient="index")
    aggregate_frame.index.name = "canonical_gid"
    aggregate_frame = aggregate_frame.reset_index()
    local = local.merge(aggregate_frame, on="canonical_gid", how="left", validate="one_to_one")
    text_columns = [
        "accepted_identity_panels",
        "raw_observed_call_panels",
        "coordinate_candidate_panels",
        "graph_projection_ready_panels",
        "direct_embedding_ready_panels",
        "unresolved_candidate_panels",
    ]
    for column in text_columns:
        local[column] = local[column].fillna("").astype(str)
    for column in ["direct_sparse_candidate", "direct_observed_ready", "has_unresolved_candidate"]:
        local[column] = local[column].map(lambda value: False if pd.isna(value) else bool(value))
    local["accepted_sample_instances"] = (
        pd.to_numeric(local["accepted_sample_instances"], errors="coerce").fillna(0).astype(int)
    )
    local["pedigree_imputation_candidate"] = local["pedigree_available"]
    local["pedigree_imputation_ready"] = False
    local["observed_sequence_equivalent"] = False
    local["phase6_ka_baseline_eligible"] = local["pedigree_available"]
    local["phase6_kz_eligible"] = False
    local["regulatory_terminal_class"] = local.apply(terminal_class, axis=1)
    local["confidence_provenance_class"] = local["regulatory_terminal_class"].map(
        {
            "direct_observed_ready": "observed_graph_projected",
            "direct_sparse_candidate": "observed_sparse_markers_not_graph_projected",
            "pedigree_imputable": "imputed_pedigree_candidate_not_fitted",
            "candidate_unresolved": "identity_or_variant_evidence_unresolved",
            "ineligible": "no_regulatory_genotype_evidence",
        }
    )
    local["confidence_gate_status"] = local["regulatory_terminal_class"].map(
        lambda value: "required_not_evaluated" if value == "pedigree_imputable" else "not_applicable"
    )
    local["next_action"] = local["regulatory_terminal_class"].map(
        {
            "direct_observed_ready": "extract_and_certify_regulatory_embeddings",
            "direct_sparse_candidate": "normalize_coordinates_project_graph_and_audit_window_coverage",
            "pedigree_imputable": "wait_for_direct_embedding_donors_then_fit_training_only_propagation",
            "candidate_unresolved": "resolve_identity_or_source_variant_provenance",
            "ineligible": "retain_K_A_or_non_genomic_baseline_only",
        }
    )
    return local.sort_values("genotype_index").reset_index(drop=True)


def split_support(
    states: pd.DataFrame,
    state_root: Path,
    readiness: pd.DataFrame,
    evidence: pd.DataFrame,
    pedigree_ids: set[str],
    *,
    minimum_training_gids: int,
) -> pd.DataFrame:
    direct_by_panel = {
        panel: set(group.loc[group["direct_sparse_evidence"], "canonical_gid"])
        for panel, group in evidence.groupby("panel_id", sort=True)
    }
    rows: list[dict[str, object]] = []
    for state in states.to_dict("records"):
        path = state_root / str(state["training_gid_path"])
        ids_frame = pd.read_csv(path, sep="\t", dtype=str)
        id_col = "canonical_gid" if "canonical_gid" in ids_frame.columns else ids_frame.columns[0]
        training_ids = set(ids_frame[id_col].fillna("").astype(str))
        pedigree_count = len(training_ids & pedigree_ids)
        for panel in readiness["panel_id"]:
            candidate_ids = direct_by_panel.get(str(panel), set())
            candidate_count = len(training_ids & candidate_ids)
            if not candidate_ids:
                support_status = "NO_DIRECT_SPARSE_CANDIDATES"
            elif candidate_count < minimum_training_gids:
                support_status = "MASKED_INSUFFICIENT_DIRECT_TRAINING_GIDS"
            else:
                support_status = "SUPPORT_PASS_PROJECTION_AND_EMBEDDING_BLOCKED"
            rows.append(
                {
                    "state_id": state["state_id"],
                    "scenario": state["scenario"],
                    "outer_fold": state["outer_fold"],
                    "inner_fold": state["inner_fold"],
                    "state_level": state["state_level"],
                    "panel_id": panel,
                    "training_state_gids": len(training_ids),
                    "training_pedigree_gids": pedigree_count,
                    "panel_direct_sparse_gids": len(candidate_ids),
                    "training_direct_sparse_gids": candidate_count,
                    "minimum_training_gids": minimum_training_gids,
                    "support_status": support_status,
                    "graph_projection_ready": False,
                    "embedding_ready": False,
                    "phase6_kz_active": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["state_id", "panel_id"]).reset_index(drop=True)


def status_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.groupby("regulatory_terminal_class", sort=True)
        .agg(
            genotype_count=("canonical_gid", "nunique"),
            primary_genotype_count=("in_primary_view", "sum"),
            selected_trait_rows=("selected_trait_rows", "sum"),
            primary_weighted_rows=("primary_weighted_rows", "sum"),
            pedigree_supported_gids=("pedigree_available", "sum"),
        )
        .reindex(TERMINAL_CLASSES, fill_value=0)
        .reset_index()
    )


def panel_summary(evidence: pd.DataFrame) -> pd.DataFrame:
    return (
        evidence.groupby("panel_id", sort=True)
        .agg(
            stage1_v2_gids=("canonical_gid", "nunique"),
            accepted_identity_gids=("accepted_identity", "sum"),
            direct_sparse_gids=("direct_sparse_evidence", "sum"),
            candidate_only_gids=("candidate_only", "sum"),
            direct_observed_ready_gids=("direct_observed_ready", "sum"),
        )
        .reset_index()
    )


def validate_contract(
    manifest: pd.DataFrame,
    evidence: pd.DataFrame,
    support: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    add(
        "stage1_v2_gid_axis_unique",
        len(manifest) == 10_722 and manifest["canonical_gid"].is_unique,
        f"rows={len(manifest)}; unique={manifest['canonical_gid'].nunique()}; expected=10722",
    )
    add(
        "stage1_v2_primary_population",
        int(manifest["in_primary_view"].sum()) == 10_656,
        f"primary_gids={int(manifest['in_primary_view'].sum())}; expected=10656",
    )
    add(
        "secondary_row_reconciliation",
        int(manifest["secondary_unweighted_rows"].sum()) == 2_242_863,
        f"rows={int(manifest['secondary_unweighted_rows'].sum())}; expected=2242863",
    )
    observed_classes = set(manifest["regulatory_terminal_class"])
    add(
        "terminal_class_domain",
        observed_classes.issubset(set(TERMINAL_CLASSES)),
        f"observed={sorted(observed_classes)}",
    )
    add(
        "one_terminal_class_per_gid",
        manifest["regulatory_terminal_class"].notna().all(),
        f"classified={manifest['regulatory_terminal_class'].notna().sum()}/{len(manifest)}",
    )
    direct_ready = manifest["regulatory_terminal_class"].eq("direct_observed_ready")
    add(
        "no_uncertified_direct_embeddings",
        not direct_ready.any() and not manifest["phase6_kz_eligible"].any(),
        f"direct_ready={int(direct_ready.sum())}; phase6_kz={int(manifest['phase6_kz_eligible'].sum())}",
    )
    sparse = evidence["direct_sparse_evidence"]
    add(
        "direct_sparse_requires_accepted_identity_and_qc",
        bool((~sparse | (evidence["accepted_identity"] & evidence["passes_panel_sample_qc"])).all()),
        f"direct_sparse_rows={int(sparse.sum())}",
    )
    candidates_80k = evidence["panel_id"].astype(str).str.startswith("dartseq80k_")
    add(
        "80k_candidates_not_promoted",
        not evidence.loc[candidates_80k, "accepted_identity"].any()
        and not evidence.loc[candidates_80k, "direct_sparse_evidence"].any(),
        f"candidate_rows={int(candidates_80k.sum())}",
    )
    imputed = manifest["regulatory_terminal_class"].eq("pedigree_imputable")
    add(
        "pedigree_imputation_is_candidate_only",
        bool(
            manifest.loc[imputed, "pedigree_available"].all()
            and not manifest.loc[imputed, "pedigree_imputation_ready"].any()
            and not manifest.loc[imputed, "observed_sequence_equivalent"].any()
        ),
        f"pedigree_imputable={int(imputed.sum())}",
    )
    add(
        "stage1_v2_state_grid",
        len(states) == 150 and states["state_id"].is_unique and len(support["state_id"].unique()) == 150,
        f"states={len(states)}; support_states={support['state_id'].nunique()}",
    )
    expected_support_rows = len(states) * support["panel_id"].nunique()
    add(
        "split_panel_support_grid",
        len(support) == expected_support_rows
        and not support.duplicated(["state_id", "panel_id"]).any(),
        f"rows={len(support)}; expected={expected_support_rows}",
    )
    add(
        "kz_inactive_without_projection",
        not support["phase6_kz_active"].any()
        and not support["graph_projection_ready"].any()
        and not support["embedding_ready"].any(),
        "all split states fail closed before graph projection and embedding",
    )
    return pd.DataFrame(checks)
