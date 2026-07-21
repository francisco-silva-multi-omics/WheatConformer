from __future__ import annotations

import pandas as pd

from server_genotype_recovery.audit_dataverse_pedigree_enrichment import (
    build_alias_candidates,
    candidate_relationships,
    extract_external_records,
    phenotype_impact,
    resolver_lineage,
    split_lineage,
    summarize_conflicts,
)


def evidence_row(
    query_id: str, query_text: str, source_row: int
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query_text": query_text,
        "evidence_class": "selection_history_exact_unique",
        "dataset_persistent_id": "doi:test",
        "datafile_id": "1",
        "filename": "germplasm.tsv",
        "local_path": "/tmp/germplasm.tsv",
        "source_part": "file",
        "source_row": source_row,
    }


def test_external_records_and_conflicts_preserve_review_boundary() -> None:
    frame = pd.DataFrame(
        [
            ["GID", "Cross Name", "Selection History", "Parent1", "Parent2"],
            ["7001", "PARENT_A/PARENT_B", "SEL-1", "PARENT_A", "PARENT_B"],
            ["7002", "OTHER_A/OTHER_B", "SEL-1", "OTHER_A", "OTHER_B"],
        ]
    )
    evidence = pd.DataFrame(
        [evidence_row("GID1", "SEL-1", 1), evidence_row("GID1", "SEL-1", 2)]
    )
    records = extract_external_records(
        evidence, {("/tmp/germplasm.tsv", "file"): frame}
    )
    resolver = pd.DataFrame(
        {
            "sample_id": ["GID1"],
            "selection_history": ["SEL-1"],
            "cross_name": ["PARENT_A/PARENT_B"],
            "parent1": ["PARENT_A"],
            "parent2": ["PARENT_B"],
        }
    )
    conflicts = summarize_conflicts(
        records, resolver_lineage(resolver), {"GID1"}
    )
    row = conflicts.iloc[0]
    assert row["external_gid_count"] == 2
    assert row["multiple_external_gid"]
    assert row["conflict_status"] == "CONFLICT_REQUIRES_REVIEW"
    aliases = build_alias_candidates(records, conflicts)
    assert set(aliases["alias_review_status"]) == {"blocked_by_record_conflict"}
    assert not aliases["automatic_pedigree_update_ready"].any()


def test_candidate_relationships_distinguish_existing_new_and_complex() -> None:
    records = pd.DataFrame(
        [
            {
                "query_id": "GID1",
                "external_parent1": "GID10",
                "external_parent2": "GID30",
                "external_lineage": "GID10/GID30",
                "filename": "pedigree.tsv",
                "source_part": "file",
                "source_row": 2,
            },
            {
                "query_id": "GID2",
                "external_parent1": "",
                "external_parent2": "",
                "external_lineage": "PARENT_A//PARENT_B/PARENT_C",
                "filename": "pedigree.tsv",
                "source_part": "file",
                "source_row": 3,
            },
        ]
    )
    conflicts = pd.DataFrame(
        {
            "query_id": ["GID1", "GID2"],
            "conflict_status": ["NO_DETECTED_CONFLICT", "NO_DETECTED_CONFLICT"],
        }
    )
    current = pd.DataFrame(
        {
            "sample_id": ["GID1", "GID10", "GID2"],
            "parent1": ["GID10", "", ""],
            "parent2": ["", "", ""],
        }
    )
    nodes, edges = candidate_relationships(
        records, conflicts, current, {"GID1", "GID2", "GID10"}
    )
    status = edges.set_index(["child_id", "parent_id"])["edge_review_status"]
    assert status.loc[("GID1", "GID10")] == "ALREADY_PRESENT"
    assert status.loc[("GID1", "GID30")] == "NEW_CANONICAL_EDGE_CANDIDATE"
    complex_nodes = nodes[nodes["query_id"].eq("GID2")]
    assert set(complex_nodes["node_role"]) == {"unresolved_ancestor_token"}
    assert not edges["child_id"].eq("GID2").any()
    assert not edges["automatic_pedigree_update_ready"].any()


def test_split_lineage_only_assigns_roles_for_simple_crosses() -> None:
    parents, tokens, status = split_lineage("PARENT_A/PARENT_B")
    assert parents == [("parent1", "PARENT_A"), ("parent2", "PARENT_B")]
    assert tokens == ["PARENT_A", "PARENT_B"]
    assert status == "simple_two_parent_cross"

    parents, tokens, status = split_lineage("PARENT_A//PARENT_B/PARENT_C")
    assert parents == []
    assert tokens == ["PARENT_A", "PARENT_B", "PARENT_C"]
    assert status == "complex_lineage_tokens_unresolved"


def test_phenotype_impact_counts_identifiers_without_outcomes() -> None:
    observations = pd.DataFrame(
        {
            "query_id": ["GID1", "GID1", "GID2", "GID9"],
            "trait_name_canonical": ["DTH", "GY", "DTH", "DTH"],
            "environment_id": ["E1", "E2", "E1", "E9"],
        }
    )
    gid, trait = phenotype_impact(observations, {"GID1", "GID2", "GID3"})
    indexed = gid.set_index("query_id")
    assert indexed.loc["GID1", "model_observation_rows"] == 2
    assert indexed.loc["GID3", "model_observation_rows"] == 0
    impact = trait.set_index("trait_name_canonical")
    assert impact.loc["DTH", "affected_query_ids"] == 2
    assert impact.loc["GY", "observation_rows"] == 1
