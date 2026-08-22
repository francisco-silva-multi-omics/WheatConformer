from __future__ import annotations

import pandas as pd

from server_genotype_recovery.regulatory_eligibility_v2 import (
    build_gid_manifest,
    gid_panel_evidence,
    observation_counts,
    panel_readiness,
    validate_contract,
)


def test_observation_counts_are_bound_to_stage1_v2_flags() -> None:
    observations = pd.DataFrame(
        {
            "canonical_gid": ["GID1", "GID1", "GID2", ""],
            "primary_weighted_training_eligible": [True, False, True, False],
            "secondary_unweighted_training_eligible": [True, True, True, False],
        }
    )
    counts = observation_counts(observations).set_index("canonical_gid")
    assert counts.loc["GID1", "selected_trait_rows"] == 2
    assert counts.loc["GID1", "primary_weighted_rows"] == 1
    assert counts.loc["GID1", "secondary_unweighted_rows"] == 2


def test_panel_readiness_does_not_confuse_sparse_calls_with_kz_readiness() -> None:
    source = pd.DataFrame(
        [
            {
                "panel_id": "seeds_of_discovery_dartseq",
                "raw_marker_availability": "RAW_CALLS_PRESENT",
                "accepted_canonical_gid_count": "100",
                "v2_terminal_disposition": "ACTIVATE_WHERE_TRAINING_SUPPORT_AND_QC_PASS",
                "allele_encoding": "REF_ALT",
                "identity_authority": "SAME_DATASET",
            },
            {
                "panel_id": "dartseq80k_hexaploid",
                "raw_marker_availability": "RAW_CALLS_PRESENT_IDENTITY_NOT_AUTHORIZED",
                "accepted_canonical_gid_count": "0",
                "v2_terminal_disposition": "BLOCKED_NO_SAME_DATASET_TYPED_IDENTITY",
                "allele_encoding": "PAV_AND_SNP",
                "identity_authority": "CANDIDATE_ONLY",
            },
        ]
    )
    overlap = pd.DataFrame(
        {
            "panel_id": ["seeds_of_discovery_dartseq", "dartseq80k_hexaploid"],
            "primary_stage1_gids": [10, 0],
        }
    )
    readiness = panel_readiness(source, overlap, cimmyt_qc_gids=0, seeds_qc_gids=9)
    seeds = readiness.set_index("panel_id").loc["seeds_of_discovery_dartseq"]
    eighty_k = readiness.set_index("panel_id").loc["dartseq80k_hexaploid"]
    assert seeds.panel_kz_class == "direct_sparse_candidate"
    assert seeds.graph_projection_status == "NOT_BUILT_OR_CERTIFIED_V2"
    assert eighty_k.panel_kz_class == "candidate_unresolved"


def test_gid_terminal_classes_prefer_direct_sparse_then_pedigree() -> None:
    genotypes = pd.DataFrame(
        {
            "genotype_index": [0, 1, 2],
            "canonical_gid": ["GID1", "GID2", "GID3"],
            "in_primary_view": [True, True, True],
            "in_secondary_view": [True, True, True],
            "pedigree_available": [True, True, False],
        }
    )
    counts = pd.DataFrame(
        {
            "canonical_gid": ["GID1", "GID2", "GID3"],
            "selected_trait_rows": [1, 1, 1],
            "primary_weighted_rows": [1, 1, 1],
            "secondary_unweighted_rows": [1, 1, 1],
        }
    )
    evidence = pd.DataFrame(
        {
            "canonical_gid": ["GID1", "GID3"],
            "panel_id": ["seeds", "80k"],
            "accepted_identity": [True, False],
            "candidate_only": [False, True],
            "accepted_sample_instances": [1, 1],
            "direct_sparse_evidence": [True, False],
            "direct_observed_ready": [False, False],
            "marker_coordinate_status": ["NOT_CERTIFIED", "NOT_CERTIFIED"],
            "graph_projection_status": ["NOT_BUILT", "NOT_BUILT"],
            "direct_embedding_status": ["NOT_BUILT", "NOT_BUILT"],
        }
    )
    manifest = build_gid_manifest(genotypes, counts, evidence).set_index("canonical_gid")
    assert manifest.loc["GID1", "regulatory_terminal_class"] == "direct_sparse_candidate"
    assert manifest.loc["GID2", "regulatory_terminal_class"] == "pedigree_imputable"
    assert manifest.loc["GID3", "regulatory_terminal_class"] == "candidate_unresolved"
    assert not manifest.phase6_kz_eligible.any()


def test_80k_candidate_identity_never_becomes_direct_evidence() -> None:
    accepted = pd.DataFrame(
        columns=["accepted_canonical_gid", "panel_id", "sample_instance_key", "evidence_type", "mapping_status"]
    )
    unresolved = pd.DataFrame(
        {
            "candidate_canonical_gid": ["GID1"],
            "panel_id": ["dartseq80k_hexaploid"],
            "sample_instance_key": ["S1"],
            "evidence_type": ["cross_panel_exact_sample_id_candidate_only"],
            "mapping_status": ["CANDIDATE_REQUIRES_REVIEW"],
        }
    )
    readiness = pd.DataFrame(
        {
            "panel_id": ["dartseq80k_hexaploid"],
            "panel_kz_class": ["candidate_unresolved"],
            "marker_coordinate_status": ["NOT_CERTIFIED"],
            "graph_projection_status": ["NOT_BUILT_OR_CERTIFIED_V2"],
            "direct_embedding_status": ["NOT_BUILT_OR_CERTIFIED_V2"],
        }
    )
    evidence = gid_panel_evidence(
        {"GID1"}, accepted, readiness, seeds_qc_ids=set(), cimmyt_qc_ids=set(), unresolved_80k=unresolved
    )
    assert evidence.candidate_only.all()
    assert not evidence.accepted_identity.any()
    assert not evidence.direct_sparse_evidence.any()
