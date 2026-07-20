from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from server_genotype_recovery.build_regulatory_eligibility_manifest import (
    build_gid_manifest,
    build_panel_evidence,
    marker_evidence,
    regulatory_retention_policy,
)


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def test_regulatory_manifest_separates_direct_and_imputed_evidence(tmp_path: Path) -> None:
    hmp_dir = tmp_path / "genotype_panels/hmp"
    recovered_dir = tmp_path / "genotype_panels/recovered/test"
    write_tsv(pd.DataFrame({"sample_id": ["GID1"]}), hmp_dir / "hmp_order.tsv")
    write_tsv(pd.DataFrame({"sample_id": ["GID2"]}), recovered_dir / "K_G_TEST_sample_order.tsv")
    write_tsv(
        pd.DataFrame({"marker_id": ["M2_A/G"]}),
        recovered_dir / "K_G_TEST_retained_marker_order.tsv.gz",
    )
    np.save(recovered_dir / "K_G_TEST_QC_dosage.npy", np.asarray([[1.0]]))
    pd.DataFrame({"sample_id": ["GID1"], "M1_A/G": [1.0]}).to_parquet(
        hmp_dir / "hmp_sample_by_marker.QCfiltered.parquet", index=False
    )
    coordinate = tmp_path / "coordinates.tsv"
    write_tsv(
        pd.DataFrame(
            {
                "marker_id": ["M1_A/G", "M2_A/G"],
                "chromosome": ["1A", "2B"],
                "position": [100, 200],
                "alleles": ["A/G", "A/G"],
            }
        ),
        coordinate,
    )
    coordinate_ids, allele_ids, _ = marker_evidence([coordinate])
    candidates = pd.DataFrame(
        [
            {
                "kernel": "K_G_HMP_LINEAR",
                "biological_role": "HMP",
                "order_path": hmp_dir / "hmp_order.tsv",
                "source_id_col": "sample_id",
                "candidate_group": "existing_HMP",
            },
            {
                "kernel": "K_G_TEST_LINEAR",
                "biological_role": "test",
                "order_path": recovered_dir / "K_G_TEST_sample_order.tsv",
                "source_id_col": "sample_id",
                "candidate_group": "K_G_TEST",
            },
        ]
    )
    write_tsv(
        pd.DataFrame(
            {
                "marker_id": ["M1_A/G"],
                "chromosome": ["1A"],
                "position": [100],
                "alleles": ["A/G"],
            }
        ),
        hmp_dir / "hmp_marker_metadata.tsv",
    )
    write_tsv(
        pd.DataFrame({"marker_id": ["M1_A/G"], "keep_marker": [True]}),
        hmp_dir / "qc_hmp_marker_stats.tsv",
    )
    evidence, samples = build_panel_evidence(
        root=tmp_path,
        candidates=candidates,
        qc_status={"K_G_HMP_LINEAR": "PASS", "K_G_TEST_LINEAR": "PASS"},
        coordinate_ids=coordinate_ids,
        allele_ids=allele_ids,
        graph_marker_ids={"M1_A/G", "M2_A/G"},
        minimum_graph_projection_fraction=0.9,
    )
    catalog = pd.DataFrame(
        {
            "canonical_gid": ["GID1", "GID2", "GID3", "GID4"],
            "canonical_observation_rows": [10, 20, 30, 40],
        }
    )
    manifest = build_gid_manifest(
        catalog=catalog,
        pedigree_ids={"GID1", "GID2", "GID3"},
        panel_evidence=evidence,
        panel_samples=samples,
        graph_path_ids={"GID1"},
        embeddings={},
    ).set_index("canonical_gid")
    assert manifest.loc["GID1", "regulatory_embedding_eligibility"] == (
        "eligible_direct_sequence_window_construction"
    )
    assert manifest.loc["GID1", "future_embedding_provenance_class"] == (
        "observed_marker_supported_sequence"
    )
    assert not manifest.loc["GID1", "observed_sequence_equivalent"]
    assert manifest.loc["GID1", "observed_sequence_equivalence_reason"] == (
        "sequence_windows_and_embeddings_not_yet_certified"
    )
    assert manifest.loc["GID2", "regulatory_embedding_eligibility"] == (
        "eligible_direct_sequence_window_construction"
    )
    assert manifest.loc["GID3", "regulatory_embedding_eligibility"] == (
        "pedigree_imputation_candidate"
    )
    assert manifest.loc["GID3", "confidence_gate_status"] == "required_not_evaluated"
    assert not manifest.loc["GID3", "observed_sequence_equivalent"]
    assert manifest.loc["GID4", "regulatory_embedding_eligibility"] == "unavailable"


def test_graph_readiness_requires_alleles_and_coordinates(tmp_path: Path) -> None:
    panel_dir = tmp_path / "genotype_panels/recovered/test"
    write_tsv(
        pd.DataFrame({"sample_id": ["GID1"]}),
        panel_dir / "K_G_TEST_sample_order.tsv",
    )
    write_tsv(
        pd.DataFrame({"marker_id": ["MISSING_EVIDENCE"]}),
        panel_dir / "K_G_TEST_retained_marker_order.tsv.gz",
    )
    np.save(panel_dir / "K_G_TEST_QC_dosage.npy", np.asarray([[1.0]]))
    candidates = pd.DataFrame(
        [
            {
                "kernel": "K_G_TEST_LINEAR",
                "biological_role": "test",
                "order_path": panel_dir / "K_G_TEST_sample_order.tsv",
                "source_id_col": "sample_id",
                "candidate_group": "K_G_TEST",
            }
        ]
    )
    evidence, _ = build_panel_evidence(
        root=tmp_path,
        candidates=candidates,
        qc_status={"K_G_TEST_LINEAR": "PASS"},
        coordinate_ids=set(),
        allele_ids=set(),
        graph_marker_ids={"MISSING_EVIDENCE"},
        minimum_graph_projection_fraction=0.9,
    )
    row = evidence.iloc[0]
    assert row["graph_projected_marker_count"] == 1
    assert row["graph_projected_projectable_marker_count"] == 0
    assert row["graph_projection_fraction"] == 0.0
    assert not row["graph_projection_ready"]


def test_regulatory_policy_rejects_quantitative_panel_discard(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.tsv"
    write_tsv(
        pd.DataFrame(
            {
                "policy": [
                    "quantitative_screen_scope",
                    "retain_certified_panels_for_regulatory_projection",
                    "direct_regulatory_embedding_status",
                    "pedigree_propagated_embedding_status",
                    "pedigree_propagation_requires_confidence_gate",
                    "pedigree_propagation_equivalent_to_observed_sequence",
                ],
                "value": [
                    "standalone_K_G_baseline_inclusion_only",
                    True,
                    "observed_marker_supported_sequence",
                    "imputed_pedigree",
                    True,
                    False,
                ],
            }
        ),
        policy_path,
    )
    values = regulatory_retention_policy(policy_path)
    assert values["retain_certified_panels_for_regulatory_projection"] == "True"
    policy = pd.read_csv(policy_path, sep="\t")
    policy.loc[
        policy["policy"].eq("retain_certified_panels_for_regulatory_projection"),
        "value",
    ] = False
    write_tsv(policy, policy_path)
    try:
        regulatory_retention_policy(policy_path)
    except ValueError as exc:
        assert "contract failed" in str(exc)
    else:
        raise AssertionError("A panel-discarding policy must be rejected")
