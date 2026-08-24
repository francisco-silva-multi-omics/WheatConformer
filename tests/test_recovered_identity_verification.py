from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from server_genotype_recovery.verify_recovered_identity_evidence import (
    MARKER_ACCEPTED,
    apply_pedigree_cycle_gate,
    declared_marker_matrix_inputs,
    marker_classification,
    structural_ledger,
    verify_pedigree_edges,
    write_deterministic_gzip_table,
    main,
)
from server_genotype_recovery.adjudicate_marker_identity_candidates import (
    stream_marker_by_sample_concordance,
)


def candidate(
    gid: str,
    sample: str,
    *,
    crop_scope: str = "WHEAT_CONFIRMED",
    classification: str = "accepted_unique_identity",
) -> dict[str, object]:
    return {
        "trial_gid": gid,
        "panel_id": "PANEL",
        "sample_id": sample,
        "normalized_sample_id": sample,
        "classification": classification,
        "classification_reasons": "",
        "external_identity_count": 1,
        "pedigree_conflict_status": "NO_DETECTED_CONFLICT",
        "mapping_filename": "SampleIDvsGID.tsv",
        "candidate_scope": "new_dataverse_two_hop",
        "external_gid": gid,
        "selection_history_unique": True,
        "marker_axis_match_count": 1,
        "selection_history": "SEL-1",
        "trial_cross": "A/B",
        "external_alias": sample,
        "mapping_source_part": "table",
        "mapping_source_row": 1,
        "marker_matrix_path": "/bounded/calls.tsv",
        "marker_matrix_locator": "column:1",
        "dataset_persistent_id": "hdl:11529/wheat",
        "marker_source_file": "calls.tsv",
        "crop_scope": crop_scope,
        "direct_gid_mapping_evidence": True,
    }


def test_marker_verification_accepts_only_explicit_terminal_paths() -> None:
    candidates = pd.DataFrame(
        [
            candidate("GID1", "S1"),
            candidate("GID2", "S2", crop_scope="NON_WHEAT_EXCLUDED"),
            candidate("GID3", "S3", classification="requires_metadata_review"),
        ]
    )
    verification, accepted, _ = marker_classification(
        candidates,
        pd.DataFrame(),
        pd.DataFrame(),
        {},
        minimum_shared=1000,
        minimum_concordance=0.995,
    )
    observed = verification.set_index("canonical_gid")["verification_class"]
    assert observed["GID1"] == "accepted_direct_gid_to_marker_sample"
    assert observed["GID2"] == "non_wheat_excluded"
    assert observed["GID3"] == "unresolved"
    assert set(accepted["canonical_gid"]) == {"GID1"}
    assert set(accepted["mapping_class"]).issubset(MARKER_ACCEPTED)


def test_unique_two_hop_identity_is_not_mislabeled_as_direct_gid() -> None:
    row = candidate("GID1", "S1")
    row["direct_gid_mapping_evidence"] = False
    verification, accepted, _ = marker_classification(
        pd.DataFrame([row]),
        pd.DataFrame(),
        pd.DataFrame(),
        {},
        minimum_shared=1000,
        minimum_concordance=0.995,
    )
    assert verification.iloc[0]["verification_class"] == "accepted_unique_two_hop_identity"
    assert accepted.iloc[0]["mapping_class"] == "accepted_unique_two_hop_identity"


def test_replicate_classification_distinguishes_overlap_and_call_conflict() -> None:
    candidates = pd.DataFrame(
        [
            candidate("GID1", "S1", classification="accepted_concordant_replicates"),
            candidate("GID1", "S2", classification="accepted_concordant_replicates"),
            candidate("GID2", "S3", classification="conflicting_marker_samples"),
            candidate("GID2", "S4", classification="conflicting_marker_samples"),
        ]
    )
    pairs = pd.DataFrame(
        [
            {
                "trial_gid": "GID1",
                "panel_id": "PANEL",
                "sample_id_left": "S1",
                "sample_id_right": "S2",
                "shared_nonmissing_markers": 1500,
                "call_concordance": 0.999,
            },
            {
                "trial_gid": "GID2",
                "panel_id": "PANEL",
                "sample_id_left": "S3",
                "sample_id_right": "S4",
                "shared_nonmissing_markers": 1500,
                "call_concordance": 0.98,
            },
        ]
    )
    verification, accepted, replicates = marker_classification(
        candidates,
        pd.DataFrame(),
        pairs,
        {},
        minimum_shared=1000,
        minimum_concordance=0.995,
    )
    observed = verification.set_index("canonical_gid")["verification_class"]
    assert observed["GID1"] == "accepted_concordant_technical_replicates"
    assert observed["GID2"] == "conflicting_marker_calls"
    assert accepted["canonical_gid"].tolist() == ["GID1"]
    assert replicates.iloc[0]["replicate_count"] == 2


def test_large_marker_matrix_access_is_streaming_and_bounded(
    tmp_path: Path, monkeypatch
) -> None:
    matrix = tmp_path / "calls.tsv"
    with matrix.open("w", encoding="utf-8", newline="") as handle:
        handle.write("MarkerID\tS1\tS2\tUNUSED\n")
        for index in range(1500):
            handle.write(f"m{index}:A>G\tA\tA\tG\n")

    def prohibited_dataframe_read(*args, **kwargs):
        raise AssertionError("large marker matrices must not enter pandas.read_csv")

    monkeypatch.setattr(pd, "read_csv", prohibited_dataframe_read)
    pairs = stream_marker_by_sample_concordance(
        matrix,
        sample_columns={"S1": 1, "S2": 2},
        replicate_groups={"GID1": ["S1", "S2"]},
        minimum_shared_markers=1000,
        minimum_call_concordance=0.995,
    )
    assert pairs.iloc[0]["shared_nonmissing_markers"] == 1500
    assert pairs.iloc[0]["call_concordance"] == 1.0


def test_pedigree_verification_uses_purdy_registry_and_blocks_conflicts() -> None:
    records = pd.DataFrame(
        [
            {
                "query_id": "GID1",
                "external_gid": "GID1",
                "external_parent1": "PARENT-A",
                "external_parent2": "PARENT-B",
                "external_lineage": "",
                "dataset_persistent_id": "D1",
                "filename": "parents.tsv",
                "source_part": "table",
                "source_row": 1,
            },
            {
                "query_id": "GID2",
                "external_gid": "GID2",
                "external_parent1": "",
                "external_parent2": "",
                "external_lineage": "A/B//C/3/D",
                "dataset_persistent_id": "D2",
                "filename": "lineage.tsv",
                "source_part": "table",
                "source_row": 2,
            },
        ]
    )
    conflicts = pd.DataFrame(
        [
            {"query_id": "GID1", "conflict_status": "NO_DETECTED_CONFLICT", "conflict_reasons": ""},
            {"query_id": "GID2", "conflict_status": "NO_DETECTED_CONFLICT", "conflict_reasons": ""},
        ]
    )
    legacy_edges = pd.DataFrame(
        columns=[
            "child_id",
            "parent_id",
            "edge_review_status",
            "edge_already_in_current_pedigree",
        ]
    )
    current = pd.DataFrame(
        {
            "sample_id": ["GID1", "GID2"],
            "parent1": ["", "OLD-A"],
            "parent2": ["", "OLD-B"],
        }
    )
    verified, registry = verify_pedigree_edges(
        records,
        conflicts,
        legacy_edges,
        pd.DataFrame(),
        current,
    )
    gid1 = verified[verified["child_id"].eq("GID1")]
    gid2 = verified[verified["child_id"].eq("GID2")]
    assert set(gid1["verification_class"]) == {"accepted_new_edge_exact_unique"}
    assert set(gid2["verification_class"]) == {"conflicts_existing_complete_parent_pair"}
    assert registry["stable_parent_id"].is_unique
    assert any(registry["node_type"].eq("derived_purdy_cross_node"))


def test_pedigree_cycle_gate_rejects_self_and_ancestral_cycles() -> None:
    current = pd.DataFrame(
        {
            "sample_id": ["GID1", "GID2"],
            "parent1": ["GID2", ""],
            "parent2": ["", ""],
        }
    )
    candidates = pd.DataFrame(
        [
            {
                "child_id": "GID2",
                "parent_role": "parent1",
                "parent_id": "GID1",
                "verification_class": "accepted_new_edge_exact_unique",
                "accepted": True,
                "is_new_edge": True,
                "conflict_reasons": "",
            },
            {
                "child_id": "GID3",
                "parent_role": "parent1",
                "parent_id": "GID3",
                "verification_class": "accepted_new_edge_exact_unique",
                "accepted": True,
                "is_new_edge": True,
                "conflict_reasons": "",
            },
        ]
    )
    gated = apply_pedigree_cycle_gate(candidates, current)
    assert not gated["accepted"].any()
    assert gated["conflict_reasons"].str.contains("pedigree_cycle").all()


def test_structural_ledger_never_reads_phenotype_values(tmp_path: Path) -> None:
    path = tmp_path / "ledger.parquet"
    pd.DataFrame(
        {
            "panel_sample_id": ["GID1"],
            "env_kernel_id": ["ENV1"],
            "cycle": ["2020"],
            "country": ["MEXICO"],
            "trait_name_canonical": ["GRAIN_YIELD"],
            "phenotype_value": [999.0],
            "outer_test_rmse": [123.0],
        }
    ).to_parquet(path, index=False)
    observed = structural_ledger(path)
    assert set(observed.columns) == {
        "panel_sample_id",
        "env_kernel_id",
        "cycle",
        "country",
        "trait_name_canonical",
    }
    assert "phenotype_value" not in observed
    assert "outer_test_rmse" not in observed


def test_deterministic_gzip_output(tmp_path: Path) -> None:
    frame = pd.DataFrame({"gid": ["GID2", "GID1"], "value": [2, 1]})
    first = tmp_path / "first.tsv.gz"
    second = tmp_path / "second.tsv.gz"
    write_deterministic_gzip_table(frame, first, ["gid"])
    write_deterministic_gzip_table(frame, second, ["gid"])
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()


def test_declared_marker_matrix_identity_is_unique_and_valid(tmp_path: Path) -> None:
    matrix = tmp_path / "calls.tsv"
    matrix.write_text("MarkerID\tS1\n", encoding="utf-8")
    digest = hashlib.sha256(matrix.read_bytes()).hexdigest()
    paths, hashes = declared_marker_matrix_inputs(
        tmp_path,
        pd.DataFrame(
            {
                "marker_matrix_path": [str(matrix), str(matrix)],
                "marker_matrix_sha256": [digest, digest],
            }
        ),
    )
    assert list(paths.values()) == [matrix]
    assert list(hashes.values()) == [digest]


def test_full_verification_pipeline_is_reproducible_and_non_destructive(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path
    wide = root / "wide"
    structured = wide / "structured_evidence"
    two_hop = structured / "two_hop_marker_bridges"
    pedigree_evidence = structured / "pedigree_enrichment"
    adjudication = root / "adjudication"
    pedigree = root / "pedigree"
    model = root / "model"
    nested = root / "nested"
    for directory in (
        two_hop,
        pedigree_evidence,
        adjudication,
        pedigree,
        model,
        nested,
        root / "hmp",
        root / "gbs",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    resolver = root / "resolver.tsv"
    pd.DataFrame(
        {
            "sample_id": ["GID1", "GID2", "GID3"],
            "selection_history": ["SEL1", "SEL2", "SEL3"],
            "cross_name": ["A/B", "C/D", "E/F"],
        }
    ).to_csv(resolver, sep="\t", index=False)
    matrix = root / "calls.tsv"
    matrix.write_text("MarkerID\tS1\nm1:A>G\tA\n", encoding="utf-8")
    structured_frame = pd.DataFrame(
        [
            {
                "query_id": "GID1",
                "evidence_class": "direct_gid_exact",
                "crop_scope": "WHEAT_CONFIRMED",
                "dataset_persistent_id": "hdl:wheat",
                "filename": "mapping.tsv",
            }
        ]
    )
    structured_frame.to_csv(
        structured / "dataverse_structured_evidence.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        [{"dataset_persistent_id": "hdl:wheat", "crop_scope": "WHEAT_CONFIRMED"}]
    ).to_csv(
        structured / "dataverse_structured_source_crop_scope.tsv",
        sep="\t",
        index=False,
    )
    bridges = pd.DataFrame(
        [
            {
                "query_id": "GID1",
                "query_text": "SEL1",
                "dataset_persistent_id": "hdl:wheat",
                "external_alias": "S1",
                "mapping_filename": "SampleIDvsGID.tsv",
                "mapping_source_part": "table",
                "mapping_source_row": 1,
                "marker_filename": "calls.tsv",
                "bridge_confidence": "moderate_candidate_requires_disambiguation",
            }
        ]
    )
    bridges.to_csv(
        two_hop / "dataverse_two_hop_marker_bridges.tsv", sep="\t", index=False
    )
    (two_hop / "dataverse_two_hop_marker_bridge_provenance.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    records = pd.DataFrame(
        [
            {
                "query_id": "GID1",
                "external_gid": "GID1",
                "external_parent1": "PARENT-A",
                "external_parent2": "PARENT-B",
                "external_lineage": "",
                "dataset_persistent_id": "hdl:wheat",
                "filename": "parents.tsv",
                "source_part": "table",
                "source_row": 1,
            }
        ]
    )
    records.to_csv(
        pedigree_evidence / "dataverse_pedigree_external_records.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        [{"query_id": "GID1", "conflict_status": "NO_DETECTED_CONFLICT", "conflict_reasons": ""}]
    ).to_csv(
        pedigree_evidence / "dataverse_pedigree_conflicts.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        columns=[
            "child_id",
            "parent_id",
            "edge_review_status",
            "edge_already_in_current_pedigree",
        ]
    ).to_csv(
        pedigree_evidence / "dataverse_pedigree_candidate_edges.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        columns=["query_id", "candidate_node", "node_role", "derivation", "source_filename"]
    ).to_csv(
        pedigree_evidence / "dataverse_pedigree_candidate_nodes.tsv",
        sep="\t",
        index=False,
    )
    (pedigree_evidence / "dataverse_pedigree_enrichment_provenance.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    candidate_frame = pd.DataFrame(
        [
            {
                **candidate("GID1", "S1"),
                "external_identity_count": 1,
                "external_record_count": 1,
                "pedigree_conflict_reasons": "",
                "certified_panel_reference": "PANEL",
                "existing_certified_in_panel": False,
                "existing_certified_in_any_panel": False,
                "direct_marker_assignment_ready": True,
                "marker_matrix_sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
                "marker_matrix_axis": "sample_column",
                "marker_matrix_axis_index": 1,
                "marker_matrix_path": str(matrix),
            }
        ]
    )
    candidate_frame.to_csv(
        adjudication / "marker_identity_candidate_paths.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        columns=[
            "trial_gid",
            "panel_id",
            "sample_id_left",
            "sample_id_right",
            "shared_nonmissing_markers",
            "call_concordance",
        ]
    ).to_csv(
        adjudication / "marker_identity_pairwise_concordance.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    (adjudication / "marker_identity_adjudication_provenance.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    pd.DataFrame(
        {"sample_id": ["GID1", "GID2", "GID3"], "parent1": ["", "", ""], "parent2": ["", "", ""]}
    ).to_csv(pedigree / "parents.tsv", sep="\t", index=False)
    for path in (
        pedigree / "order.tsv",
        model / "order.tsv",
    ):
        pd.DataFrame({"sample_id": ["GID1", "GID2", "GID3"]}).to_csv(
            path, sep="\t", index=False
        )
    pd.DataFrame({"sample_id": ["GID1"]}).to_csv(
        root / "hmp/order.tsv", sep="\t", index=False
    )
    pd.DataFrame({"sample_id": ["GID2"]}).to_csv(
        root / "gbs/order.tsv", sep="\t", index=False
    )
    identity_policy = root / "identity_policy.json"
    identity_policy.write_text(
        json.dumps(
            {
                "existing_panel_artifacts": [],
                "direct_certified_panel_orders": [
                    {"panel_id": "HMP", "sample_order_path": "hmp/order.tsv"},
                    {"panel_id": "GBS_SAWYT", "sample_order_path": "gbs/order.tsv"},
                ],
            }
        ),
        encoding="utf-8",
    )
    ledger = nested / "ledger.parquet"
    pd.DataFrame(
        {
            "panel_sample_id": ["GID1", "GID2", "GID3"],
            "env_kernel_id": ["E1", "E2", "E3"],
            "cycle": ["2018", "2019", "2020"],
            "country": ["A", "B", "C"],
            "trait_name_canonical": ["DTH", "DTH", "DTH"],
            "phenotype_value": [1.0, 2.0, 3.0],
        }
    ).to_parquet(ledger, index=False)
    entity_manifest = nested / "nested_evaluation_entities.tsv"
    pd.DataFrame(
        [
            {"scenario": "unseen_genotypes", "outer_fold": 0, "inner_fold": 0, "axis": "genotype", "partition": "outer_test", "entity_id": "GID3"},
            {"scenario": "unseen_genotypes", "outer_fold": 0, "inner_fold": 0, "axis": "genotype", "partition": "inner_validation", "entity_id": "GID2"},
        ]
    ).to_csv(entity_manifest, sep="\t", index=False)
    nested_contract = nested / "nested_evaluation_contract.json"
    nested_contract.write_text(
        json.dumps(
            {
                "status": "frozen",
                "ledger_path": str(ledger),
                "entity_manifest_sha256": hashlib.sha256(entity_manifest.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    output = root / "verification"
    project = Path(__file__).resolve().parents[1]
    base_args = [
        "verify",
        "--root",
        str(root),
        "--policy",
        str(project / "server_genotype_recovery/recovered_identity_verification_policy_v2.json"),
        "--identity-policy",
        str(identity_policy),
        "--resolver-query",
        str(resolver),
        "--wide-dir",
        str(wide),
        "--marker-adjudication-dir",
        str(adjudication),
        "--pedigree-parent-table",
        str(pedigree / "parents.tsv"),
        "--k-a-order",
        str(pedigree / "order.tsv"),
        "--model-genotype-order",
        str(model / "order.tsv"),
        "--nested-evaluation-dir",
        str(nested),
        "--out-dir",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", base_args)
    main()
    first_manifest = (output / "verification_sha256.tsv").read_bytes()
    protected_hash = hashlib.sha256((pedigree / "parents.tsv").read_bytes()).hexdigest()
    monkeypatch.setattr(sys, "argv", [*base_args, "--force"])
    main()
    assert (output / "verification_sha256.tsv").read_bytes() == first_manifest
    assert hashlib.sha256((pedigree / "parents.tsv").read_bytes()).hexdigest() == protected_hash
    provenance = json.loads((output / "verification_provenance.json").read_text())
    assert provenance["phenotype_values_read"] is False
    assert provenance["single_step_H_constructed"] is False
    assert provenance["structural_ledger_columns_read"] == sorted(
        {
            "panel_sample_id",
            "env_kernel_id",
            "cycle",
            "country",
            "trait_name_canonical",
        }
    )
