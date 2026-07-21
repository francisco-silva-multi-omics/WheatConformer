from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest

from server_genotype_recovery.adjudicate_marker_identity_candidates import (
    adjudicate_new_candidates,
    classify_replicates,
    marker_by_sample_axis,
    regulatory_overlay,
    resolver_identity_summary,
    stream_marker_by_sample_concordance,
    validate_upstream_provenance,
)
from server_genotype_recovery.fetch_brapi_pedigree_markers import sha256_file


def write_matrix(path: Path) -> None:
    path.write_text(
        "MarkerID\tS1\tS2\tS3\tS4\tS5\n"
        "m1:A>G\tA\tA\tA\tG\tA\n"
        "m2:C>T\tT\tT\tC\tT\tC\n"
        "m3:G>C\tG\tG\tG\tC\tG\n",
        encoding="utf-8",
    )


def test_resolver_selection_history_uniqueness_is_gid_based() -> None:
    resolver = pd.DataFrame(
        {
            "sample_id": ["GID1", "GID1", "GID2", "GID3"],
            "selection_history": ["SEL-1", "SEL-1", "SEL-2", "SEL-2"],
            "cross_name": ["A/B", "A/B", "C/D", "C/D"],
        }
    )
    summary = resolver_identity_summary(resolver).set_index("trial_gid")
    assert bool(summary.loc["GID1", "selection_history_unique"])
    assert not bool(summary.loc["GID2", "selection_history_unique"])
    assert summary.loc["GID2", "selection_history_gid_count"] == 2


def test_selective_marker_concordance_and_terminal_classes(tmp_path: Path) -> None:
    matrix = tmp_path / "calls.tsv"
    write_matrix(matrix)
    axis, header_row = marker_by_sample_axis(matrix)
    assert header_row == 0
    assert axis["S1"] == [(1, "S1")]

    pairs = stream_marker_by_sample_concordance(
        matrix,
        sample_columns={"S1": 1, "S2": 2, "S3": 3, "S4": 4},
        replicate_groups={"GID2": ["S1", "S2"], "GID3": ["S3", "S4"]},
        minimum_shared_markers=3,
        minimum_call_concordance=0.995,
    )
    accepted = pairs[pairs["trial_gid"].eq("GID2")]
    conflicting = pairs[pairs["trial_gid"].eq("GID3")]
    assert accepted.iloc[0]["call_concordance"] == 1.0
    assert bool(accepted.iloc[0]["concordance_pass"])
    assert conflicting.iloc[0]["call_concordance"] == 0.0
    assert classify_replicates(accepted, 2)[0] == "accepted_concordant_replicates"
    assert classify_replicates(conflicting, 2)[0] == "conflicting_marker_samples"

    low_overlap = accepted.copy()
    low_overlap["overlap_pass"] = False
    low_overlap["concordance_pass"] = False
    classification, reasons = classify_replicates(low_overlap, 2)
    assert classification == "requires_metadata_review"
    assert reasons == ["replicate_overlap_below_panel_minimum"]


def test_end_to_end_candidate_adjudication_is_fail_closed(tmp_path: Path) -> None:
    matrix = tmp_path / "calls.tsv"
    write_matrix(matrix)
    resolver = pd.DataFrame(
        {
            "sample_id": ["GID1", "GID2", "GID3", "GID4", "GID5"],
            "selection_history": ["SEL-1", "SEL-2", "SEL-3", "SHARED", "SHARED"],
            "cross_name": ["A/B", "C/D", "E/F", "G/H", "G/H"],
        }
    )
    resolver_summary = resolver_identity_summary(resolver)
    rows = []
    for gid, samples in {
        "GID1": ["S5"],
        "GID2": ["S1", "S2"],
        "GID3": ["S3", "S4"],
        "GID4": ["S5"],
    }.items():
        for sample in samples:
            rows.append(
                {
                    "query_id": gid,
                    "external_alias": sample,
                    "mapping_filename": "SampleIDvsGID_45610samples.txt",
                    "mapping_source_part": "file",
                    "mapping_source_row": 1,
                }
            )
    bridges = pd.DataFrame(rows)
    external = pd.DataFrame(
        columns=[
            "trial_gid",
            "external_gid",
            "external_identity_count",
            "external_record_count",
        ]
    )
    conflicts = pd.DataFrame(
        columns=["query_id", "conflict_status", "conflict_reasons"]
    )
    candidates, pairs = adjudicate_new_candidates(
        bridges=bridges,
        resolver_summary=resolver_summary,
        external_summary=external,
        conflicts=conflicts,
        matrix_path=matrix,
        matrix_sha256=sha256_file(matrix),
        panel_id="TEST_PANEL",
        minimum_shared_markers=3,
        minimum_call_concordance=0.995,
        existing_certified_ids=set(),
    )
    by_gid = candidates.groupby("trial_gid")["classification"].first()
    assert by_gid["GID1"] == "accepted_unique_identity"
    assert by_gid["GID2"] == "accepted_concordant_replicates"
    assert by_gid["GID3"] == "conflicting_marker_samples"
    assert by_gid["GID4"] == "requires_metadata_review"
    assert candidates.loc[
        candidates["trial_gid"].eq("GID1"), "marker_matrix_locator"
    ].iloc[0] == "column:5"
    assert set(pairs["trial_gid"]) == {"GID2", "GID3"}


def test_regulatory_overlay_preserves_mixed_panel_status() -> None:
    candidates = pd.DataFrame(
        [
            {
                "trial_gid": "GID1",
                "panel_id": "PANEL_ACCEPTED",
                "classification": "accepted_unique_identity",
                "direct_marker_assignment_ready": True,
            },
            {
                "trial_gid": "GID1",
                "panel_id": "PANEL_REVIEW",
                "classification": "requires_metadata_review",
                "direct_marker_assignment_ready": False,
            },
        ]
    )
    row = regulatory_overlay(candidates).iloc[0]
    assert row["marker_identity_adjudication_status"] == (
        "accepted_identity_and_candidate_unresolved"
    )
    assert bool(row["accepted_for_new_kernel_input"])
    assert bool(row["candidate_unresolved"])


def test_upstream_provenance_must_use_same_evidence_snapshot(tmp_path: Path) -> None:
    resolver = tmp_path / "resolver.tsv"
    resolver.write_text("sample_id\nGID1\n", encoding="utf-8")
    resolver_hash = sha256_file(resolver)
    two_hop = tmp_path / "two_hop.json"
    pedigree = tmp_path / "pedigree.json"
    two_hop.write_text(
        json.dumps(
            {
                "status": "complete",
                "structured_evidence_sha256": "evidence-a",
                "resolver_query_sha256": resolver_hash,
            }
        ),
        encoding="utf-8",
    )
    pedigree.write_text(
        json.dumps(
            {
                "status": "complete",
                "inputs": {
                    "evidence": {"sha256": "evidence-a"},
                    "resolver": {"sha256": resolver_hash},
                },
            }
        ),
        encoding="utf-8",
    )
    observed = validate_upstream_provenance(
        two_hop_provenance_path=two_hop,
        pedigree_provenance_path=pedigree,
        resolver_path=resolver,
    )
    assert observed["structured_evidence_sha256"] == "evidence-a"

    payload = json.loads(pedigree.read_text(encoding="utf-8"))
    payload["inputs"]["evidence"]["sha256"] = "evidence-b"
    pedigree.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="different structured-evidence snapshots"):
        validate_upstream_provenance(
            two_hop_provenance_path=two_hop,
            pedigree_provenance_path=pedigree,
            resolver_path=resolver,
        )
