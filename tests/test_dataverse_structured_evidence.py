from __future__ import annotations

import pandas as pd

from server_genotype_recovery.audit_dataverse_structured_evidence import (
    evidence_class,
    marker_bridge_class,
    scan_frame,
    summarize_gid_evidence,
)


def test_evidence_class_does_not_promote_family_cross_to_individual() -> None:
    assert evidence_class("selection_history", 1) == (
        "selection_history_exact_unique",
        "individual_candidate",
    )
    assert evidence_class("bcid", 4)[1] == "family_or_batch"
    assert evidence_class("cross_name", 1)[1] == "family_only"


def test_marker_bridge_remains_candidate_until_sample_axis_certification() -> None:
    assert marker_bridge_class(
        "selection_history",
        "individual_candidate",
        "marker",
        "marker_matrix_candidate",
    ) == "candidate_unique_line_in_marker_matrix"


def test_scan_frame_records_exact_cell_and_summary_is_not_direct_ready() -> None:
    frame = pd.DataFrame([["sample", "SEL-1"], ["cross", "P1/P2"]])
    terms = {
        "SEL1": [
            {
                "query_id": "GID1",
                "query_kind": "selection_history",
                "query_text": "SEL-1",
                "resolver_term_gid_count": 1,
            }
        ],
        "P1P2": [
            {
                "query_id": "GID2",
                "query_kind": "cross_name",
                "query_text": "P1/P2",
                "resolver_term_gid_count": 2,
            }
        ],
    }
    source = {
        "dataset_persistent_id": "doi:test",
        "datafile_id": "1",
        "candidate_role": "marker",
        "filename": "calls_transposed.txt",
        "local_path": "/tmp/calls.txt",
        "description": "SNP calls",
    }
    evidence = pd.DataFrame(scan_frame(frame, terms, source, "file"))
    summary = summarize_gid_evidence(evidence).set_index("query_id")
    assert summary.loc["GID1", "unique_selection_history_exact"]
    assert summary.loc["GID1", "marker_bridge_candidate"]
    assert not summary.loc["GID1", "direct_marker_assignment_ready"]
    assert summary.loc["GID2", "strongest_evidence"] == "family_only"
