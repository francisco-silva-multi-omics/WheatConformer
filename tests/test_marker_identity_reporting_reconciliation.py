from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from server_genotype_recovery.adjudicate_marker_identity_candidates import (
    CANDIDATE_COLUMNS,
    PAIR_COLUMNS,
    adjudication_qc,
)
from server_genotype_recovery.reconcile_marker_identity_reporting import (
    apply_certified_panel_flags,
    classification_evidence_sha256,
    pairwise_evidence_sha256,
)


def candidate_row(gid: str) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in CANDIDATE_COLUMNS}
    row.update(
        {
            "trial_gid": gid,
            "candidate_scope": "new_dataverse_two_hop",
            "panel_id": "SEEDS_DARTSEQ_DATAVERSE_RECOVERY",
            "sample_id": f"sample-{gid}",
            "marker_axis_match_count": 1,
            "classification": "accepted_unique_identity",
            "classification_reasons": "",
            "direct_marker_assignment_ready": True,
        }
    )
    return row


def test_reconciliation_distinguishes_panel_and_global_novelty(tmp_path: Path) -> None:
    policy_path = (
        Path(__file__).resolve().parents[1]
        / "server_genotype_recovery/marker_identity_concordance_policy_v1.json"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    source = pd.DataFrame(
        [candidate_row("GID1"), candidate_row("GID2"), candidate_row("GID3")]
    )
    source["existing_certified_gid"] = False
    before = classification_evidence_sha256(source)
    reconciled = apply_certified_panel_flags(
        source,
        policy=policy,
        certified_ids={
            "SEEDS_DARTSEQ": {"GID1"},
            "HMP": {"GID2"},
        },
    ).set_index("trial_gid")

    assert bool(reconciled.loc["GID1", "existing_certified_in_panel"])
    assert bool(reconciled.loc["GID1", "existing_certified_in_any_panel"])
    assert not bool(reconciled.loc["GID2", "existing_certified_in_panel"])
    assert bool(reconciled.loc["GID2", "existing_certified_in_any_panel"])
    assert not bool(reconciled.loc["GID3", "existing_certified_in_panel"])
    assert not bool(reconciled.loc["GID3", "existing_certified_in_any_panel"])
    assert "existing_certified_gid" not in reconciled.columns
    assert before == classification_evidence_sha256(reconciled.reset_index())

    qc = adjudication_qc(reconciled.reset_index()).set_index("metric")["value"]
    assert int(qc["accepted_two_hop_candidate_gids"]) == 3
    assert qc["new_two_hop_certified_panel_reference"] == "SEEDS_DARTSEQ"
    assert int(qc["accepted_two_hop_new_to_certified_reference_panel_gids"]) == 2
    assert int(qc["accepted_two_hop_new_to_any_certified_panel_gids"]) == 1


def test_reconciliation_rejects_unknown_candidate_panel() -> None:
    policy_path = (
        Path(__file__).resolve().parents[1]
        / "server_genotype_recovery/marker_identity_concordance_policy_v1.json"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    source = pd.DataFrame([candidate_row("GID1")])
    source["panel_id"] = "UNDECLARED_PANEL"
    try:
        apply_certified_panel_flags(
            source,
            policy=policy,
            certified_ids={"SEEDS_DARTSEQ": set()},
        )
    except ValueError as exc:
        assert "No certified-panel reference" in str(exc)
    else:
        raise AssertionError("Unknown panel membership must fail closed")


def test_pairwise_hash_is_stable_across_boolean_serialization() -> None:
    row: dict[str, object] = {column: "" for column in PAIR_COLUMNS}
    row.update(
        {
            "trial_gid": "GID1",
            "panel_id": "SEEDS_DARTSEQ_DATAVERSE_RECOVERY",
            "sample_id_left": "S1",
            "sample_id_right": "S2",
            "shared_nonmissing_markers": 6000,
            "concordant_markers": 5999,
            "call_concordance": 5999 / 6000,
            "minimum_shared_markers": 5000,
            "minimum_call_concordance": 0.995,
            "overlap_pass": True,
            "concordance_pass": "true",
            "pair_status": "PASS",
        }
    )
    serialized = pd.DataFrame([row])
    normalized = serialized.copy()
    normalized["overlap_pass"] = True
    normalized["concordance_pass"] = True
    assert pairwise_evidence_sha256(serialized) == pairwise_evidence_sha256(normalized)
