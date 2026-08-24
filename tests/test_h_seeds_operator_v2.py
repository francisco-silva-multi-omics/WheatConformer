from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from server_genotype_recovery.h_seeds_operator_v2 import (
    HSeedsProtocol,
    build_state_binding,
    observed_pedigree_axis,
    seeds_axis,
)


ROOT = Path(__file__).resolve().parents[1]


def test_h_seeds_binding_uses_training_only_positive_alignment() -> None:
    node_registry = pd.DataFrame(
        {
            "node_index": [0, 1, 2, 3],
            "node_id": ["FOUNDER", "GID1", "GID2", "GID3"],
            "is_observed_gid": [False, True, True, True],
            "raw_relationship_diagonal": [1.0, 1.0, 1.1, 1.2],
        }
    )
    consensus = pd.DataFrame(
        {
            "canonical_gid": ["GID1", "GID2", "GID3", "GID4"],
            "retained_for_component": [True, True, True, True],
        }
    )
    dosage = np.asarray(
        [[0, 0, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [0, 2, 1, 0]],
        dtype=np.uint8,
    )
    factor = sparse.eye(4, format="csr", dtype=np.float64)
    d_values = np.asarray([1.0, 1.0, 1.1, 1.2])
    protocol = HSeedsProtocol(
        genomic_blend_weight=0.95,
        minimum_training_overlap=2,
        diagnostic_sample_size=3,
        marker_block_size=2,
    )
    row, arrays = build_state_binding(
        state_id="STATE",
        scenario="GNEW_EOBS",
        state_level="INNER",
        training_gids={"GID1", "GID2"},
        observed_axis=observed_pedigree_axis(node_registry),
        panel_axis=seeds_axis(consensus),
        dosage=dosage,
        marker_indices=np.arange(4, dtype=np.int64),
        allele_frequency=np.asarray([0.25, 0.25, 0.5, 0.5]),
        denominator=2.0,
        pedigree_factor=factor,
        mendelian_variance=d_values,
        ka_scale=1.05,
        ka_state_hash="ka",
        seeds_state_hash="seeds",
        protocol=protocol,
    )
    assert row["status"] == "PASS"
    assert row["operator_overlap_gids"] == 3
    assert row["training_overlap_gids"] == 2
    assert row["genomic_alignment_scale"] > 0
    assert row["sample_minimum_blended_eigenvalue"] > 0
    assert arrays["training_overlap_mask"].tolist() == [True, True, False]


def test_h_seeds_protocol_is_frozen_before_metrics() -> None:
    import json

    protocol = json.loads(
        (ROOT / "server_genotype_recovery/h_seeds_operator_protocol_v1.json").read_text()
    )
    assert protocol["stage1_version"] == "Stage-1 v2"
    assert protocol["genomic_blend_weight"] == 0.95
    assert protocol["phenotype_values_read"] is False
    assert protocol["inner_validation_metrics_read"] is False
    assert protocol["addition_after_inner_metric_access_allowed"] is False


def test_real_h_seeds_release_covers_all_states_when_built() -> None:
    decision_path = ROOT / "audit/v2/phase6_h_seeds_operator_v1/H_SEEDS_OPERATOR_DECISION.json"
    if not decision_path.exists():
        return
    import json

    decision = json.loads(decision_path.read_text())
    registry = pd.read_csv(
        ROOT / "audit/v2/phase6_h_seeds_operator_v1/h_seeds_operator_registry.tsv",
        sep="\t",
    )
    assert decision["status"] == "PASS_H_SEEDS_150_STATE_OPERATOR_CERTIFIED"
    assert len(registry) == 150
    assert registry["state_id"].is_unique
    assert registry["status"].str.startswith("PASS").all()
