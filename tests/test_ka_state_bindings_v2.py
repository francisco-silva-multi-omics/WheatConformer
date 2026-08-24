from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from server_genotype_recovery.ka_state_bindings_v2 import (
    build_state_binding,
    compare_replayed_binding,
    index_signature,
    observed_pedigree_registry,
    validate_combined_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic_registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node_index": [0, 1, 2, 3],
            "node_id": ["LEAF", "GID1", "GID2", "GID3"],
            "is_observed_gid": [False, True, True, True],
            "raw_relationship_diagonal": [1.0, 1.0, 1.25, 0.75],
        }
    )


def test_build_state_binding_uses_training_only_scale_and_fixed_order() -> None:
    observed = observed_pedigree_registry(synthetic_registry())
    state = {"state_id": "TEMPORAL_YEAR__OUTER1", "scenario": "TEMPORAL_YEAR", "outer_fold": 1, "inner_fold": None}
    entities, row, node_indices = build_state_binding(
        state,
        {"GID1", "GID2", "UNSUPPORTED"},
        observed,
        entity_order_path="pedigree/states/state.tsv",
        raw_operator_factor="factor.npz",
        raw_operator_d="d.npy",
    )
    assert entities["canonical_gid"].tolist() == ["GID1", "GID2", "GID3"]
    assert entities["partition"].tolist() == ["TRAINING", "TRAINING", "APPLICATION"]
    assert row["training_scale_mean_diagonal"] == 1.125
    assert row["training_observed_gids"] == 2
    assert node_indices == [1, 2]
    assert row["entity_order_signature"] == index_signature(["GID1", "GID2", "GID3"])


def test_replay_requires_hash_scale_and_partition_identity() -> None:
    observed = observed_pedigree_registry(synthetic_registry())
    state = {"state_id": "GNEW_EOBS__OUTER1", "scenario": "GNEW_EOBS", "outer_fold": 1, "inner_fold": None}
    entities, row, _ = build_state_binding(
        state,
        {"GID1", "GID3"},
        observed,
        entity_order_path="same.tsv",
        raw_operator_factor="factor.npz",
        raw_operator_d="d.npy",
    )
    replay = compare_replayed_binding(row, entities, row.copy(), entities.copy())
    assert replay["status"] == "PASS"
    altered = row.copy()
    altered["state_hash"] = "different"
    assert compare_replayed_binding(row, entities, altered, entities)["status"] == "FAIL"


def test_combined_registry_requires_exact_90_60_source_split() -> None:
    states = pd.DataFrame(
        {
            "state_id": [f"S{i}" for i in range(150)],
            "scenario": sum(([scenario] * 30 for scenario in ["A", "B", "C", "D", "E"]), []),
        }
    )
    combined = states.copy()
    combined["binding_source"] = ["IMMUTABLE_PHASE5"] * 90 + ["TEMPORAL_COUNTRY_EXTENSION"] * 60
    combined["status"] = "PASS"
    combined["training_scale_mean_diagonal"] = 1.0
    combined["raw_operator_factor_root_relative"] = "factor.npz"
    combined["raw_operator_d_root_relative"] = "d.npy"
    checks = validate_combined_registry(combined, states)
    assert checks["status"].eq("PASS").all()


def test_real_existing_binding_constructor_replays_one_phase5_state() -> None:
    phase5 = ROOT / "audit/v2/phase5_split_bound_kernel_validation_v2"
    parity = ROOT / "audit/v2/phase5_panel_environment_scenario_parity_extension_v2"
    node_registry = pd.read_csv(phase5 / "pedigree/pedigree_node_registry.tsv", sep="\t", dtype=str)
    observed = observed_pedigree_registry(node_registry)
    states = pd.read_csv(parity / "splits/state_registry.tsv", sep="\t", dtype=str)
    state = states.loc[states["state_id"].eq("GNEW_EOBS__OUTER1")].iloc[0].to_dict()
    training = pd.read_csv(parity / state["training_gid_path"], sep="\t", dtype=str)
    registry = pd.read_csv(phase5 / "pedigree/ka_registry.tsv", sep="\t", dtype=str)
    frozen = registry.loc[registry["state_id"].eq(state["state_id"])].iloc[0].to_dict()
    frozen_entities = pd.read_csv(phase5 / frozen["entity_order_path"], sep="\t")
    generated_entities, generated, _ = build_state_binding(
        state,
        set(training["canonical_gid"].astype(str)),
        observed,
        entity_order_path=frozen["entity_order_path"],
        raw_operator_factor=frozen["raw_operator_factor"],
        raw_operator_d=frozen["raw_operator_d"],
    )
    replay = compare_replayed_binding(
        generated, generated_entities, frozen, frozen_entities
    )
    assert replay["status"] == "PASS"
    assert np.isclose(float(generated["training_scale_mean_diagonal"]), 1.010959905624992)
