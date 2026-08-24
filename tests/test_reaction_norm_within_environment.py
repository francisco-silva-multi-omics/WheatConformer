from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from server_training_pipeline.summarize_reaction_norm_within_environment_screen import (
    validation_within_environment_metrics,
)
from server_training_pipeline.within_environment_objective import (
    deterministic_pair_assignments,
    leave_one_genotype_out_targets,
    validate_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "server_training_pipeline"
    / "reaction_norm_within_environment_protocol_v1.json"
)


def candidate(name: str = "mean_deviation_rank_low") -> dict[str, object]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    return next(value for value in protocol["candidates"] if value["name"] == name)


def training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trait_name_canonical": ["YIELD"] * 5,
            "environment_id": ["E1"] * 4 + ["E2"],
            "genotype_id": ["G1", "G2", "G2", "G3", "G4"],
            "phenotype_value": [10.0, 12.0, 14.0, 18.0, 20.0],
            "y_scaled": [-1.0, -0.5, 0.0, 1.5, 2.0],
            "weight_g_e": [1.0, 1.0, 3.0, 2.0, 1.0],
            "var_g_e": [0.01] * 5,
        }
    )


def test_protocol_is_inner_only_and_candidate_count_is_bounded() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_before_inner_validation"
    assert protocol["selection_data"] == "inner_validation_only"
    assert protocol["prior_routed_outer_metrics_known"] is True
    assert protocol["prior_routed_outer_metrics_used_as_screen_inputs"] is False
    assert protocol["outer_test_metrics_read"] is False
    assert protocol["final_holdout_outcomes_read"] is False
    assert len(protocol["candidates"]) == 4
    for value in protocol["candidates"]:
        validate_candidate(value)


def test_leave_one_genotype_out_mean_excludes_all_rows_for_target_genotype() -> None:
    frame = training_frame()
    targets = leave_one_genotype_out_targets(frame)
    expected_g1 = (-0.5 * 1.0 + 0.0 * 3.0 + 1.5 * 2.0) / 6.0
    expected_g2 = (-1.0 * 1.0 + 1.5 * 2.0) / 3.0
    expected_g3 = (-1.0 * 1.0 - 0.5 * 1.0 + 0.0 * 3.0) / 5.0
    assert targets.loc[0, "environment_mean_target"] == pytest.approx(expected_g1)
    assert targets.loc[1, "environment_mean_target"] == pytest.approx(expected_g2)
    assert targets.loc[2, "environment_mean_target"] == pytest.approx(expected_g2)
    assert targets.loc[3, "environment_mean_target"] == pytest.approx(expected_g3)
    assert targets.loc[4, "decomposition_weight"] == 0.0
    assert targets.loc[0, "genotype_deviation_target"] == pytest.approx(
        frame.loc[0, "y_scaled"] - expected_g1
    )


def test_pair_assignments_are_deterministic_bounded_and_cross_genotype() -> None:
    frame = training_frame().iloc[:4].copy()
    policy = candidate()
    left, diagnostics_left = deterministic_pair_assignments(frame, policy, seed=123)
    right, diagnostics_right = deterministic_pair_assignments(frame, policy, seed=123)
    pd.testing.assert_frame_equal(left, right)
    pd.testing.assert_frame_equal(diagnostics_left, diagnostics_right)
    selected = left[left["pair_weight"].gt(0)]
    assert not selected.empty
    assert len(selected) <= int(policy["maximum_pairs_per_environment_trait"])
    for row_index, row in selected.iterrows():
        partner = int(row["partner_position"])
        assert frame.iloc[row_index]["genotype_id"] != frame.iloc[partner]["genotype_id"]
        assert np.sign(
            frame.iloc[row_index]["y_scaled"] - frame.iloc[partner]["y_scaled"]
        ) == row["pair_direction"]
    assert selected["pair_weight"].sum() == pytest.approx(len(frame))


def test_uncertainty_gate_can_exclude_apparent_near_ties() -> None:
    frame = training_frame().iloc[:4].copy()
    frame["var_g_e"] = 1e8
    assignments, diagnostics = deterministic_pair_assignments(frame, candidate(), seed=123)
    assert assignments["pair_weight"].eq(0).all()
    assert diagnostics["selected_pairs"].eq(0).all()


def test_pair_weights_balance_traits_then_environments() -> None:
    first = training_frame().iloc[:4].copy()
    second = first.copy()
    second["trait_name_canonical"] = "HEIGHT"
    second["environment_id"] = ["E2", "E2", "E3", "E3"]
    frame = pd.concat([first, second], ignore_index=True)
    assignments, _ = deterministic_pair_assignments(frame, candidate(), seed=321)
    weighted = frame[["trait_name_canonical"]].join(assignments[["pair_weight"]])
    totals = weighted.groupby("trait_name_canonical")["pair_weight"].sum()
    assert totals["YIELD"] == pytest.approx(totals["HEIGHT"])
    assert totals.sum() == pytest.approx(len(frame))


def test_validation_metrics_remove_environment_offsets_and_recover_ordering() -> None:
    predictions = pd.DataFrame(
        {
            "trait_name_canonical": ["GRAIN_YIELD"] * 6,
            "environment_id": ["E1"] * 3 + ["E2"] * 3,
            "y_scaled": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
            "y_pred_scaled": [100.0, 101.0, 102.0, -50.0, -49.0, -48.0],
        }
    )
    policy = {
        "environment_id_column": "environment_id",
        "minimum_rows_per_environment_trait": 3,
        "minimum_comparable_pairs_per_environment_trait": 1,
        "pairwise_tie_tolerance_standardized": 0.1,
        "top_k_fraction": 0.34,
        "minimum_top_k": 1,
        "maximum_top_k_fraction": 0.5,
    }
    result = validation_within_environment_metrics(predictions, policy).iloc[0]
    assert result["centered_pearson"] == pytest.approx(1.0)
    assert result["centered_spearman"] == pytest.approx(1.0)
    assert result["pairwise_ordering_accuracy"] == pytest.approx(1.0)
    assert result["tail_regret_sd"] == pytest.approx(0.0)


def test_runner_uses_fresh_contract_and_never_requests_outer_evaluation() -> None:
    source = (
        ROOT / "scripts" / "run_reaction_norm_within_environment_inner_screen.sh"
    ).read_text(encoding="utf-8")
    assert "reaction_norm_within_environment_evaluation_v1" in source
    assert "build_final_evaluation_manifests" in source
    assert "--evaluation-stage inner_selection" in source
    assert "--evaluation-stage outer_evaluation" not in source
    assert "final_holdout" in source.lower()
