from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.v2 import run_stage1_v2_phase6_private_head_screen as screen


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_private_head_screen_protocol_v1.json"
)
SOURCE_PROTOCOL = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_trait_balance_screen_protocol_v1.json"
)
PLAN = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_post_hierarchy_screen_plan_v2.json"
)
TRAINER = (
    ROOT
    / "server_training_pipeline"
    / "train_stage1_v2_phase6_private_heads_tf.py"
)
MODEL_BUILDER = (
    ROOT
    / "server_training_pipeline"
    / "train_stage1_v2_phase6_remediation_tf.py"
)
RUNNER = ROOT / "scripts" / "v2" / "run_stage1_v2_phase6_private_head_screen.py"
FREEZE = ROOT / "scripts" / "v2" / "freeze_stage1_v2_phase6_private_head_screen.py"
PACKAGER = (
    ROOT
    / "scripts"
    / "v2"
    / "package_stage1_v2_phase6_private_head_results.py"
)


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_private_head_gate_changes_only_decoder_sharing() -> None:
    value = protocol()
    source = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    policy = value["architecture_policy"]
    assert policy["only_mutable_component"] == "trait_decoder_sharing"
    assert policy["fixed_authoritative_row_mass"] is True
    assert policy["fixed_factor_backbone"] is True
    assert value["fixed_configuration"] == source["fixed_configuration"]
    assert value["trial_environment_hierarchy"] == source[
        "trial_environment_hierarchy"
    ]
    assert value["trait_specific_regularization"] == source[
        "trait_specific_regularization"
    ]
    assert value["positive_training_calibration"] == source[
        "positive_training_calibration"
    ]
    assert value["test_weight_environment_oof_calibration"] == source[
        "test_weight_environment_oof_calibration"
    ]
    assert value["fixed_configuration"]["batch_size"] == 8192
    assert value["product_policy"]["projection_compatible_product"] == (
        "unchanged_and_separate"
    )
    assert value["outer_test_metrics_read"] is False
    assert value["final_holdout_outcomes_read"] is False


def test_two_private_decoder_candidates_are_preregistered() -> None:
    value = protocol()
    candidates = {
        name: candidate
        for name, candidate in value["candidates"].items()
        if candidate.get("source_reuse") is False
    }
    assert set(candidates) == {
        "trait_private_residual_heads",
        "family_shared_trait_private_residual_heads",
    }
    assert {
        candidate["decoder_policy"]["mode"] for candidate in candidates.values()
    } == {
        "trait_private_residual",
        "family_shared_trait_private_residual",
    }
    traits = {*value["primary_traits"], *value["exploratory_traits"]}
    assert set(value["trait_families"]) == traits
    for candidate in candidates.values():
        decoder = candidate["decoder_policy"]
        assert decoder["trait_family_map"] == value["trait_families"]
        assert decoder["trait_private_penalty_multiplier"] == 4.0
        assert decoder["family_penalty_multiplier"] == 1.0


def test_private_head_gate_is_second_in_the_frozen_plan() -> None:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    gate = value["ordered_gates"][1]
    assert gate["name"] == "trait_family_private_heads"
    assert gate["only_mutable_component"] == "trait_decoder_sharing"


def test_implementation_binds_decoder_activity_and_parent_terminal_decision() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    model_builder = MODEL_BUILDER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    freeze = FREEZE.read_text(encoding="utf-8")
    assert "decoder_policy=decoder_policy" in trainer
    assert "authoritative_row_mass_changed" in trainer
    assert "PrivateDecoderHierarchicalReactionNorm" in model_builder
    assert "private_trait_decoder_" in model_builder
    assert "family_decoder_" in model_builder
    assert '"decoder_policies_active"' in runner
    assert '"outer_evaluation_allowed": False' in runner
    assert "parent_trait_balance_terminal" in freeze
    for implementation in (
        "trainer",
        "calibration_helper",
        "calibration_trainer",
        "remediation_helper",
        "model_builder",
        "factor_builder",
        "trainer_interface",
        "runner",
    ):
        assert f'"{implementation}": sha256_file(paths["{implementation}"])' in freeze


def test_result_packager_requires_sealed_complete_phase1_outputs() -> None:
    source = PACKAGER.read_text(encoding="utf-8")
    assert "EXPECTED_STATUSES" in source
    assert "PRIVATE_HEAD_PHASE1_DECISION.json" in source
    assert "decoder_parameter_inventory.tsv" in source
    assert 'value.get("outer_test_outcomes_read") is not False' in source
    assert 'value.get("final_holdout_outcomes_read") is not False' in source
    assert "len(runs) != 15" in source
    assert "len(candidate_runs) != 10" in source


def test_grid_ignores_blank_outer_level_inner_folds(
    tmp_path: Path, monkeypatch
) -> None:
    parity = Path("parity")
    source_runs = Path("source_runs")
    split_dir = tmp_path / parity / "splits"
    split_dir.mkdir(parents=True)
    rows = [
        {
            "state_id": "GNEW_EOBS__OUTER1",
            "state_level": "OUTER",
            "scenario": "GNEW_EOBS",
            "outer_fold": "1",
            "inner_fold": "",
        }
    ]
    for outer_fold in range(1, 6):
        state_id = f"GNEW_EOBS__OUTER{outer_fold}__INNER1"
        rows.append(
            {
                "state_id": state_id,
                "state_level": "INNER",
                "scenario": "GNEW_EOBS",
                "outer_fold": str(outer_fold),
                "inner_fold": "1",
            }
        )
        metadata = (
            tmp_path
            / source_runs
            / state_id
            / screen.SOURCE_REFERENCE
            / "run_metadata.json"
        )
        metadata.parent.mkdir(parents=True)
        metadata.write_text(
            json.dumps({"status": "PASS", "seed": 900 + outer_fold}),
            encoding="utf-8",
        )
    pd.DataFrame(rows).to_csv(
        split_dir / "state_registry.tsv", sep="\t", index=False
    )
    monkeypatch.setattr(screen, "PARITY", parity)
    monkeypatch.setattr(screen, "SOURCE_RUNS", source_runs)

    grid = screen.build_grid(tmp_path, protocol())

    assert len(grid) == 5
    assert grid["inner_fold"].eq(1).all()
    assert grid["outer_fold"].tolist() == [1, 2, 3, 4, 5]
    assert grid["seed"].tolist() == [901, 902, 903, 904, 905]
