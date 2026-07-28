from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from server_training_pipeline.build_final_evaluation_manifests import (
    genotype_expert_support_table,
    nested_genotype_expert_support_table,
)
from server_training_pipeline.final_evaluation_contract import load_protocol
from server_training_pipeline.prepare_reaction_norm_trial_hierarchy_screen import (
    write_freeze,
)
from server_training_pipeline.trial_hierarchy import (
    fit_hierarchy_support,
    hierarchy_indices,
    support_matrix,
    trial_ids,
)


ROOT = Path(__file__).resolve().parents[1]


def hierarchy_candidate(
    *, trial: bool = True, environment: bool = True
) -> dict[str, object]:
    return {
        "name": "test",
        "trial_effect_enabled": trial,
        "environment_intercept_enabled": environment,
        "minimum_trial_trait_training_rows": 2 if trial else 0,
        "minimum_environment_trait_training_rows": 2 if environment else 0,
        "trial_penalty": 0.01 if trial else 0.0,
        "environment_penalty": 0.05 if environment else 0.0,
    }


def test_trial_hierarchy_protocol_is_frozen_before_metrics() -> None:
    evaluation = load_protocol(
        ROOT
        / "server_training_pipeline"
        / "reaction_norm_trial_hierarchy_evaluation_protocol_v1.json"
    )
    hierarchy = json.loads(
        (
            ROOT
            / "server_training_pipeline"
            / "reaction_norm_trial_hierarchy_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert evaluation["scenario_assignment_id"] == (
        "reaction_norm_trial_hierarchy_development_v1"
    )
    assert evaluation["final_holdout_assignment_id"] == (
        "multitrait_quantitative_final_v4"
    )
    assert evaluation["protected_genotype_experts"] == []
    assert hierarchy["status"] == "frozen_before_inner_validation"
    assert hierarchy["outer_test_metrics_read"] is False
    assert hierarchy["final_holdout_outcomes_read"] is False
    assert hierarchy["selected_loss"] == "current_trait_row_balanced"


def test_empty_protected_expert_tables_preserve_schema() -> None:
    ledger = pd.DataFrame(
        {
            "panel_sample_id": ["G1", "G2"],
            "env_kernel_id": ["T1|E1", "T1|E2"],
        }
    )
    final = genotype_expert_support_table(
        ledger,
        {"T1|E2"},
        {},
        {
            "minimum_development_unique_genotypes": 1,
            "minimum_development_unique_fraction": 0.5,
            "minimum_development_observation_rows": 1,
            "minimum_holdout_unique_genotypes": 1,
            "minimum_holdout_unique_fraction": 0.5,
            "minimum_holdout_observation_rows": 1,
        },
    )
    assert final.empty
    assert "support_status" in final

    manifest = pd.DataFrame(
        columns=[
            "scenario",
            "outer_fold",
            "inner_fold",
            "axis",
            "partition",
            "entity_id",
        ]
    )
    nested = nested_genotype_expert_support_table(
        ledger,
        manifest,
        {},
        {
            "minimum_train_unique_genotypes": 1,
            "minimum_train_unique_fraction": 0.5,
            "minimum_train_observation_rows": 1,
        },
        {},
        3,
        {},
    )
    assert nested.empty
    assert "support_status" in nested


def test_trial_identity_prefers_metadata_and_falls_back_to_environment() -> None:
    frame = pd.DataFrame(
        {
            "environment_id": ["31ESWYT|52|MALI", "43IBWSN|96|MALI"],
            "trial_name": ["", "CURATED_43IBWSN"],
        }
    )
    assert trial_ids(frame).tolist() == ["31ESWYT", "CURATED_43IBWSN"]


def test_hierarchy_support_is_training_only_and_trait_specific() -> None:
    training = pd.DataFrame(
        {
            "environment_id": [
                "TRIAL_A|E1",
                "TRIAL_A|E1",
                "TRIAL_A|E2",
                "TRIAL_A|E2",
                "TRIAL_B|E3",
            ],
            "trait_name_canonical": ["T1", "T1", "T2", "T2", "T1"],
        }
    )
    maps, support = fit_hierarchy_support(
        training, ["T1", "T2"], hierarchy_candidate()
    )
    assert maps["trial"] == {"TRIAL_A": 0}
    assert maps["environment"] == {"TRIAL_A|E1": 0, "TRIAL_A|E2": 1}
    trial_support = support_matrix(support, "trial", maps["trial"], ["T1", "T2"])
    np.testing.assert_array_equal(trial_support, np.asarray([[True, True]]))
    environment_support = support_matrix(
        support, "environment", maps["environment"], ["T1", "T2"]
    )
    np.testing.assert_array_equal(
        environment_support,
        np.asarray([[True, False], [False, True]]),
    )

    validation = pd.DataFrame(
        {
            "environment_id": ["TRIAL_A|E1", "TRIAL_C|E9"],
            "trait_name_canonical": ["T1", "T1"],
        }
    )
    trial_index, environment_index = hierarchy_indices(validation, maps)
    assert trial_index.tolist() == [0, -1]
    assert environment_index.tolist() == [0, -1]


def test_launcher_is_inner_only_and_uses_exact_kernel_opt_ins() -> None:
    source = (
        ROOT / "scripts" / "run_reaction_norm_trial_hierarchy_inner_screen.sh"
    ).read_text(encoding="utf-8")
    assert "--evaluation-stage inner_selection" in source
    assert "--include-disabled-kernel K_A_CANONICAL_V3" in source
    assert "--include-disabled-kernel K_E_TGW_V2" in source
    assert "--include-disabled-kernel K_E_REACTION_NORM_V1" in source
    assert "outer_evaluation" not in source
    assert "final-holdout" in source


def test_failed_freeze_can_be_replaced_but_pass_freeze_is_immutable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "freeze.json"
    write_freeze(path, {"status": "FAIL", "checks": {"old": False}})
    corrected = {"status": "PASS", "checks": {"corrected": True}}
    write_freeze(path, corrected)
    assert json.loads(path.read_text(encoding="utf-8")) == corrected
    write_freeze(path, corrected)
    with pytest.raises(SystemExit, match="Existing PASS"):
        write_freeze(path, {"status": "PASS", "checks": {"changed": True}})
