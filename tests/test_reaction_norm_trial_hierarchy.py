from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from server_training_pipeline.build_final_evaluation_manifests import (
    genotype_expert_support_table,
    nested_genotype_expert_support_table,
)
from server_training_pipeline.final_evaluation_contract import load_protocol
from server_training_pipeline.freeze_reaction_norm_routed_hierarchy_selection import (
    source_decision_checks,
)
from server_training_pipeline.prepare_reaction_norm_trial_hierarchy_screen import (
    write_freeze,
)
from server_training_pipeline.summarize_reaction_norm_trial_hierarchy_screen import (
    scenario_noninferiority_pass,
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


def test_cross_scenario_guard_is_frozen_and_bound_to_confirmation() -> None:
    protocol = json.loads(
        (
            ROOT
            / "server_training_pipeline"
            / "reaction_norm_trial_hierarchy_cross_scenario_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["status"] == "frozen_before_inner_validation"
    assert protocol["outer_test_metrics_read"] is False
    assert protocol["final_holdout_outcomes_read"] is False
    assert set(protocol["selection_scenarios"]) == {
        "unseen_environments",
        "unseen_genotypes_and_environments",
        "temporal_holdout",
        "country_holdout",
    }
    assert protocol["source_confirmation"]["selected_candidate"] == (
        "trial_and_environment_intercepts"
    )
    assert set(protocol["acceptance"]["required_scenarios"]) == set(
        protocol["selection_scenarios"]
    )


def test_cross_scenario_guard_requires_every_scenario_to_be_noninferior() -> None:
    scenarios = [
        "unseen_environments",
        "unseen_genotypes_and_environments",
        "temporal_holdout",
        "country_holdout",
    ]
    summary = pd.DataFrame(
        {
            "candidate": ["candidate"] * 4,
            "scenario": scenarios,
            "relative_normalized_rmse_gain_mean": [0.02, 0.01, 0.0, -0.005],
            "normalized_rmse_win_rate": [1.0, 0.67, 0.5, 0.34],
            "pearson_gain_mean": [0.01, 0.0, -0.005, -0.009],
            "calibration_error_delta_mean": [-0.01, 0.0, 0.005, 0.009],
        }
    )
    acceptance = {
        "required_scenarios": scenarios,
        "maximum_scenario_relative_nrmse_loss": 0.01,
        "minimum_scenario_fold_win_rate": 1.0 / 3.0,
        "maximum_scenario_pearson_drop": 0.01,
        "maximum_scenario_calibration_error_increase": 0.01,
    }
    assert scenario_noninferiority_pass(summary, "candidate", acceptance)
    degraded = summary.copy()
    degraded.loc[degraded["scenario"].eq("temporal_holdout"), "pearson_gain_mean"] = -0.02
    assert not scenario_noninferiority_pass(degraded, "candidate", acceptance)
    assert not scenario_noninferiority_pass(
        summary[summary["scenario"].ne("country_holdout")],
        "candidate",
        acceptance,
    )


def test_routed_outer_protocol_uses_hierarchy_only_for_known_environments() -> None:
    protocol = json.loads(
        (
            ROOT
            / "server_training_pipeline"
            / "reaction_norm_routed_hierarchy_outer_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["status"] == "frozen_after_inner_validation_before_outer_test"
    assert protocol["outer_test_metrics_read_at_freeze"] is False
    assert protocol["outer_test_metrics_used_for_routing"] is False
    assert protocol["final_holdout_outcomes_read"] is False
    routes = {
        scenario: value["trial_hierarchy_candidate"]
        for scenario, value in protocol["scenario_routes"].items()
    }
    assert routes == {
        "unseen_environments": "current_reaction_norm",
        "unseen_genotypes": "trial_and_environment_intercepts",
        "unseen_genotypes_and_environments": "current_reaction_norm",
        "temporal_holdout": "current_reaction_norm",
        "country_holdout": "current_reaction_norm",
    }
    assert protocol["model_contract"]["future_environment_route"] == (
        "current_reaction_norm"
    )
    assert protocol["model_contract"]["outer_model_fit_count"] == 69


def test_routed_outer_protocol_binds_canonical_git_text_bytes() -> None:
    protocol = json.loads(
        (
            ROOT
            / "server_training_pipeline"
            / "reaction_norm_routed_hierarchy_outer_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    implementations = {
        "hierarchy_trainer_sha256": (
            ROOT
            / "server_training_pipeline"
            / "train_multitrait_reaction_norm_trial_hierarchy_tf.py"
        ),
        "base_trainer_sha256": (
            ROOT
            / "server_training_pipeline"
            / "train_multitrait_reaction_norm_tf.py"
        ),
        "factorization_sha256": (
            ROOT / "server_training_pipeline" / "kernel_factorization.py"
        ),
        "run_verifier_sha256": (
            ROOT / "server_training_pipeline" / "verify_reaction_norm_run.py"
        ),
        "outer_verifier_sha256": (
            ROOT
            / "server_training_pipeline"
            / "verify_reaction_norm_outer_evaluation.py"
        ),
    }
    for label, path in implementations.items():
        canonical_bytes = path.read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(canonical_bytes).hexdigest() == (
            protocol["implementation"][label]
        )


def test_routed_source_decision_requires_exact_artifact_hashes(tmp_path: Path) -> None:
    artifact = tmp_path / "decision.tsv"
    artifact.write_text("decision\naccepted\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "status": "PASS",
                "phase": "confirmation",
                "selected_candidate": "winner",
                "paired_inner_fold_count": 3,
                "hierarchy_protocol_sha256": "protocol",
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
            }
        ),
        encoding="utf-8",
    )
    specification = {
        "phase": "confirmation",
        "selected_candidate": "winner",
        "paired_inner_fold_count": 3,
        "protocol_sha256": "protocol",
        "required_artifact_sha256": {"decision.tsv": digest},
    }
    checks, paths = source_decision_checks(provenance, specification)
    assert all(checks.values())
    assert set(paths) == {provenance, artifact}
    artifact.write_text("changed\n", encoding="utf-8")
    checks, _ = source_decision_checks(provenance, specification)
    assert checks["artifact_hashes"] is False


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
    assert "--hyperparameter-label explicit_E_REACTION_NORM_V1" in source
    assert '--hyperparameter-label "$candidate"' not in source
    assert "outer_evaluation" not in source
    assert "final-holdout" in source


def test_cross_scenario_launcher_refits_environment_and_remains_inner_only() -> None:
    source = (
        ROOT
        / "scripts"
        / "run_reaction_norm_trial_hierarchy_cross_scenario_guard.sh"
    ).read_text(encoding="utf-8")
    assert "build_environment_component_kernels.py" in source
    assert "build_reaction_norm_environment_v1" in source
    assert "--fit-environment-ids \"$OUTER_ENV_IDS\"" in source
    assert "--only-kernel" in source
    assert "--evaluation-stage inner_selection" in source
    assert "--evaluation-stage outer_evaluation" not in source
    assert "--hierarchy-confirmation-provenance" in source
    assert "trial_hierarchy_inner_cross_" in source


def test_routed_outer_launcher_freezes_route_and_keeps_holdout_sealed() -> None:
    suite = (
        ROOT
        / "scripts"
        / "run_reaction_norm_routed_hierarchy_outer_suite.sh"
    ).read_text(encoding="utf-8")
    fold = (
        ROOT / "scripts" / "run_multitrait_reaction_norm_outer_fold.sh"
    ).read_text(encoding="utf-8")
    assert "freeze_reaction_norm_routed_hierarchy_selection" in suite
    assert "REACTION_TRIAL_HIERARCHY_PROTOCOL" in suite
    assert "verify_reaction_norm_outer_evaluation" in suite
    assert "final holdout remains sealed" in suite
    assert "REACTION_TRIAL_HIERARCHY_PROTOCOL" in fold
    assert 'RUN_CANDIDATE="$SELECTED_CANDIDATE"' in fold
    assert "train_multitrait_reaction_norm_trial_hierarchy_tf" in fold


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
