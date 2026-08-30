from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from server_training_pipeline.stage1_v2_phase6_trait_balance_v1 import (
    apply_trait_mass_policy,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_trait_balance_screen_protocol_v1.json"
)
PLAN = (
    ROOT
    / "server_training_pipeline"
    / "stage1_v2_phase6_post_hierarchy_screen_plan_v1.json"
)
TRAINER = (
    ROOT
    / "server_training_pipeline"
    / "train_stage1_v2_phase6_trait_balance_tf.py"
)
RUNNER = ROOT / "scripts" / "v2" / "run_stage1_v2_phase6_trait_balance_screen.py"


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def frame() -> pd.DataFrame:
    traits = [*protocol()["primary_traits"], *protocol()["exploratory_traits"]]
    rows = []
    for index, trait in enumerate(traits):
        for repeat in range(index + 2):
            rows.append(
                {
                    "selection_role": "TRAINING",
                    "trait": trait,
                    "loss_weight": float(repeat + 1),
                }
            )
        rows.append(
            {
                "selection_role": "TRAINING",
                "trait": trait,
                "loss_weight": 0.0,
            }
        )
        rows.append(
            {
                "selection_role": "INNER_VALIDATION",
                "trait": trait,
                "loss_weight": 7.0 + index,
            }
        )
    return pd.DataFrame(rows)


def test_equal_trait_mass_preserves_total_within_trait_weights_and_masks() -> None:
    source = frame()
    value = protocol()
    candidate = "equal_seven_trait_mass"
    balanced, diagnostics = apply_trait_mass_policy(
        source,
        candidate=candidate,
        candidate_policy=value["candidates"][candidate],
        primary_traits=value["primary_traits"],
        exploratory_traits=value["exploratory_traits"],
    )
    training = source["selection_role"].eq("TRAINING")
    assert np.isclose(
        balanced.loc[training, "loss_weight"].sum(),
        source.loc[training, "loss_weight"].sum(),
    )
    assert np.allclose(diagnostics["final_loss_weight_share"], 1.0 / 7.0)
    assert balanced.loc[training & source["loss_weight"].eq(0), "loss_weight"].eq(0).all()
    assert np.array_equal(
        balanced.loc[~training, "loss_weight"].to_numpy(),
        source.loc[~training, "loss_weight"].to_numpy(),
    )


def test_primary_equal_exploratory_quarter_mass_is_preregistered() -> None:
    source = frame()
    value = protocol()
    candidate = "primary_equal_exploratory_quarter_mass"
    _, diagnostics = apply_trait_mass_policy(
        source,
        candidate=candidate,
        candidate_policy=value["candidates"][candidate],
        primary_traits=value["primary_traits"],
        exploratory_traits=value["exploratory_traits"],
    )
    observed = diagnostics.set_index("trait_name_canonical")["final_loss_weight_share"]
    for trait in value["primary_traits"]:
        assert np.isclose(observed.loc[trait], 1.0 / 5.5)
    for trait in value["exploratory_traits"]:
        assert np.isclose(observed.loc[trait], 0.25 / 5.5)


def test_screen_changes_only_loss_and_keeps_products_separate() -> None:
    value = protocol()
    assert value["architecture_policy"]["only_mutable_component"] == (
        "training_loss_trait_mass"
    )
    assert value["fixed_configuration"]["batch_size"] == 8192
    assert list(value["calibration_candidates"]) == [
        "hierarchy_test_weight_environment_oof_huber_v2"
    ]
    assert value["calibration_candidates"][
        "hierarchy_test_weight_environment_oof_huber_v2"
    ]["method"] == "environment_oof_huber"
    assert value["phase_1_scope"]["state_count"] == 5
    assert value["phase_1_scope"]["new_candidate_fit_count"] == 10
    assert value["product_policy"]["projection_compatible_product"] == (
        "unchanged_and_separate"
    )
    assert value["outer_test_metrics_read"] is False
    trainer = TRAINER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "apply_trait_mass_policy" in trainer
    assert "environment_oof_huber" in trainer
    assert '"outer_evaluation_allowed": False' in runner
    freeze = (
        ROOT
        / "scripts"
        / "v2"
        / "freeze_stage1_v2_phase6_trait_balance_screen.py"
    ).read_text(encoding="utf-8")
    for implementation in (
        "trainer",
        "loss_helper",
        "calibration_helper",
        "calibration_trainer",
        "remediation_helper",
        "remediation_trainer",
        "factor_builder",
        "trainer_interface",
        "runner",
    ):
        assert f'"{implementation}": sha256_file(paths["{implementation}"])' in freeze


def test_follow_on_plan_orders_candidates_before_outer_protocol() -> None:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    names = [entry["name"] for entry in value["ordered_gates"]]
    assert names == [
        "trait_balanced_loss",
        "trait_specific_reaction_heads",
        "empirical_bayes_reml_or_bounded_megalmm_hierarchy",
        "phenotype_safe_phenology_random_regression",
        "projection_residual_comparators",
        "nonnegative_cross_fitted_stacking",
    ]
    assert value["products"]["historical_known_environment"][
        "identifier_intercepts_allowed"
    ] is True
    assert value["products"]["projection_compatible"][
        "identifier_intercepts_allowed"
    ] is False
    assert value["outer_protocol_policy"][
        "create_only_after_all_entered_inner_gates_are_terminal"
    ] is True
