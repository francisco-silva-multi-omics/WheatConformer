from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from server_training_pipeline.summarize_reaction_norm_screen import (
    MATCHED_REFERENCE_LABEL,
    REFERENCE,
    prediction_file,
    reference_candidate,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_reaction_protocol_declares_exact_model_contract() -> None:
    protocol = json.loads(
        (ROOT / "server_training_pipeline/reaction_norm_protocol_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["status"] == "frozen_before_inner_validation"
    assert set(protocol["required_kernels"]) == {
        "K_A_CANONICAL_V3",
        "K_E_GEO",
        "K_E_WEATHER",
        "K_E_STRESS",
        "K_E_MGMT",
        "K_E_TGW_V2",
    }
    assert protocol["model_contract"]["trait_covariance_fit_partition"] == "inner_training_only"
    assert protocol["model_contract"]["outer_test_available_during_selection"] is False
    assert {candidate["trait_covariance_shrinkage"] for candidate in protocol["candidates"]} == {
        0.25,
        1.0,
    }


def test_trait_covariance_is_train_only_shrunk_and_psd() -> None:
    from server_training_pipeline.train_multitrait_reaction_norm_tf import (
        estimate_trait_covariance,
    )

    rows = []
    for index, value in enumerate([-2.0, -1.0, 1.0, 2.0]):
        rows.extend(
            [
                {
                    "genotype_id": f"G{index}",
                    "environment_id": "E1",
                    "trait_name_canonical": "T1",
                    "y_scaled": value,
                },
                {
                    "genotype_id": f"G{index}",
                    "environment_id": "E1",
                    "trait_name_canonical": "T2",
                    "y_scaled": value * 2.0,
                },
            ]
        )
    train = pd.DataFrame(rows)
    identity, identity_root, counts = estimate_trait_covariance(
        train, ["T1", "T2"], shrinkage=1.0, minimum_pairs=3
    )
    correlated, correlated_root, _ = estimate_trait_covariance(
        train, ["T1", "T2"], shrinkage=0.25, minimum_pairs=3
    )
    np.testing.assert_allclose(identity, np.eye(2), atol=1e-7)
    np.testing.assert_allclose(identity_root @ identity_root.T, identity, atol=1e-7)
    np.testing.assert_allclose(
        correlated_root @ correlated_root.T, correlated, atol=1e-7
    )
    assert counts[0, 1] == 4
    assert correlated[0, 1] > 0.5
    assert np.linalg.eigvalsh(correlated).min() > 0


def test_reaction_projection_is_deterministic_and_component_specific() -> None:
    from server_training_pipeline.train_multitrait_reaction_norm_tf import (
        deterministic_sign_projection,
    )

    first = deterministic_sign_projection(8, 4, 61001, "K_AxK_E_GEO")
    repeated = deterministic_sign_projection(8, 4, 61001, "K_AxK_E_GEO")
    other = deterministic_sign_projection(8, 4, 61001, "K_AxK_E_WEATHER")
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)
    assert set(np.unique(first)) == {-1.0, 1.0}


def test_reaction_model_has_trait_specific_residuals_and_gxe_components() -> None:
    import tensorflow as tf

    from server_training_pipeline.train_multitrait_reaction_norm_tf import (
        MultiTraitReactionNorm,
    )

    specs = [
        {
            "kernel": "K_A_CANONICAL_V3",
            "axis": "genotype",
            "eligible_traits": "*",
            "interaction_enabled": True,
        },
        {
            "kernel": "K_E_TGW_V2",
            "axis": "environment",
            "eligible_traits": "T2",
            "interaction_enabled": True,
        },
    ]
    factors = [
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32),
    ]
    model = MultiTraitReactionNorm(
        specs,
        factors,
        ["T1", "T2"],
        "K_A_CANONICAL_V3",
        np.eye(2, dtype=np.float32),
        reaction_rank=4,
        ridge_penalty=1e-4,
        residual_scale_floor=0.05,
        initialization_seed=61001,
    )
    prediction = model(
        (
            tf.constant([[0, 0], [1, 1]], dtype=tf.int32),
            tf.constant([0, 1], dtype=tf.int32),
        )
    ).numpy()
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()
    assert np.all(model.residual_scales().numpy() > 0.05)
    components = model.component_variance_frame()
    assert set(components["component_type"]) == {"main", "reaction"}
    assert "K_A_CANONICAL_V3xK_E_TGW_V2" in set(components["component"])


def test_reaction_acceptance_requires_gain_stability_pearson_and_calibration() -> None:
    paired = pd.DataFrame(
        [
            {
                "architecture": architecture,
                "outer_fold": fold,
                "inner_fold": 0,
                "val_normalized_rmse": rmse,
                "val_pearson": pearson,
                "val_nrmse_gain_vs_train_mean": 0.2,
                "relative_nrmse_gain_vs_reference": relative_gain,
                "nrmse_gain_vs_reference": gain,
                "pearson_gain_vs_reference": pearson_gain,
                "calibration_error_delta_vs_reference": calibration,
            }
            for fold in range(3)
            for architecture, rmse, pearson, relative_gain, gain, pearson_gain, calibration in [
                ("safe", 0.68, 0.61, 0.02, 0.02, 0.01, -0.01),
                ("miscalibrated", 0.68, 0.61, 0.02, 0.02, 0.01, 0.01),
            ]
        ]
    )
    reference_runs = pd.DataFrame(
        [
            {
                "architecture": REFERENCE,
                "outer_fold": fold,
                "inner_fold": 0,
                "val_normalized_rmse": 0.70,
                "val_pearson": 0.60,
                "val_nrmse_gain_vs_train_mean": 0.18,
            }
            for fold in range(3)
        ]
    )
    result = summarize(
        reference_runs,
        paired,
        minimum_relative_gain=0.01,
        minimum_win_rate=2.0 / 3.0,
        maximum_pearson_drop=0.005,
        maximum_calibration_increase=0.0,
    ).set_index("architecture")
    assert result.loc["safe", "quantitative_model_decision"] == (
        "advance_as_primary_quantitative_model"
    )
    assert result.loc["miscalibrated", "quantitative_model_decision"] == (
        "retain_as_interpretable_mixed_baseline"
    )
    assert result.loc[REFERENCE, "quantitative_model_decision"] == "nonlinear_reference"


def test_prediction_file_prefers_parquet_when_tsv_mirror_exists(tmp_path: Path) -> None:
    parquet = tmp_path / "run_predictions.parquet"
    mirror = tmp_path / "run_predictions.tsv.gz"
    parquet.write_bytes(b"parquet")
    mirror.write_bytes(b"mirror")
    assert prediction_file(tmp_path) == parquet


def test_only_exact_matched_nonlinear_reference_is_accepted() -> None:
    assert reference_candidate(MATCHED_REFERENCE_LABEL)
    assert not reference_candidate("pedigree_environment_only_cfg7b8af9c8b5")


def test_prepare_reaction_inputs_is_phenotype_blind(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    np.save(canonical / "K_A_CANONICAL_V3.npy", np.eye(2, dtype=np.float32))
    pd.DataFrame({"sample_id": ["G1", "G2"], "compact_kernel_index": [0, 1]}).to_csv(
        canonical / "K_A_CANONICAL_V3_sample_order.tsv", sep="\t", index=False
    )
    (canonical / "canonical_pedigree_decision.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "protocol_version": "canonical_trial_pedigree_v3_verified_recovery_overlay",
                "phenotype_values_read": False,
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
            }
        ),
        encoding="utf-8",
    )
    (canonical / "canonical_pedigree_artifacts.sha256").write_text(
        "certified\n", encoding="utf-8"
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "status": "frozen_before_inner_validation",
                "protocol_version": "test_reaction",
                "genotype_kernel": "K_A_CANONICAL_V3",
                "phenotype_values_read": False,
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
                "training": {"max_rank_genotype": 2},
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "server_training_pipeline.prepare_reaction_norm_inputs",
            "--root",
            str(tmp_path),
            "--protocol",
            str(protocol),
            "--canonical-dir",
            str(canonical),
            "--out-dir",
            "prepared",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    provenance = json.loads(
        (tmp_path / "prepared/reaction_norm_input_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = pd.read_csv(
        tmp_path / "prepared/reaction_norm_genotype_manifest.tsv", sep="\t"
    )
    assert provenance["status"] == "PASS"
    assert provenance["phenotype_values_read"] is False
    assert manifest["kernel"].tolist() == ["K_A_CANONICAL_V3"]
    assert not manifest["enabled_default"].astype(bool).any()


def test_runner_is_inner_only_and_prepares_exact_kernel_subset() -> None:
    text = (ROOT / "scripts/run_multitrait_reaction_norm_inner_screen.sh").read_text(
        encoding="utf-8"
    )
    assert "--only-kernel" in text
    assert "--evaluation-stage inner_selection" in text
    assert "train_multitrait_reaction_norm_tf" in text
    assert "reaction_norm_matched_nonlinear_reference_v1_runs" in text
    assert "TRAIN matched nonlinear reference" in text
    assert "outer_evaluation" not in text


def test_environment_screen_runner_is_inner_only_and_blocks_old_outer_suite() -> None:
    screen = (
        ROOT / "scripts/run_reaction_norm_environment_inner_screen.sh"
    ).read_text(encoding="utf-8")
    outer = (
        ROOT / "scripts/run_multitrait_reaction_norm_outer_suite.sh"
    ).read_text(encoding="utf-8")
    assert "--evaluation-stage inner_selection" in screen
    assert "summarize_reaction_norm_environment_screen" in screen
    assert "E_REACTION_NORM_V1_certification.json" in screen
    assert "outer_evaluation" not in screen
    assert "STOP: reaction-norm outer evaluation is blocked" in outer


def test_outer_protocol_is_blocked_until_environment_architecture_is_selected() -> None:
    import hashlib

    protocol = json.loads(
        (
            ROOT
            / "server_training_pipeline/reaction_norm_outer_evaluation_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["status"] == "blocked_pending_environment_architecture_selection"
    assert "E_REACTION_NORM_V1" in protocol["blocked_reason"]
    assert protocol["selected_candidate"] == "reaction_norm_identity_covariance"
    assert protocol["model_contract"]["no_further_hyperparameter_selection"] is True
    assert protocol["model_contract"]["final_holdout_available"] is False
    assert protocol["trait_reporting_policy"]["ABOVE_GROUND_BIOMASS"] == (
        "exploratory_not_improved_in_inner_validation"
    )
    assert protocol["scenarios"] == {
        "unseen_environments": 5,
        "unseen_genotypes": 5,
        "unseen_genotypes_and_environments": 5,
        "temporal_holdout": 3,
        "country_holdout": 5,
    }
    assert protocol["outer_member_policy"]["member_count"] == 3
    assert protocol["inner_reaction_protocol_sha256"] == hashlib.sha256(
        (ROOT / "server_training_pipeline/reaction_norm_protocol_v1.json").read_bytes()
    ).hexdigest()
    assert protocol["evaluation_protocol_sha256"] == hashlib.sha256(
        (ROOT / "server_training_pipeline/final_evaluation_protocol.json").read_bytes()
    ).hexdigest()
    assert protocol["outer_member_policy"]["support_policy_sha256"] == hashlib.sha256(
        (
            ROOT / "server_training_pipeline/outer_ensemble_support_policy.json"
        ).read_bytes()
    ).hexdigest()


def test_environment_protocol_freezes_two_arm_inner_only_comparison() -> None:
    protocol = json.loads(
        (
            ROOT
            / "server_training_pipeline/reaction_norm_environment_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["status"] == "frozen_before_inner_validation"
    assert protocol["outer_test_metrics_read"] is False
    candidates = {value["name"]: value for value in protocol["candidates"]}
    assert set(candidates) == {
        "current_corrected_generic_environment",
        "explicit_E_REACTION_NORM_V1",
    }
    assert candidates["current_corrected_generic_environment"][
        "reaction_feature_mode"
    ] == "kernel_product"
    explicit = candidates["explicit_E_REACTION_NORM_V1"]
    assert explicit["reaction_feature_mode"] == "explicit_environment_axes"
    assert "K_E_REACTION_NORM_V1" in explicit["required_kernels"]
    assert explicit["kernel_interaction_allowlist"] == ["K_E_TGW_V2"]
    assert protocol["trait_slope_penalty_multiplier"]["TEST_WEIGHT"] == 4.0


def test_fold_local_environment_scaling_does_not_use_held_out_values() -> None:
    from server_training_pipeline.build_reaction_norm_environment_v1 import (
        standardize_fold_local,
    )

    index = pd.Index(["E1", "E2", "E3"])
    first = pd.DataFrame({"water": [1.0, 3.0, 1000.0]}, index=index)
    second = pd.DataFrame({"water": [1.0, 3.0, -9000.0]}, index=index)
    fit = pd.Index(["E1", "E2"])
    z_first, scaling_first, _ = standardize_fold_local(first, fit)
    z_second, scaling_second, _ = standardize_fold_local(second, fit)
    pd.testing.assert_frame_equal(
        scaling_first.reset_index(drop=True), scaling_second.reset_index(drop=True)
    )
    np.testing.assert_allclose(
        z_first.loc[fit, "water"], z_second.loc[fit, "water"], atol=1e-7
    )
    np.testing.assert_allclose(z_first.loc[fit, "water"].mean(), 0.0, atol=1e-7)
    np.testing.assert_allclose(z_first.loc[fit, "water"].std(ddof=0), 1.0, atol=1e-7)


def test_reaction_model_supports_trait_masked_explicit_environment_axes() -> None:
    import tensorflow as tf

    from server_training_pipeline.train_multitrait_reaction_norm_tf import (
        MultiTraitReactionNorm,
    )

    specs = [
        {
            "kernel": "K_A_CANONICAL_V3",
            "axis": "genotype",
            "eligible_traits": "*",
            "interaction_enabled": True,
        },
        {
            "kernel": "K_E_TGW_V2",
            "axis": "environment",
            "eligible_traits": "T2",
            "interaction_enabled": True,
        },
    ]
    factors = [
        np.eye(2, dtype=np.float32),
        np.eye(2, dtype=np.float32),
    ]
    design = np.asarray([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    eligibility = np.asarray([[True, True], [True, False], [False, True]])
    model = MultiTraitReactionNorm(
        specs,
        factors,
        ["T1", "T2"],
        "K_A_CANONICAL_V3",
        np.eye(2, dtype=np.float32),
        reaction_rank=2,
        ridge_penalty=1e-4,
        residual_scale_floor=0.05,
        initialization_seed=61001,
        reaction_feature_mode="explicit_environment_axes",
        kernel_interaction_allowlist={"K_E_TGW_V2"},
        environment_design=design,
        environment_trait_eligibility=eligibility,
        trait_slope_penalty_multiplier=np.asarray([1.0, 4.0], dtype=np.float32),
    )
    prediction = model(
        (
            tf.constant([[0, 0], [1, 1]], dtype=tf.int32),
            tf.constant([0, 1], dtype=tf.int32),
            tf.constant([0, 1], dtype=tf.int32),
        )
    ).numpy()
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()
    components = model.component_variance_frame()
    assert "K_A_CANONICAL_V3xE_REACTION_NORM_V1" in set(components["component"])
    assert np.isfinite(float(model.regularization_loss().numpy()))


def test_freeze_reaction_selection_records_biomass_caveat_without_trait_removal(
    tmp_path: Path,
) -> None:
    summary_dir = tmp_path / "summary"
    models_dir = tmp_path / "models"
    reference_dir = tmp_path / "references"
    summary_dir.mkdir()
    models_dir.mkdir()
    reference_dir.mkdir()
    kernels = ["K_A_CANONICAL_V3"]
    traits = ["DAYS_TO_HEADING", "ABOVE_GROUND_BIOMASS"]
    candidates = [
        {
            "name": "reaction_norm_identity_covariance",
            "trait_covariance_shrinkage": 1.0,
            "reaction_rank": 2,
            "ridge_penalty": 0.1,
        },
        {
            "name": "reaction_norm_correlated_traits",
            "trait_covariance_shrinkage": 0.25,
            "reaction_rank": 2,
            "ridge_penalty": 0.1,
        },
    ]
    reaction_protocol = {
        "protocol_version": "test_inner",
        "status": "frozen_before_inner_validation",
        "scenario": "unseen_genotypes",
        "required_kernels": kernels,
        "traits": traits,
        "candidates": candidates,
        "selection": {
            "minimum_relative_nrmse_gain_vs_nonlinear_reference": 0.01,
            "minimum_fold_win_rate": 2.0 / 3.0,
            "maximum_mean_pearson_drop": 0.005,
            "maximum_mean_calibration_error_increase": 0.0,
        },
    }
    reaction_path = tmp_path / "reaction.json"
    reaction_path.write_text(json.dumps(reaction_protocol), encoding="utf-8")
    import hashlib

    outer_protocol = {
        "protocol_version": "test_outer",
        "status": "frozen_after_inner_validation_before_outer_test",
        "selected_candidate": "reaction_norm_identity_covariance",
        "selected_model_label": "frozen_identity",
        "selected_configuration": {},
        "inner_reaction_protocol_sha256": hashlib.sha256(
            reaction_path.read_bytes()
        ).hexdigest(),
        "trait_reporting_policy": {
            "ABOVE_GROUND_BIOMASS": "exploratory_not_improved_in_inner_validation",
            "default": "primary_quantitative_result",
        },
    }
    outer_path = tmp_path / "outer.json"
    outer_path.write_text(json.dumps(outer_protocol), encoding="utf-8")
    (summary_dir / "reaction_norm_inner_screen_provenance.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "selection_data": "inner_validation_metrics_only",
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
                "selected_reaction_candidate": "reaction_norm_identity_covariance",
                "outer_folds": [0],
                "inner_fold_count": 1,
                "reaction_run_count": 2,
                "reference_run_count": 1,
                "matched_seed_status": "pass",
                "matched_validation_observation_status": "pass",
                "matched_common_kernel_identity_status": "pass",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "architecture": "reaction_norm_identity_covariance",
                "paired_inner_folds": 1,
                "outer_folds": 1,
                "relative_nrmse_gain_vs_reference_mean": 0.02,
                "nrmse_win_rate_vs_reference": 1.0,
                "pearson_gain_vs_reference_mean": 0.01,
                "calibration_error_delta_vs_reference_mean": -0.01,
                "quantitative_model_decision": "advance_as_primary_quantitative_model",
            },
            {
                "architecture": "reaction_norm_correlated_traits",
                "paired_inner_folds": 1,
                "outer_folds": 1,
                "relative_nrmse_gain_vs_reference_mean": 0.0,
                "nrmse_win_rate_vs_reference": 0.0,
                "pearson_gain_vs_reference_mean": 0.0,
                "calibration_error_delta_vs_reference_mean": 0.0,
                "quantitative_model_decision": "retain_as_interpretable_mixed_baseline",
            },
        ]
    ).to_csv(
        summary_dir / "reaction_norm_inner_screen_summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        [
            {
                "architecture": "reaction_norm_identity_covariance",
                "outer_fold": 0,
                "inner_fold": 0,
                "trait_name_canonical": "DAYS_TO_HEADING",
                "normalized_rmse_candidate": 0.7,
                "normalized_rmse_reference": 0.8,
                "nrmse_gain_vs_reference": 0.1,
                "pearson_candidate": 0.7,
                "pearson_reference": 0.6,
                "pearson_gain_vs_reference": 0.1,
                "calibration_error_delta_vs_reference": -0.1,
            },
            {
                "architecture": "reaction_norm_identity_covariance",
                "outer_fold": 0,
                "inner_fold": 0,
                "trait_name_canonical": "ABOVE_GROUND_BIOMASS",
                "normalized_rmse_candidate": 0.9,
                "normalized_rmse_reference": 0.8,
                "nrmse_gain_vs_reference": -0.1,
                "pearson_candidate": 0.5,
                "pearson_reference": 0.6,
                "pearson_gain_vs_reference": -0.1,
                "calibration_error_delta_vs_reference": -0.1,
            },
        ]
    ).to_csv(
        summary_dir / "reaction_norm_inner_screen_trait_metrics.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame({"x": [1]}).to_csv(
        summary_dir / "reaction_norm_inner_screen_runs.tsv", sep="\t", index=False
    )
    pd.DataFrame({"x": [1]}).to_csv(
        summary_dir / "reaction_norm_inner_screen_paired_metrics.tsv",
        sep="\t",
        index=False,
    )

    def write_run(run_dir: Path, label: str, reaction_run: bool) -> None:
        run_dir.mkdir()
        metadata = {
            "evaluation_stage": "inner_selection",
            "external_split": {
                "scenario": "unseen_genotypes",
                "outer_fold": 0,
                "inner_fold": 0,
            },
            "hyperparameter_label": label,
            "active_kernels": kernels,
            "trainer_sha256": "reaction" if reaction_run else "reference",
        }
        if reaction_run:
            metadata.update(
                {
                    "status": "PASS",
                    "outer_test_metrics_read": False,
                    "final_holdout_outcomes_read": False,
                    "phenotype_preprocessing": {"outer_test_outcomes_used": False},
                }
            )
        (run_dir / "x_run_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        pd.DataFrame({"split": ["val"]}).to_csv(
            run_dir / "x_macro_metrics.tsv", sep="\t", index=False
        )
        pd.DataFrame({"split": ["val"]}).to_csv(
            run_dir / "x_trait_metrics.tsv", sep="\t", index=False
        )
        pd.DataFrame({"split": ["val"], "y_pred": [0.0]}).to_csv(
            run_dir / "x_predictions.tsv.gz", sep="\t", index=False
        )

    for candidate in candidates:
        write_run(
            models_dir
            / f"reaction_inner_unseen_genotypes_outer0_{candidate['name']}_inner0",
            candidate["name"],
            True,
        )
    write_run(
        reference_dir / "reaction_reference_inner_unseen_genotypes_outer0_inner0",
        "nonlinear_canonical_v3_matched_reference",
        False,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "server_training_pipeline.freeze_reaction_norm_selection",
            "--root",
            str(tmp_path),
            "--summary-dir",
            "summary",
            "--models-dir",
            "models",
            "--reference-models-dir",
            "references",
            "--reaction-protocol",
            "reaction.json",
            "--outer-protocol",
            "outer.json",
            "--expected-outer-folds",
            "1",
            "--expected-inner-folds",
            "1",
            "--out-dir",
            "frozen",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    lock = json.loads(
        (tmp_path / "frozen/reaction_norm_selection_lock.json").read_text(
            encoding="utf-8"
        )
    )
    reporting = pd.read_csv(
        tmp_path / "frozen/reaction_norm_selected_trait_reporting.tsv", sep="\t"
    )
    assert lock["status"] == "PASS"
    assert lock["trait_architecture_preserved"] is True
    assert lock["outer_evaluation_allowed"] is True
    biomass = reporting[
        reporting["trait_name_canonical"].eq("ABOVE_GROUND_BIOMASS")
    ].iloc[0]
    assert biomass["reporting_class"] == (
        "exploratory_not_improved_in_inner_validation"
    )


def test_outer_runners_do_not_reopen_inner_selection() -> None:
    fold = (ROOT / "scripts/run_multitrait_reaction_norm_outer_fold.sh").read_text(
        encoding="utf-8"
    )
    suite = (ROOT / "scripts/run_multitrait_reaction_norm_outer_suite.sh").read_text(
        encoding="utf-8"
    )
    assert "--evaluation-stage outer_evaluation" in fold
    assert "--reaction-selection-lock" in fold
    assert "--outer-evaluation-protocol" in fold
    assert "summarize_reaction_norm_screen" not in fold
    assert "verify_reaction_norm_outer_evaluation" in suite
    assert "final_holdout_environment_ids.tsv" in suite
