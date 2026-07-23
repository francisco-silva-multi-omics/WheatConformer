from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from server_training_pipeline.summarize_reaction_norm_screen import (
    REFERENCE,
    prediction_file,
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
    assert "single_step_H_inner_screen_v3_canonical_runs" in text
    assert "outer_evaluation" not in text
