from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from build_dth_env_features_v2 import (
    build_window_features,
    feature_export_frame,
    kernel_from_features,
    zscore_with_missing,
)
from fetch_dth_api_weather_windows import build_window_manifest, parse_window
from server_training_pipeline.prepare_multitrait_kernel_registry import (
    load_trait_environment_candidates,
)
from server_training_pipeline.summarize_trait_environment_ablation import (
    Candidate,
    compare_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_custom_sowing_windows_are_parsed_and_materialized() -> None:
    assert parse_window("90:120") == (90, 120)
    assert parse_window("120-150") == (120, 150)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_window("120:90")

    fetch_manifest = pd.DataFrame(
        {
            "env_id": ["e1"],
            "ready_to_fetch": ["TRUE"],
            "sowing_date": ["2020-01-01"],
            "latitude": [10.0],
            "longitude": [-20.0],
        }
    )
    manifest = build_window_manifest(fetch_manifest, windows=[(0, 30), (90, 120)])
    assert manifest["window_label"].tolist() == ["d0_30", "d90_120"]
    assert manifest["window_start_date"].tolist() == ["2020-01-01", "2020-03-31"]


def test_window_features_filter_labels_and_metrics(tmp_path: Path) -> None:
    path = tmp_path / "weather.tsv"
    pd.DataFrame(
        {
            "env_id": ["e1", "e1", "e2", "e2"],
            "window_label": ["d0_30", "d90_120", "d0_30", "d90_120"],
            "fetch_status": ["ok", "ok", "ok", "failed"],
            "gdd_base5_sum": [10, 30, 20, 40],
            "precipitation_total_mm": [1, 3, 2, 4],
        }
    ).to_csv(path, sep="\t", index=False)
    features = build_window_features(
        path,
        pd.Series(["e1", "e2"]),
        allowed_labels={"d0_30"},
        allowed_metrics={"gdd_base5_sum"},
    )
    assert features.columns.tolist() == ["api_d0_30_gdd_base5_sum"]
    assert features.iloc[:, 0].tolist() == [10, 20]


def test_feature_kernel_is_symmetric_unit_diagonal_and_psd() -> None:
    features = pd.DataFrame(
        [[-1.0, 0.5], [0.0, -0.5], [1.0, 0.5]], columns=["a", "b"]
    )
    kernel = kernel_from_features(features)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-7)
    np.testing.assert_allclose(np.diag(kernel), np.ones(3), atol=1e-7)
    assert float(np.linalg.eigvalsh(kernel).min()) >= -1e-6


def test_nonfinite_features_are_imputed_flagged_or_dropped_explicitly() -> None:
    features = pd.DataFrame(
        {
            "partly_finite": [1.0, np.inf, 3.0],
            "all_invalid": [np.inf, -np.inf, np.nan],
            "constant": [2.0, 2.0, 2.0],
        },
        index=["e1", "e2", "e3"],
    )
    standardized, scaling = zscore_with_missing(features)
    assert standardized.columns.tolist() == [
        "partly_finite",
        "partly_finite__missing",
    ]
    assert np.isfinite(standardized.to_numpy()).all()
    status = scaling.set_index("feature")["status"].to_dict()
    assert status == {
        "partly_finite": "retained",
        "all_invalid": "dropped_no_finite_values",
        "constant": "dropped_zero_variance",
    }
    partly = scaling.set_index("feature").loc["partly_finite"]
    assert partly["n_positive_inf"] == 1
    assert partly["n_missing_or_invalid"] == 1
    assert bool(partly["missing_indicator_added"])


def test_kernel_builder_rejects_nonfinite_standardized_input() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        kernel_from_features(pd.DataFrame({"x": [0.0, np.inf]}))


def test_feature_export_consolidates_blocks_and_preserves_environment_order() -> None:
    features = pd.DataFrame({"a": [1.0, 2.0]}, index=["e2", "e1"])
    features["b"] = [3.0, 4.0]
    exported = feature_export_frame(features)
    assert exported.columns.tolist() == ["env_id", "a", "b"]
    assert exported["env_id"].tolist() == ["e2", "e1"]
    np.testing.assert_allclose(exported[["a", "b"]], features[["a", "b"]])


def test_trait_environment_manifest_is_loaded_as_opt_in(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.tsv"
    kernel_path = tmp_path / "K.npy"
    order_path = tmp_path / "order.tsv"
    np.save(kernel_path, np.eye(2, dtype=np.float32))
    pd.DataFrame({"env_id": ["e1", "e2"]}).to_csv(order_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "kernel": ["K_E_DTM_V2"],
            "biological_role": ["fixed_sowing_windows"],
            "kernel_path": [kernel_path.name],
            "order_path": [order_path.name],
            "eligible_traits": ["DAYS_TO_MATURITY"],
            "enabled_default": [False],
            "interaction_enabled": [True],
            "rank": [64],
            "minimum_ledger_coverage": [0.95],
        }
    ).to_csv(manifest_path, sep="\t", index=False)
    candidates = load_trait_environment_candidates(
        manifest_path,
        root=tmp_path,
        base_e_order=pd.DataFrame({"env_id": ["e1", "e2"]}),
    )
    assert len(candidates) == 1
    assert candidates[0]["kernel"] == "K_E_DTM_V2"
    assert candidates[0]["enabled_default"] is False
    assert candidates[0]["eligible_traits"] == "DAYS_TO_MATURITY"


def write_toy_run(
    models_root: Path,
    *,
    variant: str,
    seed: int,
    active_kernels: list[str],
    normalized_rmse: float,
    pearson: float,
    sd_ratio: float,
) -> None:
    run_dir = models_root / f"multitrait_quantitative_{variant}_env_seed{seed}"
    run_dir.mkdir(parents=True)
    prefix = f"multitrait_quantitative_{variant}_env_seed{seed}"
    pd.DataFrame(
        {
            "split": ["val", "val"],
            "coverage_group": ["all", "all"],
            "model": [variant, "train_mean"],
            "trait_name_canonical": ["DAYS_TO_MATURITY", "DAYS_TO_MATURITY"],
            "normalized_rmse": [normalized_rmse, 1.0],
            "pearson": [pearson, np.nan],
            "prediction_sd_ratio": [sd_ratio, 0.0],
        }
    ).to_csv(run_dir / f"{prefix}_trait_metrics.tsv", sep="\t", index=False)
    (run_dir / f"{prefix}_run_metadata.json").write_text(
        json.dumps({"seed": seed, "active_kernels": active_kernels}), encoding="utf-8"
    )
    pd.DataFrame({"leakage_status": ["pass"]}).to_csv(
        run_dir / f"{prefix}_split_leakage_qc.tsv", sep="\t", index=False
    )


def test_ablation_acceptance_requires_isolated_kernel_and_repeated_seed_gain(
    tmp_path: Path,
) -> None:
    for seed in [2026, 2027, 2028, 2029]:
        write_toy_run(
            tmp_path,
            variant="uniform_env_generic",
            seed=seed,
            active_kernels=["K_E_GEO", "K_E_WEATHER"],
            normalized_rmse=0.90,
            pearson=0.60,
            sd_ratio=0.80,
        )
        write_toy_run(
            tmp_path,
            variant="uniform_env_dtm_v2",
            seed=seed,
            active_kernels=["K_E_GEO", "K_E_WEATHER", "K_E_DTM_V2"],
            normalized_rmse=0.85,
            pearson=0.61,
            sd_ratio=0.85,
        )
    detail, decision = compare_candidate(
        models_root=tmp_path,
        baseline_variant="uniform_env_generic",
        candidate=Candidate("K_E_DTM_V2", "DAYS_TO_MATURITY", "uniform_env_dtm_v2"),
        seeds=[2026, 2027, 2028, 2029],
        specific_kernels={"K_E_DTM_V2", "K_E_DTH_V2"},
    )
    assert len(detail) == 4
    assert decision["candidate_win_count"] == 4
    assert decision["accepted"] is True
    assert decision["decision"] == "accept_for_multitrait_baseline"


def test_server_runner_isolates_trait_specific_kernel_candidates() -> None:
    source = (ROOT / "scripts" / "run_trait_environment_kernel_ablation.sh").read_text(
        encoding="utf-8"
    )
    assert 'run_variant "uniform_env_generic" "" "$SPECIFIC_KERNELS"' in source
    for kernel in ["K_E_DTH_V2", "K_E_DTM_V2", "K_E_GY_V2", "K_E_TGW_V2", "K_E_PH_V2"]:
        assert kernel in source
    assert "MULTITRAIT_INCLUDE_DISABLED_KERNELS" in source
    assert "summarize_trait_environment_ablation" in source
