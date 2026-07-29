from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from server_training_pipeline.plan_reaction_norm_rcp_projection import (
    classify_population,
    main as plan_rcp,
)
from server_training_pipeline.certify_reaction_norm_rcp_covariates import (
    main as certify_rcp,
)
from server_training_pipeline.report_reaction_norm_routed_diagnostics import (
    calibration_diagnostics,
    environment_range_diagnostics,
    top_k_regret,
    within_environment_diagnostics,
    resolve_provenance_source,
)
from server_training_pipeline.final_evaluation_contract import file_sha256


PROTOCOL = json.loads(
    Path("server_training_pipeline/reaction_norm_reporting_protocol_v1.json").read_text(
        encoding="utf-8"
    )
)


def prediction_rows() -> pd.DataFrame:
    rows = []
    for environment, offset in [("e1", 0.0), ("e2", 100.0)]:
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            rows.append(
                {
                    "scenario": "unseen_genotypes",
                    "outer_fold": 0,
                    "model_label": "model",
                    "trait_name_canonical": "GRAIN_YIELD",
                    "split": "test",
                    "env_kernel_id": environment,
                    "phenotype_value": value + offset,
                    "y_pred": value + 2.0 * offset,
                }
            )
    return pd.DataFrame(rows)


def test_within_environment_centering_removes_environment_offset() -> None:
    summary, detail = within_environment_diagnostics(prediction_rows(), PROTOCOL)
    assert len(summary) == 1
    assert summary.iloc[0]["centered_rmse"] == 0.0
    assert summary.iloc[0]["centered_pearson"] == 1.0
    assert summary.iloc[0]["centered_spearman"] == 1.0
    assert len(detail) == 2
    assert detail["directional_top_k_regret"].eq(0.0).all()


def test_top_k_regret_detects_reversed_ranking() -> None:
    frame = pd.DataFrame(
        {
            "phenotype_value": np.arange(1.0, 11.0),
            "y_pred": np.arange(10.0, 0.0, -1.0),
        }
    )
    result = top_k_regret(
        frame, fraction=0.2, minimum_top_k=1, maximum_fraction=0.5
    )
    assert result["k"] == 2
    assert result["upper_tail_regret"] == 8.0
    assert result["lower_tail_regret"] == 8.0


def test_negative_validation_calibration_slope_is_flagged() -> None:
    frame = pd.DataFrame(
        {
            "scenario": ["country_holdout"] * 8,
            "outer_fold": [0] * 8,
            "model_label": ["model"] * 8,
            "trait_name_canonical": ["GRAIN_YIELD"] * 8,
            "split": ["val"] * 4 + ["test"] * 4,
            "phenotype_value": [4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0],
            "y_pred": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    result = calibration_diagnostics(frame)
    assert len(result) == 1
    assert bool(result.iloc[0]["negative_calibration_slope"])
    assert result.iloc[0]["calibration_status"] == "FLAG_NEGATIVE_VALIDATION_SLOPE"
    assert result.iloc[0]["raw_spearman"] == 1.0
    assert result.iloc[0]["calibrated_spearman"] == -1.0


def test_environment_range_diagnostics_detect_extrapolation() -> None:
    raw = pd.DataFrame(
        {
            "env_id": ["train1", "train2", "test1"],
            "heat__days": [0.0, 2.0, 10.0],
        }
    )
    standardized = pd.DataFrame(
        {
            "env_id": ["train1", "train2", "test1"],
            "heat__days": [-1.0, 1.0, 9.0],
        }
    )
    manifest = pd.DataFrame(
        {
            "feature": ["heat__days"],
            "feature_block": ["heat"],
            "source_feature": ["days"],
            "source_artifact": ["weather"],
            "regulatory_treatment": ["heat"],
            "is_missingness_indicator": ["False"],
        }
    )
    feature, environment = environment_range_diagnostics(
        raw,
        standardized,
        manifest,
        ["train1", "train2"],
        ["test1"],
        scenario="temporal_holdout",
        outer_fold=0,
        q_low=0.01,
        q_high=0.99,
        moderate_z=3.0,
        extreme_z=5.0,
    )
    assert feature.iloc[0]["test_above_training_max_fraction"] == 1.0
    assert feature.iloc[0]["test_abs_z_gt_extreme_fraction"] == 1.0
    assert environment.iloc[0]["maximum_absolute_z"] == 9.0


def test_rcp_population_plan_separates_static_and_projected_features() -> None:
    geo = pd.Series(
        {
            "feature": "geo__latitude",
            "source_feature": "latitude",
            "feature_block": "geo",
        }
    )
    weather = pd.Series(
        {
            "feature": "heat__window__TMAX_MEAN_0_30",
            "source_feature": "window__TMAX_MEAN_0_30",
            "feature_block": "heat",
        }
    )
    missing = pd.Series(
        {
            "feature": "heat__window__TMAX_MEAN_0_30__missing",
            "source_feature": "window__TMAX_MEAN_0_30",
            "feature_block": "heat",
        }
    )
    assert classify_population(geo)[0] == "site_registry_static"
    assert classify_population(weather)[0] == "recompute_from_bias_corrected_daily_climate"
    assert classify_population(missing)[0] == "derived_missingness_indicator"


def test_provenance_source_falls_back_to_current_data_root(tmp_path: Path) -> None:
    current = tmp_path / "model_kernels" / "evaluation" / "ids.tsv"
    current.parent.mkdir(parents=True)
    current.write_text("env_id\ne1\n", encoding="utf-8")
    source = {
        "path": "/retired/server/root/model_kernels/evaluation/ids.tsv",
        "sha256": file_sha256(current),
    }
    assert resolve_provenance_source(source, data_root=tmp_path) == current.resolve()


def test_rcp_plan_and_range_certification_are_fold_local(
    tmp_path: Path, monkeypatch
) -> None:
    outer_dir = tmp_path / "outer"
    environment_dir = (
        outer_dir
        / "folds"
        / "temporal_holdout"
        / "outer_0"
        / "E_REACTION_NORM_V1"
    )
    environment_dir.mkdir(parents=True)
    manifest = pd.DataFrame(
        {
            "feature": ["heat__temperature"],
            "source_feature": ["temperature"],
            "source_artifact": ["daily_climate"],
            "feature_block": ["heat"],
            "eligible_traits": ["*"],
            "regulatory_treatment": ["heat"],
            "is_missingness_indicator": [False],
            "phenotype_derived": [False],
            "fit_partition": ["outer_training_environments_only"],
        }
    )
    manifest.to_csv(
        environment_dir / "E_REACTION_NORM_V1_feature_manifest.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        {
            "feature": ["heat__temperature"],
            "mean": [1.0],
            "std": [1.0],
            "status": ["retained"],
        }
    ).to_csv(
        environment_dir / "E_REACTION_NORM_V1_scaling.tsv", sep="\t", index=False
    )
    (environment_dir / "E_REACTION_NORM_V1_certification.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    fit_ids = tmp_path / "fit_ids.tsv"
    pd.DataFrame({"env_id": ["e1", "e2", "e3"]}).to_csv(
        fit_ids, sep="\t", index=False
    )
    pd.DataFrame(
        {"env_id": ["e1", "e2", "e3"], "heat__temperature": [0.0, 1.0, 2.0]}
    ).to_parquet(environment_dir / "E_REACTION_NORM_V1_raw.parquet", index=False)
    pd.DataFrame(
        {"env_id": ["e1", "e2", "e3"], "heat__temperature": [-1.0, 0.0, 1.0]}
    ).to_parquet(environment_dir / "E_REACTION_NORM_V1.parquet", index=False)
    pd.DataFrame(
        columns=["feature", "source_feature", "fit_missing_fraction"]
    ).to_csv(
        environment_dir / "E_REACTION_NORM_V1_missingness_indicators.tsv",
        sep="\t",
        index=False,
    )
    (environment_dir / "E_REACTION_NORM_V1_provenance.json").write_text(
        json.dumps({"sources": {"fit_environment_ids": {"path": str(fit_ids)}}}),
        encoding="utf-8",
    )

    outer_protocol = tmp_path / "outer.json"
    outer_protocol.write_text(
        json.dumps(
            {
                "status": "frozen_after_inner_validation_before_outer_test",
                "selected_environment_architecture": "explicit_E_REACTION_NORM_V1",
                "scenarios": {"temporal_holdout": 1},
            }
        ),
        encoding="utf-8",
    )
    environment_protocol = tmp_path / "environment.json"
    environment_protocol.write_text(
        json.dumps({"status": "frozen_before_inner_validation"}), encoding="utf-8"
    )
    projection_protocol = Path(
        "server_training_pipeline/reaction_norm_rcp_projection_protocol_v1.json"
    ).resolve()
    plan_dir = tmp_path / "plan"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan",
            "--outer-dir",
            str(outer_dir),
            "--outer-protocol",
            str(outer_protocol),
            "--environment-protocol",
            str(environment_protocol),
            "--projection-protocol",
            str(projection_protocol),
            "--out-dir",
            str(plan_dir),
        ],
    )
    plan_rcp()
    plan = json.loads((plan_dir / "E_REACTION_NORM_RCP_V1_plan.json").read_text())
    assert plan["status"] == "PASS"
    assert plan["projection_allowed"] is False
    population = pd.read_csv(
        plan_dir / "E_REACTION_NORM_RCP_V1_feature_population_plan.tsv", sep="\t"
    )
    assert population.iloc[0]["population_method"] == (
        "recompute_from_bias_corrected_daily_climate"
    )

    future = tmp_path / "future.parquet"
    pd.DataFrame(
        {"future_env_id": ["future1"], "heat__temperature": [1.5]}
    ).to_parquet(future, index=False)
    certified_dir = tmp_path / "certified"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "certify",
            "--future-raw-matrix",
            str(future),
            "--projection-plan",
            str(plan_dir / "E_REACTION_NORM_RCP_V1_plan.json"),
            "--outer-dir",
            str(outer_dir),
            "--outer-protocol",
            str(outer_protocol),
            "--projection-protocol",
            str(projection_protocol),
            "--out-dir",
            str(certified_dir),
        ],
    )
    certify_rcp()
    certification = json.loads(
        (
            certified_dir
            / "E_REACTION_NORM_RCP_V1_covariate_certification.json"
        ).read_text()
    )
    assert certification["status"] == "PASS"
    assert certification["projection_allowed"] is True
    projected = pd.read_parquet(
        certified_dir
        / "folds"
        / "temporal_holdout"
        / "outer_0"
        / "E_REACTION_NORM_RCP_V1.parquet"
    )
    assert projected.iloc[0]["heat__temperature"] == 0.5
