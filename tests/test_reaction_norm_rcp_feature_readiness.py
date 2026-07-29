from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from server_training_pipeline.audit_reaction_norm_rcp_feature_readiness import (
    canonical_source_identity,
    classify_feature,
    main as audit_readiness,
)
from server_training_pipeline.plan_reaction_norm_rcp_projection import main as plan_rcp


def manifest_row(
    feature: str,
    source_feature: str,
    source_artifact: str,
    feature_block: str,
    *,
    missing: bool = False,
) -> dict[str, object]:
    return {
        "feature": feature,
        "source_feature": source_feature,
        "source_artifact": source_artifact,
        "feature_block": feature_block,
        "eligible_traits": "*",
        "regulatory_treatment": "none",
        "is_missingness_indicator": missing,
        "phenotype_derived": False,
        "fit_partition": "outer_training_environments_only",
    }


def test_feature_classification_separates_historical_and_sparse_fields() -> None:
    historical = pd.Series(
        manifest_row(
            "water__observed__PPN_MONTH_OF_HARVESTED",
            "observed__PPN_MONTH_OF_HARVESTED",
            "envdata.tsv",
            "water",
        )
    )
    management = pd.Series(
        manifest_row(
            "management__FERTILIZER_%K2O_3",
            "FERTILIZER_%K2O_3",
            "envdata.tsv",
            "management",
        )
    )
    climate = pd.Series(
        manifest_row(
            "heat__window__api_d0_30_temperature_max_c",
            "window__api_d0_30_temperature_max_c",
            "trait_weather.tsv",
            "heat",
        )
    )
    assert classify_feature(historical)["projectability_class"] == (
        "historical_only_unprojectable"
    )
    assert classify_feature(management)["range_rule_class"] == "sparse_management"
    assert classify_feature(climate)["projectability_class"] == "fixed_window_climate"
    legacy_rain_count = pd.Series(
        manifest_row(
            "water__observed__NO_OF_RAINS_DURING_CYCLE_OLD",
            "observed__NO_OF_RAINS_DURING_CYCLE_OLD",
            "envdata.tsv",
            "water",
        )
    )
    assert classify_feature(legacy_rain_count)["projectability_class"] == (
        "historical_only_unprojectable"
    )


def test_duplicate_identity_recognizes_observed_envdata_copy() -> None:
    management = pd.Series(
        manifest_row(
            "management__PRE_SOWING_IRRIGATION",
            "PRE_SOWING_IRRIGATION",
            "envdata.tsv",
            "management",
        )
    )
    observed = pd.Series(
        manifest_row(
            "water__observed__PRE_SOWING_IRRIGATION",
            "observed__PRE_SOWING_IRRIGATION",
            "envdata.tsv",
            "water",
        )
    )
    assert canonical_source_identity(management)[0] == canonical_source_identity(observed)[0]


def test_readiness_audit_is_complete_but_does_not_authorize_projection(
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
    rows = [
        manifest_row("geo__latitude", "latitude", "envdata.tsv+locdata.tsv", "geo"),
        manifest_row(
            "management__PRE_SOWING_IRRIGATION",
            "PRE_SOWING_IRRIGATION",
            "envdata.tsv",
            "management",
        ),
        manifest_row(
            "water__observed__PRE_SOWING_IRRIGATION",
            "observed__PRE_SOWING_IRRIGATION",
            "envdata.tsv",
            "water",
        ),
        manifest_row(
            "water__observed__PPN_MONTH_OF_HARVESTED",
            "observed__PPN_MONTH_OF_HARVESTED",
            "envdata.tsv",
            "water",
        ),
        manifest_row(
            "heat__generic__weather_api_temperature_mean_c",
            "generic__weather_api_temperature_mean_c",
            "observed_api_weather_or_fold_climatology",
            "heat",
        ),
        manifest_row(
            "confidence__weather_observed",
            "weather_observed",
            "environment_expert_coverage.tsv",
            "confidence",
        ),
    ]
    manifest = pd.DataFrame(rows)
    manifest.to_csv(
        environment_dir / "E_REACTION_NORM_V1_feature_manifest.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        {
            "feature": manifest["feature"],
            "mean": 0.0,
            "std": 1.0,
            "status": "retained",
        }
    ).to_csv(
        environment_dir / "E_REACTION_NORM_V1_scaling.tsv", sep="\t", index=False
    )
    (environment_dir / "E_REACTION_NORM_V1_certification.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "env_id": ["e1", "e2", "e3"],
            "geo__latitude": [10.0, 11.0, 12.0],
            "management__PRE_SOWING_IRRIGATION": [0.0, 1.0, 0.0],
            "water__observed__PRE_SOWING_IRRIGATION": [0.0, 1.0, 0.0],
            "water__observed__PPN_MONTH_OF_HARVESTED": [1.0, 2.0, 3.0],
            "heat__generic__weather_api_temperature_mean_c": [20.0, 21.0, 22.0],
            "confidence__weather_observed": [1.0, 1.0, 0.0],
        }
    ).to_parquet(environment_dir / "E_REACTION_NORM_V1_raw.parquet", index=False)

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
    readiness_protocol = Path(
        "server_training_pipeline/reaction_norm_rcp_feature_readiness_protocol_v1.json"
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

    diagnostics = tmp_path / "environment_extrapolation_by_feature.tsv"
    diagnostic_rows = []
    for row in rows:
        diagnostic_rows.append(
            {
                "scenario": "temporal_holdout",
                "outer_fold": 0,
                "feature": row["feature"],
                "feature_block": row["feature_block"],
                "source_feature": row["source_feature"],
                "source_artifact": row["source_artifact"],
                "regulatory_treatment": "none",
                "training_nonmissing": 3,
                "test_nonmissing": 1,
                "test_missing": 0,
                "training_min": 0.0,
                "training_q01": 0.0,
                "training_q99": 1.0,
                "training_max": 1.0,
                "test_below_training_min_fraction": 0.0,
                "test_above_training_max_fraction": 0.0,
                "test_outside_training_range_fraction": 0.0,
                "test_outside_robust_range_fraction": 0.0,
                "test_abs_z_gt_moderate_fraction": 0.0,
                "test_abs_z_gt_extreme_fraction": 0.0,
                "test_max_abs_z": 1.0,
            }
        )
    pd.DataFrame(diagnostic_rows).to_csv(diagnostics, sep="\t", index=False)

    out_dir = tmp_path / "readiness"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--outer-dir",
            str(outer_dir),
            "--outer-protocol",
            str(outer_protocol),
            "--environment-protocol",
            str(environment_protocol),
            "--projection-protocol",
            str(projection_protocol),
            "--projection-plan",
            str(plan_dir / "E_REACTION_NORM_RCP_V1_plan.json"),
            "--reporting-feature-diagnostics",
            str(diagnostics),
            "--readiness-protocol",
            str(readiness_protocol),
            "--out-dir",
            str(out_dir),
        ],
    )
    audit_readiness()
    result = json.loads(
        (out_dir / "RCP_feature_readiness_certification.json").read_text()
    )
    assert result["status"] == "PASS"
    assert result["feature_contract_ready_for_population"] is False
    assert result["future_covariate_population_allowed"] is False
    assert result["rcp_predictions_allowed"] is False
    assert result["future_matrix_count_generated"] == 0
    assert result["exact_duplicate_group_count"] == 1
    duplicates = pd.read_csv(
        out_dir / "RCP_duplicate_source_consistency.tsv", sep="\t"
    )
    assert duplicates["status"].eq("PASS").all()
    assert not list(out_dir.rglob("*.parquet"))
