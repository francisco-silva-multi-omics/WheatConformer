from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from server_training_pipeline.audit_reaction_norm_rcp_historical_reconstruction import (
    HARVEST_MONTH_FIELDS,
    fit_harvest_anchors,
    main as audit_historical_reconstruction,
    monthly_window,
)
from server_training_pipeline.final_evaluation_contract import file_sha256


ID_PARTS = {
    "Trial_name": "TRIAL",
    "Occ": "1",
    "Loc_no": "100",
    "Country": "MEXICO",
    "Loc_desc": "SITE",
}


def env_id(cycle: int) -> str:
    return "|".join(["TRIAL", "1", "100", "MEXICO", "SITE", str(cycle)])


def test_harvest_anchor_model_uses_training_country_then_global_median() -> None:
    ids = pd.Index(["e1", "e2", "e3", "e4"])
    metadata = pd.DataFrame(
        {
            "sowing_date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-01", "2020-01-01"]
            ),
            "harvest_start_date": pd.to_datetime(
                ["2020-05-30", "2020-06-09", None, None]
            ),
            "harvest_finish_date": pd.to_datetime([None, None, None, None]),
            "Country": ["A", "A", "A", "B"],
        },
        index=ids,
    )
    protocol = {
        "harvest_anchor": {
            "minimum_country_donors": 2,
            "minimum_season_length_days": 60,
            "maximum_season_length_days": 365,
            "default_season_length_days_when_no_training_donors": 180,
        }
    }
    fitted, summary = fit_harvest_anchors(metadata, ids, protocol)
    assert fitted.loc["e3", "harvest_anchor_source"] == (
        "outer_training_country_median_season_length"
    )
    assert fitted.loc["e4", "harvest_anchor_source"] == (
        "outer_training_global_median_season_length"
    )
    assert summary["valid_season_length_donor_count"] == 2
    start, end = monthly_window(pd.Timestamp("2020-06-15"), 11)
    assert start == pd.Timestamp("2019-07-01")
    assert end == pd.Timestamp("2020-06-30")


def test_historical_reconstruction_audit_builds_queue_and_remains_blocked(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path
    environment = data_root / "environment"
    outer_dir = data_root / "outer"
    readiness = data_root / "readiness"
    output = data_root / "audit"
    environment.mkdir()
    readiness.mkdir()

    ids = [env_id(2018), env_id(2019), env_id(2020)]
    fit_ids = data_root / "fit_ids.tsv"
    pd.DataFrame({"env_id": ids}).to_csv(fit_ids, sep="\t", index=False)

    source_tokens = [
        "PRECIPITATION_FROM_SOWING_TO_MATURITY",
        "PRECIPITATION_ON_CROP",
        "MOISTURE_AVAILB_BEFORE_SOWING_EXCL_PRE_IRRIGATION",
        *HARVEST_MONTH_FIELDS,
    ]
    assert len(source_tokens) == 15
    envdata_rows = []
    for position, identifier in enumerate(ids):
        cycle = identifier.rsplit("|", 1)[1]
        values = {
            "PRECIPITATION_FROM_SOWING_TO_MATURITY": 100.0 + position * 10.0,
            "PRECIPITATION_ON_CROP": 100.0 + position * 10.0,
            "MOISTURE_AVAILB_BEFORE_SOWING_EXCL_PRE_IRRIGATION": 20.0 + position,
            "TOTAL_PRECIPIT_IN_12_MONTHS": 500.0 + position * 10.0,
            "ESTIMATE_TOTAL_PRECIPIT_IN_12_MONTHS": 510.0 + position * 10.0,
            **{field: 10.0 + position for field in HARVEST_MONTH_FIELDS},
        }
        for trait, value in values.items():
            envdata_rows.append(
                {
                    **ID_PARTS,
                    "Cycle": cycle,
                    "Trait_name": trait,
                    "Value": value,
                }
            )
    envdata_path = environment / "envdata.tsv"
    pd.DataFrame(envdata_rows).to_csv(envdata_path, sep="\t", index=False)

    window_path = environment / "agronomic_api_weather_windows.tsv"
    pd.DataFrame(
        {
            "env_id": ids,
            "window_label": ["d0_180"] * 3,
            "precipitation_total_mm": [100.0, 110.0, 120.0],
        }
    ).to_csv(window_path, sep="\t", index=False)
    manifest_path = environment / "trial_weather_fetch_manifest.tsv"
    pd.DataFrame(
        {
            "env_id": ids,
            "Country": ["MEXICO"] * 3,
            "latitude": [20.0, 20.0, 20.0],
            "longitude": [-100.0, -100.0, -100.0],
            "coordinate_source": ["curated"] * 3,
            "sowing_date": ["2018-01-01", "2019-01-01", "2020-01-01"],
            "sowing_date_source": ["fieldbook"] * 3,
            "harvest_start_date": ["2018-06-01", "2019-06-01", ""],
            "harvest_finish_date": ["", "", ""],
        }
    ).to_csv(manifest_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "env_id": ids,
            "n_days_weather": [180, 180, 180],
            "precipitation_total_mm": [250.0, 260.0, 270.0],
            "fetch_status": ["ok"] * 3,
        }
    ).to_csv(
        environment / "trial_weather_features_openmeteo.tsv", sep="\t", index=False
    )

    generic_qc = data_root / "generic_qc.json"
    generic_qc.write_text(
        json.dumps({"weather_feature_input_dir": str(environment)}), encoding="utf-8"
    )
    fold_environment = (
        outer_dir
        / "folds"
        / "temporal_holdout"
        / "outer_0"
        / "E_REACTION_NORM_V1"
    )
    fold_environment.mkdir(parents=True)
    (fold_environment / "E_REACTION_NORM_V1_certification.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )

    def identity(path: Path) -> dict[str, object]:
        return {"path": str(path), "sha256": file_sha256(path)}

    (fold_environment / "E_REACTION_NORM_V1_provenance.json").write_text(
        json.dumps(
            {
                "sources": {
                    "fit_environment_ids": identity(fit_ids),
                    "envdata": identity(envdata_path),
                    "window_features": identity(window_path),
                    "generic_environment_provenance": identity(generic_qc),
                }
            }
        ),
        encoding="utf-8",
    )
    outer_protocol = data_root / "outer_protocol.json"
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

    lineage_rows = []
    for source in source_tokens:
        lineage_rows.append(
            {
                "feature": f"water__observed__{source}",
                "source_feature": f"observed__{source}",
                "feature_block": "water",
                "is_missingness_indicator": False,
                "projectability_class": "historical_only_unprojectable",
                "duplicate_group_id": "",
            }
        )
    lineage_path = readiness / "RCP_feature_readiness_lineage.tsv"
    pd.DataFrame(lineage_rows).to_csv(lineage_path, sep="\t", index=False)
    range_path = readiness / "RCP_historical_range_rule_audit.tsv"
    pd.DataFrame(
        {
            "feature": ["heat__window__api_d0_30_heat_days_tmax_ge_35"],
            "feature_block": ["heat"],
            "historical_range_rule_status": [
                "HISTORICAL_BASELINE_EXCEEDS_GLOBAL_HARD_Z"
            ],
            "historical_max_abs_z": [12.0],
        }
    ).to_csv(range_path, sep="\t", index=False)
    readiness_cert = {
        "status": "PASS",
        "future_covariate_population_allowed": False,
        "artifacts": {
            lineage_path.name: file_sha256(lineage_path),
            range_path.name: file_sha256(range_path),
        },
    }
    (readiness / "RCP_feature_readiness_certification.json").write_text(
        json.dumps(readiness_cert), encoding="utf-8"
    )

    base_protocol = json.loads(
        Path(
            "server_training_pipeline/reaction_norm_rcp_historical_reconstruction_protocol_v1.json"
        ).read_text()
    )
    base_protocol["crop_precipitation_backcast"][
        "minimum_paired_training_environments"
    ] = 2
    base_protocol["crop_precipitation_backcast"]["minimum_pearson"] = 0.9
    protocol_path = data_root / "reconstruction_protocol.json"
    protocol_path.write_text(json.dumps(base_protocol), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--root",
            str(data_root),
            "--outer-dir",
            str(outer_dir),
            "--outer-protocol",
            str(outer_protocol),
            "--readiness-dir",
            str(readiness),
            "--reconstruction-protocol",
            str(protocol_path),
            "--out-dir",
            str(output),
        ],
    )
    audit_historical_reconstruction()

    certification = json.loads(
        (output / "RCP_historical_reconstruction_certification.json").read_text()
    )
    assert certification["status"] == "PASS"
    assert certification["historical_source_count"] == 15
    assert certification["certified_replacement_count"] == 2
    assert certification["blocked_replacement_count"] == 13
    assert certification["future_covariate_population_allowed"] is False
    assert certification["rcp_predictions_allowed"] is False
    queue = pd.read_csv(output / "RCP_daily_reanalysis_work_queue.tsv", sep="\t")
    assert set(queue["request_kind"]) == {
        "annual_precipitation_trailing_365_before_sowing",
        "pre_sowing_antecedent_water_balance",
        "harvest_relative_calendar_month_precipitation",
    }
    assert queue["request_status"].eq("READY_TO_FETCH").all()
    unique_queue = pd.read_csv(
        output / "RCP_daily_reanalysis_unique_requests.tsv", sep="\t"
    )
    assert set(unique_queue["request_id"].astype(str)) == set(
        queue["request_id"].astype(str)
    )
    backcast = pd.read_csv(
        output / "RCP_fixed_window_replacement_backcast.tsv", sep="\t"
    )
    assert len(backcast) == 6
    assert backcast["fit_partition"].eq("outer_training_environments_only").all()
    assert not list(output.glob("*future_matrix*"))
    assert not list(output.glob("*prediction*"))
