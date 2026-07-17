from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_trial_weather_fetch_manifest import ID_COLS, env_id, main as build_manifest
from recover_trial_weather_dates import (
    discover_date_candidates,
    merge_base_supplement,
    parse_raw_date,
    resolve_candidates,
)
from server_training_pipeline.audit_weather_recovery import (
    classify_environment_coverage,
    request_id,
)
from server_training_pipeline.build_weather_climatology_expert import (
    main as build_climatology,
)
from server_training_pipeline.summarize_weather_recovery_adoption import summarize


def test_manifest_uses_country_scoped_locations_and_curated_date_provenance(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "environment"
    output = tmp_path / "recovery"
    environment.mkdir()
    base = {
        "Occ": "1",
        "Loc_no": "7",
        "Loc_desc": "SITE",
        "Cycle": "2020",
    }
    env = pd.DataFrame(
        [
            {
                "trial_dir": "target_a",
                "Trial_name": "TRIAL_A",
                "Country": "MEXICO",
                "Trait_name": "YIELD",
                "Value": "1",
                **base,
            },
            {
                "trial_dir": "target_b",
                "Trial_name": "TRIAL_B",
                "Country": "KENYA",
                "Trait_name": "SOWING_DATE",
                "Value": "2020-01-15",
                **base,
            },
        ]
    )
    env.to_csv(environment / "envdata.tsv", sep="\t", index=False)
    loc = pd.DataFrame(
        [
            {
                "trial_dir": "source_a",
                "Country": "MEXICO",
                "Loc_no": "7",
                "Lat_degress": "20",
                "Lat_minutes": "0",
                "Latitud": "N",
                "Long_degress": "100",
                "Long_minutes": "0",
                "Longitude": "W",
                "Altitude": "1500",
            },
            {
                "trial_dir": "source_b",
                "Country": "KENYA",
                "Loc_no": "7",
                "Lat_degress": "1",
                "Lat_minutes": "0",
                "Latitud": "S",
                "Long_degress": "36",
                "Long_minutes": "0",
                "Longitude": "E",
                "Altitude": "1800",
            },
        ]
    )
    loc.to_csv(environment / "locdata.tsv", sep="\t", index=False)
    env_a_id = env_id(env.iloc[[0]])[0]
    supplement = tmp_path / "dates.tsv"
    pd.DataFrame(
        {
            "env_id": [env_a_id],
            "sowing_date": ["2020-02-01"],
            "provenance": ["reviewed_raw_fieldbook"],
        }
    ).to_csv(supplement, sep="\t", index=False)

    build_manifest(environment, output, date_supplement=supplement)

    manifest = pd.read_csv(output / "trial_weather_fetch_manifest.tsv", sep="\t")
    mexico = manifest[manifest["Country"].eq("MEXICO")].iloc[0]
    assert mexico["latitude"] == 20.0
    assert mexico["longitude"] == -100.0
    assert mexico["coordinate_source"] == "Loc_data_Country_Loc_no_stable_mean"
    assert mexico["sowing_date_source"] == "reviewed_raw_fieldbook"
    assert bool(mexico["window_inferred"])


def test_weather_audit_classifies_missing_causes_before_imputation() -> None:
    order = pd.DataFrame({"env_id": ["e1", "e2", "e3", "e4", "e5"]})
    manifest = pd.DataFrame(
        {
            "env_id": order["env_id"],
            "latitude": [1, 1, np.nan, 1, 1],
            "longitude": [2, 2, np.nan, 2, 2],
            "weather_start_date": ["2000-01-01", "", "2000-01-01", "1970-01-01", "2000-01-01"],
            "weather_end_date": ["2000-04-01", "", "2000-04-01", "1970-04-01", "2000-04-01"],
            "has_fetch_window": [True, False, True, True, True],
            "has_fetch_coordinates": [True, True, False, True, True],
            "ready_to_fetch": [True, False, False, True, True],
        }
    )
    failed = {request_id(manifest).iloc[4]}
    audit = classify_environment_coverage(
        order,
        manifest,
        pd.DataFrame({"env_id": ["e1"], "observed_nasa": [True]}),
        pd.DataFrame(columns=["env_id", "observed_openmeteo"]),
        failed,
        {"e2", "e5"},
        pd.Timestamp("1981-01-01"),
    ).set_index("env_id")

    assert audit.loc["e1", "coverage_cause"] == "observed_api"
    assert audit.loc["e2", "coverage_cause"] == "missing_fetch_window"
    assert audit.loc["e3", "coverage_cause"] == "missing_coordinates"
    assert audit.loc["e4", "coverage_cause"] == "dates_outside_nasa_coverage"
    assert audit.loc["e5", "coverage_cause"] == "fetch_failed"
    assert bool(audit.loc["e5", "used_by_pedigree_model"])


def test_weather_audit_rejects_stale_cached_request_windows() -> None:
    order = pd.DataFrame({"env_id": ["e1"]})
    manifest = pd.DataFrame(
        {
            "env_id": ["e1"],
            "latitude": [20.0],
            "longitude": [-100.0],
            "weather_start_date": ["2000-01-01"],
            "weather_end_date": ["2000-04-01"],
            "has_fetch_window": [True],
            "has_fetch_coordinates": [True],
            "ready_to_fetch": [True],
        }
    )
    audit = classify_environment_coverage(
        order,
        manifest,
        pd.DataFrame(
            {
                "env_id": ["e1"],
                "observed_nasa_raw": [True],
                "request_id_nasa": ["20.00000|-100.00000|1999-01-01|1999-04-01"],
            }
        ),
        pd.DataFrame(
            columns=["env_id", "observed_openmeteo_raw", "request_id_openmeteo"]
        ),
        set(),
        set(),
        pd.Timestamp("1981-01-01"),
    ).iloc[0]

    assert not bool(audit["weather_observed"])
    assert bool(audit["stale_cached_weather_request"])
    assert audit["coverage_cause"] == "stale_cached_request"


def test_climatology_is_separate_and_location_season_specific(
    tmp_path: Path, monkeypatch
) -> None:
    environment = tmp_path / "environment"
    weather = tmp_path / "weather"
    audit_dir = tmp_path / "audit"
    output = tmp_path / "kernels"
    for directory in [environment, weather, audit_dir, output]:
        directory.mkdir()
    env_ids = [f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)]
    pd.DataFrame({"env_id": env_ids}).to_csv(
        environment / "env_kernel_sample_order.tsv", sep="\t", index=False
    )
    manifest = pd.DataFrame(
        {
            "env_id": env_ids,
            "Country": ["MEXICO"] * 4 + ["KENYA"] * 4,
            "Loc_no": ["1"] * 4 + ["2"] * 4,
            "trial_dir": ["ta"] * 4 + ["tb"] * 4,
            "weather_start_date": ["2000-01-01"] * 8,
        }
    )
    manifest.to_csv(weather / "trial_weather_fetch_manifest.tsv", sep="\t", index=False)
    observed_ids = ["a0", "a1", "a2", "b0", "b1", "b2"]
    pd.DataFrame(
        {
            "env_id": env_ids,
            "weather_observed": [value in observed_ids for value in env_ids],
            "window_inferred": False,
            "coordinates_inferred": False,
        }
    ).to_csv(
        audit_dir / "weather_recovery_environment_audit.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {
            "env_id": observed_ids,
            "fetch_status": "ok",
            "temperature_mean_c": [10, 11, 9, 30, 31, 29],
            "precipitation_total_mm": [100, 110, 90, 20, 25, 15],
        }
    ).to_csv(weather / "trial_weather_features_nasa_power.tsv", sep="\t", index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_weather_climatology_expert",
            "--root",
            str(tmp_path),
            "--environment-dir",
            str(environment),
            "--weather-dir",
            str(weather),
            "--audit-dir",
            str(audit_dir),
            "--out-dir",
            str(output),
            "--minimum-donors",
            "3",
        ],
    )
    build_climatology()

    coverage = pd.read_csv(output / "environment_expert_coverage.tsv", sep="\t")
    recovered = coverage.set_index("env_id")["weather_climatology"]
    assert bool(recovered["a3"])
    assert bool(recovered["b3"])
    kernel = np.load(output / "K_climatology.npy")
    assert np.isfinite(kernel).all()
    assert np.allclose(kernel, kernel.T)
    assert np.diag(kernel)[3] > 0
    assert np.diag(kernel)[7] > 0


def test_weather_adoption_uses_validation_only_and_requires_consistent_seeds() -> None:
    paired = pd.DataFrame(
        [
            {
                "split": split,
                "mode": "env",
                "seed": seed,
                "trait_name_canonical": "DAYS_TO_HEADING",
                "delta_normalized_rmse": -0.1 if split == "val" else 1.0,
                "delta_pearson": 0.05 if split == "val" else -1.0,
            }
            for seed in [2026, 2027, 2028, 2029]
            for split in ["val", "test"]
        ]
    )
    contract = pd.DataFrame(
        {
            "comparison_eligible": [True] * 4,
            "status": ["PASS"] * 4,
        }
    )

    decision = summarize(paired, contract).set_index("mode")
    assert bool(decision.loc["env", "accepted"])
    assert bool(decision.loc["overall", "accepted"])


def test_raw_trial_date_parser_separates_full_partial_and_invalid_dates() -> None:
    full = parse_raw_date("15/11/2020")
    partial = parse_raw_date("00/05/1998")
    invalid = parse_raw_date("29/02/1997")

    assert full["parse_status"] == "full_date"
    assert pd.Timestamp(full["parsed_date"]) == pd.Timestamp("2020-11-15")
    assert partial["parse_status"] == "partial_month_year"
    assert partial["partial_month"] == 5
    assert partial["partial_year"] == 1998
    assert pd.isna(partial["parsed_date"])
    assert invalid["parse_status"] == "invalid_partial_date"
    assert pd.isna(invalid["parsed_date"])


def test_raw_trial_date_recovery_accepts_only_unique_cycle_plausible_dates(
    tmp_path: Path,
) -> None:
    trial_root = tmp_path / "TRIALS_AND_NURSERIES"
    trial_dir = trial_root / "trial_a"
    trial_dir.mkdir(parents=True)
    target_rows = pd.DataFrame(
        [
            {
                "env_id": "T1|1|7|MEXICO|SITE|2020",
                "trial_dir": "trial_a",
                "Trial_name": "T1",
                "Occ": "1",
                "Loc_no": "7",
                "Country": "MEXICO",
                "Loc_desc": "SITE",
                "Cycle": "2020",
            },
            {
                "env_id": "T2|2|8|KENYA|SITE2|2019-20",
                "trial_dir": "trial_a",
                "Trial_name": "T2",
                "Occ": "2",
                "Loc_no": "8",
                "Country": "KENYA",
                "Loc_desc": "SITE2",
                "Cycle": "2019-20",
            },
        ]
    )
    raw = pd.DataFrame(
        [
            {
                "Trial name": "T1",
                "Occ": "1",
                "Loc_no": "7",
                "Country": "MEXICO",
                "Loc_desc": "SITE",
                "Cycle": "2020",
                "Trait name": "SOWING_DATE",
                "Value": "15/11/2020",
            },
            {
                "Trial name": "T1",
                "Occ": "1",
                "Loc_no": "7",
                "Country": "MEXICO",
                "Loc_desc": "SITE",
                "Cycle": "2020",
                "Trait name": "HARVEST_FINISHING_DATE",
                "Value": "01/05/2023",
            },
            {
                "Trial name": "T2",
                "Occ": "2",
                "Loc_no": "8",
                "Country": "KENYA",
                "Loc_desc": "SITE2",
                "Cycle": "2019-20",
                "Trait name": "SOWING_DATE_TEXT",
                "Value": "00/05/2019",
            },
        ]
    )
    raw.to_csv(trial_dir / "trial_EnvData.xls", sep="\t", index=False)
    non_date = raw.iloc[[0]].copy()
    non_date["Trait name"] = "IRRIGATION_AFTER_SOWING"
    non_date["Value"] = "100"
    non_date.to_csv(trial_dir / "trial_no_dates_EnvData.xls", sep="\t", index=False)

    candidates, files = discover_date_candidates(trial_root, target_rows)
    supplement, resolution, conflicts = resolve_candidates(target_rows, candidates)

    assert files["status"].tolist() == ["read", "read"]
    assert supplement["env_id"].tolist() == ["T1|1|7|MEXICO|SITE|2020"]
    assert supplement["sowing_date"].tolist() == ["2020-11-15"]
    assert supplement["harvest_finish_date"].isna().all()
    second = resolution.set_index("env_id").loc["T2|2|8|KENYA|SITE2|2019-20"]
    assert second["sowing_date_resolution"] == "partial_month_year_review_only"
    assert conflicts.empty


def test_raw_trial_date_recovery_reports_conflicting_full_dates() -> None:
    targets = pd.DataFrame(
        {
            "env_id": ["E1"],
            "Cycle": ["2020"],
        }
    )
    candidates = pd.DataFrame(
        {
            "target_env_id": ["E1", "E1"],
            "date_field": ["sowing_date", "sowing_date"],
            "parsed_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
            "match_method": ["exact_env_id", "exact_env_id"],
            "cycle_date_status": ["cycle_plausible", "cycle_plausible"],
            "parse_status": ["full_date", "full_date"],
            "source_file": ["a", "b"],
        }
    )

    supplement, resolution, conflicts = resolve_candidates(targets, candidates)

    assert supplement.empty
    assert resolution.loc[0, "sowing_date_resolution"] == "conflicting_full_dates"
    assert conflicts.loc[0, "candidate_dates"] == "2020-01-01;2020-01-02"


def test_raw_trial_date_recovery_preserves_curated_base_supplement(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "curated.tsv"
    pd.DataFrame(
        {
            "env_id": ["E1"],
            "sowing_date": ["15/11/2020"],
            "provenance": ["manual_review"],
        }
    ).to_csv(base_path, sep="\t", index=False)
    recovered = pd.DataFrame(
        columns=["env_id", "sowing_date", "emergence_date", "harvest_start_date", "harvest_finish_date"]
    )

    combined = merge_base_supplement(recovered, base_path)

    assert combined.loc[0, "sowing_date"] == "2020-11-15"
    assert combined.loc[0, "provenance"] == "manual_review"


def test_matched_weather_runner_freezes_and_verifies_complete_experiment() -> None:
    repo = Path(__file__).resolve().parents[1]
    matched = (
        repo / "scripts" / "run_matched_weather_recovery_comparison.sh"
    ).read_text()
    pipeline = (repo / "scripts" / "run_weather_recovery_pipeline.sh").read_text()
    multitrait = (
        repo / "scripts" / "run_multitrait_quantitative_baseline.sh"
    ).read_text()

    assert matched.index('run_weather_variant "$BASELINE_TAG"') < matched.index(
        'run_weather_variant "$CORRECTED_TAG"'
    )
    assert 'MULTITRAIT_WEIGHT_POWER="0"' in matched
    assert 'MULTITRAIT_INCLUDE_DISABLED_KERNELS="K_E_TGW_V2"' in matched
    assert 'MULTITRAIT_SEEDS="$SEEDS"' in matched
    assert 'MULTITRAIT_MODES="$MODES"' in matched
    assert 'MULTITRAIT_TRAITS="$TRAITS"' in matched
    assert 'verify_variant "$variant"' in matched
    assert "run_is_complete" in multitrait
    assert "training_configuration" in multitrait
    assert "expected_retained_traits" in multitrait
    assert "MIN_TRAIN_ROWS_PER_TRAIT" in multitrait
    assert "MIN_EVAL_ROWS_PER_TRAIT" in multitrait
    for suffix in [
        "contract",
        "paired",
        "trait_summary",
        "macro_summary",
        "trait_availability",
        "adoption_decision",
    ]:
        assert suffix in matched

    assert '--seeds "$SEEDS"' in pipeline
    assert '--modes "$MODES"' in pipeline
    assert '--traits "$TRAITS"' in pipeline
