from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

import server_training_pipeline.phase6a_environment_source_recovery as recovery

from server_training_pipeline.phase6a_environment_source_recovery import (
    CMIP6_IDENTITY_FIELDS,
    CDS_DATASET,
    CDS_REQUEST_CONCURRENCY,
    CDS_REQUEST_TIMEOUT_SECONDS,
    CDS_RETRY_MAX,
    CDS_RETRY_SLEEP_SECONDS,
    build_cds_request_inventory,
    build_cmip6_preregistration_requirement,
    build_environment_map,
    build_request_inventory,
    build_soil_request_inventory,
    broad_cycle_archive_bounds,
    canonical_soilgrids_value,
    cds_api_payload,
    management_unit_resolution,
    management_value_outliers,
    parse_open_meteo,
    request_identity,
    soilgrids_wcs_url,
    soilgrids_completion_counts,
    SoilGridsStructuralUnavailable,
    stable_json_sha256,
    write_json_atomic,
)


def test_cds_fetch_policy_is_serial_and_bounded() -> None:
    assert CDS_REQUEST_CONCURRENCY == 1
    assert CDS_RETRY_MAX == 5
    assert CDS_RETRY_SLEEP_SECONDS == 30
    assert CDS_REQUEST_TIMEOUT_SECONDS == 120


def test_cds_provider_oserror_is_retryable_not_storage_failure(
    tmp_path: Path, monkeypatch
) -> None:
    contract_dir = tmp_path / "contract"
    cache_dir = tmp_path / "cache"
    contract_dir.mkdir()
    inventory = pd.DataFrame(
        [
            {
                "request_id": "a" * 64,
                "source_request_id": "b" * 64,
                "dataset": CDS_DATASET,
                "latitude": "34.55",
                "longitude": "69.2",
                "request_start_date": "2015-10-05",
                "request_end_date": "2016-05-01",
                "variable": ";".join(recovery.CDS_VARIABLES),
                "data_format": "csv",
                "mapped_environment_count": "1",
                "request_payload_json": "{}",
                "request_status": "READY_TO_FETCH",
            }
        ]
    )
    inventory_path = contract_dir / "cds_era5_land_request_inventory.tsv"
    inventory.to_csv(inventory_path, sep="\t", index=False, lineterminator="\n")
    inventory_sha = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    (contract_dir / "environment_source_contract.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "artifacts": {inventory_path.name: inventory_sha},
            }
        ),
        encoding="utf-8",
    )

    class ProviderJobError(OSError):
        pass

    class FailingClient:
        def retrieve(self, *args, **kwargs):
            raise ProviderJobError("CDS job failed")

    fake_cdsapi = SimpleNamespace(Client=lambda **kwargs: FailingClient())
    monkeypatch.setitem(sys.modules, "cdsapi", fake_cdsapi)
    monkeypatch.setattr(recovery, "credential_present", lambda: True)

    provenance = recovery.run_cds_fetch(contract_dir, cache_dir, limit=0)
    index = pd.read_csv(cache_dir / "cds_era5_land_fetch_index.tsv", sep="\t")

    assert provenance["run_status"] == "PARTIAL"
    assert provenance["status_counts"] == {"FAILED_RETRYABLE": 1}
    assert index.iloc[0].status == "FAILED_RETRYABLE"
    assert "ProviderJobError:CDS job failed" in index.iloc[0].detail


def test_atomic_json_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "request.json"
    write_json_atomic(target, {"status": "PASS", "request_id": "abc"})
    assert json.loads(target.read_text(encoding="utf-8"))["request_id"] == "abc"
    assert not target.with_suffix(".json.tmp").exists()


def test_environment_inventory_uses_exact_ids_and_sowing_relative_windows() -> None:
    static = pd.DataFrame(
        {
            "environment_id": ["E1", "E2", "E3"],
            "latitude": [20.0, 21.0, 22.0],
            "longitude": [-100.0, -101.0, -102.0],
            "elevation_m": [1000, 1100, 1200],
        }
    )
    weather = pd.DataFrame(
        {
            "env_id": ["E1", "E1", "E2"],
            "sowing_date": ["2020-11-15", "2020-11-15", "2020-12-01"],
        }
    )
    mapped = build_environment_map(static, weather)
    assert mapped["status"].tolist() == [
        "READY_TO_FETCH",
        "READY_TO_FETCH",
        "BLOCKED_NO_TRIAL_METADATA_CANDIDATE",
    ]
    assert mapped.loc[0, "request_start_date"] == "2020-10-16"
    assert mapped.loc[0, "request_end_date"] == "2021-05-13"
    inventory = build_request_inventory(mapped)
    assert len(inventory) == 2
    for row in inventory.itertuples(index=False):
        expected = stable_json_sha256(
            request_identity(
                float(row.latitude),
                float(row.longitude),
                row.request_start_date,
                row.request_end_date,
            )
        )
        assert row.request_id == expected


def test_missing_sowing_uses_broad_raw_cycle_archive_without_imputation() -> None:
    static = pd.DataFrame(
        {
            "environment_id": ["10ESWYT|11|19402|TUNISIA|BEJA|88-89"],
            "latitude": [36.73333],
            "longitude": [9.13333],
            "elevation_m": [150],
        }
    )
    weather = pd.DataFrame(columns=["env_id", "sowing_date"])
    mapped = build_environment_map(static, weather)
    assert mapped.iloc[0].status == "BLOCKED_NO_TRIAL_METADATA_CANDIDATE"
    assert mapped.iloc[0].source_archive_status == "READY_BROAD_CYCLE_ARCHIVE"
    assert mapped.iloc[0].request_start_date == "1987-07-01"
    assert mapped.iloc[0].request_end_date == "1989-12-31"
    assert len(build_request_inventory(mapped)) == 1
    assert broad_cycle_archive_bounds(
        "10ESWYT|11|19402|TUNISIA|BEJA|88-89"
    ) == ("1987-07-01", "1989-12-31")


def test_environment_inventory_accepts_only_certified_trial_aliases() -> None:
    static = pd.DataFrame(
        {
            "environment_id": [
                "26TH ELITE SPRING WHEAT YT|1|100|MEXICO|SITE|2005",
                "OTHER TRIAL|2|100|MEXICO|SITE|2005",
            ],
            "latitude": [20.0, 20.0],
            "longitude": [-100.0, -100.0],
            "elevation_m": [1000, 1000],
        }
    )
    weather = pd.DataFrame(
        {
            "env_id": [
                "26ESWYT|1|100|MEXICO|SITE|2005",
                "UNRELATED|2|100|MEXICO|SITE|2005",
            ],
            "sowing_date": ["2005-11-15", "2005-11-15"],
        }
    )
    registry = pd.DataFrame(
        {
            "trial_key": ["26ESWYT", "UNRELATED", "OTHER TRIAL"],
            "cycle": ["2005", "2005", "2005"],
            "trial_name": ["26TH ELITE SPRING WHEAT YT", "UNRELATED", "OTHER TRIAL"],
            "trial_code": ["26ESWYT", "UNRELATED", "OTHER"],
        }
    )
    aliases = pd.DataFrame(
        columns=["source_env_id", "target_env_id", "mapping_status", "alias_decision"]
    )
    mapped = build_environment_map(static, weather, registry, aliases)
    assert mapped.loc[0, "status"] == "READY_TO_FETCH"
    assert mapped.loc[0, "metadata_resolution"] == (
        "UNIQUE_NONTRIAL_IDENTITY_AND_CERTIFIED_TRIAL_GROUP"
    )
    assert mapped.loc[1, "status"] == "BLOCKED_UNCERTIFIED_TRIAL_ALIAS"


def test_management_units_separate_depths_from_legacy_marks() -> None:
    rows = []
    for feature in (
        "CALCULATED_OF_TOTAL_WATER_APPLIED_BY_IRRIGATION",
        "ESTIMATE_OF_TOTAL_WATER_APPLIED_BY_IRRIGATION",
    ):
        rows.append(
            {"Trait_name": feature, "Unit": "mm", "Value": "125", "source_file": "a.xls"}
        )
    for feature in (
        "K_FERTILIZER_APPLIED_OLD",
        "N_FERTILIZER_APPLIED_OLD",
        "P_FERTILIZER_APPLIED_OLD",
    ):
        rows.append(
            {"Trait_name": feature, "Unit": "mark", "Value": "1", "source_file": "b.xls"}
        )
    resolved = management_unit_resolution(pd.DataFrame(rows)).set_index("feature")
    assert resolved.loc[
        "ESTIMATE_OF_TOTAL_WATER_APPLIED_BY_IRRIGATION", "status"
    ] == "RESOLVED_CANONICAL_MM"
    assert resolved.loc[
        "N_FERTILIZER_APPLIED_OLD", "status"
    ] == "RESOLVED_BINARY_MARK_NOT_AMOUNT_EXCLUDED_FROM_CORE"
    assert management_value_outliers(pd.DataFrame(rows)).empty


def test_implausible_irrigation_depth_is_quarantined() -> None:
    frame = pd.DataFrame(
        [
            {
                "Trait_name": "ESTIMATE_OF_TOTAL_WATER_APPLIED_BY_IRRIGATION",
                "Unit": "mm",
                "Value": "45000",
                "source_file": "a.xls",
                "trial_dir": "trial",
                "Trial_name": "TRIAL",
                "Occ": "1",
                "Loc_no": "100",
                "Country": "MEXICO",
                "Loc_desc": "SITE",
                "Cycle": "2000",
            }
        ]
    )
    resolved = management_unit_resolution(frame).set_index("feature").loc[
        "ESTIMATE_OF_TOTAL_WATER_APPLIED_BY_IRRIGATION"
    ]
    assert resolved.status == "RESOLVED_CANONICAL_MM_WITH_QUARANTINED_OUTLIERS"
    outliers = management_value_outliers(frame)
    assert len(outliers) == 1
    assert outliers.iloc[0].numeric_value == 45000


def test_open_meteo_parser_requires_units_and_daily_coverage() -> None:
    row = pd.Series(
        {
            "request_id": "abc",
            "request_start_date": "2020-01-01",
            "request_end_date": "2020-01-02",
            "latitude": "20",
            "longitude": "-100",
        }
    )
    daily = {
        "time": ["2020-01-01", "2020-01-02"],
        "temperature_2m_mean": [20.0, 21.0],
        "temperature_2m_max": [25.0, 26.0],
        "temperature_2m_min": [15.0, 16.0],
        "precipitation_sum": [0.0, 2.0],
        "shortwave_radiation_sum": [18.0, 19.0],
        "et0_fao_evapotranspiration": [3.0, 3.2],
        "relative_humidity_2m_mean": [60.0, 58.0],
        "wind_speed_10m_mean": [2.0, 2.2],
    }
    units = {
        "time": "iso8601",
        "temperature_2m_mean": "°C",
        "temperature_2m_max": "°C",
        "temperature_2m_min": "°C",
        "precipitation_sum": "mm",
        "shortwave_radiation_sum": "MJ/m²",
        "et0_fao_evapotranspiration": "mm",
        "relative_humidity_2m_mean": "%",
        "wind_speed_10m_mean": "m/s",
    }
    frame, metadata = parse_open_meteo(
        json.dumps({"daily": daily, "daily_units": units}).encode(), row
    )
    assert len(frame) == 2
    assert frame["precipitation_sum"].tolist() == [0.0, 2.0]
    assert np.isfinite(frame["temperature_2m_mean"]).all()
    assert metadata["daily_units"]["precipitation_sum"] == "mm"


def test_authoritative_request_inventories_are_identifier_deterministic() -> None:
    environment_map = pd.DataFrame(
        {
            "environment_id": ["E1", "E2"],
            "latitude": [20.0, 20.0],
            "longitude": [-100.0, -100.0],
            "request_start_date": ["2020-10-01", "2021-10-01"],
            "request_end_date": ["2021-04-28", "2022-04-28"],
            "daily_request_id": ["a", "b"],
            "source_archive_status": [
                "READY_SOWING_RELATIVE_ARCHIVE",
                "READY_SOWING_RELATIVE_ARCHIVE",
            ],
            "status": ["READY_TO_FETCH", "READY_TO_FETCH"],
        }
    )
    daily = build_request_inventory(environment_map)
    cds = build_cds_request_inventory(daily)
    soil = build_soil_request_inventory(environment_map)
    assert len(cds) == 2
    assert cds["request_id"].is_unique
    assert cds["dataset"].eq(CDS_DATASET).all()
    payload = cds_api_payload(cds.iloc[0])
    assert set(payload) == {"variable", "location", "date", "data_format"}
    assert len(soil) == 1
    assert soil.iloc[0].mapped_environment_count == 2
    assert soil.iloc[0].coverage_request_count == 20


def test_soilgrids_request_uses_documented_wcs_coverage() -> None:
    url = soilgrids_wcs_url("wv0033", "0-5cm", 1000.0, 2000.0)
    assert "COVERAGEID=wv0033_0-5cm_Q0.5" in url
    assert url.count("SUBSET=") == 2
    assert "GEOTIFF_INT16" in url


def test_soilgrids_physical_values_reject_unlabelled_zero_nodata() -> None:
    assert canonical_soilgrids_value("wv0033", 284) == 28.4
    assert canonical_soilgrids_value("cfvo", 0) == 0
    with np.testing.assert_raises_regex(
        SoilGridsStructuralUnavailable, "water-content"
    ):
        canonical_soilgrids_value("wv0033", 0)
    with np.testing.assert_raises_regex(
        SoilGridsStructuralUnavailable, "bulk-density"
    ):
        canonical_soilgrids_value("bdod", 0)


def test_soilgrids_structural_missingness_is_terminally_resolved() -> None:
    index = pd.DataFrame(
        {
            "status": [
                "FETCHED",
                "CACHED",
                "STRUCTURALLY_UNAVAILABLE_SOIL_CELL",
                "PENDING_LIMIT",
            ]
        }
    )
    assert soilgrids_completion_counts(index) == (2, 1, 3)


def test_cmip6_retrieval_remains_blocked_without_member_identity() -> None:
    requirements = build_cmip6_preregistration_requirement()
    assert len(requirements) == 4
    assert requirements["declared_source_count"].eq(0).all()
    assert requirements["status"].eq(
        "BLOCKED_ENSEMBLE_IDENTITY_NOT_PREREGISTERED"
    ).all()
    assert set(CMIP6_IDENTITY_FIELDS).issubset(
        set(requirements.iloc[0].required_identity_fields.split(";"))
    )
