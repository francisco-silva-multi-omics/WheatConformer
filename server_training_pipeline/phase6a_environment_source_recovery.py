from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import time
from typing import Any
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "phase6a_environment_source_recovery_v10"
DEFAULT_CONTRACT = Path("audit/v2/phase6a_environment_source_contract_v10")
DEFAULT_CACHE = Path("environment/v2/phase6a_openmeteo_era5_daily_v10")
DEFAULT_CDS_CACHE = Path("environment/v2/phase6a_cds_era5_land_daily_v4")
DEFAULT_SOIL_CACHE = Path("environment/v2/phase6a_soilgrids_water_v4")
DEFAULT_STATUS = Path("audit/v2/phase6a_environment_source_staging_v8")
PHASE6A = Path("audit/v2/phase6a_environmental_projection_readiness_v1")
STATIC_BACKCAST = PHASE6A / "backcast/historical_static_site_backcast.parquet"
TRIAL_WEATHER_MANIFEST = Path("environment/trial_weather_fetch_manifest.tsv")
ENVDATA = Path("server_phase5_parity_bundle/artifacts/environment/envdata.tsv")
STAGE1_V2_REGISTRY = Path(
    "audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v8/raw_trial_registry.tsv"
)
STAGE1_V2_ENVIRONMENT_ALIASES = Path(
    "audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v8/environment_alias_registry_v2.tsv"
)

OPEN_METEO_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_MODEL = "era5"
OPEN_METEO_COVERAGE_START = pd.Timestamp("1950-01-01")
CDS_DATASET = "reanalysis-era5-land-timeseries"
CDS_REQUEST_CONCURRENCY = 1
CDS_RETRY_MAX = 5
CDS_RETRY_SLEEP_SECONDS = 30
CDS_REQUEST_TIMEOUT_SECONDS = 120
CDS_VARIABLES = (
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_pressure",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
)
SOILGRIDS_ENDPOINT = "https://maps.isric.org/mapserv"
SOILGRIDS_NATIVE_CRS = "http://www.opengis.net/def/crs/EPSG/0/152160"
SOILGRIDS_PROJ = "+proj=igh +datum=WGS84 +units=m +no_defs"
SOILGRIDS_PROPERTIES = {
    "wv0033": 10.0,
    "wv1500": 10.0,
    "cfvo": 10.0,
    "bdod": 100.0,
}
SOILGRIDS_DEPTHS = ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm")
SOIL_NUMERIC_COMPLETE_STATUSES = ("FETCHED", "CACHED")
SOIL_TERMINAL_RESOLVED_STATUSES = (
    "FETCHED",
    "CACHED",
    "STRUCTURALLY_UNAVAILABLE_SOIL_CELL",
)
CMIP6_SCENARIOS = ("SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5")
CMIP6_IDENTITY_FIELDS = (
    "source_id",
    "institution_id",
    "experiment_id",
    "variant_label",
    "grid_label",
    "version",
)
DAILY_VARIABLES = (
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
)
EXPECTED_UNITS = {
    "temperature_2m_mean": "°C",
    "temperature_2m_max": "°C",
    "temperature_2m_min": "°C",
    "precipitation_sum": "mm",
    "shortwave_radiation_sum": "MJ/m²",
    "et0_fao_evapotranspiration": "mm",
    "relative_humidity_2m_mean": "%",
    "wind_speed_10m_mean": "m/s",
}
MANAGEMENT_FIELDS = (
    "CALCULATED_OF_TOTAL_WATER_APPLIED_BY_IRRIGATION",
    "ESTIMATE_OF_TOTAL_WATER_APPLIED_BY_IRRIGATION",
    "K_FERTILIZER_APPLIED_OLD",
    "N_FERTILIZER_APPLIED_OLD",
    "P_FERTILIZER_APPLIED_OLD",
)
FORBIDDEN_OUTPUT_TOKENS = ("prediction", "future_matrix", "rcp_matrix", "ssp_matrix")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_date(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def normalize_identifier(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9]+", "", text.upper())


def split_environment_id(value: Any) -> list[str] | None:
    parts = str(value).split("|")
    return parts if len(parts) == 6 else None


def nontrial_environment_key(value: Any) -> tuple[str, ...] | None:
    parts = split_environment_id(value)
    return tuple(normalize_identifier(token) for token in parts[1:]) if parts else None


def broad_cycle_archive_bounds(environment_id: str) -> tuple[str, str] | None:
    parts = split_environment_id(environment_id)
    if parts is None:
        return None
    cycle = str(parts[-1]).strip()
    years = [int(value) for value in re.findall(r"\d{2,4}", cycle)]
    if not years:
        return None

    def expand(value: int) -> int:
        if value >= 1000:
            return value
        return 1900 + value if value >= 50 else 2000 + value

    start_year = expand(years[0])
    end_year = expand(years[-1])
    if end_year < start_year:
        end_year += 100
    if not (1950 <= start_year <= 2100 and start_year <= end_year <= start_year + 2):
        return None
    return f"{start_year - 1:04d}-07-01", f"{end_year:04d}-12-31"


def certified_trial_groups(registry: pd.DataFrame) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for row in registry.itertuples(index=False):
        group = f"{normalize_identifier(row.trial_key)}|{normalize_identifier(row.cycle)}"
        for value in (row.trial_key, row.trial_name, row.trial_code):
            if pd.notna(value) and str(value).strip():
                groups[normalize_identifier(value)].add(group)
    return groups


def request_identity(
    latitude: float, longitude: float, start_date: str, end_date: str
) -> dict[str, Any]:
    return {
        "provider": "open_meteo",
        "model": OPEN_METEO_MODEL,
        "latitude": round(float(latitude), 5),
        "longitude": round(float(longitude), 5),
        "start_date": start_date,
        "end_date": end_date,
        "daily_variables": list(DAILY_VARIABLES),
        "timezone": "GMT",
    }


def open_meteo_url(row: pd.Series) -> str:
    params = {
        "latitude": f"{float(row['latitude']):.5f}",
        "longitude": f"{float(row['longitude']):.5f}",
        "start_date": str(row["request_start_date"]),
        "end_date": str(row["request_end_date"]),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "GMT",
        "models": OPEN_METEO_MODEL,
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }
    return OPEN_METEO_ENDPOINT + "?" + urllib.parse.urlencode(params)


def credential_present() -> bool:
    if os.environ.get("CDSAPI_KEY") or os.environ.get("CDS_API_KEY"):
        return True
    return (Path.home() / ".cdsapirc").is_file()


def provider_readiness() -> pd.DataFrame:
    cds_auth = credential_present()
    rows = [
        {
            "source": "local_trial_envdata_and_locdata",
            "role": "site_sowing_and_management_metadata",
            "access_status": "READY_LOCAL",
            "authoritative_for_phase6a": True,
            "detail": "Raw trial EnvData/LocData plus certified parity extraction",
        },
        {
            "source": "open_meteo_era5",
            "role": "public_daily_historical_backcast_diagnostic",
            "access_status": "READY_PUBLIC_API",
            "authoritative_for_phase6a": False,
            "detail": "Independent daily archive candidate; requires cross-provider certification before authoritative use",
        },
        {
            "source": "copernicus_cds_era5_land",
            "role": "authoritative_daily_historical_reference",
            "access_status": "READY_AUTH" if cds_auth else "BLOCKED_MISSING_CDS_CREDENTIALS",
            "authoritative_for_phase6a": True,
            "detail": "Credential presence checked without reading or logging any token",
        },
        {
            "source": "soilgrids_wcs_or_webdav",
            "role": "static_soil_water_capacity_inputs",
            "access_status": "READY_PUBLIC_WCS",
            "authoritative_for_phase6a": True,
            "detail": "Bounded WCS adapter uses median water-content, coarse-fragment, and bulk-density layers; missing cells remain missing",
        },
        {
            "source": "copernicus_cds_cmip6",
            "role": "member_resolved_historical_and_ssp_archive",
            "access_status": "BLOCKED_ENSEMBLE_IDENTITY_NOT_PREREGISTERED",
            "authoritative_for_phase6a": True,
            "detail": (
                "SSP labels are frozen, but source_id/institution_id/variant_label/grid_label/version are not; "
                f"CDS credentials_present={cds_auth}. No CMIP6 values are fetched or inspected"
            ),
        },
        {
            "source": "esgf",
            "role": "member_resolved_cmip6_alternative",
            "access_status": "BLOCKED_ESGF_CLIENT_NOT_INSTALLED"
            if shutil.which("esgpull") is None
            else "READY_CLIENT",
            "authoritative_for_phase6a": True,
            "detail": "Member identity and checksums are mandatory; no ensemble averaging before derivation",
        },
    ]
    return pd.DataFrame(rows)


def management_unit_resolution(envdata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    trait = envdata["Trait_name"].astype(str).str.upper()
    for field in MANAGEMENT_FIELDS:
        local = envdata[trait.eq(field)].copy()
        units = sorted(local["Unit"].dropna().astype(str).str.strip().unique())
        values = pd.to_numeric(local["Value"], errors="coerce")
        if field.endswith("WATER_APPLIED_BY_IRRIGATION"):
            expected = ["mm"]
            outlier_count = int(((values < 0) | (values > 5000)).sum())
            if units != expected:
                status = "BLOCKED_UNIT_CONFLICT"
            elif outlier_count:
                status = "RESOLVED_CANONICAL_MM_WITH_QUARANTINED_OUTLIERS"
            else:
                status = "RESOLVED_CANONICAL_MM"
            role = "quantitative_irrigation_depth_mm"
            projection_action = "RETAIN_HISTORICAL;_FUTURE_REQUIRES_EXPLICIT_MM_SCENARIO"
        else:
            expected = ["mark"]
            mark_ok = units == expected and set(values.dropna().unique()).issubset({0.0, 1.0})
            status = (
                "RESOLVED_BINARY_MARK_NOT_AMOUNT_EXCLUDED_FROM_CORE"
                if mark_ok
                else "BLOCKED_UNIT_OR_VALUE_CONFLICT"
            )
            role = "binary_fertilizer_applied_indicator"
            projection_action = (
                "REJECT_AS_NUTRIENT_AMOUNT;_DERIVE_KG_HA_FROM_FERTILIZER_KG/HA_X_COMPOSITION_WHERE_AVAILABLE"
            )
            outlier_count = 0
        rows.append(
            {
                "feature": field,
                "raw_rows": len(local),
                "source_files": local["source_file"].nunique(),
                "observed_units": ";".join(units),
                "expected_units": ";".join(expected),
                "finite_values": int(values.notna().sum()),
                "minimum": float(values.min()) if values.notna().any() else np.nan,
                "maximum": float(values.max()) if values.notna().any() else np.nan,
                "physical_range": "0_to_5000_mm" if field.endswith("WATER_APPLIED_BY_IRRIGATION") else "binary_0_or_1",
                "quarantined_outlier_rows": outlier_count,
                "resolved_role": role,
                "projection_core_action": projection_action,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def management_value_outliers(envdata: pd.DataFrame) -> pd.DataFrame:
    trait = envdata["Trait_name"].astype(str).str.upper()
    local = envdata[
        trait.isin(
            [
                "CALCULATED_OF_TOTAL_WATER_APPLIED_BY_IRRIGATION",
                "ESTIMATE_OF_TOTAL_WATER_APPLIED_BY_IRRIGATION",
            ]
        )
    ].copy()
    local["numeric_value"] = pd.to_numeric(local["Value"], errors="coerce")
    local = local[(local["numeric_value"] < 0) | (local["numeric_value"] > 5000)].copy()
    local["quarantine_reason"] = "outside_preregistered_0_to_5000_mm_physical_screen"
    columns = [
        "source_file",
        "trial_dir",
        "Trial_name",
        "Occ",
        "Loc_no",
        "Country",
        "Loc_desc",
        "Cycle",
        "Trait_name",
        "Value",
        "Unit",
        "numeric_value",
        "quarantine_reason",
    ]
    return local.reindex(columns=columns).sort_values(
        ["Trait_name", "numeric_value"], kind="stable"
    )


def build_environment_map(
    static: pd.DataFrame,
    weather: pd.DataFrame,
    trial_registry: pd.DataFrame | None = None,
    environment_aliases: pd.DataFrame | None = None,
) -> pd.DataFrame:
    weather = weather.copy()
    weather["env_id"] = weather["env_id"].astype(str)
    weather["_sowing_date"] = weather["sowing_date"].map(canonical_date)
    by_environment = {key: group for key, group in weather.groupby("env_id", sort=False)}
    nontrial_candidates: dict[tuple[str, ...] | None, list[str]] = defaultdict(list)
    for environment_id in by_environment:
        key = nontrial_environment_key(environment_id)
        if key is not None:
            nontrial_candidates[key].append(environment_id)
    trial_groups = certified_trial_groups(trial_registry) if trial_registry is not None else {}
    accepted_aliases: dict[str, str] = {}
    if environment_aliases is not None:
        accepted = environment_aliases[
            environment_aliases["mapping_status"].eq("ACCEPTED_ALIAS")
            & environment_aliases["alias_decision"].eq("ACCEPT")
        ]
        accepted_aliases = dict(
            zip(accepted["target_env_id"].astype(str), accepted["source_env_id"].astype(str))
        )
    rows: list[dict[str, Any]] = []
    for record in static.itertuples(index=False):
        environment_id = str(record.environment_id)
        latitude = pd.to_numeric(pd.Series([record.latitude]), errors="coerce").iloc[0]
        longitude = pd.to_numeric(pd.Series([record.longitude]), errors="coerce").iloc[0]
        local = by_environment.get(environment_id)
        metadata_source_environment_id = environment_id if local is not None else ""
        metadata_resolution = "EXACT_ENVIRONMENT_ID" if local is not None else "UNRESOLVED"
        candidate_count = 0
        if local is None:
            explicit_source = accepted_aliases.get(environment_id)
            if explicit_source in by_environment:
                local = by_environment[explicit_source]
                metadata_source_environment_id = explicit_source
                metadata_resolution = "CERTIFIED_STAGE1_V2_ENVIRONMENT_ALIAS"
            else:
                key = nontrial_environment_key(environment_id)
                candidates = nontrial_candidates.get(key, []) if key is not None else []
                candidate_count = len(candidates)
                target_parts = split_environment_id(environment_id)
                if len(candidates) == 1 and target_parts is not None:
                    source = candidates[0]
                    source_parts = split_environment_id(source)
                    same_trial_group = bool(
                        source_parts
                        and trial_groups.get(normalize_identifier(target_parts[0]), set())
                        & trial_groups.get(normalize_identifier(source_parts[0]), set())
                    )
                    if same_trial_group:
                        local = by_environment[source]
                        metadata_source_environment_id = source
                        metadata_resolution = (
                            "UNIQUE_NONTRIAL_IDENTITY_AND_CERTIFIED_TRIAL_GROUP"
                        )
        sowing_dates: list[str] = []
        if local is not None:
            sowing_dates = sorted(set(local["_sowing_date"].dropna().astype(str)))
        if not np.isfinite(latitude) or not np.isfinite(longitude):
            status = "BLOCKED_MISSING_COORDINATES"
        elif local is None and candidate_count == 0:
            status = "BLOCKED_NO_TRIAL_METADATA_CANDIDATE"
        elif local is None and candidate_count == 1:
            status = "BLOCKED_UNCERTIFIED_TRIAL_ALIAS"
        elif local is None:
            status = "BLOCKED_AMBIGUOUS_NONTRIAL_IDENTITY"
        elif not sowing_dates:
            status = "BLOCKED_MISSING_SOWING_DATE"
        elif len(sowing_dates) > 1:
            status = "BLOCKED_CONFLICTING_SOWING_DATES"
        else:
            status = "READY_TO_FETCH"
        sowing_date = sowing_dates[0] if len(sowing_dates) == 1 else ""
        start_date = ""
        end_date = ""
        request_id = ""
        source_archive_status = ""
        source_archive_window_definition = ""
        if sowing_date:
            sowing = pd.Timestamp(sowing_date)
            start_date = (sowing - timedelta(days=30)).strftime("%Y-%m-%d")
            end_date = (sowing + timedelta(days=179)).strftime("%Y-%m-%d")
            if sowing < OPEN_METEO_COVERAGE_START:
                source_archive_status = "BLOCKED_BEFORE_ERA5_LAND_COVERAGE"
            elif np.isfinite(latitude) and np.isfinite(longitude):
                source_archive_status = "READY_SOWING_RELATIVE_ARCHIVE"
                source_archive_window_definition = (
                    "sowing_minus_30_through_sowing_plus_179_inclusive"
                )
            else:
                source_archive_status = "BLOCKED_MISSING_COORDINATES"
        elif np.isfinite(latitude) and np.isfinite(longitude):
            broad_bounds = broad_cycle_archive_bounds(environment_id)
            if broad_bounds is None:
                source_archive_status = "BLOCKED_UNPARSEABLE_TRIAL_CYCLE"
            else:
                start_date, end_date = broad_bounds
                source_archive_status = "READY_BROAD_CYCLE_ARCHIVE"
                source_archive_window_definition = (
                    "previous_July_1_through_terminal_cycle_December_31;_"
                    "raw_archive_only_no_sowing_imputation"
                )
        else:
            source_archive_status = "BLOCKED_MISSING_COORDINATES"
        if source_archive_status.startswith("READY_"):
            request_id = stable_json_sha256(
                request_identity(latitude, longitude, start_date, end_date)
            )
        rows.append(
            {
                "environment_id": environment_id,
                "latitude": latitude,
                "longitude": longitude,
                "sowing_date": sowing_date,
                "request_start_date": start_date,
                "request_end_date": end_date,
                "historical_window_definition": source_archive_window_definition,
                "daily_request_id": request_id,
                "exact_trial_metadata_rows": 0 if local is None else len(local),
                "metadata_source_environment_id": metadata_source_environment_id,
                "metadata_resolution": metadata_resolution,
                "nontrial_identity_candidate_count": candidate_count,
                "distinct_sowing_dates": len(sowing_dates),
                "source_archive_status": source_archive_status,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def build_request_inventory(environment_map: pd.DataFrame) -> pd.DataFrame:
    ready = environment_map[
        environment_map["source_archive_status"].str.startswith("READY_")
    ].copy()
    columns = [
        "daily_request_id",
        "latitude",
        "longitude",
        "request_start_date",
        "request_end_date",
    ]
    inventory = ready[columns].drop_duplicates().sort_values("daily_request_id", kind="stable")
    counts = ready.groupby("daily_request_id").size().rename("mapped_environment_count")
    inventory = inventory.merge(counts, left_on="daily_request_id", right_index=True)
    inventory = inventory.rename(columns={"daily_request_id": "request_id"})
    inventory["provider"] = "open_meteo"
    inventory["model"] = OPEN_METEO_MODEL
    inventory["required_daily_variables"] = ";".join(DAILY_VARIABLES)
    archive_kinds = (
        ready.groupby("daily_request_id")["source_archive_status"]
        .agg(lambda values: ";".join(sorted(set(values))))
        .rename("source_archive_status")
    )
    inventory = inventory.merge(archive_kinds, left_on="request_id", right_index=True)
    inventory["request_status"] = "READY_TO_FETCH"
    return inventory.reset_index(drop=True)


def cds_request_identity(row: pd.Series | Any) -> dict[str, Any]:
    return {
        "provider": "copernicus_cds",
        "dataset": CDS_DATASET,
        "location": {
            "latitude": round(float(row.latitude), 5),
            "longitude": round(float(row.longitude), 5),
        },
        "date": f"{row.request_start_date}/{row.request_end_date}",
        "variable": list(CDS_VARIABLES),
        "data_format": "csv",
    }


def build_cds_request_inventory(requests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in requests.itertuples(index=False):
        identity = cds_request_identity(row)
        rows.append(
            {
                "request_id": stable_json_sha256(identity),
                "source_request_id": row.request_id,
                "dataset": CDS_DATASET,
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "request_start_date": row.request_start_date,
                "request_end_date": row.request_end_date,
                "variable": ";".join(CDS_VARIABLES),
                "data_format": "csv",
                "mapped_environment_count": int(row.mapped_environment_count),
                "request_payload_json": json.dumps(
                    identity, sort_keys=True, separators=(",", ":")
                ),
                "request_status": (
                    "READY_TO_FETCH"
                    if credential_present()
                    else "BLOCKED_MISSING_CDS_CREDENTIALS"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("request_id", kind="stable").reset_index(drop=True)


def soil_site_identity(latitude: float, longitude: float) -> dict[str, Any]:
    return {
        "provider": "soilgrids_wcs",
        "release": "latest",
        "latitude": round(float(latitude), 5),
        "longitude": round(float(longitude), 5),
        "properties": sorted(SOILGRIDS_PROPERTIES),
        "depths": list(SOILGRIDS_DEPTHS),
        "statistic": "Q0.5",
    }


def build_soil_request_inventory(environment_map: pd.DataFrame) -> pd.DataFrame:
    local = environment_map.dropna(subset=["latitude", "longitude"])[
        ["latitude", "longitude"]
    ].copy()
    local["latitude"] = pd.to_numeric(local["latitude"]).round(5)
    local["longitude"] = pd.to_numeric(local["longitude"]).round(5)
    sites = (
        local
        .drop_duplicates()
        .sort_values(["latitude", "longitude"], kind="stable")
    )
    mapped = local.groupby(["latitude", "longitude"], dropna=True).size()
    rows = []
    for row in sites.itertuples(index=False):
        identity = soil_site_identity(float(row.latitude), float(row.longitude))
        rows.append(
            {
                "site_id": stable_json_sha256(identity),
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "mapped_environment_count": int(mapped.loc[(row.latitude, row.longitude)]),
                "properties": ";".join(sorted(SOILGRIDS_PROPERTIES)),
                "depths": ";".join(SOILGRIDS_DEPTHS),
                "statistic": "Q0.5",
                "coverage_request_count": len(SOILGRIDS_PROPERTIES)
                * len(SOILGRIDS_DEPTHS),
                "request_identity_json": json.dumps(
                    identity, sort_keys=True, separators=(",", ":")
                ),
                "request_status": "READY_TO_FETCH",
            }
        )
    return pd.DataFrame(rows).sort_values("site_id", kind="stable").reset_index(drop=True)


def build_cmip6_preregistration_requirement() -> pd.DataFrame:
    rows = []
    for scenario in CMIP6_SCENARIOS:
        rows.append(
            {
                "generation": "CMIP6",
                "scenario": scenario,
                "required_identity_fields": ";".join(CMIP6_IDENTITY_FIELDS),
                "declared_source_count": 0,
                "declared_member_count": 0,
                "status": "BLOCKED_ENSEMBLE_IDENTITY_NOT_PREREGISTERED",
                "required_action": (
                    "freeze source_id, institution_id, experiment_id, variant_label, "
                    "grid_label, and version before any historical or future CMIP6 retrieval"
                ),
            }
        )
    return pd.DataFrame(rows)


def source_inventory(root: Path, paths: list[Path]) -> pd.DataFrame:
    rows = []
    for relative in paths:
        path = resolve(root, relative)
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "status": "PASS",
            }
        )
    return pd.DataFrame(rows)


def build_contract(root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise SystemExit(f"Fail-if-exists contract directory already exists: {output}")
    required = [
        STATIC_BACKCAST,
        TRIAL_WEATHER_MANIFEST,
        ENVDATA,
        STAGE1_V2_REGISTRY,
        STAGE1_V2_ENVIRONMENT_ALIASES,
    ]
    missing = [str(path) for path in required if not resolve(root, path).is_file()]
    if missing:
        raise SystemExit(f"Missing Phase-6A source inputs: {missing}")
    output.mkdir(parents=True)

    static = pd.read_parquet(resolve(root, STATIC_BACKCAST))
    weather = pd.read_csv(
        resolve(root, TRIAL_WEATHER_MANIFEST), sep="\t", dtype=str, low_memory=False
    )
    envdata = pd.read_csv(resolve(root, ENVDATA), sep="\t", dtype=str, low_memory=False)
    trial_registry = pd.read_csv(
        resolve(root, STAGE1_V2_REGISTRY), sep="\t", dtype=str, low_memory=False
    )
    environment_aliases = pd.read_csv(
        resolve(root, STAGE1_V2_ENVIRONMENT_ALIASES),
        sep="\t",
        dtype=str,
        low_memory=False,
    )
    if static["environment_id"].duplicated().any():
        raise SystemExit("Static Phase-6A environment IDs are not unique")

    environment_map = build_environment_map(
        static, weather, trial_registry, environment_aliases
    )
    requests = build_request_inventory(environment_map)
    cds_requests = build_cds_request_inventory(requests)
    soil_requests = build_soil_request_inventory(environment_map)
    cmip6_requirements = build_cmip6_preregistration_requirement()
    management = management_unit_resolution(envdata)
    management_outliers = management_value_outliers(envdata)
    providers = provider_readiness()
    sources = source_inventory(root, required)

    write_tsv(output / "environment_daily_request_map.tsv", environment_map)
    write_tsv(output / "daily_request_inventory.tsv", requests)
    write_tsv(output / "cds_era5_land_request_inventory.tsv", cds_requests)
    write_tsv(output / "soilgrids_request_inventory.tsv", soil_requests)
    write_tsv(
        output / "cmip6_ensemble_preregistration_requirement.tsv", cmip6_requirements
    )
    write_tsv(output / "management_unit_resolution.tsv", management)
    write_tsv(output / "management_value_outliers.tsv", management_outliers)
    write_tsv(output / "provider_readiness.tsv", providers)
    write_tsv(output / "source_input_manifest.tsv", sources)

    status_counts = environment_map["status"].value_counts().sort_index().to_dict()
    checks = {
        "stage1_v2_environment_ids_unique": environment_map["environment_id"].is_unique,
        "one_terminal_status_per_environment": len(environment_map) == len(static),
        "source_archive_status_nonempty": environment_map[
            "source_archive_status"
        ].ne("").all(),
        "ready_environments_have_request_ids": environment_map.loc[
            environment_map["source_archive_status"].str.startswith("READY_"),
            "daily_request_id",
        ].ne("").all(),
        "blocked_source_environments_have_no_request_ids": environment_map.loc[
            ~environment_map["source_archive_status"].str.startswith("READY_"),
            "daily_request_id",
        ].eq("").all(),
        "normalized_aliases_require_certified_trial_group": environment_map.loc[
            environment_map["metadata_resolution"].eq(
                "UNIQUE_NONTRIAL_IDENTITY_AND_CERTIFIED_TRIAL_GROUP"
            ),
            "metadata_source_environment_id",
        ].ne("").all(),
        "request_ids_unique": requests["request_id"].is_unique,
        "request_ids_reconstruct": all(
            str(row.request_id)
            == stable_json_sha256(
                request_identity(
                    float(row.latitude),
                    float(row.longitude),
                    str(row.request_start_date),
                    str(row.request_end_date),
                )
            )
            for row in requests.itertuples(index=False)
        ),
        "cds_request_ids_unique": cds_requests["request_id"].is_unique,
        "cds_requests_match_historical_inventory": len(cds_requests) == len(requests),
        "cds_request_payloads_reconstruct": all(
            str(row.request_id) == stable_json_sha256(cds_request_identity(row))
            for row in cds_requests.itertuples(index=False)
        ),
        "soil_site_ids_unique": soil_requests["site_id"].is_unique,
        "soil_sites_reconstruct": all(
            str(row.site_id)
            == stable_json_sha256(
                soil_site_identity(float(row.latitude), float(row.longitude))
            )
            for row in soil_requests.itertuples(index=False)
        ),
        "cmip6_retrieval_blocked_until_ensemble_identity_frozen": cmip6_requirements[
            "status"
        ].eq("BLOCKED_ENSEMBLE_IDENTITY_NOT_PREREGISTERED").all(),
        "irrigation_units_resolved": management.loc[
            management["feature"].str.contains("WATER_APPLIED"), "status"
        ].str.startswith("RESOLVED_CANONICAL_MM").all(),
        "implausible_irrigation_values_quarantined": int(
            pd.to_numeric(management["quarantined_outlier_rows"], errors="coerce").sum()
        )
        == len(management_outliers),
        "legacy_fertilizer_marks_not_treated_as_amounts": management.loc[
            management["feature"].str.endswith("FERTILIZER_APPLIED_OLD"), "status"
        ].eq("RESOLVED_BINARY_MARK_NOT_AMOUNT_EXCLUDED_FROM_CORE").all(),
        "no_phenotype_or_outcome_inputs": True,
        "no_future_matrices_or_predictions": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    contract = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol_version": PROTOCOL_VERSION,
        "selection_data": "stage1_v2_environment_identifiers_and_source_metadata_only",
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
        "environment_count": len(environment_map),
        "metadata_complete_environment_count": int(
            environment_map["status"].eq("READY_TO_FETCH").sum()
        ),
        "ready_environment_count": int(
            environment_map["source_archive_status"].str.startswith("READY_").sum()
        ),
        "unique_daily_request_count": len(requests),
        "cds_era5_land_request_count": len(cds_requests),
        "soilgrids_site_request_count": len(soil_requests),
        "soilgrids_coverage_request_count": int(
            soil_requests["coverage_request_count"].sum()
        ),
        "cmip6_ensemble_preregistered": False,
        "environment_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "source_archive_status_counts": {
            str(k): int(v)
            for k, v in environment_map["source_archive_status"]
            .value_counts()
            .sort_index()
            .items()
        },
        "management_unit_fields_resolved": int(management["status"].str.startswith("RESOLVED").sum()),
        "management_value_outliers_quarantined": len(management_outliers),
        "cds_credentials_present": credential_present(),
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "artifacts": artifacts,
        "python": platform.python_version(),
    }
    write_json(output / "environment_source_contract.json", contract)
    if contract["status"] != "PASS":
        raise SystemExit("Environment source contract failed")
    return contract


@dataclass(frozen=True)
class FetchResult:
    request_id: str
    status: str
    detail: str
    raw_path: str = ""
    raw_sha256: str = ""
    daily_path: str = ""
    daily_sha256: str = ""
    daily_rows: int = 0
    first_date: str = ""
    last_date: str = ""


class SoilGridsStructuralUnavailable(ValueError):
    pass


def fetch_bytes(url: str, timeout: int, retries: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "WheatConformer-Phase6A-source-recovery/1.0"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            last_error = exc
        if attempt + 1 < retries:
            time.sleep(2**attempt)
    raise RuntimeError(f"request failed after {retries} attempts: {last_error}")


def parse_open_meteo(raw: bytes, row: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = json.loads(raw.decode("utf-8"))
    daily = payload.get("daily")
    units = payload.get("daily_units")
    if not isinstance(daily, dict) or not isinstance(units, dict) or "time" not in daily:
        raise ValueError("response does not contain a daily axis and unit registry")
    missing = sorted(set(DAILY_VARIABLES) - set(daily))
    if missing:
        raise ValueError(f"response is missing variables {missing}")
    unit_mismatches = {
        key: units.get(key) for key, expected in EXPECTED_UNITS.items() if units.get(key) != expected
    }
    if unit_mismatches:
        raise ValueError(f"unexpected response units {unit_mismatches}")
    frame = pd.DataFrame({"date": pd.to_datetime(daily["time"], errors="coerce")})
    for variable in DAILY_VARIABLES:
        frame[variable] = pd.to_numeric(pd.Series(daily[variable]), errors="coerce")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("daily dates are invalid or duplicated")
    start = pd.Timestamp(str(row["request_start_date"]))
    end = pd.Timestamp(str(row["request_end_date"]))
    expected_days = int((end - start).days) + 1
    frame = frame[frame["date"].between(start, end, inclusive="both")].copy()
    if len(frame) / expected_days < 0.98:
        raise ValueError(f"date coverage below 0.98: {len(frame)}/{expected_days}")
    required_finite = [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "shortwave_radiation_sum",
    ]
    if not np.isfinite(frame[required_finite].to_numpy(float)).all():
        raise ValueError("required daily climate values contain nonfinite cells")
    frame.insert(0, "request_id", str(row["request_id"]))
    frame["latitude"] = float(row["latitude"])
    frame["longitude"] = float(row["longitude"])
    frame["source_provider"] = "open_meteo"
    frame["source_model"] = OPEN_METEO_MODEL
    metadata = {
        "response_latitude": payload.get("latitude"),
        "response_longitude": payload.get("longitude"),
        "response_elevation": payload.get("elevation"),
        "generationtime_ms": payload.get("generationtime_ms"),
        "utc_offset_seconds": payload.get("utc_offset_seconds"),
        "timezone": payload.get("timezone"),
        "daily_units": units,
    }
    return frame, metadata


def cache_result(cache: Path, row: pd.Series) -> FetchResult | None:
    request_id = str(row["request_id"])
    metadata_path = cache / "requests" / request_id[:2] / f"{request_id}.json"
    if not metadata_path.is_file():
        return None
    metadata = read_json(metadata_path)
    raw_path = resolve(cache, Path(metadata["raw_path"]))
    daily_path = resolve(cache, Path(metadata["daily_path"]))
    if not raw_path.is_file() or not daily_path.is_file():
        return None
    if sha256_file(raw_path) != metadata["raw_sha256"]:
        return None
    if sha256_file(daily_path) != metadata["daily_sha256"]:
        return None
    return FetchResult(
        request_id=request_id,
        status="CACHED",
        detail="",
        raw_path=str(metadata["raw_path"]),
        raw_sha256=str(metadata["raw_sha256"]),
        daily_path=str(metadata["daily_path"]),
        daily_sha256=str(metadata["daily_sha256"]),
        daily_rows=int(metadata["daily_rows"]),
        first_date=str(metadata["first_date"]),
        last_date=str(metadata["last_date"]),
    )


def fetch_one(cache: Path, row: pd.Series, timeout: int, retries: int) -> FetchResult:
    request_id = str(row["request_id"])
    try:
        url = open_meteo_url(row)
        raw = fetch_bytes(url, timeout, retries)
        raw_sha = hashlib.sha256(raw).hexdigest()
        raw_relative = Path("raw") / raw_sha[:2] / f"{raw_sha}.json"
        raw_path = cache / raw_relative
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists():
            temporary = raw_path.with_suffix(f".tmp.{os.getpid()}")
            temporary.write_bytes(raw)
            temporary.replace(raw_path)
        frame, response_metadata = parse_open_meteo(raw, row)
        daily_relative = Path("daily") / request_id[:2] / f"{request_id}.parquet"
        daily_path = cache / daily_relative
        daily_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_daily = daily_path.with_suffix(f".tmp.{os.getpid()}.parquet")
        frame.to_parquet(temporary_daily, index=False, compression="zstd")
        temporary_daily.replace(daily_path)
        daily_sha = sha256_file(daily_path)
        dates = pd.to_datetime(frame["date"])
        request_relative = Path("requests") / request_id[:2] / f"{request_id}.json"
        request_path = cache / request_relative
        request_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "request_id": request_id,
            "request_identity": request_identity(
                float(row["latitude"]),
                float(row["longitude"]),
                str(row["request_start_date"]),
                str(row["request_end_date"]),
            ),
            "request_url_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
            "raw_path": raw_relative.as_posix(),
            "raw_sha256": raw_sha,
            "daily_path": daily_relative.as_posix(),
            "daily_sha256": daily_sha,
            "daily_rows": len(frame),
            "first_date": dates.min().strftime("%Y-%m-%d"),
            "last_date": dates.max().strftime("%Y-%m-%d"),
            "response_metadata": response_metadata,
        }
        write_json(request_path, metadata)
        return FetchResult(
            request_id=request_id,
            status="FETCHED",
            detail="",
            raw_path=raw_relative.as_posix(),
            raw_sha256=raw_sha,
            daily_path=daily_relative.as_posix(),
            daily_sha256=daily_sha,
            daily_rows=len(frame),
            first_date=metadata["first_date"],
            last_date=metadata["last_date"],
        )
    except Exception as exc:
        return FetchResult(
            request_id=request_id,
            status="FAILED_RETRYABLE",
            detail=f"{type(exc).__name__}:{exc}",
        )


def run_fetch(
    root: Path,
    contract_dir: Path,
    cache: Path,
    limit: int,
    workers: int,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    contract_path = contract_dir / "environment_source_contract.json"
    inventory_path = contract_dir / "daily_request_inventory.tsv"
    contract = read_json(contract_path)
    if contract.get("status") != "PASS":
        raise SystemExit("Environment source contract is not PASS")
    expected = dict(contract.get("artifacts", {})).get(inventory_path.name)
    if not expected or sha256_file(inventory_path) != expected:
        raise SystemExit("Daily request inventory is not bound to the contract")
    cache.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(inventory_path, sep="\t", dtype=str)
    cached: dict[str, FetchResult] = {}
    for _, row in inventory.iterrows():
        result = cache_result(cache, row)
        if result is not None:
            cached[result.request_id] = result
    pending = inventory[~inventory["request_id"].isin(cached)].copy()
    if limit > 0:
        pending = pending.head(limit)
    fetched: dict[str, FetchResult] = {}
    rows = [row for _, row in pending.iterrows()]
    if workers == 1:
        for row in rows:
            result = fetch_one(cache, row, timeout, retries)
            fetched[result.request_id] = result
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_one, cache, row, timeout, retries) for row in rows]
            for future in as_completed(futures):
                result = future.result()
                fetched[result.request_id] = result
    index_rows = []
    selected = set(pending["request_id"])
    for _, row in inventory.iterrows():
        request_id = str(row["request_id"])
        result = fetched.get(request_id) or cached.get(request_id)
        if result is None:
            status = "PENDING_LIMIT" if request_id not in selected else "FAILED_RETRYABLE"
            result = FetchResult(request_id, status, "not_selected_in_this_bounded_run")
        record = row.to_dict()
        record.update(result.__dict__)
        index_rows.append(record)
    index = pd.DataFrame(index_rows)
    index_path = cache / "daily_request_fetch_index.tsv"
    write_tsv(index_path, index)
    complete_statuses = {"FETCHED", "CACHED"}
    completed = int(index["status"].isin(complete_statuses).sum())
    forbidden = [
        path.name
        for path in cache.iterdir()
        if any(token in path.name.lower() for token in FORBIDDEN_OUTPUT_TOKENS)
    ]
    provenance = {
        "status": "PASS",
        "run_status": "COMPLETE" if completed == len(inventory) else "PARTIAL",
        "protocol_version": PROTOCOL_VERSION,
        "selection_data": "frozen_stage1_v2_environment_request_identifiers_only",
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
        "contract_sha256": sha256_file(contract_path),
        "request_inventory_sha256": sha256_file(inventory_path),
        "request_count": len(inventory),
        "selected_pending_count": len(pending),
        "fetched_this_run": sum(result.status == "FETCHED" for result in fetched.values()),
        "cached_before_run": len(cached),
        "completed_request_count": completed,
        "archive_complete": completed == len(inventory),
        "status_counts": {
            str(key): int(value) for key, value in index["status"].value_counts().items()
        },
        "fetch_index_sha256": sha256_file(index_path),
        "checks": {
            "contract_pass": True,
            "inventory_checksum": True,
            "cached_artifacts_validate": True,
            "no_future_matrices_or_predictions": not forbidden,
        },
        "authoritative_phase6a_archive": False,
        "authoritative_block_reason": (
            "Open-Meteo ERA5 is staged as an independent diagnostic archive; "
            "cross-provider certification against a checksummed CDS ERA5-Land reference is still required"
        ),
    }
    write_json(cache / "daily_fetch_provenance.json", provenance)
    return provenance


def cds_api_payload(row: pd.Series | Any) -> dict[str, Any]:
    identity = cds_request_identity(row)
    return {
        "variable": identity["variable"],
        "location": identity["location"],
        "date": identity["date"],
        "data_format": identity["data_format"],
    }


def run_cds_fetch(
    contract_dir: Path, cache: Path, limit: int
) -> dict[str, Any]:
    contract_path = contract_dir / "environment_source_contract.json"
    inventory_path = contract_dir / "cds_era5_land_request_inventory.tsv"
    contract = read_json(contract_path)
    expected = dict(contract.get("artifacts", {})).get(inventory_path.name)
    if contract.get("status") != "PASS" or not expected:
        raise SystemExit("CDS request inventory is not bound to a passing contract")
    if sha256_file(inventory_path) != expected:
        raise SystemExit("CDS request inventory checksum does not match the contract")
    cache.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(inventory_path, sep="\t", dtype=str)
    credentials = credential_present()
    try:
        import cdsapi  # type: ignore

        cdsapi_available = True
    except ImportError:
        cdsapi = None
        cdsapi_available = False

    records = []
    pending_selected = 0
    fetched = 0
    cached = 0
    client = None
    for _, row in inventory.iterrows():
        request_id = str(row["request_id"])
        metadata_path = cache / "requests" / request_id[:2] / f"{request_id}.json"
        status = ""
        detail = ""
        raw_path = ""
        raw_sha256 = ""
        raw_bytes = 0
        try:
            if metadata_path.is_file():
                metadata = read_json(metadata_path)
                candidate = cache / str(metadata.get("raw_path", ""))
                if candidate.is_file() and sha256_file(candidate) == metadata.get("raw_sha256"):
                    status = "CACHED"
                    raw_path = str(metadata["raw_path"])
                    raw_sha256 = str(metadata["raw_sha256"])
                    raw_bytes = candidate.stat().st_size
                    cached += 1
        except OSError as exc:
            raise RuntimeError(
                "CDS cache storage became unavailable while validating "
                f"{metadata_path}. Reconnect the drive and rerun; completed "
                "content-addressed requests will be reused."
            ) from exc
        if not status and not credentials:
            status = "BLOCKED_MISSING_CDS_CREDENTIALS"
            detail = "No ~/.cdsapirc or CDSAPI_KEY/CDS_API_KEY was present; no token was read"
        elif not status and not cdsapi_available:
            status = "BLOCKED_MISSING_CDSAPI_CLIENT"
            detail = "Install cdsapi before an authenticated retrieval"
        elif not status and limit > 0 and pending_selected >= limit:
            status = "PENDING_LIMIT"
            detail = "not selected in this bounded run"
        elif not status:
            pending_selected += 1
            temporary = cache / f".{request_id}.download"
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError as exc:
                raise RuntimeError(
                    "CDS cache storage became unavailable while preparing request "
                    f"{request_id} under {cache}. Reconnect the drive and rerun; "
                    "completed content-addressed requests will be reused."
                ) from exc
            try:
                if client is None:
                    client = cdsapi.Client(
                        quiet=True,
                        retry_max=CDS_RETRY_MAX,
                        sleep_max=CDS_RETRY_SLEEP_SECONDS,
                        timeout=CDS_REQUEST_TIMEOUT_SECONDS,
                    )
                client.retrieve(CDS_DATASET, cds_api_payload(row), str(temporary))
            except Exception as exc:
                # requests.HTTPError inherits from OSError. Keep provider/job failures
                # retryable instead of misclassifying them as local storage loss.
                status = "FAILED_RETRYABLE"
                detail = f"{type(exc).__name__}:{exc}"
            if not status:
                try:
                    if not temporary.is_file() or temporary.stat().st_size == 0:
                        raise ValueError("CDS retrieval produced an empty target")
                    raw_sha256 = sha256_file(temporary)
                    relative = Path("raw") / raw_sha256[:2] / f"{raw_sha256}.bin"
                    target = cache / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        temporary.unlink()
                    else:
                        temporary.replace(target)
                    raw_path = relative.as_posix()
                    raw_bytes = target.stat().st_size
                    write_json_atomic(
                        metadata_path,
                        {
                            "request_id": request_id,
                            "dataset": CDS_DATASET,
                            "request_payload": cds_api_payload(row),
                            "raw_path": raw_path,
                            "raw_sha256": raw_sha256,
                            "raw_bytes": raw_bytes,
                        },
                    )
                    status = "FETCHED_RAW"
                    fetched += 1
                except OSError as exc:
                    raise RuntimeError(
                        "CDS cache storage became unavailable while writing request "
                        f"{request_id} under {cache}. Reconnect the drive and rerun; "
                        "completed content-addressed requests will be reused."
                    ) from exc
                except Exception as exc:
                    status = "FAILED_RETRYABLE"
                    detail = f"{type(exc).__name__}:{exc}"
        record = row.to_dict()
        record.update(
            {
                "status": status,
                "detail": detail,
                "raw_path": raw_path,
                "raw_sha256": raw_sha256,
                "raw_bytes": raw_bytes,
            }
        )
        records.append(record)
    index = pd.DataFrame(records)
    index_path = cache / "cds_era5_land_fetch_index.tsv"
    write_tsv(index_path, index)
    completed = int(index["status"].isin(["FETCHED_RAW", "CACHED"]).sum())
    blocked_statuses = sorted(
        set(index.loc[index["status"].str.startswith("BLOCKED"), "status"])
    )
    provenance = {
        "status": "PASS",
        "run_status": (
            "COMPLETE_RAW_ARCHIVE"
            if completed == len(index)
            else "BLOCKED" if blocked_statuses else "PARTIAL"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "selection_data": "frozen_stage1_v2_environment_request_identifiers_only",
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
        "credentials_present": credentials,
        "credentials_or_token_logged": False,
        "cdsapi_available": cdsapi_available,
        "request_concurrency": CDS_REQUEST_CONCURRENCY,
        "retry_max": CDS_RETRY_MAX,
        "retry_sleep_seconds": CDS_RETRY_SLEEP_SECONDS,
        "request_timeout_seconds": CDS_REQUEST_TIMEOUT_SECONDS,
        "request_count": len(index),
        "selected_pending_count": pending_selected,
        "fetched_this_run": fetched,
        "cached_before_run": cached,
        "completed_raw_request_count": completed,
        "raw_archive_complete": completed == len(index),
        "normalized_daily_archive_complete": False,
        "status_counts": {
            str(key): int(value) for key, value in index["status"].value_counts().items()
        },
        "blocking_statuses": blocked_statuses,
        "contract_sha256": sha256_file(contract_path),
        "request_inventory_sha256": sha256_file(inventory_path),
        "fetch_index_sha256": sha256_file(index_path),
        "authoritative_phase6a_archive": False,
        "authoritative_block_reason": (
            "The full checksummed CDS raw archive must be normalized from hourly variables "
            "to certified daily units and pass cross-provider/date/coverage checks"
        ),
    }
    write_json(cache / "cds_era5_land_fetch_provenance.json", provenance)
    return provenance


def soilgrids_wcs_url(
    property_name: str, depth: str, x: float, y: float
) -> str:
    if property_name not in SOILGRIDS_PROPERTIES or depth not in SOILGRIDS_DEPTHS:
        raise ValueError("Unsupported SoilGrids property or depth")
    params = [
        ("map", f"/map/{property_name}.map"),
        ("SERVICE", "WCS"),
        ("VERSION", "2.0.1"),
        ("REQUEST", "GetCoverage"),
        ("COVERAGEID", f"{property_name}_{depth}_Q0.5"),
        ("FORMAT", "GEOTIFF_INT16"),
        ("SUBSET", f"X({x - 125:.3f},{x + 125:.3f})"),
        ("SUBSET", f"Y({y - 125:.3f},{y + 125:.3f})"),
        ("SUBSETTINGCRS", SOILGRIDS_NATIVE_CRS),
        ("OUTPUTCRS", SOILGRIDS_NATIVE_CRS),
    ]
    return SOILGRIDS_ENDPOINT + "?" + urllib.parse.urlencode(params)


def canonical_soilgrids_value(property_name: str, mapped_value: float) -> float:
    if property_name not in SOILGRIDS_PROPERTIES or not np.isfinite(mapped_value):
        raise ValueError("Unsupported SoilGrids property or nonfinite value")
    value = mapped_value / SOILGRIDS_PROPERTIES[property_name]
    if property_name in {"wv0033", "wv1500"} and value <= 0:
        raise SoilGridsStructuralUnavailable(
            f"missing water-content cell; mapped_value={mapped_value}"
        )
    if property_name in {"wv0033", "wv1500"} and value > 100:
        raise ValueError(f"invalid water-content value={value}")
    if property_name == "cfvo" and not (0 <= value <= 100):
        raise ValueError(f"invalid coarse-fragment value={value}")
    if property_name == "bdod" and value <= 0:
        raise SoilGridsStructuralUnavailable(
            f"missing bulk-density cell; mapped_value={mapped_value}"
        )
    if property_name == "bdod" and value > 3:
        raise ValueError(f"invalid bulk-density value={value}")
    return value


def soilgrids_completion_counts(index: pd.DataFrame) -> tuple[int, int, int]:
    statuses = index["status"].astype(str)
    numeric = int(statuses.isin(SOIL_NUMERIC_COMPLETE_STATUSES).sum())
    unavailable = int(statuses.eq("STRUCTURALLY_UNAVAILABLE_SOIL_CELL").sum())
    resolved = int(statuses.isin(SOIL_TERMINAL_RESOLVED_STATUSES).sum())
    return numeric, unavailable, resolved


def fetch_soil_site(
    cache: Path, row: pd.Series, timeout: int, retries: int
) -> tuple[str, str, int, str]:
    try:
        from rasterio.io import MemoryFile
        from rasterio.warp import transform
    except ImportError as exc:
        return "BLOCKED_MISSING_RASTERIO", str(exc), 0, ""
    site_id = str(row["site_id"])
    metadata_path = cache / "requests" / site_id[:2] / f"{site_id}.json"
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("status") == "STRUCTURALLY_UNAVAILABLE_SOIL_CELL":
            return (
                "STRUCTURALLY_UNAVAILABLE_SOIL_CELL",
                str(metadata.get("detail", "")),
                0,
                "",
            )
        values_path = cache / str(metadata.get("values_path", ""))
        if values_path.is_file() and sha256_file(values_path) == metadata.get(
            "values_sha256"
        ):
            return "CACHED", "", int(metadata["value_rows"]), str(metadata["values_path"])
    longitude = float(row["longitude"])
    latitude = float(row["latitude"])
    xs, ys = transform("EPSG:4326", SOILGRIDS_PROJ, [longitude], [latitude])
    values = []
    raw_bindings = []
    try:
        for property_name, divisor in SOILGRIDS_PROPERTIES.items():
            for depth in SOILGRIDS_DEPTHS:
                url = soilgrids_wcs_url(property_name, depth, xs[0], ys[0])
                raw = fetch_bytes(url, timeout, retries)
                raw_sha = hashlib.sha256(raw).hexdigest()
                relative = Path("raw") / raw_sha[:2] / f"{raw_sha}.tif"
                raw_path = cache / relative
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                if not raw_path.exists():
                    temporary = raw_path.with_suffix(f".tmp.{os.getpid()}.tif")
                    temporary.write_bytes(raw)
                    temporary.replace(raw_path)
                with MemoryFile(raw) as memory:
                    with memory.open() as source:
                        data = source.read(1)
                        if data.size != 1:
                            raise ValueError(
                                f"SoilGrids point request returned shape={data.shape}"
                            )
                        mapped_value = float(data.reshape(-1)[0])
                        canonical_value = canonical_soilgrids_value(
                            property_name, mapped_value
                        )
                canonical_unit = (
                    "volume_percent"
                    if property_name in {"wv0033", "wv1500", "cfvo"}
                    else "kg_per_dm3"
                )
                values.append(
                    {
                        "site_id": site_id,
                        "latitude": latitude,
                        "longitude": longitude,
                        "property": property_name,
                        "depth": depth,
                        "statistic": "Q0.5",
                        "mapped_integer_value": mapped_value,
                        "conversion_divisor": divisor,
                        "canonical_value": canonical_value,
                        "canonical_unit": canonical_unit,
                        "raw_sha256": raw_sha,
                    }
                )
                raw_bindings.append(
                    {
                        "property": property_name,
                        "depth": depth,
                        "raw_path": relative.as_posix(),
                        "raw_sha256": raw_sha,
                        "request_url_sha256": hashlib.sha256(
                            url.encode("utf-8")
                        ).hexdigest(),
                    }
                )
        frame = pd.DataFrame(values)
        relative_values = Path("values") / site_id[:2] / f"{site_id}.parquet"
        values_path = cache / relative_values
        values_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_values = values_path.with_suffix(f".tmp.{os.getpid()}.parquet")
        frame.to_parquet(temporary_values, index=False, compression="zstd")
        temporary_values.replace(values_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            metadata_path,
            {
                "site_id": site_id,
                "site_identity": soil_site_identity(latitude, longitude),
                "values_path": relative_values.as_posix(),
                "values_sha256": sha256_file(values_path),
                "value_rows": len(frame),
                "raw_bindings": raw_bindings,
            },
        )
        return "FETCHED", "", len(frame), relative_values.as_posix()
    except SoilGridsStructuralUnavailable as exc:
        detail = str(exc)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            metadata_path,
            {
                "site_id": site_id,
                "site_identity": soil_site_identity(latitude, longitude),
                "status": "STRUCTURALLY_UNAVAILABLE_SOIL_CELL",
                "detail": detail,
                "raw_bindings": raw_bindings,
            },
        )
        return "STRUCTURALLY_UNAVAILABLE_SOIL_CELL", detail, 0, ""
    except Exception as exc:
        return "FAILED_RETRYABLE", f"{type(exc).__name__}:{exc}", 0, ""


def run_soilgrids_fetch(
    contract_dir: Path,
    cache: Path,
    limit: int,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    contract_path = contract_dir / "environment_source_contract.json"
    inventory_path = contract_dir / "soilgrids_request_inventory.tsv"
    contract = read_json(contract_path)
    expected = dict(contract.get("artifacts", {})).get(inventory_path.name)
    if contract.get("status") != "PASS" or not expected:
        raise SystemExit("SoilGrids inventory is not bound to a passing contract")
    if sha256_file(inventory_path) != expected:
        raise SystemExit("SoilGrids inventory checksum does not match the contract")
    cache.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(inventory_path, sep="\t", dtype=str)
    records = []
    selected = 0
    for _, row in inventory.iterrows():
        site_id = str(row["site_id"])
        metadata = cache / "requests" / site_id[:2] / f"{site_id}.json"
        choose = metadata.is_file() or limit == 0 or selected < limit
        if choose:
            if not metadata.is_file():
                selected += 1
            status, detail, value_rows, values_path = fetch_soil_site(
                cache, row, timeout, retries
            )
        else:
            status, detail, value_rows, values_path = (
                "PENDING_LIMIT",
                "not selected in this bounded run",
                0,
                "",
            )
        record = row.to_dict()
        record.update(
            {
                "status": status,
                "detail": detail,
                "value_rows": value_rows,
                "values_path": values_path,
            }
        )
        records.append(record)
    index = pd.DataFrame(records)
    index_path = cache / "soilgrids_fetch_index.tsv"
    write_tsv(index_path, index)
    completed, structurally_unavailable, resolved = soilgrids_completion_counts(index)
    provenance = {
        "status": "PASS",
        "run_status": "COMPLETE" if resolved == len(index) else "PARTIAL",
        "protocol_version": PROTOCOL_VERSION,
        "selection_data": "stage1_v2_location_identifiers_only",
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
        "site_count": len(index),
        "selected_pending_count": selected,
        "completed_site_count": completed,
        "structurally_unavailable_site_count": structurally_unavailable,
        "resolved_site_count": resolved,
        "completed_value_rows": int(
            pd.to_numeric(index["value_rows"], errors="coerce").sum()
        ),
        "archive_complete": resolved == len(index),
        "status_counts": {
            str(key): int(value) for key, value in index["status"].value_counts().items()
        },
        "contract_sha256": sha256_file(contract_path),
        "request_inventory_sha256": sha256_file(inventory_path),
        "fetch_index_sha256": sha256_file(index_path),
        "authoritative_soil_archive_complete": resolved == len(index),
    }
    write_json(cache / "soilgrids_fetch_provenance.json", provenance)
    return provenance


def freeze_staging_status(
    root: Path,
    contract_dir: Path,
    cache: Path,
    cds_cache: Path,
    soil_cache: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise SystemExit(f"Fail-if-exists staging status directory already exists: {output}")
    output.mkdir(parents=True)
    contract_path = contract_dir / "environment_source_contract.json"
    fetch_path = cache / "daily_fetch_provenance.json"
    index_path = cache / "daily_request_fetch_index.tsv"
    cds_fetch_path = cds_cache / "cds_era5_land_fetch_provenance.json"
    cds_index_path = cds_cache / "cds_era5_land_fetch_index.tsv"
    soil_fetch_path = soil_cache / "soilgrids_fetch_provenance.json"
    soil_index_path = soil_cache / "soilgrids_fetch_index.tsv"
    required = [
        contract_path,
        contract_dir / "environment_daily_request_map.tsv",
        contract_dir / "management_unit_resolution.tsv",
        contract_dir / "management_value_outliers.tsv",
        contract_dir / "provider_readiness.tsv",
        fetch_path,
        index_path,
        cds_fetch_path,
        cds_index_path,
        soil_fetch_path,
        soil_index_path,
        contract_dir / "cmip6_ensemble_preregistration_requirement.tsv",
        Path(__file__).resolve(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing staging status inputs: {missing}")
    contract = read_json(contract_path)
    fetch = read_json(fetch_path)
    cds_fetch = read_json(cds_fetch_path)
    soil_fetch = read_json(soil_fetch_path)
    environment_map = pd.read_csv(
        contract_dir / "environment_daily_request_map.tsv", sep="\t", dtype=str
    )
    management = pd.read_csv(
        contract_dir / "management_unit_resolution.tsv", sep="\t", dtype=str
    )
    management_outliers = pd.read_csv(
        contract_dir / "management_value_outliers.tsv", sep="\t", dtype=str
    )
    providers = pd.read_csv(contract_dir / "provider_readiness.tsv", sep="\t", dtype=str)
    index = pd.read_csv(index_path, sep="\t", dtype=str)
    cds_index = pd.read_csv(cds_index_path, sep="\t", dtype=str)
    soil_index = pd.read_csv(soil_index_path, sep="\t", dtype=str)
    completed = index[index["status"].isin(["FETCHED", "CACHED"])]
    source_bindings = pd.DataFrame(
        [
            {
                "path": str(path.relative_to(root)).replace("\\", "/")
                if path.is_relative_to(root)
                else str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "status": "PASS",
            }
            for path in required
        ]
    )
    write_tsv(output / "source_binding_manifest.tsv", source_bindings)
    summary = pd.DataFrame(
        [
            {"metric": "stage1_v2_environments", "value": len(environment_map)},
            {
                "metric": "source_archive_ready_environments",
                "value": int(
                    environment_map["source_archive_status"]
                    .str.startswith("READY_")
                    .sum()
                ),
            },
            {
                "metric": "source_archive_blocked_environments",
                "value": int(
                    (~environment_map["source_archive_status"].str.startswith("READY_"))
                    .sum()
                ),
            },
            {
                "metric": "feature_window_metadata_gap_environments",
                "value": int(environment_map["status"].ne("READY_TO_FETCH").sum()),
            },
            {"metric": "unique_daily_requests", "value": len(index)},
            {"metric": "diagnostic_requests_completed", "value": len(completed)},
            {
                "metric": "diagnostic_daily_rows",
                "value": int(pd.to_numeric(completed["daily_rows"], errors="coerce").sum()),
            },
            {
                "metric": "cds_raw_requests_completed",
                "value": int(
                    cds_index["status"].isin(["FETCHED_RAW", "CACHED"]).sum()
                ),
            },
            {
                "metric": "soilgrids_sites_completed",
                "value": int(
                    soil_index["status"].isin(["FETCHED", "CACHED"]).sum()
                ),
            },
            {
                "metric": "management_fields_resolved",
                "value": int(management["status"].str.startswith("RESOLVED").sum()),
            },
            {
                "metric": "management_value_outliers_quarantined",
                "value": len(management_outliers),
            },
            {"metric": "future_covariate_matrices_generated", "value": 0},
            {"metric": "future_predictions_generated", "value": 0},
        ]
    )
    write_tsv(output / "source_recovery_summary.tsv", summary)
    blocker_rows = []
    for row in providers.itertuples(index=False):
        if str(row.access_status).startswith("BLOCKED") or (
            str(row.authoritative_for_phase6a).lower() == "true"
            and str(row.access_status).startswith("READY_PUBLIC_METADATA")
        ):
            blocker_rows.append(
                {
                    "scope": row.source,
                    "status": row.access_status,
                    "detail": row.detail,
                }
            )
    for status, count in environment_map.loc[
        environment_map["status"].ne("READY_TO_FETCH"), "status"
    ].value_counts().items():
        blocker_rows.append(
            {
                "scope": "historical_environment_metadata",
                "status": status,
                "detail": f"environments={count}",
            }
        )
    for status, count in environment_map.loc[
        ~environment_map["source_archive_status"].str.startswith("READY_"),
        "source_archive_status",
    ].value_counts().items():
        blocker_rows.append(
            {
                "scope": "historical_source_archive_request",
                "status": status,
                "detail": f"environments={count}",
            }
        )
    blocker_rows.append(
        {
            "scope": "authoritative_historical_archive",
            "status": (
                "BLOCKED_PENDING_CDS_CROSS_PROVIDER_CERTIFICATION"
                if not cds_fetch.get("raw_archive_complete")
                else "BLOCKED_PENDING_CDS_DAILY_NORMALIZATION_AND_CERTIFICATION"
            ),
            "detail": cds_fetch["authoritative_block_reason"],
        }
    )
    if not soil_fetch.get("archive_complete"):
        blocker_rows.append(
            {
                "scope": "authoritative_soil_archive",
                "status": "BLOCKED_INCOMPLETE_SOILGRIDS_ARCHIVE",
                "detail": (
                    f"completed_sites={soil_fetch.get('completed_site_count', 0)}/"
                    f"{soil_fetch.get('site_count', 0)}"
                ),
            }
        )
    blockers = pd.DataFrame(blocker_rows)
    write_tsv(output / "remaining_blockers.tsv", blockers)
    checks = {
        "source_contract_pass": contract.get("status") == "PASS",
        "diagnostic_fetch_pass": fetch.get("status") == "PASS",
        "cds_fetcher_preflight_pass": cds_fetch.get("status") == "PASS",
        "soilgrids_fetcher_smoke_pass": (
            soil_fetch.get("status") == "PASS"
            and int(soil_fetch.get("completed_site_count", 0)) > 0
        ),
        "diagnostic_smoke_nonempty": len(completed) > 0,
        "diagnostic_request_files_validate": bool(
            fetch.get("checks", {}).get("cached_artifacts_validate")
        ),
        "all_management_units_resolved": management["status"].str.startswith("RESOLVED").all(),
        "management_outliers_explicitly_quarantined": len(management_outliers)
        == int(
            pd.to_numeric(management["quarantined_outlier_rows"], errors="coerce").sum()
        ),
        "no_phenotype_or_metric_access": all(
            source.get(key) is False
            for source in (contract, fetch, cds_fetch, soil_fetch)
            for key in (
                "phenotype_values_read",
                "inner_validation_metrics_read",
                "outer_test_outcomes_read",
                "outer_test_metrics_read",
                "final_holdout_outcomes_read",
            )
        ),
        "no_future_matrix_or_prediction": (
            contract.get("future_covariate_matrices_generated") == 0
            and contract.get("future_predictions_generated") == 0
            and fetch.get("future_covariate_matrices_generated") == 0
            and fetch.get("future_predictions_generated") == 0
            and cds_fetch.get("future_covariate_matrices_generated") == 0
            and cds_fetch.get("future_predictions_generated") == 0
            and soil_fetch.get("future_covariate_matrices_generated") == 0
            and soil_fetch.get("future_predictions_generated") == 0
        ),
        "authoritative_archive_not_misrepresented": (
            fetch.get("authoritative_phase6a_archive") is False
            and cds_fetch.get("authoritative_phase6a_archive") is False
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    decision = {
        "status": (
            "PASS_PHASE6A_SOURCE_FETCHER_AND_DIAGNOSTIC_SMOKE_WITH_AUTHORITATIVE_SOURCES_BLOCKED"
            if all(checks.values())
            else "FAIL_PHASE6A_SOURCE_FETCHER_STAGING"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "selection_data": "stage1_v2_environment_identifiers_and_source_metadata_only",
        "phase6a_remediation_complete": False,
        "phase6b_allowed": False,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_covariate_matrices_generated": 0,
        "future_predictions_generated": 0,
        "environment_count": len(environment_map),
        "ready_environment_count": int(
            environment_map["source_archive_status"].str.startswith("READY_").sum()
        ),
        "blocked_environment_count": int(
            (~environment_map["source_archive_status"].str.startswith("READY_")).sum()
        ),
        "feature_window_metadata_gap_environment_count": int(
            environment_map["status"].ne("READY_TO_FETCH").sum()
        ),
        "unique_daily_request_count": len(index),
        "diagnostic_request_count_completed": len(completed),
        "diagnostic_archive_complete": bool(fetch.get("archive_complete")),
        "cds_raw_archive_complete": bool(cds_fetch.get("raw_archive_complete")),
        "soilgrids_archive_complete": bool(soil_fetch.get("archive_complete")),
        "authoritative_historical_archive_complete": False,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "remaining_blocker_count": len(blockers),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(output / "PHASE6A_ENVIRONMENT_SOURCE_STAGING_DECISION.json", decision)
    output_files = sorted(path for path in output.iterdir() if path.is_file())
    output_manifest = pd.DataFrame(
        [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_files
        ]
    )
    write_tsv(output / "output_manifest.tsv", output_manifest)
    if not all(checks.values()):
        raise SystemExit("Phase-6A environment source staging status failed")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and fetch the phenotype-blind Stage-1 v2 Phase-6A environment source archive"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-contract")
    build.add_argument("--root", type=Path, default=Path("."))
    build.add_argument("--out-dir", type=Path, default=DEFAULT_CONTRACT)
    fetch = subparsers.add_parser("fetch-openmeteo")
    fetch.add_argument("--root", type=Path, default=Path("."))
    fetch.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT)
    fetch.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    fetch.add_argument("--limit", type=int, default=10)
    fetch.add_argument("--workers", type=int, default=2)
    fetch.add_argument("--timeout", type=int, default=120)
    fetch.add_argument("--retries", type=int, default=5)
    cds = subparsers.add_parser("fetch-cds-era5-land")
    cds.add_argument("--root", type=Path, default=Path("."))
    cds.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT)
    cds.add_argument("--cache-dir", type=Path, default=DEFAULT_CDS_CACHE)
    cds.add_argument("--limit", type=int, default=1)
    soil = subparsers.add_parser("fetch-soilgrids")
    soil.add_argument("--root", type=Path, default=Path("."))
    soil.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT)
    soil.add_argument("--cache-dir", type=Path, default=DEFAULT_SOIL_CACHE)
    soil.add_argument("--limit", type=int, default=1)
    soil.add_argument("--timeout", type=int, default=120)
    soil.add_argument("--retries", type=int, default=5)
    freeze = subparsers.add_parser("freeze-status")
    freeze.add_argument("--root", type=Path, default=Path("."))
    freeze.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT)
    freeze.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    freeze.add_argument("--cds-cache-dir", type=Path, default=DEFAULT_CDS_CACHE)
    freeze.add_argument("--soil-cache-dir", type=Path, default=DEFAULT_SOIL_CACHE)
    freeze.add_argument("--out-dir", type=Path, default=DEFAULT_STATUS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.command == "build-contract":
        contract = build_contract(root, resolve(root, args.out_dir))
        print(json.dumps(contract, indent=2, sort_keys=True), flush=True)
        return
    if args.command == "freeze-status":
        decision = freeze_staging_status(
            root,
            resolve(root, args.contract_dir),
            resolve(root, args.cache_dir),
            resolve(root, args.cds_cache_dir),
            resolve(root, args.soil_cache_dir),
            resolve(root, args.out_dir),
        )
        print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
        return
    if args.limit < 0:
        raise SystemExit("limit must be nonnegative")
    if args.command == "fetch-cds-era5-land":
        provenance = run_cds_fetch(
            resolve(root, args.contract_dir),
            resolve(root, args.cache_dir),
            args.limit,
        )
        print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)
        return
    if args.command == "fetch-soilgrids":
        if min(args.timeout, args.retries) < 1:
            raise SystemExit("timeout and retries must be positive")
        provenance = run_soilgrids_fetch(
            resolve(root, args.contract_dir),
            resolve(root, args.cache_dir),
            args.limit,
            args.timeout,
            args.retries,
        )
        print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)
        return
    if min(args.workers, args.timeout, args.retries) < 1 or args.limit < 0:
        raise SystemExit("workers, timeout, and retries must be positive; limit must be nonnegative")
    provenance = run_fetch(
        root,
        resolve(root, args.contract_dir),
        resolve(root, args.cache_dir),
        args.limit,
        args.workers,
        args.timeout,
        args.retries,
    )
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
