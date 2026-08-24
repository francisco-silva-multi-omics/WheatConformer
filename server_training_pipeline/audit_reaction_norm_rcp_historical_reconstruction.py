from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from build_dth_env_features_v2 import env_id_from_parts, parse_numeric
from .audit_reaction_norm_rcp_feature_readiness import normalize_source_token, truthy
from .final_evaluation_contract import file_sha256
from .report_reaction_norm_routed_diagnostics import read_ids, resolve_provenance_source


ANNUAL_PRECIPITATION_FIELDS = [
    "TOTAL_PRECIPIT_IN_12_MONTHS",
    "ESTIMATE_TOTAL_PRECIPIT_IN_12_MONTHS",
]
HARVEST_MONTH_FIELDS = [
    "PPN_MONTH_OF_HARVESTED",
    *[f"PPN_{index}{suffix}_MO_BEFORE_HARVESTED" for index, suffix in [
        (1, "ST"),
        (2, "ND"),
        (3, "RD"),
        *[(value, "TH") for value in range(4, 12)],
    ]],
]
CROP_PRECIPITATION_FIELDS = {
    "PRECIPITATION_FROM_SOWING_TO_MATURITY",
    "PRECIPITATION_ON_CROP",
}
MOISTURE_FIELDS = {"MOISTURE_AVAILB_BEFORE_SOWING_EXCL_PRE_IRRIGATION"}
OUTPUT_FILENAMES = {
    "contract": "RCP_historical_source_replacement_contract.tsv",
    "backcast_values": "RCP_fixed_window_replacement_backcast.tsv",
    "backcast": "RCP_historical_backcast_metrics.tsv",
    "harvest": "RCP_harvest_anchor_audit.tsv",
    "queue": "RCP_daily_reanalysis_work_queue.tsv",
    "queue_unique": "RCP_daily_reanalysis_unique_requests.tsv",
    "annual": "RCP_annual_precipitation_audit.tsv",
    "annual_overlap": "RCP_annual_precipitation_overlap.tsv",
    "range_blocks": "RCP_block_range_certification_contract.tsv",
    "range_features": "RCP_historical_range_rule_amendment.tsv",
    "summary": "RCP_historical_reconstruction_summary.tsv",
    "provenance": "RCP_historical_reconstruction_provenance.json",
    "certification": "RCP_historical_reconstruction_certification.json",
}


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def resolve(root: Path, value: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()


def stable_id(*parts: object) -> str:
    value = "|".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def finite_metrics(observed: pd.Series, candidate: pd.Series) -> dict[str, float | int]:
    left = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(candidate, errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(left) & np.isfinite(right)
    left = left[keep]
    right = right[keep]
    if len(left) == 0:
        return {
            "paired_environments": 0,
            "rmse": math.nan,
            "normalized_rmse": math.nan,
            "mae": math.nan,
            "mean_bias": math.nan,
            "pearson": math.nan,
            "spearman": math.nan,
            "observed_median": math.nan,
            "candidate_median": math.nan,
            "median_ratio_candidate_to_observed": math.nan,
        }
    residual = right - left
    observed_std = float(np.std(left, ddof=0))
    pearson = (
        float(np.corrcoef(left, right)[0, 1])
        if len(left) >= 2 and np.std(left) > 0 and np.std(right) > 0
        else math.nan
    )
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    spearman = (
        float(np.corrcoef(left_rank, right_rank)[0, 1])
        if len(left) >= 2 and np.std(left_rank) > 0 and np.std(right_rank) > 0
        else math.nan
    )
    observed_median = float(np.median(left))
    candidate_median = float(np.median(right))
    return {
        "paired_environments": int(len(left)),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "normalized_rmse": (
            float(np.sqrt(np.mean(np.square(residual))) / observed_std)
            if observed_std > 0
            else math.nan
        ),
        "mae": float(np.mean(np.abs(residual))),
        "mean_bias": float(np.mean(residual)),
        "pearson": pearson,
        "spearman": spearman,
        "observed_median": observed_median,
        "candidate_median": candidate_median,
        "median_ratio_candidate_to_observed": (
            candidate_median / observed_median if observed_median != 0 else math.nan
        ),
    }


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def references(
    outer_dir: Path, outer_protocol: dict[str, object], data_root: Path
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario, fold_count in dict(outer_protocol["scenarios"]).items():
        for outer_fold in range(int(fold_count)):
            directory = (
                outer_dir
                / "folds"
                / str(scenario)
                / f"outer_{outer_fold}"
                / "E_REACTION_NORM_V1"
            )
            certification = read_json(directory / "E_REACTION_NORM_V1_certification.json")
            if certification.get("status") != "PASS":
                raise SystemExit(
                    f"Uncertified E_REACTION_NORM_V1 reference: {scenario} outer={outer_fold}"
                )
            provenance = read_json(directory / "E_REACTION_NORM_V1_provenance.json")
            sources = provenance.get("sources", {})
            if not isinstance(sources, dict):
                raise SystemExit(f"Missing source map in {directory}")
            required = {}
            for name in ("fit_environment_ids", "envdata", "window_features"):
                source = sources.get(name)
                if not isinstance(source, dict):
                    raise SystemExit(f"Missing {name} provenance in {directory}")
                required[name] = resolve_provenance_source(source, data_root=data_root)
            generic_qc = None
            if isinstance(sources.get("generic_environment_provenance"), dict):
                generic_qc = resolve_provenance_source(
                    sources["generic_environment_provenance"], data_root=data_root
                )
            rows.append(
                {
                    "scenario": str(scenario),
                    "outer_fold": outer_fold,
                    "directory": directory,
                    "fit_ids_path": required["fit_environment_ids"],
                    "fit_ids": read_ids(required["fit_environment_ids"]),
                    "envdata_path": required["envdata"],
                    "window_path": required["window_features"],
                    "generic_qc_path": generic_qc,
                }
            )
    return rows


def unique_reference_path(rows: list[dict[str, object]], key: str) -> Path:
    paths = [Path(row[key]) for row in rows]
    identities = {(file_sha256(path), path.stat().st_size) for path in paths}
    if len(identities) != 1:
        raise SystemExit(f"Fold references disagree on {key}: {sorted(str(path) for path in paths)}")
    return paths[0]


def resolve_recorded_directory(value: object, *, root: Path) -> Path | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    recorded = Path(text).expanduser()
    candidates = [recorded]
    parts = list(recorded.parts)
    for anchor in ("model_kernels", "audit", "environment"):
        if anchor in parts:
            candidates.append(root.joinpath(*parts[parts.index(anchor) :]))
            break
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    return None


def infer_weather_dir(
    rows: list[dict[str, object]], environment_dir: Path, explicit: Path | None, root: Path
) -> Path:
    if explicit is not None:
        path = resolve(root, explicit)
        if not path.is_dir():
            raise SystemExit(f"Weather directory does not exist: {path}")
        return path
    for row in rows:
        qc_path = row.get("generic_qc_path")
        if not isinstance(qc_path, Path):
            continue
        qc = read_json(qc_path)
        path = resolve_recorded_directory(qc.get("weather_feature_input_dir"), root=root)
        if path is not None:
            return path
    return environment_dir


def source_inventory(lineage: pd.DataFrame) -> pd.DataFrame:
    source = lineage[
        lineage["projectability_class"].eq("historical_only_unprojectable")
        & ~truthy(lineage["is_missingness_indicator"])
    ].copy()
    source["source_token"] = source["source_feature"].map(normalize_source_token)
    rows = []
    for token, frame in source.groupby("source_token", sort=True):
        rows.append(
            {
                "source_token": token,
                "frozen_feature_count": frame["feature"].nunique(),
                "frozen_features": ";".join(sorted(frame["feature"].astype(str).unique())),
                "feature_blocks": ";".join(sorted(frame["feature_block"].astype(str).unique())),
                "duplicate_group_ids": ";".join(
                    sorted(value for value in frame["duplicate_group_id"].fillna("").astype(str).unique() if value)
                ),
            }
        )
    return pd.DataFrame(rows)


def read_observed_environment_values(
    envdata: pd.DataFrame, env_ids: pd.Index, source_tokens: Iterable[str]
) -> pd.DataFrame:
    requested = set(str(value) for value in source_tokens).union(
        ANNUAL_PRECIPITATION_FIELDS
    )
    frame = envdata[envdata["Trait_name"].astype(str).isin(requested)].copy()
    frame["env_id"] = env_id_from_parts(frame)
    frame["value_num"] = frame["Value"].map(parse_numeric)
    observed = frame.pivot_table(
        index="env_id", columns="Trait_name", values="value_num", aggfunc="mean"
    )
    return observed.reindex(index=env_ids, columns=sorted(requested))


def read_window_precipitation(path: Path, env_ids: pd.Index) -> pd.Series:
    usecols = [
        column
        for column in (
            "env_id",
            "window_label",
            "fetch_status",
            "precipitation_total_mm",
        )
        if column in pd.read_csv(path, sep="\t", nrows=0).columns
    ]
    required = {"env_id", "window_label", "precipitation_total_mm"}
    if not required.issubset(usecols):
        raise SystemExit(f"Fixed-window table lacks {sorted(required - set(usecols))}: {path}")
    frame = pd.read_csv(path, sep="\t", usecols=usecols, dtype=str, low_memory=False)
    if "fetch_status" in frame:
        frame = frame[frame["fetch_status"].astype(str).str.lower().eq("ok")]
    frame = frame[frame["window_label"].astype(str).eq("d0_180")].copy()
    frame["precipitation_total_mm"] = pd.to_numeric(
        frame["precipitation_total_mm"], errors="coerce"
    )
    values = frame.groupby("env_id", sort=False)["precipitation_total_mm"].mean()
    return values.reindex(env_ids)


def read_weather_aggregate(weather_dir: Path, env_ids: pd.Index) -> pd.DataFrame:
    combined = pd.DataFrame(index=env_ids)
    for filename, source_name in (
        ("trial_weather_features_openmeteo.tsv", "openmeteo_era5"),
        ("trial_weather_features_nasa_power.tsv", "nasa_power"),
    ):
        path = weather_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            continue
        header = pd.read_csv(path, sep="\t", nrows=0).columns
        columns = [
            value
            for value in (
                "env_id",
                "fetch_status",
                "n_days_weather",
                "precipitation_total_mm",
                "et0_total_mm",
                "soil_moisture_0_7_mean_m3m3",
                "soil_moisture_7_28_mean_m3m3",
            )
            if value in header
        ]
        if "env_id" not in columns:
            continue
        frame = pd.read_csv(path, sep="\t", usecols=columns, dtype=str, low_memory=False)
        if "fetch_status" in frame:
            frame = frame[frame["fetch_status"].astype(str).str.lower().eq("ok")]
        numeric = [column for column in columns if column not in {"env_id", "fetch_status"}]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.drop_duplicates("env_id", keep="first").set_index("env_id")
        for column in numeric:
            if column not in combined:
                combined[column] = np.nan
            missing = combined[column].isna()
            combined.loc[missing, column] = frame[column].reindex(env_ids).loc[missing]
        if "weather_feature_source" not in combined:
            combined["weather_feature_source"] = ""
        source_available = frame[numeric].notna().any(axis=1) if numeric else pd.Series(False, index=frame.index)
        source_ids = source_available[source_available].index.intersection(env_ids)
        blank = combined.loc[source_ids, "weather_feature_source"].eq("")
        combined.loc[source_ids[blank], "weather_feature_source"] = source_name
    return combined


def read_fetch_manifest(path: Path, env_ids: pd.Index) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    if "env_id" not in frame:
        raise SystemExit(f"Weather manifest has no env_id: {path}")
    frame = frame.drop_duplicates("env_id", keep="first").set_index("env_id").reindex(env_ids)
    for column in (
        "sowing_date",
        "harvest_start_date",
        "harvest_finish_date",
        "weather_start_date",
        "weather_end_date",
    ):
        frame[column] = pd.to_datetime(frame.get(column), errors="coerce")
    if "Country" not in frame:
        frame["Country"] = pd.Series(frame.index, index=frame.index).str.split("|").str[3]
    for column in ("latitude", "longitude"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    for column in (
        "sowing_date_source",
        "harvest_start_date_source",
        "harvest_finish_date_source",
        "coordinate_source",
    ):
        if column not in frame:
            if column == "coordinate_source":
                available = frame[["latitude", "longitude"]].notna().all(axis=1)
            else:
                base = column.removesuffix("_source")
                available = frame[base].notna()
            frame[column] = np.where(available, "legacy_manifest_value", "missing")
    return frame


def fit_harvest_anchors(
    metadata: pd.DataFrame,
    fit_ids: pd.Index,
    protocol: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    policy = dict(protocol["harvest_anchor"])
    local = metadata.reindex(fit_ids).copy()
    explicit = local["harvest_finish_date"].combine_first(local["harvest_start_date"])
    explicit_source = pd.Series("", index=local.index, dtype="object")
    finish = local["harvest_finish_date"].notna()
    start = ~finish & local["harvest_start_date"].notna()
    explicit_source.loc[finish] = "recorded_harvest_finish_date"
    explicit_source.loc[start] = "recorded_harvest_start_date"
    season_days = (explicit - local["sowing_date"]).dt.days.astype(float)
    valid = season_days.between(
        int(policy["minimum_season_length_days"]),
        int(policy["maximum_season_length_days"]),
        inclusive="both",
    )
    donor = local.loc[valid].copy()
    donor["season_days"] = season_days.loc[valid]
    global_days = (
        float(donor["season_days"].median())
        if not donor.empty
        else float(policy["default_season_length_days_when_no_training_donors"])
    )
    country = donor.groupby("Country")["season_days"].agg(["count", "median"])
    supported = country[country["count"].ge(int(policy["minimum_country_donors"]))]
    anchor = explicit.copy()
    anchor_source = explicit_source.copy()
    missing = anchor.isna() & local["sowing_date"].notna()
    predicted_days = local["Country"].map(supported["median"]).fillna(global_days)
    anchor.loc[missing] = local.loc[missing, "sowing_date"] + pd.to_timedelta(
        predicted_days.loc[missing], unit="D"
    )
    country_supported = missing & local["Country"].isin(supported.index)
    anchor_source.loc[country_supported] = "outer_training_country_median_season_length"
    anchor_source.loc[missing & ~country_supported] = "outer_training_global_median_season_length"
    anchor_source.loc[anchor.isna()] = "missing_sowing_and_harvest_anchor"
    result = local.copy()
    result["harvest_anchor"] = anchor
    result["harvest_anchor_source"] = anchor_source
    result["modeled_season_length_days"] = predicted_days
    summary = {
        "fit_environment_count": len(local),
        "recorded_harvest_finish_count": int(finish.sum()),
        "recorded_harvest_start_only_count": int(start.sum()),
        "valid_season_length_donor_count": len(donor),
        "country_model_count": len(supported),
        "global_median_season_length_days": global_days,
        "modeled_country_anchor_count": int(country_supported.sum()),
        "modeled_global_anchor_count": int((missing & ~country_supported).sum()),
        "missing_anchor_count": int(anchor.isna().sum()),
        "harvest_anchor_coverage_fraction": float(anchor.notna().mean()),
    }
    return result, summary


def monthly_window(anchor: pd.Timestamp, months_before: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end_period = anchor.to_period("M")
    start = (end_period - months_before).start_time.normalize()
    end = end_period.end_time.normalize()
    return start, end


def queue_record(
    *,
    scenario: str,
    outer_fold: int,
    env_id: str,
    request_kind: str,
    start: pd.Timestamp | pd.NaT,
    end: pd.Timestamp | pd.NaT,
    metadata: pd.Series,
    required_variables: str,
    date_source: str,
) -> dict[str, object]:
    coordinates_ok = pd.notna(metadata["latitude"]) and pd.notna(metadata["longitude"])
    dates_ok = pd.notna(start) and pd.notna(end)
    status = "READY_TO_FETCH" if coordinates_ok and dates_ok else (
        "BLOCKED_MISSING_COORDINATES" if dates_ok else "BLOCKED_MISSING_DATE_ANCHOR"
    )
    start_text = start.strftime("%Y-%m-%d") if pd.notna(start) else ""
    end_text = end.strftime("%Y-%m-%d") if pd.notna(end) else ""
    request_parts: list[object] = [
        metadata["latitude"],
        metadata["longitude"],
        start_text,
        end_text,
        required_variables,
    ]
    if status != "READY_TO_FETCH":
        request_parts.append(env_id)
    return {
        "scenario": scenario,
        "outer_fold": outer_fold,
        "env_id": env_id,
        "request_kind": request_kind,
        "request_start_date": start_text,
        "request_end_date": end_text,
        "latitude": metadata["latitude"],
        "longitude": metadata["longitude"],
        "coordinate_source": metadata.get("coordinate_source", ""),
        "date_anchor_source": date_source,
        "required_daily_variables": required_variables,
        "request_status": status,
        "request_id": stable_id(*request_parts),
    }


def build_daily_queue(
    refs: list[dict[str, object]],
    metadata: pd.DataFrame,
    protocol: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    queue: list[dict[str, object]] = []
    harvest_rows: list[dict[str, object]] = []
    request_policy = dict(protocol["daily_backcast_requests"])
    for ref in refs:
        fit_ids = pd.Index(ref["fit_ids"])
        fitted, summary = fit_harvest_anchors(metadata, fit_ids, protocol)
        summary.update({"scenario": ref["scenario"], "outer_fold": ref["outer_fold"]})
        harvest_rows.append(summary)
        for env_id, row in fitted.iterrows():
            sowing = row["sowing_date"]
            annual_start = sowing - pd.Timedelta(days=365) if pd.notna(sowing) else pd.NaT
            annual_end = sowing - pd.Timedelta(days=1) if pd.notna(sowing) else pd.NaT
            moisture_start = (
                sowing - pd.Timedelta(days=int(request_policy["antecedent_days_before_sowing"]))
                if pd.notna(sowing)
                else pd.NaT
            )
            queue.append(
                queue_record(
                    scenario=str(ref["scenario"]),
                    outer_fold=int(ref["outer_fold"]),
                    env_id=str(env_id),
                    request_kind="annual_precipitation_trailing_365_before_sowing",
                    start=annual_start,
                    end=annual_end,
                    metadata=row,
                    required_variables="pr",
                    date_source=str(row.get("sowing_date_source", "")),
                )
            )
            queue.append(
                queue_record(
                    scenario=str(ref["scenario"]),
                    outer_fold=int(ref["outer_fold"]),
                    env_id=str(env_id),
                    request_kind="pre_sowing_antecedent_water_balance",
                    start=moisture_start,
                    end=annual_end,
                    metadata=row,
                    required_variables="pr;et0;soil_moisture_0_7;soil_moisture_7_28",
                    date_source=str(row.get("sowing_date_source", "")),
                )
            )
            harvest_anchor = row["harvest_anchor"]
            if pd.notna(harvest_anchor):
                harvest_start, harvest_end = monthly_window(
                    harvest_anchor, int(request_policy["harvest_months_before"])
                )
            else:
                harvest_start, harvest_end = pd.NaT, pd.NaT
            queue.append(
                queue_record(
                    scenario=str(ref["scenario"]),
                    outer_fold=int(ref["outer_fold"]),
                    env_id=str(env_id),
                    request_kind="harvest_relative_calendar_month_precipitation",
                    start=harvest_start,
                    end=harvest_end,
                    metadata=row,
                    required_variables="pr",
                    date_source=str(row["harvest_anchor_source"]),
                )
            )
    return pd.DataFrame(queue), pd.DataFrame(harvest_rows)


def unique_daily_requests(queue: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for request_id, frame in queue.groupby("request_id", sort=True, dropna=False):
        first = frame.iloc[0]
        contexts = sorted(
            f"{scenario}:outer_{outer_fold}"
            for scenario, outer_fold in frame[["scenario", "outer_fold"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        rows.append(
            {
                "request_id": request_id,
                "request_kind": first["request_kind"],
                "request_start_date": first["request_start_date"],
                "request_end_date": first["request_end_date"],
                "latitude": first["latitude"],
                "longitude": first["longitude"],
                "required_daily_variables": first["required_daily_variables"],
                "request_status": first["request_status"],
                "fold_context_count": len(contexts),
                "fold_contexts": ";".join(contexts),
                "environment_count": frame["env_id"].nunique(),
                "environment_ids": ";".join(sorted(frame["env_id"].astype(str).unique())),
            }
        )
    return pd.DataFrame(rows)


def fixed_window_backcast_values(
    refs: list[dict[str, object]],
    observed: pd.DataFrame,
    window_precip: pd.Series,
) -> pd.DataFrame:
    rows = []
    for ref in refs:
        fit_ids = pd.Index(ref["fit_ids"])
        for source_token in sorted(CROP_PRECIPITATION_FIELDS):
            target = numeric_column(observed.reindex(fit_ids), source_token)
            candidate = pd.to_numeric(window_precip.reindex(fit_ids), errors="coerce")
            for env_id in fit_ids:
                rows.append(
                    {
                        "scenario": ref["scenario"],
                        "outer_fold": ref["outer_fold"],
                        "env_id": env_id,
                        "source_token": source_token,
                        "observed_historical_value": target.loc[env_id],
                        "replacement_backcast_value": candidate.loc[env_id],
                        "replacement_method": "fixed_sowing_relative_d0_180_precipitation_total_mm",
                        "fit_partition": "outer_training_environments_only",
                        "phenotype_derived": False,
                    }
                )
    return pd.DataFrame(rows)


def backcast_metrics(
    inventory: pd.DataFrame,
    refs: list[dict[str, object]],
    observed: pd.DataFrame,
    window_precip: pd.Series,
    protocol: dict[str, object],
) -> pd.DataFrame:
    policy = dict(protocol["crop_precipitation_backcast"])
    rows: list[dict[str, object]] = []
    for ref in refs:
        fit_ids = pd.Index(ref["fit_ids"])
        for source_token in inventory["source_token"].astype(str):
            row: dict[str, object] = {
                "scenario": ref["scenario"],
                "outer_fold": ref["outer_fold"],
                "source_token": source_token,
                "fit_environment_count": len(fit_ids),
                "observed_nonmissing": int(
                    pd.to_numeric(observed.reindex(fit_ids).get(source_token), errors="coerce").notna().sum()
                ) if source_token in observed else 0,
                "candidate_nonmissing": 0,
                "candidate_coverage_fraction": 0.0,
                "replacement_method": "",
                "backcast_status": "",
                "backcast_reason": "",
            }
            if source_token in CROP_PRECIPITATION_FIELDS:
                candidate = window_precip.reindex(fit_ids)
                target = observed[source_token].reindex(fit_ids)
                metrics = finite_metrics(target, candidate)
                coverage = float(candidate.notna().mean())
                row.update(metrics)
                row.update(
                    {
                        "candidate_nonmissing": int(candidate.notna().sum()),
                        "candidate_coverage_fraction": coverage,
                        "replacement_method": "fixed_sowing_relative_d0_180_precipitation_total_mm",
                    }
                )
                ratio = float(metrics["median_ratio_candidate_to_observed"])
                passed = all(
                    [
                        int(metrics["paired_environments"])
                        >= int(policy["minimum_paired_training_environments"]),
                        coverage >= float(policy["minimum_candidate_coverage_fraction"]),
                        np.isfinite(metrics["pearson"])
                        and float(metrics["pearson"]) >= float(policy["minimum_pearson"]),
                        np.isfinite(metrics["normalized_rmse"])
                        and float(metrics["normalized_rmse"])
                        <= float(policy["maximum_normalized_rmse"]),
                        np.isfinite(ratio)
                        and float(policy["minimum_median_ratio"])
                        <= ratio
                        <= float(policy["maximum_median_ratio"]),
                    ]
                )
                row["backcast_status"] = "PASS" if passed else "FAIL"
                row["backcast_reason"] = (
                    "direct_same_unit_fixed_window_candidate_meets_frozen_training_checks"
                    if passed
                    else "fixed_window_candidate_fails_one_or_more_frozen_training_checks"
                )
            elif source_token in HARVEST_MONTH_FIELDS:
                row.update(
                    {
                        "replacement_method": "calendar_month_precipitation_from_daily_pr_anchored_to_recorded_or_training_modeled_harvest",
                        "backcast_status": "BLOCKED",
                        "backcast_reason": "daily_precipitation_backcast_not_yet_populated",
                    }
                )
            elif source_token in MOISTURE_FIELDS:
                row.update(
                    {
                        "replacement_method": "antecedent_90_day_pr_minus_et0_plus_declared_irrigation_with_optional_soil_moisture",
                        "backcast_status": "BLOCKED",
                        "backcast_reason": "antecedent_daily_water_balance_not_yet_populated",
                    }
                )
            else:
                row.update(
                    {
                        "replacement_method": "manual_environment_source_adjudication",
                        "backcast_status": "BLOCKED",
                        "backcast_reason": "no_frozen_reconstruction_method",
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def replacement_contract(
    inventory: pd.DataFrame, backcast: pd.DataFrame, protocol: dict[str, object]
) -> pd.DataFrame:
    rows = []
    minimum_pass_rate = float(
        dict(protocol["crop_precipitation_backcast"])["minimum_fold_pass_rate"]
    )
    for record in inventory.to_dict("records"):
        source_token = str(record["source_token"])
        metrics = backcast[backcast["source_token"].eq(source_token)]
        if source_token in CROP_PRECIPITATION_FIELDS:
            pass_rate = float(metrics["backcast_status"].eq("PASS").mean())
            status = (
                "CERTIFIED_FIXED_WINDOW_REPLACEMENT"
                if pass_rate >= minimum_pass_rate
                else "BLOCKED_FIXED_WINDOW_AGREEMENT_FAILED"
            )
            requirement = "fixed_sowing_date_and_daily_or_certified_d0_180_precipitation"
        elif source_token in HARVEST_MONTH_FIELDS:
            pass_rate = 0.0
            status = "BLOCKED_DAILY_PRECIPITATION_REQUIRED"
            requirement = "daily_pr_and_nonphenotypic_or_training_modeled_harvest_anchor"
        elif source_token in MOISTURE_FIELDS:
            pass_rate = 0.0
            status = "BLOCKED_ANTECEDENT_WATER_BALANCE_REQUIRED"
            requirement = "daily_pr_et0_optional_soil_moisture_and_declared_irrigation"
        else:
            pass_rate = 0.0
            status = "BLOCKED_MANUAL_SOURCE_ADJUDICATION"
            requirement = "manual_environment_source_adjudication"
        rows.append(
            {
                **record,
                "replacement_method": metrics["replacement_method"].iloc[0],
                "fold_backcast_pass_rate": pass_rate,
                "replacement_status": status,
                "required_future_inputs": requirement,
                "phenotype_derived": False,
                "fit_partition": "outer_training_environments_only",
                "future_population_allowed": status == "CERTIFIED_FIXED_WINDOW_REPLACEMENT",
            }
        )
    return pd.DataFrame(rows)


def annual_precipitation_audit(
    refs: list[dict[str, object]],
    observed: pd.DataFrame,
    weather: pd.DataFrame,
    protocol: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = dict(protocol["annual_precipitation_audit"])
    rows = []
    overlaps = []
    for ref in refs:
        fit_ids = pd.Index(ref["fit_ids"])
        fit_observed = observed.reindex(fit_ids)
        fit_weather = weather.reindex(fit_ids)
        precipitation = numeric_column(fit_weather, "precipitation_total_mm")
        days = numeric_column(fit_weather, "n_days_weather")
        diagnostic_annualized = precipitation * 365.0 / days.where(days.gt(0))
        for field in policy["fields"]:
            values = numeric_column(fit_observed, str(field))
            finite = values[np.isfinite(values)]
            plausible = finite.between(
                float(policy["plausible_mm_minimum"]),
                float(policy["plausible_mm_maximum"]),
                inclusive="both",
            )
            metrics = finite_metrics(values, diagnostic_annualized)
            plausible_fraction = float(plausible.mean()) if len(finite) else math.nan
            rows.append(
                {
                    "scenario": ref["scenario"],
                    "outer_fold": ref["outer_fold"],
                    "field": field,
                    "fit_environment_count": len(fit_ids),
                    "observed_nonmissing": len(finite),
                    "observed_negative_count": int((finite < 0).sum()),
                    "observed_min": float(finite.min()) if len(finite) else math.nan,
                    "observed_q01": float(finite.quantile(0.01)) if len(finite) else math.nan,
                    "observed_median": float(finite.median()) if len(finite) else math.nan,
                    "observed_q99": float(finite.quantile(0.99)) if len(finite) else math.nan,
                    "observed_max": float(finite.max()) if len(finite) else math.nan,
                    "plausible_mm_fraction": plausible_fraction,
                    "trial_season_annualized_paired_environments": metrics["paired_environments"],
                    "trial_season_annualized_pearson": metrics["pearson"],
                    "trial_season_annualized_median_ratio": metrics[
                        "median_ratio_candidate_to_observed"
                    ],
                    "unit_audit_status": (
                        "PLAUSIBLE_MM_RANGE"
                        if np.isfinite(plausible_fraction)
                        and plausible_fraction >= float(policy["minimum_plausible_fraction"])
                        else "REQUIRES_UNIT_REVIEW"
                    ),
                    "period_contract_status": "BLOCKED_DAILY_PERIOD_ADJUDICATION",
                    "diagnostic_only": True,
                }
            )
        left = numeric_column(fit_observed, str(policy["fields"][0]))
        right = numeric_column(fit_observed, str(policy["fields"][1]))
        pair = finite_metrics(left, right)
        overlaps.append(
            {
                "scenario": ref["scenario"],
                "outer_fold": ref["outer_fold"],
                "field_a": policy["fields"][0],
                "field_b": policy["fields"][1],
                **pair,
                "identity_status": "SAME_UNITS_POSSIBLE_PERIOD_STILL_UNRESOLVED",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(overlaps)


def block_range_contract(protocol: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_block": block,
                "primary_checks": values["primary_checks"],
                "global_standardized_z_is_hard_gate": values[
                    "global_standardized_z_is_hard_gate"
                ],
                "contract_status": "FROZEN_FOR_HISTORICAL_RECONSTRUCTION",
            }
            for block, values in dict(protocol["block_range_rules"]).items()
        ]
    )


def infer_window_days(feature: str) -> int | None:
    match = re.search(r"api_d(\d+)_(\d+)_", feature)
    if not match:
        return None
    return int(match.group(2)) - int(match.group(1))


def range_rule_amendment(range_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    extreme = range_audit[
        range_audit["historical_range_rule_status"].eq(
            "HISTORICAL_BASELINE_EXCEEDS_GLOBAL_HARD_Z"
        )
    ]
    for record in extreme.to_dict("records"):
        feature = str(record["feature"])
        upper = infer_window_days(feature)
        if "TOTAL_PRECIPIT_IN_12_MONTHS" in feature:
            rule = "nonnegative_mm_and_explicit_12_month_period_after_daily_adjudication"
            status = "BLOCKED_ANNUAL_PERIOD_ADJUDICATION"
            lower = 0.0
            upper_value: float | str = ""
        elif "precipitation_total_mm" in feature.lower() or "rain" in feature.lower():
            rule = "nonnegative_accumulation_plus_fold_training_robust_tail"
            status = "FROZEN_RULE_READY_PENDING_FUTURE_APPLICATION"
            lower = 0.0
            upper_value = ""
        elif "_days_" in feature.lower() or "heat_days" in feature.lower() or "cold_days" in feature.lower():
            rule = "integer_day_count_within_source_window_plus_fold_training_robust_tail"
            status = "FROZEN_RULE_READY_PENDING_FUTURE_APPLICATION"
            lower = 0.0
            upper_value = upper if upper is not None else "source_period_day_count"
        else:
            rule = "physical_domain_plus_fold_training_robust_tail"
            status = "FROZEN_RULE_READY_PENDING_FUTURE_APPLICATION"
            lower = "feature_specific"
            upper_value = "feature_specific"
        rows.append(
            {
                **record,
                "replacement_range_rule": rule,
                "physical_minimum": lower,
                "physical_maximum": upper_value,
                "fold_standardized_z_role": "diagnostic_not_sole_hard_gate",
                "range_contract_status": status,
            }
        )
    return pd.DataFrame(rows)


def forbid_future_outputs(out_dir: Path) -> bool:
    forbidden_tokens = ("E_REACTION_NORM_RCP", "prediction", "future_matrix")
    return not any(
        any(token.lower() in path.name.lower() for token in forbidden_tokens)
        for path in out_dir.iterdir()
        if path.is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit phenotype-blind historical replacements required before RCP "
            "covariate population. No future matrix or prediction is generated."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--outer-dir", type=Path, required=True)
    parser.add_argument("--outer-protocol", type=Path, required=True)
    parser.add_argument("--readiness-dir", type=Path, required=True)
    parser.add_argument("--reconstruction-protocol", type=Path, required=True)
    parser.add_argument("--environment-dir", type=Path)
    parser.add_argument("--weather-dir", type=Path)
    parser.add_argument("--fetch-manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    outer_dir = resolve(root, args.outer_dir)
    outer_protocol_path = resolve(root, args.outer_protocol)
    readiness_dir = resolve(root, args.readiness_dir)
    protocol_path = resolve(root, args.reconstruction_protocol)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outer_protocol = read_json(outer_protocol_path)
    readiness_cert_path = readiness_dir / "RCP_feature_readiness_certification.json"
    readiness_cert = read_json(readiness_cert_path)
    protocol = read_json(protocol_path)
    if protocol.get("status") != "frozen_before_historical_reconstruction":
        raise SystemExit("Historical reconstruction protocol is not frozen")
    if any(
        protocol.get(key) is not False
        for key in (
            "phenotype_values_allowed",
            "outer_test_outcomes_allowed",
            "outer_test_metrics_allowed",
            "final_holdout_outcomes_allowed",
            "model_selection_allowed",
            "future_covariate_matrices_allowed",
            "rcp_predictions_allowed",
        )
    ):
        raise SystemExit("Historical reconstruction protocol is not phenotype-blind and audit-only")
    if readiness_cert.get("status") != "PASS" or readiness_cert.get(
        "future_covariate_population_allowed"
    ) is not False:
        raise SystemExit("Corrected RCP readiness audit is absent or does not remain blocked")
    if outer_protocol.get("selected_environment_architecture") != "explicit_E_REACTION_NORM_V1":
        raise SystemExit("Outer protocol does not use explicit_E_REACTION_NORM_V1")

    lineage_path = readiness_dir / "RCP_feature_readiness_lineage.tsv"
    range_audit_path = readiness_dir / "RCP_historical_range_rule_audit.tsv"
    artifacts = readiness_cert.get("artifacts", {})
    for path in (lineage_path, range_audit_path):
        expected = artifacts.get(path.name) if isinstance(artifacts, dict) else None
        if expected and file_sha256(path) != expected:
            raise SystemExit(f"Readiness artifact checksum mismatch: {path}")

    refs = references(outer_dir, outer_protocol, root)
    envdata_path = unique_reference_path(refs, "envdata_path")
    window_path = unique_reference_path(refs, "window_path")
    environment_dir = (
        resolve(root, args.environment_dir) if args.environment_dir is not None else envdata_path.parent
    )
    weather_dir = infer_weather_dir(refs, environment_dir, args.weather_dir, root)
    manifest_path = (
        resolve(root, args.fetch_manifest)
        if args.fetch_manifest is not None
        else weather_dir / "trial_weather_fetch_manifest.tsv"
    )
    if not manifest_path.is_file():
        alternate = environment_dir / "trial_weather_fetch_manifest.tsv"
        if alternate.is_file():
            manifest_path = alternate
        else:
            raise SystemExit(f"Weather fetch manifest is missing: {manifest_path}")

    lineage = pd.read_csv(lineage_path, sep="\t", dtype=str)
    range_audit = pd.read_csv(range_audit_path, sep="\t", low_memory=False)
    inventory = source_inventory(lineage)
    expected_sources = int(protocol["expected_unique_historical_source_count"])
    if len(inventory) != expected_sources:
        raise SystemExit(
            f"Historical source inventory changed: observed={len(inventory)} expected={expected_sources}"
        )
    all_fit_ids = pd.Index(
        sorted({str(env_id) for ref in refs for env_id in pd.Index(ref["fit_ids"])})
    )
    envdata = pd.read_csv(envdata_path, sep="\t", dtype=str, low_memory=False)
    observed = read_observed_environment_values(
        envdata, all_fit_ids, inventory["source_token"].astype(str)
    )
    window_precip = read_window_precipitation(window_path, all_fit_ids)
    weather = read_weather_aggregate(weather_dir, all_fit_ids)
    metadata = read_fetch_manifest(manifest_path, all_fit_ids)

    backcast_values = fixed_window_backcast_values(refs, observed, window_precip)
    backcast = backcast_metrics(inventory, refs, observed, window_precip, protocol)
    contract = replacement_contract(inventory, backcast, protocol)
    queue, harvest = build_daily_queue(refs, metadata, protocol)
    queue_unique = unique_daily_requests(queue)
    annual, annual_overlap = annual_precipitation_audit(refs, observed, weather, protocol)
    range_blocks = block_range_contract(protocol)
    range_features = range_rule_amendment(range_audit)

    frames = {
        "contract": contract,
        "backcast_values": backcast_values,
        "backcast": backcast,
        "harvest": harvest,
        "queue": queue,
        "queue_unique": queue_unique,
        "annual": annual,
        "annual_overlap": annual_overlap,
        "range_blocks": range_blocks,
        "range_features": range_features,
    }
    for key, frame in frames.items():
        write_tsv(frame, out_dir / OUTPUT_FILENAMES[key])

    ready_queue = queue["request_status"].eq("READY_TO_FETCH")
    ready_unique_queue = queue_unique["request_status"].eq("READY_TO_FETCH")
    contract_ready = bool(contract["future_population_allowed"].all())
    annual_period_ready = bool(annual["period_contract_status"].eq("PASS").all())
    range_ready = bool(range_features["range_contract_status"].ne(
        "BLOCKED_ANNUAL_PERIOD_ADJUDICATION"
    ).all()) if not range_features.empty else True
    summary = pd.DataFrame(
        [
            {"metric": "historical_source_count", "value": len(contract)},
            {
                "metric": "certified_fixed_window_replacement_count",
                "value": int(contract["replacement_status"].eq("CERTIFIED_FIXED_WINDOW_REPLACEMENT").sum()),
            },
            {
                "metric": "blocked_daily_precipitation_source_count",
                "value": int(contract["replacement_status"].eq("BLOCKED_DAILY_PRECIPITATION_REQUIRED").sum()),
            },
            {
                "metric": "blocked_antecedent_water_balance_source_count",
                "value": int(contract["replacement_status"].eq("BLOCKED_ANTECEDENT_WATER_BALANCE_REQUIRED").sum()),
            },
            {"metric": "outer_training_fold_count", "value": len(refs)},
            {"metric": "daily_work_queue_rows", "value": len(queue)},
            {"metric": "daily_work_queue_ready_rows", "value": int(ready_queue.sum())},
            {
                "metric": "daily_work_queue_unique_requests",
                "value": int(ready_unique_queue.sum()),
            },
            {
                "metric": "annual_period_contract_ready",
                "value": annual_period_ready,
            },
            {"metric": "historical_replacement_contract_ready", "value": contract_ready},
            {"metric": "range_contract_ready", "value": range_ready},
            {"metric": "future_covariate_population_allowed", "value": False},
            {"metric": "rcp_predictions_allowed", "value": False},
        ]
    )
    write_tsv(summary, out_dir / OUTPUT_FILENAMES["summary"])

    checks = {
        "outer_protocol_frozen": outer_protocol.get("status")
        == "frozen_after_inner_validation_before_outer_test",
        "selected_environment_architecture": outer_protocol.get(
            "selected_environment_architecture"
        )
        == "explicit_E_REACTION_NORM_V1",
        "readiness_audit_pass": readiness_cert.get("status") == "PASS",
        "readiness_population_block_preserved": readiness_cert.get(
            "future_covariate_population_allowed"
        )
        is False,
        "protocol_is_phenotype_blind": protocol.get("phenotype_values_allowed") is False,
        "historical_source_count": len(contract) == expected_sources,
        "one_terminal_status_per_source": contract["source_token"].nunique() == len(contract)
        and contract["replacement_status"].astype(str).str.len().gt(0).all(),
        "all_fold_training_references_audited": backcast[["scenario", "outer_fold"]]
        .drop_duplicates()
        .shape[0]
        == len(refs),
        "daily_work_queue_covers_all_folds": queue[["scenario", "outer_fold"]]
        .drop_duplicates()
        .shape[0]
        == len(refs),
        "daily_work_queue_has_required_request_classes": {
            "annual_precipitation_trailing_365_before_sowing",
            "pre_sowing_antecedent_water_balance",
            "harvest_relative_calendar_month_precipitation",
        }.issubset(set(queue["request_kind"].astype(str))),
        "daily_unique_request_inventory_matches_queue": set(
            queue_unique["request_id"].astype(str)
        )
        == set(queue["request_id"].astype(str)),
        "fixed_window_backcast_uses_training_ids_only": len(backcast_values)
        == 2 * sum(len(pd.Index(ref["fit_ids"])) for ref in refs),
        "annual_fields_audited": set(annual["field"].astype(str))
        == set(ANNUAL_PRECIPITATION_FIELDS),
        "annual_periods_remain_blocked": not annual_period_ready,
        "future_population_remains_blocked": not contract_ready or not annual_period_ready,
        "no_future_matrix_or_prediction_generated": forbid_future_outputs(out_dir),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    input_artifacts = {
        "outer_protocol": file_sha256(outer_protocol_path),
        "readiness_certification": file_sha256(readiness_cert_path),
        "readiness_lineage": file_sha256(lineage_path),
        "readiness_range_audit": file_sha256(range_audit_path),
        "reconstruction_protocol": file_sha256(protocol_path),
        "envdata": file_sha256(envdata_path),
        "window_features": file_sha256(window_path),
        "fetch_manifest": file_sha256(manifest_path),
    }
    for filename in (
        "trial_weather_features_openmeteo.tsv",
        "trial_weather_features_nasa_power.tsv",
    ):
        path = weather_dir / filename
        if path.is_file():
            input_artifacts[filename] = file_sha256(path)
    output_artifacts = {
        path.name: file_sha256(path)
        for path in sorted(out_dir.iterdir())
        if path.is_file()
        and path.name
        not in {OUTPUT_FILENAMES["provenance"], OUTPUT_FILENAMES["certification"]}
    }
    provenance = {
        "status": "PASS" if not failed else "FAIL",
        "protocol_version": protocol["protocol_version"],
        "selection_data": protocol["selection_data"],
        "phenotype_values_read": False,
        "outer_test_environment_identifiers_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "future_matrix_count_generated": 0,
        "rcp_prediction_count_generated": 0,
        "environment_input_dir": str(environment_dir),
        "weather_input_dir": str(weather_dir),
        "historical_replacement_contract_ready": contract_ready,
        "annual_period_contract_ready": annual_period_ready,
        "future_covariate_population_allowed": False,
        "rcp_predictions_allowed": False,
        "input_artifacts": input_artifacts,
        "output_artifacts": output_artifacts,
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out_dir / OUTPUT_FILENAMES["provenance"]).write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    certification = {
        **provenance,
        "audit_complete": not failed,
        "checks": checks,
        "failed_checks": failed,
        "historical_source_count": len(contract),
        "certified_replacement_count": int(contract["future_population_allowed"].sum()),
        "blocked_replacement_count": int((~contract["future_population_allowed"]).sum()),
        "daily_work_queue_rows": len(queue),
        "daily_work_queue_ready_rows": int(ready_queue.sum()),
        "daily_work_queue_unique_requests": int(
            ready_unique_queue.sum()
        ),
        "historical_range_rule_amendment_count": len(range_features),
    }
    (out_dir / OUTPUT_FILENAMES["certification"]).write_text(
        json.dumps(certification, indent=2), encoding="utf-8"
    )
    print(json.dumps(certification, indent=2), flush=True)
    if failed:
        raise SystemExit("Historical RCP reconstruction audit failed")


if __name__ == "__main__":
    main()
