from __future__ import annotations

import argparse
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLS = ["Trial_name", "Occ", "Loc_no", "Country", "Loc_desc", "Cycle"]
DATE_TRAITS = {
    "SOWING_DATE": "sowing_date",
    "SOWING_DATE_TEXT": "sowing_date",
    "SOWING_OLD": "sowing_date",
    "EMERGENCE_DATE": "emergence_date",
    "EMERGENCE_DATE_TEXT": "emergence_date",
    "HARVEST_STARTING_DATE": "harvest_start_date",
    "HARVEST_STARTING_DATE_TEXT": "harvest_start_date",
    "HARVEST_FINISHING_DATE": "harvest_finish_date",
    "HARVEST_FINISHING_DATE_TEXT": "harvest_finish_date",
}
DATE_FIELDS = list(dict.fromkeys(DATE_TRAITS.values()))
AUTO_ACCEPT_MATCH_METHODS = {"exact_env_id", "normalized_environment_key"}
MISSING_TOKENS = {"", "UNKNOWN", "NA", "N/A", "NONE", "-", "?"}
MAX_REASONABLE_DATE = pd.Timestamp.today().normalize()


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def clean_column(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def normalized_token(value: object) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).strip()).upper()
    return re.sub(r"\.0$", "", text)


def normalized_trial_dir(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\\", "/").rstrip("/").split("/")[-1].lower()


def environment_id(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(index=frame.index, dtype=str)
    values = frame.reindex(columns=ID_COLS).fillna("").astype(str)
    return values.apply(lambda row: "|".join(row), axis=1)


def normalized_environment_key(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(index=frame.index, dtype=str)
    values = frame.reindex(columns=ID_COLS).copy()
    for column in ID_COLS:
        values[column] = values[column].map(normalized_token)
    return values.apply(lambda row: "|".join(row), axis=1)


def normalized_trial_location_key(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(index=frame.index, dtype=str)
    work = pd.DataFrame(index=frame.index)
    work["trial_dir"] = frame.get("trial_dir", pd.Series("", index=frame.index)).map(
        normalized_trial_dir
    )
    for column in ["Occ", "Loc_no", "Cycle"]:
        work[column] = frame.get(column, pd.Series("", index=frame.index)).map(
            normalized_token
        )
    return work.apply(lambda row: "|".join(row), axis=1)


def parse_year(value: str) -> int | None:
    if not value or int(value) == 0:
        return None
    year = int(value)
    if len(value) == 2:
        year += 2000 if year <= 30 else 1900
    return year


def valid_timestamp(value: date | datetime | pd.Timestamp) -> pd.Timestamp | None:
    parsed = pd.Timestamp(value).normalize()
    if parsed.year < 1900 or parsed > MAX_REASONABLE_DATE:
        return None
    return parsed


def parse_raw_date(value: object) -> dict[str, object]:
    result: dict[str, object] = {
        "parsed_date": pd.NaT,
        "parse_status": "invalid_or_unrecognized",
        "partial_year": np.nan,
        "partial_month": np.nan,
        "partial_day": np.nan,
        "date_uncertainty_days": np.nan,
    }
    if pd.isna(value):
        result["parse_status"] = "missing"
        return result
    if isinstance(value, (date, datetime, pd.Timestamp)):
        parsed = valid_timestamp(value)
        if parsed is not None:
            result.update(parsed_date=parsed, parse_status="full_date", date_uncertainty_days=0)
        return result
    text = str(value).strip()
    if text.upper() in MISSING_TOKENS:
        result["parse_status"] = "missing"
        return result

    if re.fullmatch(r"\d+(?:\.0+)?", text):
        serial = float(text)
        if 1 <= serial <= 80000:
            parsed = valid_timestamp(pd.Timestamp("1899-12-30") + pd.to_timedelta(serial, unit="D"))
            if parsed is not None:
                result.update(
                    parsed_date=parsed,
                    parse_status="full_date_excel_serial",
                    date_uncertainty_days=0,
                )
                return result

    numeric = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", text)
    if numeric:
        day_value = int(numeric.group(1))
        month_value = int(numeric.group(2))
        year_value = parse_year(numeric.group(3))
        result.update(
            partial_day=day_value or np.nan,
            partial_month=month_value or np.nan,
            partial_year=year_value or np.nan,
        )
        if year_value and day_value and month_value:
            try:
                parsed = valid_timestamp(date(year_value, month_value, day_value))
            except ValueError:
                parsed = None
            if parsed is not None:
                result.update(parsed_date=parsed, parse_status="full_date", date_uncertainty_days=0)
                return result
        if year_value and month_value and not day_value and 1 <= month_value <= 12:
            result.update(parse_status="partial_month_year", date_uncertainty_days=31)
        elif year_value and not month_value:
            result.update(parse_status="year_only_or_missing_month", date_uncertainty_days=366)
        else:
            result["parse_status"] = "invalid_partial_date"
        return result

    month_name = re.fullmatch(r"([A-Za-z]{3,9})\s+(\d{1,2})[,]?\s+(\d{4})", text)
    day_name = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]{3,9})[,]?\s+(\d{4})", text)
    iso = re.fullmatch(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", text)
    try:
        if month_name or day_name:
            parsed_value = pd.to_datetime(text, errors="raise", dayfirst=bool(day_name))
            parsed = valid_timestamp(parsed_value)
        elif iso:
            parsed = valid_timestamp(
                date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
            )
        else:
            parsed = None
    except (TypeError, ValueError, OverflowError):
        parsed = None
    if parsed is not None:
        result.update(parsed_date=parsed, parse_status="full_date", date_uncertainty_days=0)
    return result


def normalized_full_date(value: object) -> object:
    if pd.isna(value):
        return np.nan
    parsed = parse_raw_date(value)["parsed_date"]
    return pd.Timestamp(parsed).strftime("%Y-%m-%d") if pd.notna(parsed) else np.nan


def cycle_year_bounds(value: object) -> tuple[int, int] | None:
    text = normalized_token(value)
    four_digit = [int(item) for item in re.findall(r"(?:19|20)\d{2}", text)]
    if four_digit:
        return min(four_digit), max(four_digit) + 1
    short_range = re.search(r"(?<!\d)(\d{2})\s*[-/]\s*(\d{2})(?!\d)", text)
    if not short_range:
        return None
    first = parse_year(short_range.group(1))
    second = parse_year(short_range.group(2))
    if first is None or second is None:
        return None
    if second < first:
        second += 100
    return first, second


def cycle_date_status(cycle: object, parsed: object) -> str:
    if pd.isna(parsed):
        return "not_applicable"
    bounds = cycle_year_bounds(cycle)
    if bounds is None:
        return "cycle_unavailable"
    year = pd.Timestamp(parsed).year
    return "cycle_plausible" if bounds[0] <= year <= bounds[1] else "cycle_mismatch"


def read_raw_table(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        signature = handle.read(4)
    if signature[:2] == b"PK" or signature == b"\xd0\xcf\x11\xe0":
        return pd.read_excel(path, dtype=str)
    last_error: Exception | None = None
    for encoding in ["utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                dtype=str,
                encoding=encoding,
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError(f"Could not read {path}: {last_error}")


def unique_mapping(keys: pd.Series, values: pd.Series) -> dict[str, str]:
    frame = pd.DataFrame({"key": keys, "env_id": values}).drop_duplicates()
    counts = frame.groupby("key")["env_id"].nunique()
    valid = set(counts[counts.eq(1)].index)
    return frame[frame["key"].isin(valid)].set_index("key")["env_id"].to_dict()


def target_indexes(targets: pd.DataFrame) -> dict[str, object]:
    targets = targets.copy()
    targets["env_id"] = targets["env_id"].fillna("").astype(str)
    return {
        "exact": set(targets["env_id"]),
        "normalized": unique_mapping(normalized_environment_key(targets), targets["env_id"]),
        "trial_location": unique_mapping(normalized_trial_location_key(targets), targets["env_id"]),
        "cycle": targets.drop_duplicates("env_id").set_index("env_id")["Cycle"].to_dict(),
    }


def match_raw_rows(frame: pd.DataFrame, indexes: dict[str, object]) -> pd.DataFrame:
    output = frame.copy()
    output["raw_env_id"] = environment_id(output)
    output["target_env_id"] = ""
    output["match_method"] = "unmatched"
    exact = output["raw_env_id"].isin(indexes["exact"])
    output.loc[exact, "target_env_id"] = output.loc[exact, "raw_env_id"]
    output.loc[exact, "match_method"] = "exact_env_id"

    pending = output["target_env_id"].eq("")
    normalized = normalized_environment_key(output)
    normalized_match = normalized.map(indexes["normalized"]).fillna("")
    use = pending & normalized_match.ne("")
    output.loc[use, "target_env_id"] = normalized_match.loc[use]
    output.loc[use, "match_method"] = "normalized_environment_key"

    pending = output["target_env_id"].eq("")
    trial_location = normalized_trial_location_key(output)
    fallback_match = trial_location.map(indexes["trial_location"]).fillna("")
    use = pending & fallback_match.ne("")
    output.loc[use, "target_env_id"] = fallback_match.loc[use]
    output.loc[use, "match_method"] = "trial_dir_loc_occ_cycle_review_only"
    return output[output["target_env_id"].ne("")].copy()


def discover_date_candidates(
    trial_root: Path, targets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexes = target_indexes(targets)
    candidate_frames: list[pd.DataFrame] = []
    file_rows: list[dict[str, object]] = []
    paths = sorted(
        path
        for path in trial_root.rglob("*")
        if path.is_file()
        and re.search(r"env[ _-]*data", path.name, flags=re.IGNORECASE)
        and path.suffix.lower() in {".xls", ".xlsx", ".txt", ".tsv", ".csv"}
    )
    for path in paths:
        relative = path.relative_to(trial_root).as_posix()
        try:
            raw = read_raw_table(path)
            raw.columns = [clean_column(column) for column in raw.columns]
            missing = sorted(set([*ID_COLS, "Trait_name", "Value"]).difference(raw.columns))
            if missing:
                file_rows.append(
                    {"source_file": relative, "status": "skipped_missing_columns", "detail": ";".join(missing), "rows": len(raw)}
                )
                continue
            raw["Trait_name"] = raw["Trait_name"].fillna("").astype(str).str.strip().str.upper()
            selected = raw[raw["Trait_name"].isin(DATE_TRAITS)].copy()
            selected["trial_dir"] = path.parent.name
            selected = match_raw_rows(selected, indexes)
            selected["date_field"] = selected["Trait_name"].map(DATE_TRAITS)
            selected["source_file"] = relative
            selected["source_row"] = selected.index.to_numpy() + 2
            if not selected.empty:
                parsed = pd.DataFrame(selected["Value"].map(parse_raw_date).tolist(), index=selected.index)
                selected = pd.concat([selected, parsed], axis=1)
                selected["cycle_date_status"] = [
                    cycle_date_status(indexes["cycle"].get(env_id, ""), parsed_date)
                    for env_id, parsed_date in zip(selected["target_env_id"], selected["parsed_date"])
                ]
                candidate_frames.append(selected)
            file_rows.append(
                {"source_file": relative, "status": "read", "detail": "", "rows": len(raw), "matched_date_rows": len(selected)}
            )
        except Exception as exc:
            file_rows.append(
                {"source_file": relative, "status": "read_error", "detail": f"{type(exc).__name__}: {exc}", "rows": np.nan}
            )
    candidates = pd.concat(candidate_frames, ignore_index=True, sort=False) if candidate_frames else pd.DataFrame()
    return candidates, pd.DataFrame(file_rows)


def resolve_candidates(
    targets: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_ids = targets["env_id"].fillna("").astype(str).drop_duplicates()
    resolution = pd.DataFrame({"env_id": target_ids})
    supplement = pd.DataFrame({"env_id": target_ids})
    conflicts: list[dict[str, object]] = []
    for field in DATE_FIELDS:
        accepted: dict[str, str] = {}
        statuses: dict[str, str] = {}
        for env_value in target_ids:
            selected = candidates[
                candidates["target_env_id"].eq(env_value)
                & candidates["date_field"].eq(field)
            ] if not candidates.empty else pd.DataFrame()
            full = selected[
                selected["parsed_date"].notna()
                & selected["match_method"].isin(AUTO_ACCEPT_MATCH_METHODS)
                & selected["cycle_date_status"].eq("cycle_plausible")
            ] if not selected.empty else selected
            unique_dates = sorted(pd.to_datetime(full["parsed_date"]).dt.strftime("%Y-%m-%d").unique()) if not full.empty else []
            if len(unique_dates) == 1:
                accepted[env_value] = unique_dates[0]
                statuses[env_value] = "accepted_unique_cycle_plausible_full_date"
            elif len(unique_dates) > 1:
                statuses[env_value] = "conflicting_full_dates"
                conflicts.append(
                    {"env_id": env_value, "date_field": field, "candidate_dates": ";".join(unique_dates), "candidate_count": len(full)}
                )
            elif not selected.empty and selected["parse_status"].eq("partial_month_year").any():
                statuses[env_value] = "partial_month_year_review_only"
            elif not selected.empty and selected["parsed_date"].notna().any():
                statuses[env_value] = "full_date_rejected_match_or_cycle"
            elif not selected.empty:
                statuses[env_value] = "invalid_or_incomplete_raw_date"
            else:
                statuses[env_value] = "no_raw_date_evidence"
        supplement[field] = supplement["env_id"].map(accepted)
        resolution[f"{field}_resolution"] = resolution["env_id"].map(statuses)
    accepted_mask = supplement[DATE_FIELDS].notna().any(axis=1)
    supplement = supplement.loc[accepted_mask].copy()
    if not supplement.empty:
        provenance = {}
        for env_value in supplement["env_id"]:
            source_files = sorted(
                candidates.loc[
                    candidates["target_env_id"].eq(env_value)
                    & candidates["parsed_date"].notna()
                    & candidates["cycle_date_status"].eq("cycle_plausible"),
                    "source_file",
                ].dropna().astype(str).unique()
            )
            provenance[env_value] = "raw_trial_envdata_exact_match:" + ";".join(source_files)
        supplement["provenance"] = supplement["env_id"].map(provenance)
    resolution["accepted_date_field_count"] = resolution["env_id"].isin(set(supplement["env_id"]))
    if not supplement.empty:
        counts = supplement.set_index("env_id")[DATE_FIELDS].notna().sum(axis=1)
        resolution["accepted_date_field_count"] = resolution["env_id"].map(counts).fillna(0).astype(int)
    else:
        resolution["accepted_date_field_count"] = 0
    return supplement, resolution, pd.DataFrame(conflicts)


def merge_base_supplement(recovered: pd.DataFrame, base_path: Path | None) -> pd.DataFrame:
    if base_path is None:
        return recovered
    base = pd.read_csv(base_path, sep=None, engine="python", dtype=str)
    if "env_id" not in base.columns or base["env_id"].fillna("").duplicated().any():
        raise ValueError(f"{base_path} must contain unique env_id values")
    for frame in [base, recovered]:
        for field in DATE_FIELDS:
            if field not in frame.columns:
                frame[field] = np.nan
        if "provenance" not in frame.columns:
            frame["provenance"] = ""
    combined = base.merge(recovered, on="env_id", how="outer", suffixes=("_base", "_raw"))
    output = combined[["env_id"]].copy()
    for field in DATE_FIELDS:
        left = combined.get(f"{field}_base", pd.Series(np.nan, index=combined.index))
        right = combined.get(f"{field}_raw", pd.Series(np.nan, index=combined.index))
        left_normalized = left.map(normalized_full_date)
        right_normalized = right.map(normalized_full_date)
        invalid_base = left.notna() & left_normalized.isna()
        if invalid_base.any():
            values = combined.loc[invalid_base, "env_id"].astype(str).tolist()
            raise ValueError(f"Base supplement contains invalid full dates for {field}: {values[:10]}")
        conflict = (
            left_normalized.notna()
            & right_normalized.notna()
            & left_normalized.ne(right_normalized)
        )
        if conflict.any():
            values = combined.loc[conflict, "env_id"].astype(str).tolist()
            raise ValueError(f"Base and raw date supplements conflict for {field}: {values[:10]}")
        output[field] = left_normalized.fillna(right_normalized)
    base_provenance = combined.get("provenance_base", pd.Series("", index=combined.index)).fillna("")
    raw_provenance = combined.get("provenance_raw", pd.Series("", index=combined.index)).fillna("")
    output["provenance"] = [
        ";".join(value for value in [base_value, raw_value] if value)
        for base_value, raw_value in zip(base_provenance, raw_provenance)
    ]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover complete trial dates conservatively from raw EnvData files."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--trial-root", type=Path, default=Path("TRIALS_AND_NURSERIES"))
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-supplement", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    trial_root = args.trial_root.resolve() if args.trial_root.is_absolute() else (root / args.trial_root).resolve()
    target_path = args.targets.resolve() if args.targets.is_absolute() else (root / args.targets).resolve()
    out_dir = args.out_dir.resolve() if args.out_dir.is_absolute() else (root / args.out_dir).resolve()
    base_path = None if args.base_supplement is None else (
        args.base_supplement.resolve() if args.base_supplement.is_absolute() else (root / args.base_supplement).resolve()
    )
    if not trial_root.is_dir():
        raise FileNotFoundError(f"Trial root does not exist: {trial_root}")
    targets = pd.read_csv(target_path, sep="\t", dtype=str, low_memory=False)
    if "env_id" not in targets.columns:
        raise ValueError(f"{target_path} is missing env_id")
    if "coverage_cause" in targets.columns:
        targets = targets[
            targets["coverage_cause"].isin(
                ["missing_fetch_window", "missing_window_and_coordinates"]
            )
        ].copy()
    candidates, files = discover_date_candidates(trial_root, targets)
    recovered, resolution, conflicts = resolve_candidates(targets, candidates)
    supplement = merge_base_supplement(recovered, base_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(candidates, out_dir / "raw_trial_date_candidates.tsv")
    write_tsv(files, out_dir / "raw_trial_date_file_audit.tsv")
    write_tsv(resolution, out_dir / "raw_trial_date_resolution.tsv")
    write_tsv(conflicts, out_dir / "raw_trial_date_conflicts.tsv")
    write_tsv(recovered, out_dir / "raw_trial_date_recovered_only.tsv")
    write_tsv(supplement, out_dir / "weather_date_supplement.tsv")
    qc = pd.DataFrame(
        [
            {"metric": "target_environment_count", "value": targets["env_id"].nunique()},
            {"metric": "raw_envdata_files_scanned", "value": len(files)},
            {"metric": "raw_envdata_read_errors", "value": int(files.get("status", pd.Series(dtype=str)).eq("read_error").sum())},
            {"metric": "matched_raw_date_rows", "value": len(candidates)},
            {"metric": "full_date_candidate_rows", "value": int(candidates.get("parsed_date", pd.Series(dtype=object)).notna().sum())},
            {"metric": "partial_month_year_rows", "value": int(candidates.get("parse_status", pd.Series(dtype=str)).eq("partial_month_year").sum())},
            {"metric": "recovered_environment_count", "value": recovered["env_id"].nunique() if not recovered.empty else 0},
            {"metric": "conflicting_environment_fields", "value": len(conflicts)},
            {"metric": "base_supplement_environment_count", "value": len(supplement)},
        ]
    )
    write_tsv(qc, out_dir / "raw_trial_date_recovery_qc.tsv")
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
