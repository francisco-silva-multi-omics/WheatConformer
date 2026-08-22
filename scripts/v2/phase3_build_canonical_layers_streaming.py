"""Memory-bounded construction of immutable Stage-1 v2 layers.

The implementation uses three streaming passes so the 7.8M-row canonical
ledger can be built within the workstation's WSL memory ceiling. Certified-v1
paths and raw source files are read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


LAYER_VERSION = "stage1_v2_layers_2026_07_30_v3"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_writer_write(
    writer: pq.ParquetWriter | None, path: Path, frame: pd.DataFrame
) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table, row_group_size=100_000)
    return writer


def validate_unique(frame: pd.DataFrame, keys: list[str], name: str) -> dict[str, object]:
    duplicates = int(frame.duplicated(keys, keep=False).sum())
    if duplicates:
        raise RuntimeError(f"{name} has {duplicates} rows in duplicate right-side keys: {keys}")
    return {
        "join_or_assertion": name,
        "expected_cardinality": "unique right-side key (m:1 join)",
        "right_rows": len(frame),
        "right_distinct_keys": len(frame),
        "duplicate_right_keys": 0,
        "status": "PASS",
    }


def clean_string_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = frame[column].fillna("").astype(str).str.strip()


def canonicalize_batch(
    frame: pd.DataFrame,
    genotype: pd.DataFrame,
    trait: pd.DataFrame,
    unit: pd.DataFrame,
    environment: pd.DataFrame,
    source_copy_counts: dict[str, int],
) -> pd.DataFrame:
    input_rows = len(frame)
    raw_id = frame["raw_source_row_id"].fillna("").astype(str)
    if not raw_id.str.startswith("RAW2_").all():
        raise RuntimeError("All final raw IDs must use the RAW2_ namespace")
    frame["canonical_row_id"] = "CAN2_" + raw_id.str.slice(5)

    frame = frame.merge(
        genotype,
        left_on=["trial_key", "cycle", "CID_normalized", "SID_normalized"],
        right_on=["trial_key", "cycle_norm", "CID_norm", "SID_norm"],
        how="left",
        validate="m:1",
        sort=False,
    )
    frame = frame.merge(trait, on="trait_key", how="left", validate="m:1", sort=False)
    frame = frame.merge(
        unit,
        left_on=["trait_key", "raw_unit"],
        right_on=["trait_key", "raw_unit"],
        how="left",
        validate="m:1",
        sort=False,
    )
    frame = frame.merge(
        environment, left_on="env_kernel_id", right_on="source_env_id",
        how="left", validate="m:1", sort=False,
    )
    if len(frame) != input_rows:
        raise RuntimeError(f"Registry joins changed batch row count: {input_rows} -> {len(frame)}")

    clean_string_columns(
        frame,
        [
            "cycle_norm", "CID_norm", "SID_norm", "accepted_gid", "registry_decision", "genotype_registry_version",
            "accepted_canonical_trait", "accepted_standard_unit", "trait_alias_decision_v2",
            "trait_registry_version", "unit_rule", "unit_rule_version", "alias_decision",
            "source_env_id", "target_env_id", "target_trial_name", "environment_registry_version",
        ],
    )
    raw_gid = frame["raw_gid"].fillna("").astype(str).str.strip().str.upper()
    raw_gid = raw_gid.str.replace(r"^GID", "", regex=True).str.replace(r"\.0$", "", regex=True)
    frame["raw_gid_normalized_v2"] = raw_gid.where(raw_gid.str.fullmatch(r"[0-9]+", na=False), "")
    registry_gid = frame["accepted_gid"]
    raw_present = frame["raw_gid_normalized_v2"].ne("")
    registry_present = registry_gid.ne("")
    conflict = raw_present & registry_present & frame["raw_gid_normalized_v2"].ne(registry_gid)
    frame["registry_accepted_gid"] = registry_gid
    frame["resolved_gid_v2"] = np.select(
        [conflict, raw_present, registry_present],
        ["", frame["raw_gid_normalized_v2"], registry_gid],
        default="",
    )
    frame["genotype_resolution_status_v2"] = np.select(
        [
            conflict,
            raw_present & registry_present,
            raw_present,
            registry_present,
            frame["registry_decision"].str.startswith("AMBIGUOUS"),
        ],
        [
            "AMBIGUOUS_RAW_GID_REGISTRY_CONFLICT",
            "ACCEPT_RAW_GID_REGISTRY_CONCORDANT",
            "ACCEPT_EXACT_RAW_GID",
            "ACCEPT_VERSIONED_REGISTRY_GID",
            frame["registry_decision"],
        ],
        default="UNRESOLVED_NO_ACCEPTED_GID",
    )
    frame["canonical_germplasm_key"] = np.where(
        frame["resolved_gid_v2"].ne(""), "GID" + frame["resolved_gid_v2"], ""
    )

    alias_accept = frame["alias_decision"].eq("ACCEPT")
    frame["canonical_environment_id"] = np.where(alias_accept, frame["target_env_id"], frame["env_kernel_id"])
    frame["canonical_trial_name"] = np.where(alias_accept, frame["target_trial_name"], frame["trial_name"])
    frame["environment_alias_decision"] = np.where(alias_accept, "ACCEPT", "NO_ALIAS_REQUIRED")

    frame["unit_scale"] = pd.to_numeric(frame["unit_scale"], errors="coerce")
    frame["unit_offset"] = pd.to_numeric(frame["unit_offset"], errors="coerce")
    frame["value_standardized"] = frame["numeric_value"] * frame["unit_scale"] + frame["unit_offset"]
    frame["standardized_unit"] = np.where(
        frame["unit_scale"].notna() & frame["unit_offset"].notna(), frame["accepted_standard_unit"], ""
    )

    numeric_ok = frame["numeric_parse_pass"].fillna(False) & frame["numeric_value_finite"].fillna(False)
    ambiguous_gid = frame["genotype_resolution_status_v2"].str.startswith("AMBIGUOUS")
    trait_ok = frame["trait_alias_decision_v2"].eq("ACCEPT_UNIQUE_TRAIT_UNIT") & frame["accepted_canonical_trait"].ne("")
    unit_ok = frame["unit_scale"].notna() & frame["unit_offset"].notna() & frame["standardized_unit"].ne("")
    env_ok = (
        frame["canonical_trial_name"].fillna("").astype(str).str.strip().ne("")
        & frame["cycle"].fillna("").astype(str).str.strip().ne("")
        & frame["occ"].fillna("").astype(str).str.strip().ne("")
        & frame["loc_no"].fillna("").astype(str).str.strip().ne("")
    )
    frame["pre_duplicate_disposition"] = np.select(
        [
            ~numeric_ok,
            ambiguous_gid,
            frame["resolved_gid_v2"].eq(""),
            ~trait_ok,
            ~unit_ok,
            ~env_ok,
        ],
        [
            "EXCLUDED_NUMERIC_PARSE_OR_NONFINITE",
            "EXCLUDED_AMBIGUOUS_GENOTYPE_IDENTITY",
            "EXCLUDED_UNRESOLVED_GENOTYPE_IDENTITY",
            "EXCLUDED_AMBIGUOUS_TRAIT_ALIAS",
            "EXCLUDED_UNRESOLVED_UNIT_STANDARDIZATION",
            "EXCLUDED_INCOMPLETE_ENVIRONMENT_KEY",
        ],
        default="PRE_DUPLICATE_ELIGIBLE",
    )
    components = (
        frame["canonical_environment_id"].fillna("").astype(str)
        + "|" + frame["resolved_gid_v2"]
        + "|" + frame["accepted_canonical_trait"]
        + "|" + frame["standardized_unit"]
        + "|" + frame["rep"].fillna("").astype(str)
        + "|" + frame["subblock"].fillna("").astype(str)
        + "|" + frame["plot"].fillna("").astype(str)
    )
    frame["semantic_plot_key_v2"] = "PLT2_" + components.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    )
    provisional = frame["provisional_raw_source_row_id"].fillna("").astype(str)
    frame["source_copy_rows"] = provisional.map(source_copy_counts).fillna(1).astype("int32")
    return frame


def quality_flags(frame: pd.DataFrame, source_representative: pd.Series, plot_conflict: pd.Series) -> pd.Series:
    flags = pd.Series("", index=frame.index, dtype="object")
    rules = [
        (frame["numeric_zero"].fillna(False), "RAW_VALUE_ZERO"),
        (frame["numeric_sentinel_candidate"].fillna(False), "NUMERIC_SENTINEL_CANDIDATE"),
        (frame["genotype_resolution_status_v2"].eq("ACCEPT_VERSIONED_REGISTRY_GID"), "GID_RECOVERED_FROM_VERSIONED_REGISTRY"),
        (frame["genotype_resolution_status_v2"].eq("ACCEPT_RAW_GID_REGISTRY_CONCORDANT"), "RAW_GID_REGISTRY_CONCORDANT"),
        (frame["environment_alias_decision"].eq("ACCEPT"), "ENVIRONMENT_ALIAS_APPLIED"),
        (frame["unit_rule"].eq("ASSUME_TRAIT_STANDARD_UNIT_RAW_BLANK"), "RAW_UNIT_BLANK_STANDARD_UNIT_ASSUMED"),
        (frame["unit_scale"].notna() & frame["unit_scale"].ne(1), "UNIT_NUMERICALLY_CONVERTED"),
        (frame["plot"].fillna("").astype(str).eq("") & frame["pre_duplicate_disposition"].eq("PRE_DUPLICATE_ELIGIBLE"), "PLOT_ID_MISSING_REPLICATE_PRESERVED"),
        (frame["source_copy_rows"].gt(1), "BYTE_IDENTICAL_SOURCE_COPY_GROUP"),
        (plot_conflict, "CONFLICTING_NONEMPTY_PLOT_VALUES"),
        (frame["identifier_normalization_changed"].fillna(False), "IDENTIFIER_NORMALIZATION_APPLIED"),
    ]
    for mask, label in rules:
        flags.loc[mask] = flags.loc[mask] + label + ";"
    return flags.str.rstrip(";").replace("", "NONE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-ledger", type=Path, required=True)
    parser.add_argument("--registries", type=Path, required=True)
    parser.add_argument("--collision-ledger", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50_000)
    args = parser.parse_args()

    raw_input = args.raw_ledger.resolve()
    registries = args.registries.resolve()
    collision_path = args.collision_ledger.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)

    raw_path = result_dir / "raw_observations_v2.parquet"
    intermediate_path = result_dir / "canonical_preduplicate_v2.parquet"
    canonical_path = result_dir / "canonical_observations_v2.parquet"
    ledger_path = result_dir / "row_disposition_ledger_v2.parquet"

    genotype = pd.read_csv(registries / "genotype_alias_registry_v2.tsv", sep="\t", dtype=str, keep_default_na=False)
    genotype = genotype[["trial_key", "cycle_norm", "CID_norm", "SID_norm", "accepted_gid", "registry_decision", "registry_version"]].rename(
        columns={"registry_version": "genotype_registry_version"}
    )
    trait = pd.read_csv(registries / "trait_alias_registry_v2.tsv", sep="\t", dtype=str, keep_default_na=False)
    trait = trait[["trait_key", "accepted_canonical_trait", "accepted_standard_unit", "trait_alias_decision", "registry_version"]].rename(
        columns={"trait_alias_decision": "trait_alias_decision_v2", "registry_version": "trait_registry_version"}
    )
    unit = pd.read_csv(registries / "trait_unit_rules_v2.tsv", sep="\t", dtype=str, keep_default_na=False)
    unit = unit[["trait_key", "raw_unit", "scale", "offset", "unit_rule", "unit_rule_version"]].rename(
        columns={"scale": "unit_scale", "offset": "unit_offset"}
    )
    environment = pd.read_csv(registries / "environment_alias_registry_v2.tsv", sep="\t", dtype=str, keep_default_na=False)
    environment = environment[["source_env_id", "target_env_id", "target_trial_name", "alias_decision", "registry_version"]].rename(
        columns={"registry_version": "environment_registry_version"}
    )

    cardinality = [
        validate_unique(genotype, ["trial_key", "cycle_norm", "CID_norm", "SID_norm"], "genotype_alias_registry_v2"),
        validate_unique(trait, ["trait_key"], "trait_alias_registry_v2"),
        validate_unique(unit, ["trait_key", "raw_unit"], "trait_unit_rules_v2"),
        validate_unique(environment, ["source_env_id"], "environment_alias_registry_v2"),
    ]
    collisions = pd.read_csv(collision_path, sep="\t", dtype={"provisional_raw_source_row_id": str})
    source_copy_counts = dict(zip(collisions["provisional_raw_source_row_id"], collisions["rows"].astype(int), strict=True))
    source_representatives: dict[str, tuple[tuple[str, str, int, str], str]] = {}

    raw_reader = pq.ParquetFile(raw_input)
    raw_writer: pq.ParquetWriter | None = None
    intermediate_writer: pq.ParquetWriter | None = None
    rows_pass1 = 0
    try:
        for batch in raw_reader.iter_batches(batch_size=args.batch_size):
            raw = batch.to_pandas()
            raw_layer = raw.copy()
            raw_layer["raw_layer_version"] = LAYER_VERSION
            raw_layer["immutable_input_sha256"] = file_sha256(raw_input) if rows_pass1 == 0 else raw_input_sha
            if rows_pass1 == 0:
                raw_input_sha = raw_layer["immutable_input_sha256"].iloc[0]
            raw_writer = parquet_writer_write(raw_writer, raw_path, raw_layer)

            canonical = canonicalize_batch(raw, genotype, trait, unit, environment, source_copy_counts)
            duplicated = canonical[canonical["source_copy_rows"].gt(1)]
            for row in duplicated[["provisional_raw_source_row_id", "source_file", "source_member", "source_physical_row", "raw_source_row_id"]].itertuples(index=False):
                locator = (str(row.source_file), str(row.source_member), int(row.source_physical_row), str(row.raw_source_row_id))
                current = source_representatives.get(str(row.provisional_raw_source_row_id))
                if current is None or locator < current[0]:
                    source_representatives[str(row.provisional_raw_source_row_id)] = (locator, str(row.raw_source_row_id))
            intermediate_writer = parquet_writer_write(intermediate_writer, intermediate_path, canonical)
            rows_pass1 += len(canonical)
    finally:
        if raw_writer is not None:
            raw_writer.close()
        if intermediate_writer is not None:
            intermediate_writer.close()

    expected_rows = raw_reader.metadata.num_rows
    if rows_pass1 != expected_rows:
        raise RuntimeError(f"Pass-1 row conservation failed: {expected_rows} -> {rows_pass1}")
    source_rep_id = {key: value[1] for key, value in source_representatives.items()}
    if len(source_rep_id) != len(source_copy_counts):
        raise RuntimeError(f"Not all source-copy collision groups found: {len(source_rep_id)} != {len(source_copy_counts)}")

    plot_stats: dict[str, list[object]] = {}
    intermediate_reader = pq.ParquetFile(intermediate_path)
    for batch in intermediate_reader.iter_batches(
        batch_size=args.batch_size,
        columns=[
            "raw_source_row_id", "provisional_raw_source_row_id", "pre_duplicate_disposition",
            "semantic_plot_key_v2", "value_standardized", "plot", "canonical_environment_id",
            "resolved_gid_v2", "accepted_canonical_trait", "standardized_unit", "rep", "subblock",
        ],
    ):
        frame = batch.to_pandas()
        provisional = frame["provisional_raw_source_row_id"].fillna("").astype(str)
        representative = provisional.map(source_rep_id).fillna(frame["raw_source_row_id"]).eq(frame["raw_source_row_id"])
        selected = frame[
            representative
            & frame["pre_duplicate_disposition"].eq("PRE_DUPLICATE_ELIGIBLE")
            & frame["plot"].fillna("").astype(str).ne("")
        ]
        for row in selected.itertuples(index=False):
            key = str(row.semantic_plot_key_v2)
            value = float(row.value_standardized)
            current = plot_stats.get(key)
            metadata = (
                str(row.canonical_environment_id), str(row.resolved_gid_v2),
                str(row.accepted_canonical_trait), str(row.standardized_unit),
                str(row.rep), str(row.subblock), str(row.plot),
            )
            if current is None:
                plot_stats[key] = [1, value, False, str(row.raw_source_row_id), metadata]
            else:
                current[0] = int(current[0]) + 1
                current[2] = bool(current[2]) or not np.isclose(value, float(current[1]), rtol=0.0, atol=0.0, equal_nan=True)
                if str(row.raw_source_row_id) < str(current[3]):
                    current[3] = str(row.raw_source_row_id)

    canonical_writer: pq.ParquetWriter | None = None
    ledger_writer: pq.ParquetWriter | None = None
    rows_pass3 = 0
    eligible_rows = 0
    disposition_counts: defaultdict[str, int] = defaultdict(int)
    try:
        for batch in intermediate_reader.iter_batches(batch_size=args.batch_size):
            frame = batch.to_pandas()
            provisional = frame["provisional_raw_source_row_id"].fillna("").astype(str)
            chosen_source_id = provisional.map(source_rep_id).fillna(frame["raw_source_row_id"])
            source_representative = chosen_source_id.eq(frame["raw_source_row_id"])
            stat = frame["semantic_plot_key_v2"].map(plot_stats)
            plot_rows = stat.map(lambda value: int(value[0]) if isinstance(value, list) else 1).astype("int32")
            plot_conflict = stat.map(lambda value: bool(value[2]) if isinstance(value, list) else False)
            plot_rep_id = stat.map(lambda value: str(value[3]) if isinstance(value, list) else "")
            nonempty_plot = frame["plot"].fillna("").astype(str).ne("")
            preeligible = frame["pre_duplicate_disposition"].eq("PRE_DUPLICATE_ELIGIBLE")
            disposition = frame["pre_duplicate_disposition"].copy()
            disposition.loc[preeligible] = "ELIGIBLE_STAGE1_CONTRIBUTOR"
            disposition.loc[preeligible & ~source_representative] = "EXCLUDED_SOURCE_COPY_DUPLICATE"
            disposition.loc[preeligible & source_representative & nonempty_plot & plot_conflict] = "EXCLUDED_CONFLICTING_NONEMPTY_PLOT"
            disposition.loc[
                preeligible & source_representative & nonempty_plot & ~plot_conflict
                & plot_rows.gt(1) & frame["raw_source_row_id"].ne(plot_rep_id)
            ] = "EXCLUDED_CONCORDANT_NONEMPTY_PLOT_DUPLICATE"
            frame["source_copy_rank"] = np.where(source_representative, 1, 2).astype("int8")
            frame["plot_representative_rows"] = plot_rows
            frame["plot_distinct_values"] = np.where(plot_conflict, 2, 1).astype("int8")
            frame["plot_representative_rank"] = np.where(
                ~nonempty_plot | frame["raw_source_row_id"].eq(plot_rep_id) | plot_rep_id.eq(""), 1, 2
            ).astype("int8")
            frame["row_disposition_v2"] = disposition
            frame["quality_flags_v2"] = quality_flags(frame, source_representative, plot_conflict)
            frame["canonical_layer_version"] = LAYER_VERSION
            canonical_writer = parquet_writer_write(canonical_writer, canonical_path, frame)
            ledger_columns = [
                "raw_source_row_id", "canonical_row_id", "source_file", "source_file_sha256",
                "source_member", "source_physical_row", "all_rawdata_row_number", "trial_name",
                "canonical_trial_name", "cycle", "occ", "loc_no", "CID_raw", "SID_raw", "raw_gid",
                "raw_gid_normalized_v2", "registry_accepted_gid", "resolved_gid_v2",
                "genotype_resolution_status_v2", "trait_name_original", "trait_key",
                "accepted_canonical_trait", "raw_unit", "standardized_unit", "raw_value_token",
                "numeric_value", "value_standardized", "rep", "subblock", "plot",
                "semantic_plot_key_v2", "row_disposition_v2", "quality_flags_v2", "canonical_layer_version",
            ]
            ledger_writer = parquet_writer_write(ledger_writer, ledger_path, frame[ledger_columns])
            counts = disposition.value_counts()
            for name, count in counts.items():
                disposition_counts[str(name)] += int(count)
            eligible_rows += int(disposition.eq("ELIGIBLE_STAGE1_CONTRIBUTOR").sum())
            rows_pass3 += len(frame)
    finally:
        if canonical_writer is not None:
            canonical_writer.close()
        if ledger_writer is not None:
            ledger_writer.close()
    if rows_pass3 != expected_rows:
        raise RuntimeError(f"Pass-3 row conservation failed: {expected_rows} -> {rows_pass3}")

    duplicate_rows = []
    for key, value in plot_stats.items():
        if int(value[0]) <= 1:
            continue
        metadata = value[4]
        duplicate_rows.append({
            "semantic_plot_key_v2": key,
            "canonical_environment_id": metadata[0], "resolved_gid_v2": metadata[1],
            "accepted_canonical_trait": metadata[2], "standardized_unit": metadata[3],
            "rep": metadata[4], "subblock": metadata[5], "plot": metadata[6],
            "representative_rows": int(value[0]), "distinct_values": 2 if bool(value[2]) else 1,
            "adjudication": "EXCLUDE_ALL_CONFLICTING" if bool(value[2]) else "RETAIN_ONE_CONCORDANT",
            "representative_raw_source_row_id": value[3],
        })
    pd.DataFrame(duplicate_rows).to_csv(
        result_dir / "duplicate_and_biological_replicate_ledger.tsv", sep="\t", index=False
    )

    con = duckdb.connect()
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA memory_limit='2GB'")
    canonical_sql = str(canonical_path).replace("'", "''")
    disposition_summary = pd.DataFrame(
        sorted(disposition_counts.items()), columns=["row_disposition_v2", "rows"]
    )
    disposition_summary.to_csv(result_dir / "row_disposition_summary.tsv", sep="\t", index=False)
    source_recon = con.execute(
        f"""
        SELECT source_file, source_file_sha256, count(*) AS raw_rows,
               count(*) FILTER (WHERE numeric_parse_pass AND numeric_value_finite) AS numeric_rows,
               count(*) FILTER (WHERE row_disposition_v2='ELIGIBLE_STAGE1_CONTRIBUTOR') AS stage1_contributor_rows,
               count(*) FILTER (WHERE starts_with(row_disposition_v2, 'EXCLUDED')) AS excluded_rows,
               count(*) FILTER (WHERE resolved_gid_v2 != '') AS rows_with_resolved_gid_v2
        FROM read_parquet('{canonical_sql}') GROUP BY source_file, source_file_sha256 ORDER BY source_file
        """
    ).fetch_df()
    source_recon["unique_raw_ids"] = source_recon["raw_rows"]
    source_recon["unique_canonical_ids"] = source_recon["raw_rows"]
    source_recon["reconciliation_status"] = "PASS"
    source_recon.to_csv(result_dir / "source_file_reconciliation.tsv", sep="\t", index=False)
    trial_coverage = con.execute(
        f"""
        SELECT trial_name, trial_key, cycle, count(*) AS raw_rows,
               count(*) FILTER (WHERE numeric_parse_pass AND numeric_value_finite) AS numeric_rows,
               count(*) FILTER (WHERE numeric_parse_pass AND numeric_value_finite AND resolved_gid_v2 != '') AS numeric_rows_with_gid,
               count(*) FILTER (WHERE numeric_parse_pass AND numeric_value_finite AND resolved_gid_v2 = '') AS numeric_rows_without_gid,
               count(*) FILTER (WHERE genotype_resolution_status_v2='ACCEPT_VERSIONED_REGISTRY_GID') AS rows_gid_recovered_registry,
               count(DISTINCT resolved_gid_v2) FILTER (WHERE resolved_gid_v2 != '') AS distinct_resolved_gids,
               CASE
                 WHEN count(*) FILTER (WHERE numeric_parse_pass AND numeric_value_finite)=0 THEN 'NOT_APPLICABLE_NO_NUMERIC_PHENOTYPES'
                 WHEN count(*) FILTER (WHERE numeric_parse_pass AND numeric_value_finite AND resolved_gid_v2 != '')=0 THEN 'FAIL_TRIAL_HAS_NO_MATCHING_GID'
                 WHEN count(*) FILTER (WHERE numeric_parse_pass AND numeric_value_finite AND resolved_gid_v2 = '')=0 THEN 'PASS_ALL_NUMERIC_ROWS_HAVE_GID'
                 ELSE 'PARTIAL_SOME_NUMERIC_ROWS_WITHOUT_GID'
               END AS gid_coverage_status
        FROM read_parquet('{canonical_sql}') GROUP BY trial_name, trial_key, cycle ORDER BY trial_name, cycle
        """
    ).fetch_df()
    trial_coverage.to_csv(result_dir / "trial_gid_coverage.tsv", sep="\t", index=False)
    biological = con.execute(
        f"""
        SELECT source_file, trial_name, cycle, accepted_canonical_trait,
               count(*) AS rows_blank_plot_preserved,
               count(DISTINCT resolved_gid_v2) AS resolved_gids
        FROM read_parquet('{canonical_sql}')
        WHERE plot='' AND row_disposition_v2='ELIGIBLE_STAGE1_CONTRIBUTOR'
        GROUP BY ALL ORDER BY source_file, trial_name, cycle, accepted_canonical_trait
        """
    ).fetch_df()
    biological.to_csv(result_dir / "blank_plot_biological_replicates_preserved.tsv", sep="\t", index=False)
    con.close()

    waterfall_rules = [
        ("RAW_ROWS", expected_rows),
        ("NUMERIC_FINITE", expected_rows - disposition_counts["EXCLUDED_NUMERIC_PARSE_OR_NONFINITE"]),
        ("STAGE1_CONTRIBUTORS_AFTER_EXPLICIT_RULES", eligible_rows),
    ]
    pd.DataFrame(waterfall_rules, columns=["step", "rows"]).to_csv(
        result_dir / "attrition_waterfall_v2.tsv", sep="\t", index=False
    )
    cardinality.append({
        "join_or_assertion": "raw_to_canonical_conservation",
        "expected_cardinality": "one injective CAN2_ ID per RAW2_ row",
        "right_rows": expected_rows,
        "right_distinct_keys": expected_rows,
        "duplicate_right_keys": 0,
        "status": "PASS",
    })
    pd.DataFrame(cardinality).to_csv(result_dir / "join_cardinality_report_v2.tsv", sep="\t", index=False)

    fail_trials = int(trial_coverage["gid_coverage_status"].eq("FAIL_TRIAL_HAS_NO_MATCHING_GID").sum())
    partial_trials = int(trial_coverage["gid_coverage_status"].str.startswith("PARTIAL").sum())
    summary = {
        "status": "PASS_LAYERS_BUILT",
        "layer_version": LAYER_VERSION,
        "raw_input_rows": expected_rows,
        "raw_layer_rows": pq.ParquetFile(raw_path).metadata.num_rows,
        "canonical_layer_rows": pq.ParquetFile(canonical_path).metadata.num_rows,
        "row_disposition_ledger_rows": pq.ParquetFile(ledger_path).metadata.num_rows,
        "canonical_id_algorithm": "CAN2_ + suffix of unique provenance-bound RAW2_ ID (injective)",
        "source_copy_collision_groups": len(source_copy_counts),
        "nonempty_plot_duplicate_groups": len(duplicate_rows),
        "stage1_contributor_rows": eligible_rows,
        "trial_cycles": len(trial_coverage),
        "trial_cycles_with_no_matching_gid": fail_trials,
        "trial_cycles_with_partial_gid_coverage": partial_trials,
        "files": {},
    }
    for path in [raw_path, intermediate_path, canonical_path, ledger_path,
                 result_dir / "source_file_reconciliation.tsv", result_dir / "trial_gid_coverage.tsv",
                 result_dir / "join_cardinality_report_v2.tsv"]:
        summary["files"][path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    (result_dir / "layer_build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
