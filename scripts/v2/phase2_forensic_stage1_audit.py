"""Forensic, non-mutating reconstruction of the legacy raw-to-Stage-1 lineage.

This diagnostic writes only to a new versioned audit directory.  It does not fit
Stage-1 models, rebuild production outputs, train models, or read protected
evaluation content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SELECTED_TRAITS = {
    "1000_GRAIN_WEIGHT",
    "ABOVE_GROUND_BIOMASS",
    "DAYS_TO_HEADING",
    "DAYS_TO_MATURITY",
    "GRAIN_YIELD",
    "PLANT_HEIGHT",
    "TEST_WEIGHT",
}

RAW_USED_COLUMNS = [
    "source_file", "trial_dir", "Trial_name", "Occ", "Loc_no", "Country",
    "Loc_desc", "Cycle", "Cid", "Sid", "Gen_name", "Trait_name", "Rep",
    "Sub_block", "Plot", "Value", "Unit", "GID",
]

TEXTUAL_MISSING_CODES = {
    "", ".", "-", "--", "?", "NA", "N/A", "NAN", "NULL", "NONE",
    "N", "NN", "#N/A", "#N/A N/A", "#NA", "<NA>",
}
NUMERIC_SENTINEL_CANDIDATES = {
    -99999.0, -9999.0, -999.0, -99.0, -9.0, 999.0, 9999.0, 99999.0,
}
PANDAS_DEFAULT_NA_TOKENS = {
    "", "-1.#IND", "1.#QNAN", "1.#IND", "-1.#QNAN", "#N/A N/A",
    "#N/A", "N/A", "n/a", "NA", "<NA>", "#NA", "NULL", "null",
    "NaN", "-NaN", "nan", "-nan", "None",
}
PROTECTED_TOKENS = (
    "final_holdout", "final_nested_evaluation",
    "reaction_norm_routed_hierarchy_outer_v1/reporting",
    "reporting_only_diagnostics", "outer_metrics", "outer_predictions",
    "trained_models",
)


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def norm_text(series: pd.Series, *, upper: bool = False) -> pd.Series:
    output = clean_text(series).str.replace(r"\s+", " ", regex=True)
    return output.str.upper() if upper else output


def norm_key(series: pd.Series) -> pd.Series:
    return norm_text(series, upper=True)


def legacy_clean_text(series: pd.Series) -> pd.Series:
    """Emulate read_csv's default NA conversion while retaining the raw token elsewhere."""
    output = clean_text(series)
    return output.mask(output.isin(PANDAS_DEFAULT_NA_TOKENS), "")


def legacy_norm_key(series: pd.Series) -> pd.Series:
    return legacy_clean_text(series).str.upper().str.replace(r"\s+", " ", regex=True)


def legacy_cycle_year(series: pd.Series) -> pd.Series:
    cleaned = legacy_clean_text(series)
    return cleaned.str.extract(r"(\d{4})", expand=False).fillna(cleaned)


def legacy_strip_dot_zero(series: pd.Series) -> pd.Series:
    return legacy_clean_text(series).str.replace(r"\.0$", "", regex=True)


def cycle_year(series: pd.Series) -> pd.Series:
    cleaned = clean_text(series)
    return cleaned.str.extract(r"(\d{4})", expand=False).fillna(cleaned)


def strip_dot_zero(series: pd.Series) -> pd.Series:
    return clean_text(series).str.replace(r"\.0$", "", regex=True)


def canonical_gid(series: pd.Series) -> pd.Series:
    value = norm_text(series, upper=True).str.replace(r"\.0$", "", regex=True)
    core = value.str.replace(r"^GID", "", regex=True)
    return pd.Series(np.where(core.eq(""), "", "GID" + core), index=series.index)


def stable_stage1_id(frame: pd.DataFrame) -> pd.Series:
    cols = [
        "phenotype_source", "env_id_pheno", "resolved_gid",
        "trait_name_canonical", "trait_name_original", "unit",
    ]
    values = frame[cols].fillna("").astype(str).itertuples(index=False, name=None)
    return pd.Series(
        ["STG1_" + hashlib.sha1("|".join(row).encode("utf-8")).hexdigest()[:16] for row in values],
        index=frame.index,
        dtype="string",
    )


def digest_id(prefix: str, values: Iterable[tuple[object, ...]], length: int = 24) -> list[str]:
    return [
        prefix + hashlib.sha256("|".join("" if v is None else str(v) for v in row).encode("utf-8")).hexdigest()[:length]
        for row in values
    ]


def natural_key(frame: pd.DataFrame) -> pd.Series:
    gid = canonical_gid(frame["canonical_germplasm_key"])
    env = norm_text(frame["env_kernel_id"])
    trait = norm_text(frame["trait_name_canonical"], upper=True)
    original = norm_text(frame["trait_name_original"], upper=True)
    unit = norm_text(frame["unit"], upper=True)
    return gid + "\x1f" + env + "\x1f" + trait + "\x1f" + original + "\x1f" + unit


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def fail_if_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def write_tsv(path: Path, rows: pd.DataFrame) -> None:
    fail_if_exists(path)
    rows.to_csv(path, sep="\t", index=False, lineterminator="\n")


def write_json(path: Path, value: object) -> None:
    fail_if_exists(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


class BatchParquetWriter:
    def __init__(self, path: Path) -> None:
        fail_if_exists(path)
        self.path = path
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(
                self.path, self.schema, compression="zstd", use_dictionary=True
            )
        elif table.schema != self.schema:
            table = table.cast(self.schema)
        assert self.writer is not None
        self.writer.write_table(table)
        self.rows += len(frame)

    def close(self) -> None:
        if self.writer is None:
            raise RuntimeError(f"No rows were written to {self.path}")
        self.writer.close()


def verify_protocol_inputs(root: Path, out_dir: Path) -> pd.DataFrame:
    protocol_path = out_dir / "phase2_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for item in protocol["input_bindings"]:
        relative = item["path"]
        lowered = relative.lower().replace("\\", "/")
        if any(token in lowered for token in PROTECTED_TOKENS):
            raise RuntimeError(f"Protected path is not allowed as Phase-2 input: {relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        digest = sha256_file(path)
        status = "PASS" if size == item["bytes"] and digest == item["sha256"] else "FAIL"
        rows.append({
            "relative_path": relative,
            "expected_bytes": item["bytes"],
            "observed_bytes": size,
            "expected_sha256": item["sha256"],
            "observed_sha256": digest,
            "status": status,
        })
    result = pd.DataFrame(rows)
    write_tsv(out_dir / "input_manifest_verification.tsv", result)
    if not result["status"].eq("PASS").all():
        raise RuntimeError("One or more Phase-2 protocol input bindings failed")
    return result


def build_resolver_audit(path: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "CID", "SID", "trial_name", "cycle", "occ", "resolved_gid",
        "panel_sample_id_expected", "gid_resolution_status", "gid_source",
    ]
    manifest = pd.read_csv(path, sep="\t", dtype=str, usecols=columns, low_memory=False)
    manifest["trial_key"] = norm_key(manifest["trial_name"])
    manifest["cycle_key"] = cycle_year(manifest["cycle"])
    manifest["occ_key"] = clean_text(manifest["occ"])
    manifest["CID_key"] = strip_dot_zero(manifest["CID"])
    manifest["SID_key"] = strip_dot_zero(manifest["SID"])
    key_cols = ["trial_key", "cycle_key", "occ_key", "CID_key", "SID_key"]
    manifest["resolver_key"] = manifest[key_cols].fillna("").astype(str).agg("\x1f".join, axis=1)
    manifest["resolved_gid_norm"] = strip_dot_zero(manifest["resolved_gid"])
    manifest["panel_norm"] = clean_text(manifest["panel_sample_id_expected"])

    audit = (
        manifest.groupby("resolver_key", sort=True, dropna=False)
        .agg(
            manifest_rows=("resolver_key", "size"),
            trial_key=("trial_key", "first"),
            cycle=("cycle_key", "first"),
            occ=("occ_key", "first"),
            CID=("CID_key", "first"),
            SID=("SID_key", "first"),
            distinct_resolved_gids=("resolved_gid_norm", lambda x: int(clean_text(x).replace("", np.nan).nunique())),
            resolved_gid_candidates=("resolved_gid_norm", lambda x: ";".join(sorted(set(clean_text(x)) - {""}))),
            distinct_panel_ids=("panel_norm", lambda x: int(clean_text(x).replace("", np.nan).nunique())),
            panel_id_candidates=("panel_norm", lambda x: ";".join(sorted(set(clean_text(x)) - {""}))),
            resolution_statuses=("gid_resolution_status", lambda x: ";".join(sorted(set(clean_text(x)) - {""}))),
            gid_sources=("gid_source", lambda x: ";".join(sorted(set(clean_text(x)) - {""}))),
        )
        .reset_index()
    )
    audit["resolver_key_status"] = np.select(
        [
            audit["distinct_resolved_gids"].gt(1) & audit["distinct_panel_ids"].gt(1),
            audit["distinct_resolved_gids"].gt(1),
            audit["distinct_panel_ids"].gt(1),
            audit["manifest_rows"].gt(1),
        ],
        [
            "DUPLICATE_CONFLICTING_GID_AND_PANEL",
            "DUPLICATE_CONFLICTING_GID",
            "DUPLICATE_CONFLICTING_PANEL",
            "DUPLICATE_CONCORDANT",
        ],
        default="UNIQUE",
    )
    write_tsv(out_dir / "genotype_resolver_key_audit.tsv", audit)

    chosen = manifest.drop_duplicates(key_cols, keep="first").copy()
    chosen = chosen.merge(
        audit[[
            "resolver_key", "manifest_rows", "distinct_resolved_gids",
            "distinct_panel_ids", "resolver_key_status",
        ]],
        on="resolver_key",
        how="left",
        validate="m:1",
    )
    chosen = chosen.rename(columns={
        "resolved_gid_norm": "manifest_resolved_gid",
        "panel_norm": "manifest_panel_sample_id",
    })
    return chosen[[
        "resolver_key", "manifest_resolved_gid", "manifest_panel_sample_id",
        "manifest_rows", "distinct_resolved_gids", "distinct_panel_ids",
        "resolver_key_status",
    ]], audit


def build_trait_audit(path: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=["trait_name_original", "trait_name_canonical", "unit"],
        low_memory=False,
    )
    source = source.dropna(subset=["trait_name_original"]).drop_duplicates()
    source["trait_key"] = norm_key(source["trait_name_original"])
    source["canonical_key"] = norm_key(source["trait_name_canonical"])
    source["unit_norm"] = norm_text(source["unit"], upper=True)
    source["_has_unit"] = clean_text(source["unit"]).ne("")
    source = source.sort_values(["trait_key", "_has_unit"], ascending=[True, False])
    chosen = source.drop_duplicates("trait_key", keep="first").copy()

    audit = (
        source.groupby("trait_key", sort=True, dropna=False)
        .agg(
            distinct_source_rows=("trait_key", "size"),
            original_labels=("trait_name_original", lambda x: ";".join(sorted(set(clean_text(x)) - {""}))),
            canonical_label_count=("canonical_key", lambda x: int(clean_text(x).replace("", np.nan).nunique())),
            canonical_labels=("canonical_key", lambda x: ";".join(sorted(set(clean_text(x)) - {""}))),
            unit_count=("unit_norm", lambda x: int(clean_text(x).replace("", np.nan).nunique())),
            units=("unit_norm", lambda x: ";".join(sorted(set(clean_text(x)) - {""}))),
        )
        .reset_index()
    )
    audit = audit.merge(
        chosen[["trait_key", "trait_name_canonical", "unit"]].rename(columns={
            "trait_name_canonical": "chosen_trait_name_canonical",
            "unit": "chosen_unit",
        }),
        on="trait_key",
        how="left",
        validate="1:1",
    )
    audit["trait_mapping_status"] = np.select(
        [
            audit["canonical_label_count"].gt(1) & audit["unit_count"].gt(1),
            audit["canonical_label_count"].gt(1),
            audit["unit_count"].gt(1),
        ],
        ["AMBIGUOUS_CANONICAL_AND_UNIT", "AMBIGUOUS_CANONICAL", "AMBIGUOUS_UNIT"],
        default="UNIQUE_OR_CONCORDANT",
    )
    write_tsv(out_dir / "trait_mapping_key_audit.tsv", audit)
    chosen = chosen.merge(
        audit[["trait_key", "canonical_label_count", "unit_count", "trait_mapping_status"]],
        on="trait_key",
        how="left",
        validate="1:1",
    )
    del source
    return chosen[[
        "trait_key", "trait_name_canonical", "canonical_key", "unit",
        "canonical_label_count", "unit_count", "trait_mapping_status",
    ]], audit


def first_member(path: Path) -> tuple[str, str]:
    with path.open("rb") as handle:
        signature = handle.read(4)
    if signature[:2] == b"PK" or signature == b"\xd0\xcf\x11\xe0":
        try:
            book = pd.ExcelFile(path)
            return book.sheet_names[0], "excel_first_sheet"
        except Exception as exc:  # retained as an auditable state
            return "<unresolved_first_sheet>", f"excel_metadata_error:{type(exc).__name__}"
    return "<tabular_text>", "tabular_text"


def build_raw_source_registry(root: Path, phase1_manifest: Path) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    manifest = pd.read_csv(phase1_manifest, sep="\t", dtype=str, low_memory=False)
    candidates = manifest[
        manifest["relative_path"].map(lambda value: "rawdata" in Path(str(value)).name.lower())
    ].copy()
    rows: list[dict[str, object]] = []
    lookup: dict[str, dict[str, object]] = {}
    for record in candidates.itertuples(index=False):
        relative = str(record.relative_path).replace("\\", "/")
        raw_path = root / "TRIALS_AND_NURSERIES_DATA" / relative
        top_path = root / relative
        member, parser = first_member(raw_path)
        top_exists = top_path.is_file()
        top_hash = sha256_file(top_path) if top_exists else ""
        phase1_hash = str(record.sha256).lower()
        row = {
            "relative_path": relative,
            "source_file_sha256": phase1_hash,
            "source_bytes": int(record.bytes),
            "legacy_first_member": member,
            "legacy_parser_route": parser,
            "top_level_duplicate_exists": top_exists,
            "top_level_duplicate_sha256": top_hash,
            "top_level_matches_authoritative_raw": bool(top_exists and top_hash == phase1_hash),
            "rows_in_all_rawdata": 0,
            "source_file_status": "NOT_SEEN_IN_ALL_RAWDATA",
        }
        rows.append(row)
        lookup[relative.lower()] = row
    return pd.DataFrame(rows), lookup


def update_group_parts(
    parts: list[pd.DataFrame], frame: pd.DataFrame, scope: str,
    dimensions: dict[str, str], disposition_col: str,
) -> None:
    for dimension, column in dimensions.items():
        grouped = (
            frame.groupby([column, disposition_col], dropna=False, sort=False)
            .size()
            .rename("rows")
            .reset_index()
            .rename(columns={column: "dimension_value", disposition_col: "disposition"})
        )
        grouped.insert(0, "dimension", dimension)
        grouped.insert(0, "ledger_scope", scope)
        parts.append(grouped)


def collapse_group_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(parts, ignore_index=True)
    combined["dimension_value"] = clean_text(combined["dimension_value"])
    return (
        combined.groupby(
            ["ledger_scope", "dimension", "dimension_value", "disposition"],
            dropna=False,
            sort=True,
        )["rows"]
        .sum()
        .reset_index()
        .sort_values(["ledger_scope", "dimension", "dimension_value", "disposition"])
        .reset_index(drop=True)
    )


def boolean_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return norm_text(series, upper=True).isin({"1", "TRUE", "YES", "Y", "PASS"})


def classify_marker(frame: pd.DataFrame) -> pd.Series:
    marker_cols = [
        "has_hmp_qc_genotype", "has_hmp_raw_genotype",
        "has_dartseq_landrace_sample", "has_mas_sample",
        "has_80k_marker_priors", "has_80k_existing_marker_context",
        "has_dartseq_80k_weighted_kernel",
    ]
    available = pd.Series(False, index=frame.index)
    for column in marker_cols:
        if column in frame:
            available |= boolean_values(frame[column])
    return available


def load_order_ids(path: Path) -> set[str]:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    for column in ("sample_id", "genotype_id", "panel_sample_id", "canonical_gid"):
        if column in frame:
            return set(canonical_gid(frame[column])) - {""}
    raise ValueError(f"No genotype identifier column in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    if not out_dir.is_dir():
        raise FileNotFoundError(out_dir)
    if args.chunksize < 10_000:
        raise ValueError("chunksize is unreasonably small")

    paths = {
        "raw": root / "phenotypes/all_rawdata.tsv",
        "model_input": root / "phenotypes/model_input_phenotypes.tsv",
        "manifest": root / "server_phase1_bundle/artifacts/metadata_outputs/all_trials_genotype_manifest_resolved.tsv",
        "canonical": root / "server_phase1_bundle/artifacts/integrated_database/canonical_trial_genotype_environment_plot_table.parquet",
        "stage1": root / "server_phase1_bundle/artifacts/phenotypes/stage1_adjusted_phenotypes.parquet",
        "baseline_attrition": root / "server_phase1_bundle/artifacts/audit/information_attrition_v2/stage1_to_model_attrition_ledger.parquet",
        "alias_registry": root / "server_phase1_bundle/artifacts/audit/stage1_environment_alias_recovery_v1/environment_alias_registry.tsv",
        "weight_registry": root / "server_phase1_bundle/artifacts/audit/stage1_weight_recovery_v1/stage1_weight_recovery_registry.tsv",
        "alias_only": root / "server_phase1_bundle/artifacts/model_kernels/stage1_canonical_v3_environment_alias_v1/stage1_canonical_v3_environment_alias_v1_model_ready_stage1_observations.parquet",
        "alias_weight": root / "server_phase1_bundle/artifacts/model_kernels/stage1_canonical_v3_environment_alias_weight_v1/stage1_canonical_v3_environment_alias_weight_v1_model_ready_stage1_observations.parquet",
        "pedigree_order": root / "server_phase1_bundle/artifacts/genotype_panels/pedigree_canonical_v3/K_A_CANONICAL_V3_sample_order.tsv",
        "phase1_trial_manifest": root / "audit/v2/phase1_project_inventory_reproducibility_v1/trial_file_inventory_before.tsv",
    }

    verify_protocol_inputs(root, out_dir)
    resolver, resolver_audit = build_resolver_audit(paths["manifest"], out_dir)
    trait_map, trait_audit = build_trait_audit(paths["model_input"], out_dir)
    source_registry, source_lookup = build_raw_source_registry(root, paths["phase1_trial_manifest"])

    stage1_columns = [
        "canonical_observation_id", "canonical_germplasm_key", "resolved_gid",
        "env_id_pheno", "env_kernel_id", "trial_name", "cycle", "occ", "loc_no",
        "country", "loc_desc", "trait_name_canonical", "trait_name_original", "unit",
        "y_tilde_g_e", "var_g_e", "weight_g_e", "n_plot_records",
        "stage1_model_status", "phenotype_adjustment_status",
    ]
    stage1 = pd.read_parquet(paths["stage1"], columns=stage1_columns)
    if len(stage1) != 433_626:
        raise RuntimeError(f"Unexpected Stage-1 row count: {len(stage1)}")
    if stage1["canonical_observation_id"].isna().any() or stage1["canonical_observation_id"].duplicated().any():
        raise RuntimeError("Stage-1 observation IDs are blank or duplicated")
    stage1_ids = set(clean_text(stage1["canonical_observation_id"]))
    stage1["analysis_natural_key"] = natural_key(stage1)
    stage1_key_duplicates = stage1[stage1["analysis_natural_key"].duplicated(keep=False)].copy()
    if not stage1_key_duplicates.empty:
        write_tsv(
            out_dir / "stage1_natural_key_duplicates.tsv",
            stage1_key_duplicates.sort_values("analysis_natural_key"),
        )
    stage1_key_map = stage1.drop_duplicates("analysis_natural_key", keep="first").set_index("analysis_natural_key")[
        "canonical_observation_id"
    ].to_dict()
    expected_stage1_plot_counts = pd.to_numeric(stage1.set_index("canonical_observation_id")["n_plot_records"], errors="coerce").fillna(0).astype(int)

    resolver_index = resolver.set_index("resolver_key")
    trait_index = trait_map.set_index("trait_key")
    raw_header = pd.read_csv(paths["raw"], sep="\t", nrows=0).columns.tolist()
    missing_raw_cols = sorted(set(RAW_USED_COLUMNS) - set(raw_header))
    if missing_raw_cols:
        raise ValueError(f"Legacy raw concatenation lacks columns: {missing_raw_cols}")
    extra_columns = [column for column in raw_header if column not in RAW_USED_COLUMNS]

    raw_writer = BatchParquetWriter(out_dir / "raw_row_disposition_ledger.parquet")
    source_offsets: dict[str, int] = defaultdict(int)
    source_counts: Counter[str] = Counter()
    source_trait_nonempty: Counter[str] = Counter()
    source_value_nonempty: Counter[str] = Counter()
    contribution_counts: Counter[str] = Counter()
    plot_key_counts: Counter[str] = Counter()
    exact_key_counts: Counter[str] = Counter()
    plot_key_first: dict[str, dict[str, str]] = {}
    plot_key_first_value: dict[str, str] = {}
    plot_key_value_conflict: set[str] = set()
    dimension_parts: list[pd.DataFrame] = []
    numeric_failure_parts: list[pd.DataFrame] = []
    zero_parts: list[pd.DataFrame] = []
    sentinel_parts: list[pd.DataFrame] = []
    unit_conflict_parts: list[pd.DataFrame] = []
    unresolved_gid_parts: list[pd.DataFrame] = []
    incomplete_environment_parts: list[pd.DataFrame] = []
    identifier_type_parts: list[pd.DataFrame] = []
    wide_stats: dict[tuple[str, str], list[object]] = {}
    total_rows = 0
    numeric_rows = 0
    numeric_failures = 0
    unresolved_gid_rows = 0
    eligible_rows = 0
    eligible_missing_stage1 = 0
    resolver_conflicting_raw_rows = 0
    trait_ambiguous_raw_rows = 0
    unit_conflict_rows = 0
    zero_rows = 0
    numeric_sentinel_rows = 0
    nonfinite_numeric_rows = 0
    alternate_gid_ignored_rows = 0
    incomplete_environment_rows = 0
    manifest_matched_numeric_rows = 0
    trait_mapped_numeric_rows = 0

    raw_reader = pd.read_csv(
        paths["raw"], sep="\t", dtype=str, usecols=raw_header,
        chunksize=args.chunksize, low_memory=False, keep_default_na=False,
        na_filter=False,
    )
    for chunk_no, chunk in enumerate(raw_reader, start=1):
        chunk = chunk.reset_index(drop=True)
        n = len(chunk)
        all_rawdata_row_number = np.arange(total_rows + 2, total_rows + n + 2, dtype=np.int64)
        total_rows += n
        source_file = clean_text(chunk["source_file"]).str.replace("\\", "/", regex=False)
        source_norm = source_file.str.lower()
        source_ordinal = pd.Series(np.zeros(n, dtype=np.int64), index=chunk.index)
        for source, indices in source_file.groupby(source_file, sort=False).groups.items():
            start = source_offsets[source]
            values = np.arange(start + 1, start + len(indices) + 1, dtype=np.int64)
            source_ordinal.loc[indices] = values
            source_offsets[source] += len(indices)
            source_counts[source] += len(indices)
        source_row = source_ordinal + 1
        source_hash = source_norm.map(lambda value: str(source_lookup.get(value, {}).get("source_file_sha256", "")))
        source_member = source_norm.map(lambda value: str(source_lookup.get(value, {}).get("legacy_first_member", "<unmapped>")))
        source_map_status = np.where(source_hash.ne(""), "MATCHED_PHASE1_RAW_MANIFEST", "UNMAPPED_SOURCE_FILE")
        raw_source_id = digest_id(
            "RAW2_", zip(source_hash, source_member, source_row, strict=True)
        )

        raw_value = clean_text(chunk["Value"])
        numeric = pd.to_numeric(raw_value, errors="coerce")
        numeric_ok = numeric.notna()
        finite_numeric = pd.Series(np.isfinite(numeric.to_numpy(dtype=float)), index=chunk.index) & numeric_ok
        numeric_nonfinite = numeric_ok & ~finite_numeric
        zero = numeric_ok & numeric.eq(0)
        numeric_sentinel = numeric_ok & numeric.isin(NUMERIC_SENTINEL_CANDIDATES)
        token_norm = norm_text(raw_value, upper=True)
        value_token_class = np.select(
            [
                zero,
                numeric_sentinel,
                numeric_nonfinite,
                numeric_ok,
                token_norm.isin(TEXTUAL_MISSING_CODES),
            ],
            [
                "NUMERIC_ZERO_RETAINED",
                "NUMERIC_SENTINEL_CANDIDATE_RETAINED",
                "NUMERIC_NONFINITE_RETAINED",
                "NUMERIC_FINITE_RETAINED",
                "TEXTUAL_OR_BLANK_MISSING_DROPPED",
            ],
            default="NONNUMERIC_UNRECOGNIZED_DROPPED",
        )

        trial_name = legacy_clean_text(chunk["Trial_name"])
        trial_key = legacy_norm_key(chunk["Trial_name"])
        cycle_raw = legacy_clean_text(chunk["Cycle"])
        cycle = legacy_cycle_year(chunk["Cycle"])
        occ = legacy_clean_text(chunk["Occ"])
        cid_raw = legacy_clean_text(chunk["Cid"])
        sid_raw = legacy_clean_text(chunk["Sid"])
        cid = legacy_strip_dot_zero(chunk["Cid"])
        sid = legacy_strip_dot_zero(chunk["Sid"])
        resolver_key = (
            trial_key + "\x1f" + cycle + "\x1f" + occ + "\x1f" + cid + "\x1f" + sid
        )
        manifest_gid = resolver_key.map(resolver_index["manifest_resolved_gid"])
        manifest_panel = resolver_key.map(resolver_index["manifest_panel_sample_id"])
        resolver_status = resolver_key.map(resolver_index["resolver_key_status"]).fillna("NO_MANIFEST_MATCH")
        resolver_rows = resolver_key.map(resolver_index["manifest_rows"]).fillna(0).astype(np.int64)
        raw_gid = legacy_strip_dot_zero(chunk["GID"])
        resolved_gid = manifest_gid.fillna("").where(clean_text(manifest_gid).ne(""), raw_gid)
        resolved_gid = clean_text(resolved_gid)
        panel_id = manifest_panel.fillna("").where(
            clean_text(manifest_panel).ne(""),
            pd.Series(np.where(resolved_gid.eq(""), "", "GID" + resolved_gid.str.replace(r"^GID", "", regex=True)), index=chunk.index),
        )
        resolver_conflict = resolver_status.str.startswith("DUPLICATE_CONFLICTING")
        genotype_id_class = np.select(
            [
                ~numeric_ok,
                resolver_conflict & clean_text(manifest_gid).ne(""),
                clean_text(manifest_gid).ne(""),
                raw_gid.ne(""),
            ],
            [
                "NOT_EVALUATED_AFTER_NUMERIC_FILTER",
                "MANIFEST_AMBIGUOUS_KEEP_FIRST",
                "MANIFEST_RESOLVED",
                "RAW_GID_FALLBACK",
            ],
            default="UNRESOLVED",
        )

        trait_original = legacy_clean_text(chunk["Trait_name"])
        trait_key = legacy_norm_key(chunk["Trait_name"])
        mapped_canonical = trait_key.map(trait_index["trait_name_canonical"])
        mapped_unit = trait_key.map(trait_index["unit"])
        trait_status = trait_key.map(trait_index["trait_mapping_status"]).fillna("UNMAPPED_FALLBACK_TO_NORMALIZED_RAW")
        trait_canonical = clean_text(mapped_canonical).replace("", np.nan).fillna(trait_key)
        unit_raw = legacy_clean_text(chunk["Unit"])
        unit = clean_text(mapped_unit).replace("", np.nan).fillna(unit_raw)
        unit_conflict = (
            norm_text(unit_raw, upper=True).ne("")
            & norm_text(mapped_unit, upper=True).ne("")
            & norm_text(unit_raw, upper=True).ne(norm_text(mapped_unit, upper=True))
        )
        trait_ambiguous = trait_status.str.startswith("AMBIGUOUS")

        loc_no = legacy_clean_text(chunk["Loc_no"])
        country = legacy_clean_text(chunk["Country"])
        loc_desc = legacy_clean_text(chunk["Loc_desc"])
        env_id_pheno = trial_name + "|" + cycle + "|" + occ + "|" + loc_no
        env_kernel_id = trial_name + "|" + occ + "|" + loc_no + "|" + country + "|" + loc_desc + "|" + cycle
        missing_trial = trial_name.eq("")
        missing_cycle = cycle.eq("")
        missing_occ = occ.eq("")
        missing_location = loc_no.eq("")
        incomplete_env = missing_trial | missing_cycle | missing_occ | missing_location

        eligible = numeric_ok & resolved_gid.ne("")
        expected_id_frame = pd.DataFrame({
            "phenotype_source": "RawData_stage1",
            "env_id_pheno": env_id_pheno,
            "resolved_gid": resolved_gid,
            "trait_name_canonical": trait_canonical,
            "trait_name_original": trait_original,
            "unit": unit,
        })
        expected_id = stable_stage1_id(expected_id_frame)
        expected_id = expected_id.where(eligible, "")
        stage1_available = expected_id.isin(stage1_ids) & eligible
        disposition = np.select(
            [
                ~numeric_ok,
                numeric_ok & resolved_gid.eq(""),
                eligible & stage1_available,
            ],
            [
                "EXCLUDED_NUMERIC_PARSE_FAILURE",
                "EXCLUDED_UNRESOLVED_GENOTYPE",
                "RETAINED_CONTRIBUTES_TO_STAGE1",
            ],
            default="DEFECT_ELIGIBLE_STAGE1_OUTPUT_MISSING",
        )

        rep = legacy_clean_text(chunk["Rep"])
        subblock = legacy_clean_text(chunk["Sub_block"])
        plot = legacy_clean_text(chunk["Plot"])
        raw_plot_key = digest_id(
            "RPK2_",
            zip(
                trial_name, cycle, occ, loc_no, cid, sid, resolved_gid,
                trait_original, rep, subblock, plot, unit,
                strict=True,
            ),
            length=20,
        )
        raw_exact_key = digest_id(
            "REX2_",
            zip(raw_plot_key, raw_value, strict=True),
            length=20,
        )

        eligible_indices = np.flatnonzero(eligible.to_numpy())
        for pos in eligible_indices:
            sid_value = str(expected_id.iloc[pos])
            contribution_counts[sid_value] += 1
            plot_key = raw_plot_key[pos]
            exact_key = raw_exact_key[pos]
            plot_key_counts[plot_key] += 1
            exact_key_counts[exact_key] += 1
            value = str(raw_value.iloc[pos])
            if plot_key not in plot_key_first:
                plot_key_first[plot_key] = {
                    "raw_plot_key": plot_key,
                    "source_file": str(source_file.iloc[pos]),
                    "trial_name": str(trial_name.iloc[pos]),
                    "cycle": str(cycle.iloc[pos]),
                    "occ": str(occ.iloc[pos]),
                    "env_kernel_id": str(env_kernel_id.iloc[pos]),
                    "resolved_gid": str(resolved_gid.iloc[pos]),
                    "trait_name_original": str(trait_original.iloc[pos]),
                    "trait_name_canonical": str(trait_canonical.iloc[pos]),
                    "unit": str(unit.iloc[pos]),
                    "rep": str(rep.iloc[pos]),
                    "subblock": str(subblock.iloc[pos]),
                    "plot": str(plot.iloc[pos]),
                }
                plot_key_first_value[plot_key] = value
            elif plot_key_first_value[plot_key] != value:
                plot_key_value_conflict.add(plot_key)

        for source, group in chunk.assign(_source=source_file).groupby("_source", sort=False):
            source_trait_nonempty[source] += int(clean_text(group["Trait_name"]).ne("").sum())
            source_value_nonempty[source] += int(clean_text(group["Value"]).ne("").sum())

        for column in extra_columns:
            values = clean_text(chunk[column])
            nonempty = values.ne("")
            if not nonempty.any():
                continue
            numeric_extra = pd.to_numeric(values, errors="coerce").notna()
            temp = pd.DataFrame({"source_file": source_file, "value": values, "nonempty": nonempty, "numeric": numeric_extra})
            for source, group in temp[nonempty].groupby("source_file", sort=False):
                key = (source, column)
                if key not in wide_stats:
                    wide_stats[key] = [0, 0, set()]
                wide_stats[key][0] += int(group["nonempty"].sum())
                wide_stats[key][1] += int(group["numeric"].sum())
                samples: set[str] = wide_stats[key][2]  # type: ignore[assignment]
                for value in group["value"].head(10):
                    if len(samples) < 5:
                        samples.add(str(value))

        failed = pd.DataFrame({
            "source_file": source_file,
            "trial_name": trial_name,
            "cycle": cycle,
            "occ": occ,
            "trait_name_original": trait_original,
            "raw_value_token": raw_value,
            "value_token_class": value_token_class,
        })[~numeric_ok]
        if not failed.empty:
            numeric_failure_parts.append(
                failed.groupby(list(failed.columns), dropna=False, sort=False).size().rename("rows").reset_index()
            )
        zero_frame = pd.DataFrame({
            "source_file": source_file, "trial_name": trial_name, "cycle": cycle,
            "occ": occ, "trait_name_original": trait_original,
            "trait_name_canonical": trait_canonical, "unit": unit,
        })[zero]
        if not zero_frame.empty:
            zero_parts.append(zero_frame.groupby(list(zero_frame.columns), dropna=False, sort=False).size().rename("rows").reset_index())
        sentinel_frame = pd.DataFrame({
            "source_file": source_file, "trial_name": trial_name, "cycle": cycle,
            "occ": occ, "trait_name_original": trait_original,
            "trait_name_canonical": trait_canonical, "unit": unit,
            "numeric_value": numeric,
        })[numeric_sentinel]
        if not sentinel_frame.empty:
            sentinel_parts.append(sentinel_frame.groupby(list(sentinel_frame.columns), dropna=False, sort=False).size().rename("rows").reset_index())
        conflict_frame = pd.DataFrame({
            "source_file": source_file, "trait_key": trait_key,
            "trait_name_original": trait_original,
            "trait_name_canonical": trait_canonical,
            "raw_unit": unit_raw, "chosen_unit": unit,
        })[unit_conflict]
        if not conflict_frame.empty:
            unit_conflict_parts.append(conflict_frame.groupby(list(conflict_frame.columns), dropna=False, sort=False).size().rename("rows").reset_index())

        alternate_gid = clean_text(chunk["GID.1"]) if "GID.1" in chunk else pd.Series("", index=chunk.index)
        ignored_alt_gid = numeric_ok & resolved_gid.eq("") & alternate_gid.ne("")

        unresolved_frame = pd.DataFrame({
            "source_file": source_file,
            "trial_name": trial_name,
            "cycle": cycle,
            "occ": occ,
            "CID": cid,
            "SID": sid,
            "genotype_name": legacy_clean_text(chunk["Gen_name"]),
            "raw_gid": raw_gid,
            "alternate_gid_ignored": alternate_gid,
        })[numeric_ok & resolved_gid.eq("")]
        if not unresolved_frame.empty:
            unresolved_gid_parts.append(
                unresolved_frame.groupby(list(unresolved_frame.columns), dropna=False, sort=False)
                .size().rename("rows").reset_index()
            )

        incomplete_frame = pd.DataFrame({
            "trial_name": trial_name,
            "cycle": cycle,
            "occ": occ,
            "loc_no": loc_no,
            "country": country,
            "loc_desc": loc_desc,
            "env_kernel_id": env_kernel_id,
            "missing_trial": missing_trial,
            "missing_cycle": missing_cycle,
            "missing_occurrence": missing_occ,
            "missing_location": missing_location,
        })[eligible & incomplete_env]
        if not incomplete_frame.empty:
            incomplete_environment_parts.append(
                incomplete_frame.groupby(list(incomplete_frame.columns), dropna=False, sort=False)
                .size().rename("rows").reset_index()
            )

        identifier_flags = {
            "TRIAL_CASE_OR_WHITESPACE_NORMALIZED": trial_name.ne(trial_key),
            "CYCLE_YEAR_EXTRACTION_CHANGED": cycle_raw.ne(cycle),
            "CID_DOT_ZERO_STRIPPED": cid_raw.str.endswith(".0") & cid.ne(cid_raw),
            "SID_DOT_ZERO_STRIPPED": sid_raw.str.endswith(".0") & sid.ne(sid_raw),
            "CID_LEADING_ZERO": cid.str.match(r"^0\d+", na=False),
            "SID_LEADING_ZERO": sid.str.match(r"^0\d+", na=False),
            "RAW_GID_ALREADY_PREFIXED": raw_gid.str.upper().str.startswith("GID"),
        }
        for mismatch_class, mismatch_mask in identifier_flags.items():
            selected_mask = numeric_ok & mismatch_mask
            if not selected_mask.any():
                continue
            mismatch = pd.DataFrame({
                "mismatch_class": mismatch_class,
                "source_file": source_file,
                "trial_name": trial_name,
            })[selected_mask]
            identifier_type_parts.append(
                mismatch.groupby(list(mismatch.columns), dropna=False, sort=False)
                .size().rename("rows").reset_index()
            )

        ledger = pd.DataFrame({
            "raw_source_row_id": pd.Series(raw_source_id, dtype="string"),
            "all_rawdata_row_number": all_rawdata_row_number,
            "source_file": source_file,
            "source_file_sha256": source_hash,
            "source_member": source_member,
            "source_physical_row": source_row.astype(np.int64),
            "source_manifest_status": source_map_status,
            "trial_dir": clean_text(chunk["trial_dir"]),
            "trial_name": trial_name,
            "trial_key": trial_key,
            "cycle_raw": cycle_raw,
            "cycle": cycle,
            "occ": occ,
            "loc_no": loc_no,
            "country": country,
            "loc_desc": loc_desc,
            "env_id_pheno": env_id_pheno,
            "env_kernel_id": env_kernel_id,
            "missing_trial": missing_trial,
            "missing_cycle": missing_cycle,
            "missing_occurrence": missing_occ,
            "missing_location": missing_location,
            "CID_raw": cid_raw,
            "SID_raw": sid_raw,
            "CID_normalized": cid,
            "SID_normalized": sid,
            "raw_gid": raw_gid,
            "alternate_gid_ignored": alternate_gid,
            "resolver_key": resolver_key,
            "resolver_manifest_rows": resolver_rows,
            "resolver_key_status": resolver_status,
            "resolved_gid": resolved_gid,
            "panel_sample_id": panel_id,
            "genotype_name": legacy_clean_text(chunk["Gen_name"]),
            "genotype_id_class": genotype_id_class,
            "identifier_normalization_changed": trial_name.ne(trial_key) | cycle_raw.ne(cycle) | cid_raw.ne(cid) | sid_raw.ne(sid),
            "trait_name_original": trait_original,
            "trait_key": trait_key,
            "trait_name_canonical": trait_canonical,
            "trait_mapping_status": trait_status,
            "raw_unit": unit_raw,
            "unit": unit,
            "unit_conflict_with_selected_mapping": unit_conflict,
            "raw_value_token": raw_value,
            "numeric_value": numeric.astype(float),
            "value_token_class": value_token_class,
            "numeric_parse_pass": numeric_ok,
            "numeric_value_finite": finite_numeric,
            "numeric_zero": zero,
            "numeric_sentinel_candidate": numeric_sentinel,
            "rep": rep,
            "subblock": subblock,
            "plot": plot,
            "raw_plot_key": raw_plot_key,
            "raw_exact_record_key": raw_exact_key,
            "expected_stage1_observation_id": expected_id,
            "stage1_output_available": stage1_available,
            "final_raw_disposition": disposition,
        })
        raw_writer.write(ledger)
        update_group_parts(
            dimension_parts,
            ledger,
            "raw_input",
            {
                "source_file": "source_file", "trial": "trial_name",
                "cycle": "cycle", "occurrence": "occ",
                "trait": "trait_name_canonical", "environment": "env_kernel_id",
                "genotype_id_class": "genotype_id_class",
            },
            "final_raw_disposition",
        )

        numeric_count = int(numeric_ok.sum())
        numeric_rows += numeric_count
        numeric_failures += n - numeric_count
        unresolved_gid_rows += int((numeric_ok & resolved_gid.eq("")).sum())
        eligible_count = int(eligible.sum())
        eligible_rows += eligible_count
        eligible_missing_stage1 += int((eligible & ~stage1_available).sum())
        resolver_conflicting_raw_rows += int((numeric_ok & resolver_conflict).sum())
        trait_ambiguous_raw_rows += int((numeric_ok & trait_ambiguous).sum())
        manifest_matched_numeric_rows += int((numeric_ok & clean_text(manifest_gid).ne("")).sum())
        trait_mapped_numeric_rows += int((numeric_ok & clean_text(mapped_canonical).ne("")).sum())
        unit_conflict_rows += int((numeric_ok & unit_conflict).sum())
        zero_rows += int(zero.sum())
        numeric_sentinel_rows += int(numeric_sentinel.sum())
        nonfinite_numeric_rows += int(numeric_nonfinite.sum())
        alternate_gid_ignored_rows += int(ignored_alt_gid.sum())
        incomplete_environment_rows += int((eligible & incomplete_env).sum())
        if chunk_no % 5 == 0:
            print(
                f"raw chunks={chunk_no:,} rows={total_rows:,} numeric={numeric_rows:,} eligible={eligible_rows:,}",
                flush=True,
            )

    raw_writer.close()

    contribution = pd.DataFrame({
        "canonical_observation_id": list(stage1_ids),
    })
    contribution["expected_n_plot_records"] = contribution["canonical_observation_id"].map(expected_stage1_plot_counts).fillna(0).astype(int)
    contribution["reconstructed_n_plot_records"] = contribution["canonical_observation_id"].map(contribution_counts).fillna(0).astype(int)
    contribution["difference"] = contribution["reconstructed_n_plot_records"] - contribution["expected_n_plot_records"]
    contribution["status"] = np.where(contribution["difference"].eq(0), "PASS", "FAIL")
    write_tsv(out_dir / "raw_to_stage1_contribution_check.tsv", contribution.sort_values("canonical_observation_id"))

    raw_gate = {
        "raw_rows": total_rows,
        "numeric_rows": numeric_rows,
        "numeric_parse_failures": numeric_failures,
        "unresolved_gid_rows_after_numeric_filter": unresolved_gid_rows,
        "eligible_stage1_input_rows": eligible_rows,
        "eligible_rows_missing_stage1_output": eligible_missing_stage1,
        "distinct_reconstructed_stage1_ids": len(contribution_counts),
        "server_stage1_ids": len(stage1_ids),
        "stage1_plot_count_mismatches": int(contribution["status"].eq("FAIL").sum()),
        "required_eligible_rows": 581397,
        "required_stage1_ids": 433626,
    }
    raw_gate["status"] = "PASS" if (
        eligible_rows == 581_397
        and eligible_missing_stage1 == 0
        and len(contribution_counts) == 433_626
        and raw_gate["stage1_plot_count_mismatches"] == 0
    ) else "SERVER_DATA_REQUIRED"
    write_json(out_dir / "raw_to_server_stage1_reconciliation.json", raw_gate)
    if raw_gate["status"] != "PASS":
        write_json(out_dir / "server_data_required.json", {
            "status": "STOP_AND_REQUEST_DATA",
            "required_data": "Exact server phenotypes/all_rawdata.tsv used to produce the supplied Stage-1 artifact",
            "reason": raw_gate,
        })
        raise SystemExit(3)

    source_registry = source_registry.copy()
    source_registry["rows_in_all_rawdata"] = source_registry["relative_path"].map(source_counts).fillna(0).astype(int)
    source_registry["trait_nonempty_rows"] = source_registry["relative_path"].map(source_trait_nonempty).fillna(0).astype(int)
    source_registry["value_nonempty_rows"] = source_registry["relative_path"].map(source_value_nonempty).fillna(0).astype(int)
    source_registry["source_file_status"] = np.select(
        [
            source_registry["rows_in_all_rawdata"].eq(0),
            ~source_registry["top_level_matches_authoritative_raw"],
            source_registry["trait_nonempty_rows"].eq(0),
            source_registry["value_nonempty_rows"].eq(0),
        ],
        [
            "SILENTLY_ABSENT_FROM_ALL_RAWDATA",
            "TOP_LEVEL_SOURCE_DIFFERS_FROM_AUTHORITATIVE_RAW",
            "NO_TRAIT_VALUES_IN_CONCATENATION",
            "NO_VALUE_TOKENS_IN_CONCATENATION",
        ],
        default="PRESENT_HASH_MATCHED_AND_LONG_FORM_FIELDS_PRESENT",
    )
    write_tsv(out_dir / "raw_source_registry.tsv", source_registry)

    duplicate_rows = []
    for key, count in plot_key_counts.items():
        if count <= 1:
            continue
        row = dict(plot_key_first[key])
        row.update({
            "records_with_same_plot_key": count,
            "conflicting_value_tokens": key in plot_key_value_conflict,
        })
        duplicate_rows.append(row)
    duplicate_frame = pd.DataFrame(duplicate_rows)
    if duplicate_frame.empty:
        duplicate_frame = pd.DataFrame(columns=[
            "raw_plot_key", "source_file", "trial_name", "cycle", "occ",
            "env_kernel_id", "resolved_gid", "trait_name_original",
            "trait_name_canonical", "unit", "rep", "subblock", "plot",
            "records_with_same_plot_key", "conflicting_value_tokens",
        ])
    write_tsv(out_dir / "raw_plot_duplicate_groups.tsv", duplicate_frame.sort_values(["records_with_same_plot_key", "raw_plot_key"], ascending=[False, True]))

    wide_rows = []
    recognized_auxiliary = re.compile(r"^(GID|ENTRY|PLOT)(\.\d+)?$|^UNNAMED_?\d*$", re.I)
    for (source, column), (nonempty_count, numeric_count, samples) in wide_stats.items():
        classification = (
            "AUXILIARY_OR_DUPLICATED_IDENTIFIER_COLUMN"
            if recognized_auxiliary.match(column.replace(" ", "_"))
            else "POSSIBLE_WIDE_VALUE_COLUMN_REQUIRES_REVIEW"
        )
        wide_rows.append({
            "source_file": source,
            "column_name": column,
            "nonempty_rows": nonempty_count,
            "numeric_rows": numeric_count,
            "sample_values": ";".join(sorted(samples)),
            "classification": classification,
            "legacy_stage1_action": "IGNORED_COLUMN",
        })
    wide_frame = pd.DataFrame(wide_rows)
    if wide_frame.empty:
        wide_frame = pd.DataFrame(columns=["source_file", "column_name", "nonempty_rows", "numeric_rows", "sample_values", "classification", "legacy_stage1_action"])
    write_tsv(out_dir / "wide_to_long_omission_candidates.tsv", wide_frame.sort_values(["classification", "numeric_rows", "source_file"], ascending=[True, False, True]))

    def collapse_optional(parts: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame(columns=columns + ["rows"])
        combined = pd.concat(parts, ignore_index=True)
        return combined.groupby(columns, dropna=False, sort=True)["rows"].sum().reset_index()

    numeric_failures_frame = collapse_optional(
        numeric_failure_parts,
        ["source_file", "trial_name", "cycle", "occ", "trait_name_original", "raw_value_token", "value_token_class"],
    )
    zero_frame = collapse_optional(
        zero_parts,
        ["source_file", "trial_name", "cycle", "occ", "trait_name_original", "trait_name_canonical", "unit"],
    )
    sentinel_frame = collapse_optional(
        sentinel_parts,
        ["source_file", "trial_name", "cycle", "occ", "trait_name_original", "trait_name_canonical", "unit", "numeric_value"],
    )
    unit_frame = collapse_optional(
        unit_conflict_parts,
        ["source_file", "trait_key", "trait_name_original", "trait_name_canonical", "raw_unit", "chosen_unit"],
    )
    write_tsv(out_dir / "numeric_parsing_failures.tsv", numeric_failures_frame)
    write_tsv(out_dir / "zero_value_audit.tsv", zero_frame)
    write_tsv(out_dir / "numeric_missing_sentinel_candidates.tsv", sentinel_frame)
    write_tsv(out_dir / "unit_conflicts.tsv", unit_frame)
    unresolved_gid_frame = collapse_optional(
        unresolved_gid_parts,
        ["source_file", "trial_name", "cycle", "occ", "CID", "SID", "genotype_name", "raw_gid", "alternate_gid_ignored"],
    )
    incomplete_environment_frame = collapse_optional(
        incomplete_environment_parts,
        ["trial_name", "cycle", "occ", "loc_no", "country", "loc_desc", "env_kernel_id", "missing_trial", "missing_cycle", "missing_occurrence", "missing_location"],
    )
    identifier_type_frame = collapse_optional(
        identifier_type_parts,
        ["mismatch_class", "source_file", "trial_name"],
    )
    write_tsv(out_dir / "unresolved_genotype_alias_candidates.tsv", unresolved_gid_frame)
    write_tsv(out_dir / "incomplete_environment_keys.tsv", incomplete_environment_frame)
    write_tsv(out_dir / "identifier_type_mismatch_audit.tsv", identifier_type_frame)

    # Canonical row disposition reconstruction in bounded batches.
    baseline = pd.read_parquet(paths["baseline_attrition"], columns=[
        "canonical_observation_id", "stage1_to_model_status",
    ])
    baseline_status = baseline.set_index("canonical_observation_id")["stage1_to_model_status"].to_dict()
    alias_only = pd.read_parquet(paths["alias_only"], columns=[
        "canonical_observation_id", "environment_alias_applied",
        "environment_alias_registry_source_id", "env_kernel_id_original", "env_kernel_id",
    ])
    alias_only_ids = set(clean_text(alias_only["canonical_observation_id"]))
    alias_only_applied = alias_only.set_index("canonical_observation_id")["environment_alias_applied"].to_dict()
    alias_source_map = alias_only.set_index("canonical_observation_id")["environment_alias_registry_source_id"].to_dict()
    alias_weight = pd.read_parquet(paths["alias_weight"], columns=["canonical_observation_id"])
    alias_weight_ids = set(clean_text(alias_weight["canonical_observation_id"]))
    weights = pd.read_csv(paths["weight_registry"], sep="\t", dtype=str)
    weight_ids = set(clean_text(weights["canonical_observation_id"]))
    pedigree_ids = load_order_ids(paths["pedigree_order"])

    canonical_file = pq.ParquetFile(paths["canonical"])
    canonical_schema_names = set(canonical_file.schema.names)
    canonical_columns = [
        "canonical_observation_id", "canonical_germplasm_key", "germplasm_id",
        "resolved_gid", "env_id_pheno", "env_kernel_id", "trial_name", "cycle",
        "occ", "loc_no", "country", "loc_desc", "trait_name_canonical",
        "trait_name_original", "unit", "phenotype_source", "phenotype_value",
        "raw_numeric_records", "raw_plot_records", "n_records", "n_source_files",
        "duplicate_resolution", "plot_support_status", "source_level",
        "gid_resolution_status", "has_environment_kernel", "has_hmp_qc_genotype",
        "has_hmp_raw_genotype", "has_dartseq_landrace_sample", "has_mas_sample",
        "has_80k_marker_priors", "has_80k_existing_marker_context",
        "has_dartseq_80k_weighted_kernel",
    ]
    canonical_columns = [column for column in canonical_columns if column in canonical_schema_names]
    ids_table = pq.read_table(paths["canonical"], columns=["canonical_observation_id"])
    canonical_distinct_ids = pc.count_distinct(ids_table["canonical_observation_id"]).as_py()
    canonical_blank_ids = pc.sum(pc.equal(ids_table["canonical_observation_id"], "")).as_py() or 0
    del ids_table
    if canonical_distinct_ids != canonical_file.metadata.num_rows or canonical_blank_ids:
        raise RuntimeError("Canonical observation IDs are not permanent unique nonempty row IDs")

    canonical_writer = BatchParquetWriter(out_dir / "canonical_row_disposition_ledger.parquet")
    canonical_key_counts: Counter[str] = Counter()
    marker_gid_set: set[str] = set()
    canonical_rows = 0
    canonical_stage1_matches = 0
    canonical_selected_rows = 0
    canonical_selected_stage1_matches = 0
    canonical_disposition_counts: Counter[str] = Counter()
    for batch_no, batch in enumerate(canonical_file.iter_batches(batch_size=args.chunksize, columns=canonical_columns), start=1):
        frame = batch.to_pandas()
        n = len(frame)
        canonical_rows += n
        frame["canonical_row_id"] = clean_text(frame["canonical_observation_id"])
        frame["canonical_germplasm_key_norm"] = canonical_gid(frame["canonical_germplasm_key"])
        frame["analysis_natural_key"] = natural_key(frame)
        stage1_id = frame["analysis_natural_key"].map(stage1_key_map).fillna("")
        frame["stage1_observation_id"] = stage1_id
        frame["stage1_key_available"] = stage1_id.ne("")
        frame["selected_trait"] = norm_text(frame["trait_name_canonical"], upper=True).isin(SELECTED_TRAITS)
        frame["finite_canonical_target"] = pd.Series(
            np.isfinite(pd.to_numeric(frame["phenotype_value"], errors="coerce").to_numpy(dtype=float)),
            index=frame.index,
        )
        frame["gid_resolved"] = frame["canonical_germplasm_key_norm"].ne("")
        frame["canonical_pedigree_available"] = frame["canonical_germplasm_key_norm"].isin(pedigree_ids)
        frame["any_marker_available"] = classify_marker(frame)
        marker_gid_set.update(frame.loc[frame["any_marker_available"], "canonical_germplasm_key_norm"])
        frame["genotype_modality_class"] = np.select(
            [
                ~frame["gid_resolved"],
                frame["canonical_pedigree_available"] & frame["any_marker_available"],
                frame["canonical_pedigree_available"] & ~frame["any_marker_available"],
                ~frame["canonical_pedigree_available"] & frame["any_marker_available"],
            ],
            ["UNRESOLVED", "PEDIGREE_AND_MARKER", "PEDIGREE_ONLY", "MARKER_ONLY"],
            default="NO_PEDIGREE_OR_MARKER",
        )
        frame["baseline_stage1_status"] = stage1_id.map(baseline_status).fillna("NOT_IN_SELECTED_STAGE1_ATTRITION_LEDGER")
        frame["alias_only_model_available"] = stage1_id.isin(alias_only_ids)
        frame["alias_weight_model_available"] = stage1_id.isin(alias_weight_ids)
        frame["environment_alias_applied"] = stage1_id.map(alias_only_applied).fillna(False).astype(bool)
        frame["environment_alias_registry_source_id"] = stage1_id.map(alias_source_map).fillna("")
        frame["fold_local_weight_recovery_required"] = stage1_id.isin(weight_ids)
        raw_numeric = pd.to_numeric(frame["raw_numeric_records"], errors="coerce").fillna(0)
        source_level = norm_text(frame["source_level"], upper=True)
        final_disposition = np.select(
            [
                ~frame["finite_canonical_target"],
                ~frame["gid_resolved"],
                ~frame["stage1_key_available"] & source_level.eq("SUMMARY_LEVEL"),
                ~frame["stage1_key_available"] & raw_numeric.le(0),
                ~frame["stage1_key_available"],
                frame["stage1_key_available"] & ~frame["selected_trait"],
                frame["baseline_stage1_status"].eq("retained_in_stage1_model_observations"),
                frame["baseline_stage1_status"].eq("invalid_or_nonpositive_stage1_weight"),
                frame["baseline_stage1_status"].eq("genotype_not_in_stage1_model_order"),
                frame["baseline_stage1_status"].eq("environment_not_in_stage1_model_order"),
                frame["stage1_key_available"] & frame["selected_trait"] & frame["alias_weight_model_available"],
            ],
            [
                "EXCLUDED_NONFINITE_CANONICAL_TARGET",
                "EXCLUDED_UNRESOLVED_CANONICAL_GENOTYPE",
                "NOT_RECONSTRUCTED_SUMMARY_LEVEL_PARALLEL_BRANCH",
                "NOT_RECONSTRUCTED_NO_NUMERIC_RAW_STAGE1_INPUT",
                "DEFECT_RAW_LINKED_NOT_RECONSTRUCTED_BY_STAGE1",
                "STAGE1_RETAINED_TRAIT_OUTSIDE_SELECTED_SEVEN",
                "SELECTED_RETAINED_IN_BASELINE_MODEL_INPUT",
                "SELECTED_EXCLUDED_BASELINE_INVALID_WEIGHT_RECOVERED_FOLD_LOCAL",
                "SELECTED_EXCLUDED_BASELINE_GENOTYPE_ORDER",
                "SELECTED_EXCLUDED_BASELINE_ENVIRONMENT_ORDER",
                "SELECTED_RECOVERED_ONLY_IN_ALIAS_WEIGHT_INPUT",
            ],
            default="UNRESOLVED_CANONICAL_DISPOSITION",
        )
        frame["final_canonical_disposition"] = final_disposition
        canonical_key_counts.update(frame["analysis_natural_key"])
        canonical_disposition_counts.update(final_disposition)
        canonical_stage1_matches += int(frame["stage1_key_available"].sum())
        canonical_selected_rows += int(frame["selected_trait"].sum())
        canonical_selected_stage1_matches += int((frame["selected_trait"] & frame["stage1_key_available"]).sum())

        output_cols = [
            "canonical_row_id", "canonical_observation_id", "canonical_germplasm_key",
            "resolved_gid", "germplasm_id", "canonical_germplasm_key_norm",
            "genotype_modality_class", "canonical_pedigree_available",
            "any_marker_available", "env_id_pheno", "env_kernel_id", "trial_name",
            "cycle", "occ", "loc_no", "country", "loc_desc",
            "trait_name_canonical", "trait_name_original", "unit",
            "phenotype_source", "finite_canonical_target", "raw_numeric_records",
            "raw_plot_records", "n_records", "n_source_files",
            "duplicate_resolution", "plot_support_status", "source_level",
            "gid_resolution_status", "analysis_natural_key",
            "stage1_observation_id", "stage1_key_available", "selected_trait",
            "baseline_stage1_status", "alias_only_model_available",
            "alias_weight_model_available", "environment_alias_applied",
            "environment_alias_registry_source_id",
            "fold_local_weight_recovery_required", "final_canonical_disposition",
        ]
        output_cols = [column for column in output_cols if column in frame]
        canonical_writer.write(frame[output_cols])
        update_group_parts(
            dimension_parts,
            frame,
            "canonical",
            {
                "source_class": "phenotype_source", "trial": "trial_name",
                "cycle": "cycle", "occurrence": "occ",
                "trait": "trait_name_canonical", "environment": "env_kernel_id",
                "genotype_id_class": "genotype_modality_class",
            },
            "final_canonical_disposition",
        )
        if batch_no % 5 == 0:
            print(f"canonical batches={batch_no:,} rows={canonical_rows:,}", flush=True)
    canonical_writer.close()

    duplicated_canonical_keys = pd.DataFrame([
        {"analysis_natural_key": key, "canonical_rows": count}
        for key, count in canonical_key_counts.items() if count > 1
    ])
    if duplicated_canonical_keys.empty:
        duplicated_canonical_keys = pd.DataFrame(columns=["analysis_natural_key", "canonical_rows"])
    write_tsv(out_dir / "canonical_natural_key_duplicates.tsv", duplicated_canonical_keys.sort_values(["canonical_rows", "analysis_natural_key"], ascending=[False, True]))

    attrition_dimensions = collapse_group_parts(dimension_parts)
    write_tsv(out_dir / "attrition_by_dimension.tsv", attrition_dimensions)

    selected_stage1 = stage1[norm_text(stage1["trait_name_canonical"], upper=True).isin(SELECTED_TRAITS)].copy()
    selected_stage1["baseline_status"] = clean_text(selected_stage1["canonical_observation_id"]).map(baseline_status).fillna("MISSING_BASELINE_STATUS")
    selected_stage1["canonical_pedigree_available"] = canonical_gid(selected_stage1["canonical_germplasm_key"]).isin(pedigree_ids)
    selected_stage1["any_marker_available"] = canonical_gid(selected_stage1["canonical_germplasm_key"]).isin(marker_gid_set)
    selected_stage1["modality_class"] = np.select(
        [
            selected_stage1["canonical_pedigree_available"] & selected_stage1["any_marker_available"],
            selected_stage1["canonical_pedigree_available"] & ~selected_stage1["any_marker_available"],
            ~selected_stage1["canonical_pedigree_available"] & selected_stage1["any_marker_available"],
        ],
        ["PEDIGREE_AND_MARKER", "PEDIGREE_ONLY", "MARKER_ONLY"],
        default="NO_PEDIGREE_OR_MARKER",
    )
    modality_summary = (
        selected_stage1.groupby(["modality_class", "baseline_status"], dropna=False, sort=True)
        .agg(
            stage1_rows=("canonical_observation_id", "size"),
            unique_genotypes=("canonical_germplasm_key", "nunique"),
            unique_environments=("env_kernel_id", "nunique"),
            unique_traits=("trait_name_canonical", "nunique"),
            represented_raw_plot_records=("n_plot_records", "sum"),
        )
        .reset_index()
    )
    write_tsv(out_dir / "stage1_genotype_modality_attrition.tsv", modality_summary)

    alias_registry = pd.read_csv(paths["alias_registry"], sep="\t", dtype=str)
    alias_collision = alias_registry[alias_registry["match_class"].eq("trial_alias_resolved_nontrial_collision")].copy()
    write_tsv(out_dir / "environment_alias_collision_review.tsv", alias_collision)

    baseline_counts = Counter(clean_text(baseline["stage1_to_model_status"]))
    raw_exact_duplicate_excess = sum(count - 1 for count in exact_key_counts.values() if count > 1)
    raw_plot_duplicate_excess = sum(count - 1 for count in plot_key_counts.values() if count > 1)
    wide_candidate_rows = int(
        wide_frame.loc[wide_frame["classification"].eq("POSSIBLE_WIDE_VALUE_COLUMN_REQUIRES_REVIEW"), "numeric_rows"].sum()
    ) if not wide_frame.empty else 0

    join_rows = [
        {
            "join_name": "raw_source_file_to_phase1_manifest",
            "left_rows": len(source_counts), "right_rows": len(source_registry),
            "left_unique_keys": len(source_counts), "right_unique_keys": source_registry["relative_path"].nunique(),
            "expected_cardinality": "m:1", "matched_left_rows": int(sum(value > 0 for value in source_registry["rows_in_all_rawdata"])),
            "unmatched_left_rows": int(sum(1 for key in source_counts if key.lower() not in source_lookup)),
            "right_duplicate_keys": int(source_registry["relative_path"].duplicated().sum()),
            "output_rows": len(source_counts), "explosion_rows": 0,
            "status": "PASS" if all(key.lower() in source_lookup for key in source_counts) else "FAIL",
        },
        {
            "join_name": "numeric_raw_to_manifest_resolver_after_keep_first",
            "left_rows": numeric_rows, "right_rows": len(resolver),
            "left_unique_keys": "not_materialized", "right_unique_keys": resolver["resolver_key"].nunique(),
            "expected_cardinality": "m:1", "matched_left_rows": manifest_matched_numeric_rows,
            "unmatched_left_rows": numeric_rows - manifest_matched_numeric_rows, "right_duplicate_keys": int(resolver["resolver_key"].duplicated().sum()),
            "output_rows": numeric_rows, "explosion_rows": 0,
            "status": "DEFECT_HIDDEN_AMBIGUITY" if resolver_conflicting_raw_rows else "PASS",
        },
        {
            "join_name": "numeric_raw_to_trait_map_after_keep_first",
            "left_rows": numeric_rows, "right_rows": len(trait_map),
            "left_unique_keys": "not_materialized", "right_unique_keys": trait_map["trait_key"].nunique(),
            "expected_cardinality": "m:1", "matched_left_rows": trait_mapped_numeric_rows,
            "unmatched_left_rows": numeric_rows - trait_mapped_numeric_rows, "right_duplicate_keys": int(trait_map["trait_key"].duplicated().sum()),
            "output_rows": numeric_rows, "explosion_rows": 0,
            "status": "DEFECT_HIDDEN_AMBIGUITY" if trait_ambiguous_raw_rows else "PASS_WITH_FALLBACK",
        },
        {
            "join_name": "eligible_raw_contributions_to_stage1_output",
            "left_rows": eligible_rows, "right_rows": len(stage1),
            "left_unique_keys": len(contribution_counts), "right_unique_keys": len(stage1_ids),
            "expected_cardinality": "m:1 contribution", "matched_left_rows": eligible_rows - eligible_missing_stage1,
            "unmatched_left_rows": eligible_missing_stage1, "right_duplicate_keys": 0,
            "output_rows": eligible_rows - eligible_missing_stage1, "explosion_rows": 0,
            "status": "PASS" if eligible_missing_stage1 == 0 else "FAIL",
        },
        {
            "join_name": "canonical_rows_to_stage1_natural_key",
            "left_rows": canonical_rows, "right_rows": len(stage1),
            "left_unique_keys": len(canonical_key_counts), "right_unique_keys": len(stage1_key_map),
            "expected_cardinality": "m:1", "matched_left_rows": canonical_stage1_matches,
            "unmatched_left_rows": canonical_rows - canonical_stage1_matches,
            "right_duplicate_keys": len(stage1_key_duplicates), "output_rows": canonical_rows,
            "explosion_rows": 0, "status": "PASS" if stage1_key_duplicates.empty else "FAIL",
        },
        {
            "join_name": "selected_stage1_to_baseline_attrition_ledger",
            "left_rows": len(selected_stage1), "right_rows": len(baseline),
            "left_unique_keys": selected_stage1["canonical_observation_id"].nunique(),
            "right_unique_keys": baseline["canonical_observation_id"].nunique(),
            "expected_cardinality": "1:1", "matched_left_rows": int(selected_stage1["baseline_status"].ne("MISSING_BASELINE_STATUS").sum()),
            "unmatched_left_rows": int(selected_stage1["baseline_status"].eq("MISSING_BASELINE_STATUS").sum()),
            "right_duplicate_keys": int(baseline["canonical_observation_id"].duplicated().sum()),
            "output_rows": len(selected_stage1), "explosion_rows": 0, "status": "PASS",
        },
        {
            "join_name": "selected_stage1_to_alias_only_model_input",
            "left_rows": len(selected_stage1), "right_rows": len(alias_only),
            "left_unique_keys": len(selected_stage1), "right_unique_keys": len(alias_only_ids),
            "expected_cardinality": "1:0..1", "matched_left_rows": len(alias_only_ids),
            "unmatched_left_rows": len(selected_stage1) - len(alias_only_ids),
            "right_duplicate_keys": int(alias_only["canonical_observation_id"].duplicated().sum()),
            "output_rows": len(alias_only), "explosion_rows": 0,
            "status": "PASS_EXPECTED_59_INVALID_WEIGHTS_EXCLUDED",
        },
        {
            "join_name": "selected_stage1_to_alias_weight_model_input",
            "left_rows": len(selected_stage1), "right_rows": len(alias_weight),
            "left_unique_keys": len(selected_stage1), "right_unique_keys": len(alias_weight_ids),
            "expected_cardinality": "1:1", "matched_left_rows": len(alias_weight_ids),
            "unmatched_left_rows": len(selected_stage1) - len(alias_weight_ids),
            "right_duplicate_keys": int(alias_weight["canonical_observation_id"].duplicated().sum()),
            "output_rows": len(alias_weight), "explosion_rows": 0,
            "status": "PASS" if len(alias_weight_ids) == len(selected_stage1) else "FAIL",
        },
    ]
    join_report = pd.DataFrame(join_rows)
    write_tsv(out_dir / "join_cardinality_report.tsv", join_report)

    waterfall = pd.DataFrame([
        {"pipeline": "raw_stage1", "step_order": 1, "step": "all_rawdata_concatenated", "row_grain": "legacy raw row", "rows": total_rows, "lost_from_prior": 0},
        {"pipeline": "raw_stage1", "step_order": 2, "step": "numeric_value_parse_pass", "row_grain": "legacy raw row", "rows": numeric_rows, "lost_from_prior": numeric_failures},
        {"pipeline": "raw_stage1", "step_order": 3, "step": "resolved_gid_after_manifest_or_raw_fallback", "row_grain": "legacy raw row", "rows": eligible_rows, "lost_from_prior": unresolved_gid_rows},
        {"pipeline": "raw_stage1", "step_order": 4, "step": "contributes_to_server_stage1", "row_grain": "legacy raw row", "rows": eligible_rows - eligible_missing_stage1, "lost_from_prior": eligible_missing_stage1},
        {"pipeline": "raw_stage1", "step_order": 5, "step": "stage1_adjusted_output", "row_grain": "environment/GID/trait/unit", "rows": len(stage1), "lost_from_prior": eligible_rows - len(stage1)},
        {"pipeline": "selected_model", "step_order": 1, "step": "selected_stage1", "row_grain": "Stage-1 observation", "rows": len(selected_stage1), "lost_from_prior": 0},
        {"pipeline": "selected_model", "step_order": 2, "step": "baseline_model_ready", "row_grain": "Stage-1 observation", "rows": baseline_counts["retained_in_stage1_model_observations"], "lost_from_prior": len(selected_stage1) - baseline_counts["retained_in_stage1_model_observations"]},
        {"pipeline": "selected_model", "step_order": 3, "step": "alias_only_valid_weight_model_ready", "row_grain": "Stage-1 observation", "rows": len(alias_only), "lost_from_prior": len(selected_stage1) - len(alias_only)},
        {"pipeline": "selected_model", "step_order": 4, "step": "alias_plus_fold_local_weight_registry_model_ready", "row_grain": "Stage-1 observation", "rows": len(alias_weight), "lost_from_prior": len(selected_stage1) - len(alias_weight)},
        {"pipeline": "canonical_parallel", "step_order": 1, "step": "canonical_all_traits", "row_grain": "canonical summary row", "rows": canonical_rows, "lost_from_prior": 0},
        {"pipeline": "canonical_parallel", "step_order": 2, "step": "canonical_rows_with_stage1_natural_key", "row_grain": "canonical summary row", "rows": canonical_stage1_matches, "lost_from_prior": canonical_rows - canonical_stage1_matches},
        {"pipeline": "canonical_parallel", "step_order": 3, "step": "canonical_selected_traits", "row_grain": "canonical summary row", "rows": canonical_selected_rows, "lost_from_prior": canonical_rows - canonical_selected_rows},
        {"pipeline": "canonical_parallel", "step_order": 4, "step": "canonical_selected_rows_with_stage1_natural_key", "row_grain": "canonical summary row", "rows": canonical_selected_stage1_matches, "lost_from_prior": canonical_selected_rows - canonical_selected_stage1_matches},
    ])
    write_tsv(out_dir / "attrition_waterfall.tsv", waterfall)

    defects = pd.DataFrame([
        {"defect_id": "D2-001", "classification": "CONFIRMED_PROVENANCE_DEFECT", "affected_rows": total_rows, "title": "Legacy concatenation omits source sheet/member and physical row", "evidence": "all_rawdata stores source_file/trial_dir only; Phase 2 had to reconstruct first-member row locators"},
        {"defect_id": "D2-002", "classification": "CONFIRMED_SILENT_ATTRITION", "affected_rows": numeric_failures, "title": "Numeric parsing failures are silently discarded", "evidence": "pd.to_numeric(errors=coerce) followed by value.notna filter has no row ledger"},
        {"defect_id": "D2-003", "classification": "CONFIRMED_AMBIGUITY_DEFECT" if resolver_conflicting_raw_rows else "NO_CONFLICT_OBSERVED", "affected_rows": resolver_conflicting_raw_rows, "title": "Manifest key collisions are resolved by keep-first", "evidence": "drop_duplicates on trial/cycle/occ/CID/SID without ambiguity output"},
        {"defect_id": "D2-004", "classification": "CONFIRMED_AMBIGUITY_DEFECT" if trait_ambiguous_raw_rows else "NO_CONFLICT_OBSERVED", "affected_rows": trait_ambiguous_raw_rows, "title": "Trait/unit mappings are resolved by prefer-unit then keep-first", "evidence": "one mapping retained per normalized trait key"},
        {"defect_id": "D2-005", "classification": "CONFIRMED_UNIT_OVERRIDE", "affected_rows": unit_conflict_rows, "title": "Model-input unit overrides disagreeing raw units", "evidence": "mapped unit is preferred whenever nonblank"},
        {"defect_id": "D2-006", "classification": "CONFIRMED_DUPLICATE_WEIGHTING_RISK", "affected_rows": raw_plot_duplicate_excess, "title": "Repeated raw plot keys are not collapsed or conflict-adjudicated before fitting", "evidence": f"exact duplicate excess={raw_exact_duplicate_excess}; plot-key excess={raw_plot_duplicate_excess}"},
        {"defect_id": "D2-007", "classification": "CONFIRMED_ORDERING_DEFECT", "affected_rows": 22609, "title": "Baseline kernel membership was evaluated before certified environment aliases existed", "evidence": "62 aliases recover 22,609 selected Stage-1 rows"},
        {"defect_id": "D2-008", "classification": "CONFIRMED_SILENT_ATTRITION", "affected_rows": 59, "title": "Positive-weight filter removes rows before fold-local recovery", "evidence": "alias-only input has 277,942 rows; alias+weight input has 278,001"},
        {"defect_id": "D2-009", "classification": "CONFIRMED_KEY_COMPLETENESS_DEFECT" if incomplete_environment_rows else "NO_MISSING_COMPONENT_OBSERVED", "affected_rows": incomplete_environment_rows, "title": "Environment keys permit missing trial/cycle/occurrence/location components", "evidence": "legacy key construction concatenates empty strings without a completeness gate"},
        {"defect_id": "D2-010", "classification": "CONFIRMED_MISSING_CODE_POLICY_GAP", "affected_rows": zero_rows + numeric_sentinel_rows, "title": "Zero and numeric sentinel candidates are retained without trait-specific missing-code policy", "evidence": f"zeros={zero_rows}; numeric sentinel candidates={numeric_sentinel_rows}"},
        {"defect_id": "D2-011", "classification": "REVIEW_REQUIRED" if alternate_gid_ignored_rows else "NO_AFFECTED_ROWS", "affected_rows": alternate_gid_ignored_rows, "title": "Alternate GID columns are ignored", "evidence": "Stage-1 usecols reads GID but not GID.1/other suffixed identifier columns"},
        {"defect_id": "D2-012", "classification": "REVIEW_REQUIRED" if wide_candidate_rows else "NO_WIDE_PHENOTYPE_COLUMN_CONFIRMED", "affected_rows": wide_candidate_rows, "title": "Legacy parser does not perform wide-to-long conversion", "evidence": "non-core numeric columns are ignored; candidates are itemized separately"},
        {"defect_id": "D2-013", "classification": "CONFIRMED_PROVENANCE_DEFECT", "affected_rows": canonical_rows, "title": "Canonical summary rows are a parallel branch, not source-row lineage into Stage 1", "evidence": "canonical summaries and raw Stage-1 contributions have different grains and no persisted bridge"},
        {"defect_id": "D2-014", "classification": "NO_OUTLIER_REMOVAL_FOUND", "affected_rows": 0, "title": "No Stage-1 outlier removal was found", "evidence": "legacy normalization and fitting code contains no outlier filter; therefore no unledgered outlier removals were identified"},
        {"defect_id": "D2-015", "classification": "CONFIRMED_REPRODUCIBILITY_DEFECT", "affected_rows": total_rows, "title": "Source discovery is repository-wide and output paths are overwrite-prone", "evidence": "collect_trial_tables uses BASE.rglob and builders write fixed production paths"},
        {"defect_id": "D2-016", "classification": "NO_SILENT_INNER_JOIN_FOUND", "affected_rows": 0, "title": "Core Stage-1 identity and trait joins are left joins", "evidence": "No literal inner join removes raw rows; equivalent losses occur in explicit post-join filters and downstream order membership filters."},
        {"defect_id": "D2-017", "classification": "CONFIRMED_MODALITY_ORDER_ATTRITION", "affected_rows": baseline_counts["genotype_not_in_stage1_model_order"], "title": "Phenotypes are removed downstream when absent from the legacy genotype order", "evidence": "This is not Stage-1 construction attrition and is not proof of marker absence; canonical-v3 pedigree inputs later recover the affected model intersection."},
        {"defect_id": "D2-018", "classification": "CONFIRMED_FILTER_NOT_JOIN", "affected_rows": baseline_counts["invalid_or_nonpositive_stage1_weight"], "title": "Weights do not remove rows through a join; a direct positive-weight filter removes them", "evidence": "build_stage1_model_kernels filters weight_g_e before alias application unless allow-missing-weight is enabled."},
    ])
    write_tsv(out_dir / "confirmed_pipeline_defects.tsv", defects)

    legitimate = pd.DataFrame([
        {"category": "NONNUMERIC_OR_BLANK_VALUE", "rows": numeric_failures, "legitimacy": "LEGITIMATE_FOR_MODEL_EXCLUSION_BUT_LEDGER_REQUIRED", "reason": "No numeric phenotype is available for fitting; original token must remain traceable."},
        {"category": "UNRESOLVED_GENOTYPE_AFTER_DOCUMENTED_FALLBACK", "rows": unresolved_gid_rows, "legitimacy": "LEGITIMATE_FAIL_CLOSED_EXCLUSION", "reason": "No accepted GID may be invented; alias evidence belongs in human review."},
        {"category": "TRAIT_OUTSIDE_SELECTED_SEVEN", "rows": int((~norm_text(stage1["trait_name_canonical"], upper=True).isin(SELECTED_TRAITS)).sum()), "legitimacy": "LEGITIMATE_FROZEN_MODEL_SCOPE_ONLY", "reason": "Rows remain in Stage 1 and are excluded only from the frozen seven-trait development scope."},
        {"category": "INVALID_OR_NONPOSITIVE_WEIGHT", "rows": baseline_counts["invalid_or_nonpositive_stage1_weight"], "legitimacy": "NOT_A_PHENOTYPE_EXCLUSION; TRAINING_ONLY_WEIGHT_RECOVERY_REQUIRED", "reason": "Outcome remains observed/adjusted; uncertainty handling must be fit within training folds."},
        {"category": "FALLBACK_STAGE1_ADJUSTMENT", "rows": int((~stage1["stage1_model_status"].eq("linear_model_adjusted")).sum()), "legitimacy": "RETAIN_WITH_EXPLICIT_DERIVED_STATUS", "reason": "Small or unfit groups are retained as fallback means, never represented as raw observed outcomes."},
        {"category": "PEDIGREE_OR_MARKER_UNAVAILABLE", "rows": int((~selected_stage1["canonical_pedigree_available"] | ~selected_stage1["any_marker_available"]).sum()), "legitimacy": "MODEL_MODALITY_AVAILABILITY_NOT_STAGE1_ATTRITION", "reason": "Stage 1 must not remove phenotypes for missing genotype modalities; downstream kernels use masks/orders."},
    ])
    write_tsv(out_dir / "legitimate_exclusion_categories.tsv", legitimate)

    ambiguity_summary = pd.DataFrame([
        {"review_class": "GENOTYPE_RESOLVER_CONFLICTING_KEY", "items": int(resolver_audit["resolver_key_status"].str.startswith("DUPLICATE_CONFLICTING").sum()), "affected_rows": resolver_conflicting_raw_rows, "detail_artifact": "genotype_resolver_key_audit.tsv"},
        {"review_class": "UNRESOLVED_GENOTYPE_ALIAS", "items": len(unresolved_gid_frame), "affected_rows": unresolved_gid_rows, "detail_artifact": "unresolved_genotype_alias_candidates.tsv"},
        {"review_class": "TRAIT_OR_UNIT_MAPPING_AMBIGUITY", "items": int(trait_audit["trait_mapping_status"].str.startswith("AMBIGUOUS").sum()), "affected_rows": trait_ambiguous_raw_rows, "detail_artifact": "trait_mapping_key_audit.tsv"},
        {"review_class": "ENVIRONMENT_ALIAS_COLLISION_RESOLUTION", "items": len(alias_collision), "affected_rows": int(pd.to_numeric(alias_collision.get("stage1_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()), "detail_artifact": "environment_alias_collision_review.tsv"},
        {"review_class": "RAW_UNIT_CONFLICT", "items": len(unit_frame), "affected_rows": unit_conflict_rows, "detail_artifact": "unit_conflicts.tsv"},
        {"review_class": "ZERO_VALUE_TRAIT_POLICY", "items": len(zero_frame), "affected_rows": zero_rows, "detail_artifact": "zero_value_audit.tsv"},
        {"review_class": "NUMERIC_SENTINEL_POLICY", "items": len(sentinel_frame), "affected_rows": numeric_sentinel_rows, "detail_artifact": "numeric_missing_sentinel_candidates.tsv"},
        {"review_class": "POSSIBLE_WIDE_COLUMN", "items": int(wide_frame["classification"].eq("POSSIBLE_WIDE_VALUE_COLUMN_REQUIRES_REVIEW").sum()) if not wide_frame.empty else 0, "affected_rows": wide_candidate_rows, "detail_artifact": "wide_to_long_omission_candidates.tsv"},
        {"review_class": "INCOMPLETE_ENVIRONMENT_KEY", "items": len(incomplete_environment_frame), "affected_rows": incomplete_environment_rows, "detail_artifact": "incomplete_environment_keys.tsv"},
    ])
    write_tsv(out_dir / "unresolved_human_review_summary.tsv", ambiguity_summary)

    summary = {
        "status": "PASS_FORENSIC_DIAGNOSTIC_COMPLETE",
        "repository_commit": git_commit(root),
        "canonical_rows": canonical_rows,
        "canonical_distinct_permanent_row_ids": canonical_distinct_ids,
        "raw_rows": total_rows,
        "numeric_rows": numeric_rows,
        "numeric_parse_failures": numeric_failures,
        "eligible_stage1_raw_rows": eligible_rows,
        "stage1_rows": len(stage1),
        "selected_stage1_rows": len(selected_stage1),
        "raw_to_stage1_reconciliation": raw_gate,
        "canonical_rows_with_stage1_key": canonical_stage1_matches,
        "canonical_selected_rows": canonical_selected_rows,
        "canonical_selected_rows_with_stage1_key": canonical_selected_stage1_matches,
        "canonical_disposition_counts": dict(sorted(canonical_disposition_counts.items())),
        "confirmed_defects": int(defects["classification"].str.startswith("CONFIRMED").sum()),
        "protected_content_read": False,
        "stage1_rebuilt": False,
        "models_trained": False,
        "production_artifacts_modified": False,
    }
    write_json(out_dir / "phase2_forensic_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
