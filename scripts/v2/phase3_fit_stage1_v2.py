"""Fit versioned Stage-1 v2 phenotypes from explicit canonical contributors."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from build_stage1_adjusted_phenotypes import fit_group


STAGE1_VERSION = "stage1_v2_reconstruction_2026_07_30_v1"
GROUP_COLS = [
    "canonical_environment_id", "canonical_trial_name", "cycle", "occ", "loc_no",
    "country", "loc_desc", "accepted_canonical_trait", "trait_name_original", "standardized_unit",
]
SELECTED_TRAITS = {
    "1000_GRAIN_WEIGHT", "ABOVE_GROUND_BIOMASS", "DAYS_TO_HEADING", "DAYS_TO_MATURITY",
    "GRAIN_YIELD", "PLANT_HEIGHT", "TEST_WEIGHT",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage1_id(environment: str, gid: str, canonical_trait: str, original_trait: str, unit: str) -> str:
    value = "|".join(["RawData_stage1_v2", environment, gid, canonical_trait, original_trait, unit])
    return "STG2_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


class SafeParquetWriter:
    def __init__(self, path: Path):
        self.path = path
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        elif table.schema != self.schema:
            table = table.cast(self.schema, safe=False)
        self.writer.write_table(table, row_group_size=100_000)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def normalize_output_types(frame: pd.DataFrame) -> pd.DataFrame:
    string_cols = [
        "stage1_v2_row_id", "canonical_observation_id", "canonical_germplasm_key", "resolved_gid",
        "panel_sample_id", "genotype_name", "canonical_environment_id", "canonical_environment_key",
        "canonical_trial_name", "cycle", "occ", "loc_no", "country", "loc_desc",
        "accepted_canonical_trait", "trait_name_original", "canonical_trait_key", "standardized_unit",
        "phenotype_source", "value_semantics", "phenotype_adjustment_status", "stage1_model_status",
        "stage1_model_formula", "stage1_terms_used", "spatial_terms_used", "contributor_quality_flags",
        "stage1_version",
    ]
    float_cols = [
        "y_tilde_g_e", "SE_g_e", "var_g_e", "source_weight_g_e", "raw_mean", "raw_sd",
        "stage1_sigma2", "stage1_df_resid", "stage1_rank",
    ]
    int_cols = ["n_plot_records", "rep_count", "subblock_count", "plot_count"]
    for column in string_cols:
        frame[column] = frame[column].fillna("").astype(str)
    for column in float_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    for column in int_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype("int64")
    return frame


def process_group(
    group: pd.DataFrame,
    min_records: int,
    min_genotypes: int,
    max_params: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    key = {column: str(group.iloc[0][column]) for column in GROUP_COLS}
    fit_input = group.rename(
        columns={
            "resolved_gid_v2": "resolved_gid", "canonical_germplasm_key": "panel_sample_id",
            "value_standardized": "value",
        }
    )
    adjusted, info = fit_group(
        fit_input,
        min_records=min_records,
        min_genotypes=min_genotypes,
        max_params=max_params,
        include_plot_linear=False,
    )
    if adjusted.empty:
        raise RuntimeError(f"Stage-1 fitting returned no rows for nonempty group: {key}")
    for column, value in key.items():
        adjusted[column] = value
    adjusted["stage1_v2_row_id"] = [
        stage1_id(
            key["canonical_environment_id"], str(gid), key["accepted_canonical_trait"],
            key["trait_name_original"], key["standardized_unit"],
        )
        for gid in adjusted["resolved_gid"]
    ]
    adjusted["canonical_observation_id"] = adjusted["stage1_v2_row_id"]
    adjusted["canonical_germplasm_key"] = "GID" + adjusted["resolved_gid"].astype(str)
    adjusted["canonical_environment_key"] = adjusted["canonical_environment_id"]
    adjusted["canonical_trait_key"] = adjusted["accepted_canonical_trait"]
    adjusted["phenotype_source"] = "RawData_stage1_v2"
    adjusted["value_semantics"] = np.where(
        adjusted["stage1_model_status"].eq("linear_model_adjusted"),
        "DERIVED_STAGE1_ADJUSTED", "DERIVED_STAGE1_FALLBACK_MEAN",
    )
    adjusted["phenotype_adjustment_status"] = np.where(
        adjusted["stage1_model_status"].eq("linear_model_adjusted"),
        "stage1_adjusted_linear_model", "stage1_fallback_mean",
    )
    adjusted["source_weight_g_e"] = np.where(
        pd.to_numeric(adjusted["var_g_e"], errors="coerce").gt(0),
        1.0 / pd.to_numeric(adjusted["var_g_e"], errors="coerce"), np.nan,
    )
    quality = (
        group.groupby("resolved_gid_v2", sort=True)["quality_flags_v2"]
        .agg(lambda values: ";".join(sorted({flag for value in values for flag in str(value).split(";") if flag and flag != "NONE"})) or "NONE")
    )
    adjusted["contributor_quality_flags"] = adjusted["resolved_gid"].map(quality).fillna("NONE")
    adjusted["rep_count"] = int(info.get("rep_count", 0) or 0)
    adjusted["subblock_count"] = int(info.get("subblock_count", 0) or 0)
    adjusted["plot_count"] = int(info.get("plot_count", 0) or 0)
    adjusted["standardized_unit"] = key["standardized_unit"]
    adjusted["stage1_version"] = STAGE1_VERSION
    output_cols = [
        "stage1_v2_row_id", "canonical_observation_id", "canonical_germplasm_key", "resolved_gid",
        "panel_sample_id", "genotype_name", "canonical_environment_id", "canonical_environment_key",
        "canonical_trial_name", "cycle", "occ", "loc_no", "country", "loc_desc",
        "accepted_canonical_trait", "trait_name_original", "canonical_trait_key", "standardized_unit",
        "phenotype_source", "value_semantics", "y_tilde_g_e", "SE_g_e", "var_g_e", "source_weight_g_e",
        "raw_mean", "raw_sd", "n_plot_records", "rep_count", "subblock_count", "plot_count",
        "phenotype_adjustment_status", "stage1_model_status", "stage1_model_formula", "stage1_terms_used",
        "stage1_sigma2", "stage1_df_resid", "stage1_rank", "spatial_terms_used",
        "contributor_quality_flags", "stage1_version",
    ]
    adjusted = normalize_output_types(adjusted[output_cols].copy())
    id_map = adjusted.set_index("resolved_gid")["stage1_v2_row_id"].to_dict()
    bridge = group[[
        "raw_source_row_id", "canonical_row_id", "resolved_gid_v2", "quality_flags_v2",
        "source_file", "source_member", "source_physical_row",
    ]].copy()
    bridge["stage1_v2_row_id"] = bridge["resolved_gid_v2"].map(id_map)
    bridge["contribution_status"] = "CONTRIBUTED_TO_STAGE1_V2"
    bridge["stage1_version"] = STAGE1_VERSION
    for column in bridge.columns:
        if column != "source_physical_row":
            bridge[column] = bridge[column].fillna("").astype(str)
    bridge["source_physical_row"] = pd.to_numeric(bridge["source_physical_row"], errors="raise").astype("int64")
    qc = {
        **key,
        **info,
        "adjusted_rows": len(adjusted),
        "linear_model_adjusted_rows": int(adjusted["stage1_model_status"].eq("linear_model_adjusted").sum()),
        "fallback_rows": int((~adjusted["stage1_model_status"].eq("linear_model_adjusted")).sum()),
        "stage1_model_statuses": ";".join(sorted(adjusted["stage1_model_status"].unique())),
        "contributor_rows_reconciled": int(adjusted["n_plot_records"].sum()),
    }
    return adjusted, bridge, qc


def process_group_worker(
    payload: tuple[pd.DataFrame, int, int, int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Pickle-safe wrapper used by the bounded deterministic worker pool."""
    group, min_records, min_genotypes, max_params = payload
    return process_group(group, min_records, min_genotypes, max_params)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--min-records", type=int, default=6)
    parser.add_argument("--min-genotypes", type=int, default=2)
    parser.add_argument("--max-params", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-chunksize", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.worker_chunksize < 1:
        raise ValueError("--worker-chunksize must be at least 1")
    canonical = args.canonical.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    sorted_path = result_dir / "stage1_contributors_sorted.parquet"
    output_path = result_dir / "stage1_adjusted_phenotypes_v2.parquet"
    bridge_path = result_dir / "canonical_to_stage1_contribution_bridge_v2.parquet"

    con = duckdb.connect(str(result_dir / "stage1_sort.duckdb"))
    (result_dir / "duckdb_tmp").mkdir()
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("PRAGMA temp_directory=?", [str((result_dir / "duckdb_tmp").resolve())])
    source = str(canonical).replace("'", "''")
    target = str(sorted_path).replace("'", "''")
    select_cols = GROUP_COLS + [
        "raw_source_row_id", "canonical_row_id", "resolved_gid_v2", "canonical_germplasm_key",
        "genotype_name", "value_standardized", "rep", "subblock", "plot", "quality_flags_v2",
        "source_file", "source_member", "source_physical_row",
    ]
    quoted_cols = ", ".join(f'"{column}"' for column in select_cols)
    order_cols = ", ".join(f'"{column}"' for column in GROUP_COLS + ["raw_source_row_id"])
    con.execute(
        f"COPY (SELECT {quoted_cols} FROM read_parquet('{source}') "
        f"WHERE row_disposition_v2='ELIGIBLE_STAGE1_CONTRIBUTOR' ORDER BY {order_cols}) "
        f"TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    contributor_rows = int(con.execute(f"SELECT count(*) FROM read_parquet('{target}')").fetchone()[0])
    con.close()

    reader = pq.ParquetFile(sorted_path)
    output_writer = SafeParquetWriter(output_path)
    bridge_writer = SafeParquetWriter(bridge_path)
    carry = pd.DataFrame()
    output_buffer: list[pd.DataFrame] = []
    bridge_buffer: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []
    processed_contributors = 0
    output_rows = 0
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None

    def flush() -> None:
        nonlocal output_rows
        if output_buffer:
            combined = pd.concat(output_buffer, ignore_index=True)
            output_writer.write(combined)
            output_rows += len(combined)
            output_buffer.clear()
        if bridge_buffer:
            bridge_writer.write(pd.concat(bridge_buffer, ignore_index=True))
            bridge_buffer.clear()

    def accept_result(
        group: pd.DataFrame,
        result: tuple[pd.DataFrame, pd.DataFrame, dict[str, object]],
    ) -> None:
        nonlocal processed_contributors
        adjusted, bridge, qc = result
        if int(adjusted["n_plot_records"].sum()) != len(group) or len(bridge) != len(group):
            raise RuntimeError("Within-group Stage-1 contribution reconciliation failed")
        output_buffer.append(adjusted)
        bridge_buffer.append(bridge)
        qc_rows.append(qc)
        processed_contributors += len(group)
        if len(output_buffer) >= 100:
            flush()

    def handle_groups(groups: list[pd.DataFrame]) -> None:
        if not groups:
            return
        if executor is None:
            for group in groups:
                accept_result(
                    group,
                    process_group(group, args.min_records, args.min_genotypes, args.max_params),
                )
            return
        payloads = [
            (group, args.min_records, args.min_genotypes, args.max_params)
            for group in groups
        ]
        results = executor.map(
            process_group_worker,
            payloads,
            chunksize=args.worker_chunksize,
        )
        # executor.map preserves input ordering, so file ordering remains deterministic.
        for group, result in zip(groups, results, strict=True):
            accept_result(group, result)

    try:
        for batch in reader.iter_batches(batch_size=args.batch_size):
            frame = batch.to_pandas()
            if not carry.empty:
                frame = pd.concat([carry, frame], ignore_index=True)
            last_key = tuple(frame.iloc[-1][column] for column in GROUP_COLS)
            last_mask = pd.Series(True, index=frame.index)
            for column, value in zip(GROUP_COLS, last_key, strict=True):
                last_mask &= frame[column].eq(value)
            complete = frame[~last_mask]
            carry = frame[last_mask].copy()
            groups = [
                group for _, group in complete.groupby(GROUP_COLS, sort=False, dropna=False)
            ]
            handle_groups(groups)
        if not carry.empty:
            handle_groups([carry])
        flush()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        output_writer.close()
        bridge_writer.close()
    if processed_contributors != contributor_rows:
        raise RuntimeError(f"Contributor reconciliation failed: {processed_contributors} != {contributor_rows}")

    qc = pd.DataFrame(qc_rows)
    qc.to_csv(result_dir / "stage1_group_design_qc_v2.tsv", sep="\t", index=False)
    out = pq.read_table(output_path, columns=[
        "stage1_v2_row_id", "resolved_gid", "canonical_environment_id", "accepted_canonical_trait",
        "n_plot_records", "stage1_model_status", "var_g_e", "source_weight_g_e",
    ]).to_pandas()
    if out["stage1_v2_row_id"].duplicated().any():
        raise RuntimeError("Stage-1 v2 observation IDs are not unique")
    if int(out["n_plot_records"].sum()) != contributor_rows:
        raise RuntimeError("Global n_plot_records reconciliation failed")
    selected = out[out["accepted_canonical_trait"].isin(SELECTED_TRAITS)]
    summary = {
        "status": "PASS_STAGE1_V2_FIT",
        "stage1_version": STAGE1_VERSION,
        "workers": args.workers,
        "worker_chunksize": args.worker_chunksize,
        "blas_thread_environment": {
            name: os.environ.get(name, "")
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "canonical_contributor_rows": contributor_rows,
        "bridge_rows": pq.ParquetFile(bridge_path).metadata.num_rows,
        "stage1_rows_all_traits": len(out),
        "stage1_rows_selected_traits": len(selected),
        "unique_genotypes_all_traits": int(out["resolved_gid"].nunique()),
        "unique_environments_all_traits": int(out["canonical_environment_id"].nunique()),
        "unique_traits_all_traits": int(out["accepted_canonical_trait"].nunique()),
        "unique_genotypes_selected_traits": int(selected["resolved_gid"].nunique()),
        "unique_environments_selected_traits": int(selected["canonical_environment_id"].nunique()),
        "linear_model_rows": int(out["stage1_model_status"].eq("linear_model_adjusted").sum()),
        "fallback_rows": int((~out["stage1_model_status"].eq("linear_model_adjusted")).sum()),
        "rows_with_source_weight": int(np.isfinite(out["source_weight_g_e"]).sum()),
        "n_plot_records_sum": int(out["n_plot_records"].sum()),
        "files": {
            output_path.name: {"bytes": output_path.stat().st_size, "sha256": file_sha256(output_path)},
            bridge_path.name: {"bytes": bridge_path.stat().st_size, "sha256": file_sha256(bridge_path)},
        },
    }
    (result_dir / "stage1_v2_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
