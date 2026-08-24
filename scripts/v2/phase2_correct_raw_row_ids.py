"""Create a collision-free final raw ledger without overwriting the provisional audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def corrected_id(frame: pd.DataFrame) -> pd.Series:
    locator = (
        frame["source_file"].fillna("").astype(str)
        + "|" + frame["source_file_sha256"].fillna("").astype(str)
        + "|" + frame["source_member"].fillna("").astype(str)
        + "|" + frame["source_physical_row"].fillna("").astype(str)
    )
    return "RAW2_" + locator.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:24])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200_000)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    output_path = result_dir / "raw_row_disposition_ledger_final.parquet"
    parquet = pq.ParquetFile(args.input.resolve())
    writer: pq.ParquetWriter | None = None
    old_seen: set[str] = set()
    old_duplicate_counts: Counter[str] = Counter()
    new_ids: set[str] = set()
    rows = 0

    try:
        for batch in parquet.iter_batches(batch_size=args.batch_size):
            frame = batch.to_pandas()
            old = frame["raw_source_row_id"].fillna("").astype(str)
            for old_id in old:
                if old_id in old_seen:
                    old_duplicate_counts[old_id] += 1
                else:
                    old_seen.add(old_id)
            frame["provisional_raw_source_row_id"] = old
            frame["raw_source_row_id"] = corrected_id(frame)
            duplicate_in_batch = frame["raw_source_row_id"].duplicated().any()
            if duplicate_in_batch:
                raise RuntimeError("Corrected raw IDs duplicate within a batch")
            overlap = new_ids.intersection(frame["raw_source_row_id"])
            if overlap:
                raise RuntimeError(f"Corrected raw IDs duplicate across batches: {next(iter(overlap))}")
            new_ids.update(frame["raw_source_row_id"])
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(frame)
    finally:
        if writer is not None:
            writer.close()

    collision_rows = []
    for old_id, duplicate_count in sorted(old_duplicate_counts.items()):
        collision_rows.append({
            "provisional_raw_source_row_id": old_id,
            "rows": duplicate_count + 1,
            "excess_rows": duplicate_count,
            "cause": "BYTE_IDENTICAL_SOURCE_FILES_SHARE_HASH_MEMBER_AND_ROW",
        })
    pd.DataFrame(collision_rows).to_csv(
        result_dir / "provisional_raw_row_id_collision_ledger.tsv", sep="\t", index=False
    )
    summary = {
        "status": "PASS_COLLISION_FREE_FINAL_RAW_ROW_IDS",
        "rows": rows,
        "provisional_distinct_ids": len(old_seen),
        "provisional_colliding_id_groups": len(old_duplicate_counts),
        "provisional_excess_duplicate_ids": sum(old_duplicate_counts.values()),
        "final_distinct_ids": len(new_ids),
        "final_duplicate_ids": rows - len(new_ids),
        "final_algorithm": "RAW2_ + sha256(source_file|source_file_sha256|source_member|source_physical_row)[0:24]",
    }
    (result_dir / "raw_row_id_correction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if len(new_ids) != rows:
        raise RuntimeError("Final raw row IDs are not unique")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
