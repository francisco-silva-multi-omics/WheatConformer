"""Assert that Phase 1 did not alter either raw-data root."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_tsv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["relative_path"].replace("\\", "/"): row
            for row in csv.DictReader(handle, delimiter="\t")
        }


def fail_if_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    sources = {
        "TRIALS_AND_NURSERIES_DATA": "trial_file_inventory",
        "GENOTYPIC_DATA": "genotype_file_inventory",
    }
    result_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for source, stem in sources.items():
        before = read_tsv(out_dir / f"{stem}_before.tsv")
        after = read_tsv(out_dir / f"{stem}_after.tsv")
        counts: dict[str, int] = {}
        for relative in sorted(before.keys() | after.keys()):
            old = before.get(relative)
            new = after.get(relative)
            if old is None:
                status = "ADDED"
            elif new is None:
                status = "REMOVED"
            elif old["bytes"] != new["bytes"]:
                status = "SIZE_CHANGED"
            elif old["sha256"].lower() != new["sha256"].lower():
                status = "HASH_CHANGED"
            else:
                status = "MATCH"
            counts[status] = counts.get(status, 0) + 1
            result_rows.append({
                "source": source,
                "relative_path": relative,
                "status": status,
                "before_bytes": old["bytes"] if old else "",
                "after_bytes": new["bytes"] if new else "",
                "before_sha256": old["sha256"] if old else "",
                "after_sha256": new["sha256"] if new else "",
            })
        summaries.append({
            "source": source,
            "before_files": len(before),
            "after_files": len(after),
            "status_counts": counts,
        })

    table_path = out_dir / "raw_before_after_comparison.tsv"
    json_path = out_dir / "raw_before_after_comparison_summary.json"
    fail_if_exists(table_path)
    fail_if_exists(json_path)
    fields = [
        "source", "relative_path", "status", "before_bytes", "after_bytes",
        "before_sha256", "after_sha256",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_rows)
    json_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2, sort_keys=True))
    if any(set(item["status_counts"]) - {"MATCH"} for item in summaries):
        raise SystemExit("Raw roots changed during Phase 1")


if __name__ == "__main__":
    main()
