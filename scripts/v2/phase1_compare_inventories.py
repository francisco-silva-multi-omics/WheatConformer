"""Compare fresh Phase 1 raw manifests with the prior audit inventories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def key(value: str) -> str:
    return value.replace("\\", "/")


def fail_if_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def compare(name: str, old_path: Path, new_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    old = {key(row["relative_path"]): row for row in read_rows(old_path, ",")}
    new = {key(row["relative_path"]): row for row in read_rows(new_path, "\t")}
    rows: list[dict[str, object]] = []
    for relative in sorted(old.keys() | new.keys()):
        prior = old.get(relative)
        current = new.get(relative)
        if prior is None:
            status = "ADDED"
        elif current is None:
            status = "MISSING"
        elif prior["bytes"] != current["bytes"]:
            status = "SIZE_CHANGED"
        elif prior["sha256"].lower() != current["sha256"].lower():
            status = "HASH_CHANGED"
        else:
            status = "MATCH"
        rows.append(
            {
                "source": name,
                "relative_path": relative,
                "status": status,
                "prior_bytes": prior["bytes"] if prior else "",
                "current_bytes": current["bytes"] if current else "",
                "prior_sha256": prior["sha256"] if prior else "",
                "current_sha256": current["sha256"] if current else "",
                "prior_mtime_ns": prior["mtime_ns"] if prior else "",
                "current_mtime_ns": current["mtime_ns"] if current else "",
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    return rows, {"source": name, "prior_files": len(old), "current_files": len(new), "status_counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    comparisons = [
        ("TRIALS_AND_NURSERIES_DATA", root / "audit/raw_source_file_inventory.csv", out_dir / "trial_file_inventory_before.tsv"),
        ("GENOTYPIC_DATA", root / "audit/genotypic_data_inventory.csv", out_dir / "genotype_file_inventory_before.tsv"),
    ]
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for name, old_path, new_path in comparisons:
        rows, summary = compare(name, old_path, new_path)
        all_rows.extend(rows)
        summaries.append(summary)
    output = out_dir / "prior_vs_fresh_raw_inventory.tsv"
    fail_if_exists(output)
    fields = [
        "source", "relative_path", "status", "prior_bytes", "current_bytes",
        "prior_sha256", "current_sha256", "prior_mtime_ns", "current_mtime_ns",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    summary_path = out_dir / "prior_vs_fresh_raw_inventory_summary.json"
    fail_if_exists(summary_path)
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2, sort_keys=True))
    if any(set(summary["status_counts"]) - {"MATCH"} for summary in summaries):
        raise SystemExit("Fresh raw manifests differ from the prior audit inventories")


if __name__ == "__main__":
    main()
