"""Re-hash raw roots after Phase 2 and compare them with the Phase-1 baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "source_root", "dataset", "relative_path", "absolute_path", "suffix",
    "bytes", "mtime_ns", "mtime_utc", "sha256",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, label: str, workers: int) -> list[dict[str, object]]:
    paths = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda p: p.as_posix().casefold())
    hashes = list(ThreadPoolExecutor(max_workers=workers).map(sha256_file, paths))
    rows: list[dict[str, object]] = []
    for path, digest in zip(paths, hashes, strict=True):
        relative = path.relative_to(root)
        stat = path.stat()
        rows.append({
            "source_root": label,
            "dataset": relative.parts[0] if len(relative.parts) > 1 else "",
            "relative_path": relative.as_posix(),
            "absolute_path": str(path),
            "suffix": path.suffix.lower(),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": digest,
        })
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--phase1-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    phase1 = args.phase1_dir.resolve()
    out_dir = args.out_dir.resolve()
    roots = {
        "TRIALS_AND_NURSERIES_DATA": workspace / "TRIALS_AND_NURSERIES_DATA",
        "GENOTYPIC_DATA": workspace / "GENOTYPIC_DATA",
    }
    baseline_names = {
        "TRIALS_AND_NURSERIES_DATA": "trial_file_inventory_before.tsv",
        "GENOTYPIC_DATA": "genotype_file_inventory_before.tsv",
    }
    after_names = {
        "TRIALS_AND_NURSERIES_DATA": "trial_file_inventory_phase2_after.tsv",
        "GENOTYPIC_DATA": "genotype_file_inventory_phase2_after.tsv",
    }

    comparison: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for label, root in roots.items():
        current = inventory(root, label, args.workers)
        write_tsv(out_dir / after_names[label], current, FIELDS)
        baseline = {row["relative_path"]: row for row in read_tsv(phase1 / baseline_names[label])}
        after = {str(row["relative_path"]): row for row in current}
        status_counts: dict[str, int] = {}
        for relative in sorted(set(baseline) | set(after)):
            before = baseline.get(relative)
            now = after.get(relative)
            if before is None:
                status = "ADDED"
            elif now is None:
                status = "REMOVED"
            elif before["sha256"] != now["sha256"]:
                status = "CONTENT_CHANGED"
            elif int(before["bytes"]) != int(now["bytes"]):
                status = "SIZE_CHANGED_WITH_SAME_HASH_IMPOSSIBLE"
            else:
                status = "MATCH"
            status_counts[status] = status_counts.get(status, 0) + 1
            comparison.append({
                "source_root": label,
                "relative_path": relative,
                "status": status,
                "before_bytes": "" if before is None else before["bytes"],
                "after_bytes": "" if now is None else now["bytes"],
                "before_sha256": "" if before is None else before["sha256"],
                "after_sha256": "" if now is None else now["sha256"],
            })
        summaries.append({
            "source_root": label,
            "baseline_files": len(baseline),
            "phase2_after_files": len(after),
            "status_counts": status_counts,
        })

    write_tsv(
        out_dir / "raw_phase1_to_phase2_comparison.tsv",
        comparison,
        ["source_root", "relative_path", "status", "before_bytes", "after_bytes", "before_sha256", "after_sha256"],
    )
    summary_path = out_dir / "raw_phase1_to_phase2_comparison_summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    nonmatches = [row for row in comparison if row["status"] != "MATCH"]
    if nonmatches:
        raise RuntimeError(f"Raw-data immutability failure: {len(nonmatches)} files differ from the Phase-1 baseline")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
