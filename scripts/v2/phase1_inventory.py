"""Deterministic Phase 1 repository and input-file inventory.

The raw roots are read only. Output files are created only when absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CHUNK_BYTES = 8 * 1024 * 1024
RAW_ROOTS = ("TRIALS_AND_NURSERIES_DATA", "GENOTYPIC_DATA")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def utc_timestamp(mtime_ns: int) -> str:
    return datetime.fromtimestamp(mtime_ns / 1_000_000_000, timezone.utc).isoformat()


def fail_if_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def write_tsv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    fail_if_exists(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    fail_if_exists(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def raw_row(root: Path, source_name: str, path: Path) -> dict[str, object]:
    stat = path.stat()
    relative = path.relative_to(root).as_posix()
    parts = Path(relative).parts
    return {
        "source_root": source_name,
        "dataset": parts[0] if len(parts) > 1 else "",
        "relative_path": relative,
        "absolute_path": str(path.resolve()),
        "suffix": path.suffix.lower(),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mtime_utc": utc_timestamp(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def inventory_raw_root(root: Path, source_name: str, workers: int) -> list[dict[str, object]]:
    paths = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda p: p.as_posix())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda path: raw_row(root, source_name, path), paths))
    return rows


def git_lines(root: Path, args: list[str]) -> list[str]:
    output = subprocess.check_output(["git", "-C", str(root), *args])
    return [item.decode("utf-8", errors="surrogateescape") for item in output.split(b"\0") if item]


def repository_inventory(root: Path) -> list[dict[str, object]]:
    tracked = set(git_lines(root, ["ls-files", "-z"]))
    untracked = set(git_lines(root, ["ls-files", "--others", "--exclude-standard", "-z"]))
    rows: list[dict[str, object]] = []
    for relative in sorted(tracked | untracked):
        if relative.startswith(("TRIALS_AND_NURSERIES_DATA/", "GENOTYPIC_DATA/", "server_phase1_bundle/")):
            continue
        path = root / relative
        if path.is_file():
            stat = path.stat()
            rows.append(
                {
                    "git_scope": "tracked" if relative in tracked else "untracked",
                    "relative_path": relative,
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(path),
                    "present": True,
                }
            )
        else:
            rows.append(
                {
                    "git_scope": "tracked" if relative in tracked else "untracked",
                    "relative_path": relative,
                    "bytes": "",
                    "mtime_ns": "",
                    "sha256": "",
                    "present": False,
                }
            )
    return rows


def protected_class(relative: str) -> str:
    lowered = relative.lower()
    final_tokens = ("final_holdout", "final_nested_evaluation", "nested_evaluation_entities")
    outer_tokens = (
        "outer_test", "outer_metrics", "outer_predictions", "trained_models",
        "reaction_norm_routed_hierarchy_outer_v1",
    )
    if any(token in lowered for token in final_tokens):
        return "final_holdout_name_hash_only"
    if any(token in lowered for token in outer_tokens):
        return "locked_outer_name_hash_only"
    return "phase1_content_allowed"


def bundle_inventory(bundle_root: Path) -> list[dict[str, object]]:
    manifest_path = bundle_root / "BUNDLE_SHA256SUMS.txt"
    manifest: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        manifest[relative.removeprefix("./")] = digest.lower()
    rows: list[dict[str, object]] = []
    for path in sorted((item for item in bundle_root.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
        relative = path.relative_to(bundle_root).as_posix()
        stat = path.stat()
        rows.append(
            {
                "relative_path": relative,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": manifest.get(relative, "manifest_self_or_unlisted"),
                "access_class": protected_class(relative),
            }
        )
    return rows


def summarize_raw(rows: list[dict[str, object]]) -> dict[str, object]:
    datasets = sorted({str(row["dataset"]) for row in rows if row["dataset"]})
    suffix_counts: dict[str, int] = {}
    for row in rows:
        suffix = str(row["suffix"] or "<none>")
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    return {
        "files": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "datasets": len(datasets),
        "dataset_names": datasets,
        "suffix_counts": dict(sorted(suffix_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--snapshot", choices=("before", "after"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    bundle_root = args.bundle_root.resolve()
    if not out_dir.is_dir():
        raise FileNotFoundError(out_dir)
    if args.workers < 1:
        raise ValueError("workers must be positive")

    trial_rows = inventory_raw_root(root / RAW_ROOTS[0], RAW_ROOTS[0], args.workers)
    genotype_rows = inventory_raw_root(root / RAW_ROOTS[1], RAW_ROOTS[1], args.workers)
    raw_fields = [
        "source_root", "dataset", "relative_path", "absolute_path", "suffix",
        "bytes", "mtime_ns", "mtime_utc", "sha256",
    ]
    write_tsv(out_dir / f"trial_file_inventory_{args.snapshot}.tsv", trial_rows, raw_fields)
    write_tsv(out_dir / f"genotype_file_inventory_{args.snapshot}.tsv", genotype_rows, raw_fields)
    write_json(
        out_dir / f"raw_inventory_summary_{args.snapshot}.json",
        {RAW_ROOTS[0]: summarize_raw(trial_rows), RAW_ROOTS[1]: summarize_raw(genotype_rows)},
    )

    repo_rows = repository_inventory(root)
    repo_name = "repository_file_inventory.tsv" if args.snapshot == "before" else "repository_file_inventory_after.tsv"
    write_tsv(
        out_dir / repo_name,
        repo_rows,
        ["git_scope", "relative_path", "bytes", "mtime_ns", "sha256", "present"],
    )

    if args.snapshot == "before":
        bundle_rows = bundle_inventory(bundle_root)
        write_tsv(
            out_dir / "server_bundle_inventory.tsv",
            bundle_rows,
            ["relative_path", "bytes", "mtime_ns", "sha256", "access_class"],
        )

    print(json.dumps({
        "snapshot": args.snapshot,
        RAW_ROOTS[0]: summarize_raw(trial_rows),
        RAW_ROOTS[1]: summarize_raw(genotype_rows),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
