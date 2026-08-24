#!/usr/bin/env python3
"""Open and freeze the atomic Phase-4 spatial/promotion release train.

This script is intentionally fail-if-exists.  It hashes only development inputs;
the locked outer-test and sealed final-holdout trees are neither discovered nor
opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


VERSION = "v1"
RELEASE_TRAIN_ID = "P4ISP_20260802_V1_274E41DF"
INTEGRATED_STATUS_CANDIDATE = "RELEASE_CANDIDATE_NOT_YET_DECIDED"


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            h.update(block)
    return h.hexdigest()


def stable_set_hash(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["path"]):
        h.update(
            f"{row['path']}\t{row['bytes']}\t{row['sha256']}\n".encode("utf-8")
        )
    return h.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
    ).strip()


def parquet_profile(path: Path) -> tuple[str, str, str]:
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    schema_json = json.dumps(
        [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in schema],
        separators=(",", ":"),
    )
    # Full per-column missingness is produced later for analytical inputs.  The
    # opening profile records footer-level grain without scanning protected rows.
    key_hint = ";".join(
        name for name in schema.names
        if name.endswith("_id") or name in {"resolved_gid", "canonical_gid", "phase4_group_id"}
    )
    return str(pf.metadata.num_rows), schema_json, key_hint


def add_file(
    rows: list[dict[str, Any]], root: Path, path: Path, category: str, role: str,
    *, hash_now: bool = True,
) -> None:
    resolved = path.resolve()
    row_count = ""
    schema_json = ""
    key_hint = ""
    if resolved.suffix.lower() == ".parquet":
        row_count, schema_json, key_hint = parquet_profile(resolved)
    rows.append(
        {
            "release_train_id": RELEASE_TRAIN_ID,
            "integrated_release_version": VERSION,
            "category": category,
            "role": role,
            "path": resolved.as_posix(),
            "relative_path": resolved.relative_to(root.resolve()).as_posix()
            if resolved.is_relative_to(root.resolve()) else "",
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved) if hash_now else "",
            "row_count": row_count,
            "schema_json": schema_json,
            "key_hint": key_hint,
        }
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["release_train_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-start-time", required=True)
    parser.add_argument("--resume-incomplete-opening", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    release = root / "audit" / "v2" / f"phase4_integrated_spatial_promotion_release_{VERSION}"
    if release.exists():
        existing = sorted(
            p.relative_to(release).as_posix() for p in release.rglob("*") if p.is_file()
        )
        allowed_resume_files = {"logs/opening.stdout.log", "logs/opening.stderr.log"}
        unexpected = [name for name in existing if name not in allowed_resume_files]
        if not args.resume_incomplete_opening or unexpected:
            raise FileExistsError(
                f"Fail-if-exists release root already exists with files={unexpected}: {release}"
            )
    else:
        release.mkdir(parents=True)
    (release / "logs").mkdir(exist_ok=True)

    phase4 = root / "audit" / "v2" / "phase4_phenotype_reconstruction_signal_assessment_v1"
    phase3g = root / "audit" / "v2" / "phase3g_all_panel_genotype_linkage_audit_v2"
    stage1_root = root / "audit" / "v2" / "phase3_stage1_v2_reconstruction_v1"
    raw_root = root / "TRIALS_AND_NURSERIES_DATA"
    required_dirs = [phase4, phase3g, stage1_root, raw_root]
    if not all(path.is_dir() for path in required_dirs):
        raise FileNotFoundError(f"Missing required input: {[str(p) for p in required_dirs if not p.is_dir()]}")

    source_rows: list[dict[str, Any]] = []
    for path in sorted(p for p in raw_root.rglob("*") if p.is_file()):
        add_file(source_rows, root, path, "RAW_TRIAL_CORPUS", "READ_ONLY_COORDINATE_SEARCH_SOURCE")

    for path in sorted(p for p in phase4.iterdir() if p.is_file()):
        add_file(source_rows, root, path, "PHASE4_V1", "READ_ONLY_AUTHORITATIVE_PHENOTYPE_SOURCE")

    phase3g_inputs = [
        "accepted_all_panel_gid_union.parquet",
        "unresolved_phenotype_identity_candidates.parquet",
        "phase3g_r2_build_summary.json",
        "r2_protocol.json",
        "output_manifest.tsv",
        "PHASE3G_R2_CORRECTIVE_REPORT.md",
    ]
    for name in phase3g_inputs:
        add_file(source_rows, root, phase3g / name, "PHASE3G_R2", "READ_ONLY_IDENTITY_AUTHORITY")

    stage1_inputs = [
        stage1_root / "layers_v2_release_candidate_v2" / "canonical_observations_v2.parquet",
        stage1_root / "stage1_v2_release_candidate_v3" / "stage1_adjusted_phenotypes_v2.parquet",
        stage1_root / "stage1_v2_release_candidate_v3" / "canonical_to_stage1_contribution_bridge_v2.parquet",
    ]
    for path in stage1_inputs:
        add_file(source_rows, root, path, "STAGE1_V2", "READ_ONLY_AUTHORITATIVE_STAGE1_INPUT")

    documentation = [
        root / "README.md",
        root / "docs" / "v2" / "MASTER_PLAN.md",
        root / "docs" / "v2" / "STATUS.md",
        root / "docs" / "v2" / "PHASE4_REPORT.md",
        root / "docs" / "v2" / "VALIDATION_CONTRACT.md",
        root / "scripts" / "v2" / "phase4_reconstruct_phenotypes.py",
        root / "scripts" / "v2" / "phase4_validate_release.py",
        root / "tests" / "test_phase4_phenotype_reconstruction.py",
    ]
    for path in documentation:
        add_file(source_rows, root, path, "CODE_OR_DOCUMENTATION", "READ_ONLY_PROTOCOL_EVIDENCE")

    write_tsv(release / "source_artifact_inventory.tsv", source_rows)
    write_tsv(release / "OPENING_HASH_MANIFEST.tsv", source_rows)

    phase4_rows = [r for r in source_rows if r["category"] == "PHASE4_V1"]
    phase3g_rows = [r for r in source_rows if r["category"] == "PHASE3G_R2"]
    stage1_rows = [r for r in source_rows if r["category"] == "STAGE1_V2"]
    raw_rows = [r for r in source_rows if r["category"] == "RAW_TRIAL_CORPUS"]
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")
    dirty = bool(git(root, "status", "--porcelain"))
    manifest = {
        "release_train_id": RELEASE_TRAIN_ID,
        "integrated_release_version": VERSION,
        "integrated_release_root": release.as_posix(),
        "status": INTEGRATED_STATUS_CANDIDATE,
        "source_phase4_release": phase4.as_posix(),
        "source_phase4_hash": stable_set_hash(phase4_rows),
        "phase3g_identity_release": phase3g.as_posix(),
        "phase3g_identity_release_hash": stable_set_hash(phase3g_rows),
        "stage1_release": (stage1_root / "stage1_v2_release_candidate_v3").as_posix(),
        "stage1_input_set_hash": stable_set_hash(stage1_rows),
        "raw_trial_corpus": raw_root.as_posix(),
        "raw_artifact_count": len(raw_rows),
        "raw_input_set_hash": stable_set_hash(raw_rows),
        "run_start_time": args.run_start_time,
        "opening_manifest_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "git_head": head,
        "dirty_worktree_status": dirty,
        "outer_test_content_accessed": False,
        "final_holdout_content_accessed": False,
        "phase5_started": False,
        "component_authoritative_statuses_allowed": False,
        "immutable_input_policy": "all listed analytical and raw inputs are read-only",
    }
    (release / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
