#!/usr/bin/env python3
"""Open the corrective namespace/R3 release train and freeze dependencies."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from .phase4_namespace_r3_common import (
        GENOTYPE_ROOT, OVERALL_RELEASE_ID, PHASE3G_R2_ROOT, PHASE3G_R3_RELEASE_ID,
        PHASE3G_R3_ROOT, PHASE4_NS_RELEASE_ID, PHASE4_NS_ROOT, PHASE4_R3_ROOT,
        PHASE4_ROOT, PHASE5_ROOT, PINNED_R2_HASHES, REPOSITORY_ROOT, STAGE1_R3_ROOT,
        STAGE1_ROOT, TRIAL_ROOT, sha256, write_json, write_tsv,
    )
except ImportError:  # direct script execution
    from phase4_namespace_r3_common import (
    GENOTYPE_ROOT,
    OVERALL_RELEASE_ID,
    PHASE3G_R2_ROOT,
    PHASE3G_R3_RELEASE_ID,
    PHASE3G_R3_ROOT,
    PHASE4_NS_RELEASE_ID,
    PHASE4_NS_ROOT,
    PHASE4_R3_ROOT,
    PHASE4_ROOT,
    PHASE5_ROOT,
    PINNED_R2_HASHES,
    REPOSITORY_ROOT,
    STAGE1_R3_ROOT,
    STAGE1_ROOT,
    TRIAL_ROOT,
    sha256,
    write_json,
    write_tsv,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def inventory_from_signed_manifests() -> pd.DataFrame:
    p4 = pd.read_csv(PHASE4_ROOT / "source_artifact_inventory.tsv", sep="\t", dtype=str, keep_default_na=False)
    p5 = pd.read_csv(PHASE5_ROOT / "source_artifact_inventory.tsv", sep="\t", dtype=str, keep_default_na=False)
    trial = p4[p4["category"].eq("RAW_TRIAL_CORPUS")][["path", "relative_path", "bytes", "sha256"]].copy()
    trial["category"] = "RAW_TRIAL_CORPUS"
    trial["opening_hash_source"] = "P4ISP_20260802_V1_274E41DF_SIGNED_CLOSING"
    genotype = p5[p5["category"].eq("RAW_GENOTYPE_CORPUS")][["path", "relative_path", "bytes", "sha256"]].copy()
    genotype["category"] = "RAW_GENOTYPE_CORPUS"
    genotype["opening_hash_source"] = "P5KV_20260802_V1_274E41DF_SIGNED_CLOSING"
    frame = pd.concat([trial, genotype], ignore_index=True)
    frame["bytes"] = pd.to_numeric(frame["bytes"], errors="raise").astype("int64")
    frame = frame.rename(columns={"sha256": "opening_sha256", "bytes": "opening_bytes"})

    actual_relative = {
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for base in (TRIAL_ROOT, GENOTYPE_ROOT)
        for path in base.rglob("*")
        if path.is_file()
    }
    expected_relative = set(frame["relative_path"])
    if actual_relative != expected_relative:
        missing = sorted(expected_relative - actual_relative)[:10]
        added = sorted(actual_relative - expected_relative)[:10]
        raise RuntimeError(f"Raw corpus file-set drift; missing={missing}; added={added}")

    exists: list[bool] = []
    size_matches: list[bool] = []
    for row in frame.itertuples(index=False):
        path = Path(row.path)
        exists.append(path.is_file())
        size_matches.append(path.is_file() and path.stat().st_size == row.opening_bytes)
    frame["exists_at_opening"] = exists
    frame["opening_size_match"] = size_matches
    if not all(exists) or not all(size_matches):
        raise RuntimeError("Raw corpus existence or byte-size mismatch at opening")
    return frame.sort_values(["category", "relative_path"]).reset_index(drop=True)


def verify_dependencies() -> dict:
    r2_hashes = {name: sha256(PHASE3G_R2_ROOT / name) for name in PINNED_R2_HASHES}
    p3 = load_json(STAGE1_ROOT / "delivery_v1/phase3_delivery_summary.json")
    r2 = load_json(PHASE3G_R2_ROOT / "phase3g_r2_build_summary.json")
    p4 = load_json(PHASE4_ROOT / "RELEASE_DECISION.json")
    p4_pointer = load_json(PHASE4_ROOT / "authoritative_phase4_pointer.json")
    p5 = load_json(PHASE5_ROOT / "PHASE5_RELEASE_DECISION.json")
    checks = {
        "stage1_v2": p3.get("phase3_version") == "phase3_stage1_v2_reconstruction_v1" and p3.get("status") == "PASS_PHASE3_DELIVERY",
        "phase3g_r2": r2.get("status") == "PASS_R2_ARTIFACT_BUILD" and r2.get("version") == "phase3g_r2_corrective_delivery_v1" and r2.get("global_accepted_unique_gids") == 94897,
        "phase4_integrated": p4.get("status") == "PASS_PHASE4_INTEGRATED_SPATIAL_PROMOTION" and p4.get("release_train_id") == "P4ISP_20260802_V1_274E41DF",
        "phase4_pointer": p4_pointer.get("authoritative_phase4_candidate_id") == "PHASE4_V1_bfc637afdd28d976" and p4_pointer.get("authoritative_phase4_candidate_hash") == "bfc637afdd28d9763f01181070477dd330df81680b1fc00fcb69cca2a39312b5",
        "phase5_blocked": p5.get("status") == "BLOCKED_PHASE5_KERNEL_VALIDATION" and p5.get("release_train_id") == "P5KV_20260802_V1_274E41DF",
        "r2_required_hashes": r2_hashes == PINNED_R2_HASHES,
    }
    return {
        "status": "PASS_PINNED_UPSTREAM_DEPENDENCIES" if all(checks.values()) else "BLOCKED_UPSTREAM_DEPENDENCY_MISMATCH",
        "checks": checks,
        "observed_r2_hashes": r2_hashes,
        "expected_r2_hashes": PINNED_R2_HASHES,
        "authoritative_modelling_foundation": "STAGE1_V2",
        "certified_v1_consumed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-out", type=Path, default=PHASE4_NS_ROOT)
    parser.add_argument("--r3-out", type=Path, default=PHASE3G_R3_ROOT)
    args = parser.parse_args()
    phase4_out, r3_out = args.phase4_out.resolve(), args.r3_out.resolve()
    for path in (phase4_out, r3_out, STAGE1_R3_ROOT, PHASE4_R3_ROOT):
        if path.exists():
            raise FileExistsError(f"Output root must be new: {path}")

    dependency = verify_dependencies()
    if dependency["status"] != "PASS_PINNED_UPSTREAM_DEPENDENCIES":
        raise RuntimeError(json.dumps(dependency, indent=2))
    opening = inventory_from_signed_manifests()
    phase4_out.mkdir(parents=True)
    r3_out.mkdir(parents=True)
    for out in (phase4_out, r3_out):
        (out / "logs").mkdir()
        write_tsv(out / "OPENING_HASH_MANIFEST.tsv", opening)
        write_json(out / "UPSTREAM_DEPENDENCY_CHECK.json", dependency)

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    git_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPOSITORY_ROOT, text=True).strip()
    common = {
        "overall_release_id": OVERALL_RELEASE_ID,
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPOSITORY_ROOT),
        "trials_and_nurseries_data": str(TRIAL_ROOT),
        "genotypic_data": str(GENOTYPE_ROOT),
        "stage1_v2_release_root": str(STAGE1_ROOT),
        "phase3g_r2_release_root": str(PHASE3G_R2_ROOT),
        "phase4_integrated_release_root": str(PHASE4_ROOT),
        "phase5_blocked_audit_root": str(PHASE5_ROOT),
        "phase4_namespace_corrected_output_root": str(phase4_out),
        "phase3g_r3_output_root": str(r3_out),
        "stage1_r3_recovery_output_root": str(STAGE1_R3_ROOT),
        "phase4_r3_recovery_output_root": str(PHASE4_R3_ROOT),
        "stage1_r3_output_created": False,
        "phase4_r3_output_created": False,
        "raw_trial_files": int((opening["category"] == "RAW_TRIAL_CORPUS").sum()),
        "raw_genotype_files": int((opening["category"] == "RAW_GENOTYPE_CORPUS").sum()),
        "raw_bytes": int(opening["opening_bytes"].sum()),
        "git_head": git_head,
        "git_branch": git_branch,
        "python": platform.python_version(),
        "protected_outcomes_accessed": False,
        "model_training_performed": False,
        "production_kernel_construction_performed": False,
    }
    write_json(phase4_out / "run_manifest.json", {**common, "release_id": PHASE4_NS_RELEASE_ID, "status": "OPEN"})
    write_json(r3_out / "run_manifest.json", {**common, "release_id": PHASE3G_R3_RELEASE_ID, "status": "OPEN"})
    write_tsv(phase4_out / "dependencies_added.tsv", [{"dependency": "NONE", "version": "", "scope": "existing isolated environment reused"}])
    write_tsv(r3_out / "dependencies_added.tsv", [{"dependency": "NONE", "version": "", "scope": "existing isolated environment reused"}])
    print(json.dumps({"status": "PASS_OPEN_RELEASE", "phase4_release_id": PHASE4_NS_RELEASE_ID, "r3_release_id": PHASE3G_R3_RELEASE_ID, "raw_files": len(opening), "raw_bytes": int(opening.opening_bytes.sum())}, indent=2))


if __name__ == "__main__":
    main()
