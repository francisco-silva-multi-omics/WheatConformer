#!/usr/bin/env python3
"""Open and freeze the atomic Phase-5 kernel-validation release.

Only explicitly authorized training-side and upstream-release paths are walked.
Protected outer-test and final-holdout outcome trees are never discovered or read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


VERSION = "v1"
RELEASE_TRAIN_ID = "P5KV_20260802_V1_274E41DF"
STATUS_CANDIDATE = "PHASE5_AUDIT_IN_PROGRESS"
EXPECTED_P4_SET_HASH = "bfc637afdd28d9763f01181070477dd330df81680b1fc00fcb69cca2a39312b5"


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def stable_set_hash(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["path"]):
        digest.update(
            f"{row['path']}\t{row['bytes']}\t{row['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
    ).strip()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["release_train_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_output_manifest(directory: Path, manifest_name: str = "output_manifest.tsv") -> dict[str, Any]:
    manifest = directory / manifest_name
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    checked = 0
    mismatches: list[str] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            relative = row["relative_path"]
            target = directory / relative
            if not target.is_file():
                mismatches.append(f"missing:{relative}")
                continue
            observed = sha256_file(target)
            if observed != row["sha256"] or target.stat().st_size != int(row["bytes"]):
                mismatches.append(f"hash_or_size:{relative}")
            checked += 1
    return {"manifest": manifest.as_posix(), "checked": checked, "mismatches": mismatches}


def parquet_footer(path: Path) -> tuple[str, str, str]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    schema_json = json.dumps(
        [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in schema],
        separators=(",", ":"),
    )
    keys = ";".join(
        name for name in schema.names
        if name.endswith("_id") or name in {"resolved_gid", "canonical_gid", "GID"}
    )
    return str(parquet.metadata.num_rows), schema_json, keys


def add_file(
    rows: list[dict[str, Any]], root: Path, path: Path, category: str, role: str
) -> None:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    lowered = resolved.as_posix().lower()
    forbidden = ("outer_test", "outer-test", "final_holdout", "final-holdout", "final_nested_evaluation")
    if any(token in lowered for token in forbidden):
        raise RuntimeError(f"Protected path is not authorized for Phase 5: {resolved}")
    row_count = schema_json = key_hint = ""
    if resolved.suffix.lower() == ".parquet":
        try:
            row_count, schema_json, key_hint = parquet_footer(resolved)
        except Exception:
            # Raw genotype files may use a misleading extension or optional codec.
            pass
    rows.append(
        {
            "release_train_id": RELEASE_TRAIN_ID,
            "phase5_release_version": VERSION,
            "category": category,
            "role": role,
            "path": resolved.as_posix(),
            "relative_path": (
                resolved.relative_to(root.resolve()).as_posix()
                if resolved.is_relative_to(root.resolve()) else ""
            ),
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
            "row_count": row_count,
            "schema_json": schema_json,
            "key_hint": key_hint,
        }
    )


def existing_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def command_version(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-start-time", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    release = root / "audit" / "v2" / f"phase5_kernel_validation_{VERSION}"
    if release.exists():
        raise FileExistsError(f"Phase-5 release root must be new and empty: {release}")
    release.mkdir(parents=True)
    (release / "logs").mkdir()
    (release / "figures").mkdir()

    p4_integrated = root / "audit" / "v2" / "phase4_integrated_spatial_promotion_release_v1"
    p4_source = root / "audit" / "v2" / "phase4_phenotype_reconstruction_signal_assessment_v1"
    p3g = root / "audit" / "v2" / "phase3g_all_panel_genotype_linkage_audit_v2"
    genotype_root = root / "GENOTYPIC_DATA"
    trial_root = root / "TRIALS_AND_NURSERIES_DATA"
    for required in (p4_integrated, p4_source, p3g, genotype_root, trial_root):
        if not required.is_dir():
            raise FileNotFoundError(required)

    p4_manifest_check = verify_output_manifest(p4_integrated)
    p3g_manifest_check = verify_output_manifest(p3g)
    if p4_manifest_check["mismatches"] or p3g_manifest_check["mismatches"]:
        raise RuntimeError(
            f"Upstream output-manifest mismatch: p4={p4_manifest_check}; p3g={p3g_manifest_check}"
        )

    p4_decision = json.loads((p4_integrated / "RELEASE_DECISION.json").read_text(encoding="utf-8"))
    p4_pointer = json.loads((p4_integrated / "authoritative_phase4_pointer.json").read_text(encoding="utf-8"))
    p3g_decision = json.loads((p3g / "validation_summary_final.json").read_text(encoding="utf-8"))
    expected_contract = {
        "status": "PASS_PHASE4_INTEGRATED_SPATIAL_PROMOTION",
        "release_train_id": "P4ISP_20260802_V1_274E41DF",
        "integrated_release_version": "v1",
        "coordinate_outcome": "NO_VALID_COORDINATES_FOUND",
        "promoted_rows": 3_193_677,
        "trial_trait_groups": 37_206,
    }
    contract_mismatches = {
        key: {"expected": value, "observed": p4_decision.get(key)}
        for key, value in expected_contract.items() if p4_decision.get(key) != value
    }
    pointer_expectations = {
        "authoritative_phase4_candidate_id": "PHASE4_V1_bfc637afdd28d976",
        "authoritative_phase4_candidate_hash": EXPECTED_P4_SET_HASH,
        "integrated_release_version": "v1",
        "mixed_version": False,
        "phenotype_correction_required": False,
    }
    for key, value in pointer_expectations.items():
        if p4_pointer.get(key) != value:
            contract_mismatches[f"pointer.{key}"] = {"expected": value, "observed": p4_pointer.get(key)}
    if p3g_decision.get("status") != "PASS_PHASE3G_R2_CORRECTION":
        contract_mismatches["phase3g.status"] = {
            "expected": "PASS_PHASE3G_R2_CORRECTION", "observed": p3g_decision.get("status")
        }
    if contract_mismatches:
        raise RuntimeError(f"Pinned upstream contract mismatch: {contract_mismatches}")

    source_rows: list[dict[str, Any]] = []
    for path in sorted(p4_integrated.iterdir()):
        if path.is_file():
            add_file(source_rows, root, path, "PHASE4_INTEGRATED_V1", "IMMUTABLE_PROMOTION_CONTRACT")
    for path in sorted(p4_source.iterdir()):
        if path.is_file():
            add_file(source_rows, root, path, "PHASE4_SOURCE_V1", "IMMUTABLE_PHENOTYPE_SOURCE")
    for path in sorted(p3g.iterdir()):
        if path.is_file():
            add_file(source_rows, root, path, "PHASE3G_R2", "IMMUTABLE_IDENTITY_AUTHORITY")

    p4_source_rows = [row for row in source_rows if row["category"] == "PHASE4_SOURCE_V1"]
    p4_source_hash = stable_set_hash(p4_source_rows)
    if p4_source_hash != EXPECTED_P4_SET_HASH:
        raise RuntimeError(f"Phase-4 source set hash mismatch: {p4_source_hash}")

    # Raw genotype corpus is fully inventoried. This is intentionally the only
    # large opening hash pass and provides a new-timepoint integrity baseline.
    for path in existing_files(genotype_root):
        add_file(source_rows, root, path, "RAW_GENOTYPE_CORPUS", "READ_ONLY_GENOTYPE_SOURCE")

    trial_name_tokens = (
        "envdata", "loc_data", "location", "environment", "germplasm_doi",
        "germplasm-doi", "pedigree", "selection", "identifier", "gid", "doi",
    )
    for path in existing_files(trial_root):
        if any(token in path.name.lower() for token in trial_name_tokens):
            add_file(source_rows, root, path, "RAW_TRIAL_LINEAGE_SOURCE", "READ_ONLY_LINEAGE_OR_ENV_SOURCE")

    source_dirs = [
        root / "environment",
        root / "genotype_panels",
        root / "server_phase1_bundle" / "artifacts" / "genotype_panels" / "pedigree_canonical_v3",
        root / "server_phase1_bundle" / "artifacts" / "model_kernels" / "stage1_canonical_v3_environment_alias_v1",
        root / "server_phase1_bundle" / "artifacts" / "model_kernels" / "stage1_canonical_v3_environment_alias_weight_v1",
    ]
    for directory in source_dirs:
        if directory.is_dir():
            for path in existing_files(directory):
                add_file(source_rows, root, path, "PRODUCTION_ARTIFACT", "READ_ONLY_KERNEL_OR_MODEL_INPUT")

    code_paths = [
        root / "build_pedigree_kernel.py",
        root / "build_gaussian_genomic_kernel.py",
        root / "build_environment_component_kernels.py",
        root / "build_stage1_model_kernels.py",
        root / "validate_model_input_matrices.py",
        root / "environment_prep.py",
        root / "server_training_pipeline" / "split_utils.py",
        root / "server_training_pipeline" / "observation_index_bundle.py",
        root / "scripts" / "01_run_core_pipeline.sh",
        root / "scripts" / "02_run_model_inputs.sh",
        root / "scripts" / "run_forensic_kernel_corrections_server.sh",
        root / "tests" / "test_forensic_kernel_math.py",
        root / "docs" / "v2" / "VALIDATION_CONTRACT.md",
        root / "docs" / "v2" / "STATUS.md",
    ]
    for path in code_paths:
        if path.is_file():
            add_file(source_rows, root, path, "CODE_OR_CONTRACT", "AUDITED_IMPLEMENTATION_OR_PROTOCOL")

    write_tsv(release / "source_artifact_inventory.tsv", source_rows)
    write_tsv(release / "OPENING_HASH_MANIFEST.tsv", source_rows)

    raw_genotype_rows = [row for row in source_rows if row["category"] == "RAW_GENOTYPE_CORPUS"]
    upstream = {
        "status": "PASS_UPSTREAM_DEPENDENCY_CHECK",
        "phase4_integrated_manifest": p4_manifest_check,
        "phase3g_r2_manifest": p3g_manifest_check,
        "phase4_contract": expected_contract,
        "phase4_pointer": p4_pointer,
        "phase4_source_set_sha256": p4_source_hash,
        "phase3g_status": p3g_decision.get("status"),
        "contract_mismatches": contract_mismatches,
        "outer_test_content_accessed": False,
        "final_holdout_content_accessed": False,
    }
    (release / "UPSTREAM_DEPENDENCY_CHECK.json").write_text(
        json.dumps(upstream, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")
    dirty_status = git(root, "status", "--porcelain")
    manifest = {
        "release_train_id": RELEASE_TRAIN_ID,
        "phase5_release_version": VERSION,
        "status": STATUS_CANDIDATE,
        "resolved_paths": {
            "repository": root.as_posix(),
            "trials_and_nurseries_data": trial_root.as_posix(),
            "genotypic_data": genotype_root.as_posix(),
            "phase3g_r2": p3g.as_posix(),
            "phase4_integrated": p4_integrated.as_posix(),
            "phase5_release_root": release.as_posix(),
        },
        "phase4_release_id": p4_decision["release_train_id"],
        "phase4_authoritative_id": p4_pointer["authoritative_phase4_candidate_id"],
        "phase4_authoritative_sha256": p4_source_hash,
        "phase3g_status": p3g_decision["status"],
        "opening_input_set_sha256": stable_set_hash(source_rows),
        "raw_genotype_file_count": len(raw_genotype_rows),
        "raw_genotype_bytes": sum(int(row["bytes"]) for row in raw_genotype_rows),
        "raw_genotype_set_sha256": stable_set_hash(raw_genotype_rows),
        "run_start_time": args.run_start_time,
        "opening_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "git_head": head,
        "dirty_worktree": bool(dirty_status),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "nvidia_smi": command_version(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"]),
        "outer_test_content_accessed": False,
        "final_holdout_content_accessed": False,
        "phase5_started": True,
        "model_training_or_tuning_performed": False,
        "future_projection_performed": False,
        "immutable_input_policy": "all source_artifact_inventory inputs are read-only",
    }
    (release / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_tsv(
        release / "dependencies_added.tsv",
        [{"dependency": "NONE", "version": "", "scope": "existing isolated environment reused"}],
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
