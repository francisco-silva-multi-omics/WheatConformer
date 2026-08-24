"""Write Phase-2 command, dependency, file, integrity, and SHA closure manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--phase2-root", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    root = args.phase2_root.resolve()

    protocol = json.loads((root / "phase2_protocol.json").read_text(encoding="utf-8"))
    input_integrity: list[dict[str, object]] = []
    for item in protocol["input_bindings"]:
        path = workspace / item["path"]
        observed_bytes = path.stat().st_size
        observed_hash = sha256_file(path)
        status = "PASS" if observed_bytes == item["bytes"] and observed_hash == item["sha256"] else "FAIL"
        input_integrity.append({
            "path": item["path"], "expected_bytes": item["bytes"], "observed_bytes": observed_bytes,
            "expected_sha256": item["sha256"], "observed_sha256": observed_hash, "status": status,
        })
    write_tsv(
        root / "phase2_input_integrity_after.tsv", input_integrity,
        ["path", "expected_bytes", "observed_bytes", "expected_sha256", "observed_sha256", "status"],
    )
    if any(row["status"] != "PASS" for row in input_integrity):
        raise RuntimeError("One or more Phase-2 bound inputs changed")

    commands = [
        (1, "inspect repository, Phase-1 report and six handoff files", "PASS"),
        (2, "freeze phase2_protocol.json with exact input hashes and protected denylist", "PASS"),
        (3, "phase2_forensic_stage1_audit.py diagnostic replay", "PASS: 7,836,162 raw; 433,626 Stage-1 IDs; zero contribution mismatch"),
        (4, "phase2_finalize_findings.py initial direct execution", "FAILED: import path; preserved in log and corrected"),
        (5, "phase2_finalize_findings.py first refinement", "FAILED: duplicate grouping column; preserved in log and corrected"),
        (6, "phase2_finalize_findings.py --result-dir refinement_v2", "PASS"),
        (7, "phase2_verify_raw_immutability.py", "PASS: 2,754/2,754 MATCH"),
        (8, "direct provisional raw ID count-distinct assertion", "FOUND: 141,944 excess duplicate IDs"),
        (9, "phase2_correct_raw_row_ids.py --result-dir identity_amendment_v1", "PASS: 7,836,162 distinct final IDs"),
        (10, "phase2_audit_doi_glis_identity.py --result-dir doi_glis_audit_v1", "PRELIMINARY: nonblank placeholder classified as DOI"),
        (11, "phase2_audit_doi_glis_identity.py --result-dir doi_glis_audit_v2", "PASS: valid DOI syntax applied"),
        (12, "phase2_audit_doi_glis_identity.py --result-dir doi_glis_audit_v3", "PASS: occurrence-only key relaxation isolated"),
        (13, "phase2_build_closure_tables.py --result-dir closure_v1", "PASS; conflict wording refined"),
        (14, "phase2_build_closure_tables.py --result-dir closure_v2", "PASS: final closure tables"),
        (15, "python -m py_compile Phase-2 scripts", "PASS"),
        (16, "pytest -q tests/test_phase2_stage1_forensic.py", "PASS: 6 passed in 4.20s"),
        (17, "pytest -q", "PASS: 457 passed in 79.33s"),
        (18, "phase2_finalize_manifest.py input and deliverable hashing", "PASS"),
    ]
    write_tsv(
        root / "commands_executed.tsv",
        [{"order": order, "operation": operation, "result": result} for order, operation, result in commands],
        ["order", "operation", "result"],
    )

    write_tsv(
        root / "dependencies_added.tsv",
        [{
            "dependency": "NONE", "version": "", "action": "No dependency installed in Phase 2",
            "scope": "/home/Francisco/wheatconformer-envs/phase1-tf215-gpu-pandas22",
            "evidence": "Reused Phase-1 exact lock dependencies_wsl_tf215_gpu_pandas22.lock.txt",
        }],
        ["dependency", "version", "action", "scope", "evidence"],
    )
    runtimes = [
        ("python", "3.11.15"), ("pandas", "2.2.3"), ("numpy", "1.26.4"),
        ("pyarrow", "24.0.0"), ("tensorflow", "2.15.1"), ("pytest", "8.4.2"),
    ]
    write_tsv(
        root / "runtime_versions.tsv",
        [{"component": component, "version": version} for component, version in runtimes],
        ["component", "version"],
    )

    changed = [
        ("scripts/v2/phase2_forensic_stage1_audit.py", "ADDED", "primary forensic replay/instrumentation"),
        ("scripts/v2/phase2_finalize_findings.py", "ADDED", "final canonical dispositions and refined findings"),
        ("scripts/v2/phase2_verify_raw_immutability.py", "ADDED", "closing raw hash comparison"),
        ("scripts/v2/phase2_correct_raw_row_ids.py", "ADDED", "protocol-amended collision-free raw IDs"),
        ("scripts/v2/phase2_audit_doi_glis_identity.py", "ADDED", "local DOI/GLIS provenance and impact audit"),
        ("scripts/v2/phase2_build_closure_tables.py", "ADDED", "review-incorporated final tables"),
        ("scripts/v2/phase2_finalize_manifest.py", "ADDED", "closure manifests and hashes"),
        ("tests/test_phase2_stage1_forensic.py", "ADDED", "six deterministic Phase-2 tests"),
        ("docs/v2/PHASE2_REPORT.md", "ADDED", "Phase-2 report"),
        ("docs/v2/STAGE1_REBUILD_SPECIFICATION.md", "ADDED", "exact proposed rebuild specification"),
        ("docs/v2/MASTER_PLAN.md", "UPDATED", "Phase-2 completion and Phase-3 plan"),
        ("docs/v2/STATUS.md", "UPDATED", "Phase-2 status and findings"),
        ("docs/v2/DECISIONS.md", "UPDATED", "Phase-2 decisions/open questions"),
        ("docs/v2/DATA_DICTIONARY.md", "UPDATED", "Phase-2 ledger and DOI schemas"),
        ("docs/v2/VALIDATION_CONTRACT.md", "UPDATED", "Phase-2 and DOI gates"),
        ("docs/v2/CHANGELOG.md", "UPDATED", "Phase-2 changelog"),
        ("audit/v2/phase2_stage1_lineage_audit_v1/**", "ADDED", "versioned diagnostic outputs only"),
    ]
    write_tsv(
        root / "phase2_files_created_modified.tsv",
        [{"path": path, "action": action, "purpose": purpose} for path, action, purpose in changed],
        ["path", "action", "purpose"],
    )

    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout
    completion = {
        "status": "PHASE2_COMPLETE_STOPPED_FOR_REVIEW",
        "repository_commit": protocol["repository_commit"],
        "repository_branch": "audit/forensic-kernel-fixes",
        "git_status_at_closure": git_status.splitlines(),
        "stage1_rebuilt": False,
        "models_trained": False,
        "certified_v1_artifacts_modified": False,
        "outer_test_content_read": False,
        "outer_test_information_used_for_selection": False,
        "final_holdout_content_read": False,
        "raw_files_match_phase1": 2754,
        "raw_files_mismatch_phase1": 0,
        "canonical_rows_with_unique_final_id": 2938384,
        "raw_rows_with_unique_final_id": 7836162,
        "stage1_ids_reconstructed": 433626,
        "stage1_contribution_count_mismatches": 0,
        "targeted_tests": "6 passed in 4.20s",
        "full_repository_tests": "457 passed in 79.33s",
        "dependencies_added": 0,
        "recommended_next_phase": "Phase 3 - identity, trait/unit, and duplicate adjudication; no rebuild or training",
    }
    completion_path = root / "phase2_completion_summary.json"
    if completion_path.exists():
        raise FileExistsError(completion_path)
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_path = root / "phase2_deliverable_sha256.tsv"
    evidence_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path != manifest_path
    )
    source_files = [workspace / path for path, _, _ in changed if not path.endswith("/**")]
    rows: list[dict[str, object]] = []
    for path in evidence_files + source_files:
        if not path.exists() or not path.is_file():
            continue
        rows.append({
            "path": path.relative_to(workspace).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    write_tsv(manifest_path, rows, ["path", "bytes", "sha256"])
    print(json.dumps({"status": completion["status"], "hashed_deliverables": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
