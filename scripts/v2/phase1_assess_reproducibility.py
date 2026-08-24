"""Static reproducibility assessment for the frozen certified-v1 pipeline.

Protected result files are never opened. Their names and declared hashes may be
inventoried, but availability does not authorize content access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path, *, canonical_lf: bool = False) -> str:
    data = path.read_bytes()
    if canonical_lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def relative_server_path(value: str) -> str:
    marker = "/genotipoXambiente/"
    if marker in value:
        return value.split(marker, 1)[1]
    code_marker = "/WheatConformer/"
    if code_marker in value:
        return value.split(code_marker, 1)[1]
    return value.lstrip("/")


def protected_access(relative: str) -> bool:
    lowered = relative.lower()
    return any(
        token in lowered
        for token in (
            "trained_models/",
            "/reporting/",
            "outer_fold_metrics",
            "outer_fold_summary",
            "final_holdout",
            "nested_evaluation_entities",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    bundle = args.bundle_root.resolve()
    artifacts = bundle / "artifacts"
    out_dir = args.out_dir.resolve()

    protocol_path = root / "server_training_pipeline/reaction_norm_routed_hierarchy_outer_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_protocol_hash = "251ab22231e7a8c7f3cfb5bfd8721b7e2054057a21ac77109813b8aec640ab9b"
    protocol_hash = sha256(protocol_path, canonical_lf=True)

    source_map = {
        "hierarchy_trainer_sha256": root / "server_training_pipeline/train_multitrait_reaction_norm_trial_hierarchy_tf.py",
        "base_trainer_sha256": root / "server_training_pipeline/train_multitrait_reaction_norm_tf.py",
        "factorization_sha256": root / "server_training_pipeline/kernel_factorization.py",
        "run_verifier_sha256": root / "server_training_pipeline/verify_reaction_norm_run.py",
        "outer_verifier_sha256": root / "server_training_pipeline/verify_reaction_norm_outer_evaluation.py",
    }
    checks: list[dict[str, object]] = []
    checks.append({
        "check": "frozen_protocol_sha256",
        "status": "PASS" if protocol_hash == expected_protocol_hash else "FAIL",
        "detail": f"expected={expected_protocol_hash}; observed={protocol_hash}",
        "evidence": str(protocol_path),
    })
    for key, path in source_map.items():
        expected = protocol["implementation"][key]
        observed = sha256(path, canonical_lf=True)
        checks.append({
            "check": f"implementation_{key}",
            "status": "PASS" if observed == expected else "FAIL",
            "detail": f"expected={expected}; observed={observed}",
            "evidence": str(path),
        })

    completed_dir = artifacts / "audit/reaction_norm_routed_hierarchy_outer_v1/completed"
    completed_commit = (completed_dir / "code_commit.txt").read_text(encoding="utf-8").strip()
    try:
        commit_type = git_output(root, "cat-file", "-t", completed_commit)
        commit_exists = commit_type == "commit"
    except subprocess.CalledProcessError:
        commit_exists = False
    checks.append({
        "check": "completed_code_commit_available",
        "status": "PASS" if commit_exists else "FAIL",
        "detail": completed_commit,
        "evidence": str(completed_dir / "code_commit.txt"),
    })
    if commit_exists:
        diff = git_output(root, "diff", "--name-only", f"{completed_commit}..HEAD", "--", *[str(p.relative_to(root)) for p in source_map.values()])
        checks.append({
            "check": "completed_implementation_sources_unchanged_at_head",
            "status": "PASS" if not diff else "FAIL",
            "detail": diff or "no relevant source differences",
            "evidence": f"git diff {completed_commit}..HEAD",
        })

    completed_rows: list[dict[str, object]] = []
    for line in (completed_dir / "artifacts.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected_hash, server_path = line.split(maxsplit=1)
        relative = relative_server_path(server_path)
        is_protected = protected_access(relative)
        local_path = root / relative
        bundle_path = artifacts / relative
        if local_path.is_file():
            available_at = "repository"
            candidate = local_path
        elif bundle_path.is_file():
            available_at = "bundle"
            candidate = bundle_path
        else:
            available_at = "absent"
            candidate = None
        if is_protected:
            observed_hash = "NOT_READ_PROTECTED"
            hash_status = "NOT_READ_PROTECTED" if candidate else "ABSENT_PROTECTED"
        elif candidate is None:
            observed_hash = ""
            hash_status = "MISSING"
        else:
            observed_hash = sha256(candidate, canonical_lf=False)
            hash_status = "MATCH" if observed_hash == expected_hash else "MISMATCH"
        completed_rows.append({
            "relative_path": relative,
            "expected_sha256": expected_hash,
            "availability": available_at,
            "access_class": "protected_name_hash_only" if is_protected else "safe_integrity_check",
            "observed_sha256": observed_hash,
            "hash_status": hash_status,
        })

    safe_manifest_rows = [row for row in completed_rows if row["access_class"] == "safe_integrity_check"]
    checks.append({
        "check": "completed_manifest_safe_entries_available_and_matching",
        "status": "PASS" if all(row["hash_status"] == "MATCH" for row in safe_manifest_rows) else "INCOMPLETE",
        "detail": f"matching={sum(row['hash_status'] == 'MATCH' for row in safe_manifest_rows)}/{len(safe_manifest_rows)} safe entries",
        "evidence": str(completed_dir / "artifacts.sha256"),
    })
    checks.append({
        "check": "completed_protected_results_locally_verified",
        "status": "NOT_RUN_PROTECTED",
        "detail": "Protected reporting and trained-model outputs were inventoried by declared path/hash only.",
        "evidence": str(completed_dir / "artifacts.sha256"),
    })

    certification_path = artifacts / "model_kernels/multitrait_pedigree_env_uniform_tgw_certified/certification/multitrait_kernel_certification_summary.json"
    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    kernel_rows: list[dict[str, object]] = []
    for kernel, identity in certification["kernel_identities"].items():
        relative = relative_server_path(identity["path"])
        candidate = artifacts / relative
        kernel_rows.append({
            "kernel": kernel,
            "relative_path": relative,
            "bytes": identity["bytes"],
            "expected_sha256": identity["sha256"],
            "supplied_in_bundle": candidate.is_file(),
            "integrity_status": "NOT_SUPPLIED" if not candidate.is_file() else ("MATCH" if sha256(candidate) == identity["sha256"] else "MISMATCH"),
        })
    checks.append({
        "check": "certified_kernel_bytes_supplied_for_reproduction",
        "status": "PASS" if all(row["supplied_in_bundle"] for row in kernel_rows) else "INCOMPLETE",
        "detail": f"supplied={sum(bool(row['supplied_in_bundle']) for row in kernel_rows)}/{len(kernel_rows)} certified kernels",
        "evidence": str(certification_path),
    })

    certified_lineage_path = artifacts / "model_kernels/multitrait_pedigree_env_uniform_tgw_certified/multitrait_pedigree_uniform_tgw_certified_lineage.json"
    recovered_lineage_path = artifacts / "model_kernels/multitrait_stage1_recovered_v1/multitrait_stage1_recovered_v1_lineage.json"
    certified_lineage = json.loads(certified_lineage_path.read_text(encoding="utf-8"))
    recovered_lineage = json.loads(recovered_lineage_path.read_text(encoding="utf-8"))
    checks.extend([
        {
            "check": "certified_ledger_builder_commit_recorded",
            "status": "INCOMPLETE" if certified_lineage.get("git_commit") == "unknown" else "PASS",
            "detail": str(certified_lineage.get("git_commit", "missing")),
            "evidence": str(certified_lineage_path),
        },
        {
            "check": "recovered_ledger_builder_commit_recorded",
            "status": "INCOMPLETE" if recovered_lineage.get("git_commit") == "unknown" else "PASS",
            "detail": str(recovered_lineage.get("git_commit", "missing")),
            "evidence": str(recovered_lineage_path),
        },
        {
            "check": "run_bound_dependency_lock_supplied",
            "status": "INCOMPLETE",
            "detail": "Server environment snapshot exists, but no dependency lock is bound by hash to the completed certified-v1 manifest.",
            "evidence": str(bundle / "provenance/server_environment.txt"),
        },
        {
            "check": "full_certified_v1_byte_for_byte_reproduction_from_bundle",
            "status": "INCOMPLETE",
            "detail": "Certified kernel bytes, trained models, protected result bytes, and exact run-bound dependency provenance are not all supplied.",
            "evidence": str(bundle),
        },
    ])

    write_tsv(
        out_dir / "reproducibility_checks.tsv",
        checks,
        ["check", "status", "detail", "evidence"],
    )
    write_tsv(
        out_dir / "certified_v1_completed_manifest_inventory.tsv",
        completed_rows,
        ["relative_path", "expected_sha256", "availability", "access_class", "observed_sha256", "hash_status"],
    )
    write_tsv(
        out_dir / "certified_kernel_artifact_availability.tsv",
        kernel_rows,
        ["kernel", "relative_path", "bytes", "expected_sha256", "supplied_in_bundle", "integrity_status"],
    )
    report = {
        "conclusion": "PARTIALLY_REPRODUCIBLE_STATIC_CONTRACT_FULL_BYTE_REPRODUCTION_NOT_DEMONSTRABLE_FROM_BUNDLE",
        "protocol_sha256": protocol_hash,
        "completed_code_commit": completed_commit,
        "protected_artifacts_read": False,
        "checks": {row["check"]: row["status"] for row in checks},
    }
    report_path = out_dir / "reproducibility_assessment.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
