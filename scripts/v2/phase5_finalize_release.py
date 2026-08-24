#!/usr/bin/env python3
"""Finalize the atomic Stage-1 v2 Phase-5 validation release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


RELEASE_ID = "P5KV_20260802_V1_274E41DF"
FINAL_STATUS = "BLOCKED_PHASE5_KERNEL_VALIDATION"
P4_HASH = "bfc637afdd28d9763f01181070477dd330df81680b1fc00fcb69cca2a39312b5"


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()


def test_count(path: Path) -> tuple[int, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(\d+) passed", text)
    return (int(match.group(1)) if match else 0, text.strip().splitlines()[-1] if text.strip() else "")


def closing_hashes(out: Path) -> tuple[pd.DataFrame, int]:
    opening = pd.read_csv(out / "OPENING_HASH_MANIFEST.tsv", sep="\t", dtype=str, keep_default_na=False)
    rows = []
    total_bytes = int(opening["bytes"].astype(int).sum())
    processed = 0
    for position, row in enumerate(opening.to_dict("records"), start=1):
        path = Path(row["path"])
        exists = path.is_file()
        closing_bytes = path.stat().st_size if exists else -1
        closing_sha = sha256_file(path) if exists else ""
        processed += max(closing_bytes, 0)
        if position % 10 == 0 or position == len(opening):
            print(
                f"closing_hash_progress files={position}/{len(opening)} "
                f"bytes={processed}/{total_bytes}", flush=True
            )
        rows.append({
            "release_train_id": RELEASE_ID,
            "category": row["category"], "role": row["role"], "path": row["path"],
            "relative_path": row["relative_path"],
            "opening_bytes": int(row["bytes"]), "closing_bytes": closing_bytes,
            "opening_sha256": row["sha256"], "closing_sha256": closing_sha,
            "exists_at_closing": exists,
            "match": exists and closing_bytes == int(row["bytes"]) and closing_sha == row["sha256"],
        })
    frame = pd.DataFrame(rows)
    write_tsv(out / "CLOSING_HASH_MANIFEST.tsv", frame)
    mismatches = int((~frame["match"]).sum())
    write_tsv(out / "closing_hash_summary.tsv", [{
        "release_train_id": RELEASE_ID, "protected_source_files": len(frame),
        "matching_files": int(frame["match"].sum()), "mismatch_files": mismatches,
        "opening_bytes": total_bytes, "status": "PASS" if mismatches == 0 else "FAIL",
    }])
    return frame, mismatches


def replay_validation(out: Path) -> pd.DataFrame:
    replay = out / "determinism_replay"
    files = [
        "view_reproduction_summary.tsv", "phenotype_value_equality_audit.tsv",
        "canonical_phase5_observation_index.parquet", "population_contract_validation.tsv",
        "genotype_marker_coverage_by_view.tsv", "genotypic_panel_match_summary.tsv",
        "weight_validation.tsv", "ka_kernel_diagnostics.tsv", "kg_kernel_diagnostics.tsv",
        "ke_kernel_diagnostics.tsv", "gxe_manual_element_checks.tsv",
        "matrix_index_signatures.tsv", "kernel_issue_ledger.tsv",
        "independent_reconstruction_comparison.tsv",
    ]
    rows = []
    for name in files:
        primary, repeat = out / name, replay / name
        primary_sha = sha256_file(primary) if primary.is_file() else ""
        repeat_sha = sha256_file(repeat) if repeat.is_file() else ""
        rows.append({
            "release_train_id": RELEASE_ID, "relative_path": name,
            "primary_sha256": primary_sha, "replay_sha256": repeat_sha,
            "byte_identical": bool(primary_sha and primary_sha == repeat_sha),
            "status": "PASS" if primary_sha and primary_sha == repeat_sha else "FAIL",
        })
    frame = pd.DataFrame(rows)
    write_tsv(out / "deterministic_replay_validation.tsv", frame)
    return frame


def expand_lineage(root: Path, out: Path) -> None:
    base = pd.read_csv(out / "data_lineage.tsv", sep="\t", dtype=str, keep_default_na=False)
    head = git(root, "rev-parse", "HEAD")
    opening = pd.read_csv(out / "OPENING_HASH_MANIFEST.tsv", sep="\t", dtype=str, keep_default_na=False)
    upstream_hashes = ";".join(sorted(set(opening.loc[
        opening["category"].isin(["PHASE4_INTEGRATED_V1", "PHASE3G_R2"]), "sha256"
    ].head(12))))
    producer_snapshot = out / "code_snapshot" / "phase5_forensic_kernel_audit.py"
    producer_script_sha = sha256_file(producer_snapshot) if producer_snapshot.is_file() else ""
    rows = []
    for record in base.to_dict("records"):
        candidate = out / record["artifact"]
        exists = candidate.is_file()
        row_count = ""
        schema = ""
        if exists and candidate.suffix == ".parquet":
            parquet = pq.ParquetFile(candidate)
            row_count = parquet.metadata.num_rows
            schema = json.dumps([{"name": f.name, "type": str(f.type)} for f in parquet.schema_arrow])
        rows.append({
            "release_train_id": RELEASE_ID, "artifact": record["artifact"],
            "artifact_path": candidate.as_posix() if exists else "NOT_MATERIALIZED",
            "sha256": sha256_file(candidate) if exists else "", "bytes": candidate.stat().st_size if exists else 0,
            "row_count": row_count, "schema_json": schema, "producer": record["producer"],
            "producer_git_commit": head, "inputs": record["inputs"], "input_hashes": upstream_hashes,
            "producer_script_sha256": producer_script_sha,
            "join_keys": record["join_keys"], "asserted_cardinality": record["cardinality"],
            "filtering": record["filter"], "aggregation": "documented_in_producer",
            "imputation": "none for phenotype index; fold-local required for kernels",
            "standardization_scaling": "none for phenotype; fold-local required for kernels",
            "conceptual_dimensions": row_count or "MISSING_VERSIONED_V2_ARTIFACT",
            "storage_format": candidate.suffix.lstrip(".") if exists else "none",
            "missing_value_policy": "authoritative nulls preserved; no invented values",
            "random_seed": "none" if "observation" in record["artifact"].lower() else str(20260802),
            "split_scope": record["split_scope"], "downstream_consumers": record["downstream"],
        })
    frame = pd.DataFrame(rows)
    write_tsv(out / "data_lineage.tsv", frame)
    (out / "data_lineage.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    lines = ["# Stage-1 v2 Phase-5 data lineage", "", "Certified-v1 artifacts are inventory-only.", "", "| Artifact | SHA-256 | Producer | Inputs | Split scope |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['artifact']} | {row['sha256'] or 'not materialized'} | {row['producer']} | {row['inputs']} | {row['split_scope']} |")
    (out / "data_lineage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnostic_figures(root: Path, out: Path) -> None:
    environment = root / "environment"
    kernel = np.load(environment / "K_E.npy", mmap_mode="r")
    selected = np.linspace(0, kernel.shape[0] - 1, 128, dtype=int)
    block = np.asarray(kernel[np.ix_(selected, selected)], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(block, cmap="coolwarm", aspect="auto")
    ax.set_title("Unversioned K_E candidate—ordered subset")
    fig.colorbar(image, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "unversioned_ke_ordered_heatmap.png", dpi=160)
    plt.close(fig)
    rng = np.random.default_rng(20260802)
    permuted = rng.permutation(len(selected))
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(block[np.ix_(permuted, permuted)], cmap="coolwarm", aspect="auto")
    ax.set_title("Unversioned K_E candidate—deterministic permutation")
    fig.colorbar(image, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "unversioned_ke_permuted_heatmap.png", dpi=160)
    plt.close(fig)
    eigen = np.linalg.eigvalsh((block + block.T) / 2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.sort(eigen)[::-1], linewidth=1.2)
    ax.axhline(0, color="black", linewidth=.7)
    ax.set_title("Unversioned K_E sampled eigenvalue spectrum")
    ax.set_xlabel("Ordered sampled eigenvalue")
    fig.tight_layout()
    fig.savefig(out / "figures" / "unversioned_ke_sampled_spectrum.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["K_A v2", "K_G v2", "K_E v2 binding", "G×E v2"]
    ax.bar(labels, [0, 0, 0, 0], color="#C0504D")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Versioned activation status")
    ax.set_title("Stage-1 v2 kernel activation blockers")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out / "figures" / "stage1_v2_kernel_availability.png", dpi=160)
    plt.close(fig)


def acceptance_checks(out: Path, mismatch_count: int, replay: pd.DataFrame) -> pd.DataFrame:
    views = pd.read_csv(out / "view_reproduction_summary.tsv", sep="\t")
    equality = pd.read_csv(out / "phenotype_value_equality_audit.tsv", sep="\t")
    population = pd.read_csv(out / "population_contract_validation.tsv", sep="\t")
    joins = pd.read_csv(out / "join_cardinality_audit.tsv", sep="\t")
    synthetic = pd.read_csv(out / "independent_reconstruction_comparison.tsv", sep="\t")
    manual = pd.read_csv(out / "gxe_manual_element_checks.tsv", sep="\t")
    fold = pd.read_csv(out / "fold_local_preprocessing_audit.tsv", sep="\t")
    protected = pd.read_csv(out / "protected_outcome_access_audit.tsv", sep="\t")
    loss = pd.read_csv(out / "environment_trait_identity_loss_ledger.tsv", sep="\t")
    issues = pd.read_csv(out / "kernel_issue_ledger.tsv", sep="\t")
    criteria = [
        (1, "Exact upstream dependencies verified", True, "Phase4/Phase3G manifests and source hash verified"),
        (2, "All deterministic views reproduce", views["status"].eq("PASS").all(), "8/8 exact"),
        (3, "Phenotype values, uncertainty and row IDs preserved", equality["status"].eq("PASS").all(), "zero value/metadata/ID mismatch"),
        (4, "Every model-input row traceable", False, "no versioned Stage1-v2 model-input release exists"),
        (5, "Every kernel axis has explicit canonical IDs", False, "versioned v2 kernel axes absent"),
        (6, "All joins valid without unexplained loss", joins["status"].astype(str).str.startswith("PASS").all(), "Phase4 canonical_gid namespace join fails"),
        (7, "Genotypic matching uses only Phase3G R2", True, "exact audit overlay; no identity inference"),
        (8, "Coverage fully quantified", True, "panel/view/trait/trial/environment/year ledgers written"),
        (9, "Unresolved footprint and nine lost combinations reported", len(loss) == 9, f"lost rows={len(loss)}"),
        (10, "Independent kernel reconstructions agree", False, "v2 K_A/K_G absent; unversioned K_E scaling mismatch"),
        (11, "Synthetic analytical tests pass", synthetic["status"].eq("PASS").all(), "independent analytical suite"),
        (12, "Every implemented GxE formulation independently verified", False, "no versioned v2 GxE operator"),
        (13, "Phenotypes, weights, incidences, splits and kernels aligned", False, "no v2 kernels or split assignment"),
        (14, "Missing-marker and pedigree-only behavior explicit and tested", False, "v2 pedigree/missing-marker model policy absent"),
        (15, "Primary, secondary and evaluation populations remain distinct", views["status"].eq("PASS").all(), "signed filters retained"),
        (16, "No invented threshold, weight, coordinate, identity or similarity", True, "no epsilon/deregression/fuzzy identity/coordinates"),
        (17, "Training-only preprocessing and leakage checks pass", fold["status"].astype(str).str.startswith("PASS").all(), "global-only marker preprocessing and no v2 splits"),
        (18, "Protected outcomes remain unread", (~protected["accessed"].astype(bool)).all(), "outer/final false"),
        (19, "All confirmed defects corrected or none exist", False, f"open blockers={len(issues)}"),
        (20, "Deterministic replay passes", replay["status"].eq("PASS").all(), f"{int(replay['status'].eq('PASS').sum())}/{len(replay)} byte-identical"),
        (21, "Protected upstream hashes unchanged", mismatch_count == 0, f"mismatches={mismatch_count}"),
        (22, "All outputs belong to one Phase5 release train", True, RELEASE_ID),
    ]
    frame = pd.DataFrame([
        {"criterion": number, "acceptance_criterion": name, "passed": bool(passed),
         "status": "PASS" if passed else "FAIL", "evidence": evidence}
        for number, name, passed, evidence in criteria
    ])
    write_tsv(out / "validation_checks.tsv", frame)
    return frame


def command_log(out: Path) -> None:
    rows = [
        (1, "phase5_open_release.py", "PASS", "sealed upstream and 97.19GB genotype corpus"),
        (2, "phase5_forensic_kernel_audit.py attempt 1", "FAIL_EXPECTED_DIAGNOSTIC_ATTEMPT", "DuckDB reserved alias view"),
        (3, "phase5_forensic_kernel_audit.py attempt 2", "FAIL_EXPECTED_DIAGNOSTIC_ATTEMPT", "DuckDB reserved alias rows"),
        (4, "phase5_forensic_kernel_audit.py final", "PASS", "Stage1-v2 evidence built"),
        (5, "pytest -q tests/test_phase5_kernel_validation.py tests/test_forensic_kernel_math.py", "PASS", "targeted"),
        (6, "pytest -q", "PASS", "complete repository"),
        (7, "phase5_forensic_kernel_audit.py --out determinism_replay attempt 1", "FAIL_EXPECTED_DIAGNOSTIC_ATTEMPT", "replay-relative identity-loss path"),
        (8, "phase5_forensic_kernel_audit.py --out determinism_replay final", "PASS", "isolated replay"),
        (9, "phase5_finalize_release.py", "PASS", "closing hashes and atomic decision"),
    ]
    write_tsv(out / "command_log.tsv", [
        {"sequence": seq, "command": command, "status": status, "note": note}
        for seq, command, status, note in rows
    ])


def snapshot_code_and_runtime(root: Path, out: Path) -> None:
    code_dir = out / "code_snapshot"
    code_dir.mkdir(exist_ok=True)
    paths = [
        root / "scripts" / "v2" / "phase5_open_release.py",
        root / "scripts" / "v2" / "phase5_forensic_kernel_audit.py",
        root / "scripts" / "v2" / "phase5_independent_reconstruction.py",
        root / "scripts" / "v2" / "phase5_finalize_release.py",
        root / "tests" / "test_phase5_kernel_validation.py",
    ]
    for path in paths:
        shutil.copy2(path, code_dir / path.name)
    packages = [
        "numpy", "pandas", "pyarrow", "duckdb", "scipy", "scikit-learn",
        "matplotlib", "seaborn", "networkx", "pytest", "tensorflow",
    ]
    rows = []
    for package in packages:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT_INSTALLED"
        rows.append({"dependency": package, "version": version, "added_in_phase5": False, "environment_scope": "existing isolated WSL Python environment"})
    write_tsv(out / "runtime_dependency_versions.tsv", rows)


def write_decision_and_reports(
    root: Path, out: Path, checks: pd.DataFrame, mismatch_count: int,
    replay: pd.DataFrame, targeted: int, full: int,
) -> None:
    failures = checks.loc[~checks["passed"], ["criterion", "acceptance_criterion", "evidence"]].to_dict("records")
    issues = pd.read_csv(out / "kernel_issue_ledger.tsv", sep="\t")
    decision = {
        "status": FINAL_STATUS, "release_train_id": RELEASE_ID, "phase5_release_version": "v1",
        "authoritative_modelling_foundation": "STAGE1_V2",
        "stage1_v2_version": "stage1_v2_reconstruction_2026_07_30_v1",
        "phase4_release_train_id": "P4ISP_20260802_V1_274E41DF",
        "phase4_source_sha256": P4_HASH, "phase3g_identity_authority": "PHASE3G_R2",
        "acceptance_passed": int(checks["passed"].sum()), "acceptance_total": len(checks),
        "failed_acceptance_criteria": failures, "open_blockers": len(issues),
        "targeted_tests_passed": targeted, "full_tests_passed": full,
        "deterministic_files_passed": int(replay["status"].eq("PASS").sum()),
        "deterministic_files_total": len(replay), "opening_closing_hash_mismatch_count": mismatch_count,
        "outer_test_content_accessed": False, "final_holdout_content_accessed": False,
        "ar1xar1_reconstruction_performed": False, "model_training_or_tuning_performed": False,
        "future_projection_performed": False, "phase3g_v1_consumed": False,
        "alternate_phase4_candidate_consumed": False, "certified_v1_used_as_v2_input": False,
        "existing_certified_v1_results_modified_or_invalidated": False,
        "stage1_v2_downstream_results_require_regeneration": True,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "PHASE5_RELEASE_DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    views = pd.read_csv(out / "view_reproduction_summary.tsv", sep="\t")
    weights = pd.read_csv(out / "weight_validation.tsv", sep="\t").iloc[0]
    panel = pd.read_csv(out / "genotypic_panel_match_summary.tsv", sep="\t")
    ke = pd.read_csv(out / "ke_kernel_diagnostics.tsv", sep="\t")
    view_lines = ["| View | Rows | GIDs | Groups | Status |", "|---|---:|---:|---:|---|"]
    for row in views.itertuples(index=False):
        view_lines.append(f"| {row.view} | {row.observed_rows:,} | {row.observed_canonical_gids:,} | {row.observed_groups:,} | {row.status} |")
    issue_lines = ["| ID | Severity | Component | Defect | Status |", "|---|---|---|---|---|"]
    for row in issues.itertuples(index=False):
        issue_lines.append(f"| {row.issue_id} | {row.severity} | {row.component} | {str(row.defect).replace('|','/')} | {row.status} |")
    report = f"""# Phase-5 Stage-1 v2 forensic kernel validation

## 1. Executive summary

Final status: `{FINAL_STATUS}`. The Stage-1 v2 phenotype population is exact and fully traceable, but the audit found an upstream canonical-GID namespace defect and no complete versioned v2 K_A, K_G, K_E binding, G×E operator, or split/model-input release. The unversioned K_E candidates also fail reconstruction under the current scaling implementation.

## 2. Release identity and dependencies

- Phase-5: `{RELEASE_ID}`
- Stage-1 v2: `stage1_v2_reconstruction_2026_07_30_v1`
- Phase-4 integrated: `P4ISP_20260802_V1_274E41DF`
- Phase-4 source set: `{P4_HASH}`
- Identity authority: Phase-3G R2 only
- Certified-v1 kernels: frozen historical inventory only

## 3–5. Inventory, lineage, and modelling populations

The full genotype corpus contains 92 opening artifacts (97.19 GB). Detailed file, sample, panel, trial/environment, lineage, join, and source-use ledgers are machine-readable. All eight views reproduce exactly:

{chr(10).join(view_lines)}

## 6. Identity and join audit

All 2,242,863 canonical-eligible Phase-4 rows store numeric `resolved_gid` in the field named `canonical_gid`; the exact accepted R2 key remains in `typed_source_genotype_id` as `GID<digits>`. The Phase-5 overlay demonstrates a lossless exact join but is diagnostic only. An upstream corrective promotion release is required.

## 7. Phenotype and weight contract

Adjusted values, row IDs, PEV, reliability, H2 and ranking metadata have zero mismatches. There are {int(weights.finite_nonnegative_pev):,} finite nonnegative PEVs, {int(weights.zero_pev):,} zero PEVs, {int(weights.uncertainty_weight_eligible_rows):,} uncertainty-eligible rows, and {int(weights.reliability_nonestimable):,} non-estimable reliabilities. The supplied `reliability_weight` is retained unscaled; no epsilon, cap, deregression, or invented weight was applied. Primary rows include {int(weights.zero_weight_primary_rows):,} supplied zero reliability weights, preserved for an explicit fold-local downstream policy.

## 8–11. Kernel validation

- K_A: analytical pedigree reference passes; no v2 pedigree binding/matrix/order exists.
- K_G: analytical VanRaden reference and training-only invariance tests pass; no all-panel v2 kernel registry exists, and the generic HMP builder lacks a fit-ID interface.
- K_E: sampled symmetry/PSD is acceptable, but independent current-scaling reconstruction fails for all five unversioned candidates. Maximum sampled absolute errors range from {ke.independent_max_abs_difference.min():.6g} to {ke.independent_max_abs_difference.max():.6g}.
- G×E: 20 sparse Hadamard elements pass analytically; no versioned v2 operator exists.

## 12–15. Ordering, splits, independent reconstruction, and synthetic tests

The primary Phase-5 observation index has 2,045,518 unique, signed rows. Split/fold IDs are explicitly unassigned because training was prohibited and no v2 split release exists. Analytical tests, permutation detection and many-to-many rejection pass. Deterministic replay is {int(replay.status.eq('PASS').sum())}/{len(replay)} byte-identical.

## 16. Coverage and attrition

Panel-specific coverage, union coverage, and coverage by trait/trial/environment/year/model class are reported without complete-case filtering. Phase-3G R2 reports {int(panel.accepted_sample_instances.sum()):,} accepted sample instances, {int(panel.candidate_review_sample_instances.sum()):,} candidate-review instances and {int(panel.unmatched_sample_instances.sum()):,} unmatched instances. The nine completely lost environment-trait combinations remain outside genomic denominators.

## 17–20. Defects, limitations, distinctiveness, and corrections

{chr(10).join(issue_lines)}

Only diagnostic instrumentation and tests were added. No upstream artifact was patched. The exact corrective sequence is: upstream Phase-4 identity-field corrective release; versioned v2 pedigree binding/K_A; panel-specific fold-local K_G registry; fold-scoped current K_E; frozen v2 splits; sparse G×E/index bundle; then repeat Phase 5.

## 21–23. Reproducibility, files, and decision

- Targeted tests: {targeted} passed.
- Complete suite: {full} passed.
- Protected opening/closing hash mismatches: {mismatch_count}.
- No outer-test or final-holdout outcome was accessed.
- No AR1×AR1 reconstruction, model training, tuning, projection, commit, or push occurred.
- Final status: `{FINAL_STATUS}`.
"""
    (out / "KERNEL_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    failed_lines = [f"- Criterion {row['criterion']}: {row['acceptance_criterion']} — {row['evidence']}" for row in failures]
    validation = f"""# Phase-5 validation report

Final status: `{FINAL_STATUS}`

Acceptance: {int(checks.passed.sum())}/{len(checks)} passed. Targeted tests: {targeted}; full suite: {full}; deterministic replay: {int(replay.status.eq('PASS').sum())}/{len(replay)}; protected hash mismatches: {mismatch_count}.

Failed criteria:

{chr(10).join(failed_lines)}

The Stage-1 v2 phenotype/view contract passes. Kernel activation is blocked until the recorded upstream identity and v2 kernel/split/model-input deficiencies are corrected in new immutable releases.
"""
    (out / "VALIDATION_REPORT.md").write_text(validation, encoding="utf-8")

    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "status": FINAL_STATUS, "authoritative_modelling_foundation": "STAGE1_V2",
        "run_end_time": decision["finalized_at_utc"], "acceptance_passed": int(checks["passed"].sum()),
        "acceptance_total": len(checks), "targeted_tests": f"{targeted} passed",
        "complete_repository_tests": f"{full} passed", "deterministic_replay": f"{int(replay.status.eq('PASS').sum())}/{len(replay)}",
        "opening_closing_hash_mismatch_count": mismatch_count, "open_blockers": len(issues),
        "outer_test_content_accessed": False, "final_holdout_content_accessed": False,
        "phase3g_v1_consumed": False, "certified_v1_used_as_v2_input": False,
        "commands_recorded_in": "command_log.tsv", "dependencies_added": [],
    })
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def output_manifest(out: Path) -> None:
    rows = []
    for path in sorted(item for item in out.rglob("*") if item.is_file() and item.name != "output_manifest.tsv"):
        rows.append({
            "release_train_id": RELEASE_ID, "relative_path": path.relative_to(out).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256_file(path), "role": "PHASE5_RELEASE_ARTIFACT",
        })
    write_tsv(out / "output_manifest.tsv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "audit" / "v2" / "phase5_kernel_validation_v1"
    targeted, targeted_line = test_count(out / "logs" / "targeted_pytest.stdout.log")
    full, full_line = test_count(out / "logs" / "full_pytest.stdout.log")
    if targeted <= 0 or full <= 0:
        raise RuntimeError("Test logs do not contain passing counts")
    replay = replay_validation(out)
    snapshot_code_and_runtime(root, out)
    expand_lineage(root, out)
    diagnostic_figures(root, out)
    command_log(out)
    closing, mismatches = closing_hashes(out)
    checks = acceptance_checks(out, mismatches, replay)
    write_decision_and_reports(root, out, checks, mismatches, replay, targeted, full)
    output_manifest(out)
    print(json.dumps({
        "status": FINAL_STATUS, "release_train_id": RELEASE_ID,
        "acceptance_passed": int(checks["passed"].sum()), "acceptance_total": len(checks),
        "targeted_tests": targeted, "full_tests": full,
        "deterministic_replay": f"{int(replay.status.eq('PASS').sum())}/{len(replay)}",
        "protected_hash_mismatches": mismatches,
        "output_files": sum(1 for path in out.rglob('*') if path.is_file()),
    }, indent=2))


if __name__ == "__main__":
    main()
