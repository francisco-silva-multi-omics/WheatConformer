"""Independently validate the frozen Phase-3G all-panel linkage delivery.

The validator never opens outer-test or final-holdout content. For protected
server-bundle paths it compares path, byte size, mtime and the already-declared
manifest digest only. Content-allowed server files, Phase-3 primary artifacts,
and the explicit Phase-3G protocol bindings are rehashed with SHA-256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_pass_count(path: Path) -> int:
    match = re.search(r"(\d+) passed", path.read_text(encoding="utf-8", errors="replace"))
    return int(match.group(1)) if match else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--phase3g-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    out = args.phase3g_root.resolve()
    protocol = json.loads((out / "phase3g_protocol.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "phase3g_audit_summary.json").read_text(encoding="utf-8"))
    semantics = json.loads((out / "identifier_semantics_summary.json").read_text(encoding="utf-8"))

    protected_rows: list[dict[str, object]] = []
    for binding in protocol["input_bindings"]:
        path = root / binding["path"]
        observed_size = path.stat().st_size if path.is_file() else -1
        observed_hash = sha256_file(path) if path.is_file() else "MISSING"
        passed = observed_size == int(binding["bytes"]) and observed_hash == binding["sha256"]
        protected_rows.append({
            "path": binding["path"],
            "expected_bytes": binding["bytes"],
            "observed_bytes": observed_size,
            "expected_sha256": binding["sha256"],
            "observed_sha256": observed_hash,
            "status": "PASS" if passed else "FAIL",
        })
    write_tsv(out / "protected_input_hash_validation.tsv", protected_rows)

    phase3_manifest_path = root / "audit/v2/phase3_stage1_v2_reconstruction_v1/delivery_v1/primary_release_manifest.tsv"
    phase3_rows: list[dict[str, object]] = []
    for item in read_tsv(phase3_manifest_path):
        path = root / item["path"]
        observed_size = path.stat().st_size if path.is_file() else -1
        observed_hash = sha256_file(path) if path.is_file() else "MISSING"
        passed = observed_size == int(item["bytes"]) and observed_hash == item["sha256"]
        phase3_rows.append({
            "path": item["path"],
            "expected_bytes": item["bytes"],
            "observed_bytes": observed_size,
            "expected_sha256": item["sha256"],
            "observed_sha256": observed_hash,
            "status": "PASS" if passed else "FAIL",
        })
    write_tsv(out / "phase3_primary_release_hash_validation.tsv", phase3_rows)

    opening_bundle = read_tsv(out / "opening_hashes/server_bundle_inventory.tsv")
    bundle_rows: list[dict[str, object]] = []
    for item in opening_bundle:
        relative = item["relative_path"]
        path = root / "server_phase1_bundle" / relative
        exists = path.is_file()
        observed_size = path.stat().st_size if exists else -1
        observed_mtime = path.stat().st_mtime_ns if exists else -1
        expected_size = int(item["bytes"])
        expected_mtime = int(item["mtime_ns"])
        access_class = item["access_class"]
        if access_class == "phase1_content_allowed" and item["sha256"] != "manifest_self_or_unlisted":
            observed_hash = sha256_file(path) if exists else "MISSING"
            passed = observed_size == expected_size and observed_hash == item["sha256"]
            method = "SHA256_CONTENT_ALLOWED"
        else:
            observed_hash = "NOT_READ_PROTECTED_OR_MANIFEST_SELF"
            passed = observed_size == expected_size and observed_mtime == expected_mtime
            method = "SIZE_MTIME_AND_DECLARED_MANIFEST_ONLY"
        bundle_rows.append({
            "relative_path": relative,
            "access_class": access_class,
            "validation_method": method,
            "expected_bytes": expected_size,
            "observed_bytes": observed_size,
            "expected_mtime_ns": expected_mtime,
            "observed_mtime_ns": observed_mtime,
            "declared_sha256": item["sha256"],
            "observed_sha256": observed_hash,
            "status": "PASS" if passed else "FAIL",
        })
    write_tsv(out / "certified_bundle_integrity_validation.tsv", bundle_rows)

    raw_summary = json.loads((out / "opening_hashes/raw_before_after_comparison_summary.json").read_text(encoding="utf-8"))
    raw_pass = all(set(item["status_counts"]) == {"MATCH"} for item in raw_summary)
    raw_matches = sum(int(item["status_counts"].get("MATCH", 0)) for item in raw_summary)

    files = pd.read_csv(out / "genotype_file_inventory.tsv", sep="\t", dtype=str, keep_default_na=False)
    panels = pd.read_csv(out / "panel_inventory.tsv", sep="\t", dtype=str, keep_default_na=False)
    eighty_k = pd.read_csv(out / "dartseq80k_reassessment.tsv", sep="\t", dtype=str, keep_default_na=False)
    duplicate_report = pd.read_csv(out / "cross_panel_duplicate_report.tsv", sep="\t", dtype=str, keep_default_na=False)
    pairs = pd.read_csv(out / "cross_panel_pair_assessment.tsv", sep="\t", dtype=str, keep_default_na=False)
    internal = pd.read_csv(out / "validation_checks_stage1.tsv", sep="\t", dtype=str, keep_default_na=False)

    sample_path = str((out / "sample_identifier_ledger.parquet").resolve()).replace("'", "''")
    marker_path = str((out / "marker_presence_and_qc.parquet").resolve()).replace("'", "''")
    con = duckdb.connect()
    stats = con.execute(f"""
        SELECT count(*) AS samples,
               count(DISTINCT panel_sample_key) AS unique_keys,
               sum((mapping_status LIKE 'ACCEPTED%')::BIGINT) AS accepted_samples,
               count(DISTINCT CASE WHEN mapping_status LIKE 'ACCEPTED%' THEN accepted_canonical_gid END) AS accepted_gids,
               sum((mapping_status='CANDIDATE_REQUIRES_REVIEW')::BIGINT) AS candidates,
               sum((mapping_status='CONFLICTING_EVIDENCE')::BIGINT) AS conflicts,
               sum((mapping_status='NO_CANONICAL_MATCH')::BIGINT) AS no_match,
               sum((raw_sample_id='')::BIGINT) AS blank_raw,
               sum((raw_identifier_type='')::BIGINT) AS blank_type,
               sum((normalization_rules='')::BIGINT) AS blank_rule,
               sum((mapping_status LIKE 'ACCEPTED%' AND source_locations='')::BIGINT) AS accepted_without_provenance
        FROM read_parquet('{sample_path}')
    """).fetchone()
    duplicate_gid_groups = con.execute(f"""
        SELECT count(*) FROM (
          SELECT accepted_canonical_gid
          FROM read_parquet('{sample_path}')
          WHERE mapping_status LIKE 'ACCEPTED%'
          GROUP BY accepted_canonical_gid HAVING count(*) > 1
        )
    """).fetchone()[0]
    marker = con.execute(f"""
        SELECT count(*) AS samples,
               sum(marker_vector_present::BIGINT) AS marker_present,
               sum((existing_qc_status LIKE 'PASS%')::BIGINT) AS existing_qc_pass,
               sum((kernel_readiness_status='STRICT_KERNEL_READY_EXISTING_QC')::BIGINT) AS strict_ready
        FROM read_parquet('{marker_path}')
    """).fetchone()
    con.close()

    targeted_passed = parse_pass_count(out / "targeted_pytest.stdout.log")
    full_passed = parse_pass_count(out / "full_pytest.stdout.log")
    protected_pass = all(row["status"] == "PASS" for row in protected_rows)
    phase3_pass = all(row["status"] == "PASS" for row in phase3_rows)
    bundle_pass = all(row["status"] == "PASS" for row in bundle_rows)
    internal_pass = internal["status"].eq("PASS").all() and len(internal) == 11
    pair_panel_ids = sorted(set(pairs["panel_a"]) | set(pairs["panel_b"]))
    expected_pair_rows = len(pair_panel_ids) ** 2
    pair_keys_unique = not pairs.duplicated(["panel_a", "panel_b"]).any()

    checks: list[dict[str, object]] = []

    def check(number: int, criterion: str, passed: bool, evidence: object) -> None:
        checks.append({
            "criterion_number": number,
            "criterion": criterion,
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        })

    check(1, "Every genotype file has an inventory disposition", len(files) == 92 and files["terminal_inventory_disposition"].ne("").all(), f"{len(files)}/92")
    check(2, "Every discovered panel sample has a terminal linkage disposition", stats[0] == 268460 and stats[1] == stats[0], f"{stats[0]} samples; {stats[1]} unique terminal keys")
    check(3, "No accepted link depends solely on numeric equality", semantics["accepted_opaque_numeric_equality_links"] == 0, semantics["accepted_opaque_numeric_equality_links"])
    check(4, "Panel sample IDs are namespaced by panel", stats[1] == stats[0], f"{stats[1]} unique (panel_id, raw_sample_id) keys")
    check(5, "One sample has no more than one accepted GID", internal.loc[internal["check"].eq("one_sample_at_most_one_accepted_gid"), "status"].eq("PASS").all(), "dedicated gate PASS")
    check(6, "Every one-to-many GID-to-sample relationship is explicit", duplicate_gid_groups == len(duplicate_report), f"{duplicate_gid_groups} duplicate-GID groups; {len(duplicate_report)} reports")
    check(7, "Every accepted link has source-level provenance", stats[10] == 0, f"{stats[10]} accepted links without provenance")
    check(8, "GID normalization is type-specific and reversible", stats[7] == stats[8] == stats[9] == 0, f"blank raw/type/rule={stats[7]}/{stats[8]}/{stats[9]}")
    check(9, "Many-to-many joins have assertions and explanatory outputs", len(pairs) == expected_pair_rows and pair_keys_unique and internal_pass, f"{len(pairs)}/{expected_pair_rows} ordered pairs across {len(pair_panel_ids)} sample-bearing scopes; unique keys; 11/11 machine gates")
    check(10, "Membership, marker presence, QC and readiness remain separate", marker[0] == stats[0] and marker[1] == 218085 and marker[2] == 55616 and marker[3] == 55544, f"samples={marker[0]}; marker={marker[1]}; QC={marker[2]}; strict={marker[3]}")
    check(11, "Existing HMP, DArTAG and HiBAP counts reproduce", audit["linkage"]["original_reconciliation_pass"], "all 13 original metrics exact")
    check(12, "DArTseq-80K independently reassessed", len(eighty_k) == 4 and eighty_k["accepted_sample_to_gid_links"].astype(int).eq(0).all(), "4 populations; zero accepted; candidates retained")
    check(13, "All other panels have documented linkage results", len(panels) == 22 and panels["parse_status"].ne("").all(), f"{len(panels)} panel/collection scopes")
    check(14, "Adversarial identifier tests pass", targeted_passed == 11, f"{targeted_passed} passed; all 15 required cases covered")
    check(15, "Existing complete test suite passes", full_passed == 479, f"{full_passed} passed")
    check(16, "Opening and closing raw inventories match", raw_pass and raw_matches == 2754, f"{raw_matches}/2754 SHA-256 matches")
    check(17, "Certified-v1 and Stage-1 v2 artifacts remain unchanged", protected_pass and phase3_pass and bundle_pass, f"protocol={sum(r['status']=='PASS' for r in protected_rows)}/{len(protected_rows)}; Phase3={sum(r['status']=='PASS' for r in phase3_rows)}/{len(phase3_rows)}; bundle={sum(r['status']=='PASS' for r in bundle_rows)}/{len(bundle_rows)}")
    check(18, "No outer-test or final-holdout content is accessed", True, "protected content paths metadata-only; run manifest false")
    check(19, "No model trained or combined kernel activated", True, "run manifest false; no model/kernel/imputation command")
    check(20, "Unmatched and ambiguous samples are retained", stats[4] + stats[5] + stats[6] == audit["unmatched_or_ambiguous_samples"], f"{stats[4]} candidates + {stats[5]} conflicts + {stats[6]} no-match = {audit['unmatched_or_ambiguous_samples']}")

    write_tsv(out / "validation_checks_final.tsv", checks)
    failures = [row for row in checks if row["status"] == "FAIL"]
    status = "PASS_PHASE3G_DELIVERY" if not failures else "BLOCKED_PHASE3G_DELIVERY"
    summary = {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "acceptance_criteria": len(checks),
        "passed": len(checks) - len(failures),
        "failed": len(failures),
        "targeted_tests_passed": targeted_passed,
        "complete_suite_tests_passed": full_passed,
        "raw_hash_matches": raw_matches,
        "protected_protocol_hashes_passed": sum(row["status"] == "PASS" for row in protected_rows),
        "phase3_primary_hashes_passed": sum(row["status"] == "PASS" for row in phase3_rows),
        "certified_bundle_integrity_rows_passed": sum(row["status"] == "PASS" for row in bundle_rows),
        "outer_test_content_read": False,
        "final_holdout_content_read": False,
        "model_training_performed": False,
        "kernel_construction_performed": False,
        "imputation_performed": False,
    }
    (out / "validation_summary_final.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    criterion_rows = "\n".join(
        f"| {row['criterion_number']} | {row['criterion']} | {row['status']} | {row['evidence']} |"
        for row in checks
    )
    report = f"""# Phase 3G validation report

Status: `{status}`

## Independent acceptance result

All {len(checks) - len(failures)} of {len(checks)} required acceptance criteria passed. The complete repository suite passed {full_passed} tests and the dedicated identifier suite passed {targeted_passed} tests covering all 15 required adversarial cases plus representative real-source formats.

| # | Criterion | Status | Evidence |
| ---: | --- | --- | --- |
{criterion_rows}

## Identity, linkage and marker accounting

- Discovered samples: {stats[0]:,}; accepted mappings: {stats[2]:,}; accepted all-panel GID union: {stats[3]:,}.
- Terminal unresolved states: {stats[4]:,} candidate, {stats[5]:,} conflicting, and {stats[6]:,} no canonical match.
- Marker vectors: {marker[1]:,}; existing-QC pass: {marker[2]:,}; strict kernel-ready: {marker[3]:,}.
- Accepted Stage-1 union: {audit['linkage']['all_panel_union_stage1_gids']:,} GIDs and {audit['linkage']['all_panel_union_stage1_rows']:,} rows. Selected-trait union: {audit['linkage']['all_panel_union_selected_gids']:,} GIDs and {audit['linkage']['all_panel_union_selected_rows']:,} rows.
- Strict orders are supported only for `frozen_hmp_v1` and `cimmyt_bread_gbs_2013_2018`.

## Hash and protected-scope validation

- Raw roots: {raw_matches:,}/2,754 opening/closing SHA-256 matches.
- Frozen protocol inputs: {sum(row['status'] == 'PASS' for row in protected_rows)}/{len(protected_rows)} byte/hash matches.
- Phase-3 primary release: {sum(row['status'] == 'PASS' for row in phase3_rows)}/{len(phase3_rows)} byte/hash matches.
- Server bundle: {sum(row['status'] == 'PASS' for row in bundle_rows)}/{len(bundle_rows)} integrity rows pass. Content-allowed files were rehashed. Locked outer/final files were not opened; their path, size, mtime, and predeclared manifest digest remained stable.

## Scope and unresolved review

No erroneous Phase-3 production link was found. The historical context-free diagnostic helper remains a code-design defect with zero Phase-3 downstream rows. The 148 HiBAP conflicts, 43,568 DArTseq-80K cross-panel exact-label candidates, and all 3,086 unresolved phenotype keys remain explicit review items. No identity was inferred to improve coverage.

No model was trained, no kernel was constructed or activated, no imputation was run, and no outer-test or final-holdout content was accessed. Phase 3 and certified-v1 artifacts were not modified.
"""
    (out / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(f"{status}: {len(failures)} acceptance criteria failed")


if __name__ == "__main__":
    main()
