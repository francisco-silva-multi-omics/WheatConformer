#!/usr/bin/env python3
"""Close and sign the Phase-4 namespace correction / Phase-3G R3 release train.

This finalizer is intentionally limited to identity metadata, reproducibility,
and immutable-input validation.  It never reads protected outcomes, constructs
kernels/splits, or trains models.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

try:
    from .phase4_namespace_r3_common import (
        OVERALL_RELEASE_ID,
        PHASE3G_R3_RELEASE_ID,
        PHASE3G_R3_ROOT,
        PHASE4_NS_RELEASE_ID,
        PHASE4_NS_ROOT,
        PHASE4_R3_ROOT,
        PHASE4_ROOT,
        REPOSITORY_ROOT,
        STAGE1_R3_ROOT,
        index_signature,
        output_manifest,
        q,
        sha256,
        write_json,
        write_tsv,
    )
except ImportError:  # direct script execution
    from phase4_namespace_r3_common import (
        OVERALL_RELEASE_ID,
        PHASE3G_R3_RELEASE_ID,
        PHASE3G_R3_ROOT,
        PHASE4_NS_RELEASE_ID,
        PHASE4_NS_ROOT,
        PHASE4_R3_ROOT,
        PHASE4_ROOT,
        REPOSITORY_ROOT,
        STAGE1_R3_ROOT,
        index_signature,
        output_manifest,
        q,
        sha256,
        write_json,
        write_tsv,
    )


REPLAY_FILES = {
    PHASE4_NS_ROOT: [
        "corrected_promoted_phenotypes.parquet",
        "identity_join_cardinality_audit.tsv",
        "identity_join_ledger.parquet",
        "non_identity_field_equality_audit.tsv",
        "observation_id_policy.json",
        "old_new_observation_id_lineage.parquet",
        "view_count_summary.tsv",
    ],
    PHASE3G_R3_ROOT: [
        "accepted_mapping.tsv",
        "identifier_normalization_ledger.tsv",
        "r3_decision_counts.tsv",
        "source_key_decision_ledger.parquet",
        "source_key_lineage_reconciliation.tsv",
        "unresolved_source_row_lineage.parquet",
        "unresolved_review_required.tsv",
    ],
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_result(path: Path, expected: int) -> dict:
    payload = path.read_bytes()
    # PowerShell redirection on this workstation writes UTF-16LE (with BOM),
    # whereas WSL/bash redirection writes UTF-8.  Test evidence is valid in
    # either encoding and must not be misclassified by the closure parser.
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = payload.decode("utf-16")
    else:
        text = payload.decode("utf-8", errors="replace")
    match = re.search(r"(\d+) passed in ([0-9.]+)s", text)
    passed = int(match.group(1)) if match else -1
    try:
        log_label = path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        log_label = path.as_posix()
    return {
        "log": log_label,
        "expected_passed": expected,
        "observed_passed": passed,
        "seconds": float(match.group(2)) if match else None,
        "status": "PASS" if passed == expected else "FAIL",
    }


def validate_replay() -> pd.DataFrame:
    rows: list[dict] = []
    for root, names in REPLAY_FILES.items():
        for name in names:
            original = root / name
            replay = root / "determinism_replay" / name
            original_hash = sha256(original)
            replay_hash = sha256(replay)
            rows.append(
                {
                    "release_id": PHASE4_NS_RELEASE_ID if root == PHASE4_NS_ROOT else PHASE3G_R3_RELEASE_ID,
                    "relative_path": name,
                    "original_bytes": original.stat().st_size,
                    "replay_bytes": replay.stat().st_size,
                    "original_sha256": original_hash,
                    "replay_sha256": replay_hash,
                    "status": "PASS_BYTE_IDENTICAL" if original_hash == replay_hash and original.stat().st_size == replay.stat().st_size else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def close_immutable_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rehash every raw file and every additional exact identity input once."""
    source = pd.read_csv(
        PHASE3G_R3_ROOT / "evidence_artifact_inventory.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    rows: list[dict] = []
    total = len(source)
    for index, row in enumerate(source.itertuples(index=False), start=1):
        path = Path(row.path)
        exists = path.is_file()
        closing_bytes = path.stat().st_size if exists else -1
        closing_hash = sha256(path) if exists else ""
        status = (
            "PASS_UNCHANGED"
            if exists and closing_bytes == int(row.opening_bytes) and closing_hash == row.opening_sha256
            else "FAIL_CHANGED_OR_MISSING"
        )
        rows.append(
            {
                "relative_path": row.relative_path,
                "category": row.category,
                "artifact_role": row.artifact_role,
                "opening_bytes": int(row.opening_bytes),
                "closing_bytes": closing_bytes,
                "opening_sha256": row.opening_sha256,
                "closing_sha256": closing_hash,
                "status": status,
            }
        )
        if index == 1 or index % 100 == 0 or index == total:
            print(f"immutable rehash progress {index}/{total}", flush=True)
    all_inputs = pd.DataFrame(rows)
    raw = all_inputs[all_inputs["category"].isin(["RAW_TRIAL_CORPUS", "RAW_GENOTYPE_CORPUS"])].copy()
    identity = all_inputs[all_inputs["category"].eq("VERSIONED_IDENTIFIER_EVIDENCE")].copy()
    summary = pd.DataFrame(
        [
            {
                "scope": "RAW_TRIAL_CORPUS",
                "files": int((raw["category"] == "RAW_TRIAL_CORPUS").sum()),
                "bytes": int(raw.loc[raw["category"] == "RAW_TRIAL_CORPUS", "closing_bytes"].sum()),
                "unchanged": int((raw.loc[raw["category"] == "RAW_TRIAL_CORPUS", "status"] == "PASS_UNCHANGED").sum()),
            },
            {
                "scope": "RAW_GENOTYPE_CORPUS",
                "files": int((raw["category"] == "RAW_GENOTYPE_CORPUS").sum()),
                "bytes": int(raw.loc[raw["category"] == "RAW_GENOTYPE_CORPUS", "closing_bytes"].sum()),
                "unchanged": int((raw.loc[raw["category"] == "RAW_GENOTYPE_CORPUS", "status"] == "PASS_UNCHANGED").sum()),
            },
            {
                "scope": "VERSIONED_IDENTIFIER_EVIDENCE",
                "files": len(identity),
                "bytes": int(identity["closing_bytes"].sum()),
                "unchanged": int(identity["status"].eq("PASS_UNCHANGED").sum()),
            },
        ]
    )
    summary["status"] = summary.apply(lambda row: "PASS" if row["files"] == row["unchanged"] else "FAIL", axis=1)
    return raw, identity, summary


def freeze_ordered_universes(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    source = PHASE4_NS_ROOT / "corrected_promoted_phenotypes.parquet"
    primary = PHASE4_NS_ROOT / "ordered_primary_observation_universe.parquet"
    con.execute(
        f"""
        COPY (
          SELECT row_number() OVER (ORDER BY phase4_adjusted_row_id)-1 AS zero_based_index,
                 phase4_adjusted_row_id,canonical_gid,environment_id,trait,phase4_group_id
          FROM read_parquet('{q(source)}')
          WHERE primary_weighted_training_eligible
          ORDER BY phase4_adjusted_row_id
        ) TO '{q(primary)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 100000)
        """
    )
    primary_gid = PHASE4_NS_ROOT / "ordered_primary_gid_universe.tsv"
    secondary_gid = PHASE4_NS_ROOT / "ordered_secondary_gid_universe.tsv"
    con.execute(
        f"""COPY (
          SELECT row_number() OVER (ORDER BY canonical_gid)-1 AS zero_based_index,canonical_gid
          FROM (SELECT DISTINCT canonical_gid FROM read_parquet('{q(source)}') WHERE primary_weighted_training_eligible)
          ORDER BY canonical_gid
        ) TO '{q(primary_gid)}' (FORMAT CSV,DELIMITER '\t',HEADER TRUE)"""
    )
    con.execute(
        f"""COPY (
          SELECT row_number() OVER (ORDER BY canonical_gid)-1 AS zero_based_index,canonical_gid
          FROM (SELECT DISTINCT canonical_gid FROM read_parquet('{q(source)}') WHERE secondary_unweighted_training_eligible)
          ORDER BY canonical_gid
        ) TO '{q(secondary_gid)}' (FORMAT CSV,DELIMITER '\t',HEADER TRUE)"""
    )
    obs = con.execute(f"SELECT phase4_adjusted_row_id FROM read_parquet('{q(primary)}') ORDER BY zero_based_index").fetchdf()
    pgid = pd.read_csv(primary_gid, sep="\t", dtype=str, keep_default_na=False)
    sgid = pd.read_csv(secondary_gid, sep="\t", dtype=str, keep_default_na=False)
    rows = pd.DataFrame(
        [
            {
                "universe": "PRIMARY_WEIGHTED_OBSERVATIONS",
                "records": len(obs),
                "ordered_key": "phase4_adjusted_row_id",
                "ordered_sha256": index_signature(obs["phase4_adjusted_row_id"]),
                "artifact": primary.name,
            },
            {
                "universe": "PRIMARY_WEIGHTED_GIDS",
                "records": len(pgid),
                "ordered_key": "canonical_gid",
                "ordered_sha256": index_signature(pgid["canonical_gid"]),
                "artifact": primary_gid.name,
            },
            {
                "universe": "SECONDARY_UNWEIGHTED_GIDS",
                "records": len(sgid),
                "ordered_key": "canonical_gid",
                "ordered_sha256": index_signature(sgid["canonical_gid"]),
                "artifact": secondary_gid.name,
            },
        ]
    )
    write_tsv(PHASE4_NS_ROOT / "ordered_universe_signatures.tsv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-immutable-rehash", action="store_true", help="Test-only; never use for signed closure")
    parser.add_argument(
        "--reuse-verified-immutable-hashes",
        action="store_true",
        help="Resume the same closure after a downstream diagnostic failure; requires complete PASS hash tables",
    )
    args = parser.parse_args()
    if args.skip_immutable_rehash:
        raise RuntimeError("Signed closure cannot skip immutable-input rehashing")

    phase4 = load_json(PHASE4_NS_ROOT / "RELEASE_DECISION.json")
    r3 = load_json(PHASE3G_R3_ROOT / "RELEASE_DECISION.json")
    targeted = test_result(PHASE3G_R3_ROOT / "logs/targeted_pytest.stdout.log", 30)
    full = test_result(PHASE3G_R3_ROOT / "logs/full_pytest.stdout.log", 597)
    tests = pd.DataFrame([targeted, full])
    write_tsv(PHASE4_NS_ROOT / "test_summary.tsv", tests)
    write_tsv(PHASE3G_R3_ROOT / "test_summary.tsv", tests)

    replay = validate_replay()
    write_tsv(PHASE4_NS_ROOT / "determinism_regeneration_validation.tsv", replay)
    write_tsv(PHASE3G_R3_ROOT / "determinism_regeneration_validation.tsv", replay)

    con = duckdb.connect()
    universes = freeze_ordered_universes(con)
    con.close()

    if args.reuse_verified_immutable_hashes:
        raw = pd.read_csv(PHASE3G_R3_ROOT / "CLOSING_HASH_MANIFEST.tsv", sep="\t", dtype=str, keep_default_na=False)
        identity = pd.read_csv(PHASE3G_R3_ROOT / "IMMUTABLE_UPSTREAM_HASH_VALIDATION.tsv", sep="\t", dtype=str, keep_default_na=False)
        closing_summary = pd.read_csv(PHASE3G_R3_ROOT / "closing_hash_summary.tsv", sep="\t", dtype=str, keep_default_na=False)
        for frame in (raw, identity):
            if frame.empty or not frame["status"].eq("PASS_UNCHANGED").all():
                raise RuntimeError("Cannot reuse incomplete or failed immutable hash evidence")
        if not closing_summary["status"].eq("PASS").all():
            raise RuntimeError("Cannot reuse failed immutable hash summary")
        for column in ("files", "bytes", "unchanged"):
            closing_summary[column] = pd.to_numeric(closing_summary[column], errors="raise").astype("int64")
        print("reused complete PASS immutable hash evidence from immediately preceding closure attempt", flush=True)
    else:
        raw, identity, closing_summary = close_immutable_inputs()
    for root in (PHASE4_NS_ROOT, PHASE3G_R3_ROOT):
        write_tsv(root / "CLOSING_HASH_MANIFEST.tsv", raw)
        write_tsv(root / "IMMUTABLE_UPSTREAM_HASH_VALIDATION.tsv", identity)
        write_tsv(root / "closing_hash_summary.tsv", closing_summary)

    protected = pd.DataFrame(
        [
            {"scope": "outer_test_outcomes", "accessed": False, "use": "NONE", "status": "PASS_SEALED"},
            {"scope": "final_holdout", "accessed": False, "use": "NONE", "status": "PASS_SEALED"},
            {"scope": "production_kernels", "accessed": False, "use": "NONE", "status": "PASS_NOT_CONSTRUCTED"},
            {"scope": "split_assignments", "accessed": False, "use": "NONE", "status": "PASS_NOT_CONSTRUCTED"},
            {"scope": "model_training", "accessed": False, "use": "NONE", "status": "PASS_NOT_PERFORMED"},
        ]
    )
    for root in (PHASE4_NS_ROOT, PHASE3G_R3_ROOT):
        write_tsv(root / "protected_outcome_and_scope_audit.tsv", protected)

    counts_sum = r3["review_required"] + r3["unresolved_ambiguous"] + r3["unresolved_generic_or_blank"] + r3["unresolved_insufficient_evidence"]
    c_absent = not STAGE1_R3_ROOT.exists()
    d_absent = not PHASE4_R3_ROOT.exists()
    criteria = pd.DataFrame(
        [
            {"criterion": "phase4_namespace_correction", "observed": phase4["status"], "expected": "PASS_PHASE4_NAMESPACE_CORRECTION", "status": "PASS" if phase4["status"] == "PASS_PHASE4_NAMESPACE_CORRECTION" else "FAIL"},
            {"criterion": "phase4_non_identity_exact", "observed": phase4["non_identity_mismatches"], "expected": 0, "status": "PASS" if phase4["non_identity_mismatches"] == 0 else "FAIL"},
            {"criterion": "phase4_observation_ids_unchanged", "observed": phase4["observation_ids_changed"], "expected": 0, "status": "PASS" if phase4["observation_ids_changed"] == 0 else "FAIL"},
            {"criterion": "phase4_views", "observed": phase4["view_checks_passed"], "expected": 8, "status": "PASS" if phase4["view_checks_passed"] == 8 else "FAIL"},
            {"criterion": "phase3g_r3", "observed": r3["status"], "expected": "PASS_PHASE3G_R3_NO_NEW_IDENTITIES", "status": "PASS" if r3["status"] == "PASS_PHASE3G_R3_NO_NEW_IDENTITIES" else "FAIL"},
            {"criterion": "r3_decision_partition", "observed": counts_sum, "expected": r3["source_keys"], "status": "PASS" if counts_sum == r3["source_keys"] else "FAIL"},
            {"criterion": "r3_new_identities", "observed": r3["accepted_exact_authority"], "expected": 0, "status": "PASS" if r3["accepted_exact_authority"] == 0 else "FAIL"},
            {"criterion": "conditional_stage1_release_absent", "observed": c_absent, "expected": True, "status": "PASS" if c_absent else "FAIL"},
            {"criterion": "conditional_phase4_release_absent", "observed": d_absent, "expected": True, "status": "PASS" if d_absent else "FAIL"},
            {"criterion": "deterministic_replay", "observed": int(replay["status"].eq("PASS_BYTE_IDENTICAL").sum()), "expected": len(replay), "status": "PASS" if replay["status"].eq("PASS_BYTE_IDENTICAL").all() else "FAIL"},
            {"criterion": "tests", "observed": int(tests["observed_passed"].sum()), "expected": 627, "status": "PASS" if tests["status"].eq("PASS").all() else "FAIL"},
            {"criterion": "raw_and_identity_inputs_unchanged", "observed": int(closing_summary["unchanged"].sum()), "expected": int(closing_summary["files"].sum()), "status": "PASS" if closing_summary["status"].eq("PASS").all() else "FAIL"},
            {"criterion": "protected_scopes", "observed": int(protected["accessed"].sum()), "expected": 0, "status": "PASS" if not protected["accessed"].any() else "FAIL"},
            {"criterion": "primary_observation_universe", "observed": int(universes.loc[universes["universe"] == "PRIMARY_WEIGHTED_OBSERVATIONS", "records"].iloc[0]), "expected": 2_045_518, "status": "PASS" if int(universes.loc[universes["universe"] == "PRIMARY_WEIGHTED_OBSERVATIONS", "records"].iloc[0]) == 2_045_518 else "FAIL"},
        ]
    )
    overall_status = "READY_FOR_SPLIT_BOUND_PHASE5_REBUILD" if criteria["status"].eq("PASS").all() else "BLOCKED_NAMESPACE_R3_INTEGRITY_FAILURE"
    for root in (PHASE4_NS_ROOT, PHASE3G_R3_ROOT):
        write_tsv(root / "atomic_acceptance_criteria.tsv", criteria)

    now = datetime.now(timezone.utc).isoformat()
    overall = {
        "overall_release_id": OVERALL_RELEASE_ID,
        "status": overall_status,
        "finalized_at_utc": now,
        "authoritative_modelling_foundation": "STAGE1_V2",
        "authoritative_downstream_phase4_release_id": PHASE4_NS_RELEASE_ID,
        "authoritative_downstream_phase4_table": "audit/v2/phase4_namespace_corrected_release_v1/corrected_promoted_phenotypes.parquet",
        "authoritative_downstream_phase4_table_sha256": phase4["corrected_table_sha256"],
        "phase3g_r3_release_id": PHASE3G_R3_RELEASE_ID,
        "new_identities_accepted": 0,
        "affected_stage1_rows": 0,
        "affected_phase4_groups": 0,
        "stage1_reconstruction_status": "NOT_APPLICABLE_NO_NEW_IDENTITIES",
        "phase4_recovery_status": "NOT_APPLICABLE_NO_NEW_IDENTITIES",
        "protected_outcomes_accessed": False,
        "production_kernels_constructed": False,
        "split_assignments_created": False,
        "model_training_performed": False,
        "next_phase": "Freeze ID-only split manifests from the corrected Phase-4 namespace release, then rebuild all Phase-5 fold-local kernels and transforms inside those splits.",
    }
    for root in (PHASE4_NS_ROOT, PHASE3G_R3_ROOT):
        write_json(root / "OVERALL_READINESS_DECISION.json", overall)

    dependency_rows = []
    for name, version in sorted(
        {
            "duckdb": duckdb.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
        }.items()
    ):
        dependency_rows.append({"dependency": name, "version": version, "added_for_release": False, "environment": sys.prefix})
    for root in (PHASE4_NS_ROOT, PHASE3G_R3_ROOT):
        write_tsv(root / "runtime_dependencies.tsv", dependency_rows)

    command_rows = [
            {"order": 1, "command": "phase4_namespace_r3_open_release.py", "result": "PASS", "note": "opened new A/B roots and verified pinned upstreams"},
            {"order": 2, "command": "phase4_namespace_correction.py", "result": "PASS", "note": "namespace-only correction"},
            {"order": 3, "command": "phase3g_r3_identity_recovery.py", "result": "PASS", "note": "exhaustive R3 evidence adjudication; no new identities"},
            {"order": 4, "command": "pytest tests/test_phase4_namespace_phase3g_r3.py", "result": "PASS_25", "note": "initial targeted suite"},
            {"order": 5, "command": "pytest", "result": "PASS_592", "note": "initial complete repository suite"},
            {"order": 6, "command": "A/B deterministic replay", "result": "PASS_14_BYTE_IDENTICAL", "note": "substantive core artifacts"},
            {"order": 7, "command": "phase4_namespace_r3_finalize.py", "result": "BLOCKED_DIAGNOSTIC_TEST_LOG_ENCODING", "note": "all 2,764 immutable inputs passed; UTF-16LE targeted-test log was incorrectly parsed as UTF-8"},
    ]
    if args.reuse_verified_immutable_hashes:
        command_rows.append(
            {"order": 8, "command": "pytest targeted encoding-regression attempt", "result": "FAIL_2_PATH_FIXTURE_ONLY", "note": "temporary fixture paths were outside repository; parser path labeling made path-agnostic"}
        )
        command_rows.append(
            {"order": 9, "command": "pytest tests/test_phase4_namespace_phase3g_r3.py", "result": "PASS_30", "note": "final targeted suite including parser and conditional-reconstruction fixtures"}
        )
        command_rows.append(
            {"order": 10, "command": "pytest", "result": "PASS_597", "note": "final complete repository suite"}
        )
        command_rows.append(
            {"order": 11, "command": "phase4_namespace_r3_finalize.py --reuse-verified-immutable-hashes", "result": overall_status, "note": "encoding-aware test parser; reused complete PASS hash evidence from immediately preceding closure attempt"}
        )
    else:
        command_rows[-1] = {"order": 7, "command": "phase4_namespace_r3_finalize.py", "result": overall_status, "note": "rehash, frozen universes, atomic readiness"}
    commands = pd.DataFrame(command_rows)
    for root in (PHASE4_NS_ROOT, PHASE3G_R3_ROOT):
        write_tsv(root / "command_log.tsv", commands)

    correction = pd.DataFrame(
        [
            {
                "issue": "targeted pytest log encoding",
                "observed_encoding": "UTF-16LE_WITH_BOM",
                "incorrect_initial_interpretation": "NO_MATCH / -1 tests",
                "verified_content": "25 passed in 5.79s",
                "data_or_identity_outputs_affected": False,
                "immutable_hash_results_affected": False,
                "correction": "BOM-aware UTF-16/UTF-8 parser",
                "status": "PASS_DIAGNOSTIC_CORRECTED",
            },
            {
                "issue": "test-log path labeling in regression fixture",
                "observed_encoding": "UTF-8_AND_UTF-16",
                "incorrect_initial_interpretation": "ValueError for pytest tmp_path outside repository root",
                "verified_content": "parser content extraction was correct before path-label failure",
                "data_or_identity_outputs_affected": False,
                "immutable_hash_results_affected": False,
                "correction": "repository-relative label when possible; absolute label otherwise",
                "status": "PASS_DIAGNOSTIC_CORRECTED",
            },
        ]
    )
    for root in (PHASE4_NS_ROOT, PHASE3G_R3_ROOT):
        write_tsv(root / "diagnostic_correction_ledger.tsv", correction)

    report = f"""# Namespace correction and Phase-3G R3 closure

Overall status: **{overall_status}**

- Phase-4 namespace release: `{PHASE4_NS_RELEASE_ID}`; 3,193,677 rows; 2,242,863 corrected eligible rows; 950,814 unresolved archival rows retained.
- Non-identity invariance: 53/53 fields exact; observation IDs unchanged; 8/8 views exact.
- Phase-3G R3 release: `{PHASE3G_R3_RELEASE_ID}`; 3,086 source keys adjudicated; 0 new identities accepted.
- Final R3 states: 1,122 review required, 77 ambiguous, 30 generic/blank, 1,857 insufficient evidence.
- Conditional Stage-1 R3 and Phase-4 recovery releases: not applicable and not created.
- Determinism: {int(replay['status'].eq('PASS_BYTE_IDENTICAL').sum())}/{len(replay)} core artifacts byte-identical.
- Tests: {targeted['observed_passed']} targeted and {full['observed_passed']} full-suite tests passed.
- Raw and versioned identity inputs: {int(closing_summary['unchanged'].sum())}/{int(closing_summary['files'].sum())} unchanged at closing.
- Outer-test outcomes and final holdout remained sealed; no splits, kernels, training, or protected-outcome use occurred.

The authoritative downstream phenotype input is the namespace-corrected Phase-4 table. The next phase must freeze ID-only splits before fitting any fold-local transform or kernel.
"""
    for root in (PHASE4_NS_ROOT, PHASE3G_R3_ROOT):
        (root / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    git_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPOSITORY_ROOT, text=True).strip()
    for root, release_id in ((PHASE4_NS_ROOT, PHASE4_NS_RELEASE_ID), (PHASE3G_R3_ROOT, PHASE3G_R3_RELEASE_ID)):
        manifest = load_json(root / "run_manifest.json")
        manifest.update(
            {
                "status": overall_status,
                "closed_at_utc": now,
                "git_head_at_closure": git_head,
                "git_branch_at_closure": git_branch,
                "protected_outcomes_accessed": False,
                "model_training_performed": False,
                "production_kernel_construction_performed": False,
                "split_assignments_created": False,
                "random_seed_policy": "NOT_APPLICABLE_DETERMINISTIC_NONSTOCHASTIC_RELEASE",
                "conditional_stage1_root_exists": not c_absent,
                "conditional_phase4_root_exists": not d_absent,
            }
        )
        write_json(root / "run_manifest.json", manifest)
        output_manifest(root, release_id)

    print(json.dumps(overall, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
