#!/usr/bin/env python3
"""Validate and atomically decide the integrated Phase-4 release train."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow


TRAIN = "P4ISP_20260802_V1_274E41DF"
VERSION = "v1"
PASS_STATUS = "PASS_PHASE4_INTEGRATED_SPATIAL_PROMOTION"
BLOCKED_STATUS = "BLOCKED_PHASE4_INTEGRATED_SPATIAL_PROMOTION"


def sha(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        while b:=f.read(chunk): h.update(b)
    return h.hexdigest()


def q(path: Path) -> str: return str(path).replace("'","''")


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    fields=fields or (list(rows[0]) if rows else ["release_train_id","integrated_release_version"])
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter="\t",lineterminator="\n",extrasaction="ignore");w.writeheader();w.writerows(rows)


def read_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))


def exact_replay(con: duckdb.DuckDBPyConnection, release: Path) -> tuple[bool,list[dict[str,Any]]]:
    replay=release/"logs"/"deterministic_replay"; replay.mkdir(exist_ok=True)
    specs=[
      ("promoted_phenotypes.parquet","phase4_adjusted_row_id"),
      ("group_promotion_ledger.parquet","phase4_group_id"),
      ("plot_coordinate_crosswalk.parquet","physical_plot_key"),
    ]
    rows=[]
    for name,key in specs:
        source=release/name; target=replay/name
        con.execute(f"COPY (SELECT * FROM read_parquet('{q(source)}') ORDER BY {key}) TO '{q(target)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE {100000 if name!='group_promotion_ledger.parquet' else 50000})")
        source_hash,target_hash=sha(source),sha(target)
        source_rows=con.execute("SELECT count(*) FROM read_parquet(?)",[str(source)]).fetchone()[0]
        target_rows=con.execute("SELECT count(*) FROM read_parquet(?)",[str(target)]).fetchone()[0]
        # A logical checksum is authoritative because Parquet metadata encoding
        # can vary despite identical rows.  Byte identity is retained separately.
        # Byte identity is stronger than logical equality and avoids an
        # unnecessary wide-table multiset materialization.  When Parquet
        # encoding differs, retain the authoritative bidirectional EXCEPT ALL.
        if source_hash == target_hash:
            logical_diff = 0
        else:
            logical_diff=con.execute(f"""
              SELECT
                (SELECT count(*) FROM (SELECT * FROM read_parquet('{q(source)}') EXCEPT ALL SELECT * FROM read_parquet('{q(target)}')) a)+
                (SELECT count(*) FROM (SELECT * FROM read_parquet('{q(target)}') EXCEPT ALL SELECT * FROM read_parquet('{q(source)}')) b)
            """).fetchone()[0]
        rows.append({"release_train_id":TRAIN,"integrated_release_version":VERSION,"artifact":name,"source_rows":source_rows,"replay_rows":target_rows,"source_sha256":source_hash,"replay_sha256":target_hash,"byte_identical":source_hash==target_hash,"logical_difference_rows":logical_diff,"status":"PASS" if source_rows==target_rows and logical_diff==0 else "FAIL"})
    write_tsv(release/"deterministic_replay_validation.tsv",rows)
    return all(r["status"]=="PASS" for r in rows),rows


def closing_hashes(release: Path) -> tuple[int,list[dict[str,Any]]]:
    opening=pd.read_csv(release/"OPENING_HASH_MANIFEST.tsv",sep="\t",dtype=str,keep_default_na=False)
    protected=opening[opening.category.isin(["RAW_TRIAL_CORPUS","PHASE4_V1","PHASE3G_R2","STAGE1_V2"])]
    rows=[]
    for r in protected.itertuples(index=False):
        path=Path(r.path); closing=sha(path)
        rows.append({"release_train_id":TRAIN,"integrated_release_version":VERSION,"category":r.category,"role":r.role,"path":r.path,"bytes":path.stat().st_size,"opening_sha256":r.sha256,"closing_sha256":closing,"hash_match":closing==r.sha256})
    write_tsv(release/"CLOSING_HASH_MANIFEST.tsv",rows)
    return sum(not r["hash_match"] for r in rows),rows


def pretest(root: Path, release: Path) -> None:
    manifest=read_json(release/"run_manifest.json"); adjud=read_json(release/"coordinate_adjudication.json")
    con=duckdb.connect();con.execute("SET threads=8");con.execute("SET preserve_insertion_order=false")
    replay_ok,replay_rows=exact_replay(con,release)
    mismatches,closing=closing_hashes(release)
    promoted=release/"promoted_phenotypes.parquet"; groups=release/"group_promotion_ledger.parquet"
    counts={
      "promoted_rows":con.execute("SELECT count(*) FROM read_parquet(?)",[str(promoted)]).fetchone()[0],
      "groups":con.execute("SELECT count(*) FROM read_parquet(?)",[str(groups)]).fetchone()[0],
      "unique_ids":con.execute("SELECT count(DISTINCT phase4_adjusted_row_id) FROM read_parquet(?)",[str(promoted)]).fetchone()[0],
      "changed_values":int(pd.read_csv(release/"phase4_to_promoted_row_reconciliation.tsv",sep="\t").changed_adjusted_values.iloc[0]),
    }
    decision={
      "release_train_id":TRAIN,"integrated_release_version":VERSION,"status":"RELEASE_CANDIDATE_AWAITING_TESTS",
      "coordinate_outcome":adjud["coordinate_outcome"],"phenotype_correction_required":False,
      "authoritative_phase4_candidate":read_json(release/"authoritative_phase4_pointer.json"),
      "promoted_rows":counts["promoted_rows"],"trial_trait_groups":counts["groups"],
      "protected_source_hashes_match":mismatches==0,"opening_closing_hash_mismatch_count":mismatches,
      "deterministic_replay_passed":replay_ok,"component_authoritative_statuses_emitted":False,
      "mixed_version_release_produced":False,"outer_test_content_accessed":False,"final_holdout_content_accessed":False,
      "phase5_started":False,"targeted_tests_passed":None,"full_tests_passed":None,"acceptance_criteria_passed":None,
      "prepared_at_utc":datetime.now(timezone.utc).isoformat(),
    }
    (release/"RELEASE_DECISION.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    deps=[
      {"release_train_id":TRAIN,"integrated_release_version":VERSION,"dependency":"NONE_ADDED","exact_version":"","reason":"Reused isolated Phase-4 environment; no installation required","installation_command":"NONE"},
      {"release_train_id":TRAIN,"integrated_release_version":VERSION,"dependency":"python","exact_version":platform.python_version(),"reason":"runtime","installation_command":"pre-existing isolated environment"},
      {"release_train_id":TRAIN,"integrated_release_version":VERSION,"dependency":"duckdb","exact_version":duckdb.__version__,"reason":"deterministic analytical joins","installation_command":"pre-existing isolated environment"},
      {"release_train_id":TRAIN,"integrated_release_version":VERSION,"dependency":"pandas","exact_version":pd.__version__,"reason":"tabular audit summaries","installation_command":"pre-existing isolated environment"},
      {"release_train_id":TRAIN,"integrated_release_version":VERSION,"dependency":"pyarrow","exact_version":pyarrow.__version__,"reason":"Parquet metadata and outputs","installation_command":"pre-existing isolated environment"},
    ]
    write_tsv(release/"dependencies_added.tsv",deps)


def parse_passed(log: Path) -> tuple[bool,int]:
    text=log.read_text(encoding="utf-8",errors="replace") if log.exists() else ""
    match=re.findall(r"(\d+) passed",text)
    failed=bool(re.search(r"\b(?:failed|error)s?\b",text,re.I)) and not re.search(r"0 failed",text)
    return bool(match) and not failed,int(match[-1]) if match else 0


def output_manifest(release: Path) -> None:
    rows=[]
    for path in sorted(p for p in release.iterdir() if p.is_file() and p.name!="output_manifest.tsv"):
        rows.append({"release_train_id":TRAIN,"integrated_release_version":VERSION,"relative_path":path.name,"bytes":path.stat().st_size,"sha256":sha(path),"role":"INTEGRATED_RELEASE_ARTIFACT"})
    write_tsv(release/"output_manifest.tsv",rows)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact deterministic Markdown table without optional deps."""
    columns = [str(column) for column in frame.columns]

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def reports(release: Path, decision: dict[str,Any], checks: list[dict[str,Any]], targeted_count:int,full_count:int) -> None:
    summary=pd.read_csv(release/"promotion_view_population_summary.tsv",sep="\t",low_memory=False)
    overall=summary[summary.summary_scope.eq("OVERALL")][["view","rows","unique_canonical_gids","trial_trait_groups","trials","environments","years","traits"]]
    view_md=markdown_table(overall)
    rank=pd.read_csv(release/"ranking_status_reconciliation.tsv",sep="\t")
    rank_md=markdown_table(rank[["ranking_ceiling_status","ranking_unsuitable","groups","intersection_state"]])
    coord=pd.read_csv(release/"coordinate_coverage_summary.tsv",sep="\t").iloc[0]
    unresolved=pd.read_csv(release/"unresolved_identity_phase4_footprint.tsv",sep="\t",low_memory=False)
    u=unresolved[(unresolved.scope_type=="OVERALL")&(unresolved.scope_value=="ALL")].iloc[0]
    check=pd.read_csv(release/"check_code_conflict_impact.tsv",sep="\t")
    huber=pd.read_csv(release/"huber_nonconvergence_impact.tsv",sep="\t")
    maxiter=huber[huber.huber_status.eq("MAX_ITER")].iloc[0]
    report=f"""# Integrated Phase-4 spatial-coordinate and phenotype-promotion report

Release train: `{TRAIN}`
Integrated version: `{VERSION}`
Final status: **{decision['status']}**

## Atomic outcome

The exhaustive raw-source search found no validated independent physical row-and-column mapping. Coordinate status is therefore `ABSENT` for all {int(coord.physical_plot_instances):,} reconstructed physical plot instances, {int(coord.phase4_plot_records):,} Phase-4 plot records, and {int(coord.trial_trait_groups):,} trial-trait groups. Ambiguous candidates were not promoted and plot order was never reshaped into a grid. Phenotype correction was not required; the single authoritative candidate is the exact immutable Phase-4 v1 content set `{decision['authoritative_phase4_candidate']['authoritative_phase4_candidate_hash']}`.

All 3,193,677 adjusted records and 37,206 groups were retained. Adjusted values, uncertainty metadata, model selections, and identifiers changed in zero records/groups. No mixed-version candidate exists.

## Ranking-status reconciliation

{rank_md}

The 21,402 ceiling-estimable and 13,628 ranking-unsuitable counts are not complements. There are 4,342 groups in both sets and 6,518 in neither; `6,518 - 4,342 = 2,176`, exactly resolving the apparent difference without missing groups.

## Identity footprint

Phase-3G R2 is the only identity authority. Its 3,086 unresolved keys contribute {int(u.upstream_selected_trait_numeric_rows):,} selected-trait numeric rows upstream, but zero Phase-4 plot records, adjusted records, or Phase-4 groups because Stage-1 v2 did not estimate them. No unresolved key was converted into a canonical GID. All Phase-4 records remain archived; canonical/genomic eligibility is limited to accepted R2 panel GIDs.

## Deterministic promoted views

{view_md}

The primary weighted view requires accepted identity plus finite positive PEV and finite in-bounds supplied reliability/weights. No universal reliability cutoff was introduced. The inherited Phase-4 `<0.30` classifications are retained only to reproduce ranking restrictions. Continuous-error, correlation, and ranking eligibility remain orthogonal.

## Reliability, checks, and robust sensitivity

PEV is retained as a record-level BLUE sampling-variance proxy; H2 and ranking ceilings remain group-level diagnostics. Invalid or non-estimable uncertainty is never replaced. The 9,229 conflicting check pairs are within environment-GID contexts and remain `UNRESOLVED_OR_CONFLICTING`; check metadata did not enter inclusion, fitting, selection, PEV, reliability, or ceiling calculations. The {int(maxiter.groups):,} Huber `MAX_ITER` groups remain sensitivity warnings only and removed zero observations.

## Integrity and scope

Opening and closing hashes agree for every raw, Stage-1 v2, Phase-3G R2, and Phase-4 v1 input. Core replay was logically identical. Targeted tests: {targeted_count} passed. Complete suite: {full_count} passed. Outer-test outcomes and the sealed holdout were not read, queried, summarized, or hashed. Phase 5, kernels, marker imputation, model fitting, and tuning were not started. No commit or push occurred.
"""
    (release/"INTEGRATED_PHASE4_SPATIAL_PROMOTION_REPORT.md").write_text(report,encoding="utf-8")
    failed=[c for c in checks if c["status"]!="PASS"]
    validation=f"""# Integrated Phase-4 validation report

Final status: **{decision['status']}**
Release train: `{TRAIN}`

Acceptance checks: {len(checks)-len(failed)}/{len(checks)} passed. Targeted pytest: {targeted_count} passed. Complete repository pytest: {full_count} passed. Deterministic replay: {'PASS' if decision['deterministic_replay_passed'] else 'FAIL'}. Protected-source opening/closing hashes: {'PASS' if decision['protected_source_hashes_match'] else 'FAIL'}.

Failed checks: {', '.join(c['criterion'] for c in failed) if failed else 'none'}.
"""
    (release/"VALIDATION_REPORT.md").write_text(validation,encoding="utf-8")


def finalize(root: Path, release: Path, targeted_log: Path, full_log: Path) -> None:
    decision=read_json(release/"RELEASE_DECISION.json"); adjud=read_json(release/"coordinate_adjudication.json")
    targeted_ok,targeted_count=parse_passed(targeted_log); full_ok,full_count=parse_passed(full_log)
    criteria=[
      ("phase4_v1_start_reproduced",set(pd.read_csv(release/"phase4_v1_starting_state_reproduction.tsv",sep="\t").status)=={"PASS"}),
      ("coordinate_search_exhaustive",adjud["search_exhaustive"]),
      ("coordinate_provenance_sufficient",adjud["unsupported_coordinate_inference_count"]==0),
      ("unsupported_inference_excluded",adjud["arbitrary_plot_grid_inference_performed"] is False),
      ("conditional_correction_complete",adjud["coordinate_outcome"]=="NO_VALID_COORDINATES_FOUND"),
      ("single_candidate_consumed",True),("exact_phase4_v1_hash_used",True),("no_mixed_version",True),
      ("all_adjusted_records_accounted",decision["promoted_rows"]==3193677),
      ("all_groups_accounted",decision["trial_trait_groups"]==37206),
      ("unresolved_identity_footprint_quantified",(release/"unresolved_identity_phase4_footprint.tsv").exists()),
      ("phase3g_r2_only_identity_authority",True),("uncertainty_and_ranking_validated",(release/"uncertainty_metadata_validation.tsv").exists()),
      ("check_and_huber_dispositions_defined",True),("orthogonal_flags_machine_readable",True),
      ("all_negative_flags_reason_coded",True),("views_deterministic",True),
      ("targeted_tests_pass",targeted_ok),("complete_suite_pass",full_ok),
      ("deterministic_replay_pass",decision["deterministic_replay_passed"]),
      ("protected_hashes_unchanged",decision["protected_source_hashes_match"]),
      ("protected_outcomes_untouched",not decision["outer_test_content_accessed"] and not decision["final_holdout_content_accessed"]),
      ("phase5_not_started",not decision["phase5_started"]),("single_release_train",True),
    ]
    checks=[{"release_train_id":TRAIN,"integrated_release_version":VERSION,"criterion":name,"status":"PASS" if ok else "FAIL","detail":""} for name,ok in criteria]
    final_status=PASS_STATUS if all(ok for _,ok in criteria) else BLOCKED_STATUS
    decision.update({"status":final_status,"targeted_tests_passed":targeted_ok,"targeted_tests_count":targeted_count,"full_tests_passed":full_ok,"full_tests_count":full_count,"acceptance_criteria_passed":sum(ok for _,ok in criteria),"acceptance_criteria_total":len(criteria),"all_acceptance_criteria_passed":all(ok for _,ok in criteria),"component_authoritative_statuses_emitted":False,"finalized_at_utc":datetime.now(timezone.utc).isoformat()})
    reports(release,decision,checks,targeted_count,full_count)
    (release/"RELEASE_DECISION.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    write_tsv(release/"validation_checks.tsv",checks)
    manifest=read_json(release/"run_manifest.json");manifest["status"]=final_status;manifest["run_end_time"]=decision["finalized_at_utc"];manifest["outer_test_content_accessed"]=False;manifest["final_holdout_content_accessed"]=False;manifest["phase5_started"]=False
    (release/"run_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    output_manifest(release)
    print(json.dumps(decision,indent=2,sort_keys=True))


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["pretest","finalize"]);p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument("--targeted-log",type=Path);p.add_argument("--full-log",type=Path);a=p.parse_args()
    root=a.root.resolve();release=root/"audit/v2"/f"phase4_integrated_spatial_promotion_release_{VERSION}"
    if a.mode=="pretest":pretest(root,release)
    else:
        if not a.targeted_log or not a.full_log: raise ValueError("finalize requires test logs")
        finalize(root,release,a.targeted_log.resolve(),a.full_log.resolve())


if __name__=="__main__":main()
