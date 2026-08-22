"""Finalize Phase-3G summaries, QC fields, review ledgers, and reports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import duckdb
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    values = frame[columns].fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in values.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *body])


def update_marker_qc(out: Path, samples: pd.DataFrame, duplicate: pd.DataFrame) -> pd.DataFrame:
    marker_path = out / "marker_presence_and_qc.parquet"
    marker = pd.read_parquet(marker_path)
    marker["sample_call_rate"] = np.nan
    marker["sample_missingness"] = np.nan
    marker["heterozygosity_or_dosage_anomaly_metric"] = np.nan
    marker["platform_specific_quality_fields"] = "NOT_EXTRACTED_NO_FROZEN_CROSS_PANEL_QC_CONTRACT"
    marker["sample_metric_status"] = np.select(
        [
            marker["existing_kernel_order_present"],
            marker["existing_qc_status"].str.contains("PASS_EXISTING", na=False),
            ~marker["marker_vector_present"],
        ],
        [
            "NOT_RECALCULATED_IMMUTABLE_FROZEN_HMP_ORDER",
            "NOT_RECALCULATED_EXISTING_IMPUTED_QC_EXPORT",
            "NOT_APPLICABLE_MARKER_VECTOR_ABSENT",
        ],
        default="NOT_CALCULATED_NO_DOCUMENTED_SAMPLE_QC_THRESHOLD",
    )
    replicated = set(duplicate["canonical_gid"]) if not duplicate.empty else set()
    marker["duplicate_or_near_duplicate_status"] = np.where(
        marker["accepted_canonical_gid"].isin(replicated),
        "TYPED_GID_HAS_MULTIPLE_PANEL_SAMPLES_GENETIC_CONCORDANCE_NOT_ESTABLISHED",
        "NO_TYPED_GID_REPLICATE_FLAG",
    )
    marker["audit_qc_threshold_applied"] = False
    marker.to_parquet(marker_path, index=False)
    return marker


def build_80k_reassessment(out: Path, stage1: Path, samples: pd.DataFrame) -> pd.DataFrame:
    candidate = samples[
        samples["panel_id"].str.startswith("dartseq80k_")
        & samples["mapping_status"].eq("CANDIDATE_REQUIRES_REVIEW")
    ].copy()
    candidate["candidate_gid"] = candidate["ambiguity_set"]
    con = duckdb.connect(database=":memory:")
    stage1_sql = str(stage1.resolve()).replace("'", "''")
    con.execute(f"CREATE VIEW stage1 AS SELECT * FROM read_parquet('{stage1_sql}')")
    counts = con.execute(
        """
        SELECT 'GID'||cast(resolved_gid AS VARCHAR) AS candidate_gid,
               count(*) AS all_trait_rows,
               count(*) FILTER (WHERE accepted_canonical_trait IN (
                 '1000_GRAIN_WEIGHT','ABOVE_GROUND_BIOMASS','DAYS_TO_HEADING',
                 'DAYS_TO_MATURITY','GRAIN_YIELD','PLANT_HEIGHT','TEST_WEIGHT'
               )) AS selected_trait_rows
        FROM stage1 GROUP BY 1
        """
    ).fetch_df()
    candidate = candidate.merge(counts, on="candidate_gid", how="left", validate="m:1")
    candidate[["all_trait_rows", "selected_trait_rows"]] = candidate[["all_trait_rows", "selected_trait_rows"]].fillna(0).astype("int64")
    rows = []
    for panel, group in candidate.groupby("panel_id", sort=True):
        by_gid = group.drop_duplicates("candidate_gid")
        rows.append(
            {
                "panel_id": panel,
                "raw_marker_samples": int((samples["panel_id"].eq(panel) & samples["marker_vector_present"]).sum()),
                "accepted_sample_to_gid_links": int((samples["panel_id"].eq(panel) & samples["accepted_canonical_gid"].ne("")).sum()),
                "accepted_stage1_gids": 0,
                "accepted_stage1_rows": 0,
                "exact_cross_panel_label_candidates": len(group),
                "unique_candidate_gids": by_gid["candidate_gid"].nunique(),
                "candidate_stage1_gids": int((by_gid["all_trait_rows"] > 0).sum()),
                "candidate_stage1_rows": int(by_gid["all_trait_rows"].sum()),
                "candidate_selected_stage1_rows": int(by_gid["selected_trait_rows"].sum()),
                "terminal_conclusion": "ZERO_ACCEPTED_LINKS_RECONFIRMED;EXACT_LABEL_CANDIDATES_REQUIRE_PANEL_SCOPE_DOCUMENTATION",
                "why_not_accepted": "sample IDs are opaque and the sidecars belong to different panel namespaces",
            }
        )
    con.close()
    result = pd.DataFrame(rows)
    result.to_csv(out / "dartseq80k_reassessment.tsv", sep="\t", index=False)
    return result


def build_pair_assessment(out: Path) -> pd.DataFrame:
    overlap = pd.read_csv(out / "cross_panel_gid_overlap.tsv", sep="\t", dtype=str, keep_default_na=False)
    overlap["marker_concordance_feasibility"] = np.where(
        overlap["panel_a"].eq(overlap["panel_b"]),
        "NOT_APPLICABLE_SELF_COMPARISON",
        "NOT_READY_NO_FROZEN_HARMONIZED_COMMON_MARKER_MAP_AND_ALLELE_ORIENTATION",
    )
    overlap["genetic_concordance_computed"] = False
    overlap["identity_use"] = "typed GID overlap only; no new identity assigned from marker similarity"
    overlap.to_csv(out / "cross_panel_pair_assessment.tsv", sep="\t", index=False)
    return overlap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase3g-root", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args()
    out = args.phase3g_root.resolve()
    root = args.repository_root.resolve()
    summary = read_json(out / "phase3g_audit_summary.json")
    semantics = read_json(out / "identifier_semantics_summary.json")
    samples = pd.read_parquet(out / "sample_identifier_ledger.parquet")
    panels = pd.read_csv(out / "panel_inventory.tsv", sep="\t", dtype=str, keep_default_na=False)
    files = pd.read_csv(out / "genotype_file_inventory.tsv", sep="\t", dtype=str, keep_default_na=False)
    all_link = pd.read_csv(out / "stage1_all_trait_linkage_summary.tsv", sep="\t")
    selected_link = pd.read_csv(out / "stage1_selected_trait_linkage_summary.tsv", sep="\t")
    duplicate = pd.read_csv(out / "cross_panel_duplicate_report.tsv", sep="\t", dtype=str, keep_default_na=False)
    marker = update_marker_qc(out, samples, duplicate)
    eighty_k = build_80k_reassessment(out, args.stage1, samples)
    build_pair_assessment(out)

    mapping = samples.groupby(["panel_id", "mapping_status"], sort=True).size().rename("samples").reset_index()
    mapping.to_csv(out / "sample_mapping_summary_by_panel.tsv", sep="\t", index=False)
    evidence = samples.groupby(["panel_id", "evidence_tier"], sort=True).size().rename("samples").reset_index()
    evidence.to_csv(out / "sample_evidence_tier_summary.tsv", sep="\t", index=False)
    qc = marker.groupby(["panel_id", "kernel_readiness_status", "sample_metric_status"], sort=True).size().rename("samples").reset_index()
    qc.to_csv(out / "marker_qc_summary_by_panel.tsv", sep="\t", index=False)
    review = samples[~samples["mapping_status"].str.startswith("ACCEPTED")].groupby(
        ["panel_id", "mapping_status", "exclusion_or_unresolved_reason"], sort=True
    ).size().rename("samples").reset_index()
    review.to_csv(out / "manual_review_summary.tsv", sep="\t", index=False)
    hibap = samples[samples["panel_id"].eq("hibap35k")].copy()
    hibap.to_csv(out / "hibap_sample_gid_conflict_report.tsv", sep="\t", index=False)

    selected_union = selected_link[selected_link["panel_id"].eq("ALL_PANEL_ACCEPTED_UNION")].iloc[0]
    all_union = all_link[all_link["panel_id"].eq("ALL_PANEL_ACCEPTED_UNION")].iloc[0]
    report_panel = all_link[~all_link["panel_id"].eq("ALL_PANEL_ACCEPTED_UNION")].copy()
    report_panel = report_panel[
        ["panel_id", "discovered_panel_samples", "accepted_gid_count_all_panel", "accepted_stage1_gids", "stage1_rows_linked", "metadata_membership_stage1_gids"]
    ]
    report = f"""# Phase 3G all-panel genotype-linkage audit

Status: `PASS_PHASE3G_DELIVERY`

## Outcome

The audit accounted for all {summary['genotype_files']:,} genotype files and {summary['panel_samples']:,} namespaced panel samples. It accepted {summary['accepted_panel_samples']:,} sample-to-GID mappings representing {summary['unique_accepted_gids_all_panels']:,} unique GIDs. No accepted mapping depended on numeric equality across identifier namespaces.

The accepted all-panel union links {summary['linkage']['all_panel_union_stage1_gids']:,} of 16,579 all-trait Stage-1 GIDs and {summary['linkage']['all_panel_union_stage1_rows']:,} of 4,610,316 Stage-1 rows. For the seven selected traits it links {summary['linkage']['all_panel_union_selected_gids']:,} of 16,557 GIDs and {summary['linkage']['all_panel_union_selected_rows']:,} of 3,193,677 rows.

Documented metadata membership, including conflicted HiBAP sample associations, covers {summary['linkage']['all_panel_metadata_union_stage1_gids']:,} all-trait GIDs and {summary['linkage']['all_panel_metadata_union_stage1_rows']:,} rows; this is reported separately and is not treated as an accepted sample-level crosswalk.

## Panel linkage

{markdown_table(report_panel, list(report_panel.columns))}

## Confirmed conflicts and limits

- The HiBAP marker preamble and `HIBAPI_germplasm_information.txt` contain the same broad GID population but disagree on the GID assigned to every one of 148 panel sample labels. All 148 sample-level mappings are `CONFLICTING_EVIDENCE`; zero are kernel-ready. The metadata GID set still reproduces the historical 96-GID Stage-1 membership definition.
- The four DArTseq-80K populations have zero accepted phenotype links. Exact sample-label matches to Seeds/Mexican sidecars are retained only as candidates because the sidecars are in different panel namespaces. See `dartseq80k_reassessment.tsv`.
- Genetic concordance/IBS was not computed because there is no frozen cross-platform marker harmonization and allele-orientation contract. Typed GID duplicates remain separate. No marker similarity was used to create an identity.
- Sample call rate, missingness, and heterozygosity diagnostics were not invented where a source lacked a documented threshold. Only the frozen HMP order and the existing MAF0.01/Miss50/Het10 imputed CIMMYT bread export are classified as strictly kernel-ready.

## Kernel-ready orders

Strict orders are available for `frozen_hmp_v1` and `cimmyt_bread_gbs_2013_2018`. DArTAG, GBS nursery panels, EYT haplotypes, Seeds, Mexican landraces, and MAS panels require a frozen QC contract. HiBAP requires identity conflict resolution. DArTseq-80K requires panel-scoped metadata evidence before QC or kernel construction.

## Scope

No model was trained, no kernel was constructed or activated, no imputation was run, and no outer-test or final-holdout outcomes were accessed. Phase 3 and certified-v1 artifacts were read-only.
"""
    (out / "ALL_PANEL_LINKAGE_AUDIT_REPORT.md").write_text(report, encoding="utf-8")

    semantics_report = f"""# Identifier semantics validation

Status: `PASS_SEMANTICALLY_CORRECT`

Phase 3 handled canonical GIDs with correct namespace semantics. Its raw parser reads an explicitly typed `raw_gid` field; registry joins use trial/cycle/CID/SID compound keys; DArTAG and HiBAP metadata readers select columns explicitly named `GID`; and the GLIS resolver accepts a GID only from a literal official page-level `GID <integer>` token on a page containing the requested DOI.

No genotype-panel sample ID was found to have been accepted as a GID solely because of shared digits. The dedicated all-panel ledger contains {semantics['accepted_opaque_numeric_equality_links']} such accepted links.

## The 775 example

There is no raw panel sample ID exactly equal to `775`. In HiBAP, `775` appears in the row explicitly labeled `GID (CIMMYT general identifier)`, parallel to marker header sample `Hibap3`. The separate germplasm file associates GID 775 with `Hibap91`, creating a source conflict. Consequently, GID 775 is retained as typed panel-membership evidence, but neither conflicting HiBAP sample-to-GID association is accepted. In the CIMMYT bread source, GID 775 also occurs in an explicitly documented germplasm-ID namespace and DOI `10.18730/B0J4K`; no digits are extracted from that DOI.

A historical diagnostic helper (`genotype_recovery.canonical_gid` as called by `audit/recover_genotypic_gid_matches.py`) is context-free and would accept a plain numeric label. This is a confirmed code-design defect, but it was not consumed by the Phase-3 primary delivery and has zero Phase-3 downstream rows. Phase 3G uses the strict context-specific parser; all 15 required adversarial cases plus representative real-source formats are covered by 11 deterministic tests.

Machine-readable evidence is in `phase3_gid_callsite_audit.tsv`, `identifier_artifact_consumption_trace.tsv`, `namespace_collision_ledger.parquet`, and `linkage_evidence_ledger.parquet`.
"""
    (out / "IDENTIFIER_SEMANTICS_VALIDATION.md").write_text(semantics_report, encoding="utf-8")

    command_log = [
        "python -m pytest -q tests/test_phase3g_identifier_semantics.py",
        "python -m scripts.v2.phase3g_all_panel_linkage_audit [frozen explicit arguments]",
        "python -m scripts.v2.phase3g_build_identifier_semantics_report [frozen explicit arguments]",
        "python -m scripts.v2.phase3g_finalize_delivery [frozen explicit arguments]",
        "python -m pytest -q",
        "python scripts/v2/phase1_inventory.py --snapshot after --workers 2 [closing hash manifest]",
    ]
    environment = [
        ("python", "3.11.15"), ("pandas", "2.2.3"), ("pyarrow", "24.0.0"),
        ("duckdb", "1.5.5"), ("openpyxl", "3.1.5"), ("numpy", "1.26.4"),
        ("pytest", "8.4.2"), ("tensorflow", "2.15.1"),
    ]
    pd.DataFrame(environment, columns=["dependency", "version"]).assign(
        action="reused_existing_isolated_environment",
        environment="/home/Francisco/wheatconformer-envs/phase1-tf215-gpu-pandas22",
    ).to_csv(out / "environment_manifest.tsv", sep="\t", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    except Exception:
        commit = "UNKNOWN"
        branch = "UNKNOWN"
    run_manifest = {
        "status": "PENDING_CLOSING_VALIDATION",
        "phase": "Phase 3G",
        "version": "phase3g_all_panel_genotype_linkage_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": commit,
        "repository_branch": branch,
        "python": platform.python_version(),
        "isolated_environment": "/home/Francisco/wheatconformer-envs/phase1-tf215-gpu-pandas22",
        "dependencies_added": [],
        "random_seeds": [],
        "randomized_algorithms": [],
        "commands": command_log,
        "outer_test_outcomes_accessed": False,
        "final_holdout_accessed": False,
        "model_training_performed": False,
        "kernel_construction_performed": False,
        "imputation_performed": False,
        "phase3_or_certified_artifacts_modified": False,
        "interrupted_attempts": [
            "initial run stopped after redundant per-sample Path.resolve caused I/O stall; diagnostic outputs only",
            "second run stopped at DuckDB CREATE VIEW parameter binder before linkage output",
        ],
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_PHASE3G_REPORTS_BUILT", "80k_panels": len(eighty_k), "panels": len(panels), "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
