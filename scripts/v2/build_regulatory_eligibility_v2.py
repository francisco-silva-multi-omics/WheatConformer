from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server_genotype_recovery.regulatory_eligibility_v2 import (  # noqa: E402
    build_gid_manifest,
    gid_panel_evidence,
    observation_counts,
    panel_readiness,
    panel_summary,
    split_support,
    status_summary,
    validate_contract,
)
from scripts.v2.phase5_parity_common import (  # noqa: E402
    ProtectedPathGuard,
    git_head,
    index_signature,
    sha256_file,
    write_json,
    write_tsv,
)


RELEASE_ID = "P5REV2_20260809_V1_274E41DF"
RELEASE_RELATIVE = Path("audit/v2/phase5_regulatory_eligibility_v2")
PHASE5 = Path("audit/v2/phase5_split_bound_kernel_validation_v2")
PARITY = Path("audit/v2/phase5_panel_environment_scenario_parity_extension_v2")
CIMMYT = Path("audit/v2/phase5_cimmyt_unimputed_recovery_v4")
PHASE3G = Path("audit/v2/phase3g_all_panel_genotype_linkage_audit_v2")
PROTOCOL = Path("server_genotype_recovery/regulatory_eligibility_v2_protocol.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_tsv(path: Path, guard: ProtectedPathGuard) -> pd.DataFrame:
    return pd.read_csv(guard.assert_allowed(path), sep="\t", dtype=str)


def read_parquet(path: Path, guard: ProtectedPathGuard, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(guard.assert_allowed(path), columns=columns)


def write_manifest(release: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(candidate for candidate in release.rglob("*") if candidate.is_file()):
        if path.name in {"output_manifest.tsv", "CLOSING_HASH_MANIFEST.tsv"}:
            continue
        rows.append(
            {
                "release_id": RELEASE_ID,
                "relative_path": path.relative_to(release).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    frame = pd.DataFrame(rows)
    write_tsv(release / "output_manifest.tsv", frame)
    return frame


def verify_state_signatures(states: pd.DataFrame, parity: Path, guard: ProtectedPathGuard) -> pd.DataFrame:
    rows = []
    for state in states.to_dict("records"):
        path = guard.assert_allowed(parity / str(state["training_gid_path"]))
        frame = pd.read_csv(path, sep="\t", dtype=str)
        column = "canonical_gid" if "canonical_gid" in frame.columns else frame.columns[0]
        ids = frame[column].fillna("").astype(str).tolist()
        observed = index_signature(ids)
        expected = str(state["training_gid_signature"])
        rows.append(
            {
                "state_id": state["state_id"],
                "training_gid_rows": len(ids),
                "expected_signature": expected,
                "observed_signature": observed,
                "status": "PASS" if observed == expected else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and freeze Stage-1 v2 regulatory eligibility.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=RELEASE_RELATIVE)
    args = parser.parse_args()
    root = args.root.resolve()
    release = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    try:
        os.mkdir(release)
    except FileExistsError as exc:
        raise SystemExit(f"FAIL_IF_EXISTS: {release}") from exc

    phase5 = root / PHASE5
    parity = root / PARITY
    cimmyt = root / CIMMYT
    phase3g = root / PHASE3G
    denylist = parity / "PROTECTED_PATH_DENYLIST.txt"
    guard = ProtectedPathGuard(root, denylist)

    protocol_path = guard.assert_allowed(root / PROTOCOL)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    parents = {
        "phase5": phase5 / "PHASE5_RELEASE_DECISION.json",
        "parity": parity / "PHASE5_PARITY_EXTENSION_DECISION.json",
        "cimmyt": cimmyt / "PHASE5_CIMMYT_UNIMPUTED_DECISION.json",
    }
    parent_values = {
        name: json.loads(guard.assert_allowed(path).read_text(encoding="utf-8"))
        for name, path in parents.items()
    }
    expected_parent_status = {
        "phase5": "PASS_PHASE5_KERNEL_VALIDATION",
        "parity": "PASS_PHASE5_PARITY_EXTENSION_WITH_EXPLICIT_COMPONENT_BLOCKERS",
        "cimmyt": "PASS_CIMMYT_UNIMPUTED_ANALYSIS_WITH_GLOBAL_MARKER_UNIVERSE_BLOCKER",
    }
    for name, status in expected_parent_status.items():
        if parent_values[name].get("status") != status:
            raise ValueError(f"Parent release {name} is not authoritative: {parent_values[name].get('status')}")

    opening = {
        "release_id": RELEASE_ID,
        "protocol_version": protocol["protocol_version"],
        "opened_at_utc": utc_now(),
        "code_commit": git_head(root),
        "stage1_version": "Stage-1 v2",
        "stage1_authority": parent_values["phase5"]["release_id"],
        "panel_and_scenario_authority": parent_values["parity"]["release_id"],
        "cimmyt_unimputed_authority": parent_values["cimmyt"]["release_id"],
        "phenotype_blind": True,
        "v1_regulatory_manifest_consumed": False,
        "protected_outcome_content_accessed": False,
    }
    write_json(release / "OPENING_RELEASE.json", opening)
    write_json(release / "REGULATORY_ELIGIBILITY_V2_PROTOCOL.json", protocol)

    genotype_registry = read_tsv(phase5 / "indices/genotype_entity_registry.tsv", guard)
    ka_registry = read_tsv(phase5 / "pedigree/ka_registry.tsv", guard)
    observations = read_parquet(
        phase5 / "indices/canonical_phase5_observation_index.parquet",
        guard,
        [
            "canonical_gid",
            "primary_weighted_training_eligible",
            "secondary_unweighted_training_eligible",
        ],
    )
    states = read_tsv(parity / "splits/state_registry.tsv", guard)
    accepted = read_parquet(parity / "genomic/accepted_mapping_manifest.parquet", guard)
    source_axis = read_tsv(parity / "genomic/panel_source_axis_audit.tsv", guard)
    overlap = read_tsv(parity / "genomic/panel_stage1_overlap.tsv", guard)
    seeds_consensus = read_tsv(parity / "genomic/seeds_gid_consensus_summary.tsv", guard)
    seeds_marker_axis = read_parquet(parity / "genomic/seeds_marker_axis.parquet", guard)
    cimmyt_qc = read_tsv(cimmyt / "genomic/cimmyt_primary_sample_qc.tsv", guard)
    cimmyt_marker_axis = read_parquet(cimmyt / "genomic/cimmyt_marker_axis.parquet", guard)
    unresolved_80k = read_parquet(
        phase3g / "dartseq80k_cross_panel_candidate_ledger.parquet", guard
    )

    seeds_qc_ids = set(
        seeds_consensus.loc[
            seeds_consensus["retained_for_component"].astype(str).str.lower().eq("true"),
            "canonical_gid",
        ].astype(str)
    )
    cimmyt_qc_ids = set(
        cimmyt_qc.loc[
            cimmyt_qc["passes_frozen_sample_call_rate"].astype(str).str.lower().eq("true"),
            "accepted_canonical_gid",
        ].astype(str)
    )
    genotype_ids = set(genotype_registry["canonical_gid"].astype(str))
    readiness = panel_readiness(
        source_axis,
        overlap,
        cimmyt_qc_gids=len(cimmyt_qc_ids),
        seeds_qc_gids=len(seeds_qc_ids),
    )
    evidence = gid_panel_evidence(
        genotype_ids,
        accepted,
        readiness,
        seeds_qc_ids=seeds_qc_ids,
        cimmyt_qc_ids=cimmyt_qc_ids,
        unresolved_80k=unresolved_80k,
    )
    counts = observation_counts(observations)
    manifest = build_gid_manifest(genotype_registry, counts, evidence)
    pedigree_ids = set(manifest.loc[manifest["pedigree_available"], "canonical_gid"])
    support = split_support(
        states,
        parity,
        readiness,
        evidence,
        pedigree_ids,
        minimum_training_gids=int(protocol["minimum_direct_training_gids_for_structural_support"]),
    )
    signatures = verify_state_signatures(states, parity, guard)
    checks = validate_contract(manifest, evidence, support, states)
    ka_state_ids = set(ka_registry.loc[ka_registry["status"].eq("PASS"), "state_id"].astype(str))
    required_state_ids = set(states["state_id"].astype(str))
    missing_ka_states = sorted(required_state_ids.difference(ka_state_ids))
    missing_ka_scenarios = sorted(
        states.loc[states["state_id"].isin(missing_ka_states), "scenario"].astype(str).unique()
    )
    marker_axis_summary = pd.DataFrame(
        [
            {
                "panel_id": "seeds_of_discovery_dartseq",
                "markers": len(seeds_marker_axis),
                "allele_axis_complete": bool(
                    seeds_marker_axis[["reference_allele", "alternate_allele"]]
                    .fillna("")
                    .astype(str)
                    .apply(lambda column: column.str.strip().ne(""))
                    .all(axis=1)
                    .all()
                ),
                "physical_coordinates_present": False,
                "reference_assembly_certified": False,
                "graph_projection_certified": False,
                "status": "ALLELES_READY_MARKER_TAGS_REQUIRE_REFERENCE_OR_GRAPH_ALIGNMENT",
            },
            {
                "panel_id": "cimmyt_bread_gbs_2013_2018",
                "markers": len(cimmyt_marker_axis),
                "allele_axis_complete": bool(
                    cimmyt_marker_axis["alleles"].fillna("").astype(str).str.contains("/").all()
                ),
                "physical_coordinates_present": bool(
                    cimmyt_marker_axis["chrom"].fillna("").astype(str).str.strip().ne("").all()
                    and pd.to_numeric(cimmyt_marker_axis["pos"], errors="coerce").gt(0).all()
                ),
                "reference_assembly_certified": bool(
                    (~cimmyt_marker_axis["assembly"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .isin({"", "NA", "N/A", "UNKNOWN"}))
                    .all()
                ),
                "graph_projection_certified": False,
                "status": "COORDINATES_AND_ALLELES_PRESENT_ASSEMBLY_REQUIRES_CERTIFICATION",
            },
        ]
    )
    checks = pd.concat(
        [
            checks,
            pd.DataFrame(
                [
                    {
                        "check": "training_gid_signatures",
                        "status": "PASS" if signatures["status"].eq("PASS").all() else "FAIL",
                        "detail": f"matched={int(signatures['status'].eq('PASS').sum())}/{len(signatures)}",
                    },
                    {
                        "check": "protected_inputs_not_opened",
                        "status": "PASS" if not (guard.audit_frame()["decision"] == "DENY").any() else "FAIL",
                        "detail": "all input reads passed the Phase-5 parity denylist",
                    },
                    {
                        "check": "v1_regulatory_manifest_not_inherited",
                        "status": "PASS",
                        "detail": "v1 artifacts were not read; Stage-1 v2 axes and split states were rebuilt",
                    },
                    {
                        "check": "existing_K_A_state_bindings",
                        "status": "PASS" if len(ka_state_ids) == 90 and ka_registry["status"].eq("PASS").all() else "FAIL",
                        "detail": f"certified={len(ka_state_ids)}; required={len(required_state_ids)}; extension_required={len(missing_ka_states)}",
                    },
                    {
                        "check": "K_A_extension_scope",
                        "status": "PASS"
                        if len(missing_ka_states) == 60
                        and missing_ka_scenarios == ["COUNTRY_HOLDOUT", "TEMPORAL_YEAR"]
                        else "FAIL",
                        "detail": f"missing={len(missing_ka_states)}; scenarios={';'.join(missing_ka_scenarios)}",
                    },
                    {
                        "check": "marker_axis_evidence",
                        "status": "PASS"
                        if marker_axis_summary["allele_axis_complete"].all()
                        and not marker_axis_summary["graph_projection_certified"].any()
                        else "FAIL",
                        "detail": "Seeds alleles and CIMMYT coordinates/alleles verified; no graph projection promoted",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    write_tsv(release / "regulatory_panel_readiness.tsv", readiness)
    write_tsv(release / "regulatory_marker_axis_summary.tsv", marker_axis_summary)
    evidence.to_parquet(release / "regulatory_gid_panel_evidence.parquet", index=False)
    manifest.to_parquet(release / "regulatory_gid_manifest.parquet", index=False)
    write_tsv(release / "regulatory_status_summary.tsv", status_summary(manifest))
    write_tsv(release / "regulatory_panel_summary.tsv", panel_summary(evidence))
    write_tsv(release / "regulatory_split_support.tsv", support)
    ka_support = (
        support[
            [
                "state_id",
                "scenario",
                "outer_fold",
                "inner_fold",
                "state_level",
                "training_state_gids",
                "training_pedigree_gids",
            ]
        ]
        .drop_duplicates("state_id")
        .sort_values("state_id")
    )
    ka_support["existing_certified_K_A_binding"] = ka_support["state_id"].isin(ka_state_ids)
    ka_support["phase6_binding_action"] = ka_support["existing_certified_K_A_binding"].map(
        {True: "REUSE_CERTIFIED_PHASE5_BINDING", False: "BUILD_AND_CERTIFY_FROM_FROZEN_K_A_OPERATOR"}
    )
    write_tsv(release / "regulatory_ka_split_support.tsv", ka_support)
    write_tsv(release / "training_gid_signature_validation.tsv", signatures)
    write_tsv(release / "validation_checks.tsv", checks)
    write_tsv(release / "protected_outcome_access_audit.tsv", guard.audit_frame())

    direct_ready = int(manifest["direct_observed_ready"].sum())
    direct_sparse = int(manifest["direct_sparse_candidate"].sum())
    pedigree = int(manifest["pedigree_available"].sum())
    kz_allowed = direct_ready > 0 and bool(support["phase6_kz_active"].any())
    gate = {
        "status": "PASS" if checks["status"].eq("PASS").all() else "FAIL",
        "stage1_version": "Stage-1 v2",
        "K_A_baseline_supported_gids": pedigree,
        "direct_sparse_candidate_gids": direct_sparse,
        "direct_observed_ready_gids": direct_ready,
        "phase6_K_A_baseline_allowed": True,
        "phase6_K_A_existing_certified_state_bindings": len(ka_state_ids),
        "phase6_K_A_required_state_bindings": len(required_state_ids),
        "phase6_K_A_state_bindings_to_build": len(missing_ka_states),
        "phase6_K_A_training_ready": not missing_ka_states,
        "phase6_K_z_candidate_allowed": kz_allowed,
        "phase6_K_z_blockers": [
            "no certified v2 reference or graph projection",
            "no certified regulatory-window overlap",
            "no certified direct regulatory embeddings",
        ],
        "pedigree_imputation_ready": False,
        "pedigree_imputation_reason": "direct embedding donors and confidence model are not yet certified",
        "recommended_phase6_action": "proceed with K_A baseline; exclude K_z from the current one-shot selection unless a new phenotype-blind prerequisite release is frozen before metrics",
        "phase6_training_prerequisite": "build and certify K_A bindings for the 60 temporal/country states before fitting those scenarios",
    }
    write_json(release / "regulatory_phase6_gate.json", gate)

    decision = {
        "release_id": RELEASE_ID,
        "status": (
            "PASS_REGULATORY_ELIGIBILITY_V2_WITH_KZ_DEFERRED"
            if checks["status"].eq("PASS").all() and not kz_allowed
            else "PASS_REGULATORY_ELIGIBILITY_V2_KZ_READY"
            if checks["status"].eq("PASS").all()
            else "FAIL_REGULATORY_ELIGIBILITY_V2"
        ),
        "decided_at_utc": utc_now(),
        "protocol_version": protocol["protocol_version"],
        "stage1_version": "Stage-1 v2",
        "canonical_stage1_v2_gids": len(manifest),
        "primary_stage1_v2_gids": int(manifest["in_primary_view"].sum()),
        "direct_sparse_candidate_gids": direct_sparse,
        "direct_observed_ready_gids": direct_ready,
        "pedigree_supported_gids": pedigree,
        "split_state_count": int(states["state_id"].nunique()),
        "validation_checks": len(checks),
        "failed_checks": int(checks["status"].eq("FAIL").sum()),
        "phase6_K_A_baseline_allowed": True,
        "phase6_K_A_existing_certified_state_bindings": len(ka_state_ids),
        "phase6_K_A_state_bindings_to_build": len(missing_ka_states),
        "phase6_K_z_candidate_allowed": kz_allowed,
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "final_holdout_outcomes_read": False,
        "model_training_performed": False,
        "v1_regulatory_manifest_consumed": False,
    }
    write_json(release / "REGULATORY_ELIGIBILITY_V2_DECISION.json", decision)

    input_rows = []
    allowed_inputs = guard.audit_frame()
    for relative_path in sorted(
        allowed_inputs.loc[allowed_inputs["decision"].eq("ALLOW"), "relative_path"].unique()
    ):
        path = root / relative_path
        if not path.is_file():
            continue
        input_rows.append(
            {
                "relative_path": path.resolve().relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_tsv(release / "OPENING_HASH_MANIFEST.tsv", pd.DataFrame(input_rows))
    output = write_manifest(release)
    closing = output.rename(columns={"relative_path": "relative_path"}).copy()
    closing["status"] = "PASS"
    write_tsv(release / "CLOSING_HASH_MANIFEST.tsv", closing)

    report = f"""# Stage-1 v2 regulatory eligibility freeze

- Release: `{RELEASE_ID}`
- Status: `{decision['status']}`
- Canonical Stage-1 v2 GIDs: {len(manifest):,}
- Primary Stage-1 v2 GIDs: {int(manifest['in_primary_view'].sum()):,}
- K_A-supported GIDs: {pedigree:,}
- Direct sparse marker candidates: {direct_sparse:,}
- Direct graph-projected embedding-ready GIDs: {direct_ready:,}
- Split states audited: {states['state_id'].nunique():,}
- Existing certified K_A state bindings: {len(ka_state_ids):,}/{len(required_state_ids):,}

The audit used identifiers, certified genotype/QC metadata, marker metadata, and training GID
lists only. It read no phenotype values, validation metrics, outer outcomes, or final holdout
outcomes. Stage-1 v1 regulatory artifacts were not inherited.

`K_A` remains eligible as the Phase-6 baseline. `K_z` is deferred because no v2 panel has
completed reference/graph projection, regulatory-window overlap certification, and direct
embedding certification. Missing `K_z` must remain an explicit mask; pedigree propagation is
a future training-only, confidence-gated candidate and is not equivalent to observed sequence.

Before training the temporal and country scenarios, construct and certify the {len(missing_ka_states)}
missing K_A split bindings from the already frozen pedigree operator and Stage-1 v2 training IDs.
"""
    (release / "REGULATORY_ELIGIBILITY_V2_REPORT.md").write_text(report, encoding="utf-8")

    # Rebind the report into the final manifest without recursively hashing manifests.
    output = write_manifest(release)
    closing = output.copy()
    closing["status"] = "PASS"
    write_tsv(release / "CLOSING_HASH_MANIFEST.tsv", closing)

    print(json.dumps(decision, indent=2))
    print("\n=== STATUS SUMMARY ===")
    print(status_summary(manifest).to_string(index=False))
    print("\n=== PHASE 6 GATE ===")
    print(json.dumps(gate, indent=2))
    if decision["status"].startswith("FAIL"):
        raise SystemExit("Regulatory eligibility v2 certification failed")


if __name__ == "__main__":
    main()
