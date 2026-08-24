from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from server_genotype_recovery.adjudicate_marker_identity_candidates import (
    CANDIDATE_COLUMNS,
    PAIR_COLUMNS,
    accepted_entities,
    adjudication_qc,
    classification_summary,
    load_certified_panel_ids,
    load_policy,
    regulatory_overlay,
    resolve,
    validation_checks,
)
from server_genotype_recovery.fetch_brapi_pedigree_markers import (
    read_table,
    sha256_file,
    write_json_atomic,
)


CLASSIFICATION_EVIDENCE_COLUMNS = [
    "trial_gid",
    "candidate_scope",
    "panel_id",
    "sample_id",
    "classification",
    "classification_reasons",
    "direct_marker_assignment_ready",
]


def boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    invalid = sorted(set(normalized) - {"", "0", "1", "false", "true", "no", "yes"})
    if invalid:
        raise ValueError(f"Cannot interpret boolean values: {invalid[:10]}")
    return normalized.isin({"1", "true", "yes"})


def classification_evidence_sha256(candidates: pd.DataFrame) -> str:
    missing = sorted(set(CLASSIFICATION_EVIDENCE_COLUMNS) - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate artifact lacks classification evidence columns: {missing}")
    local = candidates[CLASSIFICATION_EVIDENCE_COLUMNS].copy()
    local["direct_marker_assignment_ready"] = boolean_series(
        local["direct_marker_assignment_ready"]
    ).map({True: "1", False: "0"})
    for column in CLASSIFICATION_EVIDENCE_COLUMNS[:-1]:
        local[column] = local[column].fillna("").astype(str)
    local = local.sort_values(CLASSIFICATION_EVIDENCE_COLUMNS, kind="stable")
    payload = local.to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pairwise_evidence_sha256(pairs: pd.DataFrame) -> str:
    missing = sorted(set(PAIR_COLUMNS) - set(pairs.columns))
    if missing:
        raise ValueError(f"Pairwise artifact lacks evidence columns: {missing}")
    local = pairs[PAIR_COLUMNS].copy()
    for column in ("overlap_pass", "concordance_pass"):
        local[column] = boolean_series(local[column]).map({True: "1", False: "0"})
    for column in (set(PAIR_COLUMNS) - {"overlap_pass", "concordance_pass"}):
        local[column] = local[column].fillna("").astype(str)
    local = local.sort_values(PAIR_COLUMNS, kind="stable")
    payload = local.to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def apply_certified_panel_flags(
    candidates: pd.DataFrame,
    *,
    policy: dict[str, object],
    certified_ids: dict[str, set[str]],
) -> pd.DataFrame:
    local = candidates.copy()
    required = {
        "trial_gid",
        "panel_id",
        "classification",
        "direct_marker_assignment_ready",
    }
    missing = sorted(required - set(local.columns))
    if missing:
        raise ValueError(f"Candidate artifact lacks required columns: {missing}")

    new_spec = dict(policy["new_candidate_panel"])
    new_panel = str(new_spec["panel_id"])
    panel_reference = {
        panel_id: panel_id for panel_id in certified_ids
    }
    panel_reference[new_panel] = str(
        new_spec.get("certified_panel_reference", new_panel)
    )
    unknown = sorted(set(local["panel_id"].fillna("").astype(str)) - set(panel_reference))
    if unknown:
        raise ValueError(f"No certified-panel reference is defined for candidate panels: {unknown}")

    local["trial_gid"] = local["trial_gid"].fillna("").astype(str).str.strip()
    local["panel_id"] = local["panel_id"].fillna("").astype(str).str.strip()
    local["direct_marker_assignment_ready"] = boolean_series(
        local["direct_marker_assignment_ready"]
    )
    local["certified_panel_reference"] = local["panel_id"].map(panel_reference)
    local["existing_certified_in_panel"] = [
        gid in certified_ids.get(reference, set())
        for gid, reference in zip(
            local["trial_gid"], local["certified_panel_reference"], strict=True
        )
    ]
    any_certified = set().union(*certified_ids.values()) if certified_ids else set()
    local["existing_certified_in_any_panel"] = local["trial_gid"].isin(any_certified)
    local = local.drop(columns=["existing_certified_gid"], errors="ignore")

    missing_contract = sorted(set(CANDIDATE_COLUMNS) - set(local.columns))
    if missing_contract:
        raise ValueError(
            f"Reconciled candidate artifact does not satisfy the current contract: {missing_contract}"
        )
    extra = [column for column in local.columns if column not in CANDIDATE_COLUMNS]
    return local[CANDIDATE_COLUMNS + extra]


def certified_order_provenance(
    root: Path, policy: dict[str, object]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in policy.get("existing_panel_artifacts", []):
        artifact_dir = resolve(root, str(spec["artifact_dir"]))
        path = artifact_dir / f"{spec['prefix']}_sample_order.tsv"
        rows.append(
            {
                "panel_id": str(spec["panel_id"]),
                "path": str(path),
                "present": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else "",
            }
        )
    for spec in policy.get("direct_certified_panel_orders", []):
        path = resolve(root, str(spec["sample_order_path"]))
        rows.append(
            {
                "panel_id": str(spec["panel_id"]),
                "path": str(path),
                "present": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else "",
            }
        )
    return rows


def write_outputs(
    *,
    out_dir: Path,
    candidates: pd.DataFrame,
    pairs: pd.DataFrame,
    inventory: pd.DataFrame,
    checks: pd.DataFrame,
    protocol_version: str,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted = candidates[candidates["direct_marker_assignment_ready"]].copy()
    unresolved = candidates[
        candidates["classification"].isin(
            {"requires_metadata_review", "family_only_not_assignable"}
        )
    ].copy()
    conflicting = candidates[
        candidates["classification"].eq("conflicting_marker_samples")
    ].copy()
    outputs = {
        "candidates": out_dir / "marker_identity_candidate_paths.tsv.gz",
        "pairs": out_dir / "marker_identity_pairwise_concordance.tsv.gz",
        "accepted": out_dir / "marker_identity_accepted.tsv",
        "accepted_entities": out_dir / "marker_identity_accepted_entities.tsv",
        "unresolved": out_dir / "marker_identity_unresolved.tsv",
        "conflicting": out_dir / "marker_identity_conflicting.tsv",
        "overlay": out_dir / "regulatory_eligibility_overlay.tsv",
        "summary": out_dir / "marker_identity_classification_summary.tsv",
        "inventory": out_dir / "marker_identity_panel_inventory.tsv",
        "validation": out_dir / "marker_identity_validation.tsv",
        "qc": out_dir / "marker_identity_adjudication_qc.tsv",
    }
    candidates.to_csv(outputs["candidates"], sep="\t", index=False, compression="gzip")
    pairs.to_csv(outputs["pairs"], sep="\t", index=False, compression="gzip")
    accepted.to_csv(outputs["accepted"], sep="\t", index=False)
    accepted_entities(candidates).to_csv(
        outputs["accepted_entities"], sep="\t", index=False
    )
    unresolved.to_csv(outputs["unresolved"], sep="\t", index=False)
    conflicting.to_csv(outputs["conflicting"], sep="\t", index=False)
    regulatory_overlay(candidates).to_csv(outputs["overlay"], sep="\t", index=False)
    classification_summary(candidates).to_csv(outputs["summary"], sep="\t", index=False)
    inventory.to_csv(outputs["inventory"], sep="\t", index=False)
    checks.to_csv(outputs["validation"], sep="\t", index=False)
    adjudication_qc(
        candidates, protocol_version=protocol_version
    ).to_csv(outputs["qc"], sep="\t", index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile panel-local and global novelty reporting without rereading marker calls "
            "or changing completed identity classifications."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(
            "server_genotype_recovery/marker_identity_concordance_policy_v1.json"
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("genotype_panels/marker_identity_adjudication_v1"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("genotype_panels/marker_identity_adjudication_v1_reconciled"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    policy_path = resolve(root, args.policy)
    source_dir = resolve(root, args.source_dir)
    out_dir = resolve(root, args.out_dir)
    if source_dir == out_dir:
        raise ValueError("Reconciliation output must be isolated from the source directory")
    source_paths = {
        "candidates": source_dir / "marker_identity_candidate_paths.tsv.gz",
        "pairs": source_dir / "marker_identity_pairwise_concordance.tsv.gz",
        "inventory": source_dir / "marker_identity_panel_inventory.tsv",
        "provenance": source_dir / "marker_identity_adjudication_provenance.json",
    }
    required = [source_paths["candidates"], source_paths["pairs"], source_paths["inventory"]]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Completed adjudication artifacts are missing: {missing}")

    policy = load_policy(policy_path)
    source_candidates = read_table(source_paths["candidates"])
    source_evidence_hash = classification_evidence_sha256(source_candidates)
    certified_ids = load_certified_panel_ids(
        root, policy, require_all_orders=True
    )
    candidates = apply_certified_panel_flags(
        source_candidates, policy=policy, certified_ids=certified_ids
    )
    reconciled_evidence_hash = classification_evidence_sha256(candidates)
    if source_evidence_hash != reconciled_evidence_hash:
        raise ValueError("Reconciliation changed identity classifications or assignment decisions")

    pairs = read_table(source_paths["pairs"])
    source_pairwise_hash = pairwise_evidence_sha256(pairs)
    for column in ("overlap_pass", "concordance_pass"):
        if column in pairs.columns:
            pairs[column] = boolean_series(pairs[column])
    reconciled_pairwise_hash = pairwise_evidence_sha256(pairs)
    if source_pairwise_hash != reconciled_pairwise_hash:
        raise ValueError("Reconciliation changed pairwise marker-call concordance evidence")
    inventory = read_table(source_paths["inventory"])
    overlay = regulatory_overlay(candidates)
    expected_new = set(
        candidates.loc[
            candidates["candidate_scope"].eq("new_dataverse_two_hop"), "trial_gid"
        ]
    )
    checks = validation_checks(candidates, pairs, overlay, expected_new)
    checks = pd.concat(
        [
            checks,
            pd.DataFrame(
                [
                    {
                        "check": "classification_evidence_preserved",
                        "status": "PASS",
                        "detail": source_evidence_hash,
                    },
                    {
                        "check": "pairwise_concordance_evidence_preserved",
                        "status": "PASS",
                        "detail": source_pairwise_hash,
                    },
                    {
                        "check": "panel_membership_implies_global_membership",
                        "status": (
                            "PASS"
                            if (
                                ~candidates["existing_certified_in_panel"]
                                | candidates["existing_certified_in_any_panel"]
                            ).all()
                            else "FAIL"
                        ),
                        "detail": (
                            f"panel_members={int(candidates['existing_certified_in_panel'].sum())}; "
                            f"global_members={int(candidates['existing_certified_in_any_panel'].sum())}"
                        ),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    if checks["status"].eq("FAIL").any():
        failed = checks[checks["status"].eq("FAIL")].to_dict("records")
        raise ValueError(f"Marker identity reporting reconciliation failed: {failed}")

    outputs = write_outputs(
        out_dir=out_dir,
        candidates=candidates,
        pairs=pairs,
        inventory=inventory,
        checks=checks,
        protocol_version=str(policy["protocol_version"]),
    )
    if source_paths["provenance"].is_file():
        source_provenance = json.loads(
            source_paths["provenance"].read_text(encoding="utf-8")
        )
        write_json_atomic(
            source_provenance,
            out_dir / "marker_identity_source_adjudication_provenance.json",
        )
    provenance = {
        "status": "PASS",
        "protocol_version": policy["protocol_version"],
        "selection_data": "completed_identity_classifications_and_certified_panel_orders_only",
        "source_adjudication_dir": str(source_dir),
        "source_artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
            if path.is_file()
        },
        "policy_path": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "certified_panel_orders": certified_order_provenance(root, policy),
        "classification_evidence_sha256_before": source_evidence_hash,
        "classification_evidence_sha256_after": reconciled_evidence_hash,
        "pairwise_concordance_sha256_before": source_pairwise_hash,
        "pairwise_concordance_sha256_after": reconciled_pairwise_hash,
        "classification_evidence_reused": True,
        "classification_or_concordance_modified": False,
        "marker_calls_read": False,
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "kernels_modified": False,
        "output_hashes": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in outputs.items()
        },
    }
    write_json_atomic(
        provenance, out_dir / "marker_identity_reporting_reconciliation.json"
    )
    write_json_atomic(
        provenance, out_dir / "marker_identity_adjudication_provenance.json"
    )
    qc = adjudication_qc(
        candidates, protocol_version=str(policy["protocol_version"])
    )
    print(qc.to_string(index=False))
    print("\n=== RECONCILIATION VALIDATION ===")
    print(checks.to_string(index=False))
    print(f"\nReconciled identity reports: {out_dir}")


if __name__ == "__main__":
    main()
