from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from build_pedigree_kernel import additive_relationship, assert_relationship_valid
from server_genotype_recovery.build_canonical_pedigree import (
    REGISTRY_COLUMNS,
    build_resolution,
    read_manual_decisions,
    resolve,
    source_lineages,
    write_checksums,
)
from server_genotype_recovery.fetch_brapi_pedigree_markers import (
    clean,
    read_table,
    sha256_file,
)


ACCEPTED_EDGE_CLASSES = {
    "accepted_new_edge_exact_unique",
    "accepted_new_edge_corroborated",
}
STABLE_PARENT_ID = re.compile(r"^(?:GID[0-9]+|PED[FX]_[A-F0-9]{16})$")
EDGE_COLUMNS = {
    "child_id",
    "parent_role",
    "parent_id",
    "verification_class",
    "accepted",
    "is_new_edge",
}


def bool_value(value: object) -> bool:
    return clean(value).upper() in {"1", "TRUE", "T", "YES", "Y", "PASS"}


def require_nonempty(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required {label} is missing or empty: {path}")
    return path


def verify_sha256_manifest(directory: Path) -> dict[str, str]:
    manifest_path = require_nonempty(
        directory / "verification_sha256.tsv", "verification SHA256 manifest"
    )
    manifest = read_table(manifest_path)
    required = {"path", "sha256"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Verification SHA256 manifest is missing columns: {missing}")
    if manifest["path"].map(clean).duplicated().any():
        raise ValueError("Verification SHA256 manifest contains duplicate paths")
    observed: dict[str, str] = {}
    for row in manifest.itertuples(index=False):
        name = clean(row.path)
        expected = clean(row.sha256).lower()
        path = require_nonempty(directory / name, f"verified artifact {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Verified artifact hash mismatch for {name}: "
                f"expected={expected} observed={actual}"
            )
        observed[name] = actual
    return observed


def load_verified_bundle(
    directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, str]]:
    hashes = verify_sha256_manifest(directory)
    required_names = {
        "verification_contract.json",
        "verification_provenance.json",
        "verification_qc.tsv",
        "accepted_new_pedigree_edges.tsv",
        "verification_parent_registry.tsv",
    }
    missing = sorted(required_names - set(hashes))
    if missing:
        raise ValueError(f"Verification bundle is missing certified artifacts: {missing}")

    contract = json.loads(
        (directory / "verification_contract.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (directory / "verification_provenance.json").read_text(encoding="utf-8")
    )
    if contract.get("protocol_version") != "recovered_identity_verification_v2":
        raise ValueError("Canonical pedigree v3 requires recovered identity verification v2")
    if provenance.get("status") != "PASS":
        raise ValueError("Recovered identity verification did not pass")
    for key in (
        "phenotype_values_read",
        "outer_test_metrics_read",
        "final_holdout_outcomes_read",
        "model_performance_read",
        "kernels_modified",
        "single_step_H_constructed",
    ):
        if provenance.get(key) is not False:
            raise ValueError(f"Recovered verification safety flag is not false: {key}")

    qc = read_table(directory / "verification_qc.tsv")
    if "status" not in qc or qc["status"].map(clean).eq("FAIL").any():
        raise ValueError("Recovered identity verification QC contains a failure")
    edges = read_table(directory / "accepted_new_pedigree_edges.tsv")
    registry = read_table(directory / "verification_parent_registry.tsv")
    return edges, registry, provenance, hashes


def validate_accepted_edges(edges: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(EDGE_COLUMNS - set(edges.columns))
    if missing:
        raise ValueError(f"Accepted recovered edges are missing columns: {missing}")
    local = edges.copy()
    for column in ("child_id", "parent_role", "parent_id", "verification_class"):
        local[column] = local[column].map(clean)
    if local.empty:
        return local
    invalid = local[
        ~local["accepted"].map(bool_value)
        | ~local["is_new_edge"].map(bool_value)
        | ~local["verification_class"].isin(ACCEPTED_EDGE_CLASSES)
        | ~local["parent_role"].isin({"parent1", "parent2"})
        | local["child_id"].eq("")
        | local["parent_id"].eq("")
    ]
    if not invalid.empty:
        raise ValueError(
            "Recovered edge bundle contains rows outside the accepted edge contract"
        )
    bad_ids = sorted(
        {
            value
            for column in ("child_id", "parent_id")
            for value in local[column]
            if not STABLE_PARENT_ID.fullmatch(value)
        }
    )
    if bad_ids:
        raise ValueError(f"Recovered edges contain unstable IDs: {bad_ids[:10]}")
    duplicates = local.duplicated(["child_id", "parent_role"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Recovered edges contain multiple assignments for one child parent role: "
            f"{local.loc[duplicates, ['child_id', 'parent_role']].drop_duplicates().head().to_dict('records')}"
        )
    if (local["child_id"] == local["parent_id"]).any():
        raise ValueError("Recovered edges contain a self-parent assignment")
    return local.sort_values(
        ["child_id", "parent_role", "parent_id"], kind="stable"
    ).reset_index(drop=True)


def registry_closure(
    accepted_parent_ids: set[str], recovered_registry: pd.DataFrame
) -> pd.DataFrame:
    missing_columns = sorted(set(REGISTRY_COLUMNS) - set(recovered_registry.columns))
    if missing_columns:
        raise ValueError(
            f"Recovered parent registry is missing columns: {missing_columns}"
        )
    registry = recovered_registry.copy()
    for column in REGISTRY_COLUMNS:
        registry[column] = registry[column].map(clean)
    if registry["stable_parent_id"].duplicated().any():
        raise ValueError("Recovered parent registry contains duplicate stable IDs")
    by_id = registry.set_index("stable_parent_id", drop=False)
    required: set[str] = set()
    queue = deque(sorted(accepted_parent_ids))
    while queue:
        parent = queue.popleft()
        if parent in required or parent.startswith("GID"):
            continue
        if parent not in by_id.index:
            raise ValueError(
                f"Accepted recovered parent lacks a certified registry row: {parent}"
            )
        required.add(parent)
        row = by_id.loc[parent]
        for ancestor in (clean(row["parent1"]), clean(row["parent2"])):
            if ancestor and ancestor not in required:
                queue.append(ancestor)
    if not required:
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    return registry[registry["stable_parent_id"].isin(required)].sort_values(
        ["node_type", "stable_parent_id"], kind="stable"
    )


def merge_registries(base: pd.DataFrame, recovered: pd.DataFrame) -> pd.DataFrame:
    base_local = base.copy()
    recovered_local = recovered.copy()
    for frame in (base_local, recovered_local):
        for column in REGISTRY_COLUMNS:
            if column not in frame:
                frame[column] = ""
            frame[column] = frame[column].map(clean)
    base_by_id = base_local.set_index("stable_parent_id", drop=False)
    additions: list[dict[str, object]] = []
    for row in recovered_local.to_dict("records"):
        stable_id = clean(row["stable_parent_id"])
        if stable_id in base_by_id.index:
            existing = base_by_id.loc[stable_id]
            for column in ("node_type", "parent1", "parent2"):
                if clean(existing[column]) != clean(row[column]):
                    raise ValueError(
                        f"Recovered parent registry conflicts with v2 for {stable_id}: {column}"
                    )
            continue
        row["provenance"] = ";".join(
            filter(
                None,
                [
                    clean(row.get("provenance")),
                    "certified_recovered_identity_verification_v2",
                ],
            )
        )
        additions.append(row)
    merged = pd.concat(
        [base_local, pd.DataFrame(additions, columns=REGISTRY_COLUMNS)],
        ignore_index=True,
    )
    if merged["stable_parent_id"].duplicated().any():
        raise ValueError("Merged parent registry contains duplicate stable IDs")
    return merged.sort_values(["node_type", "stable_parent_id"], kind="stable")


def overlay_recovered_edges(
    pedigree: pd.DataFrame,
    resolution: pd.DataFrame,
    base_registry: pd.DataFrame,
    edges: pd.DataFrame,
    recovered_registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accepted = validate_accepted_edges(edges)
    child_ids = set(pedigree.loc[pedigree["node_role"].eq("trial_child"), "sample_id"])
    missing_children = sorted(set(accepted["child_id"]) - child_ids)
    if missing_children:
        raise ValueError(
            f"Recovered edges reference children absent from canonical v2: {missing_children[:10]}"
        )
    closure = registry_closure(set(accepted["parent_id"]), recovered_registry)
    merged_registry = merge_registries(base_registry, closure)

    output = pedigree.copy()
    output = output.set_index("sample_id", drop=False)
    audit_rows: list[dict[str, object]] = []
    changed_children: set[str] = set()
    for row in accepted.to_dict("records"):
        child = clean(row["child_id"])
        role = clean(row["parent_role"])
        parent = clean(row["parent_id"])
        previous = clean(output.at[child, role])
        if previous and previous != parent:
            raise ValueError(
                f"Recovered edge conflicts with canonical v2: child={child} "
                f"role={role} v2={previous} recovered={parent}"
            )
        status = "already_represented_in_canonical_v2"
        if not previous:
            output.at[child, role] = parent
            changed_children.add(child)
            status = "applied_certified_recovered_edge"
        audit_rows.append(
            {
                "child_id": child,
                "parent_role": role,
                "previous_parent_id": previous,
                "recovered_parent_id": parent,
                "verification_class": clean(row["verification_class"]),
                "overlay_status": status,
                "source_datasets": clean(row.get("source_datasets")),
                "source_files": clean(row.get("source_files")),
            }
        )

    existing_ids = set(output.index)
    registry_by_id = merged_registry.set_index("stable_parent_id", drop=False)
    required_parent_ids = {
        clean(value)
        for column in ("parent1", "parent2")
        for value in output[column]
        if clean(value)
    }
    for parent in sorted(required_parent_ids - existing_ids):
        if parent in registry_by_id.index:
            row = registry_by_id.loc[parent]
            output.loc[parent] = {
                "sample_id": parent,
                "parent1": clean(row["parent1"]),
                "parent2": clean(row["parent2"]),
                "node_role": clean(row["node_type"]),
                "resolution_status": "recovered_registry_node",
            }
        elif parent.startswith("GID"):
            output.loc[parent] = {
                "sample_id": parent,
                "parent1": "",
                "parent2": "",
                "node_role": "canonical_gid_founder",
                "resolution_status": "recovered_canonical_gid_founder",
            }
        else:
            raise ValueError(f"Recovered parent cannot be materialized: {parent}")
        existing_ids.add(parent)

    # A newly added registry node can introduce another ancestor level.
    while True:
        required_parent_ids = {
            clean(value)
            for column in ("parent1", "parent2")
            for value in output[column]
            if clean(value)
        }
        missing = sorted(required_parent_ids - set(output.index))
        if not missing:
            break
        for parent in missing:
            if parent in registry_by_id.index:
                row = registry_by_id.loc[parent]
                output.loc[parent] = {
                    "sample_id": parent,
                    "parent1": clean(row["parent1"]),
                    "parent2": clean(row["parent2"]),
                    "node_role": clean(row["node_type"]),
                    "resolution_status": "recovered_registry_node",
                }
            elif parent.startswith("GID"):
                output.loc[parent] = {
                    "sample_id": parent,
                    "parent1": "",
                    "parent2": "",
                    "node_role": "canonical_gid_founder",
                    "resolution_status": "recovered_canonical_gid_founder",
                }
            else:
                raise ValueError(f"Recovered ancestor cannot be materialized: {parent}")

    output = output.reset_index(drop=True).sort_values("sample_id", kind="stable")
    if output["sample_id"].duplicated().any():
        raise ValueError("Canonical pedigree v3 contains duplicate nodes")

    updated_resolution = resolution.copy().set_index("sample_id", drop=False)
    child_lookup = output.set_index("sample_id")
    for child in sorted(changed_children):
        updated_resolution.at[child, "selected_parent1"] = clean(
            child_lookup.at[child, "parent1"]
        )
        updated_resolution.at[child, "selected_parent2"] = clean(
            child_lookup.at[child, "parent2"]
        )
        updated_resolution.at[
            child, "resolution_status"
        ] = "resolved_with_certified_recovered_edge_overlay"
        prior = clean(updated_resolution.at[child, "review_reason"])
        updated_resolution.at[child, "review_reason"] = ";".join(
            filter(None, [prior, "certified recovered pedigree edge overlay v2"])
        )
    updated_resolution = updated_resolution.reset_index(drop=True)

    # This performs the definitive cycle/dependency check before any artifact is written.
    additive_relationship(output[["sample_id", "parent1", "parent2"]], None)
    audit = pd.DataFrame(
        audit_rows,
        columns=[
            "child_id",
            "parent_role",
            "previous_parent_id",
            "recovered_parent_id",
            "verification_class",
            "overlay_status",
            "source_datasets",
            "source_files",
        ],
    )
    return output, merged_registry, updated_resolution, audit


def write_kernel(
    pedigree: pd.DataFrame, out_dir: Path, prefix: str
) -> tuple[list[Path], dict[str, object]]:
    relationship, order, qc = additive_relationship(
        pedigree[["sample_id", "parent1", "parent2"]], None
    )
    mean_diagonal = float(np.diag(relationship).mean())
    relationship = (relationship / mean_diagonal).astype(np.float32)
    assert_relationship_valid(relationship, order)
    paths = {
        "kernel": out_dir / f"{prefix}.npy",
        "order": out_dir / f"{prefix}_sample_order.tsv",
        "row": out_dir / f"{prefix}_row_order.tsv",
        "column": out_dir / f"{prefix}_column_order.tsv",
        "qc": out_dir / f"{prefix}_qc.tsv",
    }
    np.save(paths["kernel"], relationship)
    pd.DataFrame(
        {"sample_id": order, "compact_kernel_index": np.arange(len(order))}
    ).to_csv(paths["order"], sep="\t", index=False)
    pd.DataFrame({"row_index": np.arange(len(order)), "sample_id": order}).to_csv(
        paths["row"], sep="\t", index=False
    )
    pd.DataFrame(
        {"column_index": np.arange(len(order)), "sample_id": order}
    ).to_csv(paths["column"], sep="\t", index=False)
    qc.loc[len(qc)] = ["scaled_mean_diagonal", float(np.diag(relationship).mean())]
    qc.to_csv(paths["qc"], sep="\t", index=False)
    metrics = {
        "K_A_shape": f"{relationship.shape[0]}x{relationship.shape[1]}",
        "K_A_mean_diagonal": float(np.diag(relationship).mean()),
        "K_A_min_diagonal": float(np.diag(relationship).min()),
        "K_A_max_diagonal": float(np.diag(relationship).max()),
    }
    return list(paths.values()), metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an isolated canonical pedigree v3 by overlaying only certified "
            "recovered pedigree edges onto the canonical v2 reconstruction."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("metadata_outputs/all_trials_genotype_manifest_resolved.tsv"),
    )
    parser.add_argument("--manual-lineage-decisions", type=Path)
    parser.add_argument(
        "--verified-identity-dir",
        type=Path,
        default=Path("genotype_panels/recovered_identity_verification_v2"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("genotype_panels/pedigree_canonical_v3")
    )
    parser.add_argument("--prefix", default="K_A_CANONICAL_V3")
    parser.add_argument("--allow-conservative-founder-fallback", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    source_path = resolve(root, args.source_manifest)
    verification_dir = resolve(root, args.verified_identity_dir)
    out_dir = resolve(root, args.out_dir)
    manual_path = (
        resolve(root, args.manual_lineage_decisions)
        if args.manual_lineage_decisions
        else None
    )
    require_nonempty(source_path, "source pedigree manifest")
    out_dir.mkdir(parents=True, exist_ok=True)

    edges, recovered_registry, verification, verification_hashes = load_verified_bundle(
        verification_dir
    )
    source = source_lineages(source_path)
    manual = read_manual_decisions(manual_path)
    pedigree, registry, resolution, selfing, base_metrics = build_resolution(
        source,
        manual,
        allow_conservative_founder_fallback=args.allow_conservative_founder_fallback,
    )
    blockers = base_metrics.pop("blockers")
    if not blockers.empty:
        raise ValueError(
            "Canonical v2 reconstruction remains blocked; recovered edges cannot bypass "
            "unreviewed base pedigree decisions"
        )
    pedigree, registry, resolution, overlay = overlay_recovered_edges(
        pedigree, resolution, registry, edges, recovered_registry
    )

    paths = {
        "pedigree": out_dir / "canonical_pedigree_parent_table.tsv",
        "registry": out_dir / "canonical_parent_registry.tsv",
        "resolution": out_dir / "child_lineage_resolution.tsv",
        "selfing": out_dir / "selfing_review.tsv",
        "blockers": out_dir / "pedigree_resolution_blockers.tsv",
        "manual_template": out_dir / "manual_lineage_decisions_template.tsv",
        "overlay": out_dir / "recovered_edge_overlay.tsv",
        "overlay_qc": out_dir / "recovered_edge_overlay_qc.tsv",
    }
    pedigree.to_csv(paths["pedigree"], sep="\t", index=False)
    registry.to_csv(paths["registry"], sep="\t", index=False)
    resolution.to_csv(paths["resolution"], sep="\t", index=False)
    selfing.to_csv(paths["selfing"], sep="\t", index=False)
    blockers.to_csv(paths["blockers"], sep="\t", index=False)
    resolution.loc[
        resolution["resolution_status"].isin(
            [
                "conservative_founder_due_conflicting_lineages",
                "founder_due_unreviewed_selfing",
            ]
        ),
        ["sample_id", "source_lineages", "resolution_status"],
    ].assign(
        decision="", selected_lineage="", reviewed=False, reviewer="", evidence=""
    ).to_csv(paths["manual_template"], sep="\t", index=False)
    overlay.to_csv(paths["overlay"], sep="\t", index=False)
    applied = (
        overlay["overlay_status"].eq("applied_certified_recovered_edge")
        if not overlay.empty
        else pd.Series(dtype=bool)
    )
    overlay_qc = pd.DataFrame(
        [
            {"metric": "certified_recovered_edge_rows", "value": len(overlay)},
            {
                "metric": "certified_recovered_children",
                "value": int(overlay["child_id"].nunique()) if not overlay.empty else 0,
            },
            {"metric": "applied_edge_rows", "value": int(applied.sum())},
            {
                "metric": "already_represented_edge_rows",
                "value": int((~applied).sum()) if not overlay.empty else 0,
            },
            {
                "metric": "applied_children",
                "value": int(overlay.loc[applied, "child_id"].nunique())
                if not overlay.empty
                else 0,
            },
            {"metric": "merged_parent_registry_rows", "value": len(registry)},
            {"metric": "canonical_pedigree_rows", "value": len(pedigree)},
            {"metric": "phenotype_values_read", "value": False},
            {"metric": "outer_test_metrics_read", "value": False},
            {"metric": "final_holdout_outcomes_read", "value": False},
        ]
    )
    overlay_qc.to_csv(paths["overlay_qc"], sep="\t", index=False)

    kernel_paths, kernel_metrics = write_kernel(pedigree, out_dir, args.prefix)
    decision = {
        "status": "PASS",
        "canonical_K_A_construction_allowed": True,
        "protocol_version": "canonical_trial_pedigree_v3_verified_recovery_overlay",
        "selection_data": "identifiers_pedigree_strings_and_certified_recovered_edges_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "allow_conservative_founder_fallback": args.allow_conservative_founder_fallback,
        "source_manifest": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "manual_lineage_decisions": (
            {"path": str(manual_path), "sha256": sha256_file(manual_path)}
            if manual_path is not None
            else None
        ),
        "recovered_identity_verification": {
            "directory": str(verification_dir),
            "protocol_version": verification.get("protocol_version"),
            "verification_provenance_sha256": verification_hashes[
                "verification_provenance.json"
            ],
            "accepted_edges_sha256": verification_hashes[
                "accepted_new_pedigree_edges.tsv"
            ],
            "parent_registry_sha256": verification_hashes[
                "verification_parent_registry.tsv"
            ],
            "verification_sha256_manifest": sha256_file(
                verification_dir / "verification_sha256.tsv"
            ),
        },
        "metrics": {
            **base_metrics,
            "certified_recovered_edge_rows": len(overlay),
            "certified_recovered_children": int(overlay["child_id"].nunique())
            if not overlay.empty
            else 0,
            "applied_recovered_edge_rows": int(applied.sum()),
            "applied_recovered_children": int(
                overlay.loc[applied, "child_id"].nunique()
            )
            if not overlay.empty
            else 0,
            "canonical_pedigree_v3_rows": len(pedigree),
            "stable_parent_registry_v3_rows": len(registry),
            **kernel_metrics,
        },
        "blocking_reasons": [],
        "interpretation_contract": {
            "canonical_v2_artifacts_modified": False,
            "unverified_recovered_edges_included": False,
            "existing_nonempty_parent_overwritten": False,
            "recovered_parent_registry_closure_required": True,
            "recovered_edges_selected_using_outcomes": False,
        },
    }
    decision_path = out_dir / "canonical_pedigree_decision.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    checksum_path = out_dir / "canonical_pedigree_artifacts.sha256"
    write_checksums(
        [*paths.values(), *kernel_paths, decision_path], checksum_path, root
    )
    print(json.dumps(decision, indent=2, allow_nan=False))
    print("\n=== RECOVERED EDGE OVERLAY ===")
    print(overlay_qc.to_string(index=False))


if __name__ == "__main__":
    main()
