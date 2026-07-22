from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from build_pedigree_kernel import additive_relationship, assert_relationship_valid
from server_genotype_recovery.build_regulatory_eligibility_manifest import detect_column
from server_genotype_recovery.fetch_brapi_pedigree_markers import (
    clean,
    read_table,
    sha256_file,
)


CANONICAL_GID = re.compile(r"^GID[0-9]+$", re.IGNORECASE)
NUMBERED_CROSS = re.compile(r"/([3-9][0-9]*)/")
SINGLE_SLASH = re.compile(r"(?<!/)/(?!/)")
TEXT_CROSS = re.compile(r"\s+[Xx]\s+")
LEFT_RECURRENT = re.compile(r"^(.+?)\*([2-9][0-9]*)$")
RIGHT_RECURRENT = re.compile(r"^([2-9][0-9]*)\*(.+)$")
REGISTRY_COLUMNS = [
    "stable_parent_id",
    "node_type",
    "source_expression",
    "normalized_expression",
    "parent1",
    "parent2",
    "identity_scope",
    "construction_eligible",
    "accepted_by_policy",
    "human_reviewed",
    "provenance",
]
RESOLUTION_COLUMNS = [
    "sample_id",
    "source_lineage_count",
    "source_lineages",
    "structural_lineage_count",
    "selected_lineage",
    "selected_parent1",
    "selected_parent2",
    "resolution_status",
    "construction_eligible",
    "human_reviewed",
    "review_reason",
]
SELFING_COLUMNS = [
    "sample_id",
    "selected_lineage",
    "parent_id",
    "resolution_status",
    "construction_eligible",
    "human_reviewed",
]


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()


def bool_value(value: object) -> bool:
    return clean(value).upper() in {"1", "TRUE", "T", "YES", "Y", "PASS", "ACCEPTED"}


def normalize_expression(value: object) -> str:
    text = clean(value).upper()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*\\\s*", "/", text)
    return text.strip()


def canonical_gid(value: object) -> str:
    text = clean(value).upper()
    match = re.fullmatch(r"GID([0-9]+)(?:\.0+)?", text)
    return f"GID{int(match.group(1))}" if match else ""


def stable_digest(prefix: str, payload: str) -> str:
    digest = hashlib.sha256(f"{prefix}|{payload}".encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest.upper()}"


@dataclass(frozen=True)
class PedigreeNode:
    kind: str
    expression: str
    left: "PedigreeNode | None" = None
    right: "PedigreeNode | None" = None

    @property
    def structural_key(self) -> str:
        if self.kind == "cross":
            assert self.left is not None and self.right is not None
            return f"X({self.left.structural_key},{self.right.structural_key})"
        gid = canonical_gid(self.expression)
        if gid:
            return f"G:{gid}"
        return f"L:{normalize_expression(self.expression)}"


def root_cross_split(expression: str) -> tuple[str, str, str] | None:
    numbered = list(NUMBERED_CROSS.finditer(expression))
    if numbered:
        highest = max(int(match.group(1)) for match in numbered)
        matches = [match for match in numbered if int(match.group(1)) == highest]
        if len(matches) != 1:
            return None
        match = matches[0]
        left, right = expression[: match.start()], expression[match.end() :]
        return (left, right, f"numbered_cross_{highest}") if left and right else None

    if expression.count("//") == 1:
        left, right = expression.split("//", 1)
        return (left, right, "second_cross") if left and right else None
    if expression.count("//") > 1:
        return None

    slashes = list(SINGLE_SLASH.finditer(expression))
    if len(slashes) == 1:
        match = slashes[0]
        left, right = expression[: match.start()], expression[match.end() :]
        return (left, right, "first_cross") if left and right else None
    if len(slashes) > 1:
        return None

    text_crosses = list(TEXT_CROSS.finditer(expression))
    if len(text_crosses) == 1:
        match = text_crosses[0]
        left, right = expression[: match.start()], expression[match.end() :]
        return (left, right, "text_cross") if left and right else None
    return None


def parse_purdy_pedigree(value: object) -> PedigreeNode:
    expression = normalize_expression(value)
    if not expression:
        raise ValueError("Pedigree expression is empty")
    split = root_cross_split(expression)
    if split is None:
        return PedigreeNode("leaf", expression)

    left_expression, right_expression, _ = split
    left_expression = normalize_expression(left_expression)
    right_expression = normalize_expression(right_expression)
    left_dose: int | None = None
    right_dose: int | None = None

    left_match = LEFT_RECURRENT.fullmatch(left_expression)
    if left_match:
        left_expression = normalize_expression(left_match.group(1))
        left_dose = int(left_match.group(2))
    right_match = RIGHT_RECURRENT.fullmatch(right_expression)
    if right_match:
        right_dose = int(right_match.group(1))
        right_expression = normalize_expression(right_match.group(2))
    if left_dose is not None and right_dose is not None:
        return PedigreeNode("opaque", expression)

    left = parse_purdy_pedigree(left_expression)
    right = parse_purdy_pedigree(right_expression)
    result = PedigreeNode("cross", expression, left, right)
    if left_dose is not None:
        for dose in range(2, left_dose + 1):
            result = PedigreeNode(
                "cross", f"{left_expression}*{dose}/{right_expression}", left, result
            )
    if right_dose is not None:
        for dose in range(2, right_dose + 1):
            result = PedigreeNode(
                "cross", f"{left_expression}/{dose}*{right_expression}", result, right
            )
    return result


def read_manual_decisions(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file() or path.stat().st_size == 0:
        return {}
    frame = read_table(path)
    required = {"sample_id", "decision", "reviewed"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manual lineage decisions are missing columns: {missing}")
    accepted = frame[frame["reviewed"].map(bool_value)].copy()
    duplicated = accepted[accepted["sample_id"].map(clean).duplicated(keep=False)]
    if not duplicated.empty:
        raise ValueError("Manual lineage decisions contain repeated reviewed sample IDs")
    return {
        clean(row["sample_id"]): {key: clean(value) for key, value in row.items()}
        for row in accepted.to_dict("records")
        if clean(row["sample_id"])
    }


class RegistryBuilder:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}

    def materialize(self, node: PedigreeNode) -> str:
        if node.kind != "cross":
            gid = canonical_gid(node.expression)
            if gid:
                stable_id = gid
                node_type = "canonical_gid_founder"
                scope = "global_canonical_gid"
            else:
                stable_id = stable_digest("PEDF", node.structural_key)
                node_type = (
                    "opaque_founder_expression"
                    if node.kind == "opaque"
                    else "named_founder_designation"
                )
                scope = "stable_local_designation_not_verified_global_germplasm"
            candidate = {
                "stable_parent_id": stable_id,
                "node_type": node_type,
                "source_expression": node.expression,
                "normalized_expression": normalize_expression(node.expression),
                "parent1": "",
                "parent2": "",
                "identity_scope": scope,
                "construction_eligible": True,
                "accepted_by_policy": True,
                "human_reviewed": False,
                "provenance": "deterministic_exact_source_designation",
                "_structural_key": node.structural_key,
            }
            existing = self.rows.get(stable_id)
            if existing is not None and existing["_structural_key"] != node.structural_key:
                raise ValueError(f"Stable founder ID collision: {stable_id}")
            self.rows.setdefault(stable_id, candidate)
            return stable_id

        assert node.left is not None and node.right is not None
        parent1 = self.materialize(node.left)
        parent2 = self.materialize(node.right)
        if parent1 == parent2:
            stable_id = stable_digest("PEDF", f"UNREVIEWED_SELFING|{node.structural_key}")
            candidate = {
                "stable_parent_id": stable_id,
                "node_type": "opaque_unreviewed_selfing_subtree",
                "source_expression": node.expression,
                "normalized_expression": normalize_expression(node.expression),
                "parent1": "",
                "parent2": "",
                "identity_scope": "conservative_local_founder_not_verified_global_germplasm",
                "construction_eligible": True,
                "accepted_by_policy": True,
                "human_reviewed": False,
                "provenance": "duplicate_parent_subtree_collapsed_to_founder",
                "_structural_key": node.structural_key,
            }
            existing = self.rows.get(stable_id)
            if existing is not None and existing["_structural_key"] != node.structural_key:
                raise ValueError(f"Stable selfing fallback ID collision: {stable_id}")
            self.rows.setdefault(stable_id, candidate)
            return stable_id
        stable_id = stable_digest("PEDX", node.structural_key)
        candidate = {
            "stable_parent_id": stable_id,
            "node_type": "derived_purdy_cross_node",
            "source_expression": node.expression,
            "normalized_expression": normalize_expression(node.expression),
            "parent1": parent1,
            "parent2": parent2,
            "identity_scope": "stable_local_cross_node_not_verified_external_gid",
            "construction_eligible": True,
            "accepted_by_policy": True,
            "human_reviewed": False,
            "provenance": "deterministic_purdy_cross_structure",
            "_structural_key": node.structural_key,
        }
        existing = self.rows.get(stable_id)
        if existing is not None and existing["_structural_key"] != node.structural_key:
            raise ValueError(f"Stable cross ID collision: {stable_id}")
        self.rows.setdefault(stable_id, candidate)
        return stable_id

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=REGISTRY_COLUMNS)
        return pd.DataFrame(self.rows.values(), columns=REGISTRY_COLUMNS).sort_values(
            ["node_type", "stable_parent_id"], kind="stable"
        )


def source_lineages(path: Path) -> pd.DataFrame:
    frame = read_table(path)
    id_col = detect_column(
        frame,
        ["sample_id", "panel_sample_id_expected", "panel_sample_id", "genotype_id"],
    )
    cross_col = detect_column(frame, ["cross_name", "cross", "pedigree", "designation"])
    if id_col is None or cross_col is None:
        raise ValueError("Source manifest lacks a sample ID or cross/pedigree column")
    output = pd.DataFrame(
        {
            "sample_id": frame[id_col].map(clean).str.upper(),
            "source_lineage": frame[cross_col].map(clean),
        }
    )
    output = output[output["sample_id"].ne("")]
    invalid = output.loc[
        ~output["sample_id"].str.fullmatch(r"GID[0-9]+"), "sample_id"
    ].unique()
    if len(invalid):
        raise ValueError(f"Source manifest contains noncanonical child IDs: {invalid[:10]}")
    return output.drop_duplicates().reset_index(drop=True)


def selected_tree(
    sample_id: str,
    lineages: list[str],
    manual: dict[str, dict[str, str]],
) -> tuple[PedigreeNode | None, str, bool, str, str]:
    decision = manual.get(sample_id)
    if decision:
        action = clean(decision.get("decision")).lower()
        if action == "treat_as_founder":
            return None, "manual_founder", True, "manual decision", ""
        if action == "accept_lineage":
            selected = clean(decision.get("selected_lineage"))
            if selected not in lineages:
                raise ValueError(
                    f"Manual lineage for {sample_id} is not present in the source manifest"
                )
            return (
                parse_purdy_pedigree(selected),
                "manual_lineage",
                True,
                "manual decision",
                selected,
            )
        raise ValueError(f"Unsupported manual decision for {sample_id}: {action!r}")

    if not lineages:
        return None, "founder_no_source_lineage", False, "no source lineage", ""
    parsed = [(lineage, parse_purdy_pedigree(lineage)) for lineage in lineages]
    structural = {node.structural_key for _, node in parsed}
    if len(structural) == 1:
        lineage, node = parsed[0]
        if node.kind == "cross":
            status = (
                "resolved_equivalent_source_lineages"
                if len(lineages) > 1
                else "resolved_unique_source_lineage"
            )
            return node, status, False, "deterministic Purdy structure", lineage
        return None, "founder_source_designation_only", False, "no cross structure", lineage
    return (
        None,
        "conservative_founder_due_conflicting_lineages",
        False,
        "multiple non-equivalent source lineages",
        "",
    )


def build_resolution(
    source: pd.DataFrame,
    manual: dict[str, dict[str, str]],
    *,
    allow_conservative_founder_fallback: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    registry = RegistryBuilder()
    child_rows: list[dict[str, object]] = []
    resolution_rows: list[dict[str, object]] = []
    selfing_rows: list[dict[str, object]] = []
    blocker_rows: list[dict[str, object]] = []

    for sample_id, group in source.groupby("sample_id", sort=True):
        lineages = sorted(
            {clean(value) for value in group["source_lineage"] if clean(value)},
            key=normalize_expression,
        )
        tree, status, reviewed, reason, selected_lineage = selected_tree(
            sample_id, lineages, manual
        )
        parent1 = ""
        parent2 = ""
        structural_count = len(
            {parse_purdy_pedigree(lineage).structural_key for lineage in lineages}
        )
        eligible = True
        if tree is not None and tree.kind == "cross":
            assert tree.left is not None and tree.right is not None
            parent1 = registry.materialize(tree.left)
            parent2 = registry.materialize(tree.right)
            if parent1 == parent2:
                manual_selfing = (
                    reviewed
                    and clean(manual[sample_id].get("decision")).lower()
                    == "accept_lineage"
                )
                selfing_status = (
                    "accepted_reviewed_selfing"
                    if manual_selfing
                    else "founder_due_unreviewed_selfing"
                )
                if not manual_selfing:
                    parent1 = ""
                    parent2 = ""
                    status = selfing_status
                    reason = "duplicate immediate parent requires explicit selfing review"
                    eligible = allow_conservative_founder_fallback
                selfing_rows.append(
                    {
                        "sample_id": sample_id,
                        "selected_lineage": selected_lineage,
                        "parent_id": tree.left.structural_key,
                        "resolution_status": selfing_status,
                        "construction_eligible": eligible,
                        "human_reviewed": manual_selfing,
                    }
                )
        if status == "conservative_founder_due_conflicting_lineages":
            eligible = allow_conservative_founder_fallback
        if not eligible:
            blocker_rows.append(
                {"sample_id": sample_id, "resolution_status": status, "reason": reason}
            )
        child_rows.append(
            {
                "sample_id": sample_id,
                "parent1": parent1,
                "parent2": parent2,
                "node_role": "trial_child",
                "resolution_status": status,
            }
        )
        resolution_rows.append(
            {
                "sample_id": sample_id,
                "source_lineage_count": len(lineages),
                "source_lineages": ";".join(lineages),
                "structural_lineage_count": structural_count,
                "selected_lineage": selected_lineage,
                "selected_parent1": parent1,
                "selected_parent2": parent2,
                "resolution_status": status,
                "construction_eligible": eligible,
                "human_reviewed": reviewed,
                "review_reason": reason,
            }
        )

    registry_frame = registry.frame()
    child_ids = set(source["sample_id"])
    ancestor_rows = []
    for row in registry_frame.to_dict("records"):
        node_id = clean(row["stable_parent_id"])
        if node_id in child_ids:
            continue
        ancestor_rows.append(
            {
                "sample_id": node_id,
                "parent1": clean(row["parent1"]),
                "parent2": clean(row["parent2"]),
                "node_role": clean(row["node_type"]),
                "resolution_status": "registry_node",
            }
        )
    pedigree = pd.concat(
        [pd.DataFrame(child_rows), pd.DataFrame(ancestor_rows)], ignore_index=True
    )
    if pedigree["sample_id"].duplicated().any():
        duplicates = pedigree.loc[pedigree["sample_id"].duplicated(False), "sample_id"]
        raise ValueError(
            "Canonical pedigree contains duplicate node rows: "
            f"{duplicates.head().tolist()}"
        )

    resolution = pd.DataFrame(resolution_rows, columns=RESOLUTION_COLUMNS)
    selfing = pd.DataFrame(selfing_rows, columns=SELFING_COLUMNS)
    blockers = pd.DataFrame(
        blocker_rows, columns=["sample_id", "resolution_status", "reason"]
    )
    metrics = {
        "source_child_count": source["sample_id"].nunique(),
        "source_row_count": len(source),
        "source_children_with_multiple_lineages": int(
            resolution["source_lineage_count"].gt(1).sum()
        ),
        "resolved_parent_pair_children": int(
            resolution[["selected_parent1", "selected_parent2"]].ne("").all(axis=1).sum()
        ),
        "founder_children": int(
            resolution[["selected_parent1", "selected_parent2"]].eq("").all(axis=1).sum()
        ),
        "conflicting_lineage_founder_fallbacks": int(
            resolution["resolution_status"]
            .eq("conservative_founder_due_conflicting_lineages")
            .sum()
        ),
        "unreviewed_selfing_founder_fallbacks": int(
            resolution["resolution_status"].eq("founder_due_unreviewed_selfing").sum()
        ),
        "manual_resolution_count": int(resolution["human_reviewed"].sum()),
        "stable_parent_registry_rows": len(registry_frame),
        "canonical_pedigree_rows": len(pedigree),
        "blocking_resolution_rows": len(blockers),
    }
    return pedigree, registry_frame, resolution, selfing, {**metrics, "blockers": blockers}


def write_checksums(paths: list[Path], output: Path, root: Path) -> None:
    display_paths = []
    for path in paths:
        try:
            display_paths.append(path.relative_to(root))
        except ValueError:
            display_paths.append(path)
    lines = [
        f"{sha256_file(path)}  {display.as_posix()}"
        for path, display in zip(paths, display_paths)
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an isolated, provenance-aware canonical wheat pedigree and K_A."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("metadata_outputs/all_trials_genotype_manifest_resolved.tsv"),
    )
    parser.add_argument("--manual-lineage-decisions", type=Path)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("genotype_panels/pedigree_canonical_v2")
    )
    parser.add_argument("--prefix", default="K_A_CANONICAL_V2")
    parser.add_argument("--allow-conservative-founder-fallback", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    source_path = resolve(root, args.source_manifest)
    manual_path = (
        resolve(root, args.manual_lineage_decisions)
        if args.manual_lineage_decisions
        else None
    )
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise SystemExit(f"Source pedigree manifest is missing or empty: {source_path}")

    source = source_lineages(source_path)
    manual = read_manual_decisions(manual_path)
    pedigree, registry, resolution, selfing, metrics = build_resolution(
        source,
        manual,
        allow_conservative_founder_fallback=args.allow_conservative_founder_fallback,
    )
    blockers = metrics.pop("blockers")

    pedigree_path = out_dir / "canonical_pedigree_parent_table.tsv"
    registry_path = out_dir / "canonical_parent_registry.tsv"
    resolution_path = out_dir / "child_lineage_resolution.tsv"
    selfing_path = out_dir / "selfing_review.tsv"
    blocker_path = out_dir / "pedigree_resolution_blockers.tsv"
    pedigree.to_csv(pedigree_path, sep="\t", index=False)
    registry.to_csv(registry_path, sep="\t", index=False)
    resolution.to_csv(resolution_path, sep="\t", index=False)
    selfing.to_csv(selfing_path, sep="\t", index=False)
    blockers.to_csv(blocker_path, sep="\t", index=False)

    manual_template = resolution[
        resolution["resolution_status"].isin(
            [
                "conservative_founder_due_conflicting_lineages",
                "founder_due_unreviewed_selfing",
            ]
        )
    ][["sample_id", "source_lineages", "resolution_status"]].copy()
    manual_template["decision"] = ""
    manual_template["selected_lineage"] = ""
    manual_template["reviewed"] = False
    manual_template["reviewer"] = ""
    manual_template["evidence"] = ""
    manual_template.to_csv(
        out_dir / "manual_lineage_decisions_template.tsv", sep="\t", index=False
    )

    status = "PASS" if blockers.empty else "BLOCKED"
    generated_paths = [
        pedigree_path,
        registry_path,
        resolution_path,
        selfing_path,
        blocker_path,
        out_dir / "manual_lineage_decisions_template.tsv",
    ]
    kernel_paths: list[Path] = []
    kernel_metrics: dict[str, object] = {}
    if status == "PASS":
        relationship_input = pedigree[["sample_id", "parent1", "parent2"]]
        relationship, order, qc = additive_relationship(relationship_input, None)
        mean_diagonal = float(np.diag(relationship).mean())
        relationship = (relationship / mean_diagonal).astype(np.float32)
        assert_relationship_valid(relationship, order)
        kernel_path = out_dir / f"{args.prefix}.npy"
        order_path = out_dir / f"{args.prefix}_sample_order.tsv"
        row_path = out_dir / f"{args.prefix}_row_order.tsv"
        column_path = out_dir / f"{args.prefix}_column_order.tsv"
        qc_path = out_dir / f"{args.prefix}_qc.tsv"
        np.save(kernel_path, relationship)
        pd.DataFrame(
            {"sample_id": order, "compact_kernel_index": np.arange(len(order))}
        ).to_csv(order_path, sep="\t", index=False)
        pd.DataFrame({"row_index": np.arange(len(order)), "sample_id": order}).to_csv(
            row_path, sep="\t", index=False
        )
        pd.DataFrame(
            {"column_index": np.arange(len(order)), "sample_id": order}
        ).to_csv(column_path, sep="\t", index=False)
        qc.loc[len(qc)] = ["scaled_mean_diagonal", float(np.diag(relationship).mean())]
        qc.to_csv(qc_path, sep="\t", index=False)
        kernel_paths = [kernel_path, order_path, row_path, column_path, qc_path]
        kernel_metrics = {
            "K_A_shape": f"{relationship.shape[0]}x{relationship.shape[1]}",
            "K_A_mean_diagonal": float(np.diag(relationship).mean()),
            "K_A_min_diagonal": float(np.diag(relationship).min()),
            "K_A_max_diagonal": float(np.diag(relationship).max()),
        }

    decision = {
        "status": status,
        "canonical_K_A_construction_allowed": status == "PASS",
        "protocol_version": "canonical_trial_pedigree_v2_purdy",
        "selection_data": "identifiers_and_pedigree_strings_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "allow_conservative_founder_fallback": args.allow_conservative_founder_fallback,
        "source_manifest": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "manual_lineage_decisions": (
            {"path": str(manual_path), "sha256": sha256_file(manual_path)}
            if manual_path is not None and manual_path.is_file()
            else None
        ),
        "metrics": {**metrics, **kernel_metrics},
        "blocking_reasons": (
            [] if status == "PASS" else ["lineage_or_selfing_decisions_require_review"]
        ),
        "interpretation_contract": {
            "stable_local_founder_id_implies_verified_global_germplasm_identity": False,
            "conflicting_source_lineage_was_selected_implicitly": False,
            "conservative_founder_fallback_removes_observations": False,
            "compound_pedigree_notation_was_split_as_plain_text": False,
            "existing_K_A_was_modified": False,
        },
    }
    decision_path = out_dir / "canonical_pedigree_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    generated_paths.append(decision_path)
    checksum_path = out_dir / "canonical_pedigree_artifacts.sha256"
    write_checksums(generated_paths + kernel_paths, checksum_path, root)

    print(json.dumps(decision, indent=2))
    print("\n=== RESOLUTION STATUS ===")
    print(
        resolution.groupby(["resolution_status", "construction_eligible"], dropna=False)
        .size()
        .rename("children")
        .reset_index()
        .to_string(index=False)
    )
    if status != "PASS":
        raise SystemExit(
            "Canonical pedigree remains blocked; review the generated decision template"
        )


if __name__ == "__main__":
    main()
