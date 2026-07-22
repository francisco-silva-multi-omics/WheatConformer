from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from server_genotype_recovery.build_canonical_pedigree import (
    RegistryBuilder,
    build_resolution,
    parse_purdy_pedigree,
)


def test_purdy_parser_preserves_cross_order_and_backcross_dosage() -> None:
    assert parse_purdy_pedigree("A/B").structural_key == "X(L:A,L:B)"
    assert (
        parse_purdy_pedigree("A/B//C/3/D").structural_key
        == "X(X(X(L:A,L:B),L:C),L:D)"
    )
    assert (
        parse_purdy_pedigree("D/3/A/B//C/4/E").structural_key
        == "X(X(L:D,X(X(L:A,L:B),L:C)),L:E)"
    )
    assert parse_purdy_pedigree("A*2/B").structural_key == "X(L:A,X(L:A,L:B))"
    assert parse_purdy_pedigree("A/2*B").structural_key == "X(X(L:A,L:B),L:B)"


def test_local_founder_ids_do_not_collapse_punctuation_or_invent_gids() -> None:
    registry = RegistryBuilder()
    hyphenated = registry.materialize(parse_purdy_pedigree("A-B"))
    compact = registry.materialize(parse_purdy_pedigree("AB"))
    numeric = registry.materialize(parse_purdy_pedigree("12345"))

    assert hyphenated != compact
    assert numeric.startswith("PEDF_")


def toy_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [
                "GID1",
                "GID2",
                "GID3",
                "GID3",
                "GID4",
                "GID5",
            ],
            "source_lineage": ["A/B", "A/B//C", "X/Y", "X/Z", "SELF/SELF", "CHECK"],
        }
    )


def test_resolution_uses_recorded_founders_for_conflicts_and_unreviewed_selfing() -> None:
    pedigree, registry, resolution, selfing, metrics = build_resolution(
        toy_source(), {}, allow_conservative_founder_fallback=True
    )
    children = pedigree[pedigree["node_role"].eq("trial_child")].set_index("sample_id")
    statuses = resolution.set_index("sample_id")["resolution_status"]

    assert children.loc["GID1", "parent1"].startswith("PEDF_")
    assert children.loc["GID2", "parent1"].startswith("PEDX_")
    assert children.loc["GID3", ["parent1", "parent2"]].eq("").all()
    assert children.loc["GID4", ["parent1", "parent2"]].eq("").all()
    assert children.loc["GID5", ["parent1", "parent2"]].eq("").all()
    assert statuses["GID3"] == "conservative_founder_due_conflicting_lineages"
    assert statuses["GID4"] == "founder_due_unreviewed_selfing"
    assert selfing["sample_id"].tolist() == ["GID4"]
    assert registry["stable_parent_id"].is_unique
    assert registry["construction_eligible"].all()
    assert metrics["blockers"].empty


def test_strict_resolution_keeps_conflicts_blocked() -> None:
    _, _, resolution, _, metrics = build_resolution(
        toy_source(), {}, allow_conservative_founder_fallback=False
    )
    blocked = set(metrics["blockers"]["sample_id"])
    assert blocked == {"GID3", "GID4"}
    assert not resolution.set_index("sample_id").loc["GID3", "construction_eligible"]


def test_canonical_pedigree_passes_single_step_readiness(tmp_path: Path) -> None:
    source_path = tmp_path / "metadata_outputs/all_trials_genotype_manifest_resolved.tsv"
    pedigree_dir = tmp_path / "genotype_panels/pedigree_canonical_v2"
    hmp_dir = tmp_path / "genotype_panels/hmp"
    regulatory_dir = tmp_path / "model_kernels/regulatory_eligibility_v1"
    readiness_dir = tmp_path / "model_kernels/single_step_readiness_v2"
    source_path.parent.mkdir(parents=True)
    hmp_dir.mkdir(parents=True)
    regulatory_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "panel_sample_id_expected": toy_source()["sample_id"],
            "cross_name": toy_source()["source_lineage"],
        }
    ).to_csv(source_path, sep="\t", index=False)
    (regulatory_dir / "regulatory_eligibility_certification.json").write_text(
        json.dumps({"status": "PASS"}) + "\n", encoding="utf-8"
    )

    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "server_genotype_recovery.build_canonical_pedigree",
            "--root",
            str(tmp_path),
            "--allow-conservative-founder-fallback",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    order = pd.read_csv(
        pedigree_dir / "K_A_CANONICAL_V2_sample_order.tsv", sep="\t"
    )
    trial_ids = [f"GID{i}" for i in range(1, 6)]
    np.save(hmp_dir / "K_HMP.QCfiltered.meanDiag1.npy", np.eye(5, dtype=np.float32))
    pd.DataFrame({"sample_id": trial_ids}).to_csv(
        hmp_dir / "hmp_K_sample_order.QCfiltered.tsv", sep="\t", index=False
    )
    assert set(trial_ids).issubset(set(order["sample_id"]))

    subprocess.run(
        [
            sys.executable,
            "-m",
            "server_genotype_recovery.audit_single_step_readiness",
            "--root",
            str(tmp_path),
            "--pedigree-parent-table",
            "genotype_panels/pedigree_canonical_v2/canonical_pedigree_parent_table.tsv",
            "--pedigree-source-manifest",
            "metadata_outputs/all_trials_genotype_manifest_resolved.tsv",
            "--stable-parent-registry",
            "genotype_panels/pedigree_canonical_v2/canonical_parent_registry.tsv",
            "--lineage-resolution",
            "genotype_panels/pedigree_canonical_v2/child_lineage_resolution.tsv",
            "--k-a",
            "genotype_panels/pedigree_canonical_v2/K_A_CANONICAL_V2.npy",
            "--k-a-order",
            "genotype_panels/pedigree_canonical_v2/K_A_CANONICAL_V2_sample_order.tsv",
            "--regulatory-certification",
            "model_kernels/regulatory_eligibility_v1/regulatory_eligibility_certification.json",
            "--out-dir",
            "model_kernels/single_step_readiness_v2",
            "--child-id-regex",
            r"^(GID[0-9]+|PED[FX]_[A-F0-9]{16})$",
            "--parent-id-regex",
            r"^(GID[0-9]+|PED[FX]_[A-F0-9]{16})$",
            "--minimum-overlap",
            "2",
            "--sample-size",
            "5",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    decision = json.loads(
        (readiness_dir / "single_step_readiness_decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["single_step_H_construction_allowed"] is True
    assert decision["blocking_reasons"] == []
    assert decision["source_summary"]["source_conflict_children_unresolved"] == 0
    assert decision["pedigree_summary"]["local_stable_nodes_missing_registry"] == 0
    assert decision["phenotype_values_read"] is False
