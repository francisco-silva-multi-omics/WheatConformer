from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_pedigree_kernel import additive_relationship
from server_genotype_recovery.audit_single_step_readiness import (
    a22_hmp_compatibility,
    json_safe,
    kernel_integrity,
    pedigree_structure,
    readiness_decision,
    source_lineage_conflicts,
)


def write_order(path: Path, values: list[str]) -> None:
    pd.DataFrame({"sample_id": values}).to_csv(path, sep="\t", index=False)


def canonical_pedigree() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["GID1", "GID2", "GID3", "GID4"],
            "parent1": ["", "", "GID1", "GID1"],
            "parent2": ["", "", "GID2", "GID2"],
        }
    )


def test_a22_hmp_compatibility_recovers_matched_scale(tmp_path: Path) -> None:
    relationship, order, _ = additive_relationship(canonical_pedigree(), None)
    ka_path = tmp_path / "K_A.npy"
    kg_path = tmp_path / "K_G.npy"
    np.save(ka_path, relationship)
    np.save(kg_path, relationship)

    assert kernel_integrity(ka_path, order, sample_size=4)["status"] == "PASS"
    overlap, metrics = a22_hmp_compatibility(
        ka_path,
        order,
        kg_path,
        order,
        sample_size=4,
        blend_fraction=0.05,
    )

    assert overlap["sample_id"].tolist() == sorted(order)
    assert metrics["overlap_genotypes"] == 4
    assert np.isclose(metrics["alignment_alpha"], 0.0)
    assert np.isclose(metrics["alignment_beta"], 1.0)
    assert metrics["sampled_blended_G_min_eigenvalue"] >= -1e-6


def test_a22_hmp_compatibility_handles_no_overlap(tmp_path: Path) -> None:
    ka_path = tmp_path / "K_A.npy"
    kg_path = tmp_path / "K_G.npy"
    np.save(ka_path, np.eye(2, dtype=np.float32))
    np.save(kg_path, np.eye(2, dtype=np.float32))

    overlap, metrics = a22_hmp_compatibility(
        ka_path,
        ["GID1", "GID2"],
        kg_path,
        ["GID3", "GID4"],
        sample_size=2,
        blend_fraction=0.05,
    )

    assert overlap.empty
    assert metrics["overlap_genotypes"] == 0
    assert np.isnan(metrics["alignment_beta"])
    assert json_safe(metrics)["alignment_beta"] is None


def test_noncanonical_parent_tokens_block_current_k_a_even_when_alias_is_curated() -> None:
    pedigree = pd.DataFrame(
        {
            "sample_id": ["GID1", "GID2"],
            "parent1": ["PARENT A", ""],
            "parent2": ["PARENT B", ""],
        }
    )
    _, parent_issues, _, structure = pedigree_structure(
        pedigree,
        child_pattern=re.compile(r"^GID[0-9]+$"),
        parent_pattern=re.compile(r"^GID[0-9]+$"),
        curated_tokens={"PARENT A"},
    )
    structure.pop("cycle_nodes")
    status, blocking, _ = readiness_decision(
        {
            "source_manifest_present": True,
            "source_children_with_multiple_lineages": 0,
        },
        structure,
        {"status": "PASS"},
        {"status": "PASS"},
        {
            "overlap_genotypes": 100,
            "alignment_beta": 1.0,
            "sampled_blended_G_min_eigenvalue": 0.0,
            "sampled_informative_pair_count": 30,
            "sampled_offdiagonal_correlation": 0.5,
        },
        minimum_overlap=100,
    )

    assert status == "BLOCKED"
    assert structure["uncurated_noncanonical_parent_tokens"] == 1
    assert structure["curated_parent_aliases_requiring_rebuild"] == 1
    assert "parent_tokens_are_not_curated_stable_ids" in blocking
    assert "curated_parent_aliases_require_canonical_K_A_rebuild" in blocking
    assert set(parent_issues["status"]) == {
        "uncurated_noncanonical_parent_token",
        "curated_alias_requires_canonical_pedigree_rebuild",
    }


def test_source_lineage_conflicts_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "source.tsv"
    pd.DataFrame(
        {
            "sample_id": ["GID1", "GID1", "GID2"],
            "cross_name": ["A/B", "A/C", "D/E"],
        }
    ).to_csv(path, sep="\t", index=False)

    conflicts, metrics = source_lineage_conflicts(path)

    assert metrics["source_children_with_multiple_lineages"] == 1
    assert conflicts["sample_id"].unique().tolist() == ["GID1"]
    assert set(conflicts["lineage_signature"]) == {"A/B", "A/C"}


def test_single_step_readiness_cli_writes_outcome_free_decision(tmp_path: Path) -> None:
    pedigree_dir = tmp_path / "genotype_panels/pedigree"
    hmp_dir = tmp_path / "genotype_panels/hmp"
    regulatory_dir = tmp_path / "model_kernels/regulatory_eligibility_v1"
    pedigree_dir.mkdir(parents=True)
    hmp_dir.mkdir(parents=True)
    regulatory_dir.mkdir(parents=True)
    pedigree = canonical_pedigree()
    pedigree.to_csv(pedigree_dir / "pedigree_parent_table.tsv", sep="\t", index=False)
    pedigree.assign(cross_name=["", "", "GID1/GID2", "GID1/GID2"]).to_csv(
        pedigree_dir / "trial_derived_pedigree_manifest.tsv", sep="\t", index=False
    )
    relationship, order, _ = additive_relationship(pedigree, None)
    np.save(pedigree_dir / "K_A.npy", relationship)
    np.save(hmp_dir / "K_HMP.QCfiltered.meanDiag1.npy", relationship)
    write_order(pedigree_dir / "K_A_sample_order.tsv", order)
    write_order(hmp_dir / "hmp_K_sample_order.QCfiltered.tsv", order)
    (regulatory_dir / "regulatory_eligibility_certification.json").write_text(
        json.dumps({"status": "PASS"}) + "\n", encoding="utf-8"
    )

    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "server_genotype_recovery.audit_single_step_readiness",
            "--root",
            str(tmp_path),
            "--out-dir",
            "model_kernels/single_step_readiness_v1",
            "--minimum-overlap",
            "2",
            "--sample-size",
            "4",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    decision_path = (
        tmp_path
        / "model_kernels/single_step_readiness_v1"
        / "single_step_readiness_decision.json"
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    assert decision["status"] == "WARN"
    assert decision["single_step_H_construction_allowed"]
    assert decision["phenotype_values_read"] is False
    assert decision["outer_test_metrics_read"] is False
    assert decision["final_holdout_outcomes_read"] is False
    assert decision["blocking_reasons"] == []
