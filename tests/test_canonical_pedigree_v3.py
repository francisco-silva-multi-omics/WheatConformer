from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from server_genotype_recovery.build_canonical_pedigree import (
    REGISTRY_COLUMNS,
    build_resolution,
)
from server_genotype_recovery.build_canonical_pedigree_v3 import (
    overlay_recovered_edges,
)


def source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["GID1", "GID2", "GID3"],
            "source_lineage": ["A/B", "C/D", "CHECK"],
        }
    )


def accepted_edges() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "child_id": "GID3",
                "parent_role": "parent1",
                "parent_id": "GID6",
                "verification_class": "accepted_new_edge_exact_unique",
                "accepted": True,
                "is_new_edge": True,
                "source_datasets": "hdl:11529/wheat",
                "source_files": "pedigree.tsv",
            },
            {
                "child_id": "GID3",
                "parent_role": "parent2",
                "parent_id": "PEDF_AAAAAAAAAAAAAAAA",
                "verification_class": "accepted_new_edge_corroborated",
                "accepted": True,
                "is_new_edge": True,
                "source_datasets": "hdl:11529/wheat",
                "source_files": "pedigree.tsv",
            },
        ]
    )


def recovered_registry() -> pd.DataFrame:
    row = {column: "" for column in REGISTRY_COLUMNS}
    row.update(
        {
            "stable_parent_id": "PEDF_AAAAAAAAAAAAAAAA",
            "node_type": "named_founder_designation",
            "source_expression": "RECOVERED_PARENT",
            "normalized_expression": "RECOVERED_PARENT",
            "identity_scope": "stable_local_designation_not_verified_global_germplasm",
            "construction_eligible": "True",
            "accepted_by_policy": "True",
            "provenance": "verified_external_record",
        }
    )
    return pd.DataFrame([row], columns=REGISTRY_COLUMNS)


def test_v3_overlay_adds_only_verified_missing_parents() -> None:
    pedigree, registry, resolution, _, _ = build_resolution(
        source(), {}, allow_conservative_founder_fallback=True
    )
    updated, merged_registry, updated_resolution, audit = overlay_recovered_edges(
        pedigree,
        resolution,
        registry,
        accepted_edges(),
        recovered_registry(),
    )
    child = updated.set_index("sample_id").loc["GID3"]
    assert child["parent1"] == "GID6"
    assert child["parent2"] == "PEDF_AAAAAAAAAAAAAAAA"
    assert {"GID6", "PEDF_AAAAAAAAAAAAAAAA"}.issubset(set(updated["sample_id"]))
    assert "PEDF_AAAAAAAAAAAAAAAA" in set(merged_registry["stable_parent_id"])
    assert audit["overlay_status"].eq("applied_certified_recovered_edge").all()
    status = updated_resolution.set_index("sample_id").loc[
        "GID3", "resolution_status"
    ]
    assert status == "resolved_with_certified_recovered_edge_overlay"


def test_v3_overlay_rejects_overwriting_nonempty_v2_parent() -> None:
    pedigree, registry, resolution, _, _ = build_resolution(
        source(), {}, allow_conservative_founder_fallback=True
    )
    conflicting = accepted_edges().iloc[[0]].copy()
    conflicting["child_id"] = "GID1"
    with pytest.raises(ValueError, match="conflicts with canonical v2"):
        overlay_recovered_edges(
            pedigree, resolution, registry, conflicting, recovered_registry()
        )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v3_cli_requires_and_records_certified_verification_bundle(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "metadata_outputs/all_trials_genotype_manifest_resolved.tsv"
    source_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "panel_sample_id_expected": source()["sample_id"],
            "cross_name": source()["source_lineage"],
        }
    ).to_csv(source_path, sep="\t", index=False)

    verification = tmp_path / "genotype_panels/recovered_identity_verification_v2"
    verification.mkdir(parents=True)
    (verification / "verification_contract.json").write_text(
        json.dumps({"protocol_version": "recovered_identity_verification_v2"}) + "\n",
        encoding="utf-8",
    )
    (verification / "verification_provenance.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "protocol_version": "recovered_identity_verification_v2",
                "phenotype_values_read": False,
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
                "model_performance_read": False,
                "kernels_modified": False,
                "single_step_H_constructed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame({"check": ["run_status"], "status": ["PASS"]}).to_csv(
        verification / "verification_qc.tsv", sep="\t", index=False
    )
    accepted_edges().to_csv(
        verification / "accepted_new_pedigree_edges.tsv", sep="\t", index=False
    )
    recovered_registry().to_csv(
        verification / "verification_parent_registry.tsv", sep="\t", index=False
    )
    names = [
        "verification_contract.json",
        "verification_provenance.json",
        "verification_qc.tsv",
        "accepted_new_pedigree_edges.tsv",
        "verification_parent_registry.tsv",
    ]
    pd.DataFrame(
        {
            "sha256": [digest(verification / name) for name in names],
            "bytes": [(verification / name).stat().st_size for name in names],
            "path": names,
        }
    ).to_csv(verification / "verification_sha256.tsv", sep="\t", index=False)

    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "server_genotype_recovery.build_canonical_pedigree_v3",
            "--root",
            str(tmp_path),
            "--allow-conservative-founder-fallback",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    out = tmp_path / "genotype_panels/pedigree_canonical_v3"
    decision = json.loads(
        (out / "canonical_pedigree_decision.json").read_text(encoding="utf-8")
    )
    assert decision["status"] == "PASS"
    assert decision["metrics"]["certified_recovered_children"] == 1
    assert decision["metrics"]["applied_recovered_edge_rows"] == 2
    kernel = np.load(out / "K_A_CANONICAL_V3.npy")
    order = pd.read_csv(out / "K_A_CANONICAL_V3_sample_order.tsv", sep="\t")
    assert kernel.shape == (len(order), len(order))
    assert np.isfinite(kernel).all()
    assert np.isclose(np.diag(kernel).mean(), 1.0)
    assert not (tmp_path / "genotype_panels/pedigree_canonical_v2").exists()
