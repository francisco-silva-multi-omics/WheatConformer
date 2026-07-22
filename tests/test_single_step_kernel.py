from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from server_genotype_recovery.build_single_step_kernel import (
    construct_single_step_submatrix,
    tune_and_blend_genomic_relationship,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_covariance_update_matches_explicit_single_step_inverse() -> None:
    pedigree = np.array(
        [
            [1.0, 0.2, 0.4],
            [0.2, 1.0, 0.3],
            [0.4, 0.3, 1.1],
        ],
        dtype=float,
    )
    genomic_indices = np.array([0, 1], dtype=int)
    genomic = np.array([[1.05, 0.1], [0.1, 0.95]], dtype=float)
    observed, _ = construct_single_step_submatrix(
        pedigree,
        np.arange(3, dtype=int),
        genomic_indices,
        genomic,
        eigen_floor_fraction=1e-10,
    )
    update = np.zeros_like(pedigree)
    a22 = pedigree[np.ix_(genomic_indices, genomic_indices)]
    update[np.ix_(genomic_indices, genomic_indices)] = (
        np.linalg.inv(genomic) - np.linalg.inv(a22)
    )
    expected = np.linalg.inv(np.linalg.inv(pedigree) + update)
    assert np.allclose(observed, expected, atol=1e-10)
    assert np.allclose(
        observed[np.ix_(genomic_indices, genomic_indices)], genomic, atol=1e-10
    )


def test_tuning_matches_pedigree_diagonal_and_offdiagonal_moments() -> None:
    a22 = np.array(
        [[1.0, 0.25, 0.1], [0.25, 1.1, 0.2], [0.1, 0.2, 0.9]], dtype=float
    )
    genomic = np.array(
        [[1.4, 0.1, -0.1], [0.1, 1.2, 0.0], [-0.1, 0.0, 1.0]], dtype=float
    )
    working, qc = tune_and_blend_genomic_relationship(
        a22,
        genomic,
        pedigree_blend_fraction=0.05,
        eigen_floor_fraction=1e-10,
    )
    assert qc["alignment_beta"] > 0
    assert np.linalg.eigvalsh(working).min() > 0
    assert np.isclose(np.diag(working).mean(), np.diag(a22).mean(), atol=1e-8)


def test_single_step_cli_writes_certified_compact_artifacts(tmp_path: Path) -> None:
    pedigree = np.array(
        [
            [1.0, 0.2, 0.4, 0.1],
            [0.2, 1.0, 0.3, 0.2],
            [0.4, 0.3, 1.1, 0.25],
            [0.1, 0.2, 0.25, 1.0],
        ],
        dtype=np.float32,
    )
    genomic = np.array(
        [[1.1, 0.15, 0.2], [0.15, 0.95, 0.1], [0.2, 0.1, 1.05]],
        dtype=np.float32,
    )
    np.save(tmp_path / "K_A.npy", pedigree)
    np.save(tmp_path / "K_G.npy", genomic)
    pd.DataFrame({"sample_id": ["GID1", "GID2", "GID3", "GID4"]}).to_csv(
        tmp_path / "K_A_order.tsv", sep="\t", index=False
    )
    pd.DataFrame({"sample_id": ["GID1", "GID2", "GID3"]}).to_csv(
        tmp_path / "K_G_order.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {
            "sample_id": ["GID4", "GID2", "GID1"],
            "compact_kernel_index": [2, 1, 0],
        }
    ).to_csv(tmp_path / "target_order.tsv", sep="\t", index=False)
    readiness = {
        "status": "PASS",
        "single_step_H_construction_allowed": True,
        "blocking_reasons": [],
        "inputs": {
            "K_A": {"sha256": digest(tmp_path / "K_A.npy")},
            "K_A_order": {"sha256": digest(tmp_path / "K_A_order.tsv")},
        },
    }
    (tmp_path / "readiness.json").write_text(
        json.dumps(readiness) + "\n", encoding="utf-8"
    )
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "server_genotype_recovery.build_single_step_kernel",
            "--root",
            str(tmp_path),
            "--k-a",
            "K_A.npy",
            "--k-a-order",
            "K_A_order.tsv",
            "--k-g",
            "K_G.npy",
            "--k-g-order",
            "K_G_order.tsv",
            "--target-order",
            "target_order.tsv",
            "--readiness-decision",
            "readiness.json",
            "--panel",
            "TEST_PANEL",
            "--prefix",
            "K_H_TEST",
            "--out-dir",
            "single_step",
            "--minimum-overlap",
            "2",
            "--minimum-target-overlap",
            "2",
            "--sample-size",
            "3",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    output = tmp_path / "single_step"
    relationship = np.load(output / "K_H_TEST.npy")
    order = pd.read_csv(output / "K_H_TEST_sample_order.tsv", sep="\t")
    provenance = json.loads(
        (output / "K_H_TEST_construction.json").read_text(encoding="utf-8")
    )
    assert relationship.shape == (3, 3)
    assert order["sample_id"].tolist() == ["GID1", "GID2", "GID4"]
    assert order["compact_kernel_index"].tolist() == [0, 1, 2]
    assert provenance["status"] == "PASS"
    assert provenance["phenotype_values_read"] is False
    assert provenance["outer_test_metrics_read"] is False
    assert provenance["metrics"]["panel_K_A_overlap_genotypes"] == 3
    assert provenance["metrics"]["overlap_genotypes_in_target_order"] == 2
    assert (output / "K_H_TEST_artifacts.sha256").is_file()
