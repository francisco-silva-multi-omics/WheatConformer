from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from build_epistatic_genomic_kernel import build_epi2
from server_training_pipeline.run_validation_ablation_suite import build_features

ROOT = Path(__file__).resolve().parents[1]


def test_epi2_is_hadamard_square_scaled_to_mean_diagonal_one(tmp_path: Path) -> None:
    linear = np.array([[2.0, -0.5, 0.2], [-0.5, 1.0, 0.1], [0.2, 0.1, 1.5]], dtype=np.float32)
    linear_path = tmp_path / "linear.npy"
    order_path = tmp_path / "order.tsv"
    out_path = tmp_path / "epi2.npy"
    np.save(linear_path, linear)
    pd.DataFrame({"sample_id": ["a", "b", "c"]}).to_csv(order_path, sep="\t", index=False)
    qc = build_epi2(linear_path, order_path, "sample_id", out_path, 2, 3, 2026)
    epi2 = np.load(out_path)
    expected = linear * linear / np.mean(np.diag(linear * linear))
    assert np.allclose(epi2, expected)
    assert np.isclose(np.diag(epi2).mean(), 1.0)
    assert qc["sampled_min_eigenvalue"] >= -1e-6


def test_validation_features_accept_optional_epi2_terms() -> None:
    factors = np.eye(3, dtype=np.float32)
    gi = np.array([0, 1, 2], dtype=np.int32)
    ei = np.array([2, 1, 0], dtype=np.int32)
    features = build_features("G+EPI2+E+GE+EPI2E", factors, factors, None, factors, gi, ei, 2, 2)
    assert features.shape[0] == 3
    assert features.shape[1] > 1


def test_stage1_compaction_writes_optional_epi2_kernel(tmp_path: Path) -> None:
    phenotypes = pd.DataFrame(
        {
            "panel_sample_id": ["g1", "g2"],
            "env_kernel_id": ["e1", "e2"],
            "y_tilde_g_e": [1.0, 2.0],
            "SE_g_e": [0.1, 0.1],
            "var_g_e": [0.01, 0.01],
            "weight_g_e": [1.0, 1.0],
            "trait_name_canonical": ["Trait", "Trait"],
            "stage1_model_status": ["linear_model_adjusted", "linear_model_adjusted"],
        }
    )
    phenotype_path = tmp_path / "phenotypes.tsv"
    phenotypes.to_csv(phenotype_path, sep="\t", index=False)
    genotype_order = tmp_path / "g_order.tsv"
    environment_order = tmp_path / "e_order.tsv"
    pd.DataFrame({"sample_id": ["g1", "g2"]}).to_csv(genotype_order, sep="\t", index=False)
    pd.DataFrame({"env_id": ["e1", "e2"]}).to_csv(environment_order, sep="\t", index=False)
    linear = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float32)
    epi2 = linear * linear
    env = np.eye(2, dtype=np.float32)
    linear_path, epi2_path, env_path = tmp_path / "linear.npy", tmp_path / "epi2.npy", tmp_path / "env.npy"
    np.save(linear_path, linear)
    np.save(epi2_path, epi2)
    np.save(env_path, env)
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "build_stage1_model_kernels.py"),
            "--stage1-phenotypes",
            str(phenotype_path),
            "--geno-kernel",
            str(linear_path),
            "--geno-epi2-kernel",
            str(epi2_path),
            "--geno-order",
            str(genotype_order),
            "--env-kernel",
            str(env_path),
            "--env-order",
            str(environment_order),
            "--out-dir",
            str(out),
            "--prefix",
            "toy",
            "--write-tsv",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert np.allclose(np.load(out / "toy_K_G_EPI2_unique.npy"), epi2)
