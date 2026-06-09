from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from server_training_pipeline.kernel_factorization import (
    effective_factorization_mode,
    kernel_factors,
    retained_eigenvalues,
)


def write_kernel(path: Path, matrix: np.ndarray) -> Path:
    np.save(path, matrix.astype(np.float64))
    return path


def test_tensorflow_nystrom_excludes_heldout_ids_and_projects_all_rows(tmp_path: Path) -> None:
    kernel = np.array(
        [[2.0, 0.5, 0.2, 0.1], [0.5, 1.5, 0.3, 0.2], [0.2, 0.3, 1.2, 0.4], [0.1, 0.2, 0.4, 1.1]]
    )
    changed = kernel.copy()
    changed[2:, 2:] = [[100.0, 20.0], [20.0, 80.0]]
    train_ids = np.array([0, 1])
    factors_a, metadata_a = kernel_factors(write_kernel(tmp_path / "a.npy", kernel), 2, train_ids, jitter=1e-6)
    factors_b, metadata_b = kernel_factors(write_kernel(tmp_path / "b.npy", changed), 2, train_ids, jitter=1e-6)

    assert factors_a.shape == (4, 2)
    assert metadata_a == metadata_b
    assert metadata_a["factorization_mode"] == "train_nystrom"
    np.testing.assert_allclose(factors_a[train_ids], factors_b[train_ids], atol=1e-6)
    assert retained_eigenvalues(factors_a, train_ids).shape == (2,)


def test_full_transductive_default_and_strict_mode_resolution() -> None:
    assert effective_factorization_mode("full_transductive", "cv1_genotype") == "full_transductive"
    assert effective_factorization_mode("train_nystrom", "cv1_genotype") == "train_nystrom"
    with pytest.warns(UserWarning, match="restricted to CV1/CV0"):
        assert effective_factorization_mode("train_nystrom", "gho_environment", warn=True) == "full_transductive"


def test_cv1_genotype_train_ids_define_strict_kernel_dimension(tmp_path: Path) -> None:
    kernel = np.eye(5) + 0.1
    train_ids = np.array([0, 2, 4])
    factors, metadata = kernel_factors(write_kernel(tmp_path / "kernel.npy", kernel), 5, train_ids)
    assert factors.shape[0] == 5
    assert metadata["train_kernel_dimension"] == 3
    assert metadata["rank_retained"] <= 3
