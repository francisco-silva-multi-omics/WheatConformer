from __future__ import annotations

from pathlib import Path

import numpy as np

from server_training_pipeline.kernel_factorization import kernel_factors, top_factors


def write_kernel(path: Path, matrix: np.ndarray) -> Path:
    np.save(path, matrix.astype(np.float64))
    return path


def test_nystrom_features_use_train_only_eigendecomposition(tmp_path: Path) -> None:
    kernel = np.array(
        [
            [2.0, 0.5, 0.2, 0.1],
            [0.5, 1.5, 0.3, 0.2],
            [0.2, 0.3, 1.2, 0.4],
            [0.1, 0.2, 0.4, 1.1],
        ]
    )
    changed_heldout = kernel.copy()
    changed_heldout[2:, 2:] = np.array([[100.0, 25.0], [25.0, 80.0]])
    train_ids = np.array([0, 1])

    factors_a, metadata_a = kernel_factors(write_kernel(tmp_path / "a.npy", kernel), 2, train_ids)
    factors_b, metadata_b = kernel_factors(write_kernel(tmp_path / "b.npy", changed_heldout), 2, train_ids)

    assert factors_a.shape == (4, 2)
    assert metadata_a == metadata_b
    assert metadata_a["factorization_mode"] == "train_nystrom"
    assert metadata_a["rank_requested"] == 2
    assert metadata_a["rank_retained"] == 2
    assert metadata_a["train_kernel_dimension"] == 2
    np.testing.assert_allclose(factors_a[train_ids], factors_b[train_ids], atol=1e-6)


def test_nystrom_projection_matches_training_kernel(tmp_path: Path) -> None:
    features = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 1.0], [1.0, 1.0]])
    kernel = features @ features.T + np.eye(4) * 0.1
    train_ids = np.array([0, 2, 3])
    factors, metadata = kernel_factors(write_kernel(tmp_path / "kernel.npy", kernel), 3, train_ids)

    assert factors.shape == (4, 3)
    assert metadata["train_kernel_dimension"] == 3
    np.testing.assert_allclose(
        factors[train_ids] @ factors[train_ids].T,
        kernel[np.ix_(train_ids, train_ids)],
        atol=1e-5,
    )


def test_full_transductive_remains_backward_compatible(tmp_path: Path) -> None:
    kernel = np.array([[2.0, 0.4, 0.1], [0.4, 1.5, 0.2], [0.1, 0.2, 1.0]])
    path = write_kernel(tmp_path / "kernel.npy", kernel)

    factors, metadata = kernel_factors(path, 2)

    np.testing.assert_allclose(factors, top_factors(path, 2), atol=1e-7)
    assert metadata["factorization_mode"] == "full_transductive"
    assert metadata["train_kernel_dimension"] == 3


def test_centered_nystrom_training_factors_reconstruct_centered_kernel(tmp_path: Path) -> None:
    features = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 1.0], [1.0, 1.0]])
    kernel = features @ features.T + np.eye(4) * 0.1
    train_ids = np.array([0, 2, 3])
    factors, metadata = kernel_factors(
        write_kernel(tmp_path / "centered.npy", kernel), 3, train_ids, center=True
    )
    train_kernel = kernel[np.ix_(train_ids, train_ids)]
    centered = (
        train_kernel
        - train_kernel.mean(axis=0, keepdims=True)
        - train_kernel.mean(axis=1, keepdims=True)
        + train_kernel.mean()
    )

    np.testing.assert_allclose(
        factors[train_ids] @ factors[train_ids].T,
        centered,
        atol=1e-5,
    )
    assert metadata["kernel_centered"] == "true"
