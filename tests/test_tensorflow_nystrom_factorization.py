from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from server_training_pipeline.kernel_factorization import (
    effective_factorization_mode,
    factorization_training_support,
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


@pytest.mark.parametrize(
    "split_mode",
    [
        "gho_environment",
        "gho_cycle",
        "gho_trial",
        "gho_country",
        "gho_family",
        "cv1_genotype",
        "cv1_environment",
        "cv0_genotype_environment",
    ],
)
def test_grouped_holdouts_use_strict_nystrom(split_mode: str) -> None:
    assert (
        effective_factorization_mode("train_nystrom", split_mode, warn=True)
        == "train_nystrom"
    )


def test_full_transductive_default_and_noninductive_mode_resolution() -> None:
    assert effective_factorization_mode("full_transductive", "cv1_genotype") == "full_transductive"
    assert effective_factorization_mode("train_nystrom", "cv1_genotype") == "train_nystrom"
    with pytest.warns(UserWarning, match="grouped entity holdout"):
        assert (
            effective_factorization_mode(
                "train_nystrom", "cv2_random_observation", warn=True
            )
            == "full_transductive"
        )


def test_cv1_genotype_train_ids_define_strict_kernel_dimension(tmp_path: Path) -> None:
    kernel = np.eye(5) + 0.1
    train_ids = np.array([0, 2, 4])
    factors, metadata = kernel_factors(write_kernel(tmp_path / "kernel.npy", kernel), 5, train_ids)
    assert factors.shape[0] == 5
    assert metadata["train_kernel_dimension"] == 3
    assert metadata["rank_retained"] <= 3


def test_centered_fold_expert_with_one_training_id_is_not_estimable() -> None:
    supported, reason = factorization_training_support(
        np.array([7, 7]), "train_nystrom", center=True
    )

    assert not supported
    assert reason == "centered_train_nystrom_requires_at_least_two_training_ids"


def test_fold_expert_with_two_training_ids_is_estimable() -> None:
    supported, reason = factorization_training_support(
        np.array([7, 11, 7]), "train_nystrom", center=True
    )

    assert supported
    assert reason == ""


def test_sparse_expert_requires_declared_training_support() -> None:
    supported, reason = factorization_training_support(
        np.array([1, 2, 3, 4]),
        "train_nystrom",
        center=True,
        minimum_ids=5,
    )
    assert not supported
    assert reason == "expert_requires_at_least_5_training_ids"

    supported, reason = factorization_training_support(
        np.array([1, 2, 3, 4, 5]),
        "train_nystrom",
        center=True,
        minimum_ids=5,
    )
    assert supported
    assert reason == ""


def test_tensorflow_trainer_records_factorization_provenance() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "server_training_pipeline"
        / "train_multikernel_gxe_tf.py"
    ).read_text(encoding="utf-8")
    for field in [
        "requested_factorization_mode",
        "effective_factorization_mode",
        "train_genotype_kernel_dimension",
        "train_environment_kernel_dimension",
        "rank_g_requested",
        "rank_g_retained",
        "rank_g_rbf_requested",
        "rank_g_rbf_retained",
        "rank_e_requested",
        "rank_e_retained",
    ]:
        assert field in source
