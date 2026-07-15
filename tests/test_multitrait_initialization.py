from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("tensorflow")
from server_training_pipeline.train_multitrait_multikernel_tf import (
    MultiTraitKernelExperts,
    regression_metrics,
    safe_weights,
)


def build_model(seed: int) -> MultiTraitKernelExperts:
    specs = [
        {
            "kernel": "K_E_A",
            "axis": "environment",
            "eligible_traits": "*",
            "interaction_enabled": True,
        },
        {
            "kernel": "K_E_B",
            "axis": "environment",
            "eligible_traits": "*",
            "interaction_enabled": True,
        },
    ]
    factors = [np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]
    return MultiTraitKernelExperts(
        specs,
        factors,
        trait_names=["TRAIT_A", "TRAIT_B"],
        latent_dim=3,
        include_genotype_main=False,
        include_environment_main=True,
        include_interaction=False,
        learn_kernel_gates=True,
        weight_decay=1e-4,
        initialization_seed=seed,
    )


def test_kernel_experts_receive_distinct_reproducible_initial_values() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = build_model(2026)
    second = build_model(2026)
    different_seed = build_model(2027)

    first_a = first.main_projection[0].numpy()
    first_b = first.main_projection[1].numpy()
    assert not np.array_equal(first_a, first_b)
    np.testing.assert_array_equal(first_a, second.main_projection[0].numpy())
    assert not np.array_equal(first_a, different_seed.main_projection[0].numpy())
    assert not any("initializer RandomNormal is unseeded" in str(item.message) for item in caught)


def test_training_metrics_reject_nonfinite_values_instead_of_dropping_rows() -> None:
    with pytest.raises(ValueError, match="weights"):
        safe_weights(np.array([1.0, np.inf]))
    with pytest.raises(ValueError, match="predictions"):
        regression_metrics(
            np.array([1.0, 2.0]),
            np.array([1.0, np.nan]),
            np.ones(2),
        )
