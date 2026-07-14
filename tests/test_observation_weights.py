from __future__ import annotations

import numpy as np
import pandas as pd

from server_training_pipeline.observation_weights import effective_sample_size, stabilize_precision_weights


def test_trait_weights_are_finite_mean_one_and_preserve_raw_values() -> None:
    frame = pd.DataFrame(
        {
            "trait_name_canonical": ["A"] * 5 + ["B"] * 4,
            "var_g_e": [1e-30, 1.0, 2.0, np.nan, 4.0, 2.0, 3.0, 4.0, 5.0],
            "weight_g_e": [1e30, 1.0, 0.5, np.nan, 0.25, 0.5, 1 / 3, 0.25, 0.2],
        }
    )

    output, qc = stabilize_precision_weights(frame)

    assert np.isfinite(output["weight_g_e"]).all()
    np.testing.assert_allclose(
        output.groupby("trait_name_canonical")["weight_g_e"].mean().to_numpy(),
        np.ones(2),
        atol=1e-12,
    )
    assert np.isclose(output.loc[0, "raw_weight_g_e"], 1e30)
    assert bool(output.loc[3, "weight_variance_imputed"])
    assert set(qc["trait_name_canonical"]) == {"A", "B"}
    assert qc["effective_sample_fraction"].between(0, 1).all()


def test_normalized_precision_weights_are_invariant_to_variance_units() -> None:
    frame = pd.DataFrame(
        {
            "trait_name_canonical": ["A"] * 5,
            "var_g_e": [1.0, 2.0, 3.0, 4.0, 5.0],
            "weight_g_e": [1.0, 0.5, 1 / 3, 0.25, 0.2],
        }
    )
    scaled = frame.copy()
    scaled["var_g_e"] *= 100.0
    scaled["weight_g_e"] /= 100.0

    output_a, _ = stabilize_precision_weights(frame)
    output_b, _ = stabilize_precision_weights(scaled)

    np.testing.assert_allclose(output_a["weight_g_e"], output_b["weight_g_e"], atol=1e-12)


def test_effective_sample_size_matches_equal_weight_count() -> None:
    assert effective_sample_size(np.ones(12)) == 12.0
