from __future__ import annotations

import numpy as np

from server_training_pipeline.build_phase6a_split_bound_projection_inputs import (
    canonicalize_svd,
    fit_preprocessing,
    transform_features,
)


def test_preprocessing_ignores_held_out_values() -> None:
    raw = np.asarray(
        [
            [1.0, 2.0, np.nan],
            [2.0, 4.0, 1.0],
            [3.0, 6.0, 0.0],
            [1000.0, -1000.0, 500.0],
        ]
    )
    fit_mask = np.asarray([True, True, True, False])
    first = fit_preprocessing(raw, fit_mask, 0.0, 1, 1e-12)
    altered = raw.copy()
    altered[-1] = [-9999.0, 9999.0, -7777.0]
    second = fit_preprocessing(altered, fit_mask, 0.0, 1, 1e-12)
    for key in ("nonmissing", "medians", "means", "scales", "retained"):
        np.testing.assert_allclose(first[key], second[key])


def test_transform_uses_training_imputation_and_inactive_zero() -> None:
    raw = np.asarray([[1.0, 2.0], [3.0, np.nan], [9.0, 9.0]])
    fit_mask = np.asarray([True, True, False])
    params = fit_preprocessing(raw, fit_mask, 0.0, 1, 1e-12)
    active = np.asarray([True, True, False])
    transformed = transform_features(raw, active, params)
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed[-1], 0.0)
    np.testing.assert_allclose(transformed[:2].mean(axis=0), 0.0, atol=1e-12)


def test_packbits_round_trip_exact_153_feature_mask() -> None:
    rng = np.random.default_rng(20260820)
    mask = rng.random((19, 153)) < 0.2
    packed = np.packbits(mask, axis=1, bitorder="little")
    restored = np.unpackbits(packed, axis=1, count=153, bitorder="little").astype(bool)
    assert packed.shape == (19, 20)
    np.testing.assert_array_equal(restored, mask)


def test_svd_sign_canonicalization() -> None:
    vectors = np.asarray([[0.1, -0.8, 0.2], [-0.9, 0.1, 0.2]])
    canonical = canonicalize_svd(vectors)
    for row in canonical:
        assert row[np.argmax(np.abs(row))] > 0
