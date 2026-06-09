from __future__ import annotations

import numpy as np
import pytest

from build_requested_outputs import vanraden_kernel


def test_vanraden_kernel_is_symmetric_and_psd() -> None:
    matrix = np.array([[0, 0, 2], [0, 2, 1], [2, 2, 0]], dtype=np.float32)
    kernel, p, denominator = vanraden_kernel(matrix)
    assert denominator > 0
    assert np.all((p >= 0) & (p <= 1))
    assert np.allclose(kernel, kernel.T, atol=1e-6)
    assert np.linalg.eigvalsh(kernel).min() >= -1e-5


def test_vanraden_rejects_missing_and_negative_values() -> None:
    with pytest.raises(ValueError, match="missing|non-finite"):
        vanraden_kernel(np.array([[0, np.nan], [2, 1]], dtype=float))
    with pytest.raises(ValueError, match="outside dosage"):
        vanraden_kernel(np.array([[0, -9], [2, 1]], dtype=float))
