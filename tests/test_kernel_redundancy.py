from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analyze_kernel_redundancy import analyze


def test_kernel_redundancy_reports_alignment_and_correlation(tmp_path: Path) -> None:
    linear = np.array(
        [[1.0, 0.4, 0.1, 0.0], [0.4, 1.0, 0.2, 0.1], [0.1, 0.2, 1.0, 0.5], [0.0, 0.1, 0.5, 1.0]],
        dtype=np.float32,
    )
    rbf = np.exp(-(1.0 - linear)).astype(np.float32)
    linear_path = tmp_path / "linear.npy"
    rbf_path = tmp_path / "rbf.npy"
    order_path = tmp_path / "order.tsv"
    np.save(linear_path, linear)
    np.save(rbf_path, rbf)
    pd.DataFrame({"sample_id": ["a", "b", "c", "d"]}).to_csv(order_path, sep="\t", index=False)
    result, spectrum = analyze(linear_path, rbf_path, order_path, "sample_id", 4, 4, 2, 2026)
    assert result["shape_consistent"] is True
    assert result["centered_kernel_alignment"] > 0.9
    assert result["off_diagonal_pearson_correlation"] > 0.9
    assert result["linear_eigen_summary"]["sampled_min_eigenvalue"] >= -1e-8
    assert set(spectrum["kernel"]) == {"linear", "rbf"}
