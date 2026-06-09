from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from build_gaussian_genomic_kernel import resolve_gamma


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "build_gaussian_genomic_kernel.py"


def run_gaussian(tmp_path: Path, multiplier: float, order_rows: int = 3) -> tuple[np.ndarray, dict[str, object], subprocess.CompletedProcess[str]]:
    linear = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 1.0]], dtype=np.float32)
    linear_path = tmp_path / f"linear_{multiplier}.npy"
    order_path = tmp_path / f"order_{multiplier}.tsv"
    out_path = tmp_path / f"gaussian_{multiplier}.npy"
    qc_path = tmp_path / f"gaussian_{multiplier}.json"
    np.save(linear_path, linear)
    pd.DataFrame({"sample_id": [f"s{i}" for i in range(order_rows)]}).to_csv(order_path, sep="\t", index=False)
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--linear-kernel",
            str(linear_path),
            "--sample-order",
            str(order_path),
            "--out-kernel",
            str(out_path),
            "--out-qc",
            str(qc_path),
            "--gamma-multiplier",
            str(multiplier),
            "--median-sample-size",
            "3",
            "--psd-sample-size",
            "3",
            "--chunk-size",
            "2",
        ],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return np.empty((0, 0)), {}, proc
    return np.load(out_path), json.loads(qc_path.read_text(encoding="utf-8")), proc


def test_gaussian_kernel_is_symmetric_psd_and_gamma_controls_similarity(tmp_path: Path) -> None:
    low, low_qc, low_proc = run_gaussian(tmp_path, 0.5)
    high, high_qc, high_proc = run_gaussian(tmp_path, 2.0)
    assert low_proc.returncode == 0
    assert high_proc.returncode == 0
    for kernel, qc in ((low, low_qc), (high, high_qc)):
        assert np.allclose(np.diag(kernel), 1.0)
        assert np.max(np.abs(kernel - kernel.T)) < 1e-6
        assert np.linalg.eigvalsh((kernel + kernel.T) / 2).min() >= -1e-5
        assert qc["gamma_source"] == "cli_gamma_multiplier"
        assert qc["number_of_samples"] == 3
    upper = np.triu_indices(3, k=1)
    assert high[upper].mean() < low[upper].mean()


def test_gamma_precedence() -> None:
    gamma, source, multiplier = resolve_gamma(0.25, 4.0, "8.0", 2.0)
    assert gamma == 0.25
    assert source == "explicit_gamma"
    assert multiplier is None

    gamma, source, multiplier = resolve_gamma(None, None, "3.0", 2.0)
    assert gamma == 1.5
    assert source == "environment_gamma_multiplier"
    assert multiplier == 3.0


def test_sample_order_length_must_match_kernel(tmp_path: Path) -> None:
    _, _, proc = run_gaussian(tmp_path, 1.0, order_rows=2)
    assert proc.returncode != 0
    assert "does not match sample-order rows" in proc.stderr


def test_invalid_environment_multiplier_fails() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        resolve_gamma(None, None, "bad", 2.0)
