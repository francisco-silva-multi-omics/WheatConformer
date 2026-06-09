from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import build_environment_component_kernels as environment_kernels
from build_environment_component_kernels import component_weights, scale_kernel_mean_diagonal


FIXTURES = Path(__file__).parent / "fixtures"


def test_nonempty_environment_kernel_scales_to_mean_diagonal_one() -> None:
    raw = np.array([[4.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    scaled, mean_raw, mean_scaled = scale_kernel_mean_diagonal(raw)
    assert mean_raw == 3.0
    assert np.isclose(mean_scaled, 1.0)
    assert np.isclose(np.diag(scaled).mean(), 1.0)


def test_component_weights_normalize_nonempty_components(monkeypatch) -> None:
    monkeypatch.setenv("ENV_WEIGHT_GEO", "2")
    monkeypatch.setenv("ENV_WEIGHT_WEATHER", "1")
    raw, normalized = component_weights(["geo", "weather"], ["geo", "weather", "stress", "mgmt"])
    assert raw["stress"] == 0.0
    assert np.isclose(sum(normalized.values()), 1.0)
    assert np.isclose(normalized["geo"], 2 / 3)


def test_environment_kernel_main_writes_scaled_components_and_provenance(tmp_path, monkeypatch) -> None:
    shutil.copyfile(FIXTURES / "toy_envdata.tsv", tmp_path / "envdata.tsv")
    shutil.copyfile(FIXTURES / "toy_locdata.tsv", tmp_path / "locdata.tsv")
    monkeypatch.setattr(environment_kernels, "OUT", tmp_path)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: None)
    environment_kernels.main()

    for component in ["geo", "weather", "stress", "mgmt"]:
        assert (tmp_path / f"K_{component}.raw.npy").exists()
        assert (tmp_path / f"K_{component}.npy").exists()
        scaled = np.load(tmp_path / f"K_{component}.npy")
        if np.diag(scaled).mean() > 0:
            assert np.isclose(np.diag(scaled).mean(), 1.0)

    weights = pd.read_csv(tmp_path / "env_kernel_component_weights.tsv", sep="\t")
    assert weights.columns.tolist() == [
        "kernel",
        "raw_weight",
        "normalized_weight",
        "feature_count",
        "mean_diag_raw",
        "mean_diag_scaled",
        "coverage_env_count",
    ]
    assert np.isclose(weights["normalized_weight"].sum(), 1.0)
    assert (tmp_path / "qc_location_key_collisions.tsv").exists()
    assert np.isclose(np.diag(np.load(tmp_path / "K_E.npy")).mean(), 1.0)
