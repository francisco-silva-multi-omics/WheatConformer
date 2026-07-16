from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audit.refresh_observation_index_bundles import selected_warning_rows
from server_training_pipeline.observation_index_bundle import (
    compare_observation_index_bundle,
    observation_index_arrays,
    write_observation_index_bundle,
)


def observation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geno_kernel_index": [10, 20],
            "env_kernel_index": [5, 8],
            "phenotype_value": [1.5, -0.5],
            "weight_g_e": [1.0, 2.0],
            "var_g_e": [0.25, 0.5],
            "SE_g_e": [0.5, np.sqrt(0.5)],
        }
    )


def test_observation_index_bundle_round_trip(tmp_path: Path) -> None:
    frame = observation_frame()
    path = tmp_path / "indices.npz"

    result = write_observation_index_bundle(frame, path)
    comparison = compare_observation_index_bundle(frame, path)

    assert result["rows"] == 2
    assert comparison == {"schema_match": True, "rows_match": True, "values_match": True}
    with np.load(path) as bundle:
        assert bundle["geno_kernel_index"].dtype == np.int32
        assert bundle["y"].dtype == np.float32


def test_observation_index_bundle_rejects_nonfinite_or_fractional_indices() -> None:
    nonfinite = observation_frame()
    nonfinite.loc[0, "weight_g_e"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        observation_index_arrays(nonfinite)

    fractional = observation_frame()
    fractional["geno_kernel_index"] = fractional["geno_kernel_index"].astype(float)
    fractional.loc[0, "geno_kernel_index"] = 1.5
    with pytest.raises(ValueError, match="nonnegative integers"):
        observation_index_arrays(fractional)


def test_observation_index_bundle_preserves_supported_missing_uncertainty(tmp_path: Path) -> None:
    frame = observation_frame()
    frame.loc[0, ["weight_g_e", "var_g_e", "SE_g_e"]] = np.nan
    path = tmp_path / "indices_with_missing_uncertainty.npz"

    write_observation_index_bundle(frame, path)

    assert compare_observation_index_bundle(frame, path)["values_match"] is True


def test_refresh_selection_only_includes_auxiliary_bundle_warnings() -> None:
    validation = pd.DataFrame(
        {
            "model_dir": ["a", "b", "c"],
            "warning_count": [1, 1, 0],
            "warnings": [
                "auxiliary observation index NPZ is absent",
                "some other warning",
                "",
            ],
        }
    )
    assert selected_warning_rows(validation)["model_dir"].tolist() == ["a"]
