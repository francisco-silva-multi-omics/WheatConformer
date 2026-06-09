from __future__ import annotations

import pandas as pd
import pytest

from server_training_pipeline.run_validation_ablation_suite import map_compact


def test_compact_kernel_sample_order_mapping(tmp_path) -> None:
    observations = pd.DataFrame({"geno_kernel_index": [3, 1, 3, 2]})
    order = pd.DataFrame(
        {
            "source_kernel_index": [1, 2, 3],
            "compact_kernel_index": [0, 1, 2],
        }
    )
    path = tmp_path / "order.tsv"
    order.to_csv(path, sep="\t", index=False)
    assert map_compact(observations, "geno_kernel_index", path).tolist() == [2, 0, 2, 1]


def test_missing_kernel_sample_order_mapping_fails(tmp_path) -> None:
    observations = pd.DataFrame({"geno_kernel_index": [1, 99]})
    path = tmp_path / "order.tsv"
    pd.DataFrame({"source_kernel_index": [1], "compact_kernel_index": [0]}).to_csv(path, sep="\t", index=False)
    with pytest.raises(SystemExit, match="Could not map all rows"):
        map_compact(observations, "geno_kernel_index", path)
