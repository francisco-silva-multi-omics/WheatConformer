from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from server_training_pipeline.select_rbf_gamma import select_gamma


def test_gamma_selection_uses_validation_not_test(tmp_path: Path) -> None:
    kernel_root = tmp_path / "kernels"
    validation_root = tmp_path / "validation"
    kernel_root.mkdir()
    for multiplier, val_rmse, val_pearson, test_rmse in [
        (0.5, 1.0, 0.4, 0.1),
        (1.0, 0.8, 0.5, 10.0),
        (2.0, 0.8, 0.7, 0.01),
    ]:
        label = format(multiplier, "g")
        (kernel_root / f"K_HMP.gaussian.gammaMultiplier_{label}.npy").write_bytes(b"kernel")
        (kernel_root / f"K_HMP.gaussian.gammaMultiplier_{label}.qc.json").write_text(
            json.dumps({"gamma": multiplier / 2, "sampled_median_squared_distance": 2.0}),
            encoding="utf-8",
        )
        run = validation_root / "grain_yield" / f"gammaMultiplier_{label}"
        (run / "model_inputs").mkdir(parents=True)
        pd.DataFrame(
            [
                {"repeat": 0, "split_mode": "loeo", "ablation": "RBF", "split": "val", "n": 10, "rmse": val_rmse, "pearson": val_pearson},
                {"repeat": 0, "split_mode": "loeo", "ablation": "RBF", "split": "test", "n": 10, "rmse": test_rmse, "pearson": 0.9},
            ]
        ).to_csv(run / "gamma_sweep_metrics.tsv", sep="\t", index=False)

    summary, selected, manifest = select_gamma(
        "Grain Yield", [0.5, 1.0, 2.0], "loeo", validation_root, kernel_root, 2026, 1
    )
    assert selected["selected_gamma_multiplier"] == 2.0
    assert selected["test_metrics_used_for_selection"] is False
    assert summary.loc[summary["selected"], "gamma_multiplier"].item() == 2.0
    assert manifest.loc[manifest["selected"], "gamma_multiplier"].item() == 2.0
