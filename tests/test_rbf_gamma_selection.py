from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from server_training_pipeline.select_rbf_gamma import combine_trait_manifests, select_gamma


def test_gamma_selection_uses_validation_not_test(tmp_path: Path) -> None:
    kernel_root = tmp_path / "kernels"
    validation_root = tmp_path / "validation"
    (kernel_root / "grain_yield").mkdir(parents=True)
    for multiplier, val_rmse, val_pearson, test_rmse in [
        (0.5, 1.0, 0.4, 0.1),
        (1.0, 0.8, 0.5, 10.0),
        (2.0, 0.8, 0.7, 0.01),
    ]:
        label = format(multiplier, "g")
        (kernel_root / "grain_yield" / f"K_HMP.gaussian.gammaMultiplier_{label}.npy").write_bytes(b"kernel")
        (kernel_root / "grain_yield" / f"K_HMP.gaussian.gammaMultiplier_{label}.qc.json").write_text(
            json.dumps({"gamma": multiplier / 2, "sampled_median_squared_distance": 2.0}),
            encoding="utf-8",
        )
        run = validation_root / "grain_yield" / f"gammaMultiplier_{label}"
        (run / "model_inputs").mkdir(parents=True)
        pd.DataFrame(
            [
                    {"repeat": 0, "split_mode": "gho_environment", "ablation": "G+RBF+E+GE+RBFE", "split": "val", "n": 10, "rmse": val_rmse, "pearson": val_pearson},
                    {"repeat": 0, "split_mode": "gho_environment", "ablation": "G+RBF+E+GE+RBFE", "split": "test", "n": 10, "rmse": test_rmse, "pearson": 0.9},
                    {"repeat": 0, "split_mode": "gho_environment", "ablation": "RBF", "split": "val", "n": 10, "rmse": 99, "pearson": 0.0},
            ]
        ).to_csv(run / "gamma_sweep_metrics.tsv", sep="\t", index=False)
        pd.DataFrame([{"split_mode": "gho_environment", "leakage_status": "pass"}]).to_csv(
            run / "split_leakage_qc.tsv", sep="\t", index=False
        )

    summary, selected, manifest = select_gamma(
        "Grain Yield", [0.5, 1.0, 2.0], "gho_environment", validation_root, kernel_root, 2026, 1
    )
    assert selected["selected_gamma_multiplier"] == 2.0
    assert selected["test_metrics_used_for_selection"] is False
    assert selected["selection_ablation"] == "G+RBF+E+GE+RBFE"
    assert "RBF" in selected["diagnostic_ablations"]
    assert summary.loc[summary["selected"], "gamma_multiplier"].item() == 2.0
    assert manifest.loc[manifest["selected"], "gamma_multiplier"].item() == 2.0


def test_two_traits_have_distinct_and_combined_manifests(tmp_path: Path) -> None:
    for slug, trait in [("grain_yield", "Grain Yield"), ("heading", "Heading")]:
        directory = tmp_path / slug
        directory.mkdir()
        pd.DataFrame([{"trait": trait, "trait_slug": slug, "selected": True}]).to_csv(
            directory / "gamma_sweep_manifest.tsv", sep="\t", index=False
        )
    combined = combine_trait_manifests(tmp_path)
    assert set(combined["trait_slug"]) == {"grain_yield", "heading"}
