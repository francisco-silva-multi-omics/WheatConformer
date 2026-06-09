from __future__ import annotations

import pandas as pd

from server_training_pipeline.tune_multikernel_hyperparameters import select_hyperparameters


def candidate_rows(
    ridge: float,
    rank_g: int,
    rank_g_rbf: int,
    rank_e: int,
    val_rmse: float,
    val_pearson: float,
    test_rmse: float,
    ablation: str = "G+RBF+E+GE+RBFE",
) -> list[dict[str, object]]:
    base = {
        "split_mode": "gho_environment",
        "ablation": ablation,
        "repeat": 0,
        "n": 20,
        "ridge": ridge,
        "rank_g": rank_g,
        "rank_g_rbf": rank_g_rbf,
        "rank_e": rank_e,
    }
    return [
        base | {"split": "val", "rmse": val_rmse, "pearson": val_pearson},
        base | {"split": "test", "rmse": test_rmse, "pearson": 0.99},
    ]


def test_selection_uses_validation_only_and_tie_breaks_with_pearson() -> None:
    rows = []
    rows += candidate_rows(0.1, 32, 32, 16, 0.8, 0.4, 0.01)
    rows += candidate_rows(1.0, 64, 64, 32, 0.7, 0.5, 10.0)
    rows += candidate_rows(10.0, 128, 128, 64, 0.7, 0.8, 0.001)
    metrics = pd.DataFrame(rows)

    summary, selected = select_hyperparameters(metrics, "gho_environment")
    changed_test = metrics.copy()
    changed_test.loc[changed_test["split"].eq("test"), "rmse"] = [999.0, 0.0, 500.0]
    _, selected_after_test_change = select_hyperparameters(changed_test, "gho_environment")

    assert selected["ridge"] == 10.0
    assert selected["rank_g"] == 128
    assert selected["test_metrics_used_for_selection"] is False
    assert selected_after_test_change == selected
    assert summary.loc[summary["selected"], "validation_pearson_mean"].item() == 0.8


def test_selection_compares_only_the_requested_integrated_ablation() -> None:
    rows = []
    rows += candidate_rows(1.0, 64, 64, 32, 0.6, 0.6, 0.5)
    rows += candidate_rows(100.0, 128, 128, 64, 0.01, 0.99, 0.01, ablation="RBF")

    summary, selected = select_hyperparameters(pd.DataFrame(rows), "gho_environment")

    assert len(summary) == 1
    assert selected["selection_ablation"] == "G+RBF+E+GE+RBFE"
    assert selected["ridge"] == 1.0
