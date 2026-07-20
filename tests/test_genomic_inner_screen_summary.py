from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from server_genotype_recovery.summarize_inner_screen import (
    architecture_name,
    load_runs,
    paired_metrics,
    summarize,
    validate_grid,
)


def write_run(
    models_dir: Path,
    architecture: str,
    outer: int,
    inner: int,
    nrmse: float,
    pearson: float,
) -> None:
    candidate = f"{architecture}_cfg0123456789"
    prefix = f"genomic_inner_unseen_genotypes_outer{outer}_{candidate}_inner{inner}"
    run_dir = models_dir / prefix
    run_dir.mkdir(parents=True)
    metadata = {
        "evaluation_stage": "inner_selection",
        "external_split": {
            "scenario": "unseen_genotypes",
            "outer_fold": outer,
            "inner_fold": inner,
        },
        "seed": 61001 + outer * 100 + inner * 10,
        "model_label": architecture,
        "hyperparameter_label": candidate,
        "training_configuration": {"latent_dim": 16},
    }
    (run_dir / "x_run_metadata.json").write_text(json.dumps(metadata))
    pd.DataFrame(
        [
            {
                "split": "val",
                "model": architecture,
                "macro_normalized_rmse": nrmse,
                "macro_pearson": pearson,
            }
        ]
    ).to_csv(run_dir / "x_macro_metrics.tsv", sep="\t", index=False)


def test_inner_screen_summary_validates_and_keeps_regulatory_panels(tmp_path: Path) -> None:
    models = tmp_path / "models"
    architectures = {
        "pedigree_environment_only": (0.65, 0.73),
        "frozen_existing_HMP_GBS": (0.66, 0.72),
        "existing_plus_K_G_TEST_LINEAR": (0.64, 0.73),
    }
    for outer in range(2):
        for inner in range(2):
            for architecture, (nrmse, pearson) in architectures.items():
                write_run(models, architecture, outer, inner, nrmse, pearson)

    runs = load_runs(models, "unseen_genotypes")
    validate_grid(runs, set(architectures), 2, 2)
    paired = paired_metrics(runs)
    summary = summarize(
        paired,
        minimum_relative_gain=0.01,
        minimum_win_rate=2.0 / 3.0,
        maximum_pearson_drop=0.005,
    )
    candidate = summary[
        summary["architecture"].eq("existing_plus_K_G_TEST_LINEAR")
    ].iloc[0]
    assert candidate["paired_inner_folds"] == 4
    assert candidate["quantitative_K_G_decision"] == "advance_pending_coverage_audit"
    assert candidate["regulatory_panel_retention"] == "retain_for_marker_to_graph_and_K_z"
    assert architecture_name("test_cfg0123456789") == "test"
