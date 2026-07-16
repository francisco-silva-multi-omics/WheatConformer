from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from server_training_pipeline.compare_multitrait_variants import compare_variants, main, summarize


def write_run(
    root: Path,
    variant: str,
    mode: str,
    seed: int,
    traits: list[str],
    rmse_shift: float,
    splits: tuple[str, ...] = ("test",),
) -> None:
    run = root / "trained_models" / f"multitrait_quantitative_{variant}_{mode}_seed{seed}"
    ledger = root / "model_kernels" / f"ledger_{variant}"
    run.mkdir(parents=True)
    ledger.mkdir(parents=True, exist_ok=True)
    factor_cache = ledger / f"factors_seed{seed}.npz"
    metadata = {
        "seed": seed,
        "canonical_split_mode": "gho_environment",
        "model_label": f"multitrait_{variant}_{mode}",
        "traits": traits,
        "rows": {"train": 100, "val": 20, "test": 30},
        "active_kernels": ["K_A", "K_E_MGMT", "K_E_TGW_V2"],
        "factor_cache": str(factor_cache),
    }
    (run / "model_run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "split": split,
                "coverage_group": "all",
                "model": metadata["model_label"],
                "trait_name_canonical": trait,
                "unweighted_rmse": 2.0 + rmse_shift,
                "normalized_rmse": 1.0 + rmse_shift,
                "pearson": 0.2 - rmse_shift,
                "prediction_sd_ratio": 0.8,
            }
            for split in splits
            for trait in traits
        ]
    ).to_csv(run / "model_trait_metrics.tsv", sep="\t", index=False)
    lineage = {
        "source_observations_sha256": "same-source",
        "weight_parameters": {"weight_power": 0.0},
    }
    (ledger / "ledger_lineage.json").write_text(json.dumps(lineage), encoding="utf-8")


def test_comparison_allows_seed_specific_but_pair_matched_traits(tmp_path: Path) -> None:
    for seed, traits in [(1, ["A", "B"]), (2, ["A"])]:
        write_run(tmp_path, "baseline", "env", seed, traits, rmse_shift=0.0)
        write_run(tmp_path, "corrected", "env", seed, traits, rmse_shift=-0.1)

    paired, contract, availability = compare_variants(
        root=tmp_path,
        models_root=tmp_path / "trained_models",
        baseline_variant="baseline",
        corrected_variant="corrected",
        modes=["env"],
        seeds=[1, 2],
        requested_traits=["A", "B"],
    )
    trait_summary, macro = summarize(paired)

    assert len(paired) == 3
    assert contract["supported_trait_count"].tolist() == [2, 1]
    missing = availability[
        availability["split"].eq("test")
        & availability["trait_name_canonical"].eq("B")
        & availability["seed"].eq(2)
    ].iloc[0]
    assert bool(missing["paired_available"]) is False
    assert trait_summary.set_index("trait_name_canonical").loc["B", "seed_count"] == 1
    assert macro.loc[0, "delta_normalized_rmse_trait_mean"] < 0


def test_comparison_audits_missing_runs_and_uses_only_matched_pairs(tmp_path: Path) -> None:
    write_run(tmp_path, "baseline", "env", 1, ["A", "B"], rmse_shift=0.0)
    write_run(tmp_path, "corrected", "env", 1, ["A", "B"], rmse_shift=-0.1)
    write_run(tmp_path, "corrected", "env", 2, ["A"], rmse_shift=-0.1)

    paired, contract, availability = compare_variants(
        root=tmp_path,
        models_root=tmp_path / "trained_models",
        baseline_variant="baseline",
        corrected_variant="corrected",
        modes=["env"],
        seeds=[1, 2, 3],
        requested_traits=["A", "B"],
    )

    _, macro = summarize(paired, contract=contract)

    assert len(paired) == 2
    assert contract["status"].tolist() == ["PASS", "SKIP", "SKIP"]
    assert contract["skip_reason"].tolist() == [
        "",
        "baseline_run_absent",
        "baseline_and_corrected_runs_absent",
    ]
    seed2 = availability[
        availability["seed"].eq(2)
        & availability["split"].eq("test")
        & availability["trait_name_canonical"].eq("A")
    ].iloc[0]
    assert bool(seed2["baseline_available"]) is False
    assert bool(seed2["corrected_available"]) is True
    assert bool(seed2["paired_available"]) is False
    assert macro.loc[0, "requested_pair_count"] == 3
    assert macro.loc[0, "matched_pair_count"] == 1
    assert macro.loc[0, "skipped_pair_count"] == 2
    assert bool(macro.loc[0, "comparison_grid_complete"]) is False


def test_comparison_reports_validation_and_test_separately(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "baseline",
        "env",
        1,
        ["A", "B"],
        rmse_shift=0.0,
        splits=("val", "test"),
    )
    write_run(
        tmp_path,
        "corrected",
        "env",
        1,
        ["A", "B"],
        rmse_shift=-0.1,
        splits=("val", "test"),
    )

    paired, contract, _ = compare_variants(
        root=tmp_path,
        models_root=tmp_path / "trained_models",
        baseline_variant="baseline",
        corrected_variant="corrected",
        modes=["env"],
        seeds=[1],
        requested_traits=["A", "B"],
    )
    trait_summary, macro = summarize(paired, contract=contract)

    assert set(paired["split"]) == {"val", "test"}
    assert len(paired) == 4
    assert set(trait_summary["split"]) == {"val", "test"}
    assert set(macro["split"]) == {"val", "test"}
    assert macro.groupby("split")["comparison_grid_complete"].first().all()


def test_comparison_cli_writes_all_reports(
    tmp_path: Path, monkeypatch
) -> None:
    write_run(tmp_path, "baseline", "env", 1, ["A", "B"], rmse_shift=0.0)
    write_run(tmp_path, "corrected", "env", 1, ["A", "B"], rmse_shift=-0.1)
    write_run(tmp_path, "corrected", "env", 2, ["A"], rmse_shift=-0.1)
    out_prefix = tmp_path / "comparisons" / "matched"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_multitrait_variants",
            "--root",
            str(tmp_path),
            "--baseline-variant",
            "baseline",
            "--corrected-variant",
            "corrected",
            "--modes",
            "env",
            "--seeds",
            "1,2",
            "--traits",
            "A,B",
            "--out-prefix",
            str(out_prefix),
        ],
    )

    main()

    for suffix in ["paired", "contract", "trait_availability", "trait_summary", "macro_summary"]:
        assert out_prefix.with_name(f"{out_prefix.name}_{suffix}.tsv").is_file()
