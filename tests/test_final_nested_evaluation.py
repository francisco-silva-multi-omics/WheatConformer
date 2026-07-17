from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from build_environment_component_kernels import (
    scale_kernel_mean_diagonal,
    standardized_kernel,
)
from server_training_pipeline.build_final_evaluation_manifests import (
    choose_final_cycle_block,
    choose_final_environment_block,
    genotype_expert_support_table,
    main as build_manifests,
)
from server_training_pipeline.final_evaluation_contract import (
    load_protocol,
    require_non_discovery_seed,
)
from server_training_pipeline.nested_evaluation import (
    assign_nested_split,
    verify_manifest_contract,
)
from server_training_pipeline.ensemble_nested_outer_predictions import main as ensemble_outer
from server_training_pipeline.select_nested_hyperparameters import main as select_hyperparameters
from server_training_pipeline.observation_weights import (
    apply_precision_weight_transform,
    fit_precision_weight_transform,
)


def synthetic_ledger() -> pd.DataFrame:
    rows = []
    genotypes = [f"g{i:02d}" for i in range(30)]
    traits = ["DAYS_TO_HEADING", "GRAIN_YIELD"]
    for environment_index in range(60):
        cycle = str(2000 + environment_index // 5)
        environment = f"e{environment_index:02d}"
        country = f"country{environment_index:02d}"
        for genotype in genotypes:
            for trait in traits:
                rows.append(
                    {
                        "canonical_observation_id": f"{environment}-{genotype}-{trait}",
                        "panel_sample_id": genotype,
                        "env_kernel_id": environment,
                        "cycle": cycle,
                        "country": country,
                        "trait_name_canonical": trait,
                        "phenotype_value": float(environment_index),
                        "raw_mean": float(environment_index),
                        "var_g_e": 1.0,
                        "weight_g_e": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def build_toy_manifests(tmp_path: Path, monkeypatch) -> tuple[pd.DataFrame, Path, Path, pd.DataFrame]:
    ledger = synthetic_ledger()
    ledger_path = tmp_path / "ledger.tsv"
    out_dir = tmp_path / "evaluation"
    protocol_path = tmp_path / "protocol.json"
    protocol = json.loads(
        Path("server_training_pipeline/final_evaluation_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    protocol.update(
        {
            "protocol_version": "toy_multitrait_quantitative_final_v2",
            "traits": ["DAYS_TO_HEADING", "GRAIN_YIELD"],
            "climatology_eligible_traits": ["DAYS_TO_HEADING", "GRAIN_YIELD"],
            "climatology_ineligible_traits": [],
            "final_holdout_support": {
                "minimum_environment_fraction": 0.15,
                "minimum_environment_count": 10,
                "maximum_environment_fraction": 0.4,
                "minimum_rows_per_trait": 20,
                "minimum_environments_per_trait": 5,
            },
        }
    )
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    ledger.to_csv(ledger_path, sep="\t", index=False)
    protected_orders = []
    for name in ["K_G_HMP", "K_G_GBS"]:
        path = tmp_path / f"{name}_order.tsv"
        pd.DataFrame({"sample_id": sorted(ledger["panel_sample_id"].unique())}).to_csv(
            path, sep="\t", index=False
        )
        protected_orders.extend(["--protected-genotype-order", f"{name}={path}"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_final_evaluation_manifests",
            "--ledger",
            str(ledger_path),
            "--out-dir",
            str(out_dir),
            "--protocol",
            str(protocol_path),
            *protected_orders,
        ],
    )
    build_manifests()
    manifest_path = out_dir / "nested_evaluation_entities.tsv"
    contract_path = out_dir / "nested_evaluation_contract.json"
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    return ledger, manifest_path, contract_path, manifest


def test_frozen_protocol_blocks_discovery_seeds() -> None:
    protocol = load_protocol()
    assert protocol["climatology_eligible_traits"] == [
        "DAYS_TO_HEADING",
        "DAYS_TO_MATURITY",
        "GRAIN_YIELD",
    ]
    try:
        require_non_discovery_seed(2026, protocol)
    except ValueError as exc:
        assert "discovery" in str(exc)
    else:
        raise AssertionError("Discovery seed was not blocked")
    require_non_discovery_seed(41001, protocol)


def test_manifest_is_hashed_and_final_holdout_is_excluded(tmp_path, monkeypatch) -> None:
    ledger, manifest_path, contract_path, manifest = build_toy_manifests(tmp_path, monkeypatch)
    contract = verify_manifest_contract(manifest_path, contract_path)
    assert contract["status"] == "frozen"
    assert contract["final_holdout_environment_count"] == 10
    assert contract["final_holdout_preflight_status"] == "pass"
    assert contract["final_holdout_policy"] == "deterministic_environment_block_minimum_support"
    support = pd.read_csv(
        contract_path.parent / "final_holdout_trait_support.tsv", sep="\t"
    )
    assert support["support_status"].eq("PASS").all()
    expert_support = pd.read_csv(
        contract_path.parent / "final_holdout_genotype_expert_support.tsv", sep="\t"
    )
    assert expert_support["support_status"].eq("PASS").all()
    assert set(expert_support["kernel_expert"]) == {"K_G_HMP", "K_G_GBS"}
    final_ids = set(
        manifest.loc[manifest["partition"].eq("final_holdout"), "entity_id"]
    )
    train, val, test, omitted, qc = assign_nested_split(
        ledger,
        manifest,
        scenario="unseen_environments",
        outer_fold=0,
        inner_fold=0,
    )
    used = set(ledger.iloc[np.concatenate([train, val, test])]["env_kernel_id"])
    assert used.isdisjoint(final_ids)
    assert set(ledger.iloc[omitted]["env_kernel_id"]) >= final_ids
    assert qc["leakage_status"] == "pass"


def test_single_recent_cycle_fails_final_holdout_preflight() -> None:
    protocol = {
        "traits": ["DAYS_TO_HEADING", "GRAIN_YIELD"],
        "final_holdout_policy": "recent_cycle_block_minimum_support",
        "final_holdout_support": {
            "minimum_environment_fraction": 0.15,
            "minimum_environment_count": 10,
            "maximum_environment_fraction": 0.4,
            "minimum_rows_per_trait": 20,
            "minimum_environments_per_trait": 5,
        },
    }
    _, environments, support, _, preflight = choose_final_cycle_block(
        synthetic_ledger(), protocol, "2011"
    )
    assert len(environments) == 5
    assert support["support_status"].eq("PASS").all()
    assert preflight["status"] == "fail"
    assert "below 10" in preflight["failures"][0]


def test_environment_block_preserves_recent_cycle_marker_support() -> None:
    ledger = synthetic_ledger()
    environment_number = pd.to_numeric(
        ledger["env_kernel_id"].str.removeprefix("e"), errors="raise"
    )
    original = ledger["panel_sample_id"].str.removeprefix("g")
    ledger["panel_sample_id"] = np.where(
        environment_number.ge(50), "h" + original, "p" + original
    )
    protocol = load_protocol().copy()
    protocol["traits"] = ["DAYS_TO_HEADING", "GRAIN_YIELD"]
    protocol["final_holdout_support"] = {
        "minimum_environment_fraction": 0.15,
        "minimum_environment_count": 10,
        "maximum_environment_fraction": 0.4,
        "minimum_rows_per_trait": 20,
        "minimum_environments_per_trait": 5,
    }
    protected = {
        "K_G_HMP": {f"h{i:02d}" for i in range(30)},
        "K_G_GBS": {f"p{i:02d}" for i in range(30)},
    }

    _, recent_environments, _, _, _ = choose_final_cycle_block(ledger, protocol, None)
    recent_support = genotype_expert_support_table(
        ledger,
        recent_environments,
        protected,
        protocol["final_holdout_genotype_expert_support"],
    ).set_index("kernel_expert")
    assert recent_support.loc["K_G_HMP", "development_unique_genotypes"] == 0
    assert recent_support.loc["K_G_HMP", "support_status"] == "FAIL"

    _, selected, _, _, preflight, expert_support = choose_final_environment_block(
        ledger, protocol, protected
    )
    expert_support = expert_support.set_index("kernel_expert")
    assert preflight["status"] == "pass"
    assert len(selected) == 10
    assert expert_support["support_status"].eq("PASS").all()
    assert expert_support.loc["K_G_HMP", "development_unique_genotypes"] == 30
    assert expert_support.loc["K_G_HMP", "holdout_unique_genotypes"] == 30


def test_manifest_contract_protects_holdout_support_artifacts(tmp_path, monkeypatch) -> None:
    _, manifest_path, contract_path, _ = build_toy_manifests(tmp_path, monkeypatch)
    support_path = contract_path.parent / "final_holdout_trait_support.tsv"
    support_path.write_text(
        support_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="trait_support hash mismatch"):
        verify_manifest_contract(manifest_path, contract_path)


def test_manifest_contract_protects_genotype_expert_orders(tmp_path, monkeypatch) -> None:
    _, manifest_path, contract_path, _ = build_toy_manifests(tmp_path, monkeypatch)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    order_path = Path(contract["protected_genotype_order_identities"]["K_G_HMP"]["path"])
    order_path.write_text(order_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Protected genotype order hash mismatch"):
        verify_manifest_contract(manifest_path, contract_path)


def test_cv0_manifest_blocks_mixed_held_axes(tmp_path, monkeypatch) -> None:
    ledger, _, _, manifest = build_toy_manifests(tmp_path, monkeypatch)
    train, val, test, omitted, qc = assign_nested_split(
        ledger,
        manifest,
        scenario="unseen_genotypes_and_environments",
        outer_fold=1,
        inner_fold=2,
    )
    assert qc["leakage_status"] == "pass"
    assert not set(ledger.iloc[train]["panel_sample_id"]) & set(
        ledger.iloc[test]["panel_sample_id"]
    )
    assert not set(ledger.iloc[train]["env_kernel_id"]) & set(
        ledger.iloc[test]["env_kernel_id"]
    )
    assert len(omitted) > 0


def test_all_nested_scenarios_accept_arrow_backed_ledger(tmp_path, monkeypatch) -> None:
    ledger, _, _, manifest = build_toy_manifests(tmp_path, monkeypatch)
    for column in ["panel_sample_id", "env_kernel_id", "cycle", "country"]:
        ledger[column] = ledger[column].astype("string[pyarrow]")
    for scenario in [
        "unseen_environments",
        "unseen_genotypes",
        "unseen_genotypes_and_environments",
        "temporal_holdout",
        "country_holdout",
    ]:
        train, val, test, _, qc = assign_nested_split(
            ledger,
            manifest,
            scenario=scenario,
            outer_fold=0,
            inner_fold=0,
        )
        assert min(len(train), len(val), len(test)) > 0
        assert qc["leakage_status"] == "pass"


def test_fold_weight_statistics_ignore_validation_outlier() -> None:
    train = pd.DataFrame(
        {
            "trait_name_canonical": ["T"] * 4,
            "var_g_e": [1.0, 2.0, 3.0, 4.0],
        }
    )
    full = pd.concat(
        [
            train,
            pd.DataFrame(
                {"trait_name_canonical": ["T"], "var_g_e": [1e-12]}
            ),
        ],
        ignore_index=True,
    )
    parameters = fit_precision_weight_transform(
        train,
        weight_power=1.0,
        min_effective_sample_fraction=0.0,
        max_top_1pct_share=1.0,
    )
    assert float(parameters.iloc[0]["variance_floor"]) > 1e-12
    transformed = apply_precision_weight_transform(full, parameters)
    assert np.isfinite(transformed["weight_g_e"]).all()
    assert transformed.loc[4, "weight_variance_floored"]


def test_environment_scaling_and_diagonal_use_training_ids_only() -> None:
    features = pd.DataFrame(
        {"x": [0.0, 2.0, 100.0, np.nan]}, index=["train_a", "train_b", "test", "missing"]
    )
    kernel, standardized, scaling = standardized_kernel(
        features, fit_index=pd.Index(["train_a", "train_b"])
    )
    assert float(scaling.iloc[0]["mean"]) == 1.0
    assert standardized.loc["missing", "x"] == 0.0
    scaled, _, scaled_mean = scale_kernel_mean_diagonal(kernel, [0, 1])
    assert np.isclose(scaled_mean, 1.0)
    assert np.isclose(np.diag(scaled)[[0, 1]].mean(), 1.0)


def test_inner_selection_reads_validation_only(tmp_path, monkeypatch) -> None:
    models = tmp_path / "models"
    for candidate, score in [("base", 0.8), ("regularized", 0.7)]:
        for inner in range(3):
            run = models / f"inner_{candidate}_{inner}"
            run.mkdir(parents=True)
            metadata = {
                "evaluation_stage": "inner_selection",
                "model_label": "model",
                "hyperparameter_label": candidate,
                "external_split": {"inner_fold": inner},
                "training_configuration": {"weight_decay": 0.1 if candidate == "regularized" else 0.0},
            }
            (run / "run_run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "split": "val",
                        "model": "model",
                        "macro_normalized_rmse": score + inner * 0.01,
                        "macro_pearson": 0.3,
                    }
                ]
            ).to_csv(run / "run_macro_metrics.tsv", sep="\t", index=False)
    output = tmp_path / "decision.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_nested_hyperparameters",
            "--models-root",
            str(models),
            "--run-glob",
            "inner_*",
            "--expected-inner-folds",
            "3",
            "--out",
            str(output),
        ],
    )
    select_hyperparameters()
    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["selected_candidate"] == "regularized"
    assert decision["outer_test_metrics_read"] is False


def test_outer_ensemble_averages_test_and_keeps_oof_validation(tmp_path, monkeypatch) -> None:
    models = tmp_path / "models"
    for inner in range(3):
        run = models / f"member_{inner}"
        run.mkdir(parents=True)
        metadata = {
            "evaluation_stage": "outer_evaluation",
            "model_label": "model",
            "external_split": {
                "scenario": "unseen_environments",
                "outer_fold": 0,
                "inner_fold": inner,
            },
        }
        (run / "run_run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        frame = pd.DataFrame(
            [
                {
                    "canonical_observation_id": f"val{inner}",
                    "split": "val",
                    "phenotype_value": 1.0,
                    "trait_name_canonical": "T",
                    "panel_sample_id": f"g{inner}",
                    "env_kernel_id": f"v{inner}",
                    "y_pred": 1.0,
                    "y_pred_scaled": 0.0,
                    "y_pred_train_mean": 0.5,
                },
                {
                    "canonical_observation_id": "test0",
                    "split": "test",
                    "phenotype_value": 4.0,
                    "trait_name_canonical": "T",
                    "panel_sample_id": "gtest",
                    "env_kernel_id": "etest",
                    "y_pred": float(inner + 1),
                    "y_pred_scaled": float(inner),
                    "y_pred_train_mean": 0.5,
                },
            ]
        )
        frame.to_csv(run / "run_predictions.tsv.gz", sep="\t", index=False)
    out = tmp_path / "ensemble"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ensemble_nested_outer_predictions",
            "--models-root",
            str(models),
            "--run-glob",
            "member_*",
            "--expected-inner-folds",
            "3",
            "--out-dir",
            str(out),
            "--prefix",
            "combined",
        ],
    )
    ensemble_outer()
    parquet = out / "combined_predictions.parquet"
    combined = (
        pd.read_parquet(parquet)
        if parquet.exists()
        else pd.read_csv(out / "combined_predictions.tsv.gz", sep="\t")
    )
    assert len(combined[combined["split"].eq("val")]) == 3
    assert combined.loc[combined["split"].eq("test"), "y_pred"].iloc[0] == 2.0
    metadata = json.loads((out / "combined_run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["ensemble"]["outer_test_used_for_selection"] is False
