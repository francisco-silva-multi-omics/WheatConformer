from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from server_training_pipeline.loss_balance import (
    fold_local_balanced_loss_weights,
    loss_weight_diagnostics,
)
from server_training_pipeline.audit_reaction_norm_loss_leverage import (
    main as leverage_main,
)
from server_training_pipeline.final_evaluation_contract import file_sha256
from server_training_pipeline.summarize_reaction_norm_loss_balance_screen import (
    main as summarize_main,
)
from server_training_pipeline.train_multitrait_multikernel_tf import (
    trait_balanced_loss_weights,
)
from server_training_pipeline import train_multitrait_reaction_norm_balanced_tf as wrapper


def policy(name: str, alpha_environment: float, alpha_genotype: float) -> dict:
    return {
        "name": name,
        "environment_count_exponent": alpha_environment,
        "genotype_count_exponent": alpha_genotype,
        "minimum_relative_weight": None,
        "maximum_relative_weight": None,
    }


def test_frozen_loss_screen_uses_the_frozen_reaction_selection_scenario() -> None:
    root = Path(__file__).resolve().parents[1]
    loss = json.loads(
        (
            root
            / "server_training_pipeline"
            / "reaction_norm_loss_balance_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    reaction = json.loads(
        (
            root / "server_training_pipeline" / "reaction_norm_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    scenarios = {
        scenario
        for phase in ("phase_1", "confirmation")
        for scenario in loss[phase]["outer_folds_by_scenario"]
    }
    assert scenarios == {loss["selection_scenario"]} == {reaction["scenario"]}


def training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trait_name_canonical": ["A"] * 6 + ["B"] * 4,
            "trait_index": [0] * 6 + [1] * 4,
            "environment_id": ["E1"] * 4 + ["E2"] * 2 + ["E1"] * 2 + ["E2"] * 2,
            "genotype_id": ["G1"] * 3 + ["G2", "G3", "G4", "G1", "G2", "G3", "G4"],
            "weight_g_e": [1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 2.0],
        }
    )


def test_current_loss_policy_exactly_matches_legacy_trait_balancing() -> None:
    frame = training_frame()
    observed = fold_local_balanced_loss_weights(frame, policy("current", 0.0, 0.0))
    expected = trait_balanced_loss_weights(
        frame["trait_index"].to_numpy(), frame["weight_g_e"].to_numpy()
    )
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-7)


def test_environment_balancing_equalizes_environment_mass_within_trait() -> None:
    frame = training_frame()
    frame["weight_g_e"] = 1.0
    weights = fold_local_balanced_loss_weights(
        frame, policy("environment", 1.0, 0.0)
    )
    local = frame.assign(loss_weight=weights)
    shares = local.groupby(["trait_name_canonical", "environment_id"])[
        "loss_weight"
    ].sum()
    np.testing.assert_allclose(shares.loc["A"].to_numpy(), [2.5, 2.5])
    np.testing.assert_allclose(shares.loc["B"].to_numpy(), [2.5, 2.5])


def test_damped_two_way_policy_reduces_repeated_genotype_leverage() -> None:
    frame = training_frame()
    frame["weight_g_e"] = 1.0
    current = fold_local_balanced_loss_weights(frame, policy("current", 0.0, 0.0))
    damped = fold_local_balanced_loss_weights(frame, policy("damped", 0.5, 0.5))
    current_diag = loss_weight_diagnostics(frame, current, policy_name="current")
    damped_diag = loss_weight_diagnostics(frame, damped, policy_name="damped")
    current_a = current_diag.set_index("trait_name_canonical").loc["A"]
    damped_a = damped_diag.set_index("trait_name_canonical").loc["A"]
    assert (
        damped_a["maximum_genotype_weight_share"]
        < current_a["maximum_genotype_weight_share"]
    )


def loss_protocol(path: Path, candidates: list[dict] | None = None) -> Path:
    content = {
        "protocol_version": "test_loss_v1",
        "status": "frozen_before_inner_validation",
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "selected_reaction_candidate": "reaction_norm_identity_covariance",
        "selected_environment_architecture": "explicit_E_REACTION_NORM_V1",
        "selection_scenario": "unseen_genotypes",
        "precision_weight_power": 0.0,
        "candidates": candidates
        or [
            policy("current_trait_row_balanced", 0.0, 0.0),
            policy("trait_environment_balanced", 1.0, 0.0),
            policy("damped_environment_genotype_balanced", 0.5, 0.5),
        ],
        "phase_1": {
            "outer_folds_by_scenario": {"unseen_genotypes": [0]},
            "inner_folds": 3,
        },
        "confirmation": {
            "outer_folds_by_scenario": {"unseen_genotypes": [0]},
            "inner_folds": 3,
        },
        "acceptance": {
            "minimum_relative_normalized_rmse_gain": 0.01,
            "minimum_paired_inner_fold_win_rate": 2 / 3,
            "maximum_mean_pearson_drop": 0.005,
            "maximum_mean_calibration_error_increase": 0.0,
            "require_positive_unseen_genotype_gain": True,
            "maximum_primary_trait_relative_nrmse_loss": 0.01,
            "primary_guard_traits": [
                "DAYS_TO_HEADING",
                "DAYS_TO_MATURITY",
                "PLANT_HEIGHT",
                "GRAIN_YIELD",
                "1000_GRAIN_WEIGHT",
            ],
            "exploratory_traits": ["ABOVE_GROUND_BIOMASS", "TEST_WEIGHT"],
        },
    }
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_balanced_wrapper_records_policy_and_restores_base_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    protocol_path = loss_protocol(
        tmp_path / "loss.json",
        [policy("trait_environment_balanced", 1.0, 0.0)],
    )
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    observed_weights: list[np.ndarray] = []

    def fake_dataset(frame, expert_columns, batch_size, shuffle, seed):
        observed_weights.append(frame["loss_weight"].to_numpy(copy=True))
        return "dataset"

    def fake_main():
        arguments = sys.argv[1:]
        prefix = arguments[arguments.index("--prefix") + 1]
        local_out = Path(arguments[arguments.index("--out-dir") + 1])
        frame = training_frame()
        frame["loss_weight"] = 1.0
        wrapper.base.make_dataset(frame, [], 16, True, 7)
        metadata = {
            "status": "PASS",
            "trainer_sha256": "base",
            "hyperparameter_label": "explicit_E_REACTION_NORM_V1",
            "model_family": "base",
            "phenotype_preprocessing": {},
        }
        (local_out / f"{prefix}_run_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    original = wrapper.base.make_dataset
    monkeypatch.setattr(wrapper.base, "make_dataset", fake_dataset)
    monkeypatch.setattr(wrapper.base, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "balanced",
            "--loss-balance-protocol",
            str(protocol_path),
            "--loss-balance-candidate",
            "trait_environment_balanced",
            "--evaluation-stage",
            "inner_selection",
            "--evaluation-scenario",
            "unseen_genotypes",
            "--reaction-candidate",
            "reaction_norm_identity_covariance",
            "--environment-architecture",
            "explicit_E_REACTION_NORM_V1",
            "--weight-power",
            "0",
            "--out-dir",
            str(out_dir),
            "--prefix",
            "run",
        ],
    )
    wrapper.main()
    metadata = json.loads((out_dir / "run_run_metadata.json").read_text())
    assert metadata["loss_balance"]["candidate"] == "trait_environment_balanced"
    assert metadata["loss_balance"]["count_fit_partition"] == "inner_training_only"
    assert metadata["loss_balance"]["outer_test_metrics_read"] is False
    assert (out_dir / "run_loss_weight_diagnostics.tsv").is_file()
    assert not np.allclose(observed_weights[0], 1.0)
    assert wrapper.base.make_dataset is fake_dataset
    monkeypatch.setattr(wrapper.base, "make_dataset", original)


def test_loss_leverage_audit_reads_identifiers_and_training_support_only(
    tmp_path: Path, monkeypatch
) -> None:
    ledger_path = tmp_path / "ledger.parquet"
    rows = []
    for genotype in ("G1", "G2", "G3"):
        for trait in ("DAYS_TO_HEADING", "GRAIN_YIELD"):
            rows.append(
                {
                    "canonical_observation_id": f"{genotype}_{trait}",
                    "trait_name_canonical": trait,
                    "genotype_id": genotype,
                    "environment_id": "E1",
                    "panel_sample_id": genotype,
                    "env_kernel_id": "E1",
                    "cycle": "2020",
                    "country": "MEXICO",
                    "weight_g_e": 1.0,
                }
            )
    ledger = pd.DataFrame(rows)
    ledger.to_parquet(ledger_path, index=False)
    readiness_path = tmp_path / "readiness.parquet"
    pd.DataFrame(
        {
            "canonical_observation_id": ledger["canonical_observation_id"],
            "canonical_germplasm_key": ledger["genotype_id"],
            "env_kernel_id": ["TRIAL|1"] * len(ledger),
            "recovery_readiness": [
                "RETAINED_REFERENCE" if value == "G1" else "P1_RECOVER_ENVIRONMENT"
                for value in ledger["genotype_id"]
            ],
        }
    ).to_parquet(readiness_path, index=False)
    manifest_path = tmp_path / "entities.tsv"
    pd.DataFrame(
        [
            {
                "scenario": "unseen_genotypes",
                "outer_fold": 0,
                "inner_fold": 0,
                "axis": "genotype",
                "partition": "outer_test",
                "entity_id": "G3",
            },
            {
                "scenario": "unseen_genotypes",
                "outer_fold": 0,
                "inner_fold": 0,
                "axis": "genotype",
                "partition": "inner_validation",
                "entity_id": "G2",
            },
        ]
    ).to_csv(manifest_path, sep="\t", index=False)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "status": "frozen",
                "ledger_sha256": file_sha256(ledger_path),
                "entity_manifest_sha256": file_sha256(manifest_path),
            }
        )
    )
    protocol_path = loss_protocol(tmp_path / "loss.json")
    protocol = json.loads(protocol_path.read_text())
    protocol["confirmation"] = {
        "outer_folds_by_scenario": {"unseen_genotypes": [0]},
        "inner_folds": 1,
    }
    protocol_path.write_text(json.dumps(protocol))
    pedigree = tmp_path / "parents.tsv"
    pd.DataFrame(
        {
            "sample_id": ["G1", "G2", "G3"],
            "parent1": ["P1", "P1", "P2"],
            "parent2": ["P2", "P2", "P3"],
        }
    ).to_csv(pedigree, sep="\t", index=False)
    out = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--ledger",
            str(ledger_path),
            "--readiness-ledger",
            str(readiness_path),
            "--split-manifest",
            str(manifest_path),
            "--split-contract",
            str(contract_path),
            "--loss-balance-protocol",
            str(protocol_path),
            "--pedigree-parent-table",
            str(pedigree),
            "--out-dir",
            str(out),
        ],
    )
    leverage_main()
    provenance = json.loads(
        (out / "reaction_norm_loss_leverage_provenance.json").read_text()
    )
    assert provenance["status"] == "PASS"
    assert provenance["fold_count"] == 1
    assert provenance["candidate_count"] == 3
    assert provenance["phenotype_values_read"] is False


def write_screen_run(
    root: Path,
    *,
    candidate: str,
    inner_fold: int,
    normalized_rmse: float,
    pearson: float,
    prediction_sd_ratio: float,
    protocol_sha: str,
) -> None:
    name = f"loss_balance_inner_unseen_genotypes_outer0_{candidate}_inner{inner_fold}"
    run_dir = root / name
    run_dir.mkdir(parents=True)
    model = f"model_{candidate}"
    metadata = {
        "status": "PASS",
        "evaluation_stage": "inner_selection",
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "model_label": model,
        "seed": 81001 + inner_fold * 10,
        "external_split": {
            "scenario": "unseen_genotypes",
            "outer_fold": 0,
            "inner_fold": inner_fold,
            "manifest_sha256": "manifest",
        },
        "evaluation_protocol": {"protocol_sha256": "evaluation"},
        "loss_balance": {"candidate": candidate, "protocol_sha256": protocol_sha},
        "training_configuration": {"same": True},
        "active_kernels": ["K_A", "K_E"],
        "training_input_identities": {"same": True},
    }
    (run_dir / "m_run_metadata.json").write_text(json.dumps(metadata))
    pd.DataFrame(
        [
            {
                "split": "val",
                "model": model,
                "macro_normalized_rmse": normalized_rmse,
                "macro_pearson": pearson,
            }
        ]
    ).to_csv(run_dir / "m_macro_metrics.tsv", sep="\t", index=False)
    traits = [
        "DAYS_TO_HEADING",
        "DAYS_TO_MATURITY",
        "PLANT_HEIGHT",
        "GRAIN_YIELD",
        "1000_GRAIN_WEIGHT",
    ]
    pd.DataFrame(
        [
            {
                "split": "val",
                "model": model,
                "coverage_group": "all",
                "trait_name_canonical": trait,
                "normalized_rmse": normalized_rmse,
                "pearson": pearson,
                "prediction_sd_ratio": prediction_sd_ratio,
            }
            for trait in traits
        ]
    ).to_csv(run_dir / "m_trait_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "canonical_observation_id": ["O1", "O2"],
            "genotype_id": ["G1", "G2"],
            "environment_id": ["E1", "E1"],
            "trait_name_canonical": ["DAYS_TO_HEADING", "DAYS_TO_MATURITY"],
            "phenotype_value": [1.0, 2.0],
            "split": ["val", "val"],
        }
    ).to_parquet(run_dir / "m_predictions.parquet", index=False)


def test_inner_loss_screen_selects_only_matched_validation_winner(
    tmp_path: Path, monkeypatch
) -> None:
    protocol_path = loss_protocol(tmp_path / "loss.json")
    import hashlib

    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    models = tmp_path / "models"
    for inner in range(3):
        write_screen_run(
            models,
            candidate="current_trait_row_balanced",
            inner_fold=inner,
            normalized_rmse=0.8,
            pearson=0.5,
            prediction_sd_ratio=0.8,
            protocol_sha=protocol_sha,
        )
        write_screen_run(
            models,
            candidate="trait_environment_balanced",
            inner_fold=inner,
            normalized_rmse=0.76,
            pearson=0.51,
            prediction_sd_ratio=0.9,
            protocol_sha=protocol_sha,
        )
        write_screen_run(
            models,
            candidate="damped_environment_genotype_balanced",
            inner_fold=inner,
            normalized_rmse=0.82,
            pearson=0.49,
            prediction_sd_ratio=0.75,
            protocol_sha=protocol_sha,
        )
    trainer = tmp_path / "trainer.py"
    trainer.write_text("# test\n")
    out = tmp_path / "summary"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize",
            "--models-dir",
            str(models),
            "--loss-balance-protocol",
            str(protocol_path),
            "--phase",
            "phase_1",
            "--trainer",
            str(trainer),
            "--out-dir",
            str(out),
        ],
    )
    summarize_main()
    provenance = json.loads(
        (out / "loss_balance_inner_screen_provenance.json").read_text()
    )
    assert provenance["outer_test_metrics_read"] is False
    assert provenance["selected_candidate"] == "trait_environment_balanced"
    decision = pd.read_csv(out / "loss_balance_inner_screen_decision.tsv", sep="\t")
    selected = decision[decision["decision"].eq("advance_to_full_inner_confirmation")]
    assert selected["candidate"].tolist() == ["trait_environment_balanced"]
