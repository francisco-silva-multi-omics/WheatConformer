from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from server_training_pipeline.compare_stage1_recovery_nested_cv import (
    main as compare_nested,
)
from server_training_pipeline.prepare_stage1_recovery_nested_evaluation import (
    main as prepare_nested,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_recovery_nested_protocol_preserves_architecture_and_assignments(
    tmp_path: Path, monkeypatch
) -> None:
    identifier_columns = {
        "panel_sample_id": "g1",
        "genotype_id": "g1",
        "env_kernel_id": "e1",
        "cycle": "2020",
        "country": "MEXICO",
        "trait_name_canonical": "GRAIN_YIELD",
    }
    base_ledger = tmp_path / "base_ledger.tsv"
    pd.DataFrame(
        [{"canonical_observation_id": "base", **identifier_columns}]
    ).to_csv(base_ledger, sep="\t", index=False)
    ledger = tmp_path / "ledger.tsv"
    pd.DataFrame(
        [
            {"canonical_observation_id": "base", **identifier_columns},
            {"canonical_observation_id": "new1", **identifier_columns},
            {"canonical_observation_id": "new2", **identifier_columns},
        ]
    ).to_csv(ledger, sep="\t", index=False)
    recovery_validation = tmp_path / "validation.json"
    required_checks = {
        "registry_accepts_exactly_all_P3_rows": True,
        "model_ids_equal_retained_environment_and_weight_recovery": True,
        "ledger_ids_equal_model_ids": True,
        "uniform_ledger_weights_equal_one": True,
        "nested_split_genotype_id_matches_certified_kernel_id": True,
        "P3_source_weights_preserved_as_invalid_in_ledger": True,
        "phenotype_values_unread": True,
        "outer_test_metrics_unread": True,
        "final_holdout_outcomes_unread": True,
    }
    recovery_validation.write_text(
        json.dumps(
            {
                "status": "PASS",
                "phenotype_values_read": False,
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
                "checks": required_checks,
                "counts": {
                    "recovered_environment_rows": 0,
                    "recovered_weight_rows": 2,
                },
                "inputs": {"multitrait_ledger": {"sha256": sha256(ledger)}},
            }
        ),
        encoding="utf-8",
    )
    base_evaluation = Path("server_training_pipeline/final_evaluation_protocol.json")
    base_outer = Path(
        "server_training_pipeline/reaction_norm_outer_evaluation_protocol_v3.json"
    )
    outer = json.loads(base_outer.read_text(encoding="utf-8"))
    selection_lock = tmp_path / "selection.json"
    selection_lock.write_text(
        json.dumps(
            {
                "status": "PASS",
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
                "selected_candidate": outer["selected_candidate"],
                "selected_configuration": outer["selected_configuration"],
            }
        ),
        encoding="utf-8",
    )
    environment_lock = tmp_path / "environment_selection.json"
    environment_lock.write_text(
        json.dumps(
            {
                "status": "PASS",
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
                "selected_environment_architecture": outer[
                    "selected_environment_architecture"
                ],
            }
        ),
        encoding="utf-8",
    )
    holdout = tmp_path / "holdout.tsv"
    holdout.write_text("env_id\ne1\n", encoding="utf-8")
    base_contract = tmp_path / "base_contract.json"
    base_contract.write_text(
        json.dumps(
            {
                "status": "frozen",
                "protocol_sha256": sha256(base_evaluation),
                "final_holdout_environment_ids_sha256": sha256(holdout),
            }
        ),
        encoding="utf-8",
    )
    selection = json.loads(selection_lock.read_text())
    selection["outer_evaluation_protocol_sha256"] = sha256(base_outer)
    selection_lock.write_text(json.dumps(selection), encoding="utf-8")
    environment_selection = json.loads(environment_lock.read_text())
    environment_selection["outer_evaluation_protocol_sha256"] = sha256(base_outer)
    environment_lock.write_text(json.dumps(environment_selection), encoding="utf-8")
    out_dir = tmp_path / "freeze"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_stage1_recovery_nested_evaluation",
            "--root",
            str(tmp_path),
            "--recovery-ledger",
            str(ledger),
            "--base-ledger",
            str(base_ledger),
            "--recovery-validation",
            str(recovery_validation),
            "--base-evaluation-protocol",
            str(base_evaluation.resolve()),
            "--base-evaluation-contract",
            str(base_contract),
            "--base-outer-protocol",
            str(base_outer.resolve()),
            "--base-selection-lock",
            str(selection_lock),
            "--base-environment-selection-lock",
            str(environment_lock),
            "--frozen-final-holdout-environments",
            str(holdout),
            "--out-dir",
            str(out_dir),
        ],
    )
    prepare_nested()
    evaluation = json.loads(
        (out_dir / "stage1_recovery_nested_evaluation_protocol.json").read_text()
    )
    recovery_outer = json.loads(
        (out_dir / "stage1_recovery_reaction_norm_outer_protocol.json").read_text()
    )
    freeze = json.loads(
        (out_dir / "stage1_recovery_nested_freeze.json").read_text()
    )
    assert freeze["status"] == "PASS"
    assert evaluation["scenario_assignment_id"] == "multitrait_quantitative_final_v4"
    assert evaluation["final_holdout_assignment_id"] == "multitrait_quantitative_final_v4"
    assert recovery_outer["selected_candidate"] == outer["selected_candidate"]
    assert recovery_outer["selected_configuration"] == outer["selected_configuration"]
    assert recovery_outer["outer_test_metrics_read_at_freeze"] is False


def write_ensemble(
    root: Path, *, model_label: str, rows: list[dict[str, object]]
) -> None:
    run_dir = root / "final_nested_reaction_norm_unseen_genotypes_outer0"
    run_dir.mkdir(parents=True)
    prefix = run_dir.name
    metadata = {
        "evaluation_stage": "outer_evaluation",
        "model_label": model_label,
        "external_split": {
            "scenario": "unseen_genotypes",
            "outer_fold": 0,
            "inner_fold": "ensemble",
        },
    }
    (run_dir / f"{prefix}_run_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(
        run_dir / f"{prefix}_predictions.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )


def test_compare_recovery_nested_reports_common_support_and_new_coverage(
    tmp_path: Path, monkeypatch
) -> None:
    common = [
        {
            "split": "test",
            "canonical_observation_id": "o1",
            "phenotype_value": 1.0,
            "trait_name_canonical": "GRAIN_YIELD",
            "panel_sample_id": "g1",
            "env_kernel_id": "e1",
            "y_pred": 0.7,
            "y_pred_train_mean": 1.5,
        },
        {
            "split": "test",
            "canonical_observation_id": "o2",
            "phenotype_value": 2.0,
            "trait_name_canonical": "GRAIN_YIELD",
            "panel_sample_id": "g2",
            "env_kernel_id": "e1",
            "y_pred": 2.3,
            "y_pred_train_mean": 1.5,
        },
    ]
    recovery = [dict(row) for row in common]
    recovery[0]["y_pred"] = 0.9
    recovery[1]["y_pred"] = 2.1
    recovery.append(
        {
            "split": "test",
            "canonical_observation_id": "new",
            "phenotype_value": 1.5,
            "trait_name_canonical": "GRAIN_YIELD",
            "panel_sample_id": "g3",
            "env_kernel_id": "e2",
            "y_pred": 1.4,
            "y_pred_train_mean": 1.5,
        }
    )
    baseline_dir = tmp_path / "baseline"
    recovery_dir = tmp_path / "recovery"
    write_ensemble(baseline_dir, model_label="baseline", rows=common)
    write_ensemble(recovery_dir, model_label="recovery", rows=recovery)
    baseline_contract = tmp_path / "baseline_contract.json"
    recovery_contract = tmp_path / "recovery_contract.json"
    contract = {
        "scenario_assignment_id": "same",
        "final_holdout_assignment_id": "same",
    }
    baseline_contract.write_text(json.dumps(contract), encoding="utf-8")
    recovery_contract.write_text(json.dumps(contract), encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps({"status": "PASS", "outer_test_metrics_read": False}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "comparison"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_stage1_recovery_nested_cv",
            "--baseline-models-dir",
            str(baseline_dir),
            "--recovery-models-dir",
            str(recovery_dir),
            "--baseline-contract",
            str(baseline_contract),
            "--recovery-contract",
            str(recovery_contract),
            "--recovery-freeze",
            str(freeze),
            "--out-dir",
            str(out_dir),
        ],
    )
    compare_nested()
    paired = pd.read_csv(
        out_dir / "stage1_recovery_nested_paired_metrics.tsv", sep="\t"
    )
    coverage = pd.read_csv(
        out_dir / "stage1_recovery_nested_coverage.tsv", sep="\t"
    )
    assert paired.loc[0, "normalized_rmse_gain"] > 0
    assert coverage.loc[0, "common_test_rows"] == 2
    assert coverage.loc[0, "recovery_only_test_rows"] == 1
