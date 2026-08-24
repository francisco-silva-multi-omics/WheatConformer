import json
import sys

import numpy as np
import pandas as pd

from server_training_pipeline.audit_fold_expert_support import (
    diagnose_fold_support,
    main,
)
from server_training_pipeline.final_evaluation_contract import file_sha256


def test_diagnoses_inner_training_concentration() -> None:
    diagnosis = diagnose_fold_support(
        order_dimension=100,
        overall_exact_ids=80,
        active_exact_ids=70,
        train_exact_ids=1,
        train_panel_ids=1,
        train_normalized_ids=1,
        positive_eigenvalues=0,
    )

    assert diagnosis == "expert_support_concentrated_outside_inner_training_partition"


def test_diagnoses_ledger_id_field_mismatch_before_fold_concentration() -> None:
    diagnosis = diagnose_fold_support(
        order_dimension=100,
        overall_exact_ids=1,
        active_exact_ids=1,
        train_exact_ids=1,
        train_panel_ids=40,
        train_normalized_ids=1,
        positive_eigenvalues=0,
    )

    assert diagnosis == "ledger_genotype_id_disagrees_with_panel_sample_id"


def test_diagnoses_numerically_degenerate_training_subkernel() -> None:
    diagnosis = diagnose_fold_support(
        order_dimension=100,
        overall_exact_ids=80,
        active_exact_ids=70,
        train_exact_ids=50,
        train_panel_ids=50,
        train_normalized_ids=50,
        positive_eigenvalues=0,
    )

    assert diagnosis == "training_subkernel_is_numerically_degenerate"


def test_healthy_support_requires_positive_centered_spectrum() -> None:
    diagnosis = diagnose_fold_support(
        order_dimension=100,
        overall_exact_ids=80,
        active_exact_ids=70,
        train_exact_ids=50,
        train_panel_ids=50,
        train_normalized_ids=50,
        positive_eigenvalues=49,
    )

    assert diagnosis == "healthy_fold_support_previous_failure_requires_artifact_identity_check"


def test_audit_reconstructs_one_id_inner_training_support(tmp_path, monkeypatch) -> None:
    rows = []
    index = 0
    for trait in ["A", "B"]:
        for genotype, environment in [("g1", "e_train"), ("g2", "e_val"), ("g3", "e_test")]:
            rows.append(
                {
                    "canonical_observation_id": f"obs{index}",
                    "trait_name_canonical": trait,
                    "genotype_id": genotype,
                    "panel_sample_id": genotype,
                    "environment_id": environment,
                    "env_kernel_id": environment,
                    "cycle": "2020",
                    "country": "X",
                }
            )
            index += 1
    ledger = pd.DataFrame(rows)
    ledger_path = tmp_path / "ledger.tsv"
    ledger.to_csv(ledger_path, sep="\t", index=False)

    manifest = pd.DataFrame(
        [
            {
                "scenario": "unseen_environments",
                "outer_fold": 0,
                "inner_fold": 0,
                "axis": "environment",
                "partition": "inner_validation",
                "entity_id": "e_val",
            },
            {
                "scenario": "unseen_environments",
                "outer_fold": 0,
                "inner_fold": 0,
                "axis": "environment",
                "partition": "outer_test",
                "entity_id": "e_test",
            },
        ]
    )
    manifest_path = tmp_path / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "status": "frozen",
                "entity_manifest_sha256": file_sha256(manifest_path),
                "ledger_sha256": file_sha256(ledger_path),
            }
        ),
        encoding="utf-8",
    )

    kernel_path = tmp_path / "K_G_HMP_LINEAR.npy"
    np.save(kernel_path, np.eye(3, dtype=np.float32))
    order_path = tmp_path / "K_G_HMP_LINEAR_order.tsv"
    pd.DataFrame(
        {"sample_id": ["g1", "g2", "g3"], "compact_kernel_index": [0, 1, 2]}
    ).to_csv(order_path, sep="\t", index=False)
    registry_path = tmp_path / "registry.tsv"
    pd.DataFrame(
        [
            {
                "kernel": "K_G_HMP_LINEAR",
                "axis": "genotype",
                "kernel_path": str(kernel_path),
                "order_path": str(order_path),
                "id_col": "sample_id",
                "eligible_traits": "*",
                "coverage_path": "",
            }
        ]
    ).to_csv(registry_path, sep="\t", index=False)
    certification_path = tmp_path / "certification.json"
    certification_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "registry_identity": {"sha256": file_sha256(registry_path)},
                "kernel_identities": {
                    "K_G_HMP_LINEAR": {"sha256": file_sha256(kernel_path)}
                },
                "order_identities": {
                    "K_G_HMP_LINEAR": {"sha256": file_sha256(order_path)}
                },
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_fold_expert_support",
            "--ledger",
            str(ledger_path),
            "--manifest",
            str(manifest_path),
            "--contract",
            str(contract_path),
            "--registry",
            str(registry_path),
            "--certification-summary",
            str(certification_path),
            "--scenario",
            "unseen_environments",
            "--outer-fold",
            "0",
            "--inner-fold",
            "0",
            "--min-train-rows-per-trait",
            "1",
            "--min-eval-rows-per-trait",
            "1",
            "--out-dir",
            str(out_dir),
        ],
    )

    main()

    summary = pd.read_csv(out_dir / "fold_expert_support_summary.tsv", sep="\t")
    report = json.loads((out_dir / "fold_expert_support_report.json").read_text())
    assert summary.loc[0, "train_exact_mapped_unique_ids"] == 1
    assert (
        summary.loc[0, "diagnosis"]
        == "expert_support_concentrated_outside_inner_training_partition"
    )
    assert report["status"] == "REVIEW"
