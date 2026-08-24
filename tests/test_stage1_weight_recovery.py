from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from audit.audit_stage1_weight_recovery import main as audit_weight_recovery
from audit.validate_stage1_weight_recovery import main as validate_weight_recovery


def test_weight_recovery_audit_accepts_fold_local_handling(
    tmp_path: Path, monkeypatch
) -> None:
    readiness = pd.DataFrame(
        {
            "canonical_observation_id": ["o1", "o2"],
            "recovery_readiness": [
                "P3_REPAIR_WEIGHT_METADATA",
                "P3_REPAIR_WEIGHT_METADATA",
            ],
        }
    )
    stage1 = pd.DataFrame(
        {
            "canonical_observation_id": ["o1", "o2"],
            "canonical_germplasm_key": ["g1", "g2"],
            "env_kernel_id": ["e1", "source   e2"],
            "trait_name_canonical": ["GRAIN_YIELD", "DAYS_TO_HEADING"],
            "stage1_model_status": ["linear_model_adjusted"] * 2,
            "weight_g_e": [np.nan, 0.0],
            "var_g_e": [2.0, np.nan],
        }
    )
    aliases = pd.DataFrame(
        {
            "source_env_id": ["source e2"],
            "target_env_id": ["e2"],
            "mapping_status": ["ACCEPTED_ALIAS"],
        }
    )
    readiness_path = tmp_path / "readiness.tsv"
    stage1_path = tmp_path / "stage1.tsv"
    aliases_path = tmp_path / "aliases.tsv"
    genotype_order = tmp_path / "genotypes.tsv"
    environment_order = tmp_path / "environments.tsv"
    out_dir = tmp_path / "audit"
    readiness.to_csv(readiness_path, sep="\t", index=False)
    stage1.to_csv(stage1_path, sep="\t", index=False)
    aliases.to_csv(aliases_path, sep="\t", index=False)
    pd.DataFrame({"sample_id": ["g1", "g2"]}).to_csv(
        genotype_order, sep="\t", index=False
    )
    pd.DataFrame({"env_id": ["e1", "e2"]}).to_csv(
        environment_order, sep="\t", index=False
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_stage1_weight_recovery",
            "--root",
            str(tmp_path),
            "--readiness-ledger",
            str(readiness_path),
            "--stage1-phenotypes",
            str(stage1_path),
            "--alias-registry",
            str(aliases_path),
            "--genotype-order",
            str(genotype_order),
            "--environment-order",
            str(environment_order),
            "--out-dir",
            str(out_dir),
        ],
    )
    audit_weight_recovery()

    registry = pd.read_csv(
        out_dir / "stage1_weight_recovery_registry.tsv", sep="\t"
    ).set_index("canonical_observation_id")
    assert registry["weight_recovery_decision"].eq(
        "ACCEPT_FOLD_LOCAL_WEIGHT_RECOVERY"
    ).all()
    assert (
        registry.loc["o1", "weight_recovery_method"]
        == "fold_local_training_transform_from_source_variance"
    )
    assert (
        registry.loc["o2", "weight_recovery_method"]
        == "fold_local_training_variance_imputation"
    )
    provenance = json.loads(
        (out_dir / "stage1_weight_recovery_provenance.json").read_text()
    )
    assert provenance["status"] == "PASS"
    assert provenance["phenotype_values_read"] is False


def test_weight_recovery_validator_requires_exact_uniform_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    readiness = pd.DataFrame(
        {
            "canonical_observation_id": ["kept", "alias", "weight"],
            "recovery_readiness": [
                "RETAINED_REFERENCE",
                "P1_RECOVER_ENVIRONMENT",
                "P3_REPAIR_WEIGHT_METADATA",
            ],
        }
    )
    registry = pd.DataFrame(
        {
            "canonical_observation_id": ["weight"],
            "weight_recovery_decision": ["ACCEPT_FOLD_LOCAL_WEIGHT_RECOVERY"],
        }
    )
    model = pd.DataFrame(
        {
            "canonical_observation_id": ["kept", "alias", "weight"],
            "canonical_germplasm_key": ["g1", "g2", "g3"],
            "env_kernel_id": ["e1", "e2", "e3"],
            "geno_kernel_index": [0, 1, 2],
            "env_kernel_index": [0, 1, 2],
            "weight_g_e": [1.0, 1.0, np.nan],
            "var_g_e": [1.0, 1.0, np.nan],
        }
    )
    ledger = pd.DataFrame(
        {
            "canonical_observation_id": ["kept", "alias", "weight"],
            "weight_g_e": [1.0, 1.0, 1.0],
            "source_weight_g_e": [1.0, 1.0, np.nan],
            "raw_var_g_e": [1.0, 1.0, np.nan],
            "weight_variance_imputed": [False, False, True],
            "panel_sample_id": ["g1", "g2", "g3"],
            "genotype_id": ["g1", "g2", "g3"],
            "geno_source_kernel_index": [0, 1, 2],
            "env_source_kernel_index": [0, 1, 2],
        }
    )
    paths = {}
    for name, frame in {
        "readiness": readiness,
        "registry": registry,
        "model": model,
        "ledger": ledger,
    }.items():
        path = tmp_path / f"{name}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        paths[name] = path
    genotype_order = tmp_path / "genotypes.tsv"
    environment_order = tmp_path / "environments.tsv"
    pd.DataFrame({"sample_id": ["g1", "g2", "g3"]}).to_csv(
        genotype_order, sep="\t", index=False
    )
    pd.DataFrame({"env_id": ["e1", "e2", "e3"]}).to_csv(
        environment_order, sep="\t", index=False
    )
    out_dir = tmp_path / "validation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_stage1_weight_recovery",
            "--root",
            str(tmp_path),
            "--readiness-ledger",
            str(paths["readiness"]),
            "--weight-registry",
            str(paths["registry"]),
            "--model-observations",
            str(paths["model"]),
            "--multitrait-ledger",
            str(paths["ledger"]),
            "--genotype-order",
            str(genotype_order),
            "--environment-order",
            str(environment_order),
            "--out-dir",
            str(out_dir),
        ],
    )
    validate_weight_recovery()
    provenance = json.loads(
        (out_dir / "stage1_weight_recovery_model_validation.json").read_text()
    )
    assert provenance["status"] == "PASS"
    assert provenance["counts"]["recovered_weight_rows"] == 1
