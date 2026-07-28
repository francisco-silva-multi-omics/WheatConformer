from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_stage1_recovery_readiness_classifies_full_order_support(
    tmp_path: Path,
) -> None:
    ledger = pd.DataFrame(
        {
            "canonical_observation_id": ["O1", "O2", "O3", "O4", "O5"],
            "canonical_germplasm_key": ["GID1", "GID2", "GID3", "GID4", "GID1"],
            "env_kernel_id": ["E1", "E1", "E2", "E3", "E1"],
            "trait_name_canonical": ["DAYS_TO_HEADING"] * 5,
            "n_plot_records": [2, 3, 4, 5, 1],
            "stage1_to_model_status": [
                "retained_in_stage1_model_observations",
                "genotype_not_in_stage1_model_order",
                "environment_not_in_stage1_model_order",
                "genotype_not_in_stage1_model_order",
                "invalid_or_nonpositive_stage1_weight",
            ],
        }
    )
    ledger.to_parquet(tmp_path / "attrition.parquet", index=False)
    pd.DataFrame({"sample_id": ["GID1", "GID3"]}).to_csv(
        tmp_path / "legacy.tsv", sep="\t", index=False
    )
    pd.DataFrame({"sample_id": ["GID1", "GID2", "GID3"]}).to_csv(
        tmp_path / "v3.tsv", sep="\t", index=False
    )
    pd.DataFrame({"env_id": ["E1"]}).to_csv(
        tmp_path / "environment.tsv", sep="\t", index=False
    )

    out_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "audit.audit_stage1_recovery_readiness",
            "--root",
            str(tmp_path),
            "--attrition-ledger",
            "attrition.parquet",
            "--legacy-pedigree-order",
            "legacy.tsv",
            "--canonical-v3-order",
            "v3.tsv",
            "--global-environment-order",
            "environment.tsv",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = pd.read_csv(
        out_dir / "stage1_recovery_readiness_summary.tsv", sep="\t"
    ).set_index("recovery_readiness")
    assert summary.loc["RETAINED_REFERENCE", "stage1_rows"] == 1
    assert summary.loc["P1_CANONICAL_V3_MODEL_INPUT_REBUILD", "stage1_rows"] == 1
    assert summary.loc["P1_RECOVER_ENVIRONMENT", "stage1_rows"] == 1
    assert summary.loc["P1_RECOVER_PEDIGREE_AND_ENVIRONMENT", "stage1_rows"] == 1
    assert summary.loc["P3_REPAIR_WEIGHT_METADATA", "stage1_rows"] == 1

    provenance = json.loads(
        (out_dir / "stage1_recovery_readiness_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["status"] == "PASS"
    assert provenance["phenotype_values_read"] is False
    assert provenance["outer_test_metrics_read"] is False
    assert provenance["kernels_modified"] is False
