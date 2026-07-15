from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_multitrait_trainer_is_joint_and_requires_certification() -> None:
    path = ROOT / "server_training_pipeline" / "train_multitrait_multikernel_tf.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)

    assert "select_single_trait" not in source
    assert "--certification-summary" in source
    assert "trait_kernel_gate_logits" in source
    assert "--kernel-registry" in source
    assert "K_G_HMP" not in source  # Experts are data-driven through the registry.
    assert "genomic_coverage_group" in source
    assert "eligible_traits" in source
    assert "trait_balanced_loss_weights" in source
    assert "train_nystrom" in source
    assert "MultiTraitKernelExperts" in source
    assert "initialization_seed" in source
    assert "seed=self.initialization_seed + self._initializer_index" in source


def test_multitrait_ledger_uses_canonical_ids_and_explicit_compact_mapping() -> None:
    path = ROOT / "server_training_pipeline" / "build_multitrait_ledger.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)

    assert "canonical_observation_id" in source
    assert "source_kernel_index" in source
    assert "compact_kernel_index" in source
    assert "stabilize_precision_weights" in source


def test_multitrait_ledger_maps_source_indices_and_writes_lineage(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "ledger"
    model_dir.mkdir()
    prefix = "toy"
    observations = pd.DataFrame(
        {
            "canonical_observation_id": ["o1", "o2", "o3", "o4", "o5"],
            "trait_name_canonical": ["A", "A", "B", "B", "A"],
            "phenotype_value": [1.0, 2.0, 3.0, 4.0, np.inf],
            "var_g_e": [0.1, 0.2, 0.3, 0.4, 0.2],
            "weight_g_e": [10.0, 5.0, 10 / 3, 2.5, 5.0],
            "geno_kernel_index": [10, 20, 10, 20, 10],
            "env_kernel_index": [5, 5, 8, 8, 5],
        }
    )
    observations.to_csv(
        model_dir / f"{prefix}_model_ready_stage1_observations.tsv.gz",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        {
            "sample_id": ["g1", "g2"],
            "source_kernel_index": [10, 20],
            "compact_kernel_index": [0, 1],
        }
    ).to_csv(model_dir / f"{prefix}_K_G_unique_order.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "env_id": ["e1", "e2"],
            "source_kernel_index": [5, 8],
            "compact_kernel_index": [0, 1],
        }
    ).to_csv(model_dir / f"{prefix}_K_E_unique_order.tsv", sep="\t", index=False)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "server_training_pipeline.build_multitrait_ledger",
            "--root",
            str(tmp_path),
            "--model-dir",
            str(model_dir),
            "--prefix",
            prefix,
            "--out-dir",
            str(out_dir),
            "--out-prefix",
            "toy_multitrait",
            "--min-trait-rows",
            "1",
            "--write-tsv",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    ledger = pd.read_csv(out_dir / "toy_multitrait_observations.tsv.gz", sep="\t")
    assert ledger["geno_compact_index"].tolist() == [0, 1, 0, 1]
    assert ledger["env_compact_index"].tolist() == [0, 0, 1, 1]
    np.testing.assert_allclose(
        ledger.groupby("trait_name_canonical")["weight_g_e"].mean().to_numpy(),
        np.ones(2),
    )
    lineage = json.loads((out_dir / "toy_multitrait_lineage.json").read_text(encoding="utf-8"))
    assert lineage["output_rows"] == 4
    summary = pd.read_csv(out_dir / "toy_multitrait_ledger_summary.tsv", sep="\t")
    summary_values = dict(zip(summary["metric"], summary["value"]))
    assert int(summary_values["removed_nonfinite_phenotype_rows"]) == 1
