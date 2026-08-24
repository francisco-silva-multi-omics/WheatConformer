from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from server_training_pipeline.final_evaluation_contract import file_sha256
from server_training_pipeline.summarize_nested_evaluation import run_record


TRAINER = Path("server_training_pipeline/train_multitrait_multikernel_tf.py")
FACTORIZATION = Path("server_training_pipeline/kernel_factorization.py")


def write_run(tmp_path: Path, effective_mode: str = "train_nystrom") -> Path:
    run_dir = tmp_path / "final_nested_unseen_environments_outer0_full"
    run_dir.mkdir()
    metadata = {
        "evaluation_stage": "outer_evaluation",
        "model_label": "model",
        "trainer_sha256": file_sha256(TRAINER),
        "kernel_factorization_sha256": file_sha256(FACTORIZATION),
        "canonical_split_mode": "gho_environment",
        "requested_factorization_mode": "train_nystrom",
        "effective_factorization_mode": effective_mode,
        "factorizations": {
            "K_A": {"factorization_mode": effective_mode},
            "K_E": {"factorization_mode": effective_mode},
        },
        "external_split": {
            "scenario": "unseen_environments",
            "outer_fold": 0,
            "inner_fold": "ensemble",
        },
        "ensemble": {"member_count": 3},
    }
    (run_dir / "run_run_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "canonical_observation_id": "val",
                "split": "val",
                "phenotype_value": 1.0,
                "trait_name_canonical": "TRAIT",
                "panel_sample_id": "g1",
                "env_kernel_id": "e1",
                "y_pred": 1.0,
                "y_pred_train_mean": 0.0,
            },
            {
                "canonical_observation_id": "test",
                "split": "test",
                "phenotype_value": 2.0,
                "trait_name_canonical": "TRAIT",
                "panel_sample_id": "g2",
                "env_kernel_id": "e2",
                "y_pred": 1.5,
                "y_pred_train_mean": 0.0,
            },
        ]
    ).to_csv(run_dir / "run_predictions.tsv.gz", sep="\t", index=False)
    return run_dir


def test_legacy_missing_member_columns_are_filled_per_run(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    record = run_record(
        run_dir, file_sha256(TRAINER), file_sha256(FACTORIZATION)
    )
    assert record is not None
    predictions = record[0].set_index("split")
    assert predictions.loc["val", "ensemble_member_count"] == 1
    assert predictions.loc["test", "ensemble_member_count"] == 3
    assert predictions.loc["test", "ensemble_expected_member_count"] == 3


def test_summarizer_rejects_transductive_ensemble(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path, effective_mode="full_transductive")
    with pytest.raises(ValueError, match="INVALID_TRANSDUCTIVE"):
        run_record(run_dir, file_sha256(TRAINER), file_sha256(FACTORIZATION))
