from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


tf = pytest.importorskip("tensorflow")

from server_training_pipeline.train_multikernel_gxe_tf import persist_and_validate_split_leakage


def leakage(status: str, **overlaps: int) -> dict[str, object]:
    return {"repeat": 0, "split_mode": "gho_environment", "leakage_status": status, **overlaps}


def test_valid_split_writes_qc_files(tmp_path: Path) -> None:
    record = leakage("pass", env_overlap_train_test=0)
    persist_and_validate_split_leakage(record, tmp_path, "toy", "gho_environment", "gho_environment")

    assert pd.read_csv(tmp_path / "toy_split_leakage_qc.tsv", sep="\t").iloc[0]["leakage_status"] == "pass"
    assert json.loads((tmp_path / "toy_split_leakage_qc.json").read_text())["leakage_status"] == "pass"


@pytest.mark.parametrize(
    ("requested", "canonical", "record"),
    [
        ("gho_environment", "gho_environment", leakage("fail", env_overlap_train_test=1)),
        ("cv1_genotype", "cv1_genotype", leakage("fail", geno_overlap_train_test=1)),
        (
            "cv0_genotype_environment",
            "cv0_genotype_environment",
            leakage("fail", geno_overlap_train_test=1, env_overlap_train_test=1),
        ),
    ],
)
def test_leakage_failure_aborts_before_training(
    tmp_path: Path, requested: str, canonical: str, record: dict[str, object]
) -> None:
    with pytest.raises(SystemExit, match="Split leakage detected"):
        persist_and_validate_split_leakage(record, tmp_path, "toy", requested, canonical)
    assert (tmp_path / "toy_split_leakage_qc.tsv").exists()
