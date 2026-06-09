from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from server_training_pipeline.run_validation_ablation_suite import (
    canonical_split_mode,
    group_kfold_splits,
    make_split,
    split_leakage_record,
)


def observations() -> pd.DataFrame:
    rows = []
    for genotype in [f"g{i}" for i in range(8)]:
        for environment in [f"e{i}" for i in range(8)]:
            rows.append(
                {
                    "panel_sample_id": genotype,
                    "env_kernel_id": environment,
                    "cycle": f"c{int(environment[1:]) % 4}",
                    "trial_name": environment,
                    "country": f"country{int(environment[1:]) % 3}",
                }
            )
    return pd.DataFrame(rows)


def test_legacy_alias_records_canonical_mode() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert canonical_split_mode("loeo", warn=True) == "gho_environment"
    assert caught


def test_grouped_holdout_has_no_environment_leakage() -> None:
    obs = observations()
    train, val, test = make_split(obs, "gho_environment", 2026, 0.2, 0.2, "env_kernel_id")
    qc = split_leakage_record(obs, 0, "gho_environment", train, val, test)
    assert qc["leakage_status"] == "pass"
    assert qc["env_overlap_train_test"] == 0


def test_cv1_genotype_has_zero_genotype_overlap() -> None:
    obs = observations()
    train, val, test = make_split(obs, "cv1_genotype", 2026, 0.2, 0.2, "panel_sample_id")
    qc = split_leakage_record(obs, 0, "cv1_genotype", train, val, test)
    assert qc["geno_overlap_train_test"] == 0
    assert qc["leakage_status"] == "pass"


def test_cv1_environment_has_zero_environment_overlap() -> None:
    obs = observations()
    train, val, test = make_split(obs, "cv1_environment", 2026, 0.2, 0.2, "env_kernel_id")
    qc = split_leakage_record(obs, 0, "cv1_environment", train, val, test)
    assert qc["env_overlap_train_test"] == 0
    assert qc["leakage_status"] == "pass"


def test_cv0_has_zero_overlap_on_both_axes() -> None:
    obs = observations()
    train, val, test = make_split(obs, "cv0_genotype_environment", 2026, 0.25, 0.25, None)
    qc = split_leakage_record(obs, 0, "cv0_genotype_environment", train, val, test)
    assert qc["geno_overlap_train_test"] == 0
    assert qc["env_overlap_train_test"] == 0
    assert qc["train_rows"] > 0 and qc["val_rows"] > 0 and qc["test_rows"] > 0
    assert qc["leakage_status"] == "pass"


def test_group_kfold_holds_every_group_out_once() -> None:
    obs = observations()
    folds = group_kfold_splits(obs, "env_kernel_id", 4, 2026, 0.2)
    held_out = []
    for repeat, (train, val, test) in enumerate(folds):
        qc = split_leakage_record(obs, repeat, "group_kfold", train, val, test, group_col="env_kernel_id")
        assert qc["env_overlap_train_test"] == 0
        assert qc["expected_env_overlap"] == "zero"
        held_out.extend(obs.iloc[test]["env_kernel_id"].unique())
    assert sorted(held_out) == sorted(obs["env_kernel_id"].unique())
