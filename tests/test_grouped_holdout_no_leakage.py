from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from server_training_pipeline.run_validation_ablation_suite import (
    build_leakage_summary,
    canonical_split_mode,
    fold_skip_reason,
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
    assert qc["env_overlap_train_val"] == 0
    assert qc["env_overlap_val_test"] == 0


def test_cv1_genotype_has_zero_genotype_overlap() -> None:
    obs = observations()
    train, val, test = make_split(obs, "cv1_genotype", 2026, 0.2, 0.2, "panel_sample_id")
    qc = split_leakage_record(obs, 0, "cv1_genotype", train, val, test)
    assert qc["geno_overlap_train_test"] == 0
    assert qc["geno_overlap_train_val"] == 0
    assert qc["geno_overlap_val_test"] == 0
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
    geno = obs["panel_sample_id"]
    env = obs["env_kernel_id"]
    assert not (set(geno.iloc[train]) & set(geno.iloc[val]))
    assert not (set(geno.iloc[val]) & set(geno.iloc[test]))
    assert not (set(env.iloc[train]) & set(env.iloc[val]))
    assert not (set(env.iloc[val]) & set(env.iloc[test]))


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


def test_leakage_failed_fold_is_skipped_before_metrics() -> None:
    reason = fold_skip_reason(
        {"leakage_status": "fail"},
        np.array([0, 1]),
        np.array([2]),
        np.array([3]),
    )
    assert reason == "split leakage detected"


def test_empty_validation_fold_is_skipped() -> None:
    reason = fold_skip_reason(
        {"leakage_status": "pass"},
        np.array([0, 1]),
        np.array([], dtype=int),
        np.array([3]),
    )
    assert reason == "empty train/validation/test partition"


def test_leakage_summary_reports_passed_failed_and_empty_skips() -> None:
    summary = build_leakage_summary(
        pd.DataFrame(
            [
                {"repeat": 0, "split_mode": "cv1_genotype", "leakage_status": "pass", "note": ""},
                {"repeat": 1, "split_mode": "cv1_genotype", "leakage_status": "fail", "note": "split leakage detected"},
                {
                    "repeat": 2,
                    "split_mode": "cv1_genotype",
                    "leakage_status": "skipped",
                    "note": "empty train/validation/test partition",
                },
            ]
        )
    ).iloc[0]
    assert summary["repeats_attempted"] == 3
    assert summary["repeats_passed"] == 1
    assert summary["repeats_failed"] == 1
    assert summary["repeats_skipped"] == 1
    assert summary["repeats_skipped_empty"] == 1
