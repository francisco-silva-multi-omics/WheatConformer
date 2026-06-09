from __future__ import annotations

import pandas as pd

from server_training_pipeline.split_utils import canonical_split_mode, make_split, split_leakage_record


def observations():
    return pd.DataFrame([{"panel_sample_id": f"g{g}", "env_kernel_id": f"e{e}"} for g in range(6) for e in range(6)])


def test_tensorflow_aliases_resolve_to_canonical_names():
    assert canonical_split_mode("loeo") == "gho_environment"
    assert canonical_split_mode("random") == "cv2_random_observation"
    assert canonical_split_mode("cv1_genotype") == "cv1_genotype"


def test_shared_splits_have_required_zero_overlap():
    obs = observations()
    for mode, col, axis in [
        ("gho_environment", "env_kernel_id", "env_overlap_train_test"),
        ("cv1_genotype", "panel_sample_id", "geno_overlap_train_test"),
        ("cv0_genotype_environment", None, "geno_overlap_train_test"),
    ]:
        train, val, test = make_split(obs, mode, 2026, 0.2, 0.2, col)
        qc = split_leakage_record(obs, 0, mode, train, val, test, col)
        assert qc[axis] == 0
        if mode == "cv0_genotype_environment":
            assert qc["env_overlap_train_test"] == 0
