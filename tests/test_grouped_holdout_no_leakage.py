from __future__ import annotations

import numpy as np
import pandas as pd

from server_training_pipeline.run_validation_ablation_suite import make_split, split_leakage_record


def test_grouped_holdout_has_no_group_leakage() -> None:
    observations = pd.DataFrame({"env_kernel_id": np.repeat(["e1", "e2", "e3", "e4", "e5"], 3)})
    train, val, test = make_split(observations, "loeo", 2026, 0.2, 0.2, "env_kernel_id")
    qc = split_leakage_record(observations, 0, "loeo", "env_kernel_id", train, val, test)
    assert qc["leakage_status"] == "pass"
    assert qc["train_test_group_overlap"] == 0
    assert qc["train_val_group_overlap"] == 0
    assert qc["val_test_group_overlap"] == 0
