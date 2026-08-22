from __future__ import annotations

import ast
from pathlib import Path


def test_daily_normalization_certifier_does_not_import_model_or_outcome_data() -> None:
    path = Path("server_training_pipeline/certify_phase6a_daily_normalization.py")
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert "pd.read_parquet" not in source
    assert "outer_test_metrics.tsv" not in source
    assert "final_holdout_outcomes." not in source
