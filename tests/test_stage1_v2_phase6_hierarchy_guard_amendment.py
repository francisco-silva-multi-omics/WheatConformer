from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.v2.certify_stage1_v2_phase6_hierarchy_guard_amendment import (
    MASK_CANDIDATE,
    normalized_guard_rows,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v2/certify_stage1_v2_phase6_hierarchy_guard_amendment.py"


def test_amendment_is_additive_inner_only_and_selection_preserving() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "existing_nested_inner_validation_metrics_only" in source
    assert '"scientific_selection_changed"' in source
    assert '"outer_evaluation_allowed": False' in source
    assert '"final_holdout_outcomes_read": False' in source
    assert "selected_candidate_all_guards_pass" in source


def test_candidate_owned_mask_is_normalized_to_frozen_mask(tmp_path: Path) -> None:
    candidate = "hierarchy_test_weight_identity_calibration_v1"
    rows = []
    for subset in (
        "PEDIGREE_ONLY",
        "MARKER_SUPPORTED",
        "PEDIGREE_AND_MARKER",
        "NEITHER_PRODUCTION_PEDIGREE_NOR_DENSE_MARKERS",
        "RECOVERED_IDENTITY_OR_COMPONENT",
        "PROJECTION_CORE_ACTIVE",
        "PROJECTION_CORE_INACTIVE_814_ENVIRONMENTS",
    ):
        rows.append(
            {
                "mask_candidate": candidate,
                "subset": subset,
                "rows": 1000,
                "unique_genotypes": 100,
                "unique_environments": 10,
                "trait_count": 7,
                "observation_id_signature": f"signature-{subset}",
                "normalized_rmse_macro": 0.8,
                "pearson_macro": 0.7,
            }
        )
    path = tmp_path / "guards.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    observed = normalized_guard_rows(
        path,
        state_id="STATE",
        scenario="GNEW_EOBS",
        candidate=candidate,
        own_mask=True,
    )
    assert len(observed) == 7
    assert observed["mask_candidate"].eq(MASK_CANDIDATE).all()
    assert observed["candidate"].eq(candidate).all()
    assert observed["state_id"].eq("STATE").all()
