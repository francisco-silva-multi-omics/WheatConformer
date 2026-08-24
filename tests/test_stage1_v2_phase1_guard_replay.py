from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.v2.run_stage1_v2_phase6_phase1 import pair_guard_metrics


ROOT = Path(__file__).resolve().parents[1]


def test_guard_replay_is_isolated_and_preregistered() -> None:
    runner = (ROOT / "scripts/v2/run_stage1_v2_phase6_phase1.py").read_text()
    freezer = (
        ROOT / "scripts/v2/freeze_stage1_v2_phase6_phase1_guard_replay.py"
    ).read_text()
    launcher = (
        ROOT / "scripts/v2/run_stage1_v2_phase6_phase1_guard_replay_server_cpu.sh"
    ).read_text()
    assert "stage1_v2_phase6_phase1_guard_replay_v1_runs" in runner
    assert "strict_candidate_reference_mask_pairing" in freezer
    assert "parent_full_metrics_replayed_with_maximum_absolute_delta" in freezer
    assert "STAGE1_V2_PHASE1_GUARD_REPLAY=1" in launcher
    assert "nohup setsid" in launcher


def test_guard_replay_writes_and_checks_component_signatures() -> None:
    trainer = (ROOT / "server_training_pipeline/train_stage1_v2_phase6_tf.py").read_text()
    runner = (ROOT / "scripts/v2/run_stage1_v2_phase6_phase1.py").read_text()
    assert "observation_id_signature" in trainer
    assert "h_seeds_direct_marker_support_included" in trainer
    assert "projection_core_mask_candidate_independent" in trainer
    assert "observation_id_signature_reference" in runner
    assert "matched_component_mask_status" in runner


def test_guard_pairing_rejects_different_observation_signatures() -> None:
    rows = []
    for candidate, signature in (
        ("ka_identity_location_baseline", "reference-signature"),
        ("candidate", "candidate-signature"),
    ):
        rows.append(
            {
                "state_id": "state",
                "candidate": candidate,
                "configuration_label": "configuration",
                "mask_candidate": "candidate",
                "subset": "MARKER_SUPPORTED",
                "rows": 10,
                "observation_id_signature": signature,
                "normalized_rmse_macro": 1.0,
                "pearson_macro": 0.5,
            }
        )
    with pytest.raises(ValueError, match="not exactly paired"):
        pair_guard_metrics(pd.DataFrame(rows), "ka_identity_location_baseline")
