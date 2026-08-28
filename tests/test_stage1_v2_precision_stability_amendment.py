from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.v2.audit_stage1_v2_precision_stability import (
    ZERO_CLASS,
    compare_manifests,
    effective_zero_tolerance,
    manifest_path_is_identical,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / (
    "server_training_pipeline/stage1_v2_precision_stability_amendment_protocol_v1.json"
)


def test_effective_zero_tolerance_is_scale_aware_and_floored() -> None:
    tolerance = effective_zero_tolerance(
        [0.0, 2.0],
        absolute_floor=1e-12,
        epsilon=2.220446049250313e-16,
        epsilon_multiplier=1024,
    )
    assert tolerance == pytest.approx(1e-12)
    scaled = effective_zero_tolerance(
        [1_000_000.0],
        absolute_floor=1e-12,
        epsilon=2.220446049250313e-16,
        epsilon_multiplier=1024,
    )
    assert scaled > tolerance


def test_manifest_comparison_detects_byte_identity_and_changes() -> None:
    before = pd.DataFrame(
        [{"path": "a", "bytes": 3, "sha256": "x", "snapshot_stage": "before"}]
    )
    after = pd.DataFrame(
        [{"path": "a", "bytes": 3, "sha256": "x", "snapshot_stage": "after"}]
    )
    assert compare_manifests(before, after)["status"].tolist() == ["BYTE_IDENTICAL"]
    assert manifest_path_is_identical(compare_manifests(before, after), "a")
    assert not manifest_path_is_identical(compare_manifests(before, after), "missing")
    after.loc[0, "sha256"] = "y"
    changed = compare_manifests(before, after)
    assert changed["status"].tolist() == ["SHA256_CHANGED"]
    assert not manifest_path_is_identical(changed, "a")


def test_protocol_declares_additive_non_model_amendment() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert ZERO_CLASS in protocol["terminal_precision_classes"]
    assert protocol["amendment_rule"][ZERO_CLASS].endswith(
        "missing_for_every_group_row"
    )
    assert protocol["authoritative_model_weight_field"] == "reliability_weight"
    assert protocol["phase6_bound_weight_field"] == "authoritative_weight"
    assert protocol["model_inputs_rebuilt"] is False
    assert protocol["model_results_rebuilt"] is False
    assert protocol["outer_test_metrics_read"] is False
    assert protocol["final_holdout_outcomes_read"] is False
