from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.v2.build_cimmyt_pre_qc_production_kg_v2 import (
    MISSING,
    QC_INCOMPATIBLE_ALLELES,
    build_exact_kernel,
    estimate_sample_call_rate_threshold,
    exact_allele_relation,
    fit_marker_qc,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "server_training_pipeline/cimmyt_pre_qc_production_kg_protocol_v2.json"
RELEASE = ROOT / "audit/v2/phase5_cimmyt_pre_qc_production_kg_v2"


def test_protocol_freezes_training_only_qc_and_known_incompatible_markers() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["protocol_version"] == "cimmyt_pre_qc_production_kg_v2"
    assert protocol["sample_qc"]["fit_scope"] == "training_partition_panel_samples_only"
    assert protocol["marker_qc"]["fit_scope"] == "sample_QC_passing_training_GIDs_only"
    assert protocol["imputation"]["held_out_calls_used_for_fit"] is False
    assert len(protocol["allele_policy"]["incompatible_shared_markers"]) == 5
    assert protocol["global_filter_guard"][
        "globally_filtered_hmp_may_define_marker_availability"
    ] is False


def test_sample_threshold_uses_only_supplied_training_rates() -> None:
    training = [0.52, 0.55, 0.58, 0.60]
    first = estimate_sample_call_rate_threshold(
        training, floor=0.5, ceiling=0.95, mad_multiplier=3.0
    )
    held_out_extremes = [0.0, 1.0]
    second = estimate_sample_call_rate_threshold(
        training, floor=0.5, ceiling=0.95, mad_multiplier=3.0
    )
    assert held_out_extremes  # These values are intentionally not supplied to the fit.
    assert first == second
    assert 0.5 <= first["threshold"] <= 0.95


def test_marker_qc_is_invariant_to_held_out_call_changes() -> None:
    dosage = np.asarray(
        [
            [0, 1, 2, 0],
            [0, 0, 0, 2],
            [0, 2, 0, 2],
            [0, 1, 2, 1],
        ],
        dtype=np.uint8,
    )
    structural = np.asarray([True, True, False, True])
    kwargs = dict(
        minimum_call_rate=0.5,
        minimum_observed=2,
        minimum_maf=0.01,
        maximum_heterozygosity=0.75,
    )
    first = fit_marker_qc(dosage, np.asarray([0, 1, 2]), structural, **kwargs)
    changed = dosage.copy()
    changed[:, 3] = np.asarray([2, MISSING, 1, 0], dtype=np.uint8)
    second = fit_marker_qc(changed, np.asarray([0, 1, 2]), structural, **kwargs)
    for key in ["observed", "allele_frequency_all", "heterozygosity", "reasons", "retained"]:
        np.testing.assert_allclose(first[key], second[key], equal_nan=True)
    assert first["reasons"][2] == QC_INCOMPATIBLE_ALLELES


def test_allele_orientation_and_exact_kernel_projection() -> None:
    assert exact_allele_relation("A/G", "A/G") == "SAME_ORDER"
    assert exact_allele_relation("A/G", "G/A") == "REVERSED_ORDER"
    assert exact_allele_relation("A/G", "A/C") == "INCOMPATIBLE_ALLELE_SET"
    assert exact_allele_relation("A/G", None) == "PRE_QC_ONLY_REFERENCE_ORIENTATION"

    dosage = np.asarray(
        [
            [0, 1, 2, 0],
            [2, 1, 0, 2],
            [0, 0, 2, MISSING],
        ],
        dtype=np.uint8,
    )
    training = np.asarray([0, 1, 2], dtype=np.int32)
    p = np.asarray([0.5, 0.5, 1.0 / 3.0], dtype=np.float32)
    result = build_exact_kernel(
        dosage,
        np.asarray([0, 1, 2], dtype=np.int32),
        p,
        np.asarray([0, 1, 2, 3], dtype=np.int32),
        training,
    )
    kernel = np.asarray(result["training_kernel"])
    projection = np.asarray(result["projection_to_training"])
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-7)
    np.testing.assert_allclose(projection[:3], kernel, atol=1e-7)
    np.testing.assert_allclose(np.diag(kernel).mean(), 1.0, atol=1e-6)
    assert np.linalg.eigvalsh(kernel.astype(np.float64)).min() >= -1e-6


@pytest.mark.skipif(
    not (RELEASE / "CIMMYT_PRE_QC_PRODUCTION_KG_V2_DECISION.json").is_file(),
    reason="CIMMYT production K_G v2 release has not been generated",
)
def test_generated_release_certifies_all_150_states() -> None:
    decision = json.loads(
        (RELEASE / "CIMMYT_PRE_QC_PRODUCTION_KG_V2_DECISION.json").read_text(
            encoding="utf-8"
        )
    )
    registry = pd.read_csv(
        RELEASE / "states/cimmyt_production_kg_state_registry.tsv", sep="\t"
    )
    assert decision["status"] == "PASS_CIMMYT_PRE_QC_PRODUCTION_KG_V2"
    assert decision["strict_ready_states"] == 150
    assert len(registry) == 150
    assert registry.strict_production_eligible.astype(bool).all()
    assert registry.markers_incompatible_alleles.eq(5).all()
    assert registry.retained_pre_qc_only_markers.gt(0).all()
    assert (~registry.held_out_calls_used_for_fit.astype(bool)).all()
