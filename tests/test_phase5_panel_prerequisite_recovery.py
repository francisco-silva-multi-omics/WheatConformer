from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.v2.recover_phase5_panel_prerequisites import (
    canonicalize_gid,
    normalize_sample_id,
)


RELEASE = (
    Path(__file__).resolve().parents[1]
    / "audit/v2/phase5_panel_prerequisite_recovery_v1"
)


def test_identity_normalization_is_bounded() -> None:
    assert normalize_sample_id(" SEEDDIV1000 ") == "SEEDDIV1000"
    assert canonicalize_gid("194554") == ("194554", "GID194554")
    assert canonicalize_gid("GID194554") == ("194554", "GID194554")
    with pytest.raises(ValueError):
        canonicalize_gid("HGO94.12.2.37")


@pytest.fixture(scope="module")
def decision() -> dict:
    path = RELEASE / "PHASE5_PANEL_PREREQUISITE_RECOVERY_DECISION.json"
    if not path.exists():
        pytest.skip("Panel prerequisite recovery release has not been built")
    return json.loads(path.read_text(encoding="utf-8"))


def test_release_is_phenotype_blind(decision: dict) -> None:
    assert decision["stage1_version"] == "Stage-1_v2"
    assert decision["phenotype_values_read"] is False
    assert decision["inner_validation_metrics_read"] is False
    assert decision["outer_test_outcomes_read"] is False
    assert decision["outer_test_metrics_read"] is False
    assert decision["final_holdout_outcomes_read"] is False
    assert decision["model_training_performed"] is False
    assert decision["kernels_modified"] is False


def test_cimmyt_prefilter_source_is_recovered_without_activation(decision: dict) -> None:
    cimmyt = decision["cimmyt"]
    assert cimmyt["source_description_marker_rows"] == 91_680
    assert cimmyt["sample_columns"] == 53_525
    assert cimmyt["unique_sample_columns"] == 53_525
    assert cimmyt["accepted_identity_stage1_primary_gids"] > 4_512
    assert cimmyt["production_kernel_activated"] is False
    assert cimmyt["strict_production_K_G_disposition"] == (
        "READY_FOR_STREAMED_CALL_VALIDATION_AND_150_STATE_TRAINING_LOCAL_QC"
    )


def test_80k_identity_is_exact_but_production_blockers_remain(decision: dict) -> None:
    summary = decision["dartseq80k"]
    assert summary["authoritative_passport_sample_ids"] == 79_191
    assert summary["certified_unique_panel_samples"] == 94_855
    assert summary["unique_panel_samples_with_exact_typed_identity"] == 94_855
    assert summary["matrix_axis_rows_with_exact_typed_identity"] == 174_048
    assert summary["identity_blocker_resolved"] is True
    assert summary["strict_production_K_G_disposition"].startswith("BLOCKED_")

    samples = pd.read_parquet(
        RELEASE / "results/dartseq80k_unique_sample_identity_classification.parquet"
    )
    assert samples.typed_identity_exact.astype(bool).all()
    assert set(samples.identity_class).issubset(
        {
            "accepted_unique_identity",
            "accepted_identity_replicate_set_pending_concordance",
        }
    )


def test_eyt_provenance_is_not_overclaimed(decision: dict) -> None:
    eyt = decision["eyt"]
    assert eyt["source_snp_count_initial"] == 50_058
    assert eyt["source_snp_count_filtered"] == 14_027
    assert eyt["published_haplotype_block_count"] == 519
    assert eyt["complete_519_block_to_source_snp_membership_recovered"] is False
    assert eyt["source_snp_call_matrix_recovered"] is False
    assert eyt["strict_inductive_disposition"].startswith("BLOCKED_")
