from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("STAGE1_V2_DATA_ROOT", CODE_ROOT)).resolve()
RELEASE = DATA_ROOT / "audit/v2/phase5_cimmyt_pre_qc_split_local_v1"
PROTOCOL = CODE_ROOT / "scripts/v2/cimmyt_pre_qc_split_local_protocol_v1.json"


def test_protocol_is_frozen_before_streaming() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["stage1_version"] == "Stage-1_v2"
    assert protocol["source"]["description_marker_rows"] == 91_680
    assert protocol["sample_qc"]["minimum_raw_call_rate"] == 0.50
    assert protocol["marker_qc"]["minimum_training_call_rate"] == 0.80
    assert protocol["marker_qc"]["minimum_training_minor_allele_frequency"] == 0.01
    assert protocol["marker_qc"]["maximum_training_heterozygosity"] == 0.10
    assert protocol["required_state_count"] == 150


@pytest.fixture(scope="module")
def decision() -> dict:
    path = RELEASE / "CIMMYT_PRE_QC_SPLIT_LOCAL_DECISION.json"
    if not path.exists():
        pytest.skip("CIMMYT split-local release has not been built")
    return json.loads(path.read_text(encoding="utf-8"))


def test_release_is_outcome_blind(decision: dict) -> None:
    assert decision["phenotype_values_read"] is False
    assert decision["inner_validation_metrics_read"] is False
    assert decision["outer_test_outcomes_read"] is False
    assert decision["outer_test_metrics_read"] is False
    assert decision["final_holdout_outcomes_read"] is False
    assert decision["model_training_performed"] is False
    assert decision["existing_kernels_modified"] is False


def test_one_pass_source_and_shared_calls(decision: dict) -> None:
    audit = json.loads(
        (RELEASE / "genomic/cimmyt_pre_qc_source_stream_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["status"] == "PASS_SINGLE_STREAM_SOURCE_AND_CALL_CERTIFICATION"
    assert audit["single_physical_source_pass"] is True
    assert audit["marker_rows"] == 91_680
    assert audit["header_sample_columns"] == 53_525
    assert audit["selected_primary_GIDs"] == 5_629
    calls = np.load(RELEASE / decision["shared_raw_call_matrix"], mmap_mode="r")
    assert calls.shape == (91_680, 5_629)
    assert calls.dtype == np.uint8


def test_all_150_states_are_training_local_and_ready(decision: dict) -> None:
    qc = pd.read_csv(
        RELEASE / "states/cimmyt_pre_qc_fold_preprocessing_registry.tsv", sep="\t"
    )
    registry = pd.read_csv(
        RELEASE / "states/cimmyt_pre_qc_component_registry.tsv", sep="\t"
    )
    assert len(qc) == len(registry) == 150
    assert qc.state_id.nunique() == registry.state_id.nunique() == 150
    assert qc.marker_QC_fit_scope.eq("TRAINING_GIDS_ONLY").all()
    assert qc.allele_frequency_fit_scope.eq("TRAINING_GIDS_ONLY").all()
    assert qc.imputation_fit_scope.eq("TRAINING_GIDS_ONLY_2P_ON_DEMAND").all()
    assert not qc.held_out_calls_used_for_parameters.astype(bool).any()
    assert qc.strict_production_eligible.astype(bool).all()
    assert registry.strict_production_component_available.astype(bool).all()
    assert decision["strict_ready_state_count"] == 150
    assert decision["masked_state_count"] == 0


def test_artifact_manifest_verifies(decision: dict) -> None:
    import hashlib

    manifest = pd.read_csv(RELEASE / "artifact_manifest.tsv", sep="\t", dtype=str)
    failures = []
    for row in manifest.itertuples(index=False):
        digest_builder = hashlib.sha256()
        with (RELEASE / row.relative_path).open("rb") as handle:
            while chunk := handle.read(16 * 1024 * 1024):
                digest_builder.update(chunk)
        digest = digest_builder.hexdigest()
        if digest != row.sha256:
            failures.append(row.relative_path)
    assert not failures
