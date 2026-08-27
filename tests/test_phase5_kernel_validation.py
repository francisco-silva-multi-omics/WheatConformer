from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from scripts.v2.phase5_independent_reconstruction import (
    assert_many_to_one,
    assert_same_index,
    environment_linear_kernel,
    fit_marker_transform,
    gxe_elements,
    synthetic_results,
    vanraden,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("STAGE1_V2_DATA_ROOT", CODE_ROOT)).resolve()
RELEASE = DATA_ROOT / "audit" / "v2" / "phase5_kernel_validation_v1"
PROMOTED = (
    DATA_ROOT
    / "audit"
    / "v2"
    / "phase4_integrated_spatial_promotion_release_v1"
    / "promoted_phenotypes.parquet"
)


def test_independent_analytical_suite_passes() -> None:
    result = synthetic_results()
    assert result["status"].eq("PASS").all(), result


def test_vanraden_drops_monomorphic_all_heterozygote_marker() -> None:
    marker = np.array([[0, 1], [1, 1], [2, 1]], dtype=float)
    _, _, kept, _ = vanraden(marker)
    assert kept.tolist() == [0]


def test_marker_parameters_are_fit_on_training_rows_only() -> None:
    base = np.array([[0, 0], [2, 2], [0, 0]], dtype=float)
    changed = base.copy()
    changed[2] = [2, 2]
    _, p1, kept1, denominator1 = fit_marker_transform(base, [0, 1])
    _, p2, kept2, denominator2 = fit_marker_transform(changed, [0, 1])
    assert np.array_equal(kept1, kept2)
    assert np.allclose(p1, p2)
    assert denominator1 == denominator2


def test_environment_parameters_are_fit_on_training_rows_only() -> None:
    base = np.array([[0, 0], [2, 2], [4, 4]], dtype=float)
    changed = base.copy()
    changed[2] = [400, -400]
    _, _, mean1, std1 = environment_linear_kernel(base, [0, 1])
    _, _, mean2, std2 = environment_linear_kernel(changed, [0, 1])
    assert np.allclose(mean1, mean2)
    assert np.allclose(std1, std2)


def test_sparse_gxe_is_component_product() -> None:
    kg = np.array([[1.0, 0.3], [0.3, 1.2]])
    ke = np.array([[1.0, -0.2], [-0.2, 0.9]])
    values = gxe_elements(kg, ke, [0, 1, 0], [0, 0, 1], [(0, 1), (0, 2), (1, 2)])
    assert np.allclose(values, [0.3, -0.2, -0.06])


def test_alignment_permutations_are_detected() -> None:
    expected = ["G1", "G2", "G3"]
    for label, observed in {
        "genotype": ["G2", "G1", "G3"],
        "environment": ["G1", "G3", "G2"],
        "phenotype": ["G3", "G2", "G1"],
        "weight": ["G1", "G3", "G2"],
        "trait": ["G2", "G3", "G1"],
    }.items():
        with pytest.raises(ValueError, match="index mismatch"):
            assert_same_index(expected, observed, label)


def test_many_to_many_join_is_rejected() -> None:
    left = pd.DataFrame({"id": ["A", "A"]})
    right = pd.DataFrame({"id": ["A", "A"], "value": [1, 2]})
    with pytest.raises(ValueError, match="many-to-many"):
        assert_many_to_one(left, right, "id")


def test_scope_contract_is_stage1_v2_only() -> None:
    scope = json.loads((RELEASE / "PHASE5_SCOPE_CORRECTION.json").read_text(encoding="utf-8"))
    assert scope["authoritative_modelling_foundation"] == "STAGE1_V2"
    assert scope["historical_certified_v1_artifacts_consumed_as_v2_inputs"] is False


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        ("primary_weighted_training_eligible", 2_045_518),
        ("secondary_unweighted_training_eligible", 2_242_863),
        ("continuous_error_evaluation_eligible", 2_242_863),
        ("correlation_evaluation_eligible", 2_242_615),
        ("ranking_evaluation_eligible", 1_418_644),
        ("NOT canonical_gid_eligible", 950_814),
        ("phenotype_release_eligible AND NOT secondary_unweighted_training_eligible", 950_814),
        ("NOT phenotype_release_eligible", 0),
    ],
)
def test_authorized_views_reproduce(predicate: str, expected: int) -> None:
    count = duckdb.connect().execute(
        f"SELECT count(*) FROM read_parquet('{PROMOTED.as_posix()}') WHERE {predicate}"
    ).fetchone()[0]
    assert count == expected


def test_primary_index_is_complete_unique_and_accepted() -> None:
    path = RELEASE / "canonical_phase5_observation_index.parquet"
    row = duckdb.connect().execute(f"""
      SELECT count(*),count(DISTINCT phase4_adjusted_row_id),
             count(*) FILTER(WHERE canonical_gid IS NULL OR canonical_gid=''),
             count(*) FILTER(WHERE identity_status NOT LIKE 'ACCEPTED_PHASE3G_R2%'),
             count(*) FILTER(WHERE canonical_gid NOT LIKE 'GID%'),
             count(*) FILTER(WHERE NOT canonical_gid_namespace_mismatch)
      FROM read_parquet('{path.as_posix()}')
    """).fetchone()
    assert row == (2_045_518, 2_045_518, 0, 0, 0, 0)


def test_phase4_canonical_gid_namespace_defect_is_explicit() -> None:
    row = duckdb.connect().execute(f"""
      SELECT count(*) FILTER(WHERE canonical_gid_eligible),
             count(*) FILTER(WHERE canonical_gid_eligible AND typed_source_genotype_id='GID'||canonical_gid)
      FROM read_parquet('{PROMOTED.as_posix()}')
    """).fetchone()
    assert row == (2_242_863, 2_242_863)


def test_zero_pev_and_invalid_uncertainty_are_not_in_primary() -> None:
    row = duckdb.connect().execute(f"""
      SELECT count(*) FILTER(WHERE primary_weighted_training_eligible AND pev_proxy=0),
             count(*) FILTER(WHERE primary_weighted_training_eligible AND
                (reliability_weight IS NULL OR NOT isfinite(reliability_weight)))
      FROM read_parquet('{PROMOTED.as_posix()}')
    """).fetchone()
    assert row == (0, 0)


def test_ranking_unsuitable_never_enters_ranking_view() -> None:
    count = duckdb.connect().execute(f"""
      SELECT count(*) FROM read_parquet('{PROMOTED.as_posix()}')
      WHERE ranking_evaluation_eligible AND ranking_status NOT LIKE 'RANKING_SIGNAL_USABLE%'
    """).fetchone()[0]
    assert count == 0


def test_check_huber_and_coordinates_are_not_filters_or_features() -> None:
    path = RELEASE / "canonical_phase5_observation_index.parquet"
    columns = duckdb.connect().execute(f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')").fetchdf()["column_name"].tolist()
    assert "check_status" in columns and "huber_status" in columns
    assert "field_row" not in columns and "field_column" not in columns
    assert "coordinate_status" not in columns


def test_no_protected_outcome_access_is_attested() -> None:
    audit = pd.read_csv(RELEASE / "protected_outcome_access_audit.tsv", sep="\t")
    assert (~audit["accessed"].astype(bool)).all()
    assert (~audit["hashed"].astype(bool)).all()
    assert (~audit["summarized"].astype(bool)).all()


def test_missing_v2_kernels_fail_activation_loudly() -> None:
    issues = pd.read_csv(RELEASE / "kernel_issue_ledger.tsv", sep="\t")
    assert {"K_A", "K_G", "K_E", "model_inputs/GxE"}.issubset(set(issues["component"]))
    assert issues["status"].str.startswith("OPEN_").all()


def test_primary_marker_coverage_uses_exact_r2_overlay() -> None:
    coverage = pd.read_csv(RELEASE / "genotype_marker_coverage_by_view.tsv", sep="\t")
    row = coverage[
        coverage["view"].eq("PRIMARY_WEIGHTED_TRAINING") &
        coverage["panel_id"].eq("ANY_ACCEPTED_PANEL")
    ].iloc[0]
    assert int(row["view_canonical_gids"]) == 10_656
    assert int(row["marker_vector_gids"]) == 10_656


def test_unversioned_environment_candidates_are_not_activated() -> None:
    diagnostics = pd.read_csv(RELEASE / "ke_kernel_diagnostics.tsv", sep="\t")
    assert (~diagnostics["versioned_stage1_v2_binding"].astype(bool)).all()
    assert diagnostics["status"].str.contains("NOT_ACTIVATABLE|FAILS_CURRENT").all()
