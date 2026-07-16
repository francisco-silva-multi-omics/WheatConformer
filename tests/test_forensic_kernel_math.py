from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from build_environment_component_kernels import assert_kernel_valid, parse_value, standardized_kernel
from build_pedigree_kernel import additive_relationship, assert_relationship_valid, build_parent_table
from audit.validate_server_artifacts import validate_explicit_kernel_order
from audit.audit_common import (
    independent_additive_relationship,
    independent_environment_kernel,
    independent_observation_gxe,
    independent_vanraden,
    join_cardinality,
    mean_impute_markers,
)
from audit.run_forensic_audit import git_provenance, source_path_label
from server_training_pipeline.split_utils import make_split, split_group_column, split_leakage_record


def test_source_path_label_records_actual_relative_source(tmp_path: Path) -> None:
    trial_root = tmp_path / "TRIALS_AND_NURSERIES"
    trial_root.mkdir()
    assert source_path_label(tmp_path, trial_root) == "TRIALS_AND_NURSERIES"


def test_git_provenance_uses_archive_deployment_receipt(tmp_path: Path) -> None:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    commit = "94d3e73fbe8320d912dcf04b7f0f8eda7074cf7f"
    (audit_dir / "DEPLOYED_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")

    provenance = git_provenance(tmp_path, audit_dir)

    assert provenance["repository_present"] is False
    assert provenance["provenance_source"] == "deployment_receipt"
    assert provenance["commit"] == commit
    assert provenance["status_porcelain"] == "not_available_for_archive_deployment"
    assert provenance["receipt_path"] == str((audit_dir / "DEPLOYED_COMMIT.txt").resolve())


def test_git_provenance_is_explicit_when_no_provenance_exists(tmp_path: Path) -> None:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()

    provenance = git_provenance(tmp_path, audit_dir)

    assert provenance["repository_present"] is False
    assert provenance["provenance_source"] == "unavailable"
    assert provenance["commit"] == ""
    assert provenance["status_porcelain"] == "not_available"


def test_vanraden_matches_analytical_two_marker_example() -> None:
    markers = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]])
    kernel, frequency, denominator = independent_vanraden(markers)
    expected_frequency = np.array([0.5, 0.5])
    centered = markers - 2.0 * expected_frequency
    expected = centered @ centered.T / 1.0
    np.testing.assert_allclose(frequency, expected_frequency)
    assert denominator == pytest.approx(1.0)
    np.testing.assert_allclose(kernel, expected)


def test_marker_mean_imputation_is_column_specific() -> None:
    markers = np.array([[0.0, np.nan], [2.0, 1.0], [np.nan, 2.0]])
    imputed, means = mean_impute_markers(markers)
    np.testing.assert_allclose(means, [1.0, 1.5])
    np.testing.assert_allclose(imputed, [[0.0, 1.5], [2.0, 1.0], [1.0, 2.0]])


def test_additive_relationship_for_founders_and_full_sib() -> None:
    records = [("P1", "", ""), ("P2", "", ""), ("C1", "P1", "P2"), ("C2", "P1", "P2")]
    matrix, order = independent_additive_relationship(records, ["P1", "P2", "C1", "C2"])
    expected = np.array(
        [
            [1.0, 0.0, 0.5, 0.5],
            [0.0, 1.0, 0.5, 0.5],
            [0.5, 0.5, 1.0, 0.5],
            [0.5, 0.5, 0.5, 1.0],
        ]
    )
    assert order == ["P1", "P2", "C1", "C2"]
    np.testing.assert_allclose(matrix, expected)


def test_additive_relationship_rejects_conflicting_pedigrees_and_cycles() -> None:
    with pytest.raises(ValueError, match="Conflicting pedigree"):
        independent_additive_relationship([("C", "P1", "P2"), ("C", "P3", "P4")])
    with pytest.raises(ValueError, match="cycle"):
        independent_additive_relationship([("A", "B", ""), ("B", "A", "")])


def test_environment_kernel_drops_nonfinite_and_constant_columns_explicitly() -> None:
    features = np.array(
        [
            [1.0, 5.0, np.inf, 10.0],
            [2.0, 5.0, 3.0, np.nan],
            [3.0, 5.0, 4.0, 14.0],
        ]
    )
    kernel, standardized, retained = independent_environment_kernel(features)
    assert retained.tolist() == [True, False, True, True]
    assert np.isfinite(kernel).all()
    assert np.mean(np.diag(kernel)) == pytest.approx(1.0)
    assert standardized.shape == (3, 3)


def test_gxe_is_observation_indexed_hadamard_product() -> None:
    kg = np.array([[1.0, 0.25], [0.25, 1.0]])
    ke = np.array([[1.0, 0.5], [0.5, 1.0]])
    genotype = np.array([0, 0, 1])
    environment = np.array([0, 1, 1])
    actual = independent_observation_gxe(kg, ke, genotype, environment)
    expected = kg[np.ix_(genotype, genotype)] * ke[np.ix_(environment, environment)]
    np.testing.assert_allclose(actual, expected)
    assert actual[0, 1] == pytest.approx(0.5)
    assert actual[1, 2] == pytest.approx(0.25)


def test_join_cardinality_exposes_many_to_many_expansion() -> None:
    left = pd.DataFrame({"id": ["A", "A", "B"]})
    right = pd.DataFrame({"id": ["A", "A", "C"]})
    result = join_cardinality(left, right, ["id"])
    assert result["many_to_many_keys"] == 1
    assert result["joined_rows_expected"] == 4
    assert result["rows_duplicated_by_join"] == 2


@pytest.mark.parametrize("mode", ["gho_environment", "cv1_environment", "cv1_genotype", "cv0_genotype_environment"])
def test_declared_holdout_axes_have_no_overlap(mode: str) -> None:
    frame = pd.DataFrame(
        {
            "panel_sample_id": [f"G{g}" for g in range(12) for _ in range(12)],
            "env_kernel_id": [f"E{e}" for _ in range(12) for e in range(12)],
        }
    )
    group_col = split_group_column(mode)
    train, val, test = make_split(frame, mode, 20260715, 0.2, 0.1, group_col=group_col)
    record = split_leakage_record(frame, 0, mode, train, val, test, group_col=group_col)
    assert record["leakage_status"] == "pass"


def test_kernel_order_mismatch_is_detectable() -> None:
    kernel_order = pd.Series(["G1", "G2", "G3"])
    observation_order = pd.Series(["G1", "G3", "G2"])
    assert not kernel_order.equals(observation_order)


def test_server_validator_requires_explicit_kernel_order_alignment(tmp_path) -> None:
    kernel_path = tmp_path / "K_E.npy"
    order_path = tmp_path / "order.tsv"
    np.save(kernel_path, np.eye(3, dtype=np.float32))
    pd.DataFrame({"env_id": ["E1", "E2", "E3"]}).to_csv(order_path, sep="\t", index=False)
    assert validate_explicit_kernel_order(kernel_path, order_path, "env_id", "K_E")["status"] == "PASS"

    pd.DataFrame({"env_id": ["E1", "E1", "E3"]}).to_csv(order_path, sep="\t", index=False)
    assert validate_explicit_kernel_order(kernel_path, order_path, "env_id", "K_E")["status"] == "FAIL"


@pytest.mark.parametrize(
    ("value", "trait", "expected"),
    [
        ("YES", "IRRIGATION", 1.0),
        ("125 kg/ha", "N_FERTILIZER_RATE", 125.0),
        ("1,250 mm", "PRECIPITATION", 1250.0),
        ("15/11/2020", "SOWING_DATE_TEXT", 320.0),
        ("Nov 15 2020", "SOWING_DATE", 320.0),
        ("5 December 2018", "HARVEST_STARTING_DATE", 339.0),
        ("2020-11-15", "SOWING_DATE", 320.0),
    ],
)
def test_environment_parser_accepts_only_typed_values(value: str, trait: str, expected: float) -> None:
    assert parse_value(value, trait) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "trait"),
    [
        ("ROUNDUP 360", "HERBICIDE_PRODUCT(S)"),
        ("NPK 15-15-15 200 KG", "FERTILIZER_1"),
        ("UREA 46 + DAP 18-46-0", "FERTILIZER_TEXT_123"),
        ("2,4-D", "HERBICIDE_PRODUCT(S)"),
        ("N", "FUNGICIDE_PRODUCT(S)"),
        ("NO", "HERBICIDE_PRODUCT(S)"),
        ("1e999", "PRECIPITATION"),
        ("/01/97", "SOWING_DATE_TEXT"),
        ("12/2021", "SOWING_DATE_TEXT"),
        ("00/05/1998", "SOWING_DATE_TEXT"),
        ("28/01/0000", "HARVEST_FINISHING_DATE_TEXT"),
    ],
)
def test_environment_parser_rejects_categorical_or_nonfinite_numbers(value: str, trait: str) -> None:
    assert np.isnan(parse_value(value, trait))


def test_production_environment_standardization_reports_dropped_columns() -> None:
    features = pd.DataFrame(
        {
            "usable": [1.0, 2.0, 3.0],
            "constant": [5.0, 5.0, 5.0],
            "partly_nonfinite": [1.0, np.inf, 4.0],
            "all_nonfinite": [np.inf, np.nan, -np.inf],
        }
    )
    kernel, z, scaling = standardized_kernel(features)
    assert z.columns.tolist() == ["usable", "partly_nonfinite"]
    assert scaling.set_index("feature").loc["constant", "drop_reason"] == "constant_or_nonfinite_std"
    assert scaling.set_index("feature").loc["all_nonfinite", "drop_reason"] == "no_finite_values"
    assert np.isfinite(kernel).all()
    assert_kernel_valid(kernel, "synthetic_environment")


def test_production_pedigree_builder_rejects_conflicting_assignments(tmp_path) -> None:
    path = tmp_path / "pedigree.tsv"
    pd.DataFrame(
        {
            "sample_id": ["C1", "C1"],
            "parent1": ["P1", "P3"],
            "parent2": ["P2", "P4"],
        }
    ).to_csv(path, sep="\t", index=False)

    class Args:
        pedigree_table = path
        id_col = "sample_id"
        parent1_col = "parent1"
        parent2_col = "parent2"
        cross_col = None
        out_dir = tmp_path

    with pytest.raises(SystemExit, match="Conflicting pedigree assignments"):
        build_parent_table(Args())
    assert (tmp_path / "pedigree_conflicts.tsv").exists()


def test_production_pedigree_rejects_cycles_and_validates_order() -> None:
    cyclic = pd.DataFrame({"sample_id": ["A", "B"], "parent1": ["B", "A"], "parent2": ["", ""]})
    with pytest.raises(ValueError, match="cycle"):
        additive_relationship(cyclic, None)
    valid = pd.DataFrame(
        {
            "sample_id": ["P1", "P2", "C"],
            "parent1": ["", "", "P1"],
            "parent2": ["", "", "P2"],
        }
    )
    matrix, order, _ = additive_relationship(valid, ["P1", "P2", "C"])
    assert_relationship_valid(matrix, order)


def test_audited_pedigree_requires_explicit_canonical_parent_universe(tmp_path) -> None:
    cross_only = tmp_path / "cross_only.tsv"
    pd.DataFrame({"sample_id": ["C1"], "cross_name": ["P1/P2"]}).to_csv(
        cross_only, sep="\t", index=False
    )

    class CrossArgs:
        pedigree_table = cross_only
        id_col = "sample_id"
        parent1_col = None
        parent2_col = None
        cross_col = "cross_name"
        out_dir = tmp_path
        require_explicit_parent_columns = True
        require_parents_in_pedigree = True

    with pytest.raises(SystemExit, match="Explicit canonical parent columns"):
        build_parent_table(CrossArgs())

    missing_parent = tmp_path / "missing_parent.tsv"
    pd.DataFrame({"sample_id": ["C1"], "parent1": ["P1"], "parent2": [""]}).to_csv(
        missing_parent, sep="\t", index=False
    )

    class MissingParentArgs:
        pedigree_table = missing_parent
        id_col = "sample_id"
        parent1_col = "parent1"
        parent2_col = "parent2"
        cross_col = None
        out_dir = tmp_path
        require_explicit_parent_columns = True
        require_parents_in_pedigree = True

    with pytest.raises(SystemExit, match="absent from the reviewed pedigree universe"):
        build_parent_table(MissingParentArgs())
    assert (tmp_path / "pedigree_parents_missing_from_universe.tsv").exists()

    noncanonical = tmp_path / "noncanonical.tsv"
    pd.DataFrame(
        {
            "sample_id": ["GID1", "PARENT_NAME"],
            "parent1": ["PARENT_NAME", ""],
            "parent2": ["", ""],
        }
    ).to_csv(noncanonical, sep="\t", index=False)

    class NoncanonicalArgs:
        pedigree_table = noncanonical
        id_col = "sample_id"
        parent1_col = "parent1"
        parent2_col = "parent2"
        cross_col = None
        out_dir = tmp_path
        require_explicit_parent_columns = True
        require_parents_in_pedigree = True
        required_id_regex = r"GID[0-9]+"

    with pytest.raises(SystemExit, match="do not match required pattern"):
        build_parent_table(NoncanonicalArgs())
    assert (tmp_path / "pedigree_noncanonical_ids.tsv").exists()
