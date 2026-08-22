from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "v2"))

from phase5_split_bound_common import (  # noqa: E402
    PROHIBITED_SPLIT_COLUMNS,
    SPLIT_ALLOWED_COLUMNS,
    assign_balanced_entities,
    assert_many_to_one,
    build_pedigree_factor,
    consensus_dosage,
    decode_biallelic_call,
    fit_vanraden,
    geo_factor,
    index_signature,
    kernel_diagnostics,
    outer_role,
    relationship_block,
)
from phase5_split_bound_build import PANEL_CLASS, portable_selection  # noqa: E402
from phase5_split_bound_finalize import parse_pytest_log  # noqa: E402


RELEASE = ROOT / "audit" / "v2" / "phase5_split_bound_kernel_validation_v2"


def test_split_projection_excludes_every_prohibited_column() -> None:
    assert not PROHIBITED_SPLIT_COLUMNS.intersection(SPLIT_ALLOWED_COLUMNS)


def test_balanced_assignment_is_row_order_invariant() -> None:
    frame = pd.DataFrame(
        {
            "canonical_gid": [f"GID{i}" for i in range(1, 41)],
            "primary_rows": [i % 7 + 1 for i in range(40)],
            "secondary_rows": [i % 11 + 2 for i in range(40)],
        }
    )
    left = assign_balanced_entities(frame, "canonical_gid", "TEST")
    right = assign_balanced_entities(frame.sample(frac=1, random_state=17), "canonical_gid", "TEST")
    pd.testing.assert_frame_equal(left, right)


def test_secondary_only_entities_never_reassign_primary_entities() -> None:
    primary = pd.DataFrame(
        {"canonical_gid": ["GID1", "GID2"], "primary_rows": [10, 5], "secondary_rows": [10, 5]}
    )
    extended = pd.concat(
        [primary, pd.DataFrame({"canonical_gid": ["GID3"], "primary_rows": [0], "secondary_rows": [100]})],
        ignore_index=True,
    )
    a = assign_balanced_entities(primary, "canonical_gid", "TEST").set_index("entity_id")
    b = assign_balanced_entities(extended, "canonical_gid", "TEST").set_index("entity_id")
    assert a.loc["GID1", "assigned_fold"] == b.loc["GID1", "assigned_fold"]
    assert a.loc["GID2", "assigned_fold"] == b.loc["GID2", "assigned_fold"]


@pytest.mark.parametrize(
    ("scenario", "gid_fold", "env_fold", "seen", "expected"),
    [
        ("GNEW_EOBS", 1, 2, True, "TEST"),
        ("GNEW_EOBS", 1, 2, False, "EMBARGO_OTHER_ENTITY_UNSEEN"),
        ("GNEW_EOBS", 2, 1, False, "TRAIN"),
        ("GOBS_ENEW", 2, 1, True, "TEST"),
        ("GOBS_ENEW", 2, 1, False, "EMBARGO_OTHER_ENTITY_UNSEEN"),
        ("GNEW_ENEW", 1, 1, True, "TEST"),
        ("GNEW_ENEW", 2, 2, True, "TRAIN"),
        ("GNEW_ENEW", 1, 2, True, "EMBARGO_SINGLE_NOVELTY"),
    ],
)
def test_outer_role_contract(
    scenario: str, gid_fold: int, env_fold: int, seen: bool, expected: str
) -> None:
    assert outer_role(scenario, 1, gid_fold, env_fold, seen) == expected


def test_known_pedigree_relationships_and_inbreeding() -> None:
    parents = {
        "F1": ("", ""),
        "F2": ("", ""),
        "O": ("F1", "F2"),
        "SELF": ("O", "O"),
        "ONE": ("F1", ""),
        "REP": ("O", "F1"),
    }
    factor, d, order, _ = build_pedigree_factor(parents)
    index = {value: i for i, value in enumerate(order)}
    selected = [index[value] for value in ("F1", "F2", "O", "SELF", "ONE", "REP")]
    matrix = relationship_block(factor, d, selected)
    assert matrix[0, 2] == pytest.approx(0.5)
    assert matrix[1, 2] == pytest.approx(0.5)
    assert matrix[2, 2] == pytest.approx(1.0)
    assert matrix[3, 3] > 1.0
    assert np.linalg.eigvalsh(matrix).min() >= -1e-10


def test_pedigree_parent_order_does_not_change_relationships() -> None:
    left = {"A": ("", ""), "B": ("", ""), "C": ("A", "B")}
    right = {"A": ("", ""), "B": ("", ""), "C": ("B", "A")}
    lf, ld, lo, _ = build_pedigree_factor(left)
    rf, rd, ro, _ = build_pedigree_factor(right)
    lm = relationship_block(lf, ld, [lo.index(x) for x in ("A", "B", "C")])
    rm = relationship_block(rf, rd, [ro.index(x) for x in ("A", "B", "C")])
    np.testing.assert_allclose(lm, rm, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("call", "alleles", "expected"),
    [("A", "A/G", 0.0), ("G", "A/G", 2.0), ("R", "A/G", 1.0), ("N", "A/G", np.nan)],
)
def test_biallelic_iupac_dosage(call: str, alleles: str, expected: float) -> None:
    observed = decode_biallelic_call(call, alleles)
    if np.isnan(expected):
        assert np.isnan(observed)
    else:
        assert observed == expected


def test_consensus_keeps_agreement_and_masks_discordance() -> None:
    values = np.array([[0.0, 1.0, np.nan, 0.0], [0.0, 2.0, 2.0, np.nan]])
    consensus, conflicts = consensus_dosage(values)
    np.testing.assert_allclose(consensus[[0, 2, 3]], [0.0, 2.0, 0.0])
    assert np.isnan(consensus[1])
    assert conflicts == 1


def test_vanraden_training_only_fit_is_application_invariant() -> None:
    dosage = np.array(
        [[0.0, 0.0, 1.0, np.nan], [2.0, 0.0, 1.0, 2.0], [0.0, 2.0, 1.0, 0.0], [2.0, 2.0, 1.0, np.nan]]
    )
    ids = ["GID1", "GID2", "GID3", "GID4"]
    fitted = fit_vanraden(dosage, ids, {"GID1", "GID2"})
    changed = dosage.copy()
    changed[2:, :] = np.array([[2.0, 2.0, 0.0, 2.0], [0.0, 0.0, 2.0, 0.0]])
    refitted = fit_vanraden(changed, ids, {"GID1", "GID2"})
    np.testing.assert_array_equal(fitted["retained_mask"], refitted["retained_mask"])
    np.testing.assert_allclose(fitted["allele_frequency"], refitted["allele_frequency"], atol=0, rtol=0)
    assert fitted["denominator"] == refitted["denominator"]


def test_vanraden_drops_training_monomorphic_marker() -> None:
    dosage = np.array([[0.0, 1.0], [2.0, 1.0], [0.0, 2.0]])
    fitted = fit_vanraden(dosage, ["A", "B", "C"], {"A", "B"})
    assert fitted["retained_mask"].tolist() == [True, False]


def test_vanraden_kernel_is_symmetric_psd_and_finite() -> None:
    dosage = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
    fitted = fit_vanraden(dosage, ["A", "B", "C"], {"A", "B", "C"})
    diagnostics = kernel_diagnostics(fitted["factor"])
    assert diagnostics["all_finite"]
    assert diagnostics["max_symmetry_error"] <= 1e-12
    assert diagnostics["minimum_eigenvalue"] >= -1e-10


def test_environment_geo_fit_uses_training_levels_only() -> None:
    ids = ["E1", "E2", "E3"]
    locations = ["L1", "L2", "L3"]
    factor, levels, _ = geo_factor(ids, locations, {"E1", "E2"})
    assert levels == ["L1", "L2"]
    assert factor.getrow(2).nnz == 0


def test_environment_linear_factor_handles_identical_and_distant_locations() -> None:
    factor, _, _ = geo_factor(["E1", "E2", "E3"], ["L1", "L1", "L2"], {"E1", "E2", "E3"})
    kernel = (factor @ factor.T).toarray()
    assert kernel[0, 1] == pytest.approx(1.0)
    assert kernel[0, 2] == pytest.approx(0.0)


def test_sparse_gxe_complete_grid_and_deleted_observation() -> None:
    kg = np.array([[1.0, 0.5], [0.5, 1.0]])
    ke = np.array([[1.0, 0.2], [0.2, 1.0]])
    observations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    full = np.array([[kg[g1, g2] * ke[e1, e2] for g2, e2 in observations] for g1, e1 in observations])
    assert np.linalg.eigvalsh(full).min() >= -1e-10
    deleted = full[np.ix_([0, 1, 3], [0, 1, 3])]
    assert deleted.shape == (3, 3)
    assert np.linalg.eigvalsh(deleted).min() >= -1e-10


def test_many_to_many_join_is_rejected() -> None:
    left = pd.DataFrame({"id": ["A", "A"]})
    right = pd.DataFrame({"id": ["A", "A"], "value": [1, 2]})
    with pytest.raises(ValueError, match="not unique"):
        assert_many_to_one(left, right, "id")


def test_index_signature_detects_permutation() -> None:
    assert index_signature(["A", "B", "C"]) != index_signature(["B", "A", "C"])


def test_persisted_selection_contract_is_release_root_portable(tmp_path: Path) -> None:
    assignment = tmp_path / "splits" / "observation_split_assignment.parquet"
    physical = assignment.resolve().as_posix()
    expression = f"environment_id IN (SELECT environment_id FROM read_parquet('{physical}'))"
    portable = portable_selection(expression, assignment)
    assert physical not in portable
    assert "${PHASE5_RELEASE_ROOT}/splits/observation_split_assignment.parquet" in portable


def test_pytest_log_parser_accepts_utf16_powershell_output(tmp_path: Path) -> None:
    log = tmp_path / "pytest.stdout.log"
    log.write_text("35 passed, 1 deselected in 2.0s\n", encoding="utf-16")
    parsed = parse_pytest_log(log)
    assert parsed["passed"] == 35
    assert parsed["failed"] == 0
    assert parsed["deselected"] == 1
    assert parsed["status"] == "PASS"


def test_mas_panels_cannot_enter_genomewide_kg() -> None:
    mas = [panel for panel, value in PANEL_CLASS.items() if panel.startswith("mas_")]
    assert mas
    assert all(PANEL_CLASS[panel][0] == "TARGETED_MAS_COVARIATE_OR_SPARSE_KERNEL" for panel in mas)


def test_80k_panels_remain_unauthorized() -> None:
    panels = [panel for panel in PANEL_CLASS if panel.startswith("dartseq80k_")]
    assert panels
    assert all(PANEL_CLASS[panel][0] == "IDENTITY_CANDIDATE_ONLY_NOT_AUTHORIZED" for panel in panels)


@pytest.mark.skipif(not (RELEASE / "PHASE5_RELEASE_DECISION.json").exists(), reason="release not finalized")
def test_release_decision_is_atomic_pass() -> None:
    decision = json.loads((RELEASE / "PHASE5_RELEASE_DECISION.json").read_text(encoding="utf-8"))
    assert decision["status"] == "PASS_PHASE5_KERNEL_VALIDATION"
    assert decision["handoff_flag"] == "READY_FOR_PHASE6_MODEL_SELECTION"


@pytest.mark.skipif(not (RELEASE / "indices/canonical_phase5_observation_index.parquet").exists(), reason="release not built")
def test_master_index_retains_all_authorized_and_archival_rows() -> None:
    metadata = pq.ParquetFile(RELEASE / "indices/canonical_phase5_observation_index.parquet").metadata
    assert metadata.num_rows == 3_193_677


@pytest.mark.skipif(not (RELEASE / "splits/split_leakage_report.tsv").exists(), reason="release not built")
def test_release_split_leakage_checks_all_pass() -> None:
    frame = pd.read_csv(RELEASE / "splits/split_leakage_report.tsv", sep="\t")
    assert len(frame) == 110
    assert set(frame["status"]) == {"PASS"}


@pytest.mark.skipif(not (RELEASE / "model_inputs/model_input_registry.tsv").exists(), reason="release not built")
def test_release_selection_contracts_do_not_embed_physical_release_root() -> None:
    registry = pd.read_csv(RELEASE / "model_inputs/model_input_registry.tsv", sep="\t")
    rules = registry["observation_order_rule"].astype(str)
    assert not rules.str.contains(RELEASE.resolve().as_posix(), regex=False).any()
    assert rules.str.contains("${PHASE5_RELEASE_ROOT}", regex=False).any()


@pytest.mark.skipif(not (RELEASE / "genomic/panel_registry.tsv").exists(), reason="release not built")
def test_exactly_one_dense_production_panel_is_included() -> None:
    frame = pd.read_csv(RELEASE / "genomic/panel_registry.tsv", sep="\t")
    included = frame[frame["production_included"].astype(str).str.lower().eq("true")]
    assert included["panel_id"].tolist() == ["hibap35k"]


@pytest.mark.skipif(not (RELEASE / "protected_outcome_access_audit.tsv").exists(), reason="release not built")
def test_protected_outcomes_are_absent_or_not_accessed() -> None:
    frame = pd.read_csv(RELEASE / "protected_outcome_access_audit.tsv", sep="\t")
    prohibited = frame[frame["prohibited_for_stage"].astype(str).str.lower().eq("true")]
    assert not prohibited["accessed"].astype(str).str.lower().eq("true").any()
