from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from scripts.v2.phase3g_r3_identity_recovery import classify_evidence
from scripts.v2.phase4_namespace_correction import exact_authority_join
from scripts.v2.phase4_namespace_r3_finalize import test_result as parse_test_result
from scripts.v2.phase4_namespace_r3_common import (
    PHASE3G_R3_ROOT,
    PHASE4_NS_ROOT,
    PHASE4_R3_ROOT,
    PINNED_R2_HASHES,
    PHASE3G_R2_ROOT,
    STAGE1_R3_ROOT,
    index_signature,
    sha256,
    stable_id,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pinned_phase3g_r2_hashes():
    assert {name: sha256(PHASE3G_R2_ROOT / name) for name in PINNED_R2_HASHES} == PINNED_R2_HASHES


def test_namespace_release_passes():
    decision = read_json(PHASE4_NS_ROOT / "RELEASE_DECISION.json")
    assert decision["status"] == "PASS_PHASE4_NAMESPACE_CORRECTION"
    assert decision["corrected_records"] == 2_242_863
    assert decision["unresolved_archival_records"] == 950_814


def test_namespace_table_population_and_unique_ids():
    path = PHASE4_NS_ROOT / "corrected_promoted_phenotypes.parquet"
    assert pq.ParquetFile(path).metadata.num_rows == 3_193_677
    con = duckdb.connect()
    assert con.execute("select count(*)-count(distinct phase4_adjusted_row_id) from read_parquet(?)", [str(path)]).fetchone()[0] == 0


def test_all_eligible_rows_are_gid_prefixed():
    path = PHASE4_NS_ROOT / "corrected_promoted_phenotypes.parquet"
    con = duckdb.connect()
    bad = con.execute(
        "select count(*) from read_parquet(?) where canonical_gid_eligible and not regexp_full_match(canonical_gid,'GID[0-9]+')",
        [str(path)],
    ).fetchone()[0]
    assert bad == 0


def test_all_nonidentity_fields_are_equal():
    audit = pd.read_csv(PHASE4_NS_ROOT / "non_identity_field_equality_audit.tsv", sep="\t")
    assert len(audit) == 53
    assert audit["mismatch_rows"].sum() == 0
    assert set(audit["status"]) == {"PASS"}


def test_all_phase4_views_reproduce():
    views = pd.read_csv(PHASE4_NS_ROOT / "view_count_summary.tsv", sep="\t")
    assert len(views) == 8
    assert set(views["status"]) == {"PASS"}


def test_exact_authority_join_accepts_one_to_one():
    left = pd.DataFrame({"typed_source_genotype_id": ["GID1", "GID2"], "canonical_gid_eligible": [True, True]})
    right = pd.DataFrame({"canonical_gid": ["GID1", "GID2"]})
    result = exact_authority_join(left, right)
    assert result["authoritative_canonical_gid"].tolist() == ["GID1", "GID2"]


def test_exact_authority_join_rejects_duplicate_authority():
    left = pd.DataFrame({"typed_source_genotype_id": ["GID1"], "canonical_gid_eligible": [True]})
    right = pd.DataFrame({"canonical_gid": ["GID1", "GID1"]})
    with pytest.raises(ValueError):
        exact_authority_join(left, right)


def test_exact_authority_join_rejects_missing_eligible_mapping():
    left = pd.DataFrame({"typed_source_genotype_id": ["GID2"], "canonical_gid_eligible": [True]})
    right = pd.DataFrame({"canonical_gid": ["GID1"]})
    with pytest.raises(ValueError):
        exact_authority_join(left, right)


def test_name_only_singleton_cannot_be_accepted():
    decision = classify_evidence(set(), {"GID41948"}, set(), False)
    assert decision[0] == "REVIEW_REQUIRED" and decision[2] == ""


def test_ambiguous_name_cannot_be_selected():
    decision = classify_evidence(set(), {"GID1", "GID2"}, set(), False)
    assert decision[0] == "UNRESOLVED_AMBIGUOUS" and decision[2] == ""


def test_two_local_aliases_can_independently_map_to_one_gid():
    first = classify_evidence({"GID7"}, set(), set(), False)
    second = classify_evidence({"GID7"}, set(), set(), False)
    assert first[0] == second[0] == "ACCEPTED_EXACT_AUTHORITY"
    assert first[2] == second[2] == "GID7"


def test_conflicting_exact_authorities_are_rejected():
    decision = classify_evidence({"GID7", "GID8"}, set(), set(), False)
    assert decision[0] == "REJECTED_CONFLICT"


def test_conflicting_same_dataset_sidecar_is_rejected():
    decision = classify_evidence({"GID7"}, {"GID8"}, set(), False)
    assert decision[0] == "REJECTED_CONFLICT"


def test_reused_out_of_namespace_cid_sid_is_review_only():
    decision = classify_evidence(set(), set(), {"GID7"}, False)
    assert decision[0] == "REVIEW_REQUIRED"


def test_generic_label_remains_unresolved():
    decision = classify_evidence(set(), set(), {"GID7"}, True)
    assert decision[0] == "UNRESOLVED_GENERIC_OR_BLANK"


def test_permuted_identifier_orders_change_signature():
    ids = ["GID1", "GID2", "GID3"]
    assert index_signature(ids) != index_signature([ids[1], ids[0], ids[2]])
    source_ids = ["R3K_1", "R3K_2"]
    observation_ids = ["P4E_1", "P4E_2"]
    assert index_signature(source_ids) != index_signature(source_ids[::-1])
    assert index_signature(observation_ids) != index_signature(observation_ids[::-1])


def test_stable_ids_are_deterministic_and_key_sensitive():
    assert stable_id("R3K_", "T", "C", "1", "2") == stable_id("R3K_", "T", "C", "1", "2")
    assert stable_id("R3K_", "T", "C", "1", "2") != stable_id("R3K_", "T", "C", "2", "1")


def test_r3_decisions_conserve_every_source_key():
    counts = pd.read_csv(PHASE3G_R3_ROOT / "r3_decision_counts.tsv", sep="\t")
    assert counts["source_keys"].sum() == 3_086
    assert set(counts["r3_decision"]) == {
        "REVIEW_REQUIRED", "UNRESOLVED_AMBIGUOUS", "UNRESOLVED_GENERIC_OR_BLANK", "UNRESOLVED_INSUFFICIENT_EVIDENCE"
    }


def test_r3_lineage_reproduces_both_denominators():
    checks = pd.read_csv(PHASE3G_R3_ROOT / "source_key_lineage_reconciliation.tsv", sep="\t")
    assert set(checks["status"]) == {"PASS"}
    assert checks.loc[checks.scope.eq("ALL_TRAITS"), "observed_numeric_rows"].item() == 649_206
    assert checks.loc[checks.scope.eq("SEVEN_SELECTED_TRAITS"), "observed_numeric_rows"].item() == 396_262


def test_r3_no_new_identity_decision():
    decision = read_json(PHASE3G_R3_ROOT / "RELEASE_DECISION.json")
    assert decision["status"] == "PASS_PHASE3G_R3_NO_NEW_IDENTITIES"
    assert decision["accepted_exact_authority"] + decision["accepted_exact_authority_with_corroboration"] == 0


def test_no_fictitious_stage1_or_phase4_recovery_release():
    assert not STAGE1_R3_ROOT.exists()
    assert not PHASE4_R3_ROOT.exists()
    assert read_json(PHASE3G_R3_ROOT / "stage1_recovery_applicability.json")["status"] == "NOT_APPLICABLE_NO_NEW_IDENTITIES"


def test_affected_group_fixture_changes_existing_estimate():
    before = np.asarray([10.0, 12.0])
    after = np.asarray([10.0, 12.0, 30.0])
    assert before.mean() != after.mean()
    # This proves why a future accepted recovery requires the complete group,
    # not a metadata-only relabel or a fit of only the recovered row.


def test_recovered_identity_already_in_group_requires_complete_group_rerun():
    before_plots = pd.DataFrame(
        {"group": ["E1_T1", "E1_T1", "E1_T1"], "gid": ["GID1", "GID1", "GID2"], "value": [10.0, 12.0, 20.0]}
    )
    recovered_plot = pd.DataFrame(
        {"group": ["E1_T1"], "gid": ["GID1"], "value": [30.0]}
    )
    before = before_plots.groupby(["group", "gid"], as_index=False)["value"].mean()
    after = pd.concat([before_plots, recovered_plot], ignore_index=True).groupby(["group", "gid"], as_index=False)["value"].mean()
    assert before.loc[before.gid.eq("GID1"), "value"].item() == 11.0
    assert after.loc[after.gid.eq("GID1"), "value"].item() == pytest.approx(52.0 / 3.0)
    assert after.loc[after.gid.eq("GID2"), "value"].item() == 20.0


def test_unaffected_stage1_group_is_invariant_in_recovery_fixture():
    plots = pd.DataFrame(
        {"group": ["A", "A", "B", "B"], "gid": ["GID1", "GID2", "GID3", "GID4"], "value": [1.0, 2.0, 30.0, 40.0]}
    )
    recovered = pd.DataFrame({"group": ["A"], "gid": ["GID1"], "value": [5.0]})
    before_b = plots.loc[plots.group.eq("B")].sort_values("gid").reset_index(drop=True)
    after_b = pd.concat([plots, recovered], ignore_index=True).loc[lambda x: x.group.eq("B")].sort_values("gid").reset_index(drop=True)
    pd.testing.assert_frame_equal(before_b, after_b, check_exact=True)


def test_recovered_estimate_propagates_through_phase4_view_fixture():
    frame = pd.DataFrame(
        {
            "phenotype_ok": [True, True, True],
            "identity_accepted": [True, True, False],
            "uncertainty_ok": [True, False, True],
            "ranking_suitable": [True, True, True],
        }
    )
    frame["secondary"] = frame["phenotype_ok"] & frame["identity_accepted"]
    frame["primary"] = frame["secondary"] & frame["uncertainty_ok"]
    frame["ranking"] = frame["secondary"] & frame["ranking_suitable"]
    assert frame[["primary", "secondary", "ranking"]].sum().to_dict() == {
        "primary": 1, "secondary": 2, "ranking": 2
    }


def test_protected_outcomes_and_production_actions_remain_disabled():
    for root in (PHASE4_NS_ROOT, PHASE3G_R3_ROOT):
        manifest = read_json(root / "run_manifest.json")
        assert manifest["protected_outcomes_accessed"] is False
        assert manifest["model_training_performed"] is False
        assert manifest["production_kernel_construction_performed"] is False


def test_dartseq80k_remains_candidate_only():
    audit = pd.read_csv(PHASE3G_R3_ROOT / "dartseq80k_authority_search_result.tsv", sep="\t")
    assert not audit["authoritative_same_dataset_sample_gid_crosswalk"].astype(str).str.lower().isin({"true", "1"}).any()
    assert set(audit["r3_decision"]) == {"NO_SAME_DATASET_TYPED_SAMPLE_TO_GID_AUTHORITY_FOUND"}


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_pytest_summary_parser_accepts_wsl_and_powershell_encodings(tmp_path, encoding):
    log = tmp_path / f"pytest-{encoding}.log"
    log.write_text("26 passed in 5.79s\n", encoding=encoding)
    result = parse_test_result(log, 26)
    assert result["observed_passed"] == 26
    assert result["status"] == "PASS"
