"""Integration assertions for the versioned Phase-3G R2 corrective release."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("STAGE1_V2_DATA_ROOT", CODE_ROOT)).resolve()
R2 = DATA_ROOT / "audit" / "v2" / "phase3g_all_panel_genotype_linkage_audit_v2"
R1 = DATA_ROOT / "audit" / "v2" / "phase3g_all_panel_genotype_linkage_audit_v1"
pytestmark = pytest.mark.skipif(not (R2 / "phase3g_r2_build_summary.json").exists(), reason="R2 delivery not built")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def test_r2_build_summary_exact_counts() -> None:
    summary = json.loads((R2 / "phase3g_r2_build_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS_R2_ARTIFACT_BUILD"
    assert summary["hibap"]["matrix_columns"] == 148
    assert summary["hibap"]["entry_to_ent_agreement"] == 148
    assert summary["hibap"]["matrix_header_to_sidecar_sample35k_agreement"] == 0
    assert summary["hibap"]["matrix_sidecar_gid_concordant"] == 148
    assert summary["hibap"]["gid_conflicts"] == 0
    assert summary["hibap"]["unique_entry_numbers"] == 147
    assert summary["hibap"]["unique_linked_gids"] == 145
    assert summary["global_accepted_sample_instances"] == 123_169
    assert summary["global_accepted_unique_gids"] == 94_897


def test_hibap_counterexample_duplicates_and_repeated_gids_retained() -> None:
    frame = pd.read_parquet(R2 / "hibap_sample_instance_ledger.parquet")
    hibap3 = frame.loc[frame["raw_matrix_header"].eq("Hibap3")].iloc[0]
    assert hibap3["matrix_entry_number"] == "3"
    assert hibap3["sidecar_ent"] == "3"
    assert hibap3["accepted_canonical_gid"] == "GID775"
    assert hibap3["sidecar_sample_35k"] == "Hibap91"
    entry109 = frame.loc[frame["matrix_entry_number"].eq("109")]
    assert set(entry109["raw_matrix_header"]) == {"Hibap109", "Hibap109-2"}
    assert entry109["sample_instance_key"].nunique() == 2
    assert len(frame.loc[frame["accepted_canonical_gid"].duplicated(False)]) == 6


def test_hibap_replicate_concordance_exact() -> None:
    frame = pd.read_csv(R2 / "hibap_replicate_concordance_report.tsv", sep="\t")
    observed = {
        row.canonical_gid: (row.comparable_nonmissing_markers, row.matching_calls, row.discordant_calls, row.concordance_proportion)
        for row in frame.itertuples(index=False)
    }
    assert observed["GID6056237"][:3] == (9005, 8964, 41)
    assert observed["GID6176368"][:3] == (9126, 9105, 21)
    assert observed["GID6489912"][:3] == (8471, 8375, 96)
    assert observed["GID6056237"][3] == pytest.approx(0.995446973903387)


def test_80k_axes_duplicates_and_candidate_only_policy() -> None:
    axes = pd.read_csv(R2 / "dartseq80k_sample_axis_validation.tsv", sep="\t")
    assert axes["certification_status"].eq("PASS_SAMPLE_AXIS").all()
    assert dict(zip(axes["population"], axes["observed_physical_sample_columns"])) == {
        "hexaploid": 56_342,
        "tetraploid": 18_946,
        "wheat_recall": 15_666,
        "wild_relative": 3_903,
    }
    duplicates = pd.read_csv(R2 / "dartseq80k_duplicate_column_report.tsv", sep="\t")
    assert set(duplicates["raw_sample_label"]) == {"SEEDSPE86", "SEEDSPE87"}
    assert duplicates["physical_occurrence_count"].eq(2).all()
    candidates = pd.read_parquet(R2 / "dartseq80k_cross_panel_candidate_ledger.parquet")
    assert len(candidates) == 43_570
    assert candidates["accepted_canonical_gid"].fillna("").eq("").all()
    assert candidates["candidate_disposition"].eq("CANDIDATE_CROSS_PANEL_LABEL_MATCH").all()


def test_80k_csv_flapjack_certification_passes() -> None:
    frame = pd.read_csv(R2 / "dartseq80k_csv_flapjack_concordance.tsv", sep="\t")
    assert len(frame) == 8
    assert frame["certification_status"].str.startswith("PASS").all()
    paired = frame[frame["csv_source_file"].fillna("").ne("")]
    assert paired["sample_order_relation"].eq("EXACT_IDENTITY").all()
    assert paired["marker_order_mismatches"].eq(0).all()
    encoding = pd.read_csv(R2 / "dartseq80k_encoding_validation.tsv", sep="\t")
    assert len(encoding) == 8
    assert encoding["status"].str.startswith("PASS").all()
    assert encoding["unexpected_csv_tokens"].fillna("").eq("").all()
    assert encoding["unexpected_flapjack_tokens"].fillna("").eq("").all()


def test_old_to_new_population_change_is_exact() -> None:
    old = json.loads((R1 / "phase3g_audit_summary.json").read_text(encoding="utf-8"))
    new = json.loads((R2 / "phase3g_audit_summary.json").read_text(encoding="utf-8"))
    assert (old["accepted_panel_samples"], new["accepted_panel_samples"]) == (123_021, 123_169)
    assert (old["unique_accepted_gids_all_panels"], new["unique_accepted_gids_all_panels"]) == (94_824, 94_897)
    assert (old["linkage"]["all_panel_union_stage1_gids"], new["linkage"]["all_panel_union_stage1_gids"]) == (10_716, 10_744)
    assert (old["linkage"]["all_panel_union_stage1_rows"], new["linkage"]["all_panel_union_stage1_rows"]) == (3_140_500, 3_145_436)
    assert (old["linkage"]["all_panel_union_selected_gids"], new["linkage"]["all_panel_union_selected_gids"]) == (10_694, 10_722)
    assert (old["linkage"]["all_panel_union_selected_rows"], new["linkage"]["all_panel_union_selected_rows"]) == (2_239_318, 2_242_863)


def test_core_regeneration_is_byte_deterministic() -> None:
    replay = R2 / "determinism_replay"
    candidates = [path for path in replay.rglob("*") if path.is_file() and path.name != "phase3g_audit_summary.json"]
    assert len(candidates) == 90
    for replay_path in candidates:
        relative = replay_path.relative_to(replay)
        assert digest(R2 / relative) == digest(replay_path), relative.as_posix()
