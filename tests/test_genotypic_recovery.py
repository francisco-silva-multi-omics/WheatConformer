from __future__ import annotations

from pathlib import Path
import gzip
import json
import sys

import numpy as np
import pandas as pd

from audit.recover_genotypic_gid_matches import scan_marker_by_sample_header
from genotype_recovery import (
    canonical_gid,
    genotype_call_to_dosage,
    rbf_from_linear_kernel,
    validate_kernel,
)
from server_genotype_recovery.build_platform_kernel import (
    baseline_kernel_comparison,
    main as build_platform_kernel_main,
    collapse_accepted_replicates,
    duplicate_call_concordance,
    load_accepted_identity_mappings,
    parse_dartag_numeric,
    parse_iwyp,
    parse_marker_by_sample,
    parse_sample_by_marker,
    qc_markers,
    qc_samples,
    resolve_matrix_path,
    vanraden_chunked,
)
from server_genotype_recovery.build_haplotype_kernel import (
    build_categorical_haplotype_kernel,
)
from server_genotype_recovery.prepare_canonical_catalog import prepare_catalog
from server_genotype_recovery.audit_candidate_support import main as audit_candidate_support
from server_training_pipeline.prepare_multitrait_kernel_registry import (
    load_recovered_genotype_candidates,
)


def test_gid_and_iupac_call_normalization() -> None:
    assert canonical_gid("GID008246851") == "GID8246851"
    assert canonical_gid("8246851.0") == "GID8246851"
    assert genotype_call_to_dosage("A", "A", "G") == 0
    assert genotype_call_to_dosage("R", "A", "G") == 1
    assert genotype_call_to_dosage("G/G", "A", "G") == 2
    assert genotype_call_to_dosage("-", "A", "G") == -1


def test_canonical_recovery_catalog_is_rebuilt_from_trial_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "metadata_outputs/all_trials_genotype_manifest_resolved.tsv"
    observations_path = (
        tmp_path / "integrated_database/canonical_trial_genotype_environment_plot_table.parquet"
    )
    hmp_path = tmp_path / "genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv"
    output_path = tmp_path / "audit/genotypic_recovery/canonical_genotype_catalog.csv"
    manifest_path.parent.mkdir(parents=True)
    observations_path.parent.mkdir(parents=True)
    hmp_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "CID": ["C1", "C2", "C1"],
            "SID": ["S1", "S2", "S1"],
            "fieldbook_gid": ["101", "102", "101"],
            "resolved_gid": ["101", "102", "101"],
            "panel_sample_id_expected": ["GID101", "GID102", "GID101"],
            "cross_name": ["A/B", "", "A/B"],
            "gid_source": ["fieldbook", "fieldbook", "fieldbook"],
            "fieldbook_glis_gid_conflict": [False, False, False],
            "pheno_gid_conflict": [False, False, False],
        }
    ).to_csv(manifest_path, sep="\t", index=False)
    pd.DataFrame(
        {"canonical_germplasm_key": ["GID101", "GID101", "GID102"]}
    ).to_parquet(observations_path, index=False)
    pd.DataFrame({"sample_id": ["GID101"]}).to_csv(hmp_path, sep="\t", index=False)

    provenance = prepare_catalog(
        root=tmp_path,
        manifest_path=manifest_path,
        canonical_observations_path=observations_path,
        hmp_order_path=hmp_path,
        output_path=output_path,
    )

    catalog = pd.read_csv(output_path)
    assert catalog["canonical_gid"].tolist() == ["GID101", "GID102"]
    assert catalog["canonical_observation_rows"].tolist() == [2, 1]
    assert catalog["marker_available_hmp_qc"].tolist() == [True, False]
    assert catalog["audit_genotypic_match"].isna().all()
    assert provenance["prior_audit_comparison_available"] is False


def test_canonical_recovery_catalog_imports_only_compatible_prior_flags(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "metadata_outputs/all_trials_genotype_manifest_resolved.tsv"
    prior_path = tmp_path / "audit/old/canonical_genotype_mapping_audited.csv"
    output_path = tmp_path / "audit/genotypic_recovery/canonical_genotype_catalog.csv"
    manifest_path.parent.mkdir(parents=True)
    prior_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "resolved_gid": ["101", "102"],
            "panel_sample_id_expected": ["GID101", "GID102"],
            "gid_source": ["fieldbook", "fieldbook"],
        }
    ).to_csv(manifest_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "canonical_gid": ["GID101", "GID102"],
            "raw_identifiers": ["{}", "{}"],
            "audit_genotypic_match": [True, False],
        }
    ).to_csv(prior_path, index=False)

    provenance = prepare_catalog(
        root=tmp_path,
        manifest_path=manifest_path,
        canonical_observations_path=None,
        hmp_order_path=None,
        output_path=output_path,
        explicit_prior_audit_catalog=prior_path,
    )

    catalog = pd.read_csv(output_path)
    assert catalog["audit_genotypic_match"].tolist() == [True, False]
    assert provenance["prior_audit_comparison_available"] is True
    assert provenance["prior_audit_catalog_path"] == str(prior_path)


def test_sample_by_marker_parser_reads_only_resolved_samples(tmp_path: Path) -> None:
    matrix_path = tmp_path / "sample_by_marker.txt"
    matrix_path.write_text(
        "# notice\n"
        "MarkerID\tm1:A>G\tm2:C>T\n"
        "S1\tA\tY\n"
        "S2\tG\tT\n"
        "UNMATCHED\tA\tC\n",
        encoding="utf-8",
    )
    matrix, gids, samples, markers, alleles = parse_sample_by_marker(
        matrix_path, {"S1": {"GID1"}, "S2": {"GID2"}}
    )
    assert gids == ["GID1", "GID2"]
    assert samples == ["S1", "S2"]
    assert markers == ["m1:A>G", "m2:C>T"]
    assert alleles == ["A/G", "C/T"]
    np.testing.assert_array_equal(matrix, np.array([[0, 1], [2, 2]], dtype=np.int8))


def test_marker_by_sample_parser_streams_selected_columns(tmp_path: Path) -> None:
    matrix_path = tmp_path / "marker_by_sample.txt"
    matrix_path.write_text(
        "MarkerID\tS1\tUNMATCHED\tS2\n"
        "m1:A>G\tA\tG\tG\n"
        "m2:C>T\tY\tC\tT\n",
        encoding="utf-8",
    )
    matrix, gids, samples, markers, alleles = parse_marker_by_sample(
        matrix_path, {"S1": {"GID1"}, "S2": {"GID2"}}
    )
    assert gids == ["GID1", "GID2"]
    assert samples == ["S1", "S2"]
    assert markers == ["m1:A>G", "m2:C>T"]
    assert alleles == ["A/G", "C/T"]
    np.testing.assert_array_equal(matrix, np.array([[0, 1], [2, 2]], dtype=np.int8))


def test_marker_by_sample_parser_certifies_adjudicated_matrix_hash(tmp_path: Path) -> None:
    matrix_path = tmp_path / "marker_by_sample.txt"
    matrix_path.write_text(
        "MarkerID\tS1\tSEEDYT&N350\n"
        "m1:A>G\tA\tG\n",
        encoding="utf-8",
    )
    import hashlib

    expected = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    matrix, gids, sources, _, _ = parse_marker_by_sample(
        matrix_path,
        {"S1": {"GID1"}},
        expected_sha256=expected,
        forced_sample_columns={2: ("GID2", "SEEDYT&N350", "SEEDYTN350")},
    )
    assert gids == ["GID1", "GID2"]
    assert sources == ["S1", "SEEDYT&N350"]
    try:
        parse_marker_by_sample(
            matrix_path, {"S1": {"GID1"}}, expected_sha256="0" * 64
        )
    except SystemExit as exc:
        assert "changed after identity adjudication" in str(exc)
    else:
        raise AssertionError("A marker matrix hash mismatch must fail closed")


def test_accepted_replicates_are_materialized_as_consensus() -> None:
    matrix = np.asarray(
        [
            [0, 1, -1, 0],
            [0, 1, 2, 2],
            [2, 2, 0, 0],
        ],
        dtype=np.int8,
    )
    collapsed, gids, sources, audit = collapse_accepted_replicates(
        matrix,
        ["GID1", "GID1", "GID2"],
        ["S1", "S2", "S3"],
        {"GID1": {"S1", "S2"}},
    )
    assert gids == ["GID2", "GID1"]
    assert sources == ["S3", "CONSENSUS:S1|S2"]
    np.testing.assert_array_equal(
        collapsed,
        np.asarray([[2, 2, 0, 0], [0, 1, 2, -1]], dtype=np.int8),
    )
    row = audit.iloc[0]
    assert row["unanimous_multisample_markers"] == 2
    assert row["single_observed_sample_markers"] == 1
    assert row["discordant_observed_markers_set_missing"] == 1


def test_accepted_identity_loader_requires_reconciled_evidence(tmp_path: Path) -> None:
    identity_dir = tmp_path / "identity"
    identity_dir.mkdir()
    matrix_path = tmp_path / "matrix.tsv"
    matrix_path.write_text("MarkerID\tS&1\nm1:A>G\tA\n", encoding="utf-8")
    import hashlib

    matrix_hash = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    pd.DataFrame(
        {
            "trial_gid": ["GID1"],
            "panel_id": ["RECOVERY"],
            "sample_id": ["S&1"],
            "normalized_sample_id": ["S1"],
            "marker_matrix_path": [str(matrix_path.resolve())],
            "marker_matrix_sha256": [matrix_hash],
            "marker_matrix_axis_index": [1],
            "marker_axis_match_count": [1],
            "classification": ["accepted_unique_identity"],
            "direct_marker_assignment_ready": [True],
            "existing_certified_in_panel": [False],
            "existing_certified_in_any_panel": [False],
        }
    ).to_csv(
        identity_dir / "marker_identity_candidate_paths.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        columns=[
            "trial_gid",
            "panel_id",
            "sample_id_left",
            "sample_id_right",
            "overlap_pass",
            "concordance_pass",
            "pair_status",
        ]
    ).to_csv(
        identity_dir / "marker_identity_pairwise_concordance.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        {
            "check": [
                "classification_evidence_preserved",
                "pairwise_concordance_evidence_preserved",
                "regulatory_overlay_is_gated",
            ],
            "status": ["PASS", "PASS", "PASS"],
        }
    ).to_csv(identity_dir / "marker_identity_validation.tsv", sep="\t", index=False)
    (identity_dir / "marker_identity_adjudication_provenance.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "classification_evidence_reused": True,
                "classification_or_concordance_modified": False,
                "marker_calls_read": False,
                "phenotype_values_read": False,
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
            }
        ),
        encoding="utf-8",
    )
    accepted, columns, replicates, observed_hash, _ = load_accepted_identity_mappings(
        identity_dir=identity_dir,
        panel_ids={"RECOVERY"},
        canonical_ids={"GID1"},
        matrix_path=matrix_path,
    )
    assert accepted["trial_gid"].tolist() == ["GID1"]
    assert columns == {1: ("GID1", "S&1", "S1")}
    assert replicates == {}
    assert observed_hash == matrix_hash


def test_identity_recovered_kernel_must_retain_and_match_baseline(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.npy"
    order_path = tmp_path / "order.tsv"
    baseline = np.asarray(
        [[1.0, 0.2, 0.3], [0.2, 1.0, 0.4], [0.3, 0.4, 1.0]],
        dtype=np.float32,
    )
    np.save(baseline_path, baseline)
    pd.DataFrame({"sample_id": ["GID1", "GID2", "GID3"]}).to_csv(
        order_path, sep="\t", index=False
    )
    candidate_gids = ["GID4", "GID2", "GID1", "GID3"]
    candidate = np.eye(4, dtype=np.float32)
    positions = [candidate_gids.index(gid) for gid in ["GID1", "GID2", "GID3"]]
    candidate[np.ix_(positions, positions)] = baseline
    comparison = baseline_kernel_comparison(
        baseline_kernel_path=baseline_path,
        baseline_order_path=order_path,
        candidate_kernel=candidate,
        candidate_gids=candidate_gids,
        minimum_correlation=0.99,
    )
    assert comparison.iloc[0]["status"] == "PASS"
    assert comparison.iloc[0]["new_candidate_gids"] == 1


def test_identity_recovered_builder_writes_isolated_gated_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    matrix_path = tmp_path / "GENOTYPIC_DATA/seeds/matrix.tsv"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(
        "MarkerID\tGID1\tGID2\tGID3\tS&4\n"
        "m1:A>G\tA\tA\tG\tG\n"
        "m2:A>G\tA\tG\tG\tA\n"
        "m3:A>G\tG\tG\tA\tA\n"
        "m4:A>G\tA\tG\tA\tG\n",
        encoding="utf-8",
    )
    import hashlib

    matrix_hash = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    catalog_path = tmp_path / "audit/genotypic_recovery/canonical_genotype_catalog.csv"
    catalog_path.parent.mkdir(parents=True)
    pd.DataFrame({"canonical_gid": ["GID1", "GID2", "GID3", "GID4"]}).to_csv(
        catalog_path, index=False
    )
    identity_dir = tmp_path / "identity"
    identity_dir.mkdir()
    pd.DataFrame(
        {
            "trial_gid": ["GID4"],
            "panel_id": ["RECOVERY"],
            "sample_id": ["S&4"],
            "normalized_sample_id": ["S4"],
            "marker_matrix_path": [str(matrix_path.resolve())],
            "marker_matrix_sha256": [matrix_hash],
            "marker_matrix_axis_index": [4],
            "marker_axis_match_count": [1],
            "classification": ["accepted_unique_identity"],
            "direct_marker_assignment_ready": [True],
            "existing_certified_in_panel": [False],
            "existing_certified_in_any_panel": [False],
        }
    ).to_csv(
        identity_dir / "marker_identity_candidate_paths.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        columns=[
            "trial_gid",
            "panel_id",
            "sample_id_left",
            "sample_id_right",
            "overlap_pass",
            "concordance_pass",
            "pair_status",
        ]
    ).to_csv(
        identity_dir / "marker_identity_pairwise_concordance.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        {
            "check": [
                "classification_evidence_preserved",
                "pairwise_concordance_evidence_preserved",
                "regulatory_overlay_is_gated",
            ],
            "status": ["PASS", "PASS", "PASS"],
        }
    ).to_csv(identity_dir / "marker_identity_validation.tsv", sep="\t", index=False)
    (identity_dir / "marker_identity_adjudication_provenance.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "classification_evidence_reused": True,
                "classification_or_concordance_modified": False,
                "marker_calls_read": False,
                "phenotype_values_read": False,
                "outer_test_metrics_read": False,
                "final_holdout_outcomes_read": False,
            }
        ),
        encoding="utf-8",
    )
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline_kernel = baseline_dir / "K_LINEAR.npy"
    baseline_order = baseline_dir / "K_order.tsv"
    np.save(
        baseline_kernel,
        np.asarray(
            [[1.0, 0.1, 0.4], [0.1, 1.0, -0.2], [0.4, -0.2, 1.0]],
            dtype=np.float32,
        ),
    )
    pd.DataFrame({"sample_id": ["GID1", "GID2", "GID3"]}).to_csv(
        baseline_order, sep="\t", index=False
    )
    out_dir = tmp_path / "candidate"
    prefix = "K_G_TEST_IDENTITY_V2"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_platform_kernel",
            "--root",
            str(tmp_path),
            "--platform",
            "seeds_dartseq",
            "--matrix",
            str(matrix_path),
            "--canonical-catalog",
            str(catalog_path.relative_to(tmp_path)),
            "--identity-adjudication-dir",
            str(identity_dir),
            "--identity-panel",
            "RECOVERY",
            "--baseline-kernel",
            str(baseline_kernel),
            "--baseline-order",
            str(baseline_order),
            "--minimum-baseline-kernel-correlation",
            "-1",
            "--out-dir",
            str(out_dir),
            "--prefix",
            prefix,
            "--save-dosage",
        ],
    )
    build_platform_kernel_main()
    order = pd.read_csv(out_dir / f"{prefix}_sample_order.tsv", sep="\t")
    assert set(order["sample_id"]) == {"GID1", "GID2", "GID3", "GID4"}
    summary = pd.read_csv(
        out_dir / f"{prefix}_identity_recovery_summary.tsv", sep="\t"
    ).set_index("metric")["value"]
    assert int(summary["certified_new_to_any_prior_panel_gids"]) == 1
    registry = pd.read_csv(out_dir / f"{prefix}_registry_fragment.tsv", sep="\t")
    assert not registry["enabled_default"].any()
    assert (out_dir / f"{prefix}_artifacts.sha256").is_file()


def test_marker_by_sample_audit_uses_real_matrix_header(tmp_path: Path) -> None:
    genotypic_root = tmp_path / "GENOTYPIC_DATA"
    dataset = genotypic_root / "Seeds"
    dataset.mkdir(parents=True)
    matrix_path = dataset / "matrix.txt"
    matrix_path.write_text("MarkerID\tS1\tS2\nm1:A>G\tA\tG\n", encoding="utf-8")
    rows, status = scan_marker_by_sample_header(matrix_path, genotypic_root)
    assert status == "parsed_marker_by_sample_header_axis"
    assert [row["sample_identifier"] for row in rows] == ["S1", "S2"]
    assert all(row["matrix_backed"] for row in rows)


def test_iwyp_parser_uses_gid_preamble_and_marker_alleles(tmp_path: Path) -> None:
    matrix_path = tmp_path / "iwyp.txt"
    prefix = [""] * 10
    rows = [
        ["note", *prefix, "", ""],
        ["entry", *prefix, "1", "2"],
        ["GID", *prefix, "101", "102"],
        ["rs#", "alleles", "chrom", "pos", "strand", "assembly", "center", "prot", "assay", "panel", "QC", "H1", "H2"],
        ["m1", "A/G", "1A", "1", "+", "NA", "NA", "NA", "NA", "NA", "NA", "A", "G"],
        ["m2", "C/T", "1A", "2", "+", "NA", "NA", "NA", "NA", "NA", "NA", "Y", "T"],
    ]
    matrix_path.write_text("\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8")
    matrix, gids, _, markers, _ = parse_iwyp(matrix_path, {"GID101", "GID102"})
    assert gids == ["GID101", "GID102"]
    assert markers == ["m1", "m2"]
    np.testing.assert_array_equal(matrix, np.array([[0, 1], [2, 2]], dtype=np.int8))


def test_dartag_parser_unions_batches_and_records_duplicate_concordance(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text(
        "GID,101,102\n"
        "m1,0,2\n"
        "m2,1,-\n",
        encoding="utf-8",
    )
    second.write_text(
        "Subject_ID,S101,S103\n"
        "GID,101,103\n"
        "m2,1,2\n"
        "m3,2,0\n",
        encoding="utf-8",
    )

    matrix, gids, sources, markers, alleles = parse_dartag_numeric(
        [first, second], {"GID101", "GID102", "GID103"}
    )

    assert gids == ["GID101", "GID102", "GID101", "GID103"]
    assert markers == ["m1", "m2", "m3"]
    assert alleles == ["numeric_0_1_2"] * 3
    np.testing.assert_array_equal(
        matrix,
        np.array([[0, 1, -1], [2, -1, -1], [-1, 1, 2], [-1, 2, 0]], dtype=np.int8),
    )
    concordance = duplicate_call_concordance(matrix, gids, sources)
    assert concordance.loc[0, "overlapping_observed_markers"] == 1
    assert concordance.loc[0, "call_concordance"] == 1.0


def test_dartag_parser_reads_gzip_and_default_path_discovery_is_unique(tmp_path: Path) -> None:
    expected = Path("GENOTYPIC_DATA/expected/DArTAG_numeric.csv")
    actual = tmp_path / "GENOTYPIC_DATA/relocated/DArTAG_numeric.csv.gz"
    actual.parent.mkdir(parents=True)
    with gzip.open(actual, "wt", encoding="utf-8") as handle:
        handle.write("GID,101,102\nm1,0,2\nm2,1,1\n")

    resolved = resolve_matrix_path(
        tmp_path,
        expected,
        discover_by_basename=True,
        allow_gzip=True,
    )
    matrix, gids, _, markers, _ = parse_dartag_numeric(
        [resolved], {"GID101", "GID102"}
    )

    assert resolved == actual.resolve()
    assert gids == ["GID101", "GID102"]
    assert markers == ["m1", "m2"]
    np.testing.assert_array_equal(matrix, np.array([[0, 1], [2, 1]], dtype=np.int8))


def test_categorical_haplotype_kernel_is_psd_and_not_dosage_coerced() -> None:
    frame = pd.DataFrame(
        {
            "GID": ["101", "102", "103", "104"],
            "EYT": ["A", "A", "B", "B"],
            "1A.1": ["ACT", "ACT", "GGA", "GGA"],
            "1A.2": ["CC", "TT", "CC", "TT"],
            "1B.1": ["NA", "AA", "GG", "AA"],
        }
    )

    kernel, gids, sample_qc, _, block_qc = build_categorical_haplotype_kernel(
        frame,
        sample_missing_max=0.5,
        block_missing_max=0.5,
        state_frequency_min=0.20,
    )

    assert gids == ["GID101", "GID102", "GID103", "GID104"]
    assert sample_qc["selected_for_kernel"].all()
    assert block_qc["retained"].all()
    assert validate_kernel(kernel, name="haplotype")["finite"] == "true"
    np.testing.assert_allclose(np.diag(kernel).mean(), 1.0, atol=1e-6)


def test_qc_vanraden_and_rbf_produce_certified_kernels() -> None:
    matrix = np.array(
        [
            [0, 0, 2, 0],
            [0, 1, 2, 2],
            [2, 2, 0, 2],
            [2, 1, 0, 0],
        ],
        dtype=np.int8,
    )
    matrix, gids, _, sample_qc, _ = qc_samples(
        matrix,
        ["GID1", "GID2", "GID3", "GID4"],
        ["S1", "S2", "S3", "S4"],
        missing_max=0.2,
        heterozygosity_max=0.5,
    )
    assert len(gids) == 4
    assert sample_qc["selected_for_kernel"].all()
    matrix, marker_qc, frequency = qc_markers(
        matrix,
        ["m1", "m2", "m3", "m4"],
        ["A/G"] * 4,
        missing_max=0.2,
        maf_min=0.01,
        heterozygosity_max=0.5,
    )
    assert marker_qc["retained"].all()
    linear, _ = vanraden_chunked(matrix, frequency, chunk_size=2)
    rbf, _ = rbf_from_linear_kernel(linear)
    assert validate_kernel(linear, name="linear")["finite"] == "true"
    assert validate_kernel(rbf, name="rbf")["finite"] == "true"
    np.testing.assert_allclose(np.diag(linear).mean(), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.diag(rbf), 1.0, atol=1e-6)


def test_recovered_genotype_manifest_is_loaded_as_partial_expert(tmp_path: Path) -> None:
    kernel_path = tmp_path / "K.npy"
    order_path = tmp_path / "order.tsv"
    manifest_path = tmp_path / "manifest.tsv"
    np.save(kernel_path, np.eye(2, dtype=np.float32))
    pd.DataFrame({"sample_id": ["GID1", "GID2"]}).to_csv(order_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "kernel": ["K_G_80K_LINEAR"],
            "biological_role": ["80k_marker_linear"],
            "kernel_path": [kernel_path.name],
            "order_path": [order_path.name],
            "source_id_col": ["sample_id"],
            "eligible_traits": ["*"],
            "enabled_default": [True],
            "interaction_enabled": [True],
            "rank": [2],
            "minimum_ledger_coverage": [0.01],
        }
    ).to_csv(manifest_path, sep="\t", index=False)
    candidates = load_recovered_genotype_candidates(
        manifest_path,
        root=tmp_path,
        base_g_order=pd.DataFrame({"sample_id": ["GID1", "GID2", "GID3"]}),
    )
    assert candidates[0]["axis"] == "genotype"
    assert candidates[0]["enabled_default"] is True
    assert candidates[0]["target_order"]["sample_id"].tolist() == ["GID1", "GID2", "GID3"]


def test_candidate_support_audit_reads_ids_and_inner_support_only(
    tmp_path: Path, monkeypatch
) -> None:
    hmp_dir = tmp_path / "genotype_panels/hmp"
    gbs_dir = tmp_path / "genotype_panels/gbs_sawyt"
    recovered_dir = tmp_path / "genotype_panels/recovered/dartag"
    hmp_dir.mkdir(parents=True)
    gbs_dir.mkdir(parents=True)
    recovered_dir.mkdir(parents=True)
    np.save(hmp_dir / "K_HMP.QCfiltered.meanDiag1.npy", np.eye(3, dtype=np.float32))
    pd.DataFrame({"sample_id": ["GID1", "GID2", "GID3"]}).to_csv(
        hmp_dir / "hmp_K_sample_order.QCfiltered.tsv", sep="\t", index=False
    )
    np.save(gbs_dir / "K_GBS_SAWYT.QCfiltered.npy", np.eye(2, dtype=np.float32))
    pd.DataFrame({"sample_id": ["GID4", "GID5"]}).to_csv(
        gbs_dir / "gbs_sawyt_K_sample_order.QCfiltered.tsv", sep="\t", index=False
    )
    np.save(recovered_dir / "K_G_DARTAG_LINEAR.npy", np.eye(3, dtype=np.float32))
    pd.DataFrame({"sample_id": ["GID1", "GID4", "GID6"]}).to_csv(
        recovered_dir / "K_G_DARTAG_sample_order.tsv", sep="\t", index=False
    )
    recovered_manifest = tmp_path / "genotype_panels/recovered/recovered_genotype_kernel_manifest.tsv"
    pd.DataFrame(
        [
            {
                "kernel": "K_G_DARTAG_LINEAR",
                "biological_role": "dartag_linear",
                "kernel_path": "genotype_panels/recovered/dartag/K_G_DARTAG_LINEAR.npy",
                "order_path": "genotype_panels/recovered/dartag/K_G_DARTAG_sample_order.tsv",
                "source_id_col": "sample_id",
            }
        ]
    ).to_csv(recovered_manifest, sep="\t", index=False)
    ledger = pd.DataFrame(
        {
            "panel_sample_id": ["GID1", "GID4", "GID2", "GID5", "GID3", "GID6"],
            "env_kernel_id": ["E1", "E1", "E2", "E2", "E3", "E3"],
            "trait_name_canonical": ["T"] * 6,
            "cycle": ["2020"] * 6,
            "country": ["X"] * 6,
        }
    )
    ledger_path = tmp_path / "ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)
    entity_manifest = pd.DataFrame(
        [
            {
                "scenario": "unseen_environments",
                "outer_fold": 0,
                "inner_fold": 0,
                "axis": "environment",
                "partition": "outer_test",
                "entity_id": "E3",
            },
            {
                "scenario": "unseen_environments",
                "outer_fold": 0,
                "inner_fold": 0,
                "axis": "environment",
                "partition": "inner_validation",
                "entity_id": "E2",
            },
        ]
    )
    entity_path = tmp_path / "entities.tsv"
    entity_manifest.to_csv(entity_path, sep="\t", index=False)
    out_dir = tmp_path / "screen"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_candidate_support",
            "--root",
            str(tmp_path),
            "--ledger",
            str(ledger_path),
            "--entity-manifest",
            str(entity_path),
            "--recovered-manifest",
            str(recovered_manifest),
            "--out-dir",
            str(out_dir),
            "--minimum-training-ids",
            "1",
        ],
    )

    audit_candidate_support()

    provenance = json.loads(
        (out_dir / "genomic_candidate_screen_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["status"] == "PASS"
    assert provenance["phenotype_values_read"] is False
    assert provenance["outer_test_metrics_read"] is False
    interpretation = provenance["interpretation_contract"]
    assert interpretation["quantitative_screen_scope"] == (
        "standalone_K_G_baseline_inclusion_only"
    )
    assert interpretation["retain_certified_panels_for_regulatory_projection"] is True
    assert interpretation["pedigree_propagated_embedding_status"] == "imputed_pedigree"
    assert interpretation["pedigree_propagation_requires_confidence_gate"] is True
    assert interpretation["pedigree_propagation_equivalent_to_observed_sequence"] is False
    policy = pd.read_csv(
        out_dir / "genomic_candidate_regulatory_retention_policy.tsv", sep="\t", dtype=str
    )
    policy_values = dict(zip(policy["policy"], policy["value"]))
    assert policy_values["direct_regulatory_embedding_status"] == (
        "observed_marker_supported_sequence"
    )
    assert policy_values["pedigree_propagated_embedding_status"] == "imputed_pedigree"
    plan = pd.read_csv(out_dir / "genomic_candidate_ablation_plan.tsv", sep="\t")
    dartag = plan[plan["architecture"].eq("existing_plus_K_G_DARTAG_LINEAR")].iloc[0]
    assert dartag["status"] == "ready"
    combined = plan[
        plan["architecture"].eq("existing_plus_all_supported_linear_candidates")
    ].iloc[0]
    assert combined["screen_phase"] == "phase_2_combination_after_individual"
    assert combined["status"] == "deferred_until_individual_candidates_selected"


def test_genomic_inner_screen_uses_matched_candidate_seeds() -> None:
    script = Path("scripts/run_genomic_expert_inner_screen.sh").read_text(encoding="utf-8")

    assert "GENOMIC_SCREEN_SEED_BASE:-61001" in script
    assert "architecture_index * 1000" not in script
