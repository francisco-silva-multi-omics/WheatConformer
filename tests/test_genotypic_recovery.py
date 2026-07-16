from __future__ import annotations

from pathlib import Path

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
    parse_iwyp,
    parse_marker_by_sample,
    parse_sample_by_marker,
    qc_markers,
    qc_samples,
    vanraden_chunked,
)
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
