from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import struct
import zlib
from pathlib import Path
from typing import Any

import pandas as pd
from lxml import etree


RELEASE_ID = "P5PPR_20260822_V1_274E41DF"
DEFAULT_OUTPUT = Path("audit/v2/phase5_panel_prerequisite_recovery_v1")
DOWNLOADS = DEFAULT_OUTPUT / "downloads"
PASSPORT_FILE = DOWNLOADS / "41467_2020_18404_MOESM5_ESM.xlsx"
CIMMYT_ARCHIVE = DOWNLOADS / "CIMMYT-2013-2018.hmp.txt.zip.gz"
EYT_ARTICLE_XML = DOWNLOADS / "fgene-11-589490.xml"
AXIS_LEDGER = Path(
    "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/"
    "dartseq80k_sample_instance_ledger.parquet"
)
STAGE1_INDEX = Path(
    "audit/v2/phase5_split_bound_kernel_validation_v2/indices/"
    "canonical_phase5_observation_index.parquet"
)
EYT_CALLS = Path(
    "GENOTYPIC_DATA/Haplotype-based_genome-wide_association_study/"
    "Haplotype_blocks_EYT2011-12_to_EYT2017-18.csv"
)

EXPECTED_PASSPORT_SHA256 = (
    "83cb91b3012ef5b38e504e57681c2ff30fd4651967378501936f9cae1206ec66"
)
EXPECTED_CIMMYT_MD5 = "61865f5b1002f5a6e14dc555ba700663"
EXPECTED_CIMMYT_BYTES = 862_429_679
EXPECTED_EYT_XML_SHA256 = (
    "5a4e216360c85c235894f49f9381cd2dc98ecca149f776523973fad85af1bf19"
)
HAPMAP_METADATA_COLUMNS = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Perform a phenotype-blind bounded recovery of CIMMYT pre-QC calls, "
            "EYT source provenance, and the DArTseq-80K sample-to-GID crosswalk"
        )
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def normalize_sample_id(value: object) -> str:
    return re.sub(r"\s+", "", str(value).strip()).upper()


def canonicalize_gid(value: object) -> tuple[str, str]:
    text = str(value).strip().upper()
    if text.endswith(".0"):
        text = text[:-2]
    match = re.fullmatch(r"(?:GID)?(\d+)", text)
    if not match:
        raise ValueError(f"Invalid typed GID: {value!r}")
    numeric = match.group(1)
    return numeric, f"GID{numeric}"


def require_file(root: Path, relative: Path) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def parse_zip64_sizes(extra: bytes) -> tuple[int | None, int | None]:
    cursor = 0
    while cursor + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, cursor)
        payload = extra[cursor + 4 : cursor + 4 + size]
        if header_id == 0x0001 and len(payload) >= 16:
            return struct.unpack_from("<QQ", payload, 0)
        cursor += 4 + size
    return None, None


def read_gzip_wrapped_zip_hapmap_header(path: Path) -> dict[str, Any]:
    """Read only the first HapMap line from the nested gzip/ZIP source."""
    with gzip.open(path, "rb") as outer:
        local = outer.read(30)
        if len(local) != 30 or local[:4] != b"PK\x03\x04":
            raise ValueError("CIMMYT archive is not a gzip-wrapped ZIP local member")
        flags = struct.unpack_from("<H", local, 6)[0]
        method = struct.unpack_from("<H", local, 8)[0]
        filename_length, extra_length = struct.unpack_from("<HH", local, 26)
        member_name = outer.read(filename_length).decode("utf-8")
        extra = outer.read(extra_length)
        uncompressed_bytes, compressed_bytes = parse_zip64_sizes(extra)
        if flags & 0x1:
            raise ValueError("Encrypted ZIP member is unsupported")
        if method != 8:
            raise ValueError(f"Expected DEFLATE ZIP member, observed method={method}")

        inflater = zlib.decompressobj(-zlib.MAX_WBITS)
        output = bytearray()
        compressed_bytes_read = 0
        while b"\n" not in output:
            chunk = outer.read(256 * 1024)
            if not chunk:
                raise ValueError("HapMap header newline not found in nested archive")
            compressed_bytes_read += len(chunk)
            output.extend(inflater.decompress(chunk))
            if len(output) > 64 * 1024 * 1024:
                raise ValueError("HapMap header exceeded the bounded 64 MiB limit")

    header = bytes(output).split(b"\n", 1)[0].rstrip(b"\r").split(b"\t")
    if len(header) <= HAPMAP_METADATA_COLUMNS:
        raise ValueError("Recovered HapMap header has no sample columns")
    expected_prefix = [
        b"rs#",
        b"alleles",
        b"chrom",
        b"pos",
        b"strand",
        b"assembly#",
        b"center",
        b"protLSID",
        b"assayLSID",
        b"panelLSID",
        b"QCcode",
    ]
    if header[:HAPMAP_METADATA_COLUMNS] != expected_prefix:
        raise ValueError("Unexpected HapMap metadata columns")
    sample_ids = [value.decode("utf-8") for value in header[HAPMAP_METADATA_COLUMNS:]]
    return {
        "member_name": member_name,
        "member_uncompressed_bytes": uncompressed_bytes,
        "member_compressed_bytes": compressed_bytes,
        "compressed_bytes_read_for_header": compressed_bytes_read,
        "header_bytes": sum(len(value) for value in header) + len(header) - 1,
        "sample_ids": sample_ids,
    }


def read_passport_workbook(path: Path) -> tuple[pd.DataFrame, set[str], set[str]]:
    specifications = (
        ("Hexaploid", "Sample ID", "hexaploid"),
        ("Tetraploid", "SampleID", "tetraploid"),
        ("CWR", "SampleID", "wild_relative"),
    )
    frames: list[pd.DataFrame] = []
    for sheet, sample_column, population in specifications:
        source = pd.read_excel(path, sheet_name=sheet, dtype=str)
        if sample_column not in source or "GID" not in source:
            raise ValueError(f"Missing typed identity columns in sheet {sheet}")

        def text_column(*names: str) -> pd.Series:
            for name in names:
                if name in source:
                    return source[name].fillna("").astype(str)
            return pd.Series("", index=source.index, dtype=str)

        rows = pd.DataFrame(
            {
                "source_sheet": sheet,
                "source_population": population,
                "raw_sample_id": source[sample_column].astype(str),
                "raw_gid": source["GID"].astype(str),
                "institution": text_column("Institution"),
                "pedigree": text_column("Pedigree"),
                "taxonomy": text_column("Taxonomy"),
                "biological_status": text_column(
                    "Biological status", "Biological Status"
                ),
                "doi": text_column("DOI"),
            }
        )
        frames.append(rows)

    passport = pd.concat(frames, ignore_index=True)
    passport["normalized_sample_id"] = passport.raw_sample_id.map(normalize_sample_id)
    canonical = passport.raw_gid.map(canonicalize_gid)
    passport["external_gid_numeric"] = canonical.map(lambda value: value[0])
    passport["canonical_gid"] = canonical.map(lambda value: value[1])
    if passport.normalized_sample_id.eq("").any():
        raise ValueError("Blank sample identifier in 80K passport workbook")
    if passport.normalized_sample_id.duplicated().any():
        duplicates = passport.loc[
            passport.normalized_sample_id.duplicated(False), "normalized_sample_id"
        ].unique()
        raise ValueError(f"Duplicate 80K passport sample identifiers: {duplicates[:10]}")

    recall = pd.read_excel(path, sheet_name="Recall_H_T_WR", dtype=str)
    if "label" not in recall:
        raise ValueError("Recall_H_T_WR sheet has no label column")
    recall_ids = set(recall.label.dropna().map(normalize_sample_id))
    unknown_recall = recall_ids - set(passport.normalized_sample_id)
    return passport, recall_ids, unknown_recall


def build_80k_bindings(
    axis: pd.DataFrame, passport: pd.DataFrame, recall_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "panel_id",
        "population",
        "representation",
        "source_file",
        "physical_column_index",
        "raw_sample_label",
        "sample_instance_key",
    }
    missing = required - set(axis.columns)
    if missing:
        raise ValueError(f"80K axis ledger missing columns: {sorted(missing)}")

    local = axis.copy()
    local["normalized_sample_id"] = local.raw_sample_label.map(normalize_sample_id)
    passport_columns = [
        "source_sheet",
        "source_population",
        "normalized_sample_id",
        "external_gid_numeric",
        "canonical_gid",
        "institution",
        "pedigree",
        "taxonomy",
        "biological_status",
        "doi",
    ]
    bound = local.merge(
        passport[passport_columns],
        on="normalized_sample_id",
        how="left",
        validate="many_to_one",
    )
    bound["recall_membership_documented"] = bound.normalized_sample_id.isin(recall_ids)
    expected_population = {
        "hexaploid": "hexaploid",
        "tetraploid": "tetraploid",
        "wild_relative": "wild_relative",
    }
    non_recall = bound.population.ne("wheat_recall")
    bound["source_population_matches_axis"] = True
    bound.loc[non_recall, "source_population_matches_axis"] = bound.loc[
        non_recall, "population"
    ].map(expected_population).eq(bound.loc[non_recall, "source_population"])
    # The published recall-membership sheet is incomplete relative to the matrix.
    # An exact same-study matrix label plus typed passport row remains authoritative;
    # recall membership is retained as a separate consistency diagnostic.
    bound.loc[bound.population.eq("wheat_recall"), "source_population_matches_axis"] = (
        bound.loc[bound.population.eq("wheat_recall"), "source_population"].notna()
    )
    bound["typed_identity_exact"] = (
        bound.canonical_gid.notna() & bound.source_population_matches_axis
    )

    sample = (
        bound.groupby(
            ["panel_id", "population", "normalized_sample_id"],
            dropna=False,
            as_index=False,
        )
        .agg(
            raw_sample_id=("raw_sample_label", "first"),
            source_population=("source_population", "first"),
            external_gid_numeric=("external_gid_numeric", "first"),
            canonical_gid=("canonical_gid", "first"),
            matrix_representations=("representation", "nunique"),
            matrix_axis_instances=("sample_instance_key", "nunique"),
            recall_membership_documented=("recall_membership_documented", "max"),
            typed_identity_exact=("typed_identity_exact", "min"),
        )
    )
    gid_sample_counts = (
        sample[sample.typed_identity_exact]
        .groupby("canonical_gid").normalized_sample_id.nunique()
    )
    sample["samples_per_typed_gid"] = sample.canonical_gid.map(gid_sample_counts).fillna(0).astype(int)
    sample["identity_class"] = "conflicting_or_missing_identity"
    exact = sample.typed_identity_exact
    sample.loc[exact & sample.samples_per_typed_gid.eq(1), "identity_class"] = (
        "accepted_unique_identity"
    )
    sample.loc[exact & sample.samples_per_typed_gid.gt(1), "identity_class"] = (
        "accepted_identity_replicate_set_pending_concordance"
    )
    sample["direct_genotype_assignment_ready"] = sample.identity_class.eq(
        "accepted_unique_identity"
    )
    return bound, sample


def stage1_overlap(
    master: pd.DataFrame, panel_gids: set[str], assignment_ready_gids: set[str]
) -> dict[str, int]:
    primary = master.primary_weighted_training_eligible.fillna(False).astype(bool)
    secondary = master.secondary_unweighted_training_eligible.fillna(False).astype(bool)
    in_panel = master.canonical_gid.isin(panel_gids)
    assignment = master.canonical_gid.isin(assignment_ready_gids)
    return {
        "accepted_identity_stage1_all_gids": int(master.loc[in_panel, "canonical_gid"].nunique()),
        "accepted_identity_stage1_primary_gids": int(
            master.loc[primary & in_panel, "canonical_gid"].nunique()
        ),
        "accepted_identity_stage1_primary_rows": int((primary & in_panel).sum()),
        "accepted_identity_stage1_secondary_gids": int(
            master.loc[secondary & in_panel, "canonical_gid"].nunique()
        ),
        "assignment_ready_stage1_primary_gids": int(
            master.loc[primary & assignment, "canonical_gid"].nunique()
        ),
        "assignment_ready_stage1_primary_rows": int((primary & assignment).sum()),
    }


def audit_eyt(article_xml: Path, calls: Path) -> dict[str, Any]:
    tree = etree.parse(str(article_xml))
    article_text = " ".join(tree.xpath("//text()"))
    required_text = ("50,058", "14,027", "6,333", "519 haplotype blocks")
    missing = [value for value in required_text if value not in article_text]
    if missing:
        raise ValueError(f"EYT source article lacks expected provenance text: {missing}")
    header = pd.read_csv(calls, nrows=0)
    columns = list(header.columns)
    gid_columns = [value for value in columns if str(value).strip().upper() == "GID"]
    block_columns = [value for value in columns if re.fullmatch(r"\d+[ABD]\.\d+", str(value))]
    return {
        "article_doi": "10.3389/fgene.2020.589490",
        "source_snp_count_initial": 50_058,
        "source_snp_count_filtered": 14_027,
        "source_line_count_after_qc": 6_333,
        "published_haplotype_block_count": 519,
        "local_haplotype_block_columns": len(block_columns),
        "local_gid_column_count": len(gid_columns),
        "source_snp_maximum_missing_fraction": 0.30,
        "source_snp_minimum_maf": 0.15,
        "source_snp_imputation": "none",
        "snp_calling_pipeline": "TASSEL_5",
        "physical_reference": "IWGSC_RefSeq_v1.1",
        "block_algorithm": "Gabriel_95pct_confidence_interval",
        "block_ci_lower": 0.60,
        "block_ci_upper": 0.95,
        "block_construction_scope": "combined_6333_lines_across_all_seven_EYTs",
        "complete_519_block_to_source_snp_membership_recovered": False,
        "source_snp_call_matrix_recovered": False,
        "strict_inductive_disposition": (
            "BLOCKED_COMPLETE_BLOCK_MEMBERSHIP_AND_SOURCE_SNP_CALLS_NOT_PUBLISHED"
        ),
    }


def source_inventory(root: Path, paths: list[tuple[str, Path, str]]) -> pd.DataFrame:
    rows = []
    for source_id, relative, source_url in paths:
        path = require_file(root, relative)
        rows.append(
            {
                "source_id": source_id,
                "relative_path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest_file(path),
                "source_url": source_url,
                "phenotype_values_read": False,
                "evaluation_outcomes_read": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.repository_root.resolve()
    output = (args.output_root or (root / DEFAULT_OUTPUT)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    decision_path = output / "PHASE5_PANEL_PREREQUISITE_RECOVERY_DECISION.json"
    if decision_path.exists():
        raise FileExistsError(f"Immutable recovery decision already exists: {decision_path}")

    passport_path = require_file(root, PASSPORT_FILE)
    cimmyt_path = require_file(root, CIMMYT_ARCHIVE)
    article_path = require_file(root, EYT_ARTICLE_XML)
    axis_path = require_file(root, AXIS_LEDGER)
    master_path = require_file(root, STAGE1_INDEX)
    eyt_calls_path = require_file(root, EYT_CALLS)

    if digest_file(passport_path) != EXPECTED_PASSPORT_SHA256:
        raise ValueError("80K passport workbook checksum mismatch")
    if cimmyt_path.stat().st_size != EXPECTED_CIMMYT_BYTES:
        raise ValueError("CIMMYT pre-QC archive byte size mismatch")
    if digest_file(cimmyt_path, "md5") != EXPECTED_CIMMYT_MD5:
        raise ValueError("CIMMYT pre-QC archive MD5 mismatch")
    if digest_file(article_path) != EXPECTED_EYT_XML_SHA256:
        raise ValueError("EYT article XML checksum mismatch")

    master = pd.read_parquet(
        master_path,
        columns=[
            "canonical_gid",
            "primary_weighted_training_eligible",
            "secondary_unweighted_training_eligible",
        ],
    )

    passport, recall_ids, untyped_recall_ids = read_passport_workbook(passport_path)
    axis = pd.read_parquet(axis_path)
    bindings, samples = build_80k_bindings(axis, passport, recall_ids)
    if not bindings.typed_identity_exact.all():
        mismatch_columns = [
            "panel_id",
            "population",
            "representation",
            "source_file",
            "raw_sample_label",
            "source_sheet",
            "source_population",
            "canonical_gid",
            "recall_membership_documented",
            "source_population_matches_axis",
        ]
        write_tsv(
            results / "dartseq80k_identity_binding_mismatches.tsv",
            bindings.loc[~bindings.typed_identity_exact, mismatch_columns]
            .drop_duplicates()
            .sort_values(["population", "raw_sample_label"]),
        )
        raise ValueError(
            f"80K authoritative workbook failed to bind {int((~bindings.typed_identity_exact).sum())} axis rows"
        )
    if not samples.typed_identity_exact.all():
        raise ValueError("80K authoritative workbook did not bind every unique panel sample")

    accepted_gids = set(samples.canonical_gid.dropna())
    assignment_ready_gids = set(
        samples.loc[samples.direct_genotype_assignment_ready, "canonical_gid"].dropna()
    )
    overlap_80k = stage1_overlap(master, accepted_gids, assignment_ready_gids)
    workbook_unused = set(passport.normalized_sample_id) - set(samples.normalized_sample_id)
    result_80k = {
        "authoritative_passport_rows": int(len(passport)),
        "authoritative_passport_sample_ids": int(passport.normalized_sample_id.nunique()),
        "authoritative_passport_gids": int(passport.canonical_gid.nunique()),
        "certified_matrix_axis_instance_rows": int(len(bindings)),
        "certified_unique_panel_samples": int(len(samples)),
        "matrix_axis_rows_with_exact_typed_identity": int(bindings.typed_identity_exact.sum()),
        "unique_panel_samples_with_exact_typed_identity": int(samples.typed_identity_exact.sum()),
        "passport_samples_not_on_certified_axes": int(len(workbook_unused)),
        "recall_sheet_labels_without_typed_passport_gid": int(len(untyped_recall_ids)),
        "wheat_recall_axis_samples_without_recall_sheet_membership": int(
            (
                samples.population.eq("wheat_recall")
                & ~samples.recall_membership_documented.astype(bool)
            ).sum()
        ),
        "gids_with_multiple_sample_ids": int(
            (samples.groupby("canonical_gid").normalized_sample_id.nunique() > 1).sum()
        ),
        "unique_identity_samples": int(
            samples.identity_class.eq("accepted_unique_identity").sum()
        ),
        "replicate_pending_samples": int(
            samples.identity_class.eq(
                "accepted_identity_replicate_set_pending_concordance"
            ).sum()
        ),
        **overlap_80k,
        "identity_blocker_resolved": True,
        "strict_production_K_G_disposition": (
            "BLOCKED_SOURCE_MATRIX_MARKERS_GLOBALLY_FILTERED_MISSING_GT_50_MAF_LE_0.001"
        ),
        "regulatory_identity_disposition": (
            "READY_FOR_COORDINATE_AND_REPLICATE_CONCORDANCE_CERTIFICATION"
        ),
    }
    passport.to_parquet(results / "dartseq80k_authoritative_passport_crosswalk.parquet")
    bindings.to_parquet(results / "dartseq80k_matrix_axis_identity_bindings.parquet")
    samples.to_parquet(results / "dartseq80k_unique_sample_identity_classification.parquet")
    write_json(results / "dartseq80k_identity_recovery_summary.json", result_80k)

    cimmyt = read_gzip_wrapped_zip_hapmap_header(cimmyt_path)
    cimmyt_sample_ids = cimmyt.pop("sample_ids")
    if len(cimmyt_sample_ids) != len(set(cimmyt_sample_ids)):
        raise ValueError("Duplicate sample identifiers in recovered CIMMYT HapMap header")
    cimmyt_gids = set(cimmyt_sample_ids)
    cimmyt_overlap = stage1_overlap(master, cimmyt_gids, cimmyt_gids)
    result_cimmyt = {
        **cimmyt,
        "source_description_marker_rows": 91_680,
        "marker_rows_independently_stream_counted": False,
        "sample_columns": len(cimmyt_sample_ids),
        "unique_sample_columns": len(cimmyt_gids),
        **cimmyt_overlap,
        "prior_filtered_marker_rows": 18_239,
        "additional_marker_rows_available_before_current_global_filters": 73_441,
        "prior_filtered_primary_stage1_gids": 4_512,
        "additional_primary_stage1_gids_in_recovered_header": (
            cimmyt_overlap["accepted_identity_stage1_primary_gids"] - 4_512
        ),
        "prior_filtered_primary_stage1_rows": 721_033,
        "additional_primary_stage1_rows_in_recovered_header": (
            cimmyt_overlap["accepted_identity_stage1_primary_rows"] - 721_033
        ),
        "strict_production_K_G_disposition": (
            "READY_FOR_STREAMED_CALL_VALIDATION_AND_150_STATE_TRAINING_LOCAL_QC"
        ),
        "production_kernel_activated": False,
    }
    write_json(results / "cimmyt_pre_qc_archive_recovery_summary.json", result_cimmyt)
    pd.DataFrame({"canonical_gid": sorted(cimmyt_gids)}).to_parquet(
        results / "cimmyt_pre_qc_hapmap_sample_order.parquet"
    )

    result_eyt = audit_eyt(article_path, eyt_calls_path)
    write_json(results / "eyt_source_snp_provenance_recovery.json", result_eyt)

    decisions = pd.DataFrame(
        [
            {
                "panel": "CIMMYT_BREAD_GBS",
                "attempt_status": "RECOVERED",
                "recovered_object": "public_unimputed_91680_marker_hapmap",
                "resolved_blocker": "current_18239_marker_global_QC_universe_only",
                "remaining_blocker": "streamed_call_validation_and_150_state_training_local_QC",
                "production_K_G_ready": False,
                "next_action": "stream calls once; fit sample/marker QC and imputation within each training state",
            },
            {
                "panel": "EYT_HAPLOTYPES",
                "attempt_status": "PARTIAL_PROVENANCE_ONLY",
                "recovered_object": "source_SNP_QC_and_519_block_construction_protocol",
                "resolved_blocker": "unknown_source_SNP_processing",
                "remaining_blocker": "complete_519_block_to_SNP_membership_and_source_SNP_calls_unpublished",
                "production_K_G_ready": False,
                "next_action": "request block definition and source calls from authors or reconstruct from the exact 50058-SNP study matrix",
            },
            {
                "panel": "DARTSEQ_80K",
                "attempt_status": "IDENTITY_RECOVERED",
                "recovered_object": "same_study_79191_accession_passport_sample_to_GID_crosswalk",
                "resolved_blocker": "same_dataset_typed_identity",
                "remaining_blocker": "global_marker_filtering_and_multi_sample_GID_concordance",
                "production_K_G_ready": False,
                "next_action": "certify replicate call concordance and seek prefilter calls; coordinates may proceed independently",
            },
        ]
    )
    write_tsv(results / "panel_recovery_decisions.tsv", decisions)

    inventory = source_inventory(
        root,
        [
            (
                "DARTSEQ80K_PASSPORT_SUPPLEMENT",
                PASSPORT_FILE,
                "https://www.nature.com/articles/s41467-020-18404-w",
            ),
            (
                "CIMMYT_PUBLIC_UNIMPUTED_91680_HAPMAP",
                CIMMYT_ARCHIVE,
                "https://data.cimmyt.org/dataset.xhtml?persistentId=hdl:11529/10695&version=2.1",
            ),
            (
                "EYT_PRIMARY_ARTICLE_JATS_XML",
                EYT_ARTICLE_XML,
                "https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2020.589490/full",
            ),
            ("DARTSEQ80K_CERTIFIED_AXIS_LEDGER", AXIS_LEDGER, "local_certified_stage1_v2"),
            ("STAGE1_V2_IDENTIFIER_INDEX", STAGE1_INDEX, "local_certified_stage1_v2"),
            ("EYT_LOCAL_HAPLOTYPE_CALLS", EYT_CALLS, "hdl:11529/10548504"),
        ],
    )
    write_tsv(results / "source_inventory.tsv", inventory)

    artifacts = []
    for path in sorted(results.glob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "relative_path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest_file(path),
                }
            )
    write_tsv(output / "artifact_manifest.tsv", pd.DataFrame(artifacts))
    decision = {
        "release_id": RELEASE_ID,
        "status": "PASS_BOUNDED_PANEL_PREREQUISITE_RECOVERY_WITH_EXPLICIT_REMAINING_BLOCKERS",
        "selection_data": "identifiers_genotype_source_metadata_and_public_scientific_provenance_only",
        "stage1_version": "Stage-1_v2",
        "phenotype_values_read": False,
        "inner_validation_metrics_read": False,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "model_training_performed": False,
        "kernels_modified": False,
        "cimmyt": result_cimmyt,
        "dartseq80k": result_80k,
        "eyt": result_eyt,
        "artifact_count": len(artifacts),
    }
    write_json(decision_path, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
