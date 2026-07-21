from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from genotype_recovery import canonical_gid, marker_alleles
from server_genotype_recovery.audit_candidate_support import load_candidates, load_order


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (root / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, dtype=str, low_memory=False)
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def detect_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    exact = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    return None


def true_values(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "pass", "retained"}
    )


def true_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "retained"}


def valid_coordinates(frame: pd.DataFrame) -> pd.Series:
    chrom_col = detect_column(frame, ["chromosome", "chrom", "chr"])
    pos_col = detect_column(
        frame, ["position", "pos", "physical_position", "bp", "snpposition"]
    )
    if chrom_col is None or pos_col is None:
        return pd.Series(False, index=frame.index)
    chrom = frame[chrom_col].fillna("").astype(str).str.strip().str.upper()
    position = pd.to_numeric(frame[pos_col], errors="coerce")
    invalid_chrom = chrom.isin({"", "U", "UN", "UNKNOWN", "NA", "N/A", "0"})
    return ~invalid_chrom & position.gt(0)


def allele_evidence(frame: pd.DataFrame) -> pd.Series:
    ref_col = detect_column(frame, ["ref", "ref_allele", "reference_allele"])
    alt_col = detect_column(frame, ["alt", "alt_allele", "alternate_allele"])
    allele_col = detect_column(frame, ["alleles", "allele", "allele_values"])
    evidence = pd.Series(False, index=frame.index)
    if ref_col is not None and alt_col is not None:
        evidence |= frame[ref_col].fillna("").astype(str).str.strip().ne("") & frame[
            alt_col
        ].fillna("").astype(str).str.strip().ne("")
    if allele_col is not None:
        evidence |= frame[allele_col].fillna("").astype(str).str.contains(
            r"[ACGT].*[ACGT]", regex=True, case=False
        )
    return evidence


def marker_evidence(
    paths: list[Path],
) -> tuple[set[str], set[str], list[dict[str, object]]]:
    coordinate_ids: set[str] = set()
    allele_ids: set[str] = set()
    sources: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            continue
        frame = read_table(path)
        marker_col = detect_column(frame, ["marker_id", "marker", "rs#", "rs", "snp"])
        if marker_col is None:
            sources.append(
                {"path": str(path), "rows": len(frame), "status": "no_marker_id_column"}
            )
            continue
        marker_ids = frame[marker_col].fillna("").astype(str).str.strip()
        coordinate_mask = valid_coordinates(frame) & marker_ids.ne("")
        allele_mask = allele_evidence(frame) & marker_ids.ne("")
        coordinate_ids.update(marker_ids[coordinate_mask])
        allele_ids.update(marker_ids[allele_mask])
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(frame),
                "status": "loaded",
                "markers_with_coordinates": int(coordinate_mask.sum()),
                "markers_with_alleles": int(allele_mask.sum()),
            }
        )
    return coordinate_ids, allele_ids, sources


def projected_markers(path: Path) -> set[str]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    frame = read_table(path)
    marker_col = detect_column(frame, ["marker_id", "marker", "rs#", "rs", "snp"])
    if marker_col is None:
        raise ValueError(f"Graph marker projection lacks a marker ID column: {path}")
    graph_cols = ["graph_node", "graph_path", "graph_start", "graph_end"]
    missing = [column for column in graph_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"Graph marker projection is missing columns {missing}: {path}")
    valid = frame[graph_cols].fillna("").astype(str).apply(
        lambda column: column.str.strip().ne("")
    ).all(axis=1)
    return set(frame.loc[valid, marker_col].fillna("").astype(str).str.strip())


def path_dictionary_ids(path: Path) -> set[str]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    frame = read_table(path)
    id_col = detect_column(frame, ["sample_id", "panel_sample_id", "genotype_id"])
    if id_col is None or "graph_path" not in frame.columns:
        raise ValueError(f"Genotype path dictionary has an invalid schema: {path}")
    valid = frame["graph_path"].fillna("").astype(str).str.strip().ne("")
    return {
        gid
        for gid in frame.loc[valid, id_col].map(canonical_gid)
        if gid
    }


def marker_ids_from_metadata(metadata_path: Path, qc_path: Path | None) -> set[str]:
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        return set()
    metadata = read_table(metadata_path)
    marker_col = detect_column(metadata, ["marker_id", "marker", "rs#", "rs", "snp"])
    if marker_col is None:
        return set()
    marker_ids = set(metadata[marker_col].fillna("").astype(str).str.strip())
    marker_ids.discard("")
    if qc_path is None or not qc_path.is_file() or qc_path.stat().st_size == 0:
        return marker_ids
    qc = read_table(qc_path)
    qc_marker_col = detect_column(qc, ["marker_id", "marker", "rs#", "rs", "snp"])
    keep_col = detect_column(qc, ["keep_marker", "retained", "keep"])
    if qc_marker_col is None or keep_col is None:
        return marker_ids
    retained = set(
        qc.loc[true_values(qc[keep_col]), qc_marker_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    retained.discard("")
    return marker_ids & retained


def recovered_marker_path(order_path: Path) -> Path | None:
    name = order_path.name
    suffix = "_sample_order.tsv"
    if not name.endswith(suffix):
        return None
    prefix = name[: -len(suffix)]
    candidates = [
        order_path.with_name(f"{prefix}_retained_marker_order.tsv.gz"),
        order_path.with_name(f"{prefix}_retained_marker_order.tsv"),
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def genotype_matrix_evidence(
    path: Path | None,
    *,
    sample_order_ids: list[str],
    marker_ids: set[str],
) -> dict[str, object]:
    evidence: dict[str, object] = {
        "genotype_matrix_available": False,
        "genotype_matrix_rows": 0,
        "genotype_matrix_marker_columns": 0,
        "genotype_matrix_certification_status": "missing",
        "genotype_matrix_certified": False,
    }
    if path is None or not path.is_file() or path.stat().st_size == 0:
        return evidence
    evidence["genotype_matrix_available"] = True
    try:
        if path.suffix.lower() == ".npy":
            matrix = np.load(path, mmap_mode="r")
            if matrix.ndim != 2:
                evidence["genotype_matrix_certification_status"] = "not_two_dimensional"
                return evidence
            rows, columns = map(int, matrix.shape)
            evidence["genotype_matrix_rows"] = rows
            evidence["genotype_matrix_marker_columns"] = columns
            aligned = rows == len(sample_order_ids) and columns == len(marker_ids)
        elif path.suffix.lower() == ".parquet":
            parquet = pq.ParquetFile(path)
            rows = int(parquet.metadata.num_rows)
            schema_names = list(parquet.schema.names)
            id_col = next(
                (
                    column
                    for column in ["sample_id", "panel_sample_id", "genotype_id"]
                    if column in schema_names
                ),
                None,
            )
            marker_columns = [column for column in schema_names if column != id_col]
            evidence["genotype_matrix_rows"] = rows
            evidence["genotype_matrix_marker_columns"] = len(marker_columns)
            aligned = rows == len(sample_order_ids) and set(marker_columns) == marker_ids
            if id_col is not None and aligned:
                matrix_ids = (
                    pd.read_parquet(path, columns=[id_col])[id_col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
                aligned = matrix_ids == sample_order_ids
        else:
            evidence["genotype_matrix_certification_status"] = "unsupported_format"
            return evidence
    except Exception as exc:  # Input evidence must fail closed.
        evidence["genotype_matrix_certification_status"] = (
            f"unreadable:{type(exc).__name__}"
        )
        return evidence
    evidence["genotype_matrix_certification_status"] = (
        "aligned" if aligned else "dimension_or_order_mismatch"
    )
    evidence["genotype_matrix_certified"] = bool(aligned)
    return evidence


def panel_identifier(candidate_group: str) -> str:
    aliases = {
        "existing_HMP": "HMP",
        "existing_GBS_SAWYT": "GBS_SAWYT",
    }
    return aliases.get(candidate_group, candidate_group.removeprefix("K_G_"))


def choose_panel_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for _, group in candidates.groupby("candidate_group", sort=True):
        linear = group[group["kernel"].astype(str).str.endswith("_LINEAR")]
        selected.append((linear if not linear.empty else group).iloc[0])
    return pd.DataFrame(selected).reset_index(drop=True)


def certification_statuses(path: Path) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Candidate kernel QC is missing: {path}")
    frame = read_table(path)
    if not {"kernel", "status"}.issubset(frame.columns):
        raise ValueError(f"Candidate kernel QC has an invalid schema: {path}")
    return dict(zip(frame["kernel"].astype(str), frame["status"].astype(str)))


def regulatory_retention_policy(path: Path) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Regulatory retention policy is missing: {path}")
    frame = read_table(path)
    if not {"policy", "value"}.issubset(frame.columns):
        raise ValueError(f"Regulatory retention policy has an invalid schema: {path}")
    values = dict(zip(frame["policy"].astype(str), frame["value"].astype(str)))
    expected = {
        "quantitative_screen_scope": "standalone_K_G_baseline_inclusion_only",
        "retain_certified_panels_for_regulatory_projection": "true",
        "direct_regulatory_embedding_status": "observed_marker_supported_sequence",
        "pedigree_propagated_embedding_status": "imputed_pedigree",
        "pedigree_propagation_requires_confidence_gate": "true",
        "pedigree_propagation_equivalent_to_observed_sequence": "false",
    }
    failed = {
        key: {"expected": expected_value, "observed": values.get(key, "MISSING")}
        for key, expected_value in expected.items()
        if values.get(key, "").strip().lower() != expected_value.lower()
    }
    if failed:
        raise ValueError(f"Regulatory retention policy contract failed: {failed}")
    return values


def build_panel_evidence(
    *,
    root: Path,
    candidates: pd.DataFrame,
    qc_status: dict[str, str],
    coordinate_ids: set[str],
    allele_ids: set[str],
    graph_marker_ids: set[str],
    minimum_graph_projection_fraction: float,
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    panel_rows: list[dict[str, object]] = []
    samples: dict[str, set[str]] = {}
    for row in choose_panel_candidates(candidates).to_dict("records"):
        kernel = str(row["kernel"])
        group = str(row["candidate_group"])
        panel_id = panel_identifier(group)
        order_path = Path(row["order_path"])
        id_col = str(row.get("source_id_col", "sample_id"))
        _, raw_ids = load_order(order_path, id_col)
        gids = {gid for value in raw_ids if (gid := canonical_gid(value))}
        samples[panel_id] = gids
        support_type = "haplotype_blocks" if "HAPLOTYPE" in panel_id else "marker_calls"
        marker_path: Path | None = None
        matrix_path: Path | None = None
        marker_ids: set[str] = set()
        if panel_id == "HMP":
            marker_path = root / "genotype_panels/hmp/hmp_marker_metadata.tsv"
            marker_ids = marker_ids_from_metadata(
                marker_path, root / "genotype_panels/hmp/qc_hmp_marker_stats.tsv"
            )
            matrix_path = root / "genotype_panels/hmp/hmp_sample_by_marker.QCfiltered.parquet"
        elif panel_id == "GBS_SAWYT":
            marker_path = root / "genotype_panels/gbs_sawyt/gbs_sawyt_marker_metadata.tsv"
            marker_ids = marker_ids_from_metadata(
                marker_path,
                root / "genotype_panels/gbs_sawyt/qc_gbs_sawyt_marker_stats.tsv",
            )
            matrix_path = (
                root
                / "genotype_panels/gbs_sawyt/gbs_sawyt_sample_by_marker.QCfiltered.parquet"
            )
        elif support_type == "marker_calls":
            marker_path = recovered_marker_path(order_path)
            if marker_path is not None and marker_path.is_file():
                marker_frame = read_table(marker_path)
                marker_col = detect_column(marker_frame, ["marker_id", "marker"])
                if marker_col is not None:
                    marker_ids = set(
                        marker_frame[marker_col].fillna("").astype(str).str.strip()
                    )
                    marker_ids.discard("")
            prefix = order_path.name.removesuffix("_sample_order.tsv")
            matrix_path = order_path.with_name(f"{prefix}_QC_dosage.npy")
        matrix_evidence = genotype_matrix_evidence(
            matrix_path,
            sample_order_ids=raw_ids,
            marker_ids=marker_ids,
        )
        coordinate_markers = marker_ids & coordinate_ids
        parsed_alleles = {marker for marker in marker_ids if marker_alleles(marker) is not None}
        allele_markers = marker_ids & (allele_ids | parsed_alleles)
        projectable_markers = coordinate_markers & allele_markers
        projected = marker_ids & graph_marker_ids
        projected_projectable = projectable_markers & graph_marker_ids
        projection_fraction = (
            len(projected_projectable) / len(marker_ids) if marker_ids else 0.0
        )
        graph_ready = (
            bool(marker_ids)
            and bool(projected_projectable)
            and projection_fraction >= minimum_graph_projection_fraction
        )
        certification = qc_status.get(kernel, "MISSING")
        if certification != "PASS":
            raise ValueError(f"Panel kernel is not certified PASS: {kernel}={certification}")
        if support_type == "haplotype_blocks":
            next_action = "map_haplotype_blocks_to_refseq_v1_intervals"
        elif not matrix_evidence["genotype_matrix_available"]:
            next_action = "restore_certified_QC_genotype_matrix"
        elif not matrix_evidence["genotype_matrix_certified"]:
            next_action = "certify_QC_genotype_matrix_alignment"
        elif not allele_markers:
            next_action = "recover_marker_ref_alt_alleles"
        elif not coordinate_markers:
            next_action = "map_or_align_markers_to_refseq_v1_coordinates"
        elif not graph_ready:
            next_action = "project_refseq_v1_markers_to_zenodo_graph"
        else:
            next_action = "construct_genotype_specific_sequence_windows"
        panel_rows.append(
            {
                "panel_id": panel_id,
                "kernel": kernel,
                "biological_role": row["biological_role"],
                "support_type": support_type,
                "certification_status": certification,
                "sample_order_path": str(order_path),
                "sample_order_sha256": sha256_file(order_path),
                "sample_order_count": len(raw_ids),
                "certified_gid_count": len(gids),
                "noncanonical_sample_id_count": len(raw_ids) - len(gids),
                "genotype_matrix_path": str(matrix_path) if matrix_path is not None else "",
                "genotype_matrix_sha256": (
                    sha256_file(matrix_path)
                    if matrix_path is not None and matrix_path.is_file()
                    else ""
                ),
                **matrix_evidence,
                "retained_marker_path": str(marker_path) if marker_path is not None else "",
                "retained_marker_sha256": (
                    sha256_file(marker_path)
                    if marker_path is not None and marker_path.is_file()
                    else ""
                ),
                "retained_marker_count": len(marker_ids),
                "markers_with_alleles": len(allele_markers),
                "markers_with_physical_coordinates": len(coordinate_markers),
                "projectable_variant_count": len(projectable_markers),
                "graph_projected_marker_count": len(projected),
                "graph_projected_projectable_marker_count": len(projected_projectable),
                "graph_projection_fraction": projection_fraction,
                "graph_projection_ready": graph_ready,
                "next_required_action": next_action,
            }
        )
    return pd.DataFrame(panel_rows), samples


def load_gid_set(path: Path, id_candidates: list[str]) -> set[str]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    frame = read_table(path)
    id_col = detect_column(frame, id_candidates)
    if id_col is None:
        return set()
    return {gid for value in frame[id_col] if (gid := canonical_gid(value))}


def embedding_evidence(root: Path) -> tuple[dict[str, set[str]], list[Path]]:
    by_gid: dict[str, set[str]] = defaultdict(set)
    paths = sorted(
        (root / "regulatory_model/embeddings").glob(
            "**/*_genotype_regulatory_embedding_order.tsv"
        )
    )
    for path in paths:
        frame = read_table(path)
        id_col = detect_column(frame, ["sample_id", "panel_sample_id", "genotype_id"])
        if id_col is None:
            continue
        for value in frame[id_col]:
            gid = canonical_gid(value)
            if gid:
                by_gid[gid].add(str(path))
    return by_gid, paths


def load_marker_identity_overlay(path: Path) -> pd.DataFrame:
    required = {
        "canonical_gid",
        "marker_identity_adjudication_status",
        "marker_identity_classes",
        "candidate_marker_panels",
        "accepted_marker_panels",
        "candidate_unresolved",
        "accepted_for_new_kernel_input",
        "eligible_for_K_G",
        "eligible_for_K_z",
        "eligible_for_genotype_specific_sequence",
        "next_required_action",
    }
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(columns=sorted(required))
    overlay = read_table(path)
    missing = sorted(required - set(overlay.columns))
    if missing:
        raise ValueError(f"Marker identity overlay is missing columns {missing}: {path}")
    overlay = overlay.copy()
    overlay["canonical_gid"] = overlay["canonical_gid"].map(canonical_gid)
    if overlay["canonical_gid"].eq("").any():
        raise ValueError(f"Marker identity overlay contains noncanonical GIDs: {path}")
    if overlay["canonical_gid"].duplicated().any():
        raise ValueError(f"Marker identity overlay contains duplicate GIDs: {path}")
    for column in (
        "candidate_unresolved",
        "accepted_for_new_kernel_input",
        "eligible_for_K_G",
        "eligible_for_K_z",
        "eligible_for_genotype_specific_sequence",
    ):
        overlay[column] = overlay[column].map(true_value)
    prohibited = overlay[
        overlay["eligible_for_K_G"]
        | overlay["eligible_for_K_z"]
        | overlay["eligible_for_genotype_specific_sequence"]
    ]
    if not prohibited.empty:
        raise ValueError(
            "Uncertified marker identity candidates cannot be eligible for K_G, K_z, "
            "or genotype-specific sequence"
        )
    accepted = overlay["accepted_for_new_kernel_input"]
    unresolved = overlay["candidate_unresolved"]
    if (~(accepted | unresolved)).any():
        raise ValueError(
            "Every marker identity overlay row must contain an accepted or unresolved candidate"
        )
    return overlay


def apply_marker_identity_overlay(
    manifest: pd.DataFrame, overlay: pd.DataFrame
) -> pd.DataFrame:
    output = manifest.copy()
    if overlay.empty:
        output["marker_identity_adjudication_status"] = "no_candidate"
        output["marker_identity_classes"] = ""
        output["candidate_marker_panels"] = ""
        output["accepted_marker_panels"] = ""
        output["candidate_unresolved"] = False
        output["accepted_for_new_kernel_input"] = False
        output["candidate_eligible_for_K_G"] = False
        output["candidate_eligible_for_K_z"] = False
        output["candidate_eligible_for_genotype_specific_sequence"] = False
        output["marker_identity_next_required_action"] = "not_applicable"
        return output
    missing_gids = sorted(set(overlay["canonical_gid"]) - set(output["canonical_gid"]))
    if missing_gids:
        raise ValueError(
            f"Marker identity overlay contains {len(missing_gids)} GIDs outside the "
            "regulatory manifest universe; examples={missing_gids[:5]}"
        )
    renamed = overlay.rename(
        columns={
            "eligible_for_K_G": "candidate_eligible_for_K_G",
            "eligible_for_K_z": "candidate_eligible_for_K_z",
            "eligible_for_genotype_specific_sequence": (
                "candidate_eligible_for_genotype_specific_sequence"
            ),
            "next_required_action": "marker_identity_next_required_action",
        }
    )
    output = output.merge(renamed, on="canonical_gid", how="left", validate="one_to_one")
    text_defaults = {
        "marker_identity_adjudication_status": "no_candidate",
        "marker_identity_classes": "",
        "candidate_marker_panels": "",
        "accepted_marker_panels": "",
        "marker_identity_next_required_action": "not_applicable",
    }
    for column, default in text_defaults.items():
        output[column] = output[column].fillna(default)
    for column in (
        "candidate_unresolved",
        "accepted_for_new_kernel_input",
        "candidate_eligible_for_K_G",
        "candidate_eligible_for_K_z",
        "candidate_eligible_for_genotype_specific_sequence",
    ):
        output[column] = output[column].map(
            lambda value: bool(value) if pd.notna(value) else False
        )
    return output


def build_gid_manifest(
    *,
    catalog: pd.DataFrame,
    pedigree_ids: set[str],
    panel_evidence: pd.DataFrame,
    panel_samples: dict[str, set[str]],
    graph_path_ids: set[str],
    embeddings: dict[str, set[str]],
) -> pd.DataFrame:
    catalog = catalog.copy()
    catalog["canonical_gid"] = catalog["canonical_gid"].map(canonical_gid)
    catalog = catalog[catalog["canonical_gid"].ne("")].drop_duplicates("canonical_gid")
    catalog_lookup = catalog.set_index("canonical_gid").to_dict("index")
    all_ids = set(catalog_lookup) | pedigree_ids
    for ids in panel_samples.values():
        all_ids.update(ids)
    evidence_by_panel = panel_evidence.set_index("panel_id").to_dict("index")
    rows: list[dict[str, object]] = []
    for gid in sorted(all_ids):
        supported = sorted(panel for panel, ids in panel_samples.items() if gid in ids)
        marker_panels = sorted(
            panel
            for panel in supported
            if evidence_by_panel[panel]["support_type"] == "marker_calls"
        )
        haplotype_panels = sorted(set(supported) - set(marker_panels))
        coordinate_panels = sorted(
            panel
            for panel in marker_panels
            if int(evidence_by_panel[panel]["projectable_variant_count"]) > 0
        )
        projected_panels = sorted(
            panel
            for panel in marker_panels
            if true_value(evidence_by_panel[panel]["graph_projection_ready"])
        )
        matrix_panels = sorted(
            panel
            for panel in marker_panels
            if true_value(evidence_by_panel[panel]["genotype_matrix_certified"])
        )
        direct_window_ready = sorted(set(projected_panels) & set(matrix_panels))
        in_pedigree = gid in pedigree_ids
        if direct_window_ready:
            eligibility = "eligible_direct_sequence_window_construction"
            future_class = "observed_marker_supported_sequence"
            confidence_gate = "not_required_for_direct_observation"
        elif coordinate_panels:
            eligibility = "marker_coordinates_ready_graph_projection_pending"
            future_class = "observed_marker_supported_sequence_pending"
            confidence_gate = "not_evaluated_until_projection"
        elif marker_panels:
            eligibility = "marker_supported_coordinate_mapping_pending"
            future_class = "observed_marker_supported_sequence_pending"
            confidence_gate = "not_evaluated_until_projection"
        elif haplotype_panels:
            eligibility = "haplotype_supported_block_coordinates_pending"
            future_class = "observed_haplotype_supported_sequence_pending"
            confidence_gate = "not_evaluated_until_projection"
        elif in_pedigree:
            eligibility = "pedigree_imputation_candidate"
            future_class = "imputed_pedigree"
            confidence_gate = "required_not_evaluated"
        else:
            eligibility = "unavailable"
            future_class = "unavailable"
            confidence_gate = "not_applicable"
        if coordinate_panels:
            coordinate_status = "available_for_at_least_one_certified_panel"
        elif marker_panels:
            coordinate_status = "absent_from_current_coordinate_sources"
        else:
            coordinate_status = "not_applicable_without_marker_support"
        if direct_window_ready:
            projection_status = "ready_for_at_least_one_certified_panel"
        elif marker_panels:
            projection_status = "pending_or_below_threshold"
        else:
            projection_status = "not_applicable_without_marker_support"
        embedding_sources = sorted(embeddings.get(gid, set()))
        catalog_row = catalog_lookup.get(gid, {})
        observation_rows = pd.to_numeric(
            catalog_row.get("canonical_observation_rows", 0), errors="coerce"
        )
        if pd.isna(observation_rows):
            observation_rows = 0
        if direct_window_ready:
            equivalence_reason = "sequence_windows_and_embeddings_not_yet_certified"
        elif in_pedigree:
            equivalence_reason = "pedigree_imputation_is_not_observed_sequence"
        else:
            equivalence_reason = "no_certified_direct_sequence_evidence"
        rows.append(
            {
                "canonical_gid": gid,
                "in_canonical_trial_catalog": gid in catalog_lookup,
                "canonical_observation_rows": int(observation_rows),
                "pedigree_support": in_pedigree,
                "certified_panel_count": len(supported),
                "certified_panels": ";".join(supported),
                "direct_marker_panel_count": len(marker_panels),
                "direct_marker_panels": ";".join(marker_panels),
                "haplotype_panel_count": len(haplotype_panels),
                "haplotype_panels": ";".join(haplotype_panels),
                "coordinate_ready_panels": ";".join(coordinate_panels),
                "graph_projection_ready_panels": ";".join(projected_panels),
                "genotype_matrix_ready_panels": ";".join(matrix_panels),
                "variant_coordinate_availability": coordinate_status,
                "graph_projection_status": projection_status,
                "genotype_path_status": (
                    "path_dictionary_match" if gid in graph_path_ids else "path_not_assigned"
                ),
                "regulatory_embedding_eligibility": eligibility,
                "regulatory_embedding_status": (
                    "embedding_present_provenance_unverified"
                    if embedding_sources
                    else "not_generated"
                ),
                "embedding_sources": ";".join(embedding_sources),
                "future_embedding_provenance_class": future_class,
                "confidence_gate_status": confidence_gate,
                "observed_sequence_equivalent": False,
                "observed_sequence_equivalence_reason": equivalence_reason,
            }
        )
    return pd.DataFrame(rows)


def summary_tables(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    status = (
        manifest.groupby(
            ["regulatory_embedding_eligibility", "future_embedding_provenance_class"],
            dropna=False,
        )
        .agg(
            genotype_count=("canonical_gid", "nunique"),
            trial_genotype_count=("in_canonical_trial_catalog", "sum"),
            canonical_observation_rows=("canonical_observation_rows", "sum"),
        )
        .reset_index()
    )
    panel_rows = []
    for panel in sorted(
        {
            value
            for values in manifest["certified_panels"].fillna("")
            for value in values.split(";")
            if value
        }
    ):
        selected = manifest["certified_panels"].fillna("").str.split(";").map(
            lambda values: panel in values
        )
        local = manifest[selected]
        panel_rows.append(
            {
                "panel_id": panel,
                "genotype_count": local["canonical_gid"].nunique(),
                "trial_genotype_count": int(local["in_canonical_trial_catalog"].sum()),
                "canonical_observation_rows": int(local["canonical_observation_rows"].sum()),
            }
        )
    return status, pd.DataFrame(panel_rows)


def projection_work_queue(panel_evidence: pd.DataFrame) -> pd.DataFrame:
    work_queue = panel_evidence[
        ["panel_id", "certified_gid_count", "next_required_action"]
    ].copy()
    work_queue["certified_gid_count"] = pd.to_numeric(
        work_queue["certified_gid_count"], errors="raise"
    ).astype(int)
    work_queue = work_queue.sort_values(
        ["certified_gid_count", "panel_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    work_queue.insert(0, "priority", range(1, len(work_queue) + 1))
    return work_queue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a provenance-aware regulatory eligibility manifest for certified GIDs."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--recovered-manifest",
        type=Path,
        default=Path("genotype_panels/recovered/recovered_genotype_kernel_manifest.tsv"),
    )
    parser.add_argument(
        "--kernel-qc",
        type=Path,
        default=Path("model_kernels/genomic_candidate_screen_v1/genomic_candidate_kernel_qc.tsv"),
    )
    parser.add_argument(
        "--regulatory-policy",
        type=Path,
        default=Path(
            "model_kernels/genomic_candidate_screen_v1/"
            "genomic_candidate_regulatory_retention_policy.tsv"
        ),
    )
    parser.add_argument(
        "--canonical-catalog",
        type=Path,
        default=Path("audit/genotypic_recovery/canonical_genotype_catalog.csv"),
    )
    parser.add_argument(
        "--pedigree-order",
        type=Path,
        default=Path("genotype_panels/pedigree/K_A_sample_order.tsv"),
    )
    parser.add_argument(
        "--marker-projection",
        type=Path,
        default=Path("pangenome_resources/graph/marker_to_graph_interval.tsv"),
    )
    parser.add_argument(
        "--path-dictionary",
        type=Path,
        default=Path("pangenome_resources/graph/genotype_path_dictionary.tsv"),
    )
    parser.add_argument("--coordinate-table", type=Path, action="append", default=[])
    parser.add_argument(
        "--marker-identity-overlay",
        type=Path,
        default=Path(
            "genotype_panels/marker_identity_adjudication_v1/"
            "regulatory_eligibility_overlay.tsv"
        ),
    )
    parser.add_argument("--require-marker-identity-overlay", action="store_true")
    parser.add_argument("--minimum-graph-projection-fraction", type=float, default=0.90)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("model_kernels/regulatory_eligibility_v1")
    )
    args = parser.parse_args()
    if not 0 < args.minimum_graph_projection_fraction <= 1:
        parser.error("--minimum-graph-projection-fraction must be in (0, 1]")
    root = args.root.resolve()
    recovered_manifest = resolve(root, args.recovered_manifest)
    kernel_qc_path = resolve(root, args.kernel_qc)
    regulatory_policy_path = resolve(root, args.regulatory_policy)
    catalog_path = resolve(root, args.canonical_catalog)
    pedigree_path = resolve(root, args.pedigree_order)
    marker_projection_path = resolve(root, args.marker_projection)
    path_dictionary_path = resolve(root, args.path_dictionary)
    marker_identity_overlay_path = resolve(root, args.marker_identity_overlay)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    required = [recovered_manifest, kernel_qc_path, regulatory_policy_path, catalog_path]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit(f"Required regulatory eligibility inputs are missing: {missing}")
    coordinate_paths = [
        root / "genotype_panels/hmp/hmp_marker_metadata.tsv",
        root / "genotype_panels/gbs_sawyt/gbs_sawyt_marker_metadata.tsv",
        root / "functional_annotation/marker_to_graph_region.tsv",
        *[resolve(root, path) for path in args.coordinate_table],
    ]
    coordinate_paths = list(dict.fromkeys(path.resolve() for path in coordinate_paths))
    coordinate_ids, allele_ids, coordinate_sources = marker_evidence(coordinate_paths)
    graph_marker_ids = projected_markers(marker_projection_path)
    graph_path_ids = path_dictionary_ids(path_dictionary_path)
    candidates = load_candidates(root, recovered_manifest)
    qc_status = certification_statuses(kernel_qc_path)
    retention_policy = regulatory_retention_policy(regulatory_policy_path)
    panel_evidence, panel_samples = build_panel_evidence(
        root=root,
        candidates=candidates,
        qc_status=qc_status,
        coordinate_ids=coordinate_ids,
        allele_ids=allele_ids,
        graph_marker_ids=graph_marker_ids,
        minimum_graph_projection_fraction=args.minimum_graph_projection_fraction,
    )
    catalog = pd.read_csv(catalog_path, dtype=str)
    pedigree_ids = load_gid_set(pedigree_path, ["sample_id", "panel_sample_id", "genotype_id"])
    embeddings, embedding_paths = embedding_evidence(root)
    manifest = build_gid_manifest(
        catalog=catalog,
        pedigree_ids=pedigree_ids,
        panel_evidence=panel_evidence,
        panel_samples=panel_samples,
        graph_path_ids=graph_path_ids,
        embeddings=embeddings,
    )
    if args.require_marker_identity_overlay and not marker_identity_overlay_path.is_file():
        raise SystemExit(
            f"Required marker identity overlay is missing: {marker_identity_overlay_path}"
        )
    marker_identity_overlay = load_marker_identity_overlay(marker_identity_overlay_path)
    manifest = apply_marker_identity_overlay(manifest, marker_identity_overlay)
    status_summary, panel_summary = summary_tables(manifest)
    manifest.to_csv(
        out_dir / "regulatory_genotype_eligibility_manifest.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    panel_evidence.to_csv(out_dir / "regulatory_panel_evidence.tsv", sep="\t", index=False)
    status_summary.to_csv(
        out_dir / "regulatory_eligibility_status_summary.tsv", sep="\t", index=False
    )
    panel_summary.to_csv(
        out_dir / "regulatory_eligibility_panel_summary.tsv", sep="\t", index=False
    )
    work_queue = projection_work_queue(panel_evidence)
    work_queue.to_csv(
        out_dir / "regulatory_projection_work_queue.tsv", sep="\t", index=False
    )
    provenance = {
        "status": "PASS",
        "selection_data": "certified_identifiers_and_coordinate_evidence_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "canonical_catalog": str(catalog_path),
        "canonical_catalog_sha256": sha256_file(catalog_path),
        "recovered_manifest": str(recovered_manifest),
        "recovered_manifest_sha256": sha256_file(recovered_manifest),
        "kernel_qc": str(kernel_qc_path),
        "kernel_qc_sha256": sha256_file(kernel_qc_path),
        "regulatory_retention_policy": str(regulatory_policy_path),
        "regulatory_retention_policy_sha256": sha256_file(regulatory_policy_path),
        "pedigree_order": str(pedigree_path),
        "pedigree_order_present": pedigree_path.is_file(),
        "pedigree_order_sha256": (
            sha256_file(pedigree_path) if pedigree_path.is_file() else ""
        ),
        "marker_projection": str(marker_projection_path),
        "marker_projection_present": marker_projection_path.is_file(),
        "marker_projection_sha256": (
            sha256_file(marker_projection_path) if marker_projection_path.is_file() else ""
        ),
        "path_dictionary": str(path_dictionary_path),
        "path_dictionary_present": path_dictionary_path.is_file(),
        "path_dictionary_sha256": (
            sha256_file(path_dictionary_path) if path_dictionary_path.is_file() else ""
        ),
        "marker_identity_overlay": str(marker_identity_overlay_path),
        "marker_identity_overlay_present": marker_identity_overlay_path.is_file(),
        "marker_identity_overlay_sha256": (
            sha256_file(marker_identity_overlay_path)
            if marker_identity_overlay_path.is_file()
            else ""
        ),
        "builder_path": str(Path(__file__).resolve()),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "coordinate_sources": coordinate_sources,
        "embedding_order_paths": [str(path) for path in embedding_paths],
        "embedding_order_sources": [
            {"path": str(path), "sha256": sha256_file(path)} for path in embedding_paths
        ],
        "genotype_count": len(manifest),
        "panel_count": len(panel_evidence),
        "direct_marker_supported_genotypes": int(manifest["direct_marker_panel_count"].gt(0).sum()),
        "coordinate_ready_genotypes": int(manifest["coordinate_ready_panels"].ne("").sum()),
        "graph_projection_ready_genotypes": int(
            manifest["graph_projection_ready_panels"].ne("").sum()
        ),
        "pedigree_imputation_candidates": int(
            manifest["regulatory_embedding_eligibility"]
            .eq("pedigree_imputation_candidate")
            .sum()
        ),
        "accepted_marker_identity_candidates": int(
            manifest["accepted_for_new_kernel_input"].sum()
        ),
        "unresolved_marker_identity_candidates": int(
            manifest["candidate_unresolved"].sum()
        ),
        "interpretation_contract": {
            "quantitative_K_G_rejection_discards_panel": (
                retention_policy["retain_certified_panels_for_regulatory_projection"]
                .strip()
                .lower()
                != "true"
            ),
            "observed_sequence_requires_certified_calls_and_projection": True,
            "pedigree_propagated_status": "imputed_pedigree",
            "pedigree_confidence_gate_required": True,
            "pedigree_imputation_equivalent_to_observed_sequence": False,
        },
    }
    (out_dir / "regulatory_eligibility_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    print("\n=== REGULATORY ELIGIBILITY ===")
    print(status_summary.to_string(index=False))
    print("\n=== PANEL EVIDENCE ===")
    print(panel_evidence.to_string(index=False))


if __name__ == "__main__":
    main()
