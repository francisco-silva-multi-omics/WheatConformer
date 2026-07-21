from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from genotype_recovery import genotype_call_to_dosage, marker_alleles
from server_genotype_recovery.audit_dataverse_pedigree_enrichment import (
    canonical_cimmyt_gid,
)
from server_genotype_recovery.audit_dataverse_two_hop_marker_bridges import (
    canonical_gid,
)
from server_genotype_recovery.fetch_brapi_pedigree_markers import (
    clean,
    normalized_identifier,
    read_table,
    sha256_file,
    write_json_atomic,
)


TERMINAL_CLASSES = {
    "accepted_unique_identity",
    "accepted_concordant_replicates",
    "requires_metadata_review",
    "conflicting_marker_samples",
    "family_only_not_assignable",
}

CANDIDATE_COLUMNS = [
    "trial_gid",
    "candidate_scope",
    "panel_id",
    "selection_history",
    "normalized_selection_history",
    "selection_history_gid_count",
    "selection_history_unique",
    "trial_cross",
    "external_gid",
    "external_alias",
    "sample_id",
    "normalized_sample_id",
    "mapping_filename",
    "mapping_source_part",
    "mapping_source_row",
    "marker_matrix_path",
    "marker_matrix_sha256",
    "marker_matrix_axis",
    "marker_matrix_axis_index",
    "marker_matrix_locator",
    "marker_axis_match_count",
    "external_identity_count",
    "external_record_count",
    "pedigree_conflict_status",
    "pedigree_conflict_reasons",
    "existing_certified_gid",
    "classification",
    "classification_reasons",
    "direct_marker_assignment_ready",
]

PAIR_COLUMNS = [
    "trial_gid",
    "panel_id",
    "sample_id_left",
    "sample_id_right",
    "shared_nonmissing_markers",
    "concordant_markers",
    "call_concordance",
    "minimum_shared_markers",
    "minimum_call_concordance",
    "overlap_pass",
    "concordance_pass",
    "pair_status",
]


def resolve(root: Path, value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_policy(path: Path) -> dict[str, object]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    accepted = set(policy.get("accepted_classes", []))
    unresolved = set(policy.get("unresolved_classes", []))
    conflicting = set(policy.get("conflicting_classes", []))
    if accepted | unresolved | conflicting != TERMINAL_CLASSES:
        raise ValueError("Identity policy does not define the frozen terminal classes")
    threshold = float(policy["minimum_pairwise_call_concordance"])
    if not 0.0 < threshold <= 1.0:
        raise ValueError("minimum_pairwise_call_concordance must be in (0, 1]")
    if int(policy["minimum_shared_markers_default"]) < 1:
        raise ValueError("minimum_shared_markers_default must be positive")
    if policy.get("selection_data") != "identifiers_metadata_pedigree_and_marker_calls_only":
        raise ValueError("Identity policy selection_data contract is invalid")
    return policy


def validate_upstream_provenance(
    *,
    two_hop_provenance_path: Path,
    pedigree_provenance_path: Path,
    resolver_path: Path,
) -> dict[str, str]:
    two_hop = json.loads(two_hop_provenance_path.read_text(encoding="utf-8"))
    pedigree = json.loads(pedigree_provenance_path.read_text(encoding="utf-8"))
    if two_hop.get("status") != "complete" or pedigree.get("status") != "complete":
        raise ValueError("Upstream two-hop and pedigree audits must both be complete")
    pedigree_inputs = pedigree.get("inputs", {})
    structured_hashes = {
        clean(two_hop.get("structured_evidence_sha256")),
        clean(pedigree_inputs.get("evidence", {}).get("sha256")),
    }
    structured_hashes.discard("")
    if len(structured_hashes) != 1:
        raise ValueError(
            "Two-hop and pedigree audits use different structured-evidence snapshots; "
            "rerun both audits from the completed wide inventory"
        )
    resolver_hash = sha256_file(resolver_path)
    resolver_hashes = {
        clean(two_hop.get("resolver_query_sha256")),
        clean(pedigree_inputs.get("resolver", {}).get("sha256")),
        resolver_hash,
    }
    resolver_hashes.discard("")
    if len(resolver_hashes) != 1:
        raise ValueError(
            "Two-hop, pedigree, and current resolver inputs differ; rerun the upstream audits"
        )
    return {
        "structured_evidence_sha256": next(iter(structured_hashes)),
        "resolver_query_sha256": resolver_hash,
    }


def sha256_file_with_progress(path: Path, *, progress_bytes: int = 1024**3) -> str:
    digest = hashlib.sha256()
    processed = 0
    next_report = progress_bytes
    total = path.stat().st_size
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024**2), b""):
            digest.update(chunk)
            processed += len(chunk)
            if processed >= next_report:
                print(
                    f"matrix hash progress: {processed / 1024**3:.1f}/"
                    f"{total / 1024**3:.1f} GiB",
                    flush=True,
                )
                next_report += progress_bytes
    return digest.hexdigest()


def _joined(values: pd.Series) -> str:
    observed: dict[str, str] = {}
    for value in values:
        text = clean(value)
        normalized = normalized_identifier(text)
        if normalized and normalized not in observed:
            observed[normalized] = text
    return ";".join(observed[key] for key in sorted(observed))


def resolver_identity_summary(resolver: pd.DataFrame) -> pd.DataFrame:
    id_candidates = [
        "sample_id",
        "panel_sample_id_expected",
        "panel_sample_id",
        "query_id",
        "gid",
    ]
    id_col = next((column for column in id_candidates if column in resolver.columns), None)
    if id_col is None:
        raise ValueError("Resolver query has no recognized trial GID column")
    history_col = next(
        (column for column in ("selection_history", "selection_hist") if column in resolver.columns),
        None,
    )
    cross_col = next(
        (
            column
            for column in ("cross_name", "cross", "pedigree", "designation")
            if column in resolver.columns
        ),
        None,
    )
    work = pd.DataFrame({"trial_gid": resolver[id_col].map(canonical_gid)})
    work["selection_history"] = resolver[history_col].map(clean) if history_col else ""
    work["trial_cross"] = resolver[cross_col].map(clean) if cross_col else ""
    work = work[work["trial_gid"].ne("")].copy()
    work["normalized_selection_history"] = work["selection_history"].map(
        normalized_identifier
    )
    history_counts = (
        work[work["normalized_selection_history"].ne("")]
        .groupby("normalized_selection_history")["trial_gid"]
        .nunique()
    )
    rows: list[dict[str, object]] = []
    for trial_gid, group in work.groupby("trial_gid", sort=True):
        histories = [
            value
            for value in _joined(group["selection_history"]).split(";")
            if value
        ]
        normalized_histories = sorted(
            {normalized_identifier(value) for value in histories if normalized_identifier(value)}
        )
        normalized = normalized_histories[0] if len(normalized_histories) == 1 else ""
        gid_count = int(history_counts.get(normalized, 0)) if normalized else 0
        rows.append(
            {
                "trial_gid": trial_gid,
                "selection_history": ";".join(histories),
                "normalized_selection_history": normalized,
                "selection_history_value_count": len(normalized_histories),
                "selection_history_gid_count": gid_count,
                "selection_history_unique": len(normalized_histories) == 1 and gid_count == 1,
                "trial_cross": _joined(group["trial_cross"]),
            }
        )
    return pd.DataFrame(rows)


def _external_identity(record: dict[str, object]) -> str:
    external_gid = canonical_cimmyt_gid(record.get("external_gid"))
    if external_gid:
        return f"GID:{external_gid}"
    for field in ("external_name", "external_entry", "external_cid", "external_sid"):
        value = normalized_identifier(record.get(field))
        if value:
            return f"{field}:{value}"
    return ""


def external_identity_summary(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(
            columns=[
                "trial_gid",
                "external_gid",
                "external_identity_count",
                "external_record_count",
            ]
        )
    local = records.copy()
    local["trial_gid"] = local["query_id"].map(canonical_gid)
    local = local[local["trial_gid"].ne("")]
    local["external_identity"] = [
        _external_identity(record) for record in local.to_dict("records")
    ]
    rows: list[dict[str, object]] = []
    for trial_gid, group in local.groupby("trial_gid", sort=True):
        identities = sorted(set(group["external_identity"]) - {""})
        gids = sorted(
            {
                canonical_cimmyt_gid(value)
                for value in group.get("external_gid", pd.Series(dtype=str))
                if canonical_cimmyt_gid(value)
            }
        )
        rows.append(
            {
                "trial_gid": trial_gid,
                "external_gid": ";".join(gids),
                "external_identity_count": len(identities),
                "external_record_count": len(group),
            }
        )
    return pd.DataFrame(rows)


def marker_by_sample_axis(
    path: Path, *, delimiter: str = "\t", sentinel: str = "MarkerID"
) -> tuple[dict[str, list[tuple[int, str]]], int]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row_number, line in enumerate(handle):
            values = line.rstrip("\r\n").split(delimiter)
            if values and clean(values[0]) == sentinel:
                lookup: dict[str, list[tuple[int, str]]] = defaultdict(list)
                for column, sample in enumerate(values[1:], start=1):
                    normalized = normalized_identifier(sample)
                    if normalized:
                        lookup[normalized].append((column, clean(sample)))
                return dict(lookup), row_number
    raise ValueError(f"Could not find {sentinel!r} sample header in {path}")


def _dosage_at_fields(
    stripped: bytes,
    tabs: np.ndarray,
    selected_fields: np.ndarray,
    ref: str,
    alt: str,
) -> np.ndarray:
    raw = np.frombuffer(stripped, dtype=np.uint8)
    starts = tabs[selected_fields - 1] + 1
    ends = np.empty_like(selected_fields)
    nonfinal = selected_fields < len(tabs)
    ends[nonfinal] = tabs[selected_fields[nonfinal]]
    ends[~nonfinal] = len(raw)
    dosage = np.full(len(selected_fields), -1, dtype=np.int8)
    for index, (start, end) in enumerate(zip(starts, ends)):
        value = stripped[int(start) : int(end)].decode("utf-8", errors="replace")
        dosage[index] = genotype_call_to_dosage(value, ref, alt)
    return dosage


def stream_marker_by_sample_concordance(
    path: Path,
    *,
    sample_columns: dict[str, int],
    replicate_groups: dict[str, list[str]],
    delimiter: bytes = b"\t",
    sentinel: bytes = b"MarkerID",
    minimum_shared_markers: int,
    minimum_call_concordance: float,
) -> pd.DataFrame:
    if delimiter != b"\t":
        raise ValueError("Selective concordance currently requires a tab-delimited matrix")
    selected_samples = sorted(
        {
            sample
            for samples in replicate_groups.values()
            for sample in samples
            if sample in sample_columns
        }
    )
    if not selected_samples:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    positions = np.asarray([sample_columns[sample] for sample in selected_samples], dtype=np.int64)
    selected_index = {sample: index for index, sample in enumerate(selected_samples)}
    pair_keys: list[tuple[str, str, str]] = []
    for trial_gid, samples in sorted(replicate_groups.items()):
        for left, right in combinations(sorted(set(samples)), 2):
            if left in selected_index and right in selected_index:
                pair_keys.append((trial_gid, left, right))
    overlap = np.zeros(len(pair_keys), dtype=np.int64)
    concordant = np.zeros(len(pair_keys), dtype=np.int64)
    pair_indices = np.asarray(
        [(selected_index[left], selected_index[right]) for _, left, right in pair_keys],
        dtype=np.int64,
    )
    with path.open("rb") as handle:
        for line in handle:
            if line.startswith(sentinel + delimiter):
                break
        else:
            raise ValueError(f"Could not find marker header in {path}")
        expected_fields = len(line.rstrip(b"\r\n").split(delimiter))
        for line_number, line in enumerate(handle, start=2):
            stripped = line.rstrip(b"\r\n")
            if not stripped:
                continue
            raw = np.frombuffer(stripped, dtype=np.uint8)
            tabs = np.flatnonzero(raw == ord("\t"))
            if len(tabs) != expected_fields - 1:
                raise ValueError(
                    f"{path}: line {line_number} has {len(tabs) + 1} fields; "
                    f"expected {expected_fields}"
                )
            marker_id = stripped[: int(tabs[0])].decode("utf-8", errors="replace")
            alleles = marker_alleles(marker_id)
            if alleles is None:
                continue
            dosage = _dosage_at_fields(stripped, tabs, positions, *alleles)
            left = dosage[pair_indices[:, 0]]
            right = dosage[pair_indices[:, 1]]
            observed = (left >= 0) & (right >= 0)
            overlap += observed
            concordant += observed & (left == right)
    rows: list[dict[str, object]] = []
    for index, (trial_gid, left, right) in enumerate(pair_keys):
        shared = int(overlap[index])
        agreement = float(concordant[index] / shared) if shared else np.nan
        overlap_pass = shared >= minimum_shared_markers
        concordance_pass = bool(overlap_pass and agreement >= minimum_call_concordance)
        rows.append(
            {
                "trial_gid": trial_gid,
                "panel_id": "",
                "sample_id_left": left,
                "sample_id_right": right,
                "shared_nonmissing_markers": shared,
                "concordant_markers": int(concordant[index]),
                "call_concordance": agreement,
                "minimum_shared_markers": minimum_shared_markers,
                "minimum_call_concordance": minimum_call_concordance,
                "overlap_pass": overlap_pass,
                "concordance_pass": concordance_pass,
                "pair_status": (
                    "PASS"
                    if concordance_pass
                    else "INSUFFICIENT_OVERLAP"
                    if not overlap_pass
                    else "CONFLICT"
                ),
            }
        )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def classify_replicates(
    pairs: pd.DataFrame,
    expected_sample_count: int,
) -> tuple[str, list[str]]:
    expected_pairs = expected_sample_count * (expected_sample_count - 1) // 2
    if expected_pairs == 0:
        return "accepted_unique_identity", []
    if len(pairs) != expected_pairs:
        return "requires_metadata_review", ["replicate_pair_evidence_incomplete"]
    if (~pairs["overlap_pass"].astype(bool)).any():
        return "requires_metadata_review", ["replicate_overlap_below_panel_minimum"]
    if (~pairs["concordance_pass"].astype(bool)).any():
        return "conflicting_marker_samples", ["replicate_concordance_below_threshold"]
    return "accepted_concordant_replicates", []


def _conflict_lookup(conflicts: pd.DataFrame) -> dict[str, dict[str, str]]:
    if conflicts.empty:
        return {}
    output: dict[str, dict[str, str]] = {}
    for row in conflicts.to_dict("records"):
        gid = canonical_gid(row.get("query_id"))
        if gid:
            output[gid] = {
                "status": clean(row.get("conflict_status")),
                "reasons": clean(row.get("conflict_reasons")),
            }
    return output


def adjudicate_new_candidates(
    *,
    bridges: pd.DataFrame,
    resolver_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    conflicts: pd.DataFrame,
    matrix_path: Path,
    matrix_sha256: str,
    panel_id: str,
    minimum_shared_markers: int,
    minimum_call_concordance: float,
    existing_certified_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    axis, _ = marker_by_sample_axis(matrix_path)
    resolver_lookup = resolver_summary.set_index("trial_gid").to_dict("index")
    external_lookup = external_summary.set_index("trial_gid").to_dict("index")
    conflicts_lookup = _conflict_lookup(conflicts)
    local = bridges.copy()
    local["trial_gid"] = local["query_id"].map(canonical_gid)
    local["normalized_sample_id"] = local["external_alias"].map(normalized_identifier)
    local = local[
        local["trial_gid"].ne("") & local["normalized_sample_id"].ne("")
    ].copy()
    source_rows: list[dict[str, object]] = []
    grouped_samples: dict[str, list[str]] = {}
    group_precheck: dict[str, tuple[list[str], dict[str, object]]] = {}
    for trial_gid, group in local.groupby("trial_gid", sort=True):
        identity = resolver_lookup.get(trial_gid, {})
        external = external_lookup.get(trial_gid, {})
        conflict = conflicts_lookup.get(
            trial_gid, {"status": "NO_EXTERNAL_CONFLICT_RECORD", "reasons": ""}
        )
        direct_gid_mapping = group["mapping_filename"].fillna("").str.contains(
            "SampleIDvsGID", case=False, regex=False
        ).any()
        external_count = int(external.get("external_identity_count", 0) or 0)
        external_gid = clean(external.get("external_gid"))
        if direct_gid_mapping and external_count == 0:
            external_count = 1
            external_gid = trial_gid
        samples = sorted(set(group["normalized_sample_id"]) - {""})
        reasons: list[str] = []
        if not bool(identity.get("selection_history_unique", False)):
            reasons.append("selection_history_not_unique")
        if external_count != 1:
            reasons.append("multiple_external_germplasm_identities")
        if conflict.get("status") not in {
            "",
            "NO_DETECTED_CONFLICT",
            "NO_EXTERNAL_CONFLICT_RECORD",
        }:
            reasons.append("pedigree_or_cross_conflict")
        if not samples:
            reasons.append("family_level_identity_only")
        if (
            not direct_gid_mapping
            and "selection_history_not_unique" in reasons
            and "family_level_identity_only" not in reasons
        ):
            reasons.append("family_level_identity_only")
        marker_locations = {
            sample: axis.get(sample, []) for sample in samples
        }
        if any(len(locations) != 1 for locations in marker_locations.values()):
            reasons.append("marker_sample_not_uniquely_located")
        family_only = "family_level_identity_only" in reasons
        metadata_blocked = bool(reasons)
        preliminary = (
            "family_only_not_assignable"
            if family_only
            else "requires_metadata_review"
            if metadata_blocked
            else "pending_replicate_concordance"
            if len(samples) > 1
            else "accepted_unique_identity"
        )
        grouped_samples[trial_gid] = samples
        group_precheck[trial_gid] = (reasons, {"preliminary": preliminary})
        for sample in samples or [""]:
            rows = group[group["normalized_sample_id"].eq(sample)] if sample else group
            source = rows.iloc[0].to_dict()
            locations = marker_locations.get(sample, [])
            column = locations[0][0] if len(locations) == 1 else ""
            display_sample = locations[0][1] if len(locations) == 1 else clean(
                source.get("external_alias")
            )
            source_rows.append(
                {
                    "trial_gid": trial_gid,
                    "candidate_scope": "new_dataverse_two_hop",
                    "panel_id": panel_id,
                    "selection_history": identity.get("selection_history", ""),
                    "normalized_selection_history": identity.get(
                        "normalized_selection_history", ""
                    ),
                    "selection_history_gid_count": int(
                        identity.get("selection_history_gid_count", 0) or 0
                    ),
                    "selection_history_unique": bool(
                        identity.get("selection_history_unique", False)
                    ),
                    "trial_cross": identity.get("trial_cross", ""),
                    "external_gid": external_gid,
                    "external_alias": clean(source.get("external_alias")),
                    "sample_id": display_sample,
                    "normalized_sample_id": sample,
                    "mapping_filename": clean(source.get("mapping_filename")),
                    "mapping_source_part": clean(source.get("mapping_source_part")),
                    "mapping_source_row": source.get("mapping_source_row", ""),
                    "marker_matrix_path": str(matrix_path),
                    "marker_matrix_sha256": matrix_sha256,
                    "marker_matrix_axis": "sample_column",
                    "marker_matrix_axis_index": column,
                    "marker_matrix_locator": f"column:{column}" if column != "" else "",
                    "marker_axis_match_count": len(locations),
                    "external_identity_count": external_count,
                    "external_record_count": int(
                        external.get("external_record_count", int(direct_gid_mapping)) or 0
                    ),
                    "pedigree_conflict_status": conflict.get("status", ""),
                    "pedigree_conflict_reasons": conflict.get("reasons", ""),
                    "existing_certified_gid": trial_gid in existing_certified_ids,
                    "classification": preliminary,
                    "classification_reasons": ";".join(reasons),
                    "direct_marker_assignment_ready": preliminary
                    == "accepted_unique_identity",
                }
            )
    replicates = {
        gid: samples
        for gid, samples in grouped_samples.items()
        if len(samples) > 1 and group_precheck[gid][1]["preliminary"] == "pending_replicate_concordance"
    }
    sample_columns = {
        sample: axis[sample][0][0]
        for samples in replicates.values()
        for sample in samples
        if len(axis.get(sample, [])) == 1
    }
    pairs = stream_marker_by_sample_concordance(
        matrix_path,
        sample_columns=sample_columns,
        replicate_groups=replicates,
        minimum_shared_markers=minimum_shared_markers,
        minimum_call_concordance=minimum_call_concordance,
    )
    if not pairs.empty:
        pairs["panel_id"] = panel_id
    candidates = pd.DataFrame(source_rows, columns=CANDIDATE_COLUMNS)
    for trial_gid, samples in replicates.items():
        local_pairs = pairs[pairs["trial_gid"].eq(trial_gid)]
        classification, reasons = classify_replicates(local_pairs, len(samples))
        mask = candidates["trial_gid"].eq(trial_gid)
        candidates.loc[mask, "classification"] = classification
        candidates.loc[mask, "classification_reasons"] = ";".join(reasons)
        candidates.loc[mask, "direct_marker_assignment_ready"] = classification in {
            "accepted_unique_identity",
            "accepted_concordant_replicates",
        }
    pending = candidates["classification"].eq("pending_replicate_concordance")
    if pending.any():
        raise ValueError("Internal error: replicate candidates were not assigned a terminal class")
    return candidates, pairs


def load_existing_panel_candidates(
    *,
    root: Path,
    panel_specs: list[dict[str, object]],
    resolver_summary: pd.DataFrame,
    minimum_call_concordance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resolver_lookup = resolver_summary.set_index("trial_gid").to_dict("index")
    candidate_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    for spec in panel_specs:
        panel_id = str(spec["panel_id"])
        artifact_dir = resolve(root, str(spec["artifact_dir"]))
        prefix = str(spec["prefix"])
        sample_qc_path = artifact_dir / f"{prefix}_sample_qc.tsv"
        pair_path = artifact_dir / f"{prefix}_duplicate_call_concordance.tsv"
        order_path = artifact_dir / f"{prefix}_sample_order.tsv"
        minimum_shared = int(spec.get("minimum_shared_markers", 1000))
        if not sample_qc_path.is_file():
            inventory_rows.append(
                {
                    "panel_id": panel_id,
                    "scope": "existing_platform_reaudit",
                    "status": "NOT_AVAILABLE",
                    "candidate_gids": 0,
                    "accepted_gids": 0,
                    "detail": f"missing {sample_qc_path}",
                }
            )
            continue
        sample_qc = read_table(sample_qc_path)
        if "sample_id" not in sample_qc.columns:
            raise ValueError(f"Existing panel sample QC lacks sample_id: {sample_qc_path}")
        source_col = (
            "source_sample_id" if "source_sample_id" in sample_qc.columns else "source_row"
        )
        sample_qc["trial_gid"] = sample_qc["sample_id"].map(canonical_gid)
        sample_qc["source_identity"] = sample_qc[source_col].map(clean)
        sample_qc = sample_qc[
            sample_qc["trial_gid"].ne("") & sample_qc["source_identity"].ne("")
        ].copy()
        certified_ids: set[str] = set()
        if order_path.is_file():
            order = read_table(order_path)
            if "sample_id" in order.columns:
                certified_ids = {canonical_gid(value) for value in order["sample_id"]}
                certified_ids.discard("")
        pair_evidence = (
            read_table(pair_path)
            if pair_path.is_file()
            else pd.DataFrame(
                columns=[
                    "sample_id",
                    "source_sample_left",
                    "source_sample_right",
                    "overlapping_observed_markers",
                    "call_concordance",
                ]
            )
        )
        panel_candidates: list[dict[str, object]] = []
        panel_pairs: list[dict[str, object]] = []
        for trial_gid, group in sample_qc.groupby("trial_gid", sort=True):
            sources = sorted(set(group["source_identity"]) - {""})
            local_pairs = pair_evidence[
                pair_evidence.get("sample_id", pd.Series(dtype=str)).map(canonical_gid).eq(
                    trial_gid
                )
            ]
            for row in local_pairs.to_dict("records"):
                shared = int(float(row.get("overlapping_observed_markers", 0) or 0))
                concordance = pd.to_numeric(row.get("call_concordance"), errors="coerce")
                overlap_pass = shared >= minimum_shared
                concordance_pass = bool(
                    overlap_pass
                    and pd.notna(concordance)
                    and float(concordance) >= minimum_call_concordance
                )
                panel_pairs.append(
                    {
                        "trial_gid": trial_gid,
                        "panel_id": panel_id,
                        "sample_id_left": clean(row.get("source_sample_left")),
                        "sample_id_right": clean(row.get("source_sample_right")),
                        "shared_nonmissing_markers": shared,
                        "concordant_markers": "",
                        "call_concordance": concordance,
                        "minimum_shared_markers": minimum_shared,
                        "minimum_call_concordance": minimum_call_concordance,
                        "overlap_pass": overlap_pass,
                        "concordance_pass": concordance_pass,
                        "pair_status": (
                            "PASS"
                            if concordance_pass
                            else "INSUFFICIENT_OVERLAP"
                            if not overlap_pass
                            else "CONFLICT"
                        ),
                    }
                )
            pair_frame = pd.DataFrame(panel_pairs, columns=PAIR_COLUMNS)
            gid_pairs = pair_frame[pair_frame["trial_gid"].eq(trial_gid)]
            classification, reasons = classify_replicates(gid_pairs, len(sources))
            identity = resolver_lookup.get(trial_gid, {})
            for source in sources:
                source_row = group[group["source_identity"].eq(source)].iloc[0]
                panel_candidates.append(
                    {
                        "trial_gid": trial_gid,
                        "candidate_scope": "existing_platform_reaudit",
                        "panel_id": panel_id,
                        "selection_history": identity.get("selection_history", ""),
                        "normalized_selection_history": identity.get(
                            "normalized_selection_history", ""
                        ),
                        "selection_history_gid_count": int(
                            identity.get("selection_history_gid_count", 0) or 0
                        ),
                        "selection_history_unique": bool(
                            identity.get("selection_history_unique", False)
                        ),
                        "trial_cross": identity.get("trial_cross", ""),
                        "external_gid": trial_gid,
                        "external_alias": source,
                        "sample_id": source,
                        "normalized_sample_id": normalized_identifier(source),
                        "mapping_filename": str(sample_qc_path),
                        "mapping_source_part": "certified_platform_sample_qc",
                        "mapping_source_row": source_row.get("source_row", ""),
                        "marker_matrix_path": "",
                        "marker_matrix_sha256": "",
                        "marker_matrix_axis": "builder_matched_sample_index",
                        "marker_matrix_axis_index": source_row.get("source_row", ""),
                        "marker_matrix_locator": f"source_sample_id:{source}",
                        "marker_axis_match_count": 1,
                        "external_identity_count": 1,
                        "external_record_count": 1,
                        "pedigree_conflict_status": "DIRECT_CANONICAL_GID",
                        "pedigree_conflict_reasons": "",
                        "existing_certified_gid": trial_gid in certified_ids,
                        "classification": classification,
                        "classification_reasons": ";".join(reasons),
                        "direct_marker_assignment_ready": classification in {
                            "accepted_unique_identity",
                            "accepted_concordant_replicates",
                        },
                    }
                )
        panel_frame = pd.DataFrame(panel_candidates, columns=CANDIDATE_COLUMNS)
        pairs_frame = pd.DataFrame(panel_pairs, columns=PAIR_COLUMNS)
        candidate_frames.append(panel_frame)
        pair_frames.append(pairs_frame)
        inventory_rows.append(
            {
                "panel_id": panel_id,
                "scope": "existing_platform_reaudit",
                "status": "COMPLETE",
                "candidate_gids": panel_frame["trial_gid"].nunique(),
                "accepted_gids": panel_frame.loc[
                    panel_frame["direct_marker_assignment_ready"].astype(bool), "trial_gid"
                ].nunique(),
                "detail": "identity based on direct canonical GID; duplicates rechecked under frozen policy",
            }
        )
    candidates = (
        pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames
        else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    )
    pairs = (
        pd.concat(pair_frames, ignore_index=True)
        if pair_frames
        else pd.DataFrame(columns=PAIR_COLUMNS)
    )
    return candidates, pairs, pd.DataFrame(inventory_rows)


def direct_certified_panel_inventory(
    root: Path, panel_specs: list[dict[str, object]]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in panel_specs:
        panel_id = str(spec["panel_id"])
        order_path = resolve(root, str(spec["sample_order_path"]))
        if not order_path.is_file():
            rows.append(
                {
                    "panel_id": panel_id,
                    "scope": "direct_certified_panel",
                    "status": "NOT_AVAILABLE",
                    "candidate_gids": 0,
                    "accepted_gids": 0,
                    "detail": f"missing {order_path}",
                }
            )
            continue
        order = read_table(order_path)
        if "sample_id" not in order.columns:
            raise ValueError(f"Certified panel order lacks sample_id: {order_path}")
        gids = {canonical_gid(value) for value in order["sample_id"]}
        gids.discard("")
        rows.append(
            {
                "panel_id": panel_id,
                "scope": "direct_certified_panel",
                "status": "CERTIFIED_EXISTING_IDENTITY",
                "candidate_gids": len(gids),
                "accepted_gids": len(gids),
                "detail": str(order_path),
            }
        )
    return pd.DataFrame(rows)


def classification_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "candidate_scope",
                "panel_id",
                "classification",
                "candidate_gids",
                "sample_rows",
                "new_candidate_gids",
            ]
        )
    rows: list[dict[str, object]] = []
    keys = ["candidate_scope", "panel_id", "classification"]
    for values, group in candidates.groupby(keys, dropna=False, sort=True):
        rows.append(
            {
                **dict(zip(keys, values)),
                "candidate_gids": group["trial_gid"].nunique(),
                "sample_rows": len(group),
                "new_candidate_gids": group.loc[
                    ~group["existing_certified_gid"].astype(bool), "trial_gid"
                ].nunique(),
            }
        )
    return pd.DataFrame(rows)


def regulatory_overlay(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trial_gid, group in candidates.groupby("trial_gid", sort=True):
        accepted = group[group["direct_marker_assignment_ready"].astype(bool)]
        unresolved_group = group[
            ~group["direct_marker_assignment_ready"].astype(bool)
        ]
        classes = sorted(set(group["classification"]))
        panels = sorted(set(group["panel_id"]) - {""})
        accepted_panels = sorted(set(accepted["panel_id"]) - {""})
        if not accepted.empty and not unresolved_group.empty:
            status = "accepted_identity_and_candidate_unresolved"
        elif not accepted.empty:
            status = "accepted_identity_marker_qc_pending"
        else:
            status = "candidate_unresolved"
        rows.append(
            {
                "canonical_gid": trial_gid,
                "marker_identity_adjudication_status": status,
                "marker_identity_classes": ";".join(classes),
                "candidate_marker_panels": ";".join(panels),
                "accepted_marker_panels": ";".join(accepted_panels),
                "candidate_unresolved": not unresolved_group.empty,
                "accepted_for_new_kernel_input": not accepted.empty,
                "eligible_for_K_G": False,
                "eligible_for_K_z": False,
                "eligible_for_genotype_specific_sequence": False,
                "next_required_action": (
                    "build_and_certify_panel_specific_genotype_artifact"
                    if not accepted.empty
                    else "resolve_identity_or_marker_sample_conflict"
                ),
            }
        )
    return pd.DataFrame(rows)


def accepted_entities(candidates: pd.DataFrame) -> pd.DataFrame:
    accepted = candidates[candidates["direct_marker_assignment_ready"].astype(bool)]
    rows: list[dict[str, object]] = []
    for (trial_gid, panel_id), group in accepted.groupby(
        ["trial_gid", "panel_id"], sort=True
    ):
        classes = sorted(set(group["classification"]))
        if len(classes) != 1:
            raise ValueError(f"Accepted identity has inconsistent classes: {trial_gid}/{panel_id}")
        samples = sorted(set(group["sample_id"].map(clean)) - {""})
        if not samples:
            raise ValueError(f"Accepted identity has no marker sample: {trial_gid}/{panel_id}")
        classification = classes[0]
        rows.append(
            {
                "trial_gid": trial_gid,
                "panel_id": panel_id,
                "classification": classification,
                "accepted_sample_count": len(samples),
                "accepted_sample_ids": ";".join(samples),
                "representative_sample_id": samples[0],
                "collapse_status": (
                    "approved_concordant_technical_replicates"
                    if classification == "accepted_concordant_replicates"
                    else "not_required_unique_sample"
                ),
                "collapse_materialized": False,
                "kernel_input_status": "panel_specific_qc_and_matrix_rebuild_required",
            }
        )
    return pd.DataFrame(rows)


def validation_checks(
    candidates: pd.DataFrame,
    pairs: pd.DataFrame,
    overlay: pd.DataFrame,
    expected_new_candidate_ids: set[str],
) -> pd.DataFrame:
    accepted_classes = {
        "accepted_unique_identity",
        "accepted_concordant_replicates",
    }
    checks: list[dict[str, object]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append(
            {"check": check, "status": "PASS" if passed else "FAIL", "detail": detail}
        )

    classes = set(candidates["classification"])
    add(
        "terminal_classification",
        classes.issubset(TERMINAL_CLASSES),
        f"observed={sorted(classes)}",
    )
    ready = candidates["direct_marker_assignment_ready"].astype(bool)
    add(
        "assignment_ready_only_for_accepted_classes",
        candidates.loc[ready, "classification"].isin(accepted_classes).all()
        and (~candidates.loc[~ready, "classification"].isin(accepted_classes)).all(),
        f"ready_rows={int(ready.sum())}",
    )
    group_classes = candidates.groupby(["trial_gid", "panel_id"])[
        "classification"
    ].nunique()
    add(
        "one_terminal_class_per_gid_panel",
        group_classes.le(1).all(),
        f"max_classes={int(group_classes.max()) if len(group_classes) else 0}",
    )
    observed_new_ids = set(
        candidates.loc[
            candidates["candidate_scope"].eq("new_dataverse_two_hop"), "trial_gid"
        ]
    )
    add(
        "all_two_hop_candidates_classified",
        observed_new_ids == expected_new_candidate_ids,
        f"expected={len(expected_new_candidate_ids)}; observed={len(observed_new_ids)}",
    )
    new_accepted = candidates[
        ready & candidates["candidate_scope"].eq("new_dataverse_two_hop")
    ]
    add(
        "new_accepted_matrix_axis_is_unique",
        new_accepted["marker_axis_match_count"].astype(int).eq(1).all()
        and new_accepted["marker_matrix_locator"].map(clean).ne("").all(),
        f"accepted_rows={len(new_accepted)}",
    )
    pair_concordance = pd.to_numeric(pairs["call_concordance"], errors="coerce")
    add(
        "pairwise_concordance_is_finite_when_overlap_passes",
        pair_concordance[pairs["overlap_pass"].astype(bool)].notna().all(),
        f"pair_rows={len(pairs)}",
    )
    add(
        "regulatory_overlay_is_gated",
        not overlay["eligible_for_K_G"].astype(bool).any()
        and not overlay["eligible_for_K_z"].astype(bool).any()
        and not overlay["eligible_for_genotype_specific_sequence"].astype(bool).any(),
        f"overlay_rows={len(overlay)}",
    )
    return pd.DataFrame(checks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Adjudicate trial-to-marker identities using identifiers, pedigree metadata, "
            "and selective marker-call concordance only."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(
            "server_genotype_recovery/marker_identity_concordance_policy_v1.json"
        ),
    )
    parser.add_argument(
        "--resolver-query",
        type=Path,
        default=Path("genotype_panels/germplasm_resolver/germplasm_cross_query.tsv"),
    )
    parser.add_argument(
        "--two-hop-dir",
        type=Path,
        default=Path(
            "genotype_panels/cimmyt_dataverse_recovery_v1/wide_inventory_v1/"
            "structured_evidence/two_hop_marker_bridges"
        ),
    )
    parser.add_argument(
        "--pedigree-enrichment-dir",
        type=Path,
        default=Path(
            "genotype_panels/cimmyt_dataverse_recovery_v1/wide_inventory_v1/"
            "structured_evidence/pedigree_enrichment"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("genotype_panels/marker_identity_adjudication_v1"),
    )
    parser.add_argument("--skip-existing-panel-reaudit", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    policy_path = resolve(root, args.policy)
    resolver_path = resolve(root, args.resolver_query)
    two_hop_dir = resolve(root, args.two_hop_dir)
    pedigree_dir = resolve(root, args.pedigree_enrichment_dir)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "policy": policy_path,
        "resolver": resolver_path,
        "bridges": two_hop_dir / "dataverse_two_hop_marker_bridges.tsv",
        "external_records": pedigree_dir / "dataverse_pedigree_external_records.tsv",
        "conflicts": pedigree_dir / "dataverse_pedigree_conflicts.tsv",
        "two_hop_provenance": (
            two_hop_dir / "dataverse_two_hop_marker_bridge_provenance.json"
        ),
        "pedigree_provenance": (
            pedigree_dir / "dataverse_pedigree_enrichment_provenance.json"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Identity adjudication inputs are missing: {missing}")
    policy = load_policy(policy_path)
    upstream_hashes = validate_upstream_provenance(
        two_hop_provenance_path=paths["two_hop_provenance"],
        pedigree_provenance_path=paths["pedigree_provenance"],
        resolver_path=resolver_path,
    )
    resolver = read_table(resolver_path)
    resolver_summary = resolver_identity_summary(resolver)
    bridges = read_table(paths["bridges"])
    records = read_table(paths["external_records"])
    conflicts = read_table(paths["conflicts"])
    external_summary = external_identity_summary(records)
    new_spec = dict(policy["new_candidate_panel"])
    matrix_path = resolve(root, str(new_spec["canonical_matrix_path"]))
    if not matrix_path.is_file() or matrix_path.stat().st_size == 0:
        raise FileNotFoundError(f"Canonical candidate marker matrix is missing: {matrix_path}")
    print(f"Hashing canonical marker matrix: {matrix_path}", flush=True)
    matrix_sha256 = sha256_file_with_progress(matrix_path)
    print(f"Canonical marker matrix SHA256: {matrix_sha256}", flush=True)
    existing_seed_order = (
        root
        / "genotype_panels/recovered/seeds_dartseq/"
        "K_G_SEEDS_DARTSEQ_sample_order.tsv"
    )
    existing_ids = set()
    if existing_seed_order.is_file():
        order = read_table(existing_seed_order)
        if "sample_id" in order.columns:
            existing_ids = {canonical_gid(value) for value in order["sample_id"]}
            existing_ids.discard("")
    candidates, pairs = adjudicate_new_candidates(
        bridges=bridges,
        resolver_summary=resolver_summary,
        external_summary=external_summary,
        conflicts=conflicts,
        matrix_path=matrix_path,
        matrix_sha256=matrix_sha256,
        panel_id=str(new_spec["panel_id"]),
        minimum_shared_markers=int(
            new_spec.get(
                "minimum_shared_markers", policy["minimum_shared_markers_default"]
            )
        ),
        minimum_call_concordance=float(
            policy["minimum_pairwise_call_concordance"]
        ),
        existing_certified_ids=existing_ids,
    )
    inventory = pd.DataFrame(
        [
            {
                "panel_id": new_spec["panel_id"],
                "scope": "new_dataverse_two_hop",
                "status": "COMPLETE",
                "candidate_gids": candidates["trial_gid"].nunique(),
                "accepted_gids": candidates.loc[
                    candidates["direct_marker_assignment_ready"].astype(bool), "trial_gid"
                ].nunique(),
                "detail": "canonical matrix axis verified; duplicate samples selectively streamed",
            }
        ]
    )
    if not args.skip_existing_panel_reaudit:
        existing_candidates, existing_pairs, existing_inventory = load_existing_panel_candidates(
            root=root,
            panel_specs=list(policy.get("existing_panel_artifacts", [])),
            resolver_summary=resolver_summary,
            minimum_call_concordance=float(
                policy["minimum_pairwise_call_concordance"]
            ),
        )
        candidates = pd.concat([candidates, existing_candidates], ignore_index=True)
        pairs = pd.concat([pairs, existing_pairs], ignore_index=True)
        inventory = pd.concat([inventory, existing_inventory], ignore_index=True)
    direct_inventory = direct_certified_panel_inventory(
        root, list(policy.get("direct_certified_panel_orders", []))
    )
    inventory = pd.concat([inventory, direct_inventory], ignore_index=True)
    if not set(candidates["classification"]).issubset(TERMINAL_CLASSES):
        invalid = sorted(set(candidates["classification"]) - TERMINAL_CLASSES)
        raise ValueError(f"Nonterminal identity classifications remain: {invalid}")
    accepted = candidates[candidates["direct_marker_assignment_ready"].astype(bool)].copy()
    conflicting = candidates[
        candidates["classification"].eq("conflicting_marker_samples")
    ].copy()
    unresolved = candidates[
        candidates["classification"].isin(
            {"requires_metadata_review", "family_only_not_assignable"}
        )
    ].copy()
    overlay = regulatory_overlay(candidates)
    accepted_entity_manifest = accepted_entities(candidates)
    summary = classification_summary(candidates)
    expected_new_candidate_ids = {
        canonical_gid(value) for value in bridges["query_id"] if canonical_gid(value)
    }
    checks = validation_checks(
        candidates, pairs, overlay, expected_new_candidate_ids
    )
    if checks["status"].eq("FAIL").any():
        failed = checks[checks["status"].eq("FAIL")].to_dict("records")
        raise ValueError(f"Marker identity adjudication validation failed: {failed}")
    candidates.to_csv(
        out_dir / "marker_identity_candidate_paths.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pairs.to_csv(
        out_dir / "marker_identity_pairwise_concordance.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    accepted.to_csv(out_dir / "marker_identity_accepted.tsv", sep="\t", index=False)
    accepted_entity_manifest.to_csv(
        out_dir / "marker_identity_accepted_entities.tsv", sep="\t", index=False
    )
    unresolved.to_csv(out_dir / "marker_identity_unresolved.tsv", sep="\t", index=False)
    conflicting.to_csv(out_dir / "marker_identity_conflicting.tsv", sep="\t", index=False)
    overlay.to_csv(out_dir / "regulatory_eligibility_overlay.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "marker_identity_classification_summary.tsv", sep="\t", index=False)
    inventory.to_csv(out_dir / "marker_identity_panel_inventory.tsv", sep="\t", index=False)
    checks.to_csv(out_dir / "marker_identity_validation.tsv", sep="\t", index=False)
    qc = pd.DataFrame(
        [
            {"metric": "run_status", "value": "PASS"},
            {"metric": "protocol_version", "value": policy["protocol_version"]},
            {"metric": "candidate_gids", "value": candidates["trial_gid"].nunique()},
            {"metric": "new_two_hop_candidate_gids", "value": bridges["query_id"].map(canonical_gid).nunique()},
            {"metric": "accepted_candidate_gids", "value": accepted["trial_gid"].nunique()},
            {"metric": "accepted_new_candidate_gids", "value": accepted.loc[~accepted["existing_certified_gid"].astype(bool), "trial_gid"].nunique()},
            {"metric": "unresolved_candidate_gids", "value": unresolved["trial_gid"].nunique()},
            {"metric": "conflicting_candidate_gids", "value": conflicting["trial_gid"].nunique()},
            {"metric": "phenotype_values_read", "value": False},
            {"metric": "outer_test_metrics_read", "value": False},
            {"metric": "final_holdout_outcomes_read", "value": False},
            {"metric": "kernels_modified", "value": False},
        ]
    )
    qc.to_csv(out_dir / "marker_identity_adjudication_qc.tsv", sep="\t", index=False)
    provenance = {
        "status": "PASS",
        "protocol_version": policy["protocol_version"],
        "selection_data": policy["selection_data"],
        "policy_path": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "upstream_snapshot_hashes": upstream_hashes,
        "canonical_marker_matrix": str(matrix_path),
        "canonical_marker_matrix_sha256": matrix_sha256,
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "kernels_modified": False,
        "accepted_mappings_require_panel_specific_qc_before_K_G_or_K_z": True,
        "unresolved_mappings_allowed_in_K_G_or_K_z": False,
        "unresolved_mappings_allowed_for_genotype_specific_sequence": False,
    }
    write_json_atomic(provenance, out_dir / "marker_identity_adjudication_provenance.json")
    print(qc.to_string(index=False))
    print("\n=== CLASSIFICATION SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== PANEL INVENTORY ===")
    print(inventory.to_string(index=False))


if __name__ == "__main__":
    main()
