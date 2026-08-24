from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from server_genotype_recovery.adjudicate_marker_identity_candidates import (
    marker_by_sample_axis,
    resolver_identity_summary,
    stream_marker_by_sample_concordance,
)
from server_genotype_recovery.audit_dataverse_pedigree_enrichment import (
    canonical_cimmyt_gid,
)
from server_genotype_recovery.audit_dataverse_two_hop_marker_bridges import (
    canonical_gid,
)
from server_genotype_recovery.build_canonical_pedigree import (
    PedigreeNode,
    RegistryBuilder,
    parse_purdy_pedigree,
)
from server_genotype_recovery.build_regulatory_eligibility_manifest import (
    detect_column,
)
from server_genotype_recovery.fetch_brapi_pedigree_markers import (
    clean,
    normalized_identifier,
    read_table,
    sha256_file,
)
from server_training_pipeline.nested_evaluation import (
    assign_nested_split,
    verify_manifest_contract,
)


MARKER_ACCEPTED = {
    "accepted_direct_gid_to_marker_sample",
    "accepted_unique_two_hop_identity",
    "accepted_concordant_technical_replicates",
}
MARKER_CLASSES = MARKER_ACCEPTED | {
    "metadata_match_without_marker_sample",
    "family_only_not_assignable",
    "ambiguous_multiple_external_ids",
    "ambiguous_multiple_marker_samples",
    "conflicting_marker_calls",
    "conflicting_identity_metadata",
    "non_wheat_excluded",
    "insufficient_marker_overlap",
    "unresolved",
}
PEDIGREE_ACCEPTED = {
    "already_present",
    "accepted_new_edge_exact_unique",
    "accepted_new_edge_corroborated",
}
PEDIGREE_CLASSES = PEDIGREE_ACCEPTED | {
    "unresolved_parent_identity",
    "deeper_ancestor_not_direct_parent",
    "conflicts_existing_complete_parent_pair",
    "conflicting_external_records",
    "ambiguous_cross_parse",
    "insufficient_provenance",
}
ALLOWED_LEDGER_COLUMNS = {
    "panel_sample_id",
    "env_kernel_id",
    "cycle",
    "country",
    "trait_name_canonical",
}
REQUIRED_OUTPUTS = [
    "verification_contract.json",
    "verification_input_inventory.tsv",
    "marker_candidate_paths.tsv.gz",
    "marker_identity_verification.tsv",
    "accepted_gid_marker_mapping.tsv",
    "accepted_marker_replicate_groups.tsv",
    "marker_replicate_concordance.tsv",
    "unresolved_marker_candidates.tsv",
    "marker_conflicts.tsv",
    "pedigree_candidate_edge_verification.tsv",
    "accepted_new_pedigree_edges.tsv",
    "pedigree_conflicts.tsv",
    "recovered_structural_coverage.tsv",
    "recovered_fold_support.tsv",
    "recovered_regulatory_eligibility.tsv",
    "single_step_H_input_readiness.tsv",
    "verification_qc.tsv",
    "verification_provenance.json",
    "verification_parent_registry.tsv",
]


def resolve(root: Path, value: Path | str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json(path: Path, payload: object) -> None:
    path.write_bytes(json_bytes(payload))


def write_table(frame: pd.DataFrame, path: Path, sort_by: list[str] | None = None) -> None:
    local = frame.copy()
    if sort_by and not local.empty:
        present = [column for column in sort_by if column in local.columns]
        if present:
            local = local.sort_values(present, kind="stable", na_position="last")
    local.to_csv(path, sep="\t", index=False, lineterminator="\n")


def write_deterministic_gzip_table(
    frame: pd.DataFrame, path: Path, sort_by: list[str] | None = None
) -> None:
    local = frame.copy()
    if sort_by and not local.empty:
        present = [column for column in sort_by if column in local.columns]
        if present:
            local = local.sort_values(present, kind="stable", na_position="last")
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                local.to_csv(text, sep="\t", index=False, lineterminator="\n")


def bool_value(value: object) -> bool:
    return clean(value).upper() in {"1", "TRUE", "T", "YES", "Y", "PASS"}


def stable_id(prefix: str, *values: object) -> str:
    payload = "\0".join(clean(value) for value in values)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16].upper()}"


def required_file(path: Path, role: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required {role} is missing or empty: {path}")
    return path


def load_policy(path: Path) -> dict[str, object]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if set(policy["accepted_marker_classes"]) != MARKER_ACCEPTED:
        raise ValueError("Verification policy marker acceptance classes are stale")
    if set(policy["accepted_pedigree_classes"]) != PEDIGREE_ACCEPTED:
        raise ValueError("Verification policy pedigree acceptance classes are stale")
    if float(policy["technical_replicate_concordance_threshold"]) < 0.995:
        raise ValueError("Technical-replicate concordance cannot be below 0.995")
    if int(policy["minimum_shared_nonmissing_marker_calls"]) < 1000:
        raise ValueError("Shared nonmissing marker-call minimum cannot be below 1000")
    prohibited = set(policy.get("prohibited_inputs", []))
    required = {
        "phenotype_values",
        "outer_test_metrics",
        "final_holdout_outcomes",
        "model_performance",
    }
    if not required.issubset(prohibited):
        raise ValueError("Verification policy does not prohibit outcome-based selection")
    return policy


def load_identity_policy(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def certified_order_paths(root: Path, identity_policy: dict[str, object]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for spec in identity_policy.get("existing_panel_artifacts", []):
        artifact = resolve(root, str(spec["artifact_dir"]))
        paths[str(spec["panel_id"])] = artifact / f"{spec['prefix']}_sample_order.tsv"
    for spec in identity_policy.get("direct_certified_panel_orders", []):
        paths[str(spec["panel_id"])] = resolve(root, str(spec["sample_order_path"]))
    return paths


def add_discovered_order_paths(
    root: Path, paths: dict[str, Path], out_dir: Path
) -> dict[str, Path]:
    output = dict(paths)
    known = {path.resolve() for path in output.values()}
    genotype_root = root / "genotype_panels"
    if not genotype_root.is_dir():
        return output
    for path in sorted(genotype_root.rglob("*sample_order*.tsv")):
        resolved = path.resolve()
        if resolved in known or out_dir in resolved.parents:
            continue
        relative_path = path.relative_to(genotype_root)
        if not relative_path.parts or relative_path.parts[0].lower() != "recovered":
            continue
        try:
            header = pd.read_csv(path, sep="\t", nrows=0).columns
        except Exception:
            continue
        if not any(
            column in header
            for column in ("sample_id", "panel_sample_id", "genotype_id")
        ):
            continue
        relative = relative_path.as_posix()
        panel = "DISCOVERED_ORDER:" + re.sub(r"[^A-Za-z0-9]+", "_", relative).strip("_")
        output[panel] = resolved
        known.add(resolved)
    return output


def load_order_ids(path: Path) -> set[str]:
    frame = read_table(path)
    column = detect_column(
        frame, ["sample_id", "panel_sample_id", "panel_sample_id_expected", "genotype_id"]
    )
    if column is None:
        raise ValueError(f"Order has no recognized genotype ID column: {path}")
    values = [canonical_gid(value) or clean(value) for value in frame[column]]
    values = [value for value in values if value]
    if len(values) != len(set(values)):
        raise ValueError(f"Order contains duplicate genotype IDs: {path}")
    return set(values)


def inventory_inputs(paths: dict[str, Path], declared_hashes: dict[str, str] | None = None) -> pd.DataFrame:
    declared_hashes = declared_hashes or {}
    rows: list[dict[str, object]] = []
    for role, path in sorted(paths.items()):
        required_file(path, role)
        rows.append(
            {
                "input_role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": declared_hashes.get(role) or sha256_file(path),
                "hash_basis": "upstream_certified_sha256" if role in declared_hashes else "file_bytes",
            }
        )
    return pd.DataFrame(rows)


def declared_marker_matrix_inputs(
    root: Path, candidate_manifest: pd.DataFrame
) -> tuple[dict[str, Path], dict[str, str]]:
    required = {"marker_matrix_path", "marker_matrix_sha256"}
    missing = sorted(required - set(candidate_manifest.columns))
    if missing:
        raise ValueError(f"Marker candidate manifest is missing matrix identity columns: {missing}")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    selected = candidate_manifest[
        candidate_manifest["marker_matrix_path"].map(clean).ne("")
    ].copy()
    for matrix_value, group in selected.groupby("marker_matrix_path", sort=True):
        path = resolve(root, clean(matrix_value))
        declared = sorted(set(group["marker_matrix_sha256"].map(clean)) - {""})
        if len(declared) != 1 or not re.fullmatch(r"[0-9a-fA-F]{64}", declared[0]):
            raise ValueError(
                f"Marker matrix does not have one valid declared SHA256: {path}"
            )
        role = f"declared_marker_matrix:{len(paths):03d}"
        paths[role] = path
        hashes[role] = declared[0].lower()
    return paths, hashes


def freeze_contract(
    out_dir: Path,
    policy: dict[str, object],
    policy_path: Path,
    inventory: pd.DataFrame,
    *,
    concordance_mode: str,
    force: bool,
) -> dict[str, object]:
    contract = {
        **policy,
        "status": "frozen_before_candidate_resolution",
        "policy_path": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "replicate_concordance_mode": concordance_mode,
        "input_hashes": {
            row.input_role: {"path": row.path, "sha256": row.sha256}
            for row in inventory.itertuples(index=False)
        },
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "kernels_modified": False,
        "single_step_H_constructed": False,
    }
    path = out_dir / "verification_contract.json"
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != contract:
            raise ValueError(
                "Existing verification contract differs from current inputs or policy; "
                "use a new versioned output directory"
            )
        if not force:
            raise FileExistsError(
                f"Verification outputs already exist in {out_dir}; use --force only to reproduce them"
            )
    else:
        write_json(path, contract)
    return contract


def structural_ledger(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    columns = sorted(ALLOWED_LEDGER_COLUMNS)
    if suffixes.endswith(".parquet"):
        frame = pd.read_parquet(path, columns=columns)
    else:
        separator = "," if suffixes.endswith((".csv", ".csv.gz")) else "\t"
        frame = pd.read_csv(path, sep=separator, usecols=columns, dtype=str, low_memory=False)
    if set(frame.columns) != ALLOWED_LEDGER_COLUMNS:
        raise ValueError("Structural ledger projection changed unexpectedly")
    return frame.fillna("")


def enrich_marker_paths(
    candidates: pd.DataFrame,
    bridges: pd.DataFrame,
    structured: pd.DataFrame,
) -> pd.DataFrame:
    bridge = bridges.copy()
    bridge["trial_gid"] = bridge["query_id"].map(canonical_gid)
    bridge["normalized_sample_id"] = bridge["external_alias"].map(normalized_identifier)
    candidate_fields = candidates[
        [
            "trial_gid",
            "panel_id",
            "normalized_sample_id",
            "candidate_scope",
            "classification",
            "classification_reasons",
            "direct_marker_assignment_ready",
            "marker_matrix_path",
            "marker_matrix_sha256",
            "marker_matrix_axis",
            "marker_matrix_axis_index",
        ]
    ].drop_duplicates(["trial_gid", "panel_id", "normalized_sample_id"])
    bridge["panel_id"] = "SEEDS_DARTSEQ_DATAVERSE_RECOVERY"
    bridge = bridge.merge(
        candidate_fields,
        on=["trial_gid", "panel_id", "normalized_sample_id"],
        how="left",
        validate="many_to_one",
    )
    bridge["crop_scope"] = "WHEAT_CONFIRMED"
    bridge["crop_scope_evidence"] = "upstream_wheat_gated_structured_evidence"
    bridge["candidate_path_kind"] = "dataset_local_two_hop_marker_path"

    existing = candidates[~candidates["candidate_scope"].eq("new_dataverse_two_hop")].copy()
    existing["query_id"] = existing["trial_gid"]
    existing["query_text"] = existing["selection_history"]
    existing["dataset_persistent_id"] = ""
    existing["marker_filename"] = existing["marker_matrix_path"].map(
        lambda value: Path(clean(value)).name if clean(value) else ""
    )
    existing["crop_scope"] = "WHEAT_CONFIRMED"
    existing["crop_scope_evidence"] = "certified_existing_wheat_panel"
    existing["candidate_path_kind"] = "existing_certified_panel_identity"

    direct = structured[structured["evidence_class"].eq("direct_gid_exact")].copy()
    direct["trial_gid"] = direct["query_id"].map(canonical_gid)
    direct["panel_id"] = ""
    direct["normalized_sample_id"] = ""
    direct["candidate_scope"] = "structured_direct_gid_evidence"
    direct["classification"] = ""
    direct["classification_reasons"] = "marker_sample_confirmation_required"
    direct["direct_marker_assignment_ready"] = False
    direct["marker_matrix_path"] = ""
    direct["marker_matrix_sha256"] = ""
    direct["marker_matrix_axis"] = ""
    direct["marker_matrix_axis_index"] = ""
    direct["marker_filename"] = direct["filename"]
    direct["candidate_path_kind"] = "direct_gid_appearance_requires_sample_confirmation"
    output = pd.concat([bridge, existing, direct], ignore_index=True, sort=False)
    return output.fillna("")


def attach_bridge_provenance(
    candidates: pd.DataFrame, bridges: pd.DataFrame
) -> pd.DataFrame:
    local = candidates.copy()
    local["trial_gid"] = local["trial_gid"].map(canonical_gid)
    local["normalized_sample_id"] = local["normalized_sample_id"].map(clean)
    bridge = bridges.copy()
    bridge["trial_gid"] = bridge["query_id"].map(canonical_gid)
    bridge["normalized_sample_id"] = bridge["external_alias"].map(
        normalized_identifier
    )
    rows: list[dict[str, str]] = []
    for keys, group in bridge.groupby(
        ["trial_gid", "normalized_sample_id"], sort=True
    ):
        rows.append(
            {
                "trial_gid": keys[0],
                "normalized_sample_id": keys[1],
                "dataset_persistent_id": ";".join(
                    sorted(set(group["dataset_persistent_id"].map(clean)) - {""})
                ),
                "marker_source_file": ";".join(
                    sorted(set(group["marker_filename"].map(clean)) - {""})
                ),
                "bridge_confidence": ";".join(
                    sorted(set(group["bridge_confidence"].map(clean)) - {""})
                ),
                "direct_gid_mapping_evidence": any(
                    canonical_gid(value) == keys[0]
                    for value in group["query_text"]
                ),
                "crop_scope": "WHEAT_CONFIRMED",
                "crop_scope_evidence": "upstream_wheat_gated_structured_evidence",
            }
        )
    provenance = pd.DataFrame(rows)
    if not provenance.empty:
        local = local.merge(
            provenance,
            on=["trial_gid", "normalized_sample_id"],
            how="left",
            validate="many_to_one",
        )
    for column, default in {
        "dataset_persistent_id": "",
        "marker_source_file": "",
        "bridge_confidence": "",
        "direct_gid_mapping_evidence": False,
        "crop_scope": "WHEAT_CONFIRMED",
        "crop_scope_evidence": "certified_existing_wheat_panel",
    }.items():
        if column not in local:
            local[column] = default
        local[column] = local[column].fillna("")
        if default:
            local.loc[local[column].eq(""), column] = default
    return local


def recompute_replicate_concordance(
    candidates: pd.DataFrame,
    cached_pairs: pd.DataFrame,
    *,
    minimum_shared: int,
    minimum_concordance: float,
    mode: str,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, int]]]:
    axis_status: dict[tuple[str, str], dict[str, int]] = {}
    if mode == "verify_cached":
        return cached_pairs.copy(), axis_status

    computed: list[pd.DataFrame] = []
    computed_groups: set[tuple[str, str]] = set()
    for (matrix_value, panel_id), matrix_group in candidates.groupby(
        ["marker_matrix_path", "panel_id"], sort=True
    ):
        matrix_text = clean(matrix_value)
        if not matrix_text:
            continue
        matrix_path = Path(matrix_text)
        if not matrix_path.is_file():
            raise FileNotFoundError(f"Candidate marker matrix is absent: {matrix_path}")
        axis, _ = marker_by_sample_axis(matrix_path)
        replicate_groups: dict[str, list[str]] = {}
        sample_columns: dict[str, int] = {}
        for trial_gid, group in matrix_group.groupby("trial_gid", sort=True):
            samples = sorted(set(group["normalized_sample_id"].map(clean)) - {""})
            locations = {sample: axis.get(sample, []) for sample in samples}
            axis_status[(trial_gid, panel_id)] = {
                "expected_samples": len(samples),
                "samples_found_exactly_once": sum(len(value) == 1 for value in locations.values()),
                "ambiguous_or_missing_samples": sum(len(value) != 1 for value in locations.values()),
            }
            if len(samples) < 2 or any(len(locations[sample]) != 1 for sample in samples):
                continue
            replicate_groups[trial_gid] = samples
            computed_groups.add((trial_gid, panel_id))
            for sample in samples:
                sample_columns[sample] = locations[sample][0][0]
        if replicate_groups:
            local = stream_marker_by_sample_concordance(
                matrix_path,
                sample_columns=sample_columns,
                replicate_groups=replicate_groups,
                minimum_shared_markers=minimum_shared,
                minimum_call_concordance=minimum_concordance,
            )
            local["panel_id"] = panel_id
            local["concordance_evidence"] = "recomputed_streaming_selected_sample_columns"
            computed.append(local)
    cached = cached_pairs.copy()
    if not cached.empty:
        keep = [
            (clean(row.trial_gid), clean(row.panel_id)) not in computed_groups
            for row in cached.itertuples(index=False)
        ]
        cached = cached.loc[keep].copy()
        cached["concordance_evidence"] = "upstream_certified_cached_pair_evidence"
    frames = ([cached] if not cached.empty else []) + computed
    output = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return output.fillna(""), axis_status


def marker_classification(
    candidates: pd.DataFrame,
    direct_evidence: pd.DataFrame,
    pairs: pd.DataFrame,
    axis_status: dict[tuple[str, str], dict[str, int]],
    *,
    minimum_shared: int,
    minimum_concordance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    mappings: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    for (trial_gid, panel_id), group in candidates.groupby(
        ["trial_gid", "panel_id"], sort=True
    ):
        samples = sorted(set(group["sample_id"].map(clean)) - {""})
        normalized_samples = sorted(set(group["normalized_sample_id"].map(clean)) - {""})
        original_classes = sorted(set(group["classification"].map(clean)) - {""})
        reasons = sorted(
            {
                reason
                for value in group["classification_reasons"].map(clean)
                for reason in value.split(";")
                if reason
            }
        )
        external_counts = pd.to_numeric(group["external_identity_count"], errors="coerce").fillna(0)
        external_count = int(external_counts.max()) if len(external_counts) else 0
        conflict_statuses = set(group["pedigree_conflict_status"].map(clean)) - {
            "",
            "NO_DETECTED_CONFLICT",
            "NO_EXTERNAL_CONFLICT_RECORD",
            "DIRECT_CANONICAL_GID",
        }
        direct_gid_path = (
            group.get(
                "direct_gid_mapping_evidence",
                pd.Series(False, index=group.index),
            ).map(bool_value).any()
            or group["candidate_scope"].eq("existing_platform_reaudit").all()
        )
        selection_unique = group["selection_history_unique"].map(bool_value).all()
        status = axis_status.get((trial_gid, panel_id))
        axis_exact = (
            status["ambiguous_or_missing_samples"] == 0
            if status is not None
            else pd.to_numeric(group["marker_axis_match_count"], errors="coerce").fillna(0).eq(1).all()
        )
        local_pairs = pairs[
            pairs["trial_gid"].map(clean).eq(trial_gid)
            & pairs["panel_id"].map(clean).eq(panel_id)
        ] if not pairs.empty else pairs
        expected_pairs = len(normalized_samples) * (len(normalized_samples) - 1) // 2
        minimum_overlap = (
            int(pd.to_numeric(local_pairs["shared_nonmissing_markers"], errors="coerce").min())
            if not local_pairs.empty
            else 0
        )
        minimum_pair_concordance = (
            float(pd.to_numeric(local_pairs["call_concordance"], errors="coerce").min())
            if not local_pairs.empty
            else np.nan
        )

        crop_scopes = set(group.get("crop_scope", pd.Series(dtype=str)).map(clean)) - {""}

        if crop_scopes and crop_scopes != {"WHEAT_CONFIRMED"}:
            classification = "non_wheat_excluded"
        elif conflict_statuses or "pedigree_or_cross_conflict" in reasons:
            classification = "conflicting_identity_metadata"
        elif external_count > 1 or "multiple_external_germplasm_identities" in reasons:
            classification = "ambiguous_multiple_external_ids"
        elif "family_level_identity_only" in reasons:
            classification = "family_only_not_assignable"
        elif not samples:
            classification = "metadata_match_without_marker_sample"
        elif not axis_exact:
            classification = "ambiguous_multiple_marker_samples"
        elif len(samples) > 1:
            if len(local_pairs) != expected_pairs:
                classification = "ambiguous_multiple_marker_samples"
            elif minimum_overlap < minimum_shared:
                classification = "insufficient_marker_overlap"
            elif not np.isfinite(minimum_pair_concordance) or minimum_pair_concordance < minimum_concordance:
                classification = "conflicting_marker_calls"
            else:
                classification = "accepted_concordant_technical_replicates"
        elif not direct_gid_path and not selection_unique:
            classification = "family_only_not_assignable"
        elif original_classes and original_classes[0] not in {
            "accepted_unique_identity",
            "accepted_concordant_replicates",
        }:
            classification = "unresolved"
        elif direct_gid_path:
            classification = "accepted_direct_gid_to_marker_sample"
        else:
            classification = "accepted_unique_two_hop_identity"
        if classification not in MARKER_CLASSES:
            raise ValueError(f"Unexpected marker class {classification}")

        accepted = classification in MARKER_ACCEPTED
        replicate_group = stable_id("REPL", trial_gid, panel_id) if len(samples) > 1 else ""
        bridge_confidence = (
            "explicit_accepted_class_after_identity_and_marker_axis_verification"
            if accepted
            else "candidate_unresolved_or_conflicting"
        )
        row = {
            "canonical_gid": trial_gid,
            "marker_panel": panel_id,
            "verification_class": classification,
            "accepted": accepted,
            "sample_count": len(samples),
            "sample_ids": ";".join(samples),
            "external_identity_count": external_count,
            "selection_history_unique": selection_unique,
            "direct_gid_path": direct_gid_path,
            "marker_axis_exact": axis_exact,
            "replicate_group_id": replicate_group,
            "minimum_shared_marker_calls": minimum_overlap,
            "minimum_call_concordance": minimum_pair_concordance,
            "source_datasets": ";".join(sorted(set(group.get("dataset_persistent_id", pd.Series(dtype=str)).map(clean)) - {""})),
            "source_files": ";".join(sorted(set(group["mapping_filename"].map(clean)) - {""})),
            "evidence_reasons": ";".join(reasons),
            "confidence_provenance_class": bridge_confidence,
        }
        rows.append(row)
        if accepted:
            representative = samples[0]
            source = group.sort_values(
                ["mapping_filename", "mapping_source_part", "mapping_source_row"], kind="stable"
            ).iloc[0]
            mappings.append(
                {
                    "canonical_gid": trial_gid,
                    "marker_panel": panel_id,
                    "sample_id": representative,
                    "all_sample_ids": ";".join(samples),
                    "source_file": clean(source.get("mapping_filename")),
                    "marker_source_file": clean(source.get("marker_source_file")),
                    "source_dataset_persistent_id": clean(source.get("dataset_persistent_id")),
                    "mapping_class": classification,
                    "selection_history": clean(source.get("selection_history")),
                    "cross_name": clean(source.get("trial_cross")),
                    "external_gid": clean(source.get("external_gid")),
                    "external_alias": clean(source.get("external_alias")),
                    "source_part": clean(source.get("mapping_source_part")),
                    "source_row": source.get("mapping_source_row", ""),
                    "marker_matrix_path": clean(source.get("marker_matrix_path")),
                    "marker_matrix_axis_locator": clean(source.get("marker_matrix_locator")),
                    "replicate_group_id": replicate_group,
                    "marker_overlap": minimum_overlap,
                    "call_concordance": minimum_pair_concordance,
                    "crop_scope": clean(source.get("crop_scope")),
                    "confidence_provenance_class": bridge_confidence,
                }
            )
            if replicate_group:
                replicate_rows.append(
                    {
                        "replicate_group_id": replicate_group,
                        "canonical_gid": trial_gid,
                        "marker_panel": panel_id,
                        "sample_ids": ";".join(samples),
                        "representative_sample_id": representative,
                        "replicate_count": len(samples),
                        "minimum_shared_marker_calls": minimum_overlap,
                        "minimum_call_concordance": minimum_pair_concordance,
                        "collapse_decision": "accepted_concordant_technical_replicates",
                    }
                )

    observed_gids = set(candidates["trial_gid"].map(clean))
    direct_groups = (
        direct_evidence.groupby("query_id", sort=True)
        if "query_id" in direct_evidence.columns
        else []
    )
    for trial_gid, group in direct_groups:
        gid = canonical_gid(trial_gid)
        if not gid or gid in observed_gids:
            continue
        rows.append(
            {
                "canonical_gid": gid,
                "marker_panel": "",
                "verification_class": "metadata_match_without_marker_sample",
                "accepted": False,
                "sample_count": 0,
                "sample_ids": "",
                "external_identity_count": 1,
                "selection_history_unique": False,
                "direct_gid_path": True,
                "marker_axis_exact": False,
                "replicate_group_id": "",
                "minimum_shared_marker_calls": 0,
                "minimum_call_concordance": np.nan,
                "source_datasets": ";".join(sorted(set(group["dataset_persistent_id"].map(clean)) - {""})),
                "source_files": ";".join(sorted(set(group["filename"].map(clean)) - {""})),
                "evidence_reasons": "direct_gid_appearance_has_no_certified_marker_sample_path",
                "confidence_provenance_class": "metadata_only_not_assignable",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(mappings), pd.DataFrame(replicate_rows)


def load_current_pedigree(path: Path) -> pd.DataFrame:
    frame = read_table(path)
    columns = {
        "sample_id": detect_column(frame, ["sample_id", "panel_sample_id", "genotype_id"]),
        "parent1": detect_column(frame, ["parent1", "female_parent", "mother", "dam"]),
        "parent2": detect_column(frame, ["parent2", "male_parent", "father", "sire"]),
    }
    missing = [name for name, value in columns.items() if value is None]
    if missing:
        raise ValueError(f"Current pedigree is missing columns: {missing}")
    output = pd.DataFrame({name: frame[value].map(clean) for name, value in columns.items()})
    return output[output["sample_id"].ne("")].drop_duplicates()


def _direct_parent_nodes(record: dict[str, object]) -> tuple[list[tuple[str, PedigreeNode]], str]:
    parent1 = clean(record.get("external_parent1"))
    parent2 = clean(record.get("external_parent2"))
    if parent1 or parent2:
        if not parent1 or not parent2:
            return [], "unresolved_parent_identity"
        parent1_gid = canonical_cimmyt_gid(parent1)
        parent2_gid = canonical_cimmyt_gid(parent2)
        if not canonical_gid(parent1_gid):
            parent1_gid = parent1
        if not canonical_gid(parent2_gid):
            parent2_gid = parent2
        return [
            ("parent1", PedigreeNode("leaf", parent1_gid)),
            ("parent2", PedigreeNode("leaf", parent2_gid)),
        ], "explicit_structured_parent_columns"
    lineage = clean(record.get("external_lineage"))
    if not lineage:
        return [], "insufficient_provenance"
    node = parse_purdy_pedigree(lineage)
    if node.kind != "cross" or node.left is None or node.right is None:
        if re.search(r"/|\\|\*|\s+[Xx]\s+", lineage):
            return [], "ambiguous_cross_parse"
        return [], "deeper_ancestor_not_direct_parent"
    return [("parent1", node.left), ("parent2", node.right)], "purdy_direct_parent_parse"


def verify_pedigree_edges(
    records: pd.DataFrame,
    conflicts: pd.DataFrame,
    legacy_edges: pd.DataFrame,
    legacy_nodes: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_edges = {
        (clean(row.sample_id), role, clean(parent))
        for row in current.itertuples(index=False)
        for role, parent in (("parent1", row.parent1), ("parent2", row.parent2))
        if clean(parent)
    }
    existing_pairs = {
        clean(row.sample_id): (clean(row.parent1), clean(row.parent2))
        for row in current.itertuples(index=False)
    }
    conflict_lookup = {
        canonical_gid(row.query_id): (clean(row.conflict_status), clean(row.conflict_reasons))
        for row in conflicts.itertuples(index=False)
        if canonical_gid(row.query_id)
    }
    legacy_status: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in legacy_edges.to_dict("records"):
        legacy_status[(clean(row.get("child_id")), clean(row.get("parent_id")))].add(
            clean(row.get("edge_review_status"))
        )
    registry = RegistryBuilder()
    raw_rows: list[dict[str, object]] = []
    for record in records.to_dict("records"):
        child = canonical_gid(record.get("query_id"))
        if not child:
            continue
        parents, derivation = _direct_parent_nodes(record)
        if not parents:
            raw_rows.append(
                {
                    "child_id": child,
                    "parent_role": "",
                    "parent_id": "",
                    "parent_source_expression": clean(record.get("external_lineage")),
                    "derivation": derivation,
                    "source_dataset_persistent_id": clean(record.get("dataset_persistent_id")),
                    "source_file": clean(record.get("filename")),
                    "source_part": clean(record.get("source_part")),
                    "source_row": record.get("source_row", ""),
                    "authoritative_exact_structured_record": False,
                }
            )
            continue
        exact_child = canonical_cimmyt_gid(record.get("external_gid")) == child
        authoritative = derivation == "explicit_structured_parent_columns" or exact_child
        for role, node in parents:
            parent_id = registry.materialize(node)
            raw_rows.append(
                {
                    "child_id": child,
                    "parent_role": role,
                    "parent_id": parent_id,
                    "parent_source_expression": node.expression,
                    "derivation": derivation,
                    "source_dataset_persistent_id": clean(record.get("dataset_persistent_id")),
                    "source_file": clean(record.get("filename")),
                    "source_part": clean(record.get("source_part")),
                    "source_row": record.get("source_row", ""),
                    "authoritative_exact_structured_record": authoritative,
                }
            )

    rows: list[dict[str, object]] = []
    raw = pd.DataFrame(raw_rows)
    if not raw.empty:
        raw["unresolved_group"] = np.where(
            raw["parent_id"].map(clean).eq(""), raw["derivation"], ""
        )
        group_columns = ["child_id", "parent_role", "parent_id", "unresolved_group"]
        for keys, group in raw.groupby(group_columns, dropna=False, sort=True):
            child, role, parent, unresolved_group = keys
            source_expressions = sorted(
                set(group["parent_source_expression"].map(clean)) - {""}
            )
            derivations = sorted(set(group["derivation"].map(clean)) - {""})
            source_expression = ";".join(source_expressions)
            derivation = ";".join(derivations)
            statuses = set(legacy_status.get((child, parent), set()))
            for expression in source_expressions:
                statuses.update(legacy_status.get((child, expression), set()))
            current_pair = existing_pairs.get(child, ("", ""))
            conflict_status, conflict_reason = conflict_lookup.get(
                child, ("NO_EXTERNAL_CONFLICT_RECORD", "")
            )
            source_keys = {
                (clean(row.source_dataset_persistent_id), clean(row.source_file))
                for row in group.itertuples(index=False)
            }
            independent_sources = len(source_keys)
            exact_sources = int(group["authoritative_exact_structured_record"].map(bool_value).sum())
            already = (
                "ALREADY_PRESENT" in statuses
                or (child, role, source_expression) in existing_edges
                or (child, role, parent) in existing_edges
            )
            if already:
                classification = "already_present"
            elif conflict_status not in {"", "NO_DETECTED_CONFLICT", "NO_EXTERNAL_CONFLICT_RECORD"}:
                classification = "conflicting_external_records"
            elif all(current_pair) and source_expression not in current_pair and parent not in current_pair:
                classification = "conflicts_existing_complete_parent_pair"
            elif unresolved_group in PEDIGREE_CLASSES:
                classification = unresolved_group
            elif exact_sources == 1 and independent_sources == 1:
                classification = "accepted_new_edge_exact_unique"
            elif independent_sources >= 2:
                classification = "accepted_new_edge_corroborated"
            else:
                classification = "insufficient_provenance"
            rows.append(
                {
                    "child_id": child,
                    "parent_role": role,
                    "parent_id": parent,
                    "parent_source_expression": source_expression,
                    "verification_class": classification,
                    "accepted": classification in PEDIGREE_ACCEPTED,
                    "is_new_edge": classification in {
                        "accepted_new_edge_exact_unique",
                        "accepted_new_edge_corroborated",
                    },
                    "derivation": derivation,
                    "source_record_count": len(group),
                    "independent_source_count": independent_sources,
                    "source_datasets": ";".join(sorted({key[0] for key in source_keys} - {""})),
                    "source_files": ";".join(sorted({key[1] for key in source_keys} - {""})),
                    "existing_parent1": current_pair[0],
                    "existing_parent2": current_pair[1],
                    "external_conflict_status": conflict_status,
                    "conflict_reasons": conflict_reason,
                    "parent_registry_identity_scope": (
                        "canonical_gid" if canonical_gid(parent) else "stable_local_purdy_node"
                    ),
                }
            )
    for row in legacy_nodes.to_dict("records"):
        if clean(row.get("node_role")) != "unresolved_ancestor_token":
            continue
        rows.append(
            {
                "child_id": clean(row.get("query_id")),
                "parent_role": "deeper_ancestor",
                "parent_id": "",
                "parent_source_expression": clean(row.get("candidate_node")),
                "verification_class": "deeper_ancestor_not_direct_parent",
                "accepted": False,
                "is_new_edge": False,
                "derivation": clean(row.get("derivation")),
                "source_record_count": 1,
                "independent_source_count": 1,
                "source_datasets": "",
                "source_files": clean(row.get("source_filename")),
                "existing_parent1": "",
                "existing_parent2": "",
                "external_conflict_status": "",
                "conflict_reasons": "unresolved_deeper_lineage_token_is_not_a_direct_parent",
                "parent_registry_identity_scope": "not_materialized",
            }
        )
    output = pd.DataFrame(rows)
    registry_frame = registry.frame()
    return output, registry_frame


def apply_pedigree_cycle_gate(
    verification: pd.DataFrame, current: pd.DataFrame
) -> pd.DataFrame:
    output = verification.copy()
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in current.itertuples(index=False):
        for parent in (clean(row.parent1), clean(row.parent2)):
            if parent:
                adjacency[clean(row.sample_id)].add(parent)

    def reaches(start: str, target: str) -> bool:
        stack = [start]
        visited: set[str] = set()
        while stack:
            value = stack.pop()
            if value == target:
                return True
            if value in visited:
                continue
            visited.add(value)
            stack.extend(sorted(adjacency.get(value, set()), reverse=True))
        return False

    accepted = output[output["is_new_edge"].map(bool_value)].sort_values(
        ["child_id", "parent_role", "parent_id"], kind="stable"
    )
    for index, row in accepted.iterrows():
        child = clean(row["child_id"])
        parent = clean(row["parent_id"])
        if not parent or child == parent or reaches(parent, child):
            output.loc[index, "verification_class"] = "conflicting_external_records"
            output.loc[index, "accepted"] = False
            output.loc[index, "is_new_edge"] = False
            prior = clean(output.loc[index, "conflict_reasons"])
            output.loc[index, "conflict_reasons"] = ";".join(
                filter(None, [prior, "candidate_edge_would_create_pedigree_cycle"])
            )
            continue
        adjacency[child].add(parent)
    return output


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def structural_sources(
    mappings: pd.DataFrame, accepted_edges: pd.DataFrame, panel_orders: dict[str, set[str]]
) -> dict[str, set[str]]:
    sources = {
        panel: set(group["canonical_gid"].map(clean)) - {""}
        for panel, group in mappings.groupby("marker_panel", sort=True)
    }
    for panel in ("HMP", "GBS_SAWYT"):
        if panel in panel_orders:
            sources[f"CERTIFIED_{panel}"] = set(panel_orders[panel])
    if not accepted_edges.empty:
        sources["PEDIGREE_ENRICHMENT"] = set(accepted_edges["child_id"].map(clean)) - {""}
    return sources


def structural_coverage(
    sources: dict[str, set[str]],
    ledger: pd.DataFrame,
    ka_ids: set[str],
    model_ids: set[str],
    panel_orders: dict[str, set[str]],
    current_pedigree: pd.DataFrame,
    accepted_edges: pd.DataFrame,
) -> pd.DataFrame:
    hmp = panel_orders.get("HMP", set())
    gbs = panel_orders.get("GBS_SAWYT", set())
    existing_marker_ids = set().union(*panel_orders.values()) if panel_orders else set()
    graph = UnionFind()
    for row in current_pedigree.itertuples(index=False):
        for parent in (clean(row.parent1), clean(row.parent2)):
            if parent:
                graph.union(clean(row.sample_id), parent)
    for row in accepted_edges.itertuples(index=False):
        if clean(row.parent_id):
            graph.union(clean(row.child_id), clean(row.parent_id))
    ledger_gids = ledger["panel_sample_id"].map(clean)
    rows: list[dict[str, object]] = []
    for source, gids in sorted(sources.items()):
        mask = ledger_gids.isin(gids)
        source_ledger = ledger.loc[mask]
        components = {graph.find(gid) for gid in gids if gid in graph.parent}
        connected_pedigree_only = {
            gid
            for gid in ka_ids - existing_marker_ids
            if gid in graph.parent and graph.find(gid) in components
        }
        rows.append(
            {
                "source": source,
                "accepted_unique_trial_gids": len(gids),
                "intersection_K_A_order": len(gids & ka_ids),
                "intersection_pedigree_model_genotypes": len(gids & model_ids),
                "observation_rows": int(mask.sum()),
                "trait_count": source_ledger["trait_name_canonical"].nunique(),
                "environment_count": source_ledger["env_kernel_id"].nunique(),
                "overlap_HMP_gids": len(gids & hmp),
                "overlap_GBS_gids": len(gids & gbs),
                "new_beyond_HMP_GBS_gids": len(gids - hmp - gbs),
                "connected_pedigree_components_affected": len(components),
                "pedigree_only_entries_potentially_connected": len(connected_pedigree_only),
            }
        )
    return pd.DataFrame(rows)


def fold_support(
    sources: dict[str, set[str]], ledger: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = manifest[["scenario", "outer_fold", "inner_fold"]].drop_duplicates()
    for key in keys.sort_values(["scenario", "outer_fold", "inner_fold"]).itertuples(index=False):
        scenario = clean(key.scenario)
        outer_fold = int(key.outer_fold)
        inner_fold = int(key.inner_fold)
        train, val, test, omitted, leakage = assign_nested_split(
            ledger,
            manifest,
            scenario=scenario,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
        )
        selected_manifest = manifest[
            manifest["scenario"].eq(scenario)
            & pd.to_numeric(manifest["outer_fold"], errors="coerce").eq(outer_fold)
            & pd.to_numeric(manifest["inner_fold"], errors="coerce").eq(inner_fold)
        ]
        final_envs = set(
            selected_manifest.loc[
                selected_manifest["axis"].eq("environment")
                & selected_manifest["partition"].eq("final_holdout"),
                "entity_id",
            ].map(clean)
        ) - {""}
        partition_indices = {
            "inner_training": train,
            "inner_validation": val,
            "outer_test": test,
            "outer_development": np.unique(np.concatenate([train, val])),
            "final_holdout": np.flatnonzero(ledger["env_kernel_id"].map(clean).isin(final_envs)),
        }
        for source, gids in sorted(sources.items()):
            for partition, indices in partition_indices.items():
                local = ledger.iloc[indices]
                local_mask = local["panel_sample_id"].map(clean).isin(gids)
                rows.append(
                    {
                        "source": source,
                        "scenario": scenario,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "partition": partition,
                        "training_support_partition": partition in {"inner_training", "outer_development"},
                        "unique_gids": local.loc[local_mask, "panel_sample_id"].map(clean).nunique(),
                        "observation_rows": int(local_mask.sum()),
                        "environment_count": local.loc[local_mask, "env_kernel_id"].map(clean).nunique(),
                        "leakage_status": leakage["leakage_status"],
                    }
                )
    return pd.DataFrame(rows)


def source_readiness(
    coverage: pd.DataFrame,
    support: pd.DataFrame,
    ka_ids: set[str],
    source_ids: dict[str, set[str]],
    policy: dict[str, object],
) -> pd.DataFrame:
    minimum_fold = int(policy["minimum_training_gids_per_fold"])
    minimum_source = int(policy["minimum_gids_for_H_source"])
    rows: list[dict[str, object]] = []
    for row in coverage.to_dict("records"):
        source = clean(row["source"])
        gids = set(source_ids.get(source, set()))
        training = support[
            support["source"].eq(source) & support["partition"].eq("inner_training")
        ]
        minimum_observed = int(training["unique_gids"].min()) if not training.empty else 0
        all_in_ka = bool(gids) and gids.issubset(ka_ids)
        enough = len(gids) >= minimum_source and minimum_observed >= minimum_fold
        if source == "CERTIFIED_HMP":
            can_h = all_in_ka and enough
            recommendation = (
                "ready_for_H_construction"
                if can_h
                else "structurally_valid_but_insufficient_fold_support"
            )
            reason = (
                "certified_HMP_order_aligned_to_existing_pedigree"
                if can_h
                else "HMP_order_alignment_or_fold_support_below_frozen_minimum"
            )
        elif source.startswith("CERTIFIED_"):
            can_h = all_in_ka and enough
            recommendation = (
                "ready_for_H_construction"
                if can_h
                else "structurally_valid_but_insufficient_fold_support"
            )
            reason = (
                "certified_panel_order_aligned_to_existing_pedigree"
                if can_h
                else "certified_panel_alignment_or_fold_support_below_frozen_minimum"
            )
        elif source == "PEDIGREE_ENRICHMENT":
            recommendation = "ready_for_H_construction"
            reason = "accepted_nonconflicting_edges_can_expand_pedigree_before_H"
            can_h = True
        else:
            if all_in_ka and enough:
                recommendation = "ready_for_H_construction"
                reason = "accepted_identity_marker_calls_K_A_alignment_and_fold_support_pass"
                can_h = True
            elif all_in_ka and gids:
                recommendation = "structurally_valid_but_insufficient_fold_support"
                reason = "accepted_identity_but_source_or_fold_support_below_frozen_minimum"
                can_h = False
            elif gids:
                recommendation = "useful_for_regulatory_eligibility_only"
                reason = "accepted_marker_identity_not_fully_aligned_to_current_K_A_order"
                can_h = False
            else:
                recommendation = "unresolved_not_usable"
                reason = "no_accepted_marker_or_pedigree_entities"
                can_h = False
        rows.append(
            {
                "source": source,
                "recommendation": recommendation,
                "certified_HMP_only_baseline": source == "CERTIFIED_HMP",
                "accepted_gids": int(row["accepted_unique_trial_gids"]),
                "gids_in_K_A_order": int(row["intersection_K_A_order"]),
                "all_accepted_gids_in_K_A_order": all_in_ka,
                "minimum_inner_training_gids": minimum_observed,
                "can_participate_in_expanded_G_or_H": can_h,
                "remain_separate_platform_expert": source not in {"CERTIFIED_HMP", "PEDIGREE_ENRICHMENT"},
                "dosage_matrices_merged": False,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def regulatory_eligibility(verification: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gid, group in verification.groupby("canonical_gid", sort=True):
        accepted = group[group["accepted"].map(bool_value)]
        classes = sorted(set(group["verification_class"].map(clean)) - {""})
        rows.append(
            {
                "canonical_gid": gid,
                "marker_identity_classes": ";".join(classes),
                "accepted_marker_panels": ";".join(sorted(set(accepted["marker_panel"].map(clean)) - {""})),
                "candidate_unresolved": len(accepted) != len(group),
                "marker_identity_accepted": not accepted.empty,
                "regulatory_eligibility_status": (
                    "accepted_marker_identity_graph_projection_pending"
                    if not accepted.empty
                    else "candidate_unresolved"
                ),
                "eligible_for_genotype_specific_sequence_now": False,
                "next_required_action": (
                    "variant_coordinate_normalization_and_graph_projection"
                    if not accepted.empty
                    else "resolve_identity_or_marker_conflict"
                ),
            }
        )
    return pd.DataFrame(rows)


def qc_table(
    policy: dict[str, object],
    resolver: pd.DataFrame,
    bridges: pd.DataFrame,
    structured: pd.DataFrame,
    legacy_edges: pd.DataFrame,
    marker_verification: pd.DataFrame,
    accepted_mapping: pd.DataFrame,
    pedigree_verification: pd.DataFrame,
    protected_unchanged: bool,
    *,
    concordance_mode: str,
) -> pd.DataFrame:
    expected = dict(policy["expected_evidence_counts"])
    legacy_unique = legacy_edges.drop_duplicates(["child_id", "parent_id"])
    observed = {
        "resolver_gids": resolver_identity_summary(resolver)["trial_gid"].nunique(),
        "moderate_marker_candidate_gids": bridges.loc[
            bridges["bridge_confidence"].eq("moderate_candidate_requires_disambiguation"), "query_id"
        ].nunique(),
        "two_hop_marker_bridge_rows": len(bridges),
        "direct_gid_exact_query_ids": structured.loc[
            structured["evidence_class"].eq("direct_gid_exact"), "query_id"
        ].nunique(),
        "candidate_direct_parent_links": len(legacy_unique),
        "pedigree_edges_blocked_by_conflict": int(
            legacy_unique["edge_review_status"].isin(
                ["BLOCKED_BY_EXTERNAL_RECORD_CONFLICT", "CONFLICTS_EXISTING_COMPLETE_PARENT_PAIR"]
            ).sum()
        ),
        "pedigree_edges_already_present": int(
            legacy_unique["edge_already_in_current_pedigree"].map(bool_value).sum()
        ),
    }
    rows = [
        {"check": "run_status", "status": "PASS", "observed": "PASS", "expected": "PASS", "detail": ""},
        {"check": "phenotype_values_read", "status": "PASS", "observed": False, "expected": False, "detail": "structural ledger projection only"},
        {"check": "outer_test_metrics_read", "status": "PASS", "observed": False, "expected": False, "detail": "entity assignments only"},
        {"check": "final_holdout_outcomes_read", "status": "PASS", "observed": False, "expected": False, "detail": "membership IDs only"},
        {"check": "kernels_modified", "status": "PASS" if protected_unchanged else "FAIL", "observed": not protected_unchanged, "expected": False, "detail": "input hashes compared before and after"},
        {"check": "single_step_H_constructed", "status": "PASS", "observed": False, "expected": False, "detail": "verification only"},
        {"check": "giant_matrix_access", "status": "PASS", "observed": concordance_mode, "expected": "recompute or verify_cached", "detail": "full marker string DataFrames are prohibited"},
        {"check": "accepted_marker_mapping_unique", "status": "PASS" if not accepted_mapping.duplicated(["canonical_gid", "marker_panel"]).any() else "FAIL", "observed": int(accepted_mapping.duplicated(["canonical_gid", "marker_panel"]).sum()), "expected": 0, "detail": "after replicate collapse"},
        {"check": "accepted_marker_crop_scope", "status": "PASS" if accepted_mapping.empty or accepted_mapping["crop_scope"].eq("WHEAT_CONFIRMED").all() else "FAIL", "observed": int((~accepted_mapping["crop_scope"].eq("WHEAT_CONFIRMED")).sum()) if not accepted_mapping.empty else 0, "expected": 0, "detail": "non-wheat and ambiguous sources excluded"},
        {"check": "accepted_pedigree_no_complete_pair_conflict", "status": "PASS" if pedigree_verification.loc[pedigree_verification["accepted"].map(bool_value), "verification_class"].ne("conflicts_existing_complete_parent_pair").all() else "FAIL", "observed": 0, "expected": 0, "detail": "accepted edge invariant"},
        {"check": "accepted_pedigree_edge_unique", "status": "PASS" if not pedigree_verification.loc[pedigree_verification["accepted"].map(bool_value)].duplicated(["child_id", "parent_role", "parent_id"]).any() else "FAIL", "observed": int(pedigree_verification.loc[pedigree_verification["accepted"].map(bool_value)].duplicated(["child_id", "parent_role", "parent_id"]).sum()), "expected": 0, "detail": "after stable parent resolution"},
        {"check": "marker_classes_terminal", "status": "PASS" if set(marker_verification["verification_class"]).issubset(MARKER_CLASSES) else "FAIL", "observed": ";".join(sorted(set(marker_verification["verification_class"]))), "expected": ";".join(sorted(MARKER_CLASSES)), "detail": ""},
        {"check": "pedigree_classes_terminal", "status": "PASS" if set(pedigree_verification["verification_class"]).issubset(PEDIGREE_CLASSES) else "FAIL", "observed": ";".join(sorted(set(pedigree_verification["verification_class"]))), "expected": ";".join(sorted(PEDIGREE_CLASSES)), "detail": ""},
    ]
    for metric, value in observed.items():
        target = expected[metric]
        rows.append(
            {
                "check": f"expected_evidence_count:{metric}",
                "status": "MATCH" if value == target else "DIFF_REPORTED",
                "observed": value,
                "expected": target,
                "detail": value - target,
            }
        )
    return pd.DataFrame(rows)


def sha256_manifest(out_dir: Path) -> pd.DataFrame:
    rows = []
    for name in REQUIRED_OUTPUTS:
        path = required_file(out_dir / name, f"output {name}")
        rows.append({"sha256": sha256_file(path), "bytes": path.stat().st_size, "path": name})
    manifest = pd.DataFrame(rows).sort_values("path", kind="stable")
    write_table(manifest, out_dir / "verification_sha256.tsv", ["path"])
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phenotype-blind verification of recovered marker identities and pedigree edges."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--policy", type=Path, default=Path("server_genotype_recovery/recovered_identity_verification_policy_v2.json"))
    parser.add_argument("--identity-policy", type=Path, default=Path("server_genotype_recovery/marker_identity_concordance_policy_v1.json"))
    parser.add_argument("--resolver-query", type=Path, default=Path("genotype_panels/germplasm_resolver/germplasm_cross_query.tsv"))
    parser.add_argument("--wide-dir", type=Path, default=Path("genotype_panels/cimmyt_dataverse_recovery_v1/wide_inventory_v1"))
    parser.add_argument("--marker-adjudication-dir", type=Path, default=Path("genotype_panels/marker_identity_adjudication_v1"))
    parser.add_argument("--pedigree-parent-table", type=Path, default=Path("genotype_panels/pedigree/pedigree_parent_table.tsv"))
    parser.add_argument("--k-a-order", type=Path, default=Path("genotype_panels/pedigree/K_A_sample_order.tsv"))
    parser.add_argument("--model-genotype-order", type=Path, default=Path("model_kernels/stage1_pedigree_env/stage1_pedigree_env_K_G_unique_order.tsv"))
    parser.add_argument("--nested-evaluation-dir", type=Path, default=Path("model_kernels/final_nested_evaluation_v5_fixed"))
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--replicate-concordance-mode", choices=["recompute", "verify_cached"], default="recompute")
    parser.add_argument("--out-dir", type=Path, default=Path("genotype_panels/recovered_identity_verification_v2"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_path = required_file(resolve(root, args.policy), "verification policy")
    identity_policy_path = required_file(resolve(root, args.identity_policy), "marker identity policy")
    policy = load_policy(policy_path)
    identity_policy = load_identity_policy(identity_policy_path)
    wide = resolve(root, args.wide_dir)
    structured_dir = wide / "structured_evidence"
    two_hop_dir = structured_dir / "two_hop_marker_bridges"
    pedigree_dir = structured_dir / "pedigree_enrichment"
    adjudication = resolve(root, args.marker_adjudication_dir)
    nested = resolve(root, args.nested_evaluation_dir)
    contract_path = required_file(nested / "nested_evaluation_contract.json", "nested evaluation contract")
    nested_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    ledger_path = resolve(root, args.ledger) if args.ledger else Path(str(nested_contract["ledger_path"]))
    if not ledger_path.is_file() and not args.ledger:
        ledger_path = resolve(root, Path(str(nested_contract["ledger_path"])).name)
    paths = {
        "verification_policy": policy_path,
        "marker_identity_policy": identity_policy_path,
        "resolver_query": resolve(root, args.resolver_query),
        "structured_evidence": structured_dir / "dataverse_structured_evidence.tsv.gz",
        "structured_crop_scope": structured_dir / "dataverse_structured_source_crop_scope.tsv",
        "two_hop_bridges": two_hop_dir / "dataverse_two_hop_marker_bridges.tsv",
        "two_hop_provenance": two_hop_dir / "dataverse_two_hop_marker_bridge_provenance.json",
        "pedigree_external_records": pedigree_dir / "dataverse_pedigree_external_records.tsv",
        "pedigree_external_conflicts": pedigree_dir / "dataverse_pedigree_conflicts.tsv",
        "pedigree_legacy_edges": pedigree_dir / "dataverse_pedigree_candidate_edges.tsv",
        "pedigree_legacy_nodes": pedigree_dir / "dataverse_pedigree_candidate_nodes.tsv",
        "pedigree_provenance": pedigree_dir / "dataverse_pedigree_enrichment_provenance.json",
        "marker_candidates": adjudication / "marker_identity_candidate_paths.tsv.gz",
        "marker_cached_concordance": adjudication / "marker_identity_pairwise_concordance.tsv.gz",
        "marker_adjudication_provenance": adjudication / "marker_identity_adjudication_provenance.json",
        "pedigree_parent_table": resolve(root, args.pedigree_parent_table),
        "K_A_order": resolve(root, args.k_a_order),
        "model_genotype_order": resolve(root, args.model_genotype_order),
        "nested_entity_manifest": nested / "nested_evaluation_entities.tsv",
        "nested_evaluation_contract": contract_path,
        "structural_ledger": ledger_path,
        "verification_implementation": Path(__file__).resolve(),
        "marker_streaming_implementation": Path(
            marker_by_sample_axis.__code__.co_filename
        ).resolve(),
        "purdy_parser_implementation": Path(
            parse_purdy_pedigree.__code__.co_filename
        ).resolve(),
        "nested_split_implementation": Path(
            assign_nested_split.__code__.co_filename
        ).resolve(),
    }
    order_paths = add_discovered_order_paths(
        root, certified_order_paths(root, identity_policy), out_dir
    )
    for panel, path in sorted(order_paths.items()):
        paths[f"certified_panel_order:{panel}"] = path
    candidate_inventory = read_table(paths["marker_candidates"])
    matrix_paths, declared_hashes = declared_marker_matrix_inputs(
        root, candidate_inventory
    )
    paths.update(matrix_paths)
    inventory = inventory_inputs(paths, declared_hashes)
    protected_before = {row.input_role: row.sha256 for row in inventory.itertuples(index=False)}
    contract = freeze_contract(
        out_dir,
        policy,
        policy_path,
        inventory,
        concordance_mode=args.replicate_concordance_mode,
        force=args.force,
    )
    write_table(inventory, out_dir / "verification_input_inventory.tsv", ["input_role"])

    verify_manifest_contract(paths["nested_entity_manifest"], contract_path)

    resolver = read_table(paths["resolver_query"])
    structured = read_table(paths["structured_evidence"])
    bridges = read_table(paths["two_hop_bridges"])
    candidates = candidate_inventory
    cached_pairs = read_table(paths["marker_cached_concordance"])
    records = read_table(paths["pedigree_external_records"])
    conflicts = read_table(paths["pedigree_external_conflicts"])
    legacy_edges = read_table(paths["pedigree_legacy_edges"])
    legacy_nodes = read_table(paths["pedigree_legacy_nodes"])
    current_pedigree = load_current_pedigree(paths["pedigree_parent_table"])
    ka_ids = load_order_ids(paths["K_A_order"])
    model_ids = load_order_ids(paths["model_genotype_order"])
    panel_orders = {panel: load_order_ids(path) for panel, path in order_paths.items()}
    ledger = structural_ledger(paths["structural_ledger"])
    manifest = read_table(paths["nested_entity_manifest"])

    candidates = attach_bridge_provenance(candidates, bridges)
    marker_paths = enrich_marker_paths(candidates, bridges, structured)
    write_deterministic_gzip_table(
        marker_paths,
        out_dir / "marker_candidate_paths.tsv.gz",
        ["trial_gid", "panel_id", "dataset_persistent_id", "mapping_filename", "mapping_source_row"],
    )
    pairs, axis_status = recompute_replicate_concordance(
        candidates,
        cached_pairs,
        minimum_shared=int(policy["minimum_shared_nonmissing_marker_calls"]),
        minimum_concordance=float(policy["technical_replicate_concordance_threshold"]),
        mode=args.replicate_concordance_mode,
    )
    verification, mappings, replicate_groups = marker_classification(
        candidates,
        structured[structured["evidence_class"].eq("direct_gid_exact")],
        pairs,
        axis_status,
        minimum_shared=int(policy["minimum_shared_nonmissing_marker_calls"]),
        minimum_concordance=float(policy["technical_replicate_concordance_threshold"]),
    )
    write_table(verification, out_dir / "marker_identity_verification.tsv", ["canonical_gid", "marker_panel"])
    write_table(mappings, out_dir / "accepted_gid_marker_mapping.tsv", ["canonical_gid", "marker_panel"])
    write_table(replicate_groups, out_dir / "accepted_marker_replicate_groups.tsv", ["canonical_gid", "marker_panel"])
    write_table(pairs, out_dir / "marker_replicate_concordance.tsv", ["trial_gid", "panel_id", "sample_id_left", "sample_id_right"])
    unresolved = verification[~verification["accepted"].map(bool_value) & ~verification["verification_class"].isin({"conflicting_marker_calls", "conflicting_identity_metadata", "non_wheat_excluded"})]
    marker_conflicts = verification[verification["verification_class"].isin({"conflicting_marker_calls", "conflicting_identity_metadata", "non_wheat_excluded"})]
    write_table(unresolved, out_dir / "unresolved_marker_candidates.tsv", ["canonical_gid", "marker_panel"])
    write_table(marker_conflicts, out_dir / "marker_conflicts.tsv", ["canonical_gid", "marker_panel"])

    edge_verification, parent_registry = verify_pedigree_edges(
        records, conflicts, legacy_edges, legacy_nodes, current_pedigree
    )
    edge_verification = apply_pedigree_cycle_gate(edge_verification, current_pedigree)
    accepted_new_edges = edge_verification[edge_verification["is_new_edge"].map(bool_value)].copy()
    pedigree_conflicts = edge_verification[
        edge_verification["verification_class"].isin(
            {"conflicts_existing_complete_parent_pair", "conflicting_external_records", "ambiguous_cross_parse"}
        )
    ].copy()
    write_table(edge_verification, out_dir / "pedigree_candidate_edge_verification.tsv", ["child_id", "parent_role", "parent_id", "source_files"])
    write_table(accepted_new_edges, out_dir / "accepted_new_pedigree_edges.tsv", ["child_id", "parent_role", "parent_id"])
    write_table(pedigree_conflicts, out_dir / "pedigree_conflicts.tsv", ["child_id", "parent_role", "parent_id"])
    write_table(parent_registry, out_dir / "verification_parent_registry.tsv", ["node_type", "stable_parent_id"])

    accepted_for_graph = accepted_new_edges
    sources = structural_sources(mappings, accepted_for_graph, panel_orders)
    coverage = structural_coverage(
        sources, ledger, ka_ids, model_ids, panel_orders, current_pedigree, accepted_for_graph
    )
    support = fold_support(sources, ledger, manifest)
    regulatory = regulatory_eligibility(verification)
    readiness = source_readiness(coverage, support, ka_ids, sources, policy)
    write_table(coverage, out_dir / "recovered_structural_coverage.tsv", ["source"])
    write_table(support, out_dir / "recovered_fold_support.tsv", ["source", "scenario", "outer_fold", "inner_fold", "partition"])
    write_table(regulatory, out_dir / "recovered_regulatory_eligibility.tsv", ["canonical_gid"])
    write_table(readiness, out_dir / "single_step_H_input_readiness.tsv", ["source"])

    protected_after = {
        row.input_role: sha256_file(Path(row.path))
        for row in inventory.itertuples(index=False)
        if row.hash_basis == "file_bytes"
    }
    protected_unchanged = all(
        protected_before[role] == observed for role, observed in protected_after.items()
    )
    qc = qc_table(
        policy,
        resolver,
        bridges,
        structured,
        legacy_edges,
        verification,
        mappings,
        edge_verification,
        protected_unchanged,
        concordance_mode=args.replicate_concordance_mode,
    )
    write_table(qc, out_dir / "verification_qc.tsv", ["check"])
    failed = qc[qc["status"].eq("FAIL")]
    if not failed.empty:
        raise ValueError(f"Recovered identity verification failed: {failed.to_dict('records')}")

    provisional_outputs = {
        name: sha256_file(out_dir / name)
        for name in REQUIRED_OUTPUTS
        if name != "verification_provenance.json"
    }
    provenance = {
        "status": "PASS",
        "protocol_version": policy["protocol_version"],
        "selection_data": policy["selection_data"],
        "contract_sha256": sha256_file(out_dir / "verification_contract.json"),
        "input_inventory_sha256": sha256_file(out_dir / "verification_input_inventory.tsv"),
        "output_sha256_before_provenance": provisional_outputs,
        "parent_registry_nodes_materialized_for_verification_only": len(parent_registry),
        "replicate_concordance_mode": args.replicate_concordance_mode,
        "large_marker_matrix_access": "streamed_selected_sample_columns_only" if args.replicate_concordance_mode == "recompute" else "verified_upstream_cached_streaming_evidence",
        "structural_ledger_columns_read": sorted(ALLOWED_LEDGER_COLUMNS),
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "model_performance_read": False,
        "kernels_modified": False,
        "single_step_H_constructed": False,
        "existing_input_hashes_unchanged": protected_unchanged,
    }
    write_json(out_dir / "verification_provenance.json", provenance)
    sha256_manifest(out_dir)
    print(qc.to_string(index=False), flush=True)
    print("\n=== MARKER IDENTITY SUMMARY ===", flush=True)
    print(verification.groupby("verification_class")["canonical_gid"].nunique().to_string(), flush=True)
    print("\n=== PEDIGREE EDGE SUMMARY ===", flush=True)
    print(edge_verification.groupby("verification_class")["child_id"].count().to_string(), flush=True)
    print("\n=== H INPUT READINESS ===", flush=True)
    print(readiness.to_string(index=False), flush=True)
    print(f"\nCertified verification artifacts: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
