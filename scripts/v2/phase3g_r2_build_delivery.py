"""Build the bounded corrective Phase-3G R2 semantic deliverables.

This program is diagnostic only.  It never writes to raw genotype data,
Phase-3 Stage-1 outputs, the frozen certified-v1 bundle, or Phase-3G v1.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from scripts.v2.phase3g_r2_semantics import (
    PARSER_VERSION,
    audit_80k_encoding,
    certify_80k_representations,
    hibap_replicate_concordance,
    parse_hibap_sources,
    read_80k_csv_axes,
)


VERSION = "phase3g_r2_corrective_delivery_v1"
SELECTED_TRAITS = {
    "1000_GRAIN_WEIGHT",
    "ABOVE_GROUND_BIOMASS",
    "DAYS_TO_HEADING",
    "DAYS_TO_MATURITY",
    "GRAIN_YIELD",
    "PLANT_HEIGHT",
    "TEST_WEIGHT",
}


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False)


def semantics_validation(instances: pd.DataFrame, summary: dict[str, object]) -> pd.DataFrame:
    hibap3 = instances.loc[instances["raw_matrix_header"].eq("Hibap3")].iloc[0]
    checks = [
        ("physical_matrix_columns", 148, summary["matrix_columns"], "matrix preamble aligned columns"),
        ("matrix_header_to_sidecar_sample35k_agreement", 0, summary["matrix_header_to_sidecar_sample35k_agreement"], "typed namespaces compared, never joined"),
        ("matrix_entry_to_sidecar_ent_agreement", 148, summary["entry_to_ent_agreement"], "primary exact join"),
        ("unique_matrix_entry_numbers", 147, summary["unique_entry_numbers"], "duplicate entry 109 retained"),
        ("unique_linked_gids", 145, summary["unique_linked_gids"], "after Entry-to-ENT and GID concordance"),
        ("matrix_sidecar_gid_concordant_columns", 148, summary["matrix_sidecar_gid_concordant"], "explicit typed GID comparison"),
        ("matrix_sidecar_gid_conflicts", 0, summary["gid_conflicts"], "explicit typed GID comparison"),
        ("duplicate_entry_109_physical_columns", 2, summary["duplicate_entry_109_columns"], "Hibap109 and Hibap109-2"),
        ("hibap3_links_to_ent3", "3", clean(hibap3["sidecar_ent"]), "counterexample regression"),
        ("hibap3_links_to_gid775", "GID775", clean(hibap3["accepted_canonical_gid"]), "counterexample regression"),
        ("hibap3_sidecar_sample35k_preserved", "Hibap91", clean(hibap3["sidecar_sample_35k"]), "separate typed alias"),
    ]
    return pd.DataFrame(
        {
            "metric": metric,
            "expected": expected,
            "observed": observed,
            "status": "PASS" if str(expected) == str(observed) else "FAIL",
            "evidence": evidence,
        }
        for metric, expected, observed, evidence in checks
    )


def hibap_evidence(instances: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("HIBAP35K_MATRIX_SAMPLE_HEADER", "raw_matrix_header", "", False),
        ("HIBAP35K_MATRIX_ENTRY_NUMBER", "matrix_entry_number", "", True),
        ("HIBAP35K_MATRIX_GID", "matrix_gid_raw", "matrix_canonical_gid", True),
        ("HIBAP35K_SIDECAR_ENT", "sidecar_ent", "", True),
        ("HIBAP35K_SIDECAR_SAMPLE_35K", "sidecar_sample_35k", "", False),
        ("HIBAP35K_SIDECAR_GID", "sidecar_gid_raw", "sidecar_canonical_gid", True),
    ]
    rows: list[dict[str, object]] = []
    for record in instances.to_dict("records"):
        accepted = clean(record["accepted_canonical_gid"])
        for namespace, raw_column, gid_column, authoritative in specs:
            typed_gid = clean(record[gid_column]) if gid_column else ""
            rows.append(
                {
                    "panel_id": "hibap35k",
                    "sample_instance_key": record["sample_instance_key"],
                    "source_file": record["source_file"] if namespace.startswith("HIBAP35K_MATRIX") else record["sidecar_source_file"],
                    "source_location": (
                        f"physical_column={record['physical_column_index']}"
                        if namespace.startswith("HIBAP35K_MATRIX")
                        else f"physical_row={record['sidecar_physical_row']}"
                    ),
                    "identifier_namespace": namespace,
                    "raw_identifier": clean(record[raw_column]),
                    "typed_canonical_gid": typed_gid,
                    "authoritative_for_join": authoritative,
                    "accepted_for_identity": bool(accepted and typed_gid == accepted),
                    "join_rule": record["join_rule"],
                    "linkage_status": record["linkage_status"],
                    "parser_version": PARSER_VERSION,
                }
            )
    return pd.DataFrame(rows)


def dependency_graph() -> pd.DataFrame:
    edges = [
        ("HiBAP matrix preamble", "HiBAP sample-instance ledger", "parse distinct header/Entry/GID axes"),
        ("HiBAP sidecar", "HiBAP sample-instance ledger", "exact Entry number to ENT join"),
        ("HiBAP sample-instance ledger", "HiBAP corrected crosswalk", "accept only concordant matrix/sidecar typed GIDs"),
        ("HiBAP sample-instance ledger", "HiBAP evidence/collision/conflict ledgers", "retain six identifier namespaces"),
        ("HiBAP matrix marker vectors", "HiBAP replicate concordance", "compare retained repeated GID instances"),
        ("HiBAP corrected crosswalk", "marker-present population and HiBAP orders", "retain all 148 physical columns"),
        ("80K CSV structured preambles", "80K sample-instance ledger", "physical column and occurrence identity"),
        ("80K CSV and Flapjack", "80K representation concordance", "sample/marker axes and reversible order"),
        ("non-80K panel crosswalks", "80K cross-panel candidates", "text equality only; never identity acceptance"),
        ("all corrected panel crosswalks", "accepted all-panel crosswalk", "deterministic accepted-row filter"),
        ("accepted all-panel crosswalk", "accepted all-panel GID union", "deterministic distinct typed GID group"),
        ("accepted all-panel GID union", "Stage-1 overlap tables", "typed GID membership join"),
        ("corrected sample/evidence ledgers", "unresolved reports and panel readiness", "full regeneration"),
        ("corrective release", "persistent handoff documents", "supersession and Phase-5 input contract"),
    ]
    return pd.DataFrame(
        {
            "edge_id": index,
            "upstream": upstream,
            "downstream": downstream,
            "transformation": transformation,
            "affected_by_hibap_or_80k_correction": True,
            "regeneration_required": True,
            "regeneration_status": "REGENERATED_IN_PHASE3G_R2",
        }
        for index, (upstream, downstream, transformation) in enumerate(edges, start=1)
    )


def build_80k_axis_tables(axes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_primary = {
        "hexaploid": (56_342, 56_342),
        "tetraploid": (18_946, 18_944),
        "wheat_recall": (15_666, 15_666),
        "wild_relative": (3_903, 3_903),
    }
    rows: list[dict[str, object]] = []
    for population, population_axes in axes.groupby("population", sort=True):
        pav = population_axes[population_axes["representation"].eq("CSV_PAV")].sort_values("physical_column_index")
        snp = population_axes[population_axes["representation"].eq("CSV_SNP")].sort_values("physical_column_index")
        expected_physical, expected_unique = expected_primary[population]
        same_axis = pd.NA
        replicate_or_qc_mismatches = pd.NA
        if not snp.empty:
            same_axis = (
                list(pav["raw_sample_label"]) == list(snp["raw_sample_label"])
                and list(pav["well"]) == list(snp["well"])
                and list(pav["plate_or_barcode"]) == list(snp["plate_or_barcode"])
                and list(pav["sample_group"]) == list(snp["sample_group"])
            )
            replicate_or_qc_mismatches = int(
                pav["replicate_or_index"].reset_index(drop=True).ne(
                    snp["replicate_or_index"].reset_index(drop=True)
                ).sum()
            )
        observed_physical = len(pav)
        observed_unique = pav["raw_sample_label"].nunique()
        duplicates = sorted(pav.loc[pav["raw_sample_label"].duplicated(False), "raw_sample_label"].unique())
        passed = observed_physical == expected_physical and observed_unique == expected_unique and (snp.empty or bool(same_axis))
        rows.append(
            {
                "population": population,
                "primary_representation": "CSV_PAV",
                "expected_physical_sample_columns": expected_physical,
                "observed_physical_sample_columns": observed_physical,
                "expected_unique_sample_labels": expected_unique,
                "observed_unique_sample_labels": observed_unique,
                "duplicate_labels": ";".join(duplicates),
                "snp_csv_counterpart_present": not snp.empty,
                "pav_snp_identity_bearing_preamble_and_sample_order_exact": same_axis,
                "pav_snp_representation_specific_replicate_or_qc_value_mismatches": replicate_or_qc_mismatches,
                "replicate_or_qc_interpretation": "PRESERVED_RAW;REPRESENTATION_SPECIFIC_NOT_USED_FOR_IDENTITY",
                "sample_id_physical_row": ";".join(map(str, sorted(pav["sample_id_physical_row"].unique()))),
                "certification_status": "PASS_SAMPLE_AXIS" if passed else "BLOCKED_SAMPLE_AXIS",
            }
        )
    primary = axes[axes["representation"].eq("CSV_PAV")]
    duplicate_rows: list[dict[str, object]] = []
    for (population, label), group in primary.groupby(["population", "raw_sample_label"], sort=True):
        if len(group) < 2:
            continue
        duplicate_rows.append(
            {
                "population": population,
                "raw_sample_label": label,
                "physical_occurrence_count": len(group),
                "physical_column_indices": ";".join(map(str, group["physical_column_index"])),
                "occurrence_indices": ";".join(map(str, group["occurrence_index"])),
                "sample_instance_keys": ";".join(group["sample_instance_key"]),
                "disposition": "RETAIN_ALL_PHYSICAL_INSTANCES_PENDING_PHASE5",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(duplicate_rows)


def manifest_search_report(genotype_root: Path) -> pd.DataFrame:
    directory = genotype_root / "80k"
    rows = [
        {
            "candidate_file": "80k/MANIFEST.TXT",
            "relationship_to_80k": "SAME_DIRECTORY_DATASET_FILE_INVENTORY",
            "fields_found": "file name;media type;compressed byte size",
            "typed_gid_field_found": False,
            "authoritative_same_dataset_sample_gid_crosswalk": False,
            "decision": "NOT_AUTHORITATIVE_FILE_INVENTORY_ONLY",
        },
        {
            "candidate_file": "80k/wheat_request_fastq_file_access_agreement.txt",
            "relationship_to_80k": "SAME_DIRECTORY_LICENSE_NOTICE",
            "fields_found": "license/access text only",
            "typed_gid_field_found": False,
            "authoritative_same_dataset_sample_gid_crosswalk": False,
            "decision": "NOT_A_MANIFEST_LICENSE_ONLY",
        },
        {
            "candidate_file": "Seeds_of_Discovery_-_MasAgro_Biodiversidad_Wheat_DArTseq-Derived_SNP_Data_Beta_Recall_Results_From_2011-2014/SampleIDvsGID_45610samples.txt",
            "relationship_to_80k": "DIFFERENT_DATASET_CROSS_PANEL_TEXT_OVERLAP",
            "fields_found": "SampleID;GID",
            "typed_gid_field_found": True,
            "authoritative_same_dataset_sample_gid_crosswalk": False,
            "decision": "CANDIDATE_CROSS_PANEL_LABEL_MATCH_ONLY",
        },
        {
            "candidate_file": "DArTseq-derived_SNPs_for_wheat_Mexican_landrace_accessions/Mexican_landrace_samples_for_Germinate.txt",
            "relationship_to_80k": "DIFFERENT_DATASET_CROSS_PANEL_TEXT_OVERLAP",
            "fields_found": "SampleID;GID",
            "typed_gid_field_found": True,
            "authoritative_same_dataset_sample_gid_crosswalk": False,
            "decision": "CANDIDATE_CROSS_PANEL_LABEL_MATCH_ONLY",
        },
        {
            "candidate_file": "80k/*_data*.csv structured preambles",
            "relationship_to_80k": "EMBEDDED_SAME_DATASET_SAMPLE_METADATA",
            "fields_found": "well;plate/barcode;sample group;replicate/index;sample ID;marker schema",
            "typed_gid_field_found": False,
            "authoritative_same_dataset_sample_gid_crosswalk": False,
            "decision": "AUTHORITATIVE_SAMPLE_AXIS_NOT_GID_IDENTITY",
        },
    ]
    for row in rows[:2]:
        if not (genotype_root / row["candidate_file"]).exists():
            raise RuntimeError(f"Expected 80K candidate file is absent: {row['candidate_file']}")
    return pd.DataFrame(rows)


def accepted_union_and_overlap(
    phase3g_root: Path, stage1_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Preserve native nullable numeric and boolean types in the delivered Parquet.
    samples = pd.read_parquet(phase3g_root / "sample_identifier_ledger.parquet")
    accepted = samples[samples["accepted_canonical_gid"].fillna("").ne("")].copy()
    accepted.to_parquet(phase3g_root / "accepted_all_panel_crosswalk.parquet", index=False)

    grouped = (
        accepted.groupby("accepted_canonical_gid", sort=True)
        .agg(
            accepted_sample_instances=("sample_instance_key", "nunique"),
            panels=("panel_id", lambda values: ";".join(sorted(set(map(str, values))))),
            panel_count=("panel_id", "nunique"),
            marker_present_sample_instances=("marker_vector_present", "sum"),
        )
        .reset_index()
        .rename(columns={"accepted_canonical_gid": "canonical_gid"})
    )
    con = duckdb.connect(database=":memory:")
    stage1_sql_path = str(stage1_path).replace("'", "''")
    con.execute(f"CREATE VIEW stage1 AS SELECT * FROM read_parquet('{stage1_sql_path}')")
    selected_frame = pd.DataFrame({"accepted_canonical_trait": sorted(SELECTED_TRAITS)})
    con.register("selected_traits", selected_frame)
    stage_counts = con.execute(
        "SELECT 'GID'||cast(resolved_gid AS VARCHAR) canonical_gid, count(*) stage1_rows, "
        "sum(CASE WHEN accepted_canonical_trait IN (SELECT accepted_canonical_trait FROM selected_traits) THEN 1 ELSE 0 END) selected_trait_rows "
        "FROM stage1 GROUP BY 1"
    ).fetch_df()
    con.close()
    union = grouped.merge(stage_counts, on="canonical_gid", how="left", validate="one_to_one")
    union["stage1_rows"] = union["stage1_rows"].fillna(0).astype("int64")
    union["selected_trait_rows"] = union["selected_trait_rows"].fillna(0).astype("int64")
    union["in_stage1_v2"] = union["stage1_rows"].gt(0)
    union["in_stage1_v2_selected_traits"] = union["selected_trait_rows"].gt(0)
    write_tsv(union, phase3g_root / "accepted_all_panel_gid_union.tsv")
    union.to_parquet(phase3g_root / "accepted_all_panel_gid_union.parquet", index=False)
    overlap = pd.DataFrame(
        [
            {
                "population": "all_traits",
                "accepted_all_panel_unique_gids": len(union),
                "stage1_v2_linked_gids": int(union["in_stage1_v2"].sum()),
                "stage1_v2_linked_rows": int(union["stage1_rows"].sum()),
            },
            {
                "population": "seven_selected_traits",
                "accepted_all_panel_unique_gids": len(union),
                "stage1_v2_linked_gids": int(union["in_stage1_v2_selected_traits"].sum()),
                "stage1_v2_linked_rows": int(union["selected_trait_rows"].sum()),
            },
        ]
    )
    write_tsv(overlap, phase3g_root / "stage1_v2_genotype_overlap.tsv")
    return accepted, union, overlap


def _read_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path).fillna("")
        if path.suffix in {".tsv", ".txt"}:
            return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        if path.suffix == ".csv":
            return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return None
    return None


def _row_hashes(frame: pd.DataFrame, ignored: Iterable[str] = ()) -> pd.Series:
    columns = sorted(set(frame.columns) - set(ignored))
    normalized = frame[columns].astype(str)
    return pd.util.hash_pandas_object(normalized, index=False)


def old_vs_new_diff(old_root: Path, new_root: Path) -> pd.DataFrame:
    names = [
        "sample_identifier_ledger.parquet",
        "sample_gid_crosswalk.parquet",
        "linkage_evidence_ledger.parquet",
        "unmatched_ambiguous_conflicting_samples.tsv",
        "marker_presence_and_qc.parquet",
        "namespace_collision_ledger.tsv",
        "cross_panel_duplicate_report.tsv",
        "sample_order_manifest.tsv",
        "genotype_file_inventory.tsv",
        "panel_inventory.tsv",
        "canonical_gid_panel_coverage.tsv",
        "stage1_all_trait_linkage_summary.tsv",
        "stage1_selected_trait_linkage_summary.tsv",
        "stage1_linkage_by_trait_and_panel.tsv",
        "stage1_linkage_by_trial.tsv",
        "stage1_linkage_by_trial_cycle.tsv",
        "cross_panel_gid_overlap.tsv",
        "unresolved_phenotype_identity_candidates.tsv",
        "phase3g_audit_summary.json",
    ]
    rows: list[dict[str, object]] = []
    for name in names:
        old_path, new_path = old_root / name, new_root / name
        old_frame, new_frame = _read_table(old_path), _read_table(new_path)
        old_count = len(old_frame) if old_frame is not None else (1 if old_path.exists() else 0)
        new_count = len(new_frame) if new_frame is not None else (1 if new_path.exists() else 0)
        added = removed = changed = pd.NA
        if old_frame is not None and new_frame is not None:
            old_hashes = Counter(map(int, _row_hashes(old_frame)))
            new_hashes = Counter(map(int, _row_hashes(new_frame)))
            added = sum((new_hashes - old_hashes).values())
            removed = sum((old_hashes - new_hashes).values())
            changed = min(added, removed)
        reason = "HiBAP namespace repair and/or preservation of physical 80K duplicate columns"
        if name == "genotype_file_inventory.tsv":
            reason = "parser-version/profile provenance regenerated; raw source bytes unchanged"
        if name == "phase3g_audit_summary.json":
            reason = "regenerated counts, version, and timestamp"
        rows.append(
            {
                "artifact": name,
                "old_hash": sha256(old_path) if old_path.exists() else "",
                "new_hash": sha256(new_path) if new_path.exists() else "",
                "old_row_count": old_count,
                "new_row_count": new_count,
                "added_rows": added,
                "removed_rows": removed,
                "changed_rows": changed,
                "reason_for_change": reason,
                "expected_from_hibap_80k_correction": True,
                "downstream_recertification_required": True,
                "recertification_status": "REGENERATED_PENDING_FINAL_VALIDATION",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--genotype-root", type=Path, default=Path("GENOTYPIC_DATA"))
    parser.add_argument("--phase3g-v1", type=Path, required=True)
    parser.add_argument("--phase3g-v2", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument(
        "--reuse-representation-certification",
        action="store_true",
        help="Reuse the completed streaming CSV/Flapjack table already in this v2 root.",
    )
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    genotype_root = args.genotype_root.resolve()
    old_root = args.phase3g_v1.resolve()
    out_root = args.phase3g_v2.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    instances, sidecar, hibap_summary = parse_hibap_sources(genotype_root)
    instances.to_parquet(out_root / "hibap_sample_instance_ledger.parquet", index=False)
    write_tsv(instances, out_root / "hibap_sample_instance_ledger.tsv")
    semantics = semantics_validation(instances, hibap_summary)
    write_tsv(semantics, out_root / "hibap_identifier_semantics_validation.tsv")
    (out_root / "hibap_identifier_semantics_validation.json").write_text(
        json.dumps({"summary": hibap_summary, "all_checks_pass": bool(semantics["status"].eq("PASS").all())}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    crosswalk = instances[
        [
            "panel_id", "sample_instance_key", "source_file", "physical_column_index",
            "raw_matrix_header", "header_occurrence_index", "matrix_entry_number",
            "sidecar_ent", "sidecar_sample_35k", "accepted_canonical_gid", "join_rule",
            "evidence_type", "linkage_status", "conflict_status", "replicate_status", "parser_version",
        ]
    ].copy()
    crosswalk.to_parquet(out_root / "hibap_corrected_sample_to_gid_crosswalk.parquet", index=False)
    write_tsv(crosswalk, out_root / "hibap_corrected_sample_to_gid_crosswalk.tsv")
    evidence = hibap_evidence(instances)
    evidence.to_parquet(out_root / "hibap_linkage_evidence_ledger.parquet", index=False)
    namespace = instances[
        [
            "sample_instance_key", "physical_column_index", "raw_matrix_header",
            "matrix_entry_number", "sidecar_ent", "sidecar_sample_35k",
            "matrix_canonical_gid", "sidecar_canonical_gid",
        ]
    ].copy()
    namespace["matrix_header_equals_sidecar_sample35k"] = namespace["raw_matrix_header"].eq(namespace["sidecar_sample_35k"])
    namespace["semantic_equivalence_permitted"] = False
    namespace["disposition"] = "SEPARATE_NAMESPACES;PRIMARY_JOIN_ENTRY_TO_ENT"
    write_tsv(namespace, out_root / "hibap_namespace_collision_report.tsv")
    conflicts = instances[
        [
            "sample_instance_key", "raw_matrix_header", "matrix_entry_number", "sidecar_ent",
            "matrix_canonical_gid", "sidecar_canonical_gid", "linkage_status", "conflict_status",
        ]
    ].copy()
    conflicts["gid_concordant"] = conflicts["matrix_canonical_gid"].eq(conflicts["sidecar_canonical_gid"])
    write_tsv(conflicts, out_root / "hibap_corrected_conflict_report.tsv")
    replicates, replicate_summary = hibap_replicate_concordance(genotype_root, instances)
    write_tsv(replicates, out_root / "hibap_replicate_concordance_report.tsv")
    (out_root / "hibap_replicate_concordance_summary.json").write_text(
        json.dumps(replicate_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_tsv(dependency_graph(), out_root / "affected_artifact_dependency_graph.tsv")

    axes = read_80k_csv_axes(genotype_root)
    axes.to_parquet(out_root / "dartseq80k_sample_instance_ledger.parquet", index=False)
    axis_validation, duplicates = build_80k_axis_tables(axes)
    write_tsv(axis_validation, out_root / "dartseq80k_sample_axis_validation.tsv")
    write_tsv(duplicates, out_root / "dartseq80k_duplicate_column_report.tsv")
    write_tsv(manifest_search_report(genotype_root), out_root / "dartseq80k_manifest_search_report.tsv")

    core_evidence = pd.read_parquet(out_root / "linkage_evidence_ledger.parquet").fillna("")
    core_samples = pd.read_parquet(out_root / "sample_identifier_ledger.parquet").fillna("")
    candidates = core_evidence[
        core_evidence["panel_id"].str.startswith("dartseq80k_")
        & core_evidence["candidate_only"].astype(bool)
        & core_evidence["evidence_type"].eq("cross_panel_exact_sample_id_candidate_only")
    ].copy()
    candidates = candidates.merge(
        core_samples[["panel_sample_key", "sample_instance_key", "accepted_canonical_gid", "mapping_status"]],
        on=["panel_sample_key", "sample_instance_key"], how="left", validate="many_to_one",
    )
    if candidates["accepted_canonical_gid"].ne("").any():
        raise RuntimeError("Cross-panel candidate-only evidence was promoted to an accepted GID")
    if len(candidates) != 43_570:
        raise RuntimeError(
            f"Expected 43,570 physical 80K candidate matches (43,568 labels plus two "
            f"retained duplicate columns), observed {len(candidates)}"
        )
    candidates["candidate_disposition"] = "CANDIDATE_CROSS_PANEL_LABEL_MATCH"
    candidates.to_parquet(out_root / "dartseq80k_cross_panel_candidate_ledger.parquet", index=False)
    write_tsv(
        candidates.groupby(["panel_id", "candidate_disposition"], sort=True).size().reset_index(name="candidate_sample_instances"),
        out_root / "dartseq80k_cross_panel_candidate_summary.tsv",
    )

    accepted, union, overlap = accepted_union_and_overlap(out_root, args.stage1.resolve())
    diff = old_vs_new_diff(old_root, out_root)
    write_tsv(diff, out_root / "old_vs_new_artifact_diff.tsv")

    representation_path = out_root / "dartseq80k_csv_flapjack_concordance.tsv"
    if args.reuse_representation_certification:
        if not representation_path.exists():
            raise RuntimeError("Representation certification reuse requested, but its table is absent")
        representation = pd.read_csv(representation_path, sep="\t", keep_default_na=False)
    else:
        representation = certify_80k_representations(genotype_root)
    representation.loc[
        representation["representation_pair"].str.contains("SNP", na=False),
        "missing_and_genotype_encoding",
    ] = "CSV paired 0/1 allele-presence calls transform to FJ nucleotide or slash-separated heterozygote calls; '-' missing"
    write_tsv(representation, representation_path)
    encoding = audit_80k_encoding(genotype_root)
    write_tsv(encoding, out_root / "dartseq80k_encoding_validation.tsv")

    result = {
        "version": VERSION,
        "parser_version": PARSER_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hibap": hibap_summary,
        "hibap_semantics_checks_pass": bool(semantics["status"].eq("PASS").all()),
        "hibap_replicate_pairs": len(replicates),
        "dartseq80k_csv_axis_instances_all_representations": len(axes),
        "dartseq80k_primary_physical_sample_columns": int(axes["representation"].eq("CSV_PAV").sum()),
        "dartseq80k_cross_panel_candidates": len(candidates),
        "dartseq80k_accepted_gids": int(core_samples.loc[core_samples["panel_id"].str.startswith("dartseq80k_"), "accepted_canonical_gid"].ne("").sum()),
        "dartseq80k_representation_checks": len(representation),
        "dartseq80k_representation_checks_pass": int(representation["certification_status"].str.startswith("PASS").sum()),
        "dartseq80k_encoding_checks": len(encoding),
        "dartseq80k_encoding_checks_pass": int(encoding["status"].str.startswith("PASS").sum()),
        "global_accepted_sample_instances": len(accepted),
        "global_accepted_unique_gids": len(union),
        "stage1_overlap": overlap.to_dict("records"),
        "same_dataset_80k_sample_gid_manifest_found": False,
        "status": "PASS_R2_ARTIFACT_BUILD" if semantics["status"].eq("PASS").all() and axis_validation["certification_status"].str.startswith("PASS").all() and representation["certification_status"].str.startswith("PASS").all() and encoding["status"].str.startswith("PASS").all() else "BLOCKED_R2_ARTIFACT_BUILD",
    }
    (out_root / "phase3g_r2_build_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
