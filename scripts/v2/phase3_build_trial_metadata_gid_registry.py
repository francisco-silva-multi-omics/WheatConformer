"""Recover GIDs from exact trial-metadata identifiers with full provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


REGISTRY_VERSION = "stage1_v2_registries_2026_07_30_v8"
MISSING = {"", "NA", "N/A", "NAN", "NONE", "NULL", "UNKNOWN", "DESCONOCIDO", "-", ".", "0"}


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).strip())
    return "" if text.upper() in MISSING else text


def clean_id(value: object) -> str:
    return re.sub(r"\.0$", "", clean(value))


def clean_gid(value: object) -> str:
    text = re.sub(r"^GID", "", clean_id(value), flags=re.I)
    return text if re.fullmatch(r"[0-9]+", text) else ""


def norm(value: object) -> str:
    return clean(value).upper()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {re.sub(r"[^A-Z0-9]", "", str(column).upper()): column for column in columns}
    for candidate in candidates:
        key = re.sub(r"[^A-Z0-9]", "", candidate.upper())
        if key in lookup:
            return lookup[key]
    return None


def trial_token(path: Path) -> str:
    stem = re.sub(r"(?i)_?Genotypes?_Data$", "", path.stem)
    token = re.sub(r"[^A-Z0-9]", "", stem.upper())
    match = re.fullmatch(r"(\d+)(ST|ND|RD|TH)?(.+)", token)
    return match.group(1) + match.group(3) if match else token


def read_trial_metadata(path: Path) -> list[tuple[str, pd.DataFrame]]:
    if path.suffix.lower() == ".xls":
        for encoding in ["utf-8-sig", "latin-1", "cp1252"]:
            try:
                return [("TEXT_TSV", pd.read_csv(path, sep="\t", dtype=str, encoding=encoding, keep_default_na=False))]
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError("unknown", b"", 0, 1, "No supported encoding")
    workbook = pd.ExcelFile(path)
    return [(str(sheet), pd.read_excel(path, sheet_name=sheet, dtype=str, keep_default_na=False)) for sheet in workbook.sheet_names]


def evidence_row(rows: list[dict[str, str]], gid: object, identifier: object, kind: str, source: str, column: str) -> None:
    gid_value = clean_gid(gid)
    identifier_value = norm(identifier)
    if gid_value and identifier_value:
        rows.append({
            "identifier_type": kind, "normalized_identifier": identifier_value,
            "gid": gid_value, "source_file": source, "source_column": column,
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-root", type=Path, required=True)
    parser.add_argument("--base-registries", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--child-lineage", type=Path, required=True)
    parser.add_argument("--exact-name-candidates", type=Path, required=True)
    parser.add_argument("--genotypic-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.result_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    base_dir = args.base_registries.resolve()

    evidence_rows: list[dict[str, str]] = []
    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
    conflict = manifest["fieldbook_glis_gid_conflict"].astype(str).str.upper().isin({"TRUE", "1", "YES"})
    for row in manifest[~conflict].itertuples(index=False):
        gid = getattr(row, "resolved_gid", "")
        evidence_row(evidence_rows, gid, getattr(row, "cross_name", ""), "CROSS_NAME", str(args.manifest), "cross_name")
        evidence_row(evidence_rows, gid, getattr(row, "selection_history", ""), "SELECTION_HISTORY", str(args.manifest), "selection_history")

    lineage = pd.read_csv(args.child_lineage, sep="\t", dtype=str, keep_default_na=False)
    for row in lineage.itertuples(index=False):
        gid = getattr(row, "sample_id", "")
        if clean(getattr(row, "source_lineage_count", "")) == "1":
            evidence_row(evidence_rows, gid, getattr(row, "source_lineages", ""), "CROSS_NAME", str(args.child_lineage), "source_lineages")
        evidence_row(evidence_rows, gid, getattr(row, "selected_lineage", ""), "CROSS_NAME", str(args.child_lineage), "selected_lineage")

    name_candidates = pd.read_csv(args.exact_name_candidates, sep="\t", dtype=str, keep_default_na=False)
    for row in name_candidates.itertuples(index=False):
        evidence_row(evidence_rows, row.candidate_gid, row.normalized_genotype_name, "CROSS_NAME", str(args.exact_name_candidates), "normalized_genotype_name")

    genotypic_root = args.genotypic_root.resolve()
    dartag1 = genotypic_root / "Genotypic_data_(DArTAG_panel_2)_for_the_IBWSN_and_SAWSN/germplasm_list.xlsx"
    dartag2 = genotypic_root / "Genotypic_data_(DArTAG_panel_2)_for_the_IBWSN_and_SAWSN/germplasm_list_2.xlsx"
    hibap = genotypic_root / "IWYP64_-_HiBAP_35k_Wheat_Breeders_Array_Genotyping/HIBAPI_germplasm_information.txt"
    for path, header in [(dartag1, 1), (dartag2, 0)]:
        frame = pd.read_excel(path, header=header, dtype=str, keep_default_na=False)
        gid_col = find_column(list(frame.columns), ["GID"])
        cross_col = find_column(list(frame.columns), ["cross", "pedigree"])
        history_col = find_column(list(frame.columns), ["sel_hist", "selection history"])
        for row in frame.to_dict("records"):
            evidence_row(evidence_rows, row.get(gid_col, ""), row.get(cross_col, ""), "CROSS_NAME", str(path), str(cross_col))
            evidence_row(evidence_rows, row.get(gid_col, ""), row.get(history_col, ""), "SELECTION_HISTORY", str(path), str(history_col))
    hibap_frame = pd.read_csv(hibap, sep="\t", skiprows=5, dtype=str, keep_default_na=False)
    gid_col = find_column(list(hibap_frame.columns), ["GID"])
    cross_col = find_column(list(hibap_frame.columns), ["Cross Name"])
    history_col = find_column(list(hibap_frame.columns), ["Selection History"])
    for row in hibap_frame.to_dict("records"):
        evidence_row(evidence_rows, row.get(gid_col, ""), row.get(cross_col, ""), "CROSS_NAME", str(hibap), str(cross_col))
        evidence_row(evidence_rows, row.get(gid_col, ""), row.get(history_col, ""), "SELECTION_HISTORY", str(hibap), str(history_col))

    evidence = pd.DataFrame(evidence_rows).drop_duplicates()
    evidence.to_csv(out / "canonical_identifier_gid_evidence.tsv", sep="\t", index=False)
    identifier_registry = (
        evidence.groupby(["identifier_type", "normalized_identifier"], sort=True)
        .agg(
            gids=("gid", lambda x: ";".join(sorted(set(x)))),
            gid_count=("gid", "nunique"),
            evidence_rows=("gid", "size"),
            source_files=("source_file", lambda x: ";".join(sorted(set(x)))),
        )
        .reset_index()
    )
    identifier_registry["identifier_decision"] = np.where(
        identifier_registry["gid_count"].eq(1), "ACCEPT_EXACT_GLOBALLY_UNIQUE_IDENTIFIER", "AMBIGUOUS_IDENTIFIER_MULTIPLE_GIDS"
    )
    identifier_registry["accepted_gid"] = identifier_registry["gids"].where(identifier_registry["gid_count"].eq(1), "")
    identifier_registry.to_csv(out / "canonical_identifier_gid_registry_v2.tsv", sep="\t", index=False)
    cross_map = identifier_registry[
        identifier_registry["identifier_type"].eq("CROSS_NAME") & identifier_registry["accepted_gid"].ne("")
    ].set_index("normalized_identifier")["accepted_gid"].to_dict()
    history_map = identifier_registry[
        identifier_registry["identifier_type"].eq("SELECTION_HISTORY") & identifier_registry["accepted_gid"].ne("")
    ].set_index("normalized_identifier")["accepted_gid"].to_dict()

    raw_trials = pd.read_csv(base_dir / "raw_trial_registry.tsv", sep="\t", dtype=str, keep_default_na=False)
    alias_targets = raw_trials[raw_trials["trial_code"].ne("")][["trial_code", "trial_key", "cycle"]].drop_duplicates()
    files = sorted(
        path for path in args.trial_root.resolve().rglob("*")
        if path.is_file() and re.search(r"(?i)Genotypes?_Data\.(xls|xlsx)$", path.name)
    )
    file_audit = []
    candidate_parts = []
    for path in files:
        token = trial_token(path)
        targets = alias_targets[alias_targets["trial_code"].eq(token)]
        try:
            members = read_trial_metadata(path)
            parser_status = "PARSED"
        except Exception as exc:
            members = []
            parser_status = f"READ_ERROR:{type(exc).__name__}:{str(exc)[:200]}"
        parsed_rows = 0
        for member, frame in members:
            cid_col = find_column(list(frame.columns), ["CID", "Cid"])
            sid_col = find_column(list(frame.columns), ["SID", "Sid"])
            gid_col = find_column(list(frame.columns), ["GID"])
            cross_col = find_column(list(frame.columns), ["Cross Name", "Cross", "Pedigree"])
            history_col = find_column(list(frame.columns), ["Selection History", "Sel Hist", "sel_hist"])
            if cid_col is None or sid_col is None:
                continue
            parsed_rows += len(frame)
            work = pd.DataFrame({
                "CID_norm": frame[cid_col].map(clean_id), "SID_norm": frame[sid_col].map(clean_id),
                "direct_gid": frame[gid_col].map(clean_gid) if gid_col else "",
                "cross_name": frame[cross_col].map(clean) if cross_col else "",
                "selection_history": frame[history_col].map(clean) if history_col else "",
                "source_physical_row": np.arange(len(frame), dtype=np.int64) + 2,
            })
            work["cross_candidate_gid"] = work["cross_name"].map(lambda value: cross_map.get(norm(value), ""))
            work["history_candidate_gid"] = work["selection_history"].map(lambda value: history_map.get(norm(value), ""))
            work["candidate_gids"] = work[["direct_gid", "cross_candidate_gid", "history_candidate_gid"]].apply(
                lambda row: ";".join(sorted({str(value) for value in row if str(value)})), axis=1
            )
            work["candidate_gid_count"] = work["candidate_gids"].map(lambda value: 0 if not value else len(value.split(";")))
            work["metadata_decision"] = np.select(
                [work["candidate_gid_count"].eq(1), work["candidate_gid_count"].gt(1)],
                ["ACCEPT_CONCORDANT_EXACT_TRIAL_METADATA", "AMBIGUOUS_TRIAL_METADATA_IDENTIFIER_CONFLICT"],
                default="UNRESOLVED_NO_EXACT_IDENTIFIER_GID",
            )
            work["metadata_accepted_gid"] = work["candidate_gids"].where(work["candidate_gid_count"].eq(1), "")
            work["source_file"] = str(path.relative_to(args.trial_root.resolve())).replace("\\", "/")
            work["source_file_sha256"] = file_sha256(path)
            work["source_member"] = member
            work["trial_file_token"] = token
            if targets.empty:
                work["trial_key"] = ""; work["cycle_norm"] = ""
                candidate_parts.append(work)
            else:
                work["_join"] = 1
                expanded = work.merge(
                    targets.assign(_join=1).rename(columns={"cycle": "cycle_norm"}),
                    on="_join", how="left", validate="m:m",
                ).drop(columns="_join")
                candidate_parts.append(expanded)
        file_audit.append({
            "source_file": str(path.relative_to(args.trial_root.resolve())).replace("\\", "/"),
            "source_file_sha256": file_sha256(path), "trial_file_token": token,
            "raw_alias_target_count": len(targets), "parser_status": parser_status,
            "parsed_rows": parsed_rows,
        })
    pd.DataFrame(file_audit).to_csv(out / "trial_genotype_metadata_file_ledger.tsv", sep="\t", index=False)
    candidates = pd.concat(candidate_parts, ignore_index=True).fillna("")
    candidates.to_csv(out / "trial_genotype_metadata_row_ledger.tsv", sep="\t", index=False)
    keys = ["trial_key", "cycle_norm", "CID_norm", "SID_norm"]
    grouped = (
        candidates[candidates["trial_key"].ne("")]
        .groupby(keys, dropna=False, sort=True)
        .agg(
            metadata_rows=("metadata_decision", "size"),
            metadata_candidate_gids=("metadata_accepted_gid", lambda x: ";".join(sorted(set(x) - {""}))),
            metadata_candidate_gid_count=("metadata_accepted_gid", lambda x: len(set(x) - {""})),
            metadata_decisions=("metadata_decision", lambda x: ";".join(sorted(set(x)))),
            metadata_source_files=("source_file", lambda x: ";".join(sorted(set(x)))),
        )
        .reset_index()
    )
    grouped["metadata_registry_decision"] = np.select(
        [grouped["metadata_candidate_gid_count"].eq(1), grouped["metadata_candidate_gid_count"].gt(1)],
        ["ACCEPT_UNIQUE_EXACT_TRIAL_METADATA_GID", "AMBIGUOUS_MULTIPLE_TRIAL_METADATA_GIDS"],
        default="UNRESOLVED_NO_EXACT_TRIAL_METADATA_GID",
    )
    grouped["metadata_registry_accepted_gid"] = grouped["metadata_candidate_gids"].where(grouped["metadata_candidate_gid_count"].eq(1), "")
    grouped.to_csv(out / "trial_metadata_identity_registry_v2.tsv", sep="\t", index=False)

    base = pd.read_csv(base_dir / "genotype_alias_registry_v2.tsv", sep="\t", dtype=str, keep_default_na=False)
    if base.duplicated(keys).any() or grouped.duplicated(keys).any():
        raise RuntimeError("Genotype registry keys must be unique")
    merged = base.merge(grouped, on=keys, how="outer", validate="1:1").fillna("")
    existing = merged["accepted_gid"].ne("")
    metadata_gid = merged["metadata_registry_accepted_gid"].ne("")
    conflict = existing & metadata_gid & merged["accepted_gid"].ne(merged["metadata_registry_accepted_gid"])
    recovery = ~existing & metadata_gid
    merged.loc[recovery, "accepted_gid"] = merged.loc[recovery, "metadata_registry_accepted_gid"]
    merged.loc[recovery, "registry_decision"] = "ACCEPT_UNIQUE_EXACT_TRIAL_METADATA_GID"
    # Exact raw/manifest/DOI evidence has higher declared priority than a
    # cross-name or selection-history match. Retain it, but expose the weaker
    # metadata conflict in the decision and conflict ledger.
    merged.loc[conflict, "registry_decision"] = "ACCEPT_HIGHER_PRIORITY_EXISTING_GID_METADATA_CONFLICT_FLAG"
    merged["panel_sample_id"] = merged["accepted_gid"].map(lambda value: f"GID{value}" if value else "")
    merged["registry_version"] = REGISTRY_VERSION
    merged.to_csv(out / "genotype_alias_registry_v2.tsv", sep="\t", index=False)
    for name in [
        "environment_alias_registry_v2.tsv", "trait_alias_registry_v2.tsv", "trait_unit_rules_v2.tsv",
        "doi_file_to_trial_registry.tsv", "doi_trialwide_identity_audit.tsv", "manifest_trialwide_identity_audit.tsv",
        "global_cid_sid_identity_registry_v2.tsv", "exact_name_identity_registry_v2.tsv", "raw_trial_registry.tsv",
    ]:
        source = base_dir / name
        if source.exists():
            (out / name).write_bytes(source.read_bytes())
    summary = {
        "status": "PASS_TRIAL_METADATA_REGISTRY_EXTENSION",
        "registry_version": REGISTRY_VERSION,
        "trial_metadata_files": len(files),
        "trial_metadata_files_parsed": int(pd.Series([row["parser_status"] for row in file_audit]).eq("PARSED").sum()),
        "trial_metadata_rows": len(candidates),
        "canonical_identifier_keys": len(identifier_registry),
        "canonical_identifier_keys_unique_gid": int(identifier_registry["accepted_gid"].ne("").sum()),
        "trial_metadata_registry_keys": len(grouped),
        "newly_recovered_keys": int(recovery.sum()),
        "lower_priority_metadata_conflict_flags": int(conflict.sum()),
        "accepted_registry_keys": int(merged["accepted_gid"].ne("").sum()),
        "genotype_registry_sha256": file_sha256(out / "genotype_alias_registry_v2.tsv"),
    }
    (out / "registry_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
