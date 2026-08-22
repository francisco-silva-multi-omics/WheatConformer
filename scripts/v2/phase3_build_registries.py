"""Build versioned genotype, environment, trait, and unit registries for Stage-1 v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REGISTRY_VERSION = "stage1_v2_registries_2026_07_30_v5"


def clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def norm(series: pd.Series) -> pd.Series:
    return clean(series).str.upper().str.replace(r"\s+", " ", regex=True)


def clean_id(series: pd.Series) -> pd.Series:
    return clean(series).str.replace(r"\.0$", "", regex=True)


def cycle_year(series: pd.Series) -> pd.Series:
    raw = clean(series)
    return raw.str.extract(r"(\d{4})", expand=False).fillna(raw)


def normalized_trial_token(value: str) -> str:
    token = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    match = re.fullmatch(r"(\d+)(ST|ND|RD|TH)?(.+)", token)
    return match.group(1) + match.group(3) if match else token


def trial_code_from_name(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value).strip().upper())
    number_match = re.match(r"(\d+)", text)
    if not number_match:
        return ""
    number = number_match.group(1)
    compact = normalized_trial_token(text)
    if re.fullmatch(r"\d+[A-Z0-9]{2,10}", compact):
        return compact
    families = [
        ("ELITE SPRING WHEAT", "ESWYT"),
        ("SEMI-ARID WHEAT YT", "SAWYT"),
        ("SEMI-ARID WHEAT YIELD", "SAWYT"),
        ("SEMI-ARID WHEAT SN", "SAWSN"),
        ("SEMI-ARID WHEAT SCREEN", "SAWSN"),
        ("INTL. BREAD WHEAT", "IBWSN"),
        ("INTL BREAD WHEAT", "IBWSN"),
        ("INTERNATIONAL BREAD WHEAT", "IBWSN"),
        ("HIGH RAINFALL WHEAT", "HRWYT"),
        ("HIGH TEMPERATURE WHEAT", "HTWYT"),
        ("FUSARIUM HEAD BLIGHT", "FHBSN"),
        ("STRESS ADAPTIVE TRAIT", "SATYN"),
        ("WHEAT YIELD CONSORTIUM", "WYCYT"),
    ]
    for phrase, code in families:
        if phrase in text:
            return number + code
    return ""


def joined_unique(values: pd.Series) -> str:
    return ";".join(sorted(set(clean(values)) - {""}))


def count_unique_nonempty(values: pd.Series) -> int:
    return len(set(clean(values)) - {""})


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--doi-ledger", type=Path, required=True)
    parser.add_argument("--glis-resolver", type=Path, required=True)
    parser.add_argument("--environment-aliases", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    raw_path = args.raw_ledger.resolve()

    connection = duckdb.connect(database=":memory:")
    trial_rows = connection.execute(
        """
        SELECT DISTINCT trial_key, cycle, trial_name
        FROM read_parquet(?)
        ORDER BY trial_key, cycle, trial_name
        """,
        [str(raw_path)],
    ).fetch_df()
    trial_rows["trial_code"] = trial_rows["trial_name"].map(trial_code_from_name)
    trial_rows.to_csv(result_dir / "raw_trial_registry.tsv", sep="\t", index=False)

    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str, low_memory=False).fillna("")
    manifest["trial_key"] = norm(manifest["trial_name"])
    manifest["cycle_norm"] = cycle_year(manifest["cycle"])
    manifest["occ_norm"] = clean(manifest["occ"])
    manifest["CID_norm"] = clean_id(manifest["CID"])
    manifest["SID_norm"] = clean_id(manifest["SID"])
    manifest["resolved_gid_norm"] = clean_id(manifest["resolved_gid"])
    manifest["trial_abbr_norm"] = clean(manifest["trial_abbr"]).map(normalized_trial_token)
    manifest["doi_file_norm"] = clean(manifest["doi_file"]).str.replace("\\", "/", regex=False).str.lower()
    manifest["fieldbook_glis_conflict"] = norm(manifest["fieldbook_glis_gid_conflict"]).isin({"TRUE", "1", "YES"})

    # Resolve any DOI that was unavailable when the legacy manifest was built,
    # then expand official trial abbreviations onto every observed raw alias.
    glis_for_manifest = pd.read_csv(args.glis_resolver, sep="\t", dtype=str, low_memory=False).fillna("")
    glis_for_manifest["DOI"] = clean(glis_for_manifest["DOI"])
    glis_for_manifest["phase3_glis_gid"] = clean_id(glis_for_manifest["glis_gid"])
    glis_for_manifest = glis_for_manifest[["DOI", "phase3_glis_gid"]]
    if glis_for_manifest["DOI"].duplicated().any():
        raise RuntimeError("Phase-3 GLIS resolver DOI key is not unique")
    manifest = manifest.merge(glis_for_manifest, on="DOI", how="left", validate="m:1")
    manifest["phase3_glis_gid"] = clean_id(manifest["phase3_glis_gid"])
    effective_conflict = (
        manifest["resolved_gid_norm"].ne("")
        & manifest["phase3_glis_gid"].ne("")
        & manifest["resolved_gid_norm"].ne(manifest["phase3_glis_gid"])
    )
    manifest["fieldbook_glis_conflict"] = manifest["fieldbook_glis_conflict"] | effective_conflict
    manifest["resolved_gid_norm"] = np.where(
        manifest["resolved_gid_norm"].ne(""), manifest["resolved_gid_norm"], manifest["phase3_glis_gid"]
    )
    raw_alias_targets = trial_rows[trial_rows["trial_code"].ne("")][
        ["trial_code", "trial_key", "cycle"]
    ].drop_duplicates().rename(
        columns={"trial_key": "raw_alias_trial_key", "cycle": "raw_alias_cycle"}
    )
    manifest = manifest.merge(
        raw_alias_targets,
        left_on="trial_abbr_norm", right_on="trial_code",
        how="left", validate="m:m",
    )
    alias_target_present = clean(manifest["raw_alias_trial_key"]).ne("")
    manifest["manifest_trial_key_original"] = manifest["trial_key"]
    manifest["manifest_cycle_norm_original"] = manifest["cycle_norm"]
    manifest["trial_key"] = np.where(alias_target_present, clean(manifest["raw_alias_trial_key"]), manifest["trial_key"])
    manifest["cycle_norm"] = np.where(alias_target_present, clean(manifest["raw_alias_cycle"]), manifest["cycle_norm"])
    manifest["trial_alias_resolution"] = np.where(
        alias_target_present, "ACCEPT_OFFICIAL_TRIAL_ABBR_TO_RAW_ALIAS", "NO_OBSERVED_RAW_ALIAS"
    )

    manifest_trialwide = (
        manifest.groupby(["trial_key", "cycle_norm", "CID_norm", "SID_norm"], dropna=False, sort=True)
        .agg(
            manifest_rows=("resolved_gid_norm", "size"),
            occurrence_count=("occ_norm", "nunique"),
            manifest_resolved_gids=("resolved_gid_norm", joined_unique),
            manifest_resolved_gid_count=("resolved_gid_norm", count_unique_nonempty),
            fieldbook_gids=("fieldbook_gid", joined_unique),
            glis_gids=("glis_gid", joined_unique),
            valid_DOIs=("DOI", lambda x: ";".join(sorted({str(v).strip() for v in x if re.fullmatch(r"10\.\d{4,9}/\S+", str(v).strip(), re.I)}))),
            gid_sources=("gid_source", joined_unique),
            fieldbook_glis_conflict_rows=("fieldbook_glis_conflict", "sum"),
        )
        .reset_index()
    )
    manifest_trialwide["manifest_decision"] = np.select(
        [
            manifest_trialwide["fieldbook_glis_conflict_rows"].gt(0),
            manifest_trialwide["manifest_resolved_gid_count"].eq(0),
            manifest_trialwide["manifest_resolved_gid_count"].eq(1),
        ],
        ["AMBIGUOUS_FIELDBOOK_GLIS_CONFLICT", "UNRESOLVED_NO_MANIFEST_GID", "ACCEPT_UNIQUE_TRIALWIDE_MANIFEST_GID"],
        default="AMBIGUOUS_MULTIPLE_MANIFEST_GIDS",
    )
    manifest_trialwide["manifest_accepted_gid"] = np.where(
        manifest_trialwide["manifest_decision"].eq("ACCEPT_UNIQUE_TRIALWIDE_MANIFEST_GID"),
        manifest_trialwide["manifest_resolved_gids"],
        "",
    )
    manifest_trialwide.to_csv(result_dir / "manifest_trialwide_identity_audit.tsv", sep="\t", index=False)

    doi = pq.read_table(args.doi_ledger.resolve()).to_pandas().fillna("")
    doi["doi_source_file_norm"] = clean(doi["doi_source_file"]).str.replace("\\", "/", regex=False).str.lower()
    doi["trial_token_norm"] = clean(doi["trial_file_token"]).map(normalized_trial_token)
    doi["CID_norm"] = clean_id(doi["CID"])
    doi["SID_norm"] = clean_id(doi["SID"])
    doi["DOI"] = clean(doi["DOI"])
    doi["DOI_valid"] = doi["DOI"].str.fullmatch(r"10\.\d{4,9}/\S+", case=False, na=False)

    exact_file_map = (
        manifest[manifest["doi_file_norm"].ne("")]
        .groupby("doi_file_norm", sort=True)
        .agg(
            exact_trial_keys=("trial_key", joined_unique),
            exact_trial_key_count=("trial_key", count_unique_nonempty),
            exact_cycles=("cycle_norm", joined_unique),
            exact_cycle_count=("cycle_norm", count_unique_nonempty),
        )
        .reset_index()
    )
    token_map = (
        trial_rows[trial_rows["trial_code"].ne("")]
        .groupby("trial_code", sort=True)
        .agg(
            token_trial_keys=("trial_key", joined_unique),
            token_trial_key_count=("trial_key", count_unique_nonempty),
            token_cycles=("cycle", joined_unique),
            token_cycle_count=("cycle", count_unique_nonempty),
        )
        .reset_index()
        .rename(columns={"trial_code": "trial_token_norm"})
    )
    doi_file_map = doi[["doi_source_file", "doi_source_file_norm", "trial_file_token", "trial_token_norm"]].drop_duplicates()
    doi_file_map = doi_file_map.merge(exact_file_map, left_on="doi_source_file_norm", right_on="doi_file_norm", how="left", validate="1:1")
    doi_file_map = doi_file_map.merge(token_map, on="trial_token_norm", how="left", validate="m:1")
    for column in ["exact_trial_keys", "exact_cycles", "token_trial_keys", "token_cycles"]:
        doi_file_map[column] = clean(doi_file_map[column])
    for column in ["exact_trial_key_count", "exact_cycle_count", "token_trial_key_count", "token_cycle_count"]:
        doi_file_map[column] = pd.to_numeric(doi_file_map[column], errors="coerce").fillna(0).astype("int64")
    raw_trial_cycle_pairs = {
        (str(trial_key), str(cycle))
        for trial_key, cycle in trial_rows[["trial_key", "cycle"]].itertuples(index=False, name=None)
    }
    doi_file_map["exact_raw_trial_present"] = [
        int(
            trial_count == 1
            and cycle_count == 1
            and (trial_keys, cycles) in raw_trial_cycle_pairs
        )
        for trial_keys, cycles, trial_count, cycle_count in doi_file_map[
            ["exact_trial_keys", "exact_cycles", "exact_trial_key_count", "exact_cycle_count"]
        ].itertuples(index=False, name=None)
    ]
    doi_file_map["mapping_status"] = np.select(
        [
            doi_file_map["exact_trial_key_count"].eq(1)
            & doi_file_map["exact_cycle_count"].eq(1)
            & doi_file_map["exact_raw_trial_present"].eq(1),
            doi_file_map["token_trial_key_count"].eq(1) & doi_file_map["token_cycle_count"].eq(1),
            doi_file_map["token_trial_key_count"].gt(1) & doi_file_map["token_cycle_count"].eq(1),
            doi_file_map["token_trial_key_count"].eq(0),
        ],
        [
            "ACCEPT_EXACT_MANIFEST_DOI_FILE", "ACCEPT_UNIQUE_TRIAL_TOKEN",
            "ACCEPT_MULTIPLE_EQUIVALENT_RAW_ALIASES", "NOT_APPLICABLE_NO_RAW_TRIAL",
        ],
        default="UNRESOLVED_OR_AMBIGUOUS_DOI_FILE_TRIAL",
    )
    doi_file_map["mapped_trial_key"] = np.where(
        doi_file_map["mapping_status"].eq("ACCEPT_EXACT_MANIFEST_DOI_FILE"),
        doi_file_map["exact_trial_keys"],
        np.where(doi_file_map["mapping_status"].eq("ACCEPT_UNIQUE_TRIAL_TOKEN"), doi_file_map["token_trial_keys"], ""),
    )
    doi_file_map["mapped_cycle"] = np.where(
        doi_file_map["mapping_status"].eq("ACCEPT_EXACT_MANIFEST_DOI_FILE"),
        doi_file_map["exact_cycles"],
        np.where(doi_file_map["mapping_status"].eq("ACCEPT_UNIQUE_TRIAL_TOKEN"), doi_file_map["token_cycles"], ""),
    )
    doi_file_map.to_csv(result_dir / "doi_file_to_trial_registry.tsv", sep="\t", index=False)

    doi = doi.merge(
        doi_file_map[["doi_source_file_norm", "mapped_trial_key", "mapped_cycle", "mapping_status"]],
        on="doi_source_file_norm", how="left", validate="m:1",
    )
    doi = doi.merge(
        raw_alias_targets,
        left_on="trial_token_norm", right_on="trial_code",
        how="left", validate="m:m",
    )
    raw_alias_present = clean(doi["raw_alias_trial_key"]).ne("")
    doi["mapped_trial_key"] = np.where(
        raw_alias_present, clean(doi["raw_alias_trial_key"]), clean(doi["mapped_trial_key"])
    )
    doi["mapped_cycle"] = np.where(
        raw_alias_present, clean(doi["raw_alias_cycle"]), clean(doi["mapped_cycle"])
    )
    glis = pd.read_csv(args.glis_resolver, sep="\t", dtype=str, low_memory=False).fillna("")
    glis["DOI"] = clean(glis["DOI"])
    glis["glis_gid"] = clean_id(glis["glis_gid"])
    doi = doi.merge(
        glis[["DOI", "glis_gid", "resolver_source", "response_sha256"]],
        on="DOI", how="left", validate="m:1",
    )
    doi["glis_gid"] = clean_id(doi["glis_gid"])
    doi_identity = (
        doi.groupby(["mapped_trial_key", "mapped_cycle", "CID_norm", "SID_norm"], dropna=False, sort=True)
        .agg(
            doi_record_rows=("DOI", "size"),
            doi_files=("doi_source_file", joined_unique),
            valid_DOIs=("DOI", lambda x: ";".join(sorted({str(v).strip() for v in x if re.fullmatch(r"10\.\d{4,9}/\S+", str(v).strip(), re.I)}))),
            valid_DOI_count=("DOI", lambda x: len({str(v).strip() for v in x if re.fullmatch(r"10\.\d{4,9}/\S+", str(v).strip(), re.I)})),
            doi_resolved_gids=("glis_gid", joined_unique),
            doi_resolved_gid_count=("glis_gid", count_unique_nonempty),
            resolver_sources=("resolver_source", joined_unique),
            response_hashes=("response_sha256", joined_unique),
            file_mapping_statuses=("mapping_status", joined_unique),
        )
        .reset_index()
        .rename(columns={"mapped_trial_key": "trial_key", "mapped_cycle": "cycle_norm"})
    )
    doi_identity["doi_decision"] = np.select(
        [
            doi_identity["trial_key"].eq("") | doi_identity["cycle_norm"].eq(""),
            doi_identity["valid_DOI_count"].eq(0),
            doi_identity["doi_resolved_gid_count"].eq(0),
            doi_identity["doi_resolved_gid_count"].eq(1),
        ],
        [
            "UNRESOLVED_DOI_FILE_TRIAL",
            "UNRESOLVED_NO_VALID_DOI",
            "UNRESOLVED_VALID_DOI_WITHOUT_GID",
            "ACCEPT_UNIQUE_TRIALWIDE_DOI_GID",
        ],
        default="AMBIGUOUS_MULTIPLE_DOI_GIDS",
    )
    doi_identity["doi_accepted_gid"] = np.where(
        doi_identity["doi_decision"].eq("ACCEPT_UNIQUE_TRIALWIDE_DOI_GID"),
        doi_identity["doi_resolved_gids"],
        "",
    )
    doi_identity.to_csv(result_dir / "doi_trialwide_identity_audit.tsv", sep="\t", index=False)

    genotype = manifest_trialwide.merge(
        doi_identity,
        on=["trial_key", "cycle_norm", "CID_norm", "SID_norm"],
        how="outer",
        validate="1:1",
    ).fillna("")
    genotype["manifest_accepted_gid"] = clean_id(genotype["manifest_accepted_gid"])
    genotype["doi_accepted_gid"] = clean_id(genotype["doi_accepted_gid"])
    genotype["accepted_gid"] = ""
    genotype["registry_decision"] = "UNRESOLVED_NO_ACCEPTED_GID"
    agree = genotype["manifest_accepted_gid"].ne("") & genotype["manifest_accepted_gid"].eq(genotype["doi_accepted_gid"])
    manifest_only = genotype["manifest_accepted_gid"].ne("") & genotype["doi_accepted_gid"].eq("")
    doi_only = genotype["doi_accepted_gid"].ne("") & genotype["manifest_accepted_gid"].eq("")
    conflict = genotype["manifest_accepted_gid"].ne("") & genotype["doi_accepted_gid"].ne("") & ~agree
    genotype.loc[agree, ["accepted_gid", "registry_decision"]] = np.column_stack([
        genotype.loc[agree, "manifest_accepted_gid"],
        np.repeat("ACCEPT_MANIFEST_DOI_CONCORDANT", int(agree.sum())),
    ])
    genotype.loc[manifest_only, ["accepted_gid", "registry_decision"]] = np.column_stack([
        genotype.loc[manifest_only, "manifest_accepted_gid"],
        np.repeat("ACCEPT_UNIQUE_MANIFEST_TRIALWIDE", int(manifest_only.sum())),
    ])
    genotype.loc[doi_only, ["accepted_gid", "registry_decision"]] = np.column_stack([
        genotype.loc[doi_only, "doi_accepted_gid"],
        np.repeat("ACCEPT_UNIQUE_DOI_TRIALWIDE", int(doi_only.sum())),
    ])
    genotype.loc[conflict, "registry_decision"] = "AMBIGUOUS_MANIFEST_DOI_GID_CONFLICT"
    global_cid_sid = (
        genotype[genotype["accepted_gid"].ne("")]
        .groupby(["CID_norm", "SID_norm"], dropna=False, sort=True)
        .agg(
            evidence_rows=("accepted_gid", "size"),
            evidence_gids=("accepted_gid", joined_unique),
            evidence_gid_count=("accepted_gid", count_unique_nonempty),
            evidence_trials=("trial_key", joined_unique),
        )
        .reset_index()
    )
    global_cid_sid["global_identity_decision"] = np.where(
        global_cid_sid["evidence_gid_count"].eq(1),
        "ACCEPT_GLOBAL_UNIQUE_CID_SID",
        "AMBIGUOUS_GLOBAL_CID_SID_MULTIPLE_GIDS",
    )
    global_cid_sid["global_accepted_gid"] = np.where(
        global_cid_sid["evidence_gid_count"].eq(1), global_cid_sid["evidence_gids"], ""
    )
    global_cid_sid.to_csv(result_dir / "global_cid_sid_identity_registry_v2.tsv", sep="\t", index=False)
    genotype = genotype.merge(
        global_cid_sid[["CID_norm", "SID_norm", "global_accepted_gid", "global_identity_decision"]],
        on=["CID_norm", "SID_norm"], how="left", validate="m:1",
    ).fillna("")
    global_recovery = genotype["accepted_gid"].eq("") & genotype["global_accepted_gid"].ne("")
    genotype.loc[global_recovery, "accepted_gid"] = genotype.loc[global_recovery, "global_accepted_gid"]
    genotype.loc[global_recovery, "registry_decision"] = "ACCEPT_GLOBAL_UNIQUE_CID_SID"
    genotype["panel_sample_id"] = np.where(genotype["accepted_gid"].ne(""), "GID" + genotype["accepted_gid"], "")
    genotype["registry_version"] = REGISTRY_VERSION
    genotype.to_csv(result_dir / "genotype_alias_registry_v2.tsv", sep="\t", index=False)

    env = pd.read_csv(args.environment_aliases, sep="\t", dtype=str, low_memory=False).fillna("")
    env["registry_version"] = REGISTRY_VERSION
    env["alias_decision"] = np.where(env["mapping_status"].eq("ACCEPTED_ALIAS"), "ACCEPT", "REJECT")
    env.to_csv(result_dir / "environment_alias_registry_v2.tsv", sep="\t", index=False)

    trait_distinct = connection.execute(
        """
        SELECT DISTINCT trait_key, trait_name_original, trait_name_canonical,
               trait_mapping_status, raw_unit, unit
        FROM read_parquet(?)
        ORDER BY trait_key, trait_name_original, trait_name_canonical, raw_unit, unit
        """,
        [str(raw_path)],
    ).fetch_df()
    trait = (
        trait_distinct.groupby("trait_key", dropna=False, sort=True)
        .agg(
            original_labels=("trait_name_original", joined_unique),
            canonical_labels=("trait_name_canonical", joined_unique),
            canonical_label_count=("trait_name_canonical", count_unique_nonempty),
            standard_units=("unit", joined_unique),
            standard_unit_count=("unit", count_unique_nonempty),
            mapping_statuses=("trait_mapping_status", joined_unique),
        )
        .reset_index()
    )
    trait["trait_alias_decision"] = np.where(
        trait["canonical_label_count"].eq(1)
        & trait["standard_unit_count"].le(1)
        & ~trait["mapping_statuses"].str.contains("AMBIGUOUS", case=False, na=False),
        "ACCEPT_UNIQUE_TRAIT_UNIT",
        "AMBIGUOUS_TRAIT_OR_UNIT",
    )
    trait["accepted_canonical_trait"] = np.where(
        trait["trait_alias_decision"].eq("ACCEPT_UNIQUE_TRAIT_UNIT"), trait["canonical_labels"], ""
    )
    trait["accepted_standard_unit"] = np.where(
        trait["trait_alias_decision"].eq("ACCEPT_UNIQUE_TRAIT_UNIT"), trait["standard_units"], ""
    )
    trait["registry_version"] = REGISTRY_VERSION
    trait.to_csv(result_dir / "trait_alias_registry_v2.tsv", sep="\t", index=False)

    unit_pairs = trait_distinct[["trait_key", "trait_name_canonical", "raw_unit", "unit"]].drop_duplicates().copy()
    unit_pairs = unit_pairs.merge(
        trait[["trait_key", "trait_alias_decision", "accepted_canonical_trait", "accepted_standard_unit"]],
        on="trait_key", how="left", validate="m:1",
    )
    raw_unit_norm = norm(unit_pairs["raw_unit"])
    std_unit_norm = norm(unit_pairs["accepted_standard_unit"])
    unit_pairs["scale"] = np.nan
    unit_pairs["offset"] = np.nan
    unit_pairs["unit_rule"] = "UNRESOLVED_UNIT_PAIR"
    accepted_trait = unit_pairs["trait_alias_decision"].eq("ACCEPT_UNIQUE_TRAIT_UNIT")
    blank_raw = accepted_trait & raw_unit_norm.eq("") & std_unit_norm.ne("")
    identity = accepted_trait & raw_unit_norm.eq(std_unit_norm)
    unit_pairs.loc[blank_raw | identity, ["scale", "offset"]] = [1.0, 0.0]
    unit_pairs.loc[blank_raw, "unit_rule"] = "ASSUME_TRAIT_STANDARD_UNIT_RAW_BLANK"
    unit_pairs.loc[identity, "unit_rule"] = "IDENTITY_UNIT"
    conversions = {
        ("GRAIN_YIELD", "KG/HA", "T/HA"): (0.001, 0.0, "KG_HA_TO_T_HA"),
        ("GRAIN_YIELD", "G/M2", "T/HA"): (0.01, 0.0, "G_M2_TO_T_HA"),
        ("ABOVE_GROUND_BIOMASS", "KG/HA", "T/HA"): (0.001, 0.0, "KG_HA_TO_T_HA"),
        ("ABOVE_GROUND_BIOMASS", "G/M2", "T/HA"): (0.01, 0.0, "G_M2_TO_T_HA"),
        ("PLANT_HEIGHT", "M", "CM"): (100.0, 0.0, "M_TO_CM"),
    }
    for (trait_name, raw_value, standard_value), (scale, offset, rule) in conversions.items():
        mask = accepted_trait & norm(unit_pairs["accepted_canonical_trait"]).eq(trait_name) & raw_unit_norm.eq(raw_value) & std_unit_norm.eq(standard_value)
        unit_pairs.loc[mask, ["scale", "offset", "unit_rule"]] = [scale, offset, rule]
    unit_pairs["unit_rule_version"] = REGISTRY_VERSION
    unit_pairs = unit_pairs.sort_values(["trait_key", "raw_unit", "unit_rule"]).drop_duplicates(
        ["trait_key", "raw_unit"], keep="first"
    )
    if unit_pairs[["trait_key", "raw_unit"]].duplicated().any():
        raise RuntimeError("Trait-unit registry key is not unique")
    unit_pairs.to_csv(result_dir / "trait_unit_rules_v2.tsv", sep="\t", index=False)

    registry_files = [
        "genotype_alias_registry_v2.tsv", "environment_alias_registry_v2.tsv",
        "trait_alias_registry_v2.tsv", "trait_unit_rules_v2.tsv",
        "doi_file_to_trial_registry.tsv", "global_cid_sid_identity_registry_v2.tsv",
    ]
    summary = {
        "status": "PASS_REGISTRIES_BUILT",
        "registry_version": REGISTRY_VERSION,
        "raw_trial_keys": int(trial_rows[["trial_key", "cycle"]].drop_duplicates().shape[0]),
        "doi_files": int(doi_file_map["doi_source_file_norm"].nunique()),
        "doi_files_unique_trial_mapped": int(doi_file_map["mapping_status"].str.startswith("ACCEPT").sum()),
        "doi_files_not_applicable_no_raw_trial": int(doi_file_map["mapping_status"].eq("NOT_APPLICABLE_NO_RAW_TRIAL").sum()),
        "doi_files_unresolved_or_ambiguous": int(doi_file_map["mapping_status"].eq("UNRESOLVED_OR_AMBIGUOUS_DOI_FILE_TRIAL").sum()),
        "genotype_alias_keys": len(genotype),
        "genotype_alias_keys_accepted": int(genotype["registry_decision"].str.startswith("ACCEPT").sum()),
        "genotype_alias_keys_ambiguous": int(genotype["registry_decision"].str.startswith("AMBIGUOUS").sum()),
        "genotype_alias_keys_unresolved": int(genotype["registry_decision"].str.startswith("UNRESOLVED").sum()),
        "trait_alias_keys": len(trait),
        "trait_alias_keys_accepted": int(trait["trait_alias_decision"].eq("ACCEPT_UNIQUE_TRAIT_UNIT").sum()),
        "trait_alias_keys_ambiguous": int(trait["trait_alias_decision"].ne("ACCEPT_UNIQUE_TRAIT_UNIT").sum()),
        "environment_alias_rows": len(env),
        "registry_sha256": {},
    }
    for name in registry_files:
        summary["registry_sha256"][name] = file_sha256(result_dir / name)
    (result_dir / "registry_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
