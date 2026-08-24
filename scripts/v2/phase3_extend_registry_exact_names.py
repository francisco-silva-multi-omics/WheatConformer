"""Extend a versioned genotype registry using exact, globally unique names."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


REGISTRY_VERSION = "stage1_v2_registries_2026_07_30_v6"
GENERIC_NAMES = {
    "UNKNOWN", "DESCONOCIDO", "LOCAL CHECK", "LOCAL CHECK 1", "LOCAL CHECK 2",
    "CHECK", "CHECK 1", "CHECK 2", "TESTIGO", "TESTIGO LOCAL", "ENTRY", "LINE",
    "N/A", "NA", "NONE", "NULL", "0", "-", ".",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-registries", type=Path, required=True)
    parser.add_argument("--name-candidates", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    base_dir = args.base_registries.resolve()
    base = pd.read_csv(base_dir / "genotype_alias_registry_v2.tsv", sep="\t", dtype=str, keep_default_na=False)
    candidates = pd.read_csv(args.name_candidates, sep="\t", dtype=str, keep_default_na=False)
    candidates["candidate_gid"] = candidates["candidate_gid"].str.replace(r"^GID", "", regex=True).str.replace(r"\.0$", "", regex=True)
    candidates["generic_name_rejected"] = candidates["normalized_genotype_name"].isin(GENERIC_NAMES) | candidates["normalized_genotype_name"].str.fullmatch(
        r"(?:UNKNOWN|CHECK|LOCAL CHECK|TESTIGO|DESCONOCIDO)(?:\s*#?\d+)?", case=False, na=False
    )
    keys = ["trial_name", "cycle", "CID_normalized", "SID_normalized"]
    grouped = (
        candidates.groupby(keys, dropna=False, sort=True)
        .agg(
            candidate_names=("genotype_name", lambda x: ";".join(sorted(set(map(str, x))))),
            normalized_candidate_names=("normalized_genotype_name", lambda x: ";".join(sorted(set(map(str, x))))),
            candidate_gids=("candidate_gid", lambda x: ";".join(sorted(set(map(str, x))))),
            candidate_gid_count=("candidate_gid", "nunique"),
            numeric_rows=("numeric_rows", lambda x: sum(int(v) for v in x)),
            any_generic_name=("generic_name_rejected", "any"),
        )
        .reset_index()
    )
    grouped["exact_name_decision"] = "AMBIGUOUS_OR_GENERIC_EXACT_NAME"
    accepted = grouped["candidate_gid_count"].eq(1) & ~grouped["any_generic_name"]
    grouped.loc[accepted, "exact_name_decision"] = "ACCEPT_EXACT_GLOBALLY_UNIQUE_GENOTYPE_NAME"
    grouped["exact_name_accepted_gid"] = grouped["candidate_gids"].where(accepted, "")
    grouped.to_csv(result_dir / "exact_name_identity_registry_v2.tsv", sep="\t", index=False)

    evidence = grouped.rename(columns={
        "trial_name": "trial_key", "cycle": "cycle_norm",
        "CID_normalized": "CID_norm", "SID_normalized": "SID_norm",
    })
    base_keys = ["trial_key", "cycle_norm", "CID_norm", "SID_norm"]
    if base.duplicated(base_keys).any() or evidence.duplicated(base_keys).any():
        raise RuntimeError("Base or exact-name registry key is not unique")
    merged = base.merge(
        evidence[base_keys + ["candidate_names", "normalized_candidate_names", "candidate_gids", "candidate_gid_count", "numeric_rows", "exact_name_decision", "exact_name_accepted_gid"]],
        on=base_keys, how="outer", validate="1:1",
    ).fillna("")
    if "accepted_gid" not in merged:
        merged["accepted_gid"] = ""
    if "registry_decision" not in merged:
        merged["registry_decision"] = "UNRESOLVED_NO_ACCEPTED_GID"
    existing = merged["accepted_gid"].ne("")
    name_gid = merged["exact_name_accepted_gid"].ne("")
    conflict = existing & name_gid & merged["accepted_gid"].ne(merged["exact_name_accepted_gid"])
    recovery = ~existing & name_gid
    merged.loc[recovery, "accepted_gid"] = merged.loc[recovery, "exact_name_accepted_gid"]
    merged.loc[recovery, "registry_decision"] = "ACCEPT_EXACT_GLOBALLY_UNIQUE_GENOTYPE_NAME"
    merged.loc[conflict, "accepted_gid"] = ""
    merged.loc[conflict, "registry_decision"] = "AMBIGUOUS_EXISTING_GID_EXACT_NAME_CONFLICT"
    merged["panel_sample_id"] = merged["accepted_gid"].map(lambda value: f"GID{value}" if value else "")
    merged["registry_version"] = REGISTRY_VERSION
    merged.to_csv(result_dir / "genotype_alias_registry_v2.tsv", sep="\t", index=False)
    for name in [
        "environment_alias_registry_v2.tsv", "trait_alias_registry_v2.tsv", "trait_unit_rules_v2.tsv",
        "doi_file_to_trial_registry.tsv", "doi_trialwide_identity_audit.tsv",
        "manifest_trialwide_identity_audit.tsv", "global_cid_sid_identity_registry_v2.tsv", "raw_trial_registry.tsv",
    ]:
        source = base_dir / name
        if source.exists():
            (result_dir / name).write_bytes(source.read_bytes())
    summary = {
        "status": "PASS_EXACT_NAME_REGISTRY_EXTENSION",
        "registry_version": REGISTRY_VERSION,
        "candidate_keys": len(grouped),
        "accepted_exact_name_keys": int(accepted.sum()),
        "generic_or_ambiguous_name_keys_rejected": int((~accepted).sum()),
        "new_registry_keys": len(merged),
        "newly_recovered_keys": int(recovery.sum()),
        "existing_name_conflicts": int(conflict.sum()),
        "accepted_registry_keys": int(merged["accepted_gid"].ne("").sum()),
        "genotype_registry_sha256": sha256(result_dir / "genotype_alias_registry_v2.tsv"),
    }
    (result_dir / "registry_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
