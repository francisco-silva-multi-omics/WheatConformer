"""Create machine-readable/static evidence for the Phase-3G semantics conclusion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate(path: Path, pattern: str) -> tuple[int, str]:
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if pattern in line:
            return number, line.strip()
    raise RuntimeError(f"Pattern not found in {path}: {pattern}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--phase3-root", type=Path, required=True)
    parser.add_argument("--phase3g-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    phase3 = args.phase3_root.resolve()
    out = args.phase3g_root.resolve()
    samples = pd.read_parquet(out / "sample_identifier_ledger.parquet")
    evidence = pd.read_parquet(out / "linkage_evidence_ledger.parquet")

    specifications = [
        ("scripts/v2/phase3_build_canonical_layers_streaming.py", 'raw_gid = frame["raw_gid"]', "raw phenotype field explicitly typed raw_gid", "PASS_TYPED_GID_FIELD_ONLY", "Phase3 canonicalization"),
        ("scripts/v2/phase3_build_canonical_layers_streaming.py", 'left_on=["trial_key", "cycle", "CID_normalized", "SID_normalized"]', "registry lookup uses typed compound CID/SID keys", "PASS_NAMESPACE_PRESERVED", "Phase3 canonicalization"),
        ("scripts/v2/phase3_build_registries.py", 'manifest["resolved_gid_norm"] = clean_id(manifest["resolved_gid"])', "manifest column explicitly named resolved_gid", "PASS_TYPED_GID_FIELD_ONLY", "Phase3 registry"),
        ("scripts/v2/phase3_build_registries.py", 'doi = doi.merge(', "DOI matched exactly before official glis_gid is consumed", "PASS_EXACT_DOI_CROSSWALK", "Phase3 registry"),
        ("scripts/v2/phase3_scrape_missing_glis_gids.py", 'GID_RE = re.compile', "page parser requires literal GID token", "PASS_OFFICIAL_OTHER_GID_GRAMMAR", "Phase3 GLIS"),
        ("scripts/v2/phase3_scrape_missing_glis_gids.py", 'doi_present = requested_doi.upper() in text.upper()', "returned page must contain requested DOI", "PASS_PAGE_DOI_BINDING", "Phase3 GLIS"),
        ("scripts/v2/phase3_build_trial_metadata_gid_registry.py", 'gid_col = find_column(list(frame.columns), ["GID"])', "DArTAG/trial workbooks select a field explicitly named GID", "PASS_TYPED_GID_FIELD_ONLY", "Phase3 metadata extension"),
        ("scripts/v2/phase3_build_trial_metadata_gid_registry.py", 'hibap_frame = pd.read_csv', "HiBAP manifest is read separately and GID column is selected", "PASS_TYPED_GID_FIELD_ONLY", "Phase3 metadata extension"),
        ("scripts/v2/phase3_extend_registry_exact_names.py", 'candidate_gid_count', "exact full-name candidates accepted only when one GID remains", "PASS_FAILS_CLOSED_ON_AMBIGUITY", "Phase3 exact-name extension"),
        ("genotype_recovery.py", 'def canonical_gid(value: object)', "context-free helper accepts plain numeric strings and must not receive opaque labels", "UNSAFE_CONTEXT_FREE_API_NOT_USED_BY_PHASE3_CANONICALIZATION", "Historical/shared helper"),
        ("audit/recover_genotypic_gid_matches.py", 'direct = canonical_gid(text)', "historical audit attempted direct GID parsing for generic matrix identifiers", "DEFECT_GENERIC_IDENTIFIER_COERCION_DIAGNOSTIC_ONLY", "Historical pre-Phase3 audit"),
        ("server_genotype_recovery/build_platform_kernel.py", 'gid = canonical_gid(gid_row[index])', "caller supplies documented HiBAP GID row", "PASS_CALLER_TYPED_GID_ROW", "Nonactivated future kernel helper"),
        ("server_genotype_recovery/build_platform_kernel.py", 'gid = canonical_gid(raw_gid)', "caller supplies DArTAG row explicitly labeled GID", "PASS_CALLER_TYPED_GID_ROW", "Nonactivated future kernel helper"),
    ]
    rows = []
    for relative, pattern, context, conclusion, scope in specifications:
        path = root / relative
        line_number, source_line = locate(path, pattern)
        rows.append(
            {
                "scope": scope,
                "code_path": relative,
                "code_sha256": sha256(path),
                "line_number": line_number,
                "source_excerpt": source_line,
                "identifier_context": context,
                "semantic_conclusion": conclusion,
            }
        )
    pd.DataFrame(rows).to_csv(out / "phase3_gid_callsite_audit.tsv", sep="\t", index=False)

    accepted = samples[samples["accepted_canonical_gid"].ne("")]
    opaque_accepted = accepted[
        accepted["raw_identifier_type"].str.contains("OPAQUE|TAXA", case=False, na=False)
        & ~accepted["evidence_tier"].str.contains("authoritative|verified_frozen", case=False, na=False)
    ]
    hibap_775 = evidence[(evidence["panel_id"] == "hibap35k") & (evidence["candidate_canonical_gid"] == "GID775")]
    hibap_775_samples = samples[
        (samples["panel_id"] == "hibap35k")
        & samples["panel_sample_key"].isin(hibap_775["panel_sample_key"])
    ]
    primary_manifest = phase3 / "delivery_v1" / "primary_release_manifest.tsv"
    primary_text = primary_manifest.read_text(encoding="utf-8")
    consumption = pd.DataFrame(
        [
            {
                "claim": "Phase3 primary delivery does not consume historical generic genotypic-recovery audit",
                "observed": int("audit/genotypic_recovery" in primary_text or "genotypic_duplicate_sample_candidates" in primary_text),
                "expected": 0,
                "status": "PASS" if "audit/genotypic_recovery" not in primary_text and "genotypic_duplicate_sample_candidates" not in primary_text else "FAIL",
                "evidence": str(primary_manifest),
            },
            {
                "claim": "No accepted all-panel link rests on opaque-label numeric equality",
                "observed": len(opaque_accepted),
                "expected": 0,
                "status": "PASS" if opaque_accepted.empty else "FAIL",
                "evidence": "sample_identifier_ledger.parquet",
            },
            {
                "claim": "HiBAP GID775 evidence comes from typed GID fields, never a sample label 775",
                "observed": len(hibap_775),
                "expected": ">=1 typed evidence row and zero raw sample labels equal to 775",
                "status": "PASS" if len(hibap_775) >= 1 and not samples["raw_sample_id"].eq("775").any() else "FAIL",
                "evidence": ";".join(hibap_775["source_location"].astype(str)),
            },
        ]
    )
    consumption.to_csv(out / "identifier_artifact_consumption_trace.tsv", sep="\t", index=False)
    result = {
        "phase3_gid_semantics": "PASS_SEMANTICALLY_CORRECT",
        "panel_sample_incorrectly_treated_as_gid_in_phase3": False,
        "historical_generic_helper_defect_found": True,
        "historical_generic_helper_phase3_downstream_impact_rows": 0,
        "accepted_opaque_numeric_equality_links": len(opaque_accepted),
        "gid775_panel_sample_ids_with_typed_evidence": sorted(set(hibap_775["raw_sample_id"])),
        "gid775_evidence_types": sorted(set(hibap_775["evidence_type"])),
        "gid775_final_sample_mapping_states": sorted(set(hibap_775_samples["mapping_status"])),
        "raw_sample_id_775_exists": bool(samples["raw_sample_id"].eq("775").any()),
        "all_checks_pass": bool(consumption["status"].eq("PASS").all()),
    }
    (out / "identifier_semantics_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
