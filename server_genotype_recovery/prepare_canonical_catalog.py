from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from genotype_recovery import canonical_gid, load_canonical_catalog, normalize_identifier


IDENTIFIER_COLUMNS = (
    "CID",
    "SID",
    "fieldbook_gid",
    "DOI",
    "doi_gid",
    "glis_gid",
    "resolved_gid",
    "panel_sample_id_expected",
    "pheno_gid",
)
MANIFEST_COLUMNS = (
    *IDENTIFIER_COLUMNS,
    "cross_name",
    "gid_source",
    "fieldbook_glis_gid_conflict",
    "pheno_gid_conflict",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_true(value: object) -> bool:
    return normalize_identifier(value).lower() in {"1", "true", "yes", "y"}


def resolve_optional_path(root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def build_catalog_from_authoritative_sources(
    *,
    manifest_path: Path,
    canonical_observations_path: Path | None,
    hmp_order_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise SystemExit(f"Resolved trial genotype manifest is missing or empty: {manifest_path}")

    manifest = pd.read_csv(
        manifest_path,
        sep="\t",
        dtype=str,
        usecols=lambda column: column in MANIFEST_COLUMNS,
        low_memory=False,
    )
    if "resolved_gid" not in manifest.columns:
        raise SystemExit(f"Resolved trial genotype manifest lacks resolved_gid: {manifest_path}")

    records: dict[str, dict[str, object]] = {}
    for row in manifest.itertuples(index=False):
        values = row._asdict()
        gid = canonical_gid(values.get("resolved_gid", ""))
        if not gid:
            continue
        record = records.setdefault(
            gid,
            {
                "source_rows": 0,
                "mapping_sources": set(),
                "raw_identifiers": defaultdict(set),
                "pedigree_available": False,
                "conflict_flag": False,
            },
        )
        record["source_rows"] = int(record["source_rows"]) + 1
        source = normalize_identifier(values.get("gid_source", ""))
        if source:
            record["mapping_sources"].add(source)
        record["pedigree_available"] = bool(record["pedigree_available"]) or bool(
            normalize_identifier(values.get("cross_name", ""))
        )
        record["conflict_flag"] = (
            bool(record["conflict_flag"])
            or is_true(values.get("fieldbook_glis_gid_conflict", ""))
            or is_true(values.get("pheno_gid_conflict", ""))
        )
        for column in IDENTIFIER_COLUMNS:
            value = normalize_identifier(values.get(column, ""))
            if value:
                record["raw_identifiers"][column].add(value)

    observation_counts = pd.Series(dtype="int64")
    observation_source_status = "absent"
    if canonical_observations_path is not None and canonical_observations_path.is_file():
        observation_frame = pd.read_parquet(
            canonical_observations_path,
            columns=["canonical_germplasm_key"],
        )
        observation_ids = observation_frame["canonical_germplasm_key"].map(canonical_gid)
        observation_counts = observation_ids[observation_ids.ne("")].value_counts()
        observation_source_status = "loaded"

    hmp_ids: set[str] = set()
    hmp_source_status = "absent"
    if hmp_order_path is not None and hmp_order_path.is_file():
        hmp = pd.read_csv(hmp_order_path, sep="\t", dtype=str)
        sample_column = "sample_id" if "sample_id" in hmp.columns else hmp.columns[0]
        hmp_ids = {canonical_gid(value) for value in hmp[sample_column]}
        hmp_ids.discard("")
        hmp_source_status = "loaded"

    canonical_ids = sorted(set(records) | set(observation_counts.index))
    rows: list[dict[str, object]] = []
    for gid in canonical_ids:
        record = records.get(
            gid,
            {
                "source_rows": 0,
                "mapping_sources": set(),
                "raw_identifiers": defaultdict(set),
                "pedigree_available": False,
                "conflict_flag": False,
            },
        )
        raw_identifiers = {
            column: sorted(values)
            for column, values in record["raw_identifiers"].items()
            if values
        }
        rows.append(
            {
                "canonical_gid": gid,
                "source_rows": int(record["source_rows"]),
                "mapping_sources": ";".join(sorted(record["mapping_sources"])),
                "raw_identifiers": json.dumps(raw_identifiers, sort_keys=True),
                "marker_available_hmp_qc": gid in hmp_ids,
                "pedigree_available": bool(record["pedigree_available"]),
                "conflict_flag": bool(record["conflict_flag"]),
                "canonical_observation_rows": int(observation_counts.get(gid, 0)),
            }
        )
    catalog = pd.DataFrame(rows)
    if catalog.empty or not catalog["canonical_gid"].is_unique:
        raise SystemExit("Prepared canonical genotype catalog is empty or has duplicate GIDs")

    metadata = {
        "manifest_rows": int(len(manifest)),
        "manifest_canonical_gid_count": int(len(records)),
        "canonical_observation_gid_count": int(len(observation_counts)),
        "catalog_canonical_gid_count": int(len(catalog)),
        "observation_source_status": observation_source_status,
        "hmp_source_status": hmp_source_status,
    }
    return catalog, metadata


def prior_audit_candidates(root: Path, explicit: Path | None) -> list[Path]:
    if explicit is not None:
        if not explicit.is_file() or explicit.stat().st_size == 0:
            raise SystemExit(f"Explicit prior audit catalog is missing or empty: {explicit}")
        return [explicit]
    audit_root = root / "audit"
    if not audit_root.is_dir():
        return []
    return sorted(
        {
            path.resolve()
            for path in audit_root.rglob("canonical_genotype_mapping_audited.csv")
            if path.is_file() and path.stat().st_size > 0
        }
    )


def attach_prior_audit_flags(
    catalog: pd.DataFrame,
    *,
    candidates: list[Path],
    explicit: bool,
) -> tuple[pd.DataFrame, Path | None, pd.DataFrame]:
    canonical_ids = set(catalog["canonical_gid"])
    candidate_rows: list[dict[str, object]] = []
    compatible: list[tuple[int, str, Path, pd.DataFrame]] = []
    for path in candidates:
        try:
            prior, _ = load_canonical_catalog(path)
            prior_ids = set(prior["canonical_gid"])
            has_flag = "audit_genotypic_match" in prior.columns
            status = "compatible" if prior_ids == canonical_ids and has_flag else "incompatible"
            detail = (
                "canonical IDs match and audit flag is present"
                if status == "compatible"
                else (
                    f"missing_ids={len(canonical_ids - prior_ids)}; "
                    f"extra_ids={len(prior_ids - canonical_ids)}; audit_flag_present={has_flag}"
                )
            )
            if status == "compatible":
                compatible.append((path.stat().st_mtime_ns, str(path), path, prior))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            status = "read_error"
            detail = f"{type(exc).__name__}: {exc}"
        candidate_rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "status": status,
                "detail": detail,
            }
        )

    if explicit and not compatible:
        raise SystemExit("Explicit prior audit catalog is incompatible with the authoritative GID universe")
    if not compatible:
        catalog["audit_genotypic_match"] = pd.NA
        return catalog, None, pd.DataFrame(candidate_rows)

    _, _, selected_path, selected = max(compatible)
    flags = selected.set_index("canonical_gid")["audit_genotypic_match"]
    normalized = flags.astype(str).str.lower().map({"true": True, "false": False})
    catalog["audit_genotypic_match"] = catalog["canonical_gid"].map(normalized)
    return catalog, selected_path, pd.DataFrame(candidate_rows)


def prepare_catalog(
    *,
    root: Path,
    manifest_path: Path,
    canonical_observations_path: Path | None,
    hmp_order_path: Path | None,
    output_path: Path,
    explicit_prior_audit_catalog: Path | None = None,
) -> dict[str, object]:
    catalog, metadata = build_catalog_from_authoritative_sources(
        manifest_path=manifest_path,
        canonical_observations_path=canonical_observations_path,
        hmp_order_path=hmp_order_path,
    )
    candidates = prior_audit_candidates(root, explicit_prior_audit_catalog)
    catalog, selected_prior, candidate_frame = attach_prior_audit_flags(
        catalog,
        candidates=candidates,
        explicit=explicit_prior_audit_catalog is not None,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False)
    candidate_path = output_path.with_name("canonical_catalog_prior_audit_candidates.tsv")
    candidate_frame.to_csv(candidate_path, sep="\t", index=False)

    provenance = {
        "status": "PASS",
        "catalog_path": str(output_path),
        "catalog_sha256": sha256_file(output_path),
        "authoritative_manifest_path": str(manifest_path),
        "authoritative_manifest_sha256": sha256_file(manifest_path),
        "canonical_observations_path": (
            str(canonical_observations_path)
            if canonical_observations_path is not None and canonical_observations_path.is_file()
            else None
        ),
        "hmp_order_path": (
            str(hmp_order_path)
            if hmp_order_path is not None and hmp_order_path.is_file()
            else None
        ),
        "prior_audit_catalog_path": str(selected_prior) if selected_prior else None,
        "prior_audit_catalog_sha256": sha256_file(selected_prior) if selected_prior else None,
        "prior_audit_comparison_available": selected_prior is not None,
        **metadata,
    }
    provenance_path = output_path.with_name("canonical_genotype_catalog_provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the canonical trial-GID catalog required by platform recovery."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata_outputs/all_trials_genotype_manifest_resolved.tsv"),
    )
    parser.add_argument(
        "--canonical-observations",
        type=Path,
        default=Path(
            "integrated_database/canonical_trial_genotype_environment_plot_table.parquet"
        ),
    )
    parser.add_argument(
        "--hmp-order",
        type=Path,
        default=Path("genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv"),
    )
    parser.add_argument(
        "--prior-audit-catalog",
        type=Path,
        help="Optional explicit prior forensic catalog used only for preview-audit flags.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("audit/genotypic_recovery/canonical_genotype_catalog.csv"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = resolve_optional_path(root, args.manifest)
    canonical_observations_path = resolve_optional_path(root, args.canonical_observations)
    hmp_order_path = resolve_optional_path(root, args.hmp_order)
    prior_path = resolve_optional_path(root, args.prior_audit_catalog)
    output_path = resolve_optional_path(root, args.out)
    assert manifest_path is not None and output_path is not None
    provenance = prepare_catalog(
        root=root,
        manifest_path=manifest_path,
        canonical_observations_path=canonical_observations_path,
        hmp_order_path=hmp_order_path,
        output_path=output_path,
        explicit_prior_audit_catalog=prior_path,
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
