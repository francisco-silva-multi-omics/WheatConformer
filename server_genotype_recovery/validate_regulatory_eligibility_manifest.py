from __future__ import annotations

import argparse
import json
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from genotype_recovery import canonical_gid
from server_genotype_recovery.build_regulatory_eligibility_manifest import (
    build_gid_manifest,
    detect_column,
    embedding_evidence,
    load_gid_set,
    path_dictionary_ids,
    projection_work_queue,
    read_table,
    sha256_file,
    summary_tables,
)


ARTIFACT_NAMES = [
    "regulatory_genotype_eligibility_manifest.tsv.gz",
    "regulatory_panel_evidence.tsv",
    "regulatory_eligibility_status_summary.tsv",
    "regulatory_eligibility_panel_summary.tsv",
    "regulatory_projection_work_queue.tsv",
    "regulatory_eligibility_provenance.json",
]


def add_check(
    rows: list[dict[str, object]], name: str, passed: bool, detail: str
) -> None:
    rows.append(
        {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}
    )


def bool_series(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "pass"}
    )


def file_identity_check(
    checks: list[dict[str, object]],
    *,
    name: str,
    path_value: object,
    expected_sha256: object,
    required: bool,
) -> None:
    path = Path(str(path_value))
    exists = path.is_file() and path.stat().st_size > 0
    if not required and not exists:
        add_check(checks, name, not str(expected_sha256 or ""), "optional input absent")
        return
    observed = sha256_file(path) if exists else ""
    expected = str(expected_sha256 or "")
    add_check(
        checks,
        name,
        exists and bool(expected) and observed == expected,
        f"path={path}; expected={expected}; observed={observed or 'MISSING'}",
    )


def frame_matches(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_by: list[str],
) -> tuple[bool, str]:
    missing = sorted(set(expected.columns).difference(actual.columns))
    if missing:
        return False, f"missing_columns={missing}"
    def stable_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, bool):
            return str(value).lower()
        text = str(value).strip()
        if text.lower() in {"true", "false"}:
            return text.lower()
        try:
            number = Decimal(text)
            return format(number.normalize(), "f")
        except (InvalidOperation, ValueError):
            return text

    left = actual[expected.columns].sort_values(sort_by, kind="stable").reset_index(drop=True)
    right = expected.sort_values(sort_by, kind="stable").reset_index(drop=True)
    left = left.apply(lambda column: column.map(stable_value))
    right = right.apply(lambda column: column.map(stable_value))
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as exc:
        return False, str(exc).replace("\n", " ")[:1000]
    return True, f"rows={len(left)}; columns={len(left.columns)}"


def panel_samples(panel_evidence: pd.DataFrame) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for row in panel_evidence.to_dict("records"):
        path = Path(str(row["sample_order_path"]))
        frame = read_table(path)
        id_col = detect_column(frame, ["sample_id", "panel_sample_id", "genotype_id"])
        if id_col is None:
            raise ValueError(f"Panel order lacks a recognized sample ID column: {path}")
        ids = {gid for value in frame[id_col] if (gid := canonical_gid(value))}
        output[str(row["panel_id"])] = ids
    return output


def validate_artifacts(
    root: Path, out_dir: Path, code_root: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    checks: list[dict[str, object]] = []
    missing = [name for name in ARTIFACT_NAMES if not (out_dir / name).is_file()]
    add_check(checks, "required_artifacts_present", not missing, f"missing={missing}")
    if missing:
        return pd.DataFrame(checks), {}

    provenance_path = out_dir / "regulatory_eligibility_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    no_outcomes = (
        provenance.get("status") == "PASS"
        and provenance.get("phenotype_values_read") is False
        and provenance.get("outer_test_metrics_read") is False
        and provenance.get("final_holdout_outcomes_read") is False
    )
    add_check(
        checks,
        "outcome_free_provenance",
        no_outcomes,
        "phenotype, outer-test and final-holdout outcomes must remain unread",
    )
    for key in [
        "canonical_catalog",
        "recovered_manifest",
        "kernel_qc",
        "regulatory_retention_policy",
        "pedigree_order",
    ]:
        file_identity_check(
            checks,
            name=f"input_identity_{key}",
            path_value=provenance.get(key, ""),
            expected_sha256=provenance.get(f"{key}_sha256", ""),
            required=key != "pedigree_order" or bool(provenance.get("pedigree_order_present")),
        )
    for key, presence_key in [
        ("marker_projection", "marker_projection_present"),
        ("path_dictionary", "path_dictionary_present"),
    ]:
        file_identity_check(
            checks,
            name=f"input_identity_{key}",
            path_value=provenance.get(key, ""),
            expected_sha256=provenance.get(f"{key}_sha256", ""),
            required=bool(provenance.get(presence_key)),
        )
    coordinate_failures = []
    for source in provenance.get("coordinate_sources", []):
        path = Path(str(source.get("path", "")))
        expected = str(source.get("sha256", ""))
        observed = sha256_file(path) if path.is_file() else ""
        if not expected or observed != expected:
            coordinate_failures.append(str(path))
    add_check(
        checks,
        "coordinate_source_identities",
        not coordinate_failures,
        f"sources={len(provenance.get('coordinate_sources', []))}; failures={coordinate_failures}",
    )
    embedding_failures = []
    for source in provenance.get("embedding_order_sources", []):
        path = Path(str(source.get("path", "")))
        expected = str(source.get("sha256", ""))
        observed = sha256_file(path) if path.is_file() else ""
        if not expected or observed != expected:
            embedding_failures.append(str(path))
    add_check(
        checks,
        "embedding_order_identities",
        not embedding_failures,
        f"sources={len(provenance.get('embedding_order_sources', []))}; "
        f"failures={embedding_failures}",
    )
    builder_path = code_root / "server_genotype_recovery/build_regulatory_eligibility_manifest.py"
    builder_matches = (
        builder_path.is_file()
        and provenance.get("builder_sha256") == sha256_file(builder_path)
    )
    add_check(
        checks,
        "builder_identity",
        builder_matches,
        f"expected={provenance.get('builder_sha256', 'MISSING')}; "
        f"observed={sha256_file(builder_path) if builder_path.is_file() else 'MISSING'}",
    )

    manifest = read_table(out_dir / ARTIFACT_NAMES[0])
    panel_evidence = read_table(out_dir / ARTIFACT_NAMES[1])
    status_summary = read_table(out_dir / ARTIFACT_NAMES[2])
    panel_summary = read_table(out_dir / ARTIFACT_NAMES[3])
    work_queue = read_table(out_dir / ARTIFACT_NAMES[4])
    catalog = pd.read_csv(Path(provenance["canonical_catalog"]), dtype=str)
    catalog["canonical_gid"] = catalog["canonical_gid"].map(canonical_gid)
    catalog = catalog[catalog["canonical_gid"].ne("")].drop_duplicates("canonical_gid")
    catalog_ids = set(catalog["canonical_gid"])
    manifest_ids = set(manifest["canonical_gid"].map(canonical_gid))
    canonical_manifest = manifest["canonical_gid"].map(canonical_gid).ne("").all()
    add_check(
        checks,
        "manifest_gid_integrity",
        canonical_manifest and manifest["canonical_gid"].is_unique,
        f"rows={len(manifest)}; unique={manifest['canonical_gid'].nunique()}",
    )
    add_check(
        checks,
        "canonical_catalog_gid_coverage",
        catalog_ids.issubset(manifest_ids),
        f"mapped={len(catalog_ids & manifest_ids)}/{len(catalog_ids)}",
    )
    catalog_rows = int(
        pd.to_numeric(catalog["canonical_observation_rows"], errors="coerce").fillna(0).sum()
    )
    manifest_rows = int(
        pd.to_numeric(manifest["canonical_observation_rows"], errors="coerce").fillna(0).sum()
    )
    add_check(
        checks,
        "canonical_observation_row_conservation",
        catalog_rows == manifest_rows,
        f"catalog={catalog_rows}; manifest={manifest_rows}",
    )
    panel_certified = panel_evidence["certification_status"].eq("PASS").all()
    add_check(
        checks,
        "panel_kernel_certification",
        bool(panel_certified),
        f"panels={len(panel_evidence)}; failed="
        f"{panel_evidence.loc[~panel_evidence['certification_status'].eq('PASS'), 'panel_id'].tolist()}",
    )
    panel_input_failures = []
    for row in panel_evidence.to_dict("records"):
        panel_id = str(row["panel_id"])
        for path_column, hash_column, required in [
            ("sample_order_path", "sample_order_sha256", True),
            ("retained_marker_path", "retained_marker_sha256", False),
            ("genotype_matrix_path", "genotype_matrix_sha256", False),
        ]:
            path_text = str(row.get(path_column, "") or "").strip()
            expected = str(row.get(hash_column, "") or "").strip()
            if not path_text or path_text.lower() == "nan":
                if required:
                    panel_input_failures.append(f"{panel_id}:{path_column}:missing")
                continue
            path = Path(path_text)
            exists = path.is_file() and path.stat().st_size > 0
            observed = sha256_file(path) if exists else ""
            if not required and not expected and not exists:
                continue
            if not exists or not expected or observed != expected:
                panel_input_failures.append(f"{panel_id}:{path_column}")
    add_check(
        checks,
        "panel_evidence_input_identities",
        not panel_input_failures,
        f"failures={panel_input_failures}",
    )
    matrix_available = bool_series(panel_evidence["genotype_matrix_available"])
    matrix_certified = bool_series(panel_evidence["genotype_matrix_certified"])
    matrix_status = panel_evidence["genotype_matrix_certification_status"].astype(str)
    matrix_coherent = (~matrix_certified | (matrix_available & matrix_status.eq("aligned"))).all()
    add_check(
        checks,
        "genotype_matrix_certification_coherence",
        bool(matrix_coherent),
        f"certified={int(matrix_certified.sum())}; available={int(matrix_available.sum())}",
    )

    samples = panel_samples(panel_evidence)
    pedigree_ids = load_gid_set(
        Path(str(provenance["pedigree_order"])),
        ["sample_id", "panel_sample_id", "genotype_id"],
    )
    graph_paths = path_dictionary_ids(Path(str(provenance["path_dictionary"])))
    embeddings, _ = embedding_evidence(root)
    recomputed_manifest = build_gid_manifest(
        catalog=catalog,
        pedigree_ids=pedigree_ids,
        panel_evidence=panel_evidence,
        panel_samples=samples,
        graph_path_ids=graph_paths,
        embeddings=embeddings,
    )
    matched, detail = frame_matches(
        manifest, recomputed_manifest, sort_by=["canonical_gid"]
    )
    add_check(checks, "per_gid_classification_recomputed", matched, detail)

    expected_status, expected_panel = summary_tables(recomputed_manifest)
    matched, detail = frame_matches(
        status_summary,
        expected_status,
        sort_by=["regulatory_embedding_eligibility", "future_embedding_provenance_class"],
    )
    add_check(checks, "status_summary_recomputed", matched, detail)
    matched, detail = frame_matches(
        panel_summary, expected_panel, sort_by=["panel_id"]
    )
    add_check(checks, "panel_summary_recomputed", matched, detail)
    matched, detail = frame_matches(
        work_queue, projection_work_queue(panel_evidence), sort_by=["priority"]
    )
    add_check(checks, "projection_work_queue_recomputed", matched, detail)

    observed_equivalent = bool_series(manifest["observed_sequence_equivalent"])
    add_check(
        checks,
        "no_uncertified_observed_sequence",
        not observed_equivalent.any(),
        f"observed_sequence_equivalent={int(observed_equivalent.sum())}",
    )
    imputed = manifest["regulatory_embedding_eligibility"].eq(
        "pedigree_imputation_candidate"
    )
    imputed_contract = (
        manifest.loc[imputed, "future_embedding_provenance_class"].eq("imputed_pedigree").all()
        and manifest.loc[imputed, "confidence_gate_status"]
        .eq("required_not_evaluated")
        .all()
        and not observed_equivalent[imputed].any()
    )
    add_check(
        checks,
        "pedigree_imputation_contract",
        bool(imputed_contract),
        f"pedigree_imputation_candidates={int(imputed.sum())}",
    )
    direct = manifest["regulatory_embedding_eligibility"].eq(
        "eligible_direct_sequence_window_construction"
    )
    direct_contract = (
        manifest.loc[direct, "graph_projection_ready_panels"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .all()
        and manifest.loc[direct, "genotype_matrix_ready_panels"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .all()
    )
    add_check(
        checks,
        "direct_sequence_eligibility_contract",
        bool(direct_contract),
        f"direct_eligible={int(direct.sum())}",
    )
    checks_frame = pd.DataFrame(checks)
    status = "PASS" if checks_frame["status"].eq("PASS").all() else "FAIL"
    identities = {
        name: sha256_file(out_dir / name)
        for name in ARTIFACT_NAMES
        if (out_dir / name).is_file()
    }
    certification = {
        "status": status,
        "selection_data": "identifiers_and_certified_evidence_only",
        "phenotype_values_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "check_count": len(checks_frame),
        "failed_check_count": int(checks_frame["status"].eq("FAIL").sum()),
        "genotype_count": len(manifest),
        "canonical_trial_genotype_count": len(catalog_ids),
        "canonical_observation_rows": manifest_rows,
        "panel_count": len(panel_evidence),
        "artifact_sha256": identities,
    }
    return checks_frame, certification


def git_commit(code_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_checksum_manifest(root: Path, paths: list[Path], output: Path) -> None:
    lines = []
    for path in paths:
        try:
            label = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            label = str(path.resolve())
        lines.append(f"{sha256_file(path)}  {label}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the regulatory eligibility artifact contract."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("model_kernels/regulatory_eligibility_v1")
    )
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checksum-out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (root / args.out_dir).resolve()
    code_root = args.code_root.resolve()
    checks_path = out_dir / "regulatory_eligibility_validation.tsv"
    certification_path = out_dir / "regulatory_eligibility_certification.json"
    try:
        checks, certification = validate_artifacts(root, out_dir, code_root)
    except Exception as exc:
        checks = pd.DataFrame(
            [{"check": "fatal_validation_error", "status": "FAIL", "detail": repr(exc)}]
        )
        certification = {
            "status": "FAIL",
            "selection_data": "identifiers_and_certified_evidence_only",
            "phenotype_values_read": False,
            "outer_test_metrics_read": False,
            "final_holdout_outcomes_read": False,
            "check_count": 1,
            "failed_check_count": 1,
            "fatal_error": repr(exc),
        }
    certification["code_commit"] = git_commit(code_root)
    certification["validator_sha256"] = sha256_file(Path(__file__).resolve())
    checks.to_csv(checks_path, sep="\t", index=False)
    certification_path.write_text(
        json.dumps(certification, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(certification, indent=2))
    print("\n=== VALIDATION CHECKS ===")
    print(checks.to_string(index=False))
    if certification["status"] != "PASS":
        raise SystemExit("Regulatory eligibility certification failed")
    checksum_out = args.checksum_out
    if checksum_out is None:
        short = str(certification["code_commit"])[:8]
        checksum_out = root / "audit" / f"regulatory_eligibility_{short}.sha256"
    elif not checksum_out.is_absolute():
        checksum_out = (root / checksum_out).resolve()
    freeze_paths = [out_dir / name for name in ARTIFACT_NAMES] + [
        checks_path,
        certification_path,
    ]
    write_checksum_manifest(root, freeze_paths, checksum_out)
    print(f"Wrote checksum manifest: {checksum_out}")


if __name__ == "__main__":
    main()
