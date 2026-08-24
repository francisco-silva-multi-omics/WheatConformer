"""Independent closing validator for the Phase-4 release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd


EXPECTED_PLOTS = 4_226_848
EXPECTED_ENTRIES = 3_193_677
EXPECTED_GROUPS = 37_206


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    required = [
        "adjusted_phenotypes_v1.parquet",
        "plot_design_reconstruction_v1.parquet",
        "reliability_pev_v1.parquet",
        "plot_model_diagnostics_v1.parquet",
        "trial_trait_spatial_model_selection_report.tsv",
        "candidate_model_comparison.tsv",
        "ranking_ceiling_estimates.tsv",
        "unreliable_environment_trait_groups.tsv",
        "before_after_counts.tsv",
        "phase4_summary.json",
        "input_freeze_manifest.tsv",
    ]
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, observed: object, expected: object) -> None:
        checks.append({
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "expected": expected,
        })

    missing = [name for name in required if not (root / name).is_file()]
    check("required_files_present", not missing, ";".join(missing) or "NONE", "NONE")
    if missing:
        raise SystemExit(f"Missing Phase-4 release files: {missing}")

    con = duckdb.connect()
    entries = str(root / "adjusted_phenotypes_v1.parquet")
    plots = str(root / "plot_design_reconstruction_v1.parquet")
    reliability = str(root / "reliability_pev_v1.parquet")
    diagnostics = str(root / "plot_model_diagnostics_v1.parquet")
    groups = str(root / "trial_trait_spatial_model_selection_report.tsv")
    models = str(root / "candidate_model_comparison.tsv")
    ceiling = str(root / "ranking_ceiling_estimates.tsv")
    unreliable = str(root / "unreliable_environment_trait_groups.tsv")

    entry_stats = con.execute(
        "SELECT count(*), count(DISTINCT phase4_entry_id), count(DISTINCT phase4_group_id) FROM read_parquet(?)",
        [entries],
    ).fetchone()
    check("adjusted_entry_rows", entry_stats[0] == EXPECTED_ENTRIES, entry_stats[0], EXPECTED_ENTRIES)
    check("adjusted_entry_ids_unique", entry_stats[1] == EXPECTED_ENTRIES, entry_stats[1], EXPECTED_ENTRIES)
    check("adjusted_entry_groups", entry_stats[2] == EXPECTED_GROUPS, entry_stats[2], EXPECTED_GROUPS)

    plot_stats = con.execute(
        "SELECT count(*), count(DISTINCT raw_source_row_id), sum(outlier_excluded::INTEGER), "
        "sum((field_row_status='SOURCE_NOT_PROVIDED')::INTEGER), "
        "sum((field_column_status='SOURCE_NOT_PROVIDED')::INTEGER) FROM read_parquet(?)",
        [plots],
    ).fetchone()
    check("plot_rows", plot_stats[0] == EXPECTED_PLOTS, plot_stats[0], EXPECTED_PLOTS)
    check("plot_source_ids_unique", plot_stats[1] == EXPECTED_PLOTS, plot_stats[1], EXPECTED_PLOTS)
    check("outlier_exclusions_zero", plot_stats[2] == 0, plot_stats[2], 0)
    check("field_row_unavailable_explicit", plot_stats[3] == EXPECTED_PLOTS, plot_stats[3], EXPECTED_PLOTS)
    check("field_column_unavailable_explicit", plot_stats[4] == EXPECTED_PLOTS, plot_stats[4], EXPECTED_PLOTS)

    reliability_rows = con.execute("SELECT count(*) FROM read_parquet(?)", [reliability]).fetchone()[0]
    diagnostic_rows = con.execute("SELECT count(*) FROM read_parquet(?)", [diagnostics]).fetchone()[0]
    check("reliability_rows", reliability_rows == EXPECTED_ENTRIES, reliability_rows, EXPECTED_ENTRIES)
    check("diagnostic_rows", diagnostic_rows == EXPECTED_PLOTS, diagnostic_rows, EXPECTED_PLOTS)

    group_stats = con.execute(
        "SELECT count(*), count(DISTINCT phase4_group_id), "
        "sum((ar1_by_ar1_status='NOT_IDENTIFIABLE_NO_INDEPENDENT_ROW_COLUMN_COORDINATES')::INTEGER) "
        "FROM read_csv_auto(?, delim='\\t', header=true)", [groups]
    ).fetchone()
    check("group_report_rows_unique", group_stats[0] == EXPECTED_GROUPS and group_stats[1] == EXPECTED_GROUPS,
          f"{group_stats[0]}/{group_stats[1]}", f"{EXPECTED_GROUPS}/{EXPECTED_GROUPS}")
    check("ar1xar1_nonidentifiability_explicit", group_stats[2] == EXPECTED_GROUPS, group_stats[2], EXPECTED_GROUPS)

    model_stats = con.execute(
        "SELECT count(DISTINCT phase4_group_id), "
        "count(DISTINCT phase4_group_id) FILTER (WHERE selected_model=true) "
        "FROM read_csv_auto(?, delim='\\t', header=true)", [models]
    ).fetchone()
    check("candidate_models_cover_all_groups", model_stats[0] == EXPECTED_GROUPS, model_stats[0], EXPECTED_GROUPS)
    check("one_selected_model_per_group", model_stats[1] == EXPECTED_GROUPS, model_stats[1], EXPECTED_GROUPS)

    ceiling_rows = con.execute(
        "SELECT count(*), count(DISTINCT phase4_group_id) FROM read_csv_auto(?, delim='\\t', header=true)",
        [ceiling],
    ).fetchone()
    check("ranking_ceiling_group_coverage", ceiling_rows == (EXPECTED_GROUPS, EXPECTED_GROUPS),
          f"{ceiling_rows[0]}/{ceiling_rows[1]}", f"{EXPECTED_GROUPS}/{EXPECTED_GROUPS}")
    unreliable_rows = con.execute(
        "SELECT count(*) FROM read_csv_auto(?, delim='\\t', header=true)", [unreliable]
    ).fetchone()[0]
    summary = json.loads((root / "phase4_summary.json").read_text(encoding="utf-8"))
    check("unreliable_count_matches_summary", unreliable_rows == summary["groups_too_unreliable_for_ranking_claims"],
          unreliable_rows, summary["groups_too_unreliable_for_ranking_claims"])
    check("release_summary_pass", summary.get("status") == "PASS_PHASE4_RECONSTRUCTION_AND_SIGNAL_ASSESSMENT",
          summary.get("status"), "PASS_PHASE4_RECONSTRUCTION_AND_SIGNAL_ASSESSMENT")
    con.close()

    freeze = pd.read_csv(root / "input_freeze_manifest.tsv", sep="\t", dtype=str)
    mismatches = []
    for row in freeze.itertuples(index=False):
        path = Path(row.path)
        observed = sha256(path) if path.is_file() else "MISSING"
        if observed != row.sha256:
            mismatches.append(str(path))
    check("frozen_inputs_unchanged", not mismatches, ";".join(mismatches) or "NONE", "NONE")

    validation = pd.DataFrame(checks)
    validation.to_csv(root / "validation_checks.tsv", sep="\t", index=False)
    failed = validation[validation["status"].eq("FAIL")]
    validation_summary = {
        "status": "PASS" if failed.empty else "FAIL",
        "checks": len(validation),
        "passed": int(validation["status"].eq("PASS").sum()),
        "failed": int(len(failed)),
        "failed_checks": failed["check"].tolist(),
        "manifest_scope": "all top-level release files except output_manifest.tsv itself",
    }
    (root / "validation_summary.json").write_text(
        json.dumps(validation_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not failed.empty:
        raise SystemExit(f"Phase-4 validation failed: {failed['check'].tolist()}")

    manifest_rows = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "output_manifest.tsv":
            manifest_rows.append({
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    pd.DataFrame(manifest_rows).to_csv(root / "output_manifest.tsv", sep="\t", index=False)
    print(json.dumps(validation_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
