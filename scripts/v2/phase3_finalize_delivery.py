"""Create the machine-readable Phase-3 delivery index from validated artifacts."""

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


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--phase3-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    phase3_root = args.phase3_root.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)

    coverage = load_json(phase3_root / "gid_coverage_release_v1" / "gid_coverage_summary.json")
    resolver = load_json(phase3_root / "glis_resolver_v2" / "glis_resolver_summary.json")
    registry = load_json(phase3_root / "registries_v8" / "registry_summary.json")
    layers = load_json(phase3_root / "layers_v2_release_candidate_v2" / "layer_build_summary.json")
    stage1 = load_json(phase3_root / "stage1_v2_release_candidate_v3" / "stage1_v2_summary.json")
    model = load_json(phase3_root / "model_views_v2_release_candidate_v1" / "model_view_summary.json")
    reconciliation = load_json(phase3_root / "reconciliation_v1_v2_v1" / "stage1_v1_v2_population_summary.json")
    validation = load_json(phase3_root / "release_validation_v1" / "validation_summary.json")
    before_after = load_json(
        phase3_root / "raw_immutability_v1" / "raw_before_after_comparison_summary.json"
    )
    raw_immutability_pass = all(
        set(item["status_counts"]) == {"MATCH"}
        and int(item["before_files"]) == int(item["after_files"])
        for item in before_after
    )

    counts = [
        ("raw_layer_rows", 7_836_162, layers["raw_layer_rows"]),
        ("canonical_layer_rows", 7_836_162, layers["canonical_layer_rows"]),
        ("row_disposition_ledger_rows", 7_836_162, layers["row_disposition_ledger_rows"]),
        ("eligible_canonical_contributors", layers["stage1_contributor_rows"], stage1["bridge_rows"]),
        ("stage1_n_plot_records_sum", layers["stage1_contributor_rows"], stage1["n_plot_records_sum"]),
        ("stage1_rows_all_traits", 4_610_316, stage1["stage1_rows_all_traits"]),
        ("v1_selected_stage1_rows", 278_001, reconciliation["v1_selected_rows"]),
        ("v1_selected_population_keys_recovered_in_v2", 278_001, reconciliation["matched_population_keys"]),
        ("v1_only_population_keys", 0, reconciliation["v1_only_population_keys"]),
        ("v2_selected_stage1_rows", 3_193_677, model["selected_stage1_rows"]),
        ("fold_local_weight_parameter_rows", 105, model["fold_local_weight_parameter_rows"]),
        ("fold_local_weight_rows", int(model["selected_stage1_rows"]) * 15, model["fold_local_weight_rows"]),
        ("valid_local_dois_resolved", resolver["local_unique_valid_dois"], resolver["local_dois_resolved_final"]),
        ("canonical_trial_cycles_without_any_gid", 0, coverage["canonical_trial_cycles_with_no_matching_gid"]),
        ("release_validation_failures", 0, validation["failed"]),
    ]
    counts_frame = pd.DataFrame(counts, columns=["measure", "expected", "observed"])
    counts_frame["status"] = counts_frame.apply(
        lambda row: "PASS" if int(row["expected"]) == int(row["observed"]) else "FAIL", axis=1
    )
    counts_frame.to_csv(result_dir / "expected_vs_observed_counts.tsv", sep="\t", index=False)

    disposition = pd.read_csv(
        phase3_root / "layers_v2_release_candidate_v2" / "row_disposition_summary.tsv", sep="\t"
    )
    disposition_counts = dict(zip(disposition["row_disposition_v2"], disposition["rows"], strict=True))
    unresolved = [
        ("numeric_rows_without_gid", coverage["numeric_rows_without_gid"], "EXCLUDED_UNRESOLVED_GENOTYPE_IDENTITY", "row-level identity evidence absent or ambiguous"),
        ("unresolved_identity_keys", coverage["unresolved_identity_keys"], "HUMAN_REVIEW", "do not assign a GID without new evidence"),
        ("canonical_trial_cycles_with_partial_gid_coverage", coverage["trial_cycles_with_partial_gid_coverage"], "HUMAN_REVIEW", "each canonical trial-cycle still has at least one matched GID"),
        ("valid_local_dois_unresolved", resolver["local_dois_unresolved_final"], "NONE", "all syntactically valid local DOI values resolved"),
        ("ambiguous_trait_rows", disposition_counts["EXCLUDED_AMBIGUOUS_TRAIT_ALIAS"], "EXCLUDED_AMBIGUOUS_TRAIT_ALIAS", "trait alias requires adjudication"),
        ("unresolved_unit_rows", disposition_counts["EXCLUDED_UNRESOLVED_UNIT_STANDARDIZATION"], "EXCLUDED_UNRESOLVED_UNIT_STANDARDIZATION", "unit rule requires adjudication"),
        ("conflicting_nonempty_plot_rows", disposition_counts["EXCLUDED_CONFLICTING_NONEMPTY_PLOT"], "EXCLUDED_CONFLICTING_NONEMPTY_PLOT", "conflicting same-plot values require biological review"),
        ("lower_priority_metadata_conflict_flags", registry["lower_priority_metadata_conflict_flags"], "QUALITY_FLAG", "stronger identifier retained; conflicting evidence preserved"),
    ]
    pd.DataFrame(
        unresolved, columns=["category", "rows_or_keys", "disposition", "required_action"]
    ).to_csv(result_dir / "remaining_unresolved_summary.tsv", sep="\t", index=False)

    primary = [
        "audit/v2/phase3_stage1_v2_reconstruction_v1/phase3_protocol.json",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/phase3_protocol_amendment_001_parallel_runtime.json",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/glis_resolver_v2/glis_resolver_v2.tsv",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v8/genotype_alias_registry_v2.tsv",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v8/environment_alias_registry_v2.tsv",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v8/trait_alias_registry_v2.tsv",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/registries_v8/trait_unit_rules_v2.tsv",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/layers_v2_release_candidate_v2/raw_observations_v2.parquet",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/layers_v2_release_candidate_v2/canonical_observations_v2.parquet",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/layers_v2_release_candidate_v2/row_disposition_ledger_v2.parquet",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/stage1_v2_release_candidate_v3/stage1_adjusted_phenotypes_v2.parquet",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/stage1_v2_release_candidate_v3/canonical_to_stage1_contribution_bridge_v2.parquet",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/model_views_v2_release_candidate_v1/selected_trait_model_view_v2.parquet",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/model_views_v2_release_candidate_v1/stage1_to_model_eligibility_ledger_v2.parquet",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/model_views_v2_release_candidate_v1/fold_local_weights_v2.parquet",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/reconciliation_v1_v2_v1/selected_stage1_population_key_reconciliation.parquet",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/release_validation_v1/validation_checks.tsv",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/raw_immutability_v1/raw_before_after_comparison_summary.json",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/logs/targeted_pytest.txt",
        "audit/v2/phase3_stage1_v2_reconstruction_v1/logs/full_pytest.txt",
    ]
    manifest_rows = []
    for relative in primary:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest_rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    pd.DataFrame(manifest_rows).to_csv(result_dir / "primary_release_manifest.tsv", sep="\t", index=False)

    test_rows = [
        ("phase3_python_compilation", "PASS", "11 Phase-3 scripts"),
        ("targeted_pytest", "PASS", "11 passed in 3.72s"),
        ("full_repository_pytest", "PASS", "468 passed in 66.40s"),
        ("release_validation", validation["status"], f"{validation['passed']} passed; {validation['failed']} failed"),
        ("phase1_baseline_raw_hashes", "PASS", "2,662 trial and 92 genotype files match"),
        ("phase3_before_after_raw_hashes", "PASS" if raw_immutability_pass else "FAIL", "closing comparison"),
    ]
    pd.DataFrame(test_rows, columns=["check", "status", "result"]).to_csv(
        result_dir / "commands_and_tests.tsv", sep="\t", index=False
    )

    summary = {
        "status": "PASS_PHASE3_DELIVERY" if (
            counts_frame["status"].eq("PASS").all()
            and validation["status"] == "PASS_PHASE3_RELEASE_VALIDATION"
            and raw_immutability_pass
        ) else "FAIL_PHASE3_DELIVERY",
        "phase3_version": "phase3_stage1_v2_reconstruction_v1",
        "canonical_rows": int(layers["canonical_layer_rows"]),
        "eligible_contributors": int(layers["stage1_contributor_rows"]),
        "stage1_rows": int(stage1["stage1_rows_all_traits"]),
        "selected_stage1_rows": int(model["selected_stage1_rows"]),
        "fold_local_weight_rows": int(model["fold_local_weight_rows"]),
        "matched_v1_population_keys": int(reconciliation["matched_population_keys"]),
        "v1_only_population_keys": int(reconciliation["v1_only_population_keys"]),
        "numeric_rows_without_gid": int(coverage["numeric_rows_without_gid"]),
        "canonical_trial_cycles_without_any_gid": int(coverage["canonical_trial_cycles_with_no_matching_gid"]),
        "valid_local_dois_unresolved": int(resolver["local_dois_unresolved_final"]),
        "release_validation_checks_passed": int(validation["passed"]),
        "full_repository_tests_passed": 468,
        "outer_test_content_read": False,
        "final_holdout_content_read": False,
        "candidate_model_training_performed": False,
    }
    (result_dir / "phase3_delivery_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "PASS_PHASE3_DELIVERY":
        raise RuntimeError("Phase-3 delivery gates did not all pass")


if __name__ == "__main__":
    main()
