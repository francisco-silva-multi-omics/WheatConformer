"""Validate a complete, versioned Stage-1 v2 reconstruction release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--disposition-ledger", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--model-view", type=Path, required=True)
    parser.add_argument("--fold-weights", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    repository_root = args.repository_root.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    def check(name: str, observed: object, expected: object, passed: bool) -> None:
        checks.append({
            "check": name,
            "observed": str(observed),
            "expected": str(expected),
            "status": "PASS" if passed else "FAIL",
        })

    for binding in protocol["input_bindings"]:
        path = repository_root / binding["path"]
        observed = sha256(path) if path.is_file() else "MISSING"
        expected = binding["sha256"]
        check(f"protected_input_sha256::{binding['path']}", observed, expected, observed == expected)

    con = duckdb.connect(str(result_dir / "validation.duckdb"))
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA memory_limit='2GB'")
    canonical = str(args.canonical.resolve()).replace("'", "''")
    disposition = str(args.disposition_ledger.resolve()).replace("'", "''")
    stage1 = str(args.stage1.resolve()).replace("'", "''")
    bridge = str(args.bridge.resolve()).replace("'", "''")
    model = str(args.model_view.resolve()).replace("'", "''")
    weights = str(args.fold_weights.resolve()).replace("'", "''")

    canonical_stats = con.execute(f"""
        SELECT count(*) AS n_rows,
               count(DISTINCT canonical_row_id) AS unique_ids,
               sum((row_disposition_v2='ELIGIBLE_STAGE1_CONTRIBUTOR')::BIGINT) AS eligible,
               sum((row_disposition_v2='ELIGIBLE_STAGE1_CONTRIBUTOR' AND
                    nullif(trim(cast(resolved_gid_v2 AS VARCHAR)), '') IS NULL)::BIGINT) AS eligible_missing_gid,
               sum((row_disposition_v2='ELIGIBLE_STAGE1_CONTRIBUTOR' AND
                    nullif(trim(canonical_environment_id), '') IS NULL)::BIGINT) AS eligible_missing_environment,
               sum((row_disposition_v2='ELIGIBLE_STAGE1_CONTRIBUTOR' AND
                    nullif(trim(accepted_canonical_trait), '') IS NULL)::BIGINT) AS eligible_missing_trait
        FROM read_parquet('{canonical}')
    """).fetchone()
    disposition_stats = con.execute(f"""
        SELECT count(*), count(DISTINCT canonical_row_id)
        FROM read_parquet('{disposition}')
    """).fetchone()
    check("canonical_row_count", canonical_stats[0], 7_836_162, canonical_stats[0] == 7_836_162)
    check("canonical_row_id_injective", canonical_stats[1], canonical_stats[0], canonical_stats[1] == canonical_stats[0])
    check("disposition_row_count", disposition_stats[0], canonical_stats[0], disposition_stats[0] == canonical_stats[0])
    check("disposition_row_id_injective", disposition_stats[1], disposition_stats[0], disposition_stats[1] == disposition_stats[0])
    check("eligible_missing_gid", canonical_stats[3], 0, canonical_stats[3] == 0)
    check("eligible_missing_environment", canonical_stats[4], 0, canonical_stats[4] == 0)
    check("eligible_missing_trait", canonical_stats[5], 0, canonical_stats[5] == 0)

    stage1_stats = con.execute(f"""
        SELECT count(*) AS n_rows, count(DISTINCT stage1_v2_row_id) AS unique_ids,
               sum(n_plot_records) AS contributor_sum,
               count(DISTINCT resolved_gid) AS genotypes,
               count(DISTINCT canonical_environment_id) AS environments
        FROM read_parquet('{stage1}')
    """).fetchone()
    bridge_stats = con.execute(f"""
        SELECT count(*) AS n_rows, count(DISTINCT canonical_row_id) AS unique_canonical_ids,
               count(*) FILTER (WHERE contribution_status!='CONTRIBUTED_TO_STAGE1_V2') AS invalid_status,
               count(*) FILTER (WHERE stage1_v2_row_id IS NULL OR trim(stage1_v2_row_id)='') AS missing_stage1_id
        FROM read_parquet('{bridge}')
    """).fetchone()
    bridge_orphans = con.execute(f"""
        SELECT count(*) FROM read_parquet('{bridge}') b
        LEFT JOIN read_parquet('{stage1}') s USING (stage1_v2_row_id)
        WHERE s.stage1_v2_row_id IS NULL
    """).fetchone()[0]
    check("stage1_row_id_unique", stage1_stats[1], stage1_stats[0], stage1_stats[1] == stage1_stats[0])
    check("bridge_rows_equal_eligible_contributors", bridge_stats[0], canonical_stats[2], bridge_stats[0] == canonical_stats[2])
    check("bridge_canonical_ids_unique", bridge_stats[1], bridge_stats[0], bridge_stats[1] == bridge_stats[0])
    check("stage1_n_plot_records_reconciles", stage1_stats[2], canonical_stats[2], stage1_stats[2] == canonical_stats[2])
    check("bridge_contribution_status", bridge_stats[2], 0, bridge_stats[2] == 0)
    check("bridge_missing_stage1_id", bridge_stats[3], 0, bridge_stats[3] == 0)
    check("bridge_stage1_orphans", bridge_orphans, 0, bridge_orphans == 0)

    model_stats = con.execute(f"""
        SELECT count(*) AS n_rows, count(DISTINCT stage1_v2_row_id) AS unique_ids,
               count(DISTINCT accepted_canonical_trait) AS traits,
               min(genotype_fold), max(genotype_fold),
               min(environment_fold), max(environment_fold), min(pair_fold), max(pair_fold),
               sum(protected_outer_or_final_membership_used::INT) AS protected_used
        FROM read_parquet('{model}')
    """).fetchone()
    weight_stats = con.execute(f"""
        SELECT count(*) AS n_rows,
               count(DISTINCT scenario) AS scenarios,
               count(DISTINCT fold) AS folds,
               count(DISTINCT stage1_v2_row_id || '|' || scenario || '|' || cast(fold AS VARCHAR)) AS unique_keys,
               sum((NOT isfinite(fold_local_weight) OR fold_local_weight<=0)::BIGINT) AS invalid_weights
        FROM read_parquet('{weights}')
    """).fetchone()
    expected_weights = model_stats[0] * 3 * 5
    train_weight_deviation = con.execute(f"""
        SELECT coalesce(max(abs(mean_weight - 1.0)), 0.0)
        FROM (
          SELECT scenario, fold, accepted_canonical_trait, avg(fold_local_weight) AS mean_weight
          FROM read_parquet('{weights}') WHERE membership='TRAINING'
          GROUP BY scenario, fold, accepted_canonical_trait
        )
    """).fetchone()[0]
    check("model_view_row_id_unique", model_stats[1], model_stats[0], model_stats[1] == model_stats[0])
    check("model_view_selected_trait_count", model_stats[2], 7, model_stats[2] == 7)
    check("genotype_fold_range", f"{model_stats[3]}..{model_stats[4]}", "0..4", model_stats[3] == 0 and model_stats[4] == 4)
    check("environment_fold_range", f"{model_stats[5]}..{model_stats[6]}", "0..4", model_stats[5] == 0 and model_stats[6] == 4)
    check("pair_fold_range", f"{model_stats[7]}..{model_stats[8]}", "0..4", model_stats[7] == 0 and model_stats[8] == 4)
    check("protected_membership_used", model_stats[9], 0, model_stats[9] == 0)
    check("fold_weight_row_completeness", weight_stats[0], expected_weights, weight_stats[0] == expected_weights)
    check("fold_weight_unique_keys", weight_stats[3], weight_stats[0], weight_stats[3] == weight_stats[0])
    check("fold_weight_scenarios", weight_stats[1], 3, weight_stats[1] == 3)
    check("fold_weight_folds", weight_stats[2], 5, weight_stats[2] == 5)
    check("fold_weight_positive_finite", weight_stats[4], 0, weight_stats[4] == 0)
    check("training_fold_weight_mean", train_weight_deviation, "<=1e-10", train_weight_deviation <= 1e-10)
    con.close()

    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(result_dir / "validation_checks.tsv", sep="\t", index=False)
    failures = checks_frame[checks_frame["status"].eq("FAIL")]
    summary = {
        "status": "PASS_PHASE3_RELEASE_VALIDATION" if failures.empty else "FAIL_PHASE3_RELEASE_VALIDATION",
        "checks": len(checks_frame),
        "passed": int(checks_frame["status"].eq("PASS").sum()),
        "failed": len(failures),
        "canonical_rows": int(canonical_stats[0]),
        "eligible_canonical_contributors": int(canonical_stats[2]),
        "stage1_rows": int(stage1_stats[0]),
        "stage1_genotypes": int(stage1_stats[3]),
        "stage1_environments": int(stage1_stats[4]),
        "selected_model_view_rows": int(model_stats[0]),
        "fold_local_weight_rows": int(weight_stats[0]),
        "outer_test_content_read": False,
        "final_holdout_content_read": False,
    }
    (result_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not failures.empty:
        raise RuntimeError(f"Phase-3 release validation failed {len(failures)} checks")


if __name__ == "__main__":
    main()
