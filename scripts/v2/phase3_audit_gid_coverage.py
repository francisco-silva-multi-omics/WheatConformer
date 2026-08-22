"""Audit numeric-phenotype GID coverage before the expensive layer rebuild."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-ledger", type=Path, required=True)
    parser.add_argument("--genotype-registry", type=Path, required=True)
    parser.add_argument("--raw-trial-registry", type=Path)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)

    registry = pd.read_csv(args.genotype_registry, sep="\t", dtype=str, keep_default_na=False)
    keys = ["trial_key", "cycle_norm", "CID_norm", "SID_norm"]
    if registry.duplicated(keys).any():
        raise RuntimeError("Genotype registry key is not unique")
    registry = registry[keys + ["accepted_gid", "registry_decision"]]

    trial_counts: defaultdict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    unresolved_counts: defaultdict[tuple[str, ...], int] = defaultdict(int)
    total = numeric = with_gid = raw_registry_conflicts = 0
    parquet = pq.ParquetFile(args.raw_ledger.resolve())
    columns = [
        "trial_name", "trial_key", "cycle", "CID_normalized", "SID_normalized",
        "raw_gid", "genotype_name", "numeric_parse_pass", "numeric_value_finite",
    ]
    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
        frame = batch.to_pandas()
        frame = frame.merge(
            registry,
            left_on=["trial_key", "cycle", "CID_normalized", "SID_normalized"],
            right_on=keys,
            how="left", validate="m:1", sort=False,
        )
        raw_gid = frame["raw_gid"].fillna("").astype(str).str.strip().str.upper()
        raw_gid = raw_gid.str.replace(r"^GID", "", regex=True).str.replace(r"\.0$", "", regex=True)
        raw_gid = raw_gid.where(raw_gid.str.fullmatch(r"[0-9]+", na=False), "")
        accepted = frame["accepted_gid"].fillna("").astype(str).str.strip()
        conflict = raw_gid.ne("") & accepted.ne("") & raw_gid.ne(accepted)
        resolved = (~conflict) & (raw_gid.ne("") | accepted.ne(""))
        numeric_mask = frame["numeric_parse_pass"].fillna(False) & frame["numeric_value_finite"].fillna(False)
        total += len(frame)
        numeric += int(numeric_mask.sum())
        with_gid += int((numeric_mask & resolved).sum())
        raw_registry_conflicts += int((numeric_mask & conflict).sum())
        compact = pd.DataFrame({
            "trial_name": frame["trial_name"], "trial_key": frame["trial_key"], "cycle": frame["cycle"],
            "numeric": numeric_mask.astype("int64"),
            "with_gid": (numeric_mask & resolved).astype("int64"),
            "without_gid": (numeric_mask & ~resolved).astype("int64"),
        }).groupby(["trial_name", "trial_key", "cycle"], dropna=False).sum().reset_index()
        for row in compact.itertuples(index=False):
            value = trial_counts[(str(row.trial_name), str(row.trial_key), str(row.cycle))]
            value[0] += int(row.numeric); value[1] += int(row.with_gid); value[2] += int(row.without_gid)
        unresolved = frame[numeric_mask & ~resolved][
            ["trial_name", "trial_key", "cycle", "CID_normalized", "SID_normalized", "genotype_name", "registry_decision"]
        ].fillna("")
        grouped = unresolved.groupby(list(unresolved.columns), dropna=False).size()
        for key, count in grouped.items():
            unresolved_counts[tuple(str(value) for value in key)] += int(count)

    trials = []
    for key, counts in sorted(trial_counts.items()):
        status = (
            "FAIL_TRIAL_HAS_NO_MATCHING_GID" if counts[0] > 0 and counts[1] == 0
            else "PASS_ALL_NUMERIC_ROWS_HAVE_GID" if counts[0] > 0 and counts[2] == 0
            else "PARTIAL_SOME_NUMERIC_ROWS_WITHOUT_GID" if counts[0] > 0
            else "NOT_APPLICABLE_NO_NUMERIC_PHENOTYPES"
        )
        trials.append({
            "trial_name": key[0], "trial_key": key[1], "cycle": key[2],
            "numeric_rows": counts[0], "numeric_rows_with_gid": counts[1],
            "numeric_rows_without_gid": counts[2], "gid_coverage_status": status,
        })
    trial_frame = pd.DataFrame(trials)
    trial_frame.to_csv(result_dir / "trial_gid_coverage.tsv", sep="\t", index=False)
    canonical_trial_failures = None
    if args.raw_trial_registry:
        raw_trials = pd.read_csv(args.raw_trial_registry, sep="\t", dtype=str, keep_default_na=False)
        alias = raw_trials[["trial_key", "cycle", "trial_code"]].drop_duplicates()
        trial_with_alias = trial_frame.merge(alias, on=["trial_key", "cycle"], how="left", validate="1:1")
        trial_with_alias["canonical_trial_code"] = trial_with_alias["trial_code"].fillna("").where(
            trial_with_alias["trial_code"].fillna("").ne(""), trial_with_alias["trial_key"]
        )
        canonical_trial = trial_with_alias.groupby(
            ["canonical_trial_code", "cycle"], dropna=False, sort=True
        )[["numeric_rows", "numeric_rows_with_gid", "numeric_rows_without_gid"]].sum().reset_index()
        canonical_trial["gid_coverage_status"] = "PARTIAL_SOME_NUMERIC_ROWS_WITHOUT_GID"
        canonical_trial.loc[canonical_trial["numeric_rows_with_gid"].eq(0), "gid_coverage_status"] = "FAIL_CANONICAL_TRIAL_HAS_NO_MATCHING_GID"
        canonical_trial.loc[canonical_trial["numeric_rows_without_gid"].eq(0), "gid_coverage_status"] = "PASS_ALL_NUMERIC_ROWS_HAVE_GID"
        canonical_trial.to_csv(result_dir / "canonical_trial_gid_coverage.tsv", sep="\t", index=False)
        canonical_trial_failures = int(canonical_trial["gid_coverage_status"].eq("FAIL_CANONICAL_TRIAL_HAS_NO_MATCHING_GID").sum())
    unresolved_columns = [
        "trial_name", "trial_key", "cycle", "CID_normalized", "SID_normalized",
        "genotype_name", "registry_decision", "numeric_rows",
    ]
    unresolved_rows = [dict(zip(unresolved_columns, key + (count,), strict=True)) for key, count in unresolved_counts.items()]
    pd.DataFrame(unresolved_rows, columns=unresolved_columns).sort_values(
        "numeric_rows", ascending=False
    ).to_csv(result_dir / "unresolved_numeric_identity_keys.tsv", sep="\t", index=False)
    summary = {
        "status": "PASS_COVERAGE_AUDIT_EXECUTED",
        "raw_rows": total, "numeric_rows": numeric, "numeric_rows_with_gid": with_gid,
        "numeric_rows_without_gid": numeric - with_gid,
        "numeric_raw_registry_conflicts": raw_registry_conflicts,
        "trial_cycles": len(trial_frame),
        "trial_cycles_with_no_matching_gid": int(trial_frame["gid_coverage_status"].eq("FAIL_TRIAL_HAS_NO_MATCHING_GID").sum()),
        "trial_cycles_with_partial_gid_coverage": int(trial_frame["gid_coverage_status"].str.startswith("PARTIAL").sum()),
        "canonical_trial_cycles_with_no_matching_gid": canonical_trial_failures,
        "unresolved_identity_keys": len(unresolved_counts),
    }
    (result_dir / "gid_coverage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
