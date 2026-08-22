"""Evaluate exact, globally unique genotype-name to GID recovery evidence."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def norm_name(value: object) -> str:
    return re.sub(r"\s+", " ", "" if pd.isna(value) else str(value).strip()).upper()


def attach(frame: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    return frame.merge(
        registry,
        left_on=["trial_key", "cycle", "CID_normalized", "SID_normalized"],
        right_on=["trial_key", "cycle_norm", "CID_norm", "SID_norm"],
        how="left", validate="m:1", sort=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-ledger", type=Path, required=True)
    parser.add_argument("--genotype-registry", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    out = args.result_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    registry = pd.read_csv(args.genotype_registry, sep="\t", dtype=str, keep_default_na=False)
    keys = ["trial_key", "cycle_norm", "CID_norm", "SID_norm"]
    registry = registry[keys + ["accepted_gid"]]
    if registry.duplicated(keys).any():
        raise RuntimeError("Genotype registry key is not unique")
    parquet = pq.ParquetFile(args.raw_ledger.resolve())
    columns = [
        "trial_name", "trial_key", "cycle", "CID_normalized", "SID_normalized",
        "raw_gid", "genotype_name", "numeric_parse_pass", "numeric_value_finite",
    ]
    evidence: defaultdict[str, set[str]] = defaultdict(set)
    raw_labels: defaultdict[str, set[str]] = defaultdict(set)
    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
        frame = attach(batch.to_pandas(), registry)
        raw_gid = frame["raw_gid"].fillna("").astype(str).str.strip().str.upper()
        raw_gid = raw_gid.str.replace(r"^GID", "", regex=True).str.replace(r"\.0$", "", regex=True)
        raw_gid = raw_gid.where(raw_gid.str.fullmatch(r"[0-9]+", na=False), "")
        accepted = frame["accepted_gid"].fillna("").astype(str).str.strip()
        gid = raw_gid.where(raw_gid.ne(""), accepted)
        names = frame["genotype_name"].map(norm_name)
        selected = names.ne("") & gid.ne("")
        pairs = pd.DataFrame({"name": names[selected], "raw_name": frame.loc[selected, "genotype_name"], "gid": gid[selected]}).drop_duplicates()
        for row in pairs.itertuples(index=False):
            evidence[str(row.name)].add(str(row.gid))
            raw_labels[str(row.name)].add(str(row.raw_name))
    unique_map = {name: next(iter(gids)) for name, gids in evidence.items() if len(gids) == 1}
    conflicts = {name: gids for name, gids in evidence.items() if len(gids) > 1}
    rows = []
    total_unresolved_numeric = recovered_rows = 0
    recovered_trials: defaultdict[tuple[str, str], int] = defaultdict(int)
    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
        frame = attach(batch.to_pandas(), registry)
        raw_gid = frame["raw_gid"].fillna("").astype(str).str.strip().str.upper()
        raw_gid = raw_gid.str.replace(r"^GID", "", regex=True).str.replace(r"\.0$", "", regex=True)
        raw_gid = raw_gid.where(raw_gid.str.fullmatch(r"[0-9]+", na=False), "")
        accepted = frame["accepted_gid"].fillna("").astype(str).str.strip()
        unresolved = (
            frame["numeric_parse_pass"].fillna(False) & frame["numeric_value_finite"].fillna(False)
            & raw_gid.eq("") & accepted.eq("")
        )
        total_unresolved_numeric += int(unresolved.sum())
        names = frame["genotype_name"].map(norm_name)
        recovered_gid = names.map(unique_map).fillna("")
        recovered = unresolved & recovered_gid.ne("")
        recovered_rows += int(recovered.sum())
        candidate = pd.DataFrame({
            "trial_name": frame.loc[recovered, "trial_name"], "cycle": frame.loc[recovered, "cycle"],
            "CID_normalized": frame.loc[recovered, "CID_normalized"], "SID_normalized": frame.loc[recovered, "SID_normalized"],
            "genotype_name": frame.loc[recovered, "genotype_name"], "normalized_genotype_name": names[recovered],
            "candidate_gid": recovered_gid[recovered],
        }).groupby(
            ["trial_name", "cycle", "CID_normalized", "SID_normalized", "genotype_name", "normalized_genotype_name", "candidate_gid"],
            dropna=False,
        ).size().rename("numeric_rows").reset_index()
        rows.append(candidate)
        for row in candidate.groupby(["trial_name", "cycle"])["numeric_rows"].sum().items():
            recovered_trials[(str(row[0][0]), str(row[0][1]))] += int(row[1])
    candidates = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    candidates = candidates.groupby(list(candidates.columns[:-1]), dropna=False)["numeric_rows"].sum().reset_index() if not candidates.empty else candidates
    candidates.to_csv(out / "exact_unique_name_gid_recovery_candidates.tsv", sep="\t", index=False)
    conflict_rows = [
        {"normalized_genotype_name": name, "candidate_gids": ";".join(sorted(gids)), "gid_count": len(gids),
         "raw_labels": ";".join(sorted(raw_labels[name]))}
        for name, gids in conflicts.items()
    ]
    pd.DataFrame(conflict_rows).to_csv(out / "ambiguous_name_gid_evidence.tsv", sep="\t", index=False)
    summary = {
        "status": "PASS_EXACT_NAME_RECOVERY_AUDIT",
        "authoritative_name_keys": len(evidence), "unique_name_gid_keys": len(unique_map),
        "ambiguous_name_gid_keys": len(conflicts), "unresolved_numeric_rows_before": total_unresolved_numeric,
        "numeric_rows_recoverable_by_exact_unique_name": recovered_rows,
        "unresolved_numeric_rows_after_candidate_recovery": total_unresolved_numeric - recovered_rows,
        "trial_cycles_with_candidate_recoveries": len(recovered_trials),
    }
    (out / "exact_name_recovery_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
