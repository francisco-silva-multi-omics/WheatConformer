from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


MISSING = {"", "NA", "NAN", "NONE", "NULL", ".", "-", "UNKNOWN", "0"}


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).strip())
    return "" if text.upper() in MISSING else text


def canonical_gid(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = re.sub(r"\.0$", "", text)
    if re.fullmatch(r"\d+", text):
        return f"GID{text}"
    if re.fullmatch(r"GID\d+", text, flags=re.IGNORECASE):
        return "GID" + re.search(r"\d+", text).group(0)
    return text


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a trial-derived pedigree table from the resolved genotype manifest. "
            "The output is intended as input to build_pedigree_kernel.py."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata_outputs/all_trials_genotype_manifest_resolved.tsv"),
    )
    parser.add_argument("--out-table", type=Path, default=Path("genotype_panels/pedigree/trial_derived_pedigree_table.tsv"))
    parser.add_argument("--out-qc", type=Path, default=Path("genotype_panels/pedigree/trial_derived_pedigree_qc.tsv"))
    parser.add_argument("--id-col", help="Preferred child/sample ID column. Defaults to panel_sample_id_expected or resolved_gid.")
    parser.add_argument("--cross-col", help="Preferred cross/pedigree column. Defaults to cross_name.")
    parser.add_argument("--history-col", help="Preferred selection-history column. Defaults to selection_history.")
    args = parser.parse_args()

    df = read_table(args.manifest)
    id_col = args.id_col or first_existing(df.columns.tolist(), ["panel_sample_id_expected", "resolved_gid", "fieldbook_gid", "GID", "gid"])
    cross_col = args.cross_col or first_existing(df.columns.tolist(), ["cross_name", "cross", "pedigree", "designation"])
    history_col = args.history_col or first_existing(df.columns.tolist(), ["selection_history", "history"])
    if not id_col:
        raise SystemExit(f"Could not detect an ID column in {args.manifest}")
    if not cross_col:
        raise SystemExit(f"Could not detect a cross/pedigree column in {args.manifest}")

    work = pd.DataFrame()
    work["sample_id"] = df[id_col].map(canonical_gid)
    work["cross_name"] = df[cross_col].map(clean_text)
    work["selection_history"] = df[history_col].map(clean_text) if history_col else ""
    for col in ["CID", "SID", "trial_name", "trial_dir", "cycle", "occ"]:
        work[col] = df[col].map(clean_text) if col in df.columns else ""

    work = work[work["sample_id"].ne("")].copy()
    work["has_cross_name"] = work["cross_name"].ne("")

    ranked = (
        work[work["has_cross_name"]]
        .groupby(["sample_id", "cross_name"], dropna=False)
        .agg(
            n_cross_rows=("cross_name", "size"),
            selection_history=("selection_history", lambda x: next((v for v in x if v), "")),
            n_trial_names=("trial_name", lambda x: x[x.ne("")].nunique()),
            n_trial_dirs=("trial_dir", lambda x: x[x.ne("")].nunique()),
            example_trial_name=("trial_name", lambda x: next((v for v in x if v), "")),
            example_trial_dir=("trial_dir", lambda x: next((v for v in x if v), "")),
        )
        .reset_index()
        .sort_values(["sample_id", "n_cross_rows", "cross_name"], ascending=[True, False, True])
    )
    best = ranked.drop_duplicates("sample_id", keep="first").copy()

    all_samples = (
        work.groupby("sample_id", dropna=False)
        .agg(
            n_manifest_rows=("sample_id", "size"),
            n_nonempty_cross_rows=("has_cross_name", "sum"),
            n_distinct_cross_names=("cross_name", lambda x: x[x.ne("")].nunique()),
            n_distinct_selection_history=("selection_history", lambda x: x[x.ne("")].nunique()),
        )
        .reset_index()
    )
    out = all_samples.merge(best, on="sample_id", how="left")
    out["cross_name"] = out["cross_name"].fillna("")
    out["selection_history"] = out["selection_history"].fillna("")
    out["n_cross_rows"] = out["n_cross_rows"].fillna(0).astype(int)
    out["n_trial_names"] = out["n_trial_names"].fillna(0).astype(int)
    out["n_trial_dirs"] = out["n_trial_dirs"].fillna(0).astype(int)
    out["pedigree_source"] = "trial_manifest_cross_name"
    out.loc[out["cross_name"].eq(""), "pedigree_source"] = "trial_manifest_no_cross_name"
    out = out.sort_values("sample_id")

    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_table, sep="\t", index=False)

    qc = pd.DataFrame(
        [
            {"metric": "manifest_rows", "value": len(df)},
            {"metric": "manifest_rows_with_sample_id", "value": len(work)},
            {"metric": "unique_sample_id", "value": out["sample_id"].nunique()},
            {"metric": "unique_sample_id_with_cross_name", "value": int(out["cross_name"].ne("").sum())},
            {"metric": "unique_sample_id_without_cross_name", "value": int(out["cross_name"].eq("").sum())},
            {"metric": "sample_id_with_multiple_cross_names", "value": int(out["n_distinct_cross_names"].gt(1).sum())},
            {"metric": "input_id_col", "value": id_col},
            {"metric": "input_cross_col", "value": cross_col},
            {"metric": "input_history_col", "value": history_col or ""},
        ]
    )
    qc.to_csv(args.out_qc, sep="\t", index=False)
    print(qc.to_string(index=False))
    print(f"Wrote: {args.out_table}")
    print(f"Wrote: {args.out_qc}")


if __name__ == "__main__":
    main()
