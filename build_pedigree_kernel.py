from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ID_CANDIDATES = [
    "germplasm_id",
    "resolved_gid",
    "gid",
    "GID",
    "line",
    "line_name",
    "sample_id",
    "designation",
    "name",
]
P1_CANDIDATES = ["female_parent", "parent1", "mother", "dam", "pedigree_parent1", "Female", "P1"]
P2_CANDIDATES = ["male_parent", "parent2", "father", "sire", "pedigree_parent2", "Male", "P2"]
CROSS_CANDIDATES = ["cross", "pedigree", "cross_name", "designation", "Pedigree"]
MISSING = {"", "NA", "NAN", "NONE", "NULL", ".", "-", "UNKNOWN", "0"}


def read_table(path: Path) -> pd.DataFrame:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffix.endswith(".xlsx") or suffix.endswith(".xls"):
        return pd.read_excel(path)
    sep = "\t" if suffix.endswith(".tsv") or suffix.endswith(".tsv.gz") else None
    return pd.read_csv(path, sep=sep, low_memory=False)


def clean_id(value) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).strip())
    if text.upper() in MISSING:
        return ""
    return text


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def parse_cross(text: str) -> tuple[str, str]:
    text = clean_id(text)
    if not text:
        return "", ""
    text = re.sub(r"\s+", "", text)
    for sep in ["//", "/", "\\", " X ", " x ", "X", "*"]:
        if sep in text:
            parts = [clean_id(p) for p in text.split(sep) if clean_id(p)]
            if len(parts) >= 2:
                return parts[0], parts[1]
    return "", ""


def load_sample_order(path: Path, col: str) -> list[str]:
    order = pd.read_csv(path, sep="\t", dtype=str)
    if col not in order.columns:
        raise SystemExit(f"Order file {path} does not contain column {col}")
    return [clean_id(x) for x in order[col] if clean_id(x)]


def build_parent_table(args) -> pd.DataFrame:
    df = read_table(args.pedigree_table)
    id_col = args.id_col or first_existing(df.columns.tolist(), ID_CANDIDATES)
    if not id_col:
        raise SystemExit(f"Could not detect germplasm ID column. Use --id-col. Columns: {df.columns.tolist()[:30]}")
    p1_col = args.parent1_col or first_existing(df.columns.tolist(), P1_CANDIDATES)
    p2_col = args.parent2_col or first_existing(df.columns.tolist(), P2_CANDIDATES)
    cross_col = args.cross_col or first_existing(df.columns.tolist(), CROSS_CANDIDATES)

    rows = []
    for row in df.to_dict("records"):
        gid = clean_id(row[id_col])
        if not gid:
            continue
        p1 = clean_id(row[p1_col]) if p1_col else ""
        p2 = clean_id(row[p2_col]) if p2_col else ""
        if (not p1 or not p2) and cross_col:
            cp1, cp2 = parse_cross(row[cross_col])
            p1 = p1 or cp1
            p2 = p2 or cp2
        if p1 == gid:
            p1 = ""
        if p2 == gid:
            p2 = ""
        rows.append({"sample_id": gid, "parent1": p1, "parent2": p2})
    ped = pd.DataFrame(rows).drop_duplicates("sample_id", keep="first")
    if ped.empty:
        raise SystemExit("No usable pedigree rows found")
    return ped


def additive_relationship(ped: pd.DataFrame, requested_order: list[str] | None) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    parent_map = {r.sample_id: (r.parent1, r.parent2) for r in ped.itertuples(index=False)}
    all_ids = set(parent_map)
    for p1, p2 in parent_map.values():
        if p1:
            all_ids.add(p1)
        if p2:
            all_ids.add(p2)
    for gid in sorted(all_ids):
        parent_map.setdefault(gid, ("", ""))

    resolved: list[str] = []
    unresolved = set(parent_map)
    while unresolved:
        progressed = False
        for gid in sorted(list(unresolved)):
            p1, p2 = parent_map[gid]
            if (not p1 or p1 in resolved or p1 == gid) and (not p2 or p2 in resolved or p2 == gid):
                resolved.append(gid)
                unresolved.remove(gid)
                progressed = True
        if not progressed:
            # Break pedigree cycles conservatively by treating one unresolved node as a founder.
            gid = sorted(unresolved)[0]
            parent_map[gid] = ("", "")
            resolved.append(gid)
            unresolved.remove(gid)

    idx = {gid: i for i, gid in enumerate(resolved)}
    A = np.zeros((len(resolved), len(resolved)), dtype=np.float64)
    for gid in resolved:
        i = idx[gid]
        p1, p2 = parent_map[gid]
        has1 = p1 in idx and idx[p1] < i
        has2 = p2 in idx and idx[p2] < i
        for j in range(i):
            v = 0.0
            if has1:
                v += 0.5 * A[idx[p1], j]
            if has2:
                v += 0.5 * A[idx[p2], j]
            A[i, j] = A[j, i] = v
        if has1 and has2:
            A[i, i] = 1.0 + 0.5 * A[idx[p1], idx[p2]]
        else:
            A[i, i] = 1.0

    if requested_order:
        missing = [gid for gid in requested_order if gid not in idx]
        if missing:
            start = len(resolved)
            new_n = start + len(missing)
            A2 = np.zeros((new_n, new_n), dtype=np.float64)
            A2[:start, :start] = A
            for k in range(start, new_n):
                A2[k, k] = 1.0
            A = A2
            resolved = resolved + missing
            idx = {gid: i for i, gid in enumerate(resolved)}
        keep = [idx[gid] for gid in requested_order]
        A = A[np.ix_(keep, keep)]
        resolved = requested_order

    qc = pd.DataFrame(
        {
            "metric": [
                "pedigree_rows",
                "relationship_samples",
                "mean_diagonal",
                "min_diagonal",
                "max_diagonal",
            ],
            "value": [
                len(ped),
                len(resolved),
                float(np.mean(np.diag(A))),
                float(np.min(np.diag(A))),
                float(np.max(np.diag(A))),
            ],
        }
    )
    return A.astype(np.float32), resolved, qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build additive pedigree relationship kernel K_A.")
    parser.add_argument("--pedigree-table", type=Path, required=True)
    parser.add_argument("--id-col")
    parser.add_argument("--parent1-col")
    parser.add_argument("--parent2-col")
    parser.add_argument("--cross-col")
    parser.add_argument("--sample-order", type=Path, help="Optional TSV order to force K_A sample order.")
    parser.add_argument("--sample-order-col", default="sample_id")
    parser.add_argument("--out-dir", type=Path, default=Path("genotype_panels/pedigree"))
    parser.add_argument("--prefix", default="K_A")
    parser.add_argument("--scale-mean-diagonal", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ped = build_parent_table(args)
    requested = load_sample_order(args.sample_order, args.sample_order_col) if args.sample_order else None
    K, order, qc = additive_relationship(ped, requested)
    if args.scale_mean_diagonal:
        mean_diag = float(np.mean(np.diag(K)))
        if mean_diag > 0:
            K = (K / mean_diag).astype(np.float32)
            qc.loc[len(qc)] = ["scaled_mean_diagonal", float(np.mean(np.diag(K)))]

    np.save(args.out_dir / f"{args.prefix}.npy", K)
    pd.DataFrame({"sample_id": order}).to_csv(args.out_dir / f"{args.prefix}_sample_order.tsv", sep="\t", index=False)
    ped.to_csv(args.out_dir / "pedigree_parent_table.tsv", sep="\t", index=False)
    qc.to_csv(args.out_dir / f"{args.prefix}_qc.tsv", sep="\t", index=False)
    print(K.shape, K.dtype)
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
