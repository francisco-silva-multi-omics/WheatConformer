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
    values = [clean_id(x) for x in order[col] if clean_id(x)]
    duplicates = pd.Series(values)[pd.Series(values).duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise SystemExit(f"Order file {path} contains duplicate IDs: {duplicates[:10]}")
    return values


def build_parent_table(args) -> pd.DataFrame:
    df = read_table(args.pedigree_table)
    id_col = args.id_col or first_existing(df.columns.tolist(), ID_CANDIDATES)
    if not id_col:
        raise SystemExit(f"Could not detect germplasm ID column. Use --id-col. Columns: {df.columns.tolist()[:30]}")
    p1_col = args.parent1_col or first_existing(df.columns.tolist(), P1_CANDIDATES)
    p2_col = args.parent2_col or first_existing(df.columns.tolist(), P2_CANDIDATES)
    cross_col = args.cross_col or first_existing(df.columns.tolist(), CROSS_CANDIDATES)
    require_explicit_parents = bool(getattr(args, "require_explicit_parent_columns", False))
    if require_explicit_parents and not (p1_col or p2_col):
        raise SystemExit(
            "Explicit canonical parent columns are required; cross-name parsing is disabled for audited K_A construction"
        )

    rows = []
    for row in df.to_dict("records"):
        gid = clean_id(row[id_col])
        if not gid:
            continue
        p1 = clean_id(row[p1_col]) if p1_col else ""
        p2 = clean_id(row[p2_col]) if p2_col else ""
        if (not p1 or not p2) and cross_col and not require_explicit_parents:
            cp1, cp2 = parse_cross(row[cross_col])
            p1 = p1 or cp1
            p2 = p2 or cp2
        if p1 == gid:
            p1 = ""
        if p2 == gid:
            p2 = ""
        rows.append({"sample_id": gid, "parent1": p1, "parent2": p2})
    ped = pd.DataFrame(rows).drop_duplicates(["sample_id", "parent1", "parent2"])
    if ped.empty:
        raise SystemExit("No usable pedigree rows found")
    conflicts = ped.groupby("sample_id", dropna=False).size()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        conflict_rows = ped[ped["sample_id"].isin(conflicts.index)].sort_values(["sample_id", "parent1", "parent2"])
        conflict_path = args.out_dir / "pedigree_conflicts.tsv"
        conflict_rows.to_csv(conflict_path, sep="\t", index=False)
        raise SystemExit(
            f"Conflicting pedigree assignments for {len(conflicts)} sample IDs; "
            f"review {conflict_path}. Examples: {conflicts.index[:10].tolist()}"
        )
    if bool(getattr(args, "require_parents_in_pedigree", False)):
        sample_ids = set(ped["sample_id"])
        parent_ids = set(ped["parent1"]) | set(ped["parent2"])
        missing_parent_rows = sorted(parent_ids - sample_ids - {""})
        if missing_parent_rows:
            missing_path = args.out_dir / "pedigree_parents_missing_from_universe.tsv"
            pd.DataFrame({"parent_id": missing_parent_rows}).to_csv(missing_path, sep="\t", index=False)
            raise SystemExit(
                f"{len(missing_parent_rows)} parent IDs are absent from the reviewed pedigree universe; "
                f"review {missing_path}"
            )
    required_id_regex = getattr(args, "required_id_regex", None)
    if required_id_regex:
        pattern = re.compile(required_id_regex)
        pedigree_ids = set(ped["sample_id"]) | set(ped["parent1"]) | set(ped["parent2"])
        invalid_ids = sorted(value for value in pedigree_ids - {""} if not pattern.fullmatch(value))
        if invalid_ids:
            invalid_path = args.out_dir / "pedigree_noncanonical_ids.tsv"
            pd.DataFrame({"pedigree_id": invalid_ids}).to_csv(invalid_path, sep="\t", index=False)
            raise SystemExit(
                f"{len(invalid_ids)} pedigree IDs do not match required pattern {required_id_regex!r}; "
                f"review {invalid_path}"
            )
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
            raise ValueError(
                "Pedigree contains a cycle or unresolved parent dependency involving "
                f"{sorted(unresolved)[:10]}"
            )

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


def assert_relationship_valid(matrix: np.ndarray, order: list[str], sample_size: int = 1024) -> None:
    if len(order) != len(set(order)):
        raise ValueError("K_A order contains duplicate sample IDs")
    if matrix.shape != (len(order), len(order)):
        raise ValueError(f"K_A shape {matrix.shape} does not match order length {len(order)}")
    if not np.isfinite(matrix).all():
        raise ValueError("K_A contains nonfinite values")
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    if symmetry_error > 1e-6:
        raise ValueError(f"K_A is asymmetric; max_abs_diff={symmetry_error}")
    selected = np.linspace(0, len(order) - 1, min(len(order), sample_size), dtype=int)
    block = np.asarray(matrix[np.ix_(selected, selected)], dtype=np.float64)
    min_eigenvalue = float(np.linalg.eigvalsh((block + block.T) / 2.0).min())
    if min_eigenvalue < -1e-5:
        raise ValueError(f"K_A is materially non-PSD; sampled_min_eigenvalue={min_eigenvalue}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build additive pedigree relationship kernel K_A.")
    parser.add_argument("--pedigree-table", type=Path, required=True)
    parser.add_argument("--id-col")
    parser.add_argument("--parent1-col")
    parser.add_argument("--parent2-col")
    parser.add_argument("--cross-col")
    parser.add_argument(
        "--require-explicit-parent-columns",
        action="store_true",
        help="Disable parent extraction from cross-name strings and require explicit parent columns.",
    )
    parser.add_argument(
        "--require-parents-in-pedigree",
        action="store_true",
        help="Require every nonmissing parent ID to have a row in the reviewed pedigree table.",
    )
    parser.add_argument(
        "--required-id-regex",
        help="Optional full-match regular expression required for every nonmissing child and parent ID.",
    )
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

    assert_relationship_valid(K, order)

    np.save(args.out_dir / f"{args.prefix}.npy", K)
    pd.DataFrame({"sample_id": order}).to_csv(args.out_dir / f"{args.prefix}_sample_order.tsv", sep="\t", index=False)
    pd.DataFrame({"row_index": np.arange(len(order)), "sample_id": order}).to_csv(
        args.out_dir / f"{args.prefix}_row_order.tsv", sep="\t", index=False
    )
    pd.DataFrame({"column_index": np.arange(len(order)), "sample_id": order}).to_csv(
        args.out_dir / f"{args.prefix}_column_order.tsv", sep="\t", index=False
    )
    ped.to_csv(args.out_dir / "pedigree_parent_table.tsv", sep="\t", index=False)
    qc.to_csv(args.out_dir / f"{args.prefix}_qc.tsv", sep="\t", index=False)
    print(K.shape, K.dtype)
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
