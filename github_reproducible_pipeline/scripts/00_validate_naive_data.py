from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED_ANY = {
    "hmp_gbs": [
        "Genotypic data from CIMMYT bread wheat breeding lines",
        "Genotypic_data_from_CIMMYT_bread_wheat_breeding_lines",
    ],
    "dartag": [
        "Genotypic data (DArTAG panel 2) for the IBWSN and SAWSN",
        "Genotypic_data_(DArTAG_panel_2)_for_the_IBWSN_and_SAWSN",
    ],
    "mas_57_58": [
        "57IBWSN, 42SAWSN, and 35HRWSN - Gene-based marker data for marker-assisted selection",
        "57IBWSN,_42SAWSN,_and_35HRWSN_-_Gene-based_marker_data_for_marker-assisted_selection",
        "58IBWSN and 43SAWSN - Gene-based marker data for marker-assisted selection",
        "58IBWSN_and_43SAWSN_-_Gene-based_marker_data_for_marker-assisted_selection",
    ],
    "haploblocks": [
        "Haplotype-based genome-wide association study",
        "Haplotype-based_genome-wide_association_study",
    ],
    "dartseq_landrace": [
        "DArTseq-derived SNPs for wheat Mexican landrace accessions",
        "DArTseq-derived_SNPs_for_wheat_Mexican_landrace_accessions",
    ],
    "diversity_80k": [
        "Diversity analysis of 80,000 wheat accessions reveals consequences and opportunities of selection footprints",
        "80k",
    ],
    "gbs_sawyt": [
        "GBS",
    ],
}


EXPECTED_SUBSTRINGS = {
    "phenotype_trials": ["Yield Trial", "Screening Nursery", "Elite Spring Wheat"],
}


def count_files(path: Path) -> int:
    if path.is_file():
        return 1
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate naive/raw data availability before server processing.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("logs/naive_data_validation.tsv"))
    args = parser.parse_args()

    if not args.raw_dir.exists():
        raise SystemExit(f"Raw directory does not exist: {args.raw_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    top = {p.name: p for p in args.raw_dir.iterdir()}
    rows = []

    for group, candidates in EXPECTED_ANY.items():
        matches = [top[name] for name in candidates if name in top]
        rows.append(
            {
                "group": group,
                "status": "present" if matches else "missing",
                "matches": ";".join(str(p.name) for p in matches),
                "file_count": sum(count_files(p) for p in matches),
            }
        )

    for group, tokens in EXPECTED_SUBSTRINGS.items():
        matches = [p for p in top.values() if any(token.lower() in p.name.lower() for token in tokens)]
        rows.append(
            {
                "group": group,
                "status": "present" if matches else "missing",
                "matches": ";".join(p.name for p in matches[:50]),
                "file_count": sum(count_files(p) for p in matches),
            }
        )

    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "status", "matches", "file_count"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {args.out}")
    for row in rows:
        print(f"{row['group']}\t{row['status']}\tfiles={row['file_count']}")

    required_missing = [
        row["group"]
        for row in rows
        if row["group"] not in {"gbs_sawyt", "diversity_80k"} and row["status"] == "missing"
    ]
    if required_missing:
        raise SystemExit(f"Missing required raw groups: {required_missing}")


if __name__ == "__main__":
    main()
