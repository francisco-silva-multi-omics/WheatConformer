from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine certified recovered-panel registry fragments.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("genotype_panels/recovered/recovered_genotype_kernel_manifest.tsv"))
    args = parser.parse_args()
    root = args.root.resolve()
    paths = sorted((root / "genotype_panels" / "recovered").glob("*/*_registry_fragment.tsv"))
    if not paths:
        raise SystemExit("No recovered genotype registry fragments were found")
    frames = [pd.read_csv(path, sep="\t", dtype=str) for path in paths]
    manifest = pd.concat(frames, ignore_index=True)
    if manifest["kernel"].duplicated().any():
        raise SystemExit("Recovered genotype registry contains duplicate kernel names")
    out = (root / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, sep="\t", index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
