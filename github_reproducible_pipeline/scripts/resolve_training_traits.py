from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server_training_pipeline"))

from trait_isolation import MULTITRAIT_ERROR, clean_trait_values, sanitize_trait_name


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        try:
            return pd.read_parquet(path, columns=["trait_name_canonical"])
        except ImportError:
            fallback = path.with_suffix(".tsv.gz")
            if fallback.exists():
                return pd.read_csv(fallback, sep="\t", usecols=["trait_name_canonical"])
            raise
    return pd.read_csv(path, sep="\t", usecols=["trait_name_canonical"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve one or more explicit per-model training traits.")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--train-traits", default="")
    args = parser.parse_args()

    obs = read_table(args.observations)
    available = sorted({x for x in clean_trait_values(obs["trait_name_canonical"]) if x})
    requested = [x.strip() for x in args.train_traits.split(",") if x.strip()]
    if requested:
        lookup = {trait.upper(): trait for trait in available}
        missing = [trait for trait in requested if trait.upper() not in lookup]
        if missing:
            raise SystemExit(f"TRAIN_TRAITS contains traits absent from the observation table: {missing}")
        selected = [lookup[trait.upper()] for trait in requested]
    else:
        if len(available) > 1:
            raise SystemExit(f"{MULTITRAIT_ERROR} Set TRAIN_TRAITS as a comma-separated list.")
        if not available:
            raise SystemExit("No non-empty phenotype traits are available for training.")
        selected = available

    seen: set[str] = set()
    for trait in selected:
        if trait.upper() in seen:
            continue
        seen.add(trait.upper())
        print(f"{trait}\t{sanitize_trait_name(trait)}")


if __name__ == "__main__":
    main()
