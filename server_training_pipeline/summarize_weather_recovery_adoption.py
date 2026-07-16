from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def summarize(paired: pd.DataFrame, contract: pd.DataFrame) -> pd.DataFrame:
    validation = paired[paired["split"].eq("val")].copy()
    if validation.empty:
        raise ValueError("Comparison does not contain validation rows")
    comparison_eligible = contract["comparison_eligible"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"}
    )
    complete = bool(
        not contract.empty
        and comparison_eligible.all()
        and contract["status"].eq("PASS").all()
    )
    rows = []
    for mode, frame in validation.groupby("mode", sort=True):
        by_seed = (
            frame.groupby("seed", as_index=False)
            .agg(
                delta_normalized_rmse=("delta_normalized_rmse", "mean"),
                delta_pearson=("delta_pearson", "mean"),
            )
        )
        seed_count = by_seed["seed"].nunique()
        rmse_seed_win_rate = float(by_seed["delta_normalized_rmse"].lt(0).mean())
        pearson_seed_win_rate = float(by_seed["delta_pearson"].gt(0).mean())
        pair_rmse_win_rate = float(frame["delta_normalized_rmse"].lt(0).mean())
        rmse_mean = float(frame["delta_normalized_rmse"].mean())
        pearson_mean = float(frame["delta_pearson"].mean())
        accepted = bool(
            complete
            and seed_count >= 4
            and rmse_mean < 0
            and pearson_mean > 0
            and rmse_seed_win_rate >= 0.75
            and pearson_seed_win_rate >= 0.75
            and pair_rmse_win_rate >= 0.60
        )
        rows.append(
            {
                "mode": mode,
                "seed_count": seed_count,
                "validation_pair_count": len(frame),
                "validation_delta_normalized_rmse_mean": rmse_mean,
                "validation_delta_pearson_mean": pearson_mean,
                "rmse_seed_win_rate": rmse_seed_win_rate,
                "pearson_seed_win_rate": pearson_seed_win_rate,
                "rmse_pair_win_rate": pair_rmse_win_rate,
                "comparison_grid_complete": complete,
                "accepted": accepted,
                "decision": "adopt_recovered_weather" if accepted else "retain_current_corrected_kernel",
            }
        )
    result = pd.DataFrame(rows)
    overall = {
        "mode": "overall",
        "seed_count": int(validation["seed"].nunique()),
        "validation_pair_count": len(validation),
        "validation_delta_normalized_rmse_mean": float(
            validation["delta_normalized_rmse"].mean()
        ),
        "validation_delta_pearson_mean": float(validation["delta_pearson"].mean()),
        "rmse_seed_win_rate": float(
            validation.groupby("seed")["delta_normalized_rmse"].mean().lt(0).mean()
        ),
        "pearson_seed_win_rate": float(
            validation.groupby("seed")["delta_pearson"].mean().gt(0).mean()
        ),
        "rmse_pair_win_rate": float(validation["delta_normalized_rmse"].lt(0).mean()),
        "comparison_grid_complete": complete,
        "accepted": bool(complete and not result.empty and result["accepted"].all()),
    }
    overall["decision"] = (
        "adopt_recovered_weather"
        if overall["accepted"]
        else "retain_current_corrected_kernel"
    )
    return pd.concat([result, pd.DataFrame([overall])], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply validation-only adoption rules to a weather-recovery comparison."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    paired_path = resolve(root, args.paired)
    contract_path = resolve(root, args.contract)
    out = resolve(root, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(paired_path, sep="\t")
    contract = pd.read_csv(contract_path, sep="\t")
    decision = summarize(paired, contract)
    decision.to_csv(out, sep="\t", index=False, lineterminator="\n")
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
