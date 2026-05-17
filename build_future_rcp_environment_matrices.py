from __future__ import annotations

import argparse
import platform
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOCAL_DEPS = BASE / "local_python_deps"
if platform.system() == "Windows" and LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd


OUT = BASE / "environment"
FUTURE_OUT = OUT / "future_rcp"
DEFAULT_INPUT = OUT / "future_rcp_weather_features.tsv"
RCP_LEVELS = {"2.6", "4.5", "6.0", "8.5"}
COPY_FROM_BASE_COMPONENTS = {"geo", "mgmt"}
PROJECTED_COMPONENTS = {"weather", "stress"}


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")


def normalize_rcp(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper()
    match = re.search(r"([0-9](?:\.[0-9])?)", text)
    if not match:
        return text
    number = match.group(1)
    return f"RCP{number}"


def load_weights() -> dict[str, float]:
    weights = pd.read_csv(OUT / "env_kernel_component_weights.tsv", sep="\t", dtype={"kernel": str})
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce").fillna(0.0)
    return dict(zip(weights["kernel"], weights["weight"]))


def load_scaling() -> pd.DataFrame:
    path = OUT / "env_feature_scaling_parameters.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Rerun build_environment_component_kernels.py once to write historical scalers."
        )
    scaling = pd.read_csv(path, sep="\t", dtype={"kernel": str, "feature": str})
    scaling["mean"] = pd.to_numeric(scaling["mean"], errors="coerce")
    scaling["std"] = pd.to_numeric(scaling["std"], errors="coerce")
    return scaling


def load_historical_z(component: str) -> pd.DataFrame:
    path = OUT / f"env_features_{component}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Rerun build_environment_component_kernels.py first.")
    z = pd.read_parquet(path).set_index("env_id")
    order = pd.read_csv(OUT / "env_kernel_sample_order.tsv", sep="\t", dtype=str)["env_id"]
    return z.reindex(order)


def projection_column(df: pd.DataFrame, feature: str) -> str | None:
    if feature in df.columns:
        return feature
    if feature.startswith("weather_api_"):
        unprefixed = feature.removeprefix("weather_api_")
        if unprefixed in df.columns:
            return unprefixed
    return None


def standardize_projected_component(
    future: pd.DataFrame, scaling: pd.DataFrame, component: str
) -> tuple[pd.DataFrame, int]:
    params = scaling[scaling["kernel"].eq(component)].copy()
    z = pd.DataFrame(index=future["future_env_id"].astype(str))
    observed_feature_count = 0
    for row in params.itertuples(index=False):
        col = projection_column(future, row.feature)
        if col is None:
            values = pd.Series(np.nan, index=future.index, dtype="float64")
        else:
            values = pd.to_numeric(future[col], errors="coerce")
            observed_feature_count += int(values.notna().any())
        std = row.std if pd.notna(row.std) and row.std != 0 else np.nan
        standardized = ((values.fillna(row.mean) - row.mean) / std).fillna(0.0)
        z[row.feature] = standardized.to_numpy(dtype=np.float32)
    return z, observed_feature_count


def copy_component_from_base(future: pd.DataFrame, component: str) -> pd.DataFrame:
    hist = load_historical_z(component)
    base_ids = future["base_env_id"].astype(str)
    missing = sorted(set(base_ids) - set(hist.index))
    if missing:
        examples = "; ".join(missing[:5])
        raise ValueError(f"{component}: {len(missing)} base_env_id values are absent from historical order. Examples: {examples}")
    z = hist.reindex(base_ids).copy()
    z.index = future["future_env_id"].astype(str)
    return z


def component_kernels(z_future: pd.DataFrame, z_hist: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p = max(z_future.shape[1], 1)
    future_values = z_future.to_numpy(dtype=np.float32)
    hist_values = z_hist[z_future.columns].to_numpy(dtype=np.float32)
    k_future = (future_values @ future_values.T) / p
    k_future_vs_historical = (future_values @ hist_values.T) / p
    return k_future.astype(np.float32), k_future_vs_historical.astype(np.float32)


def build_future_id(future: pd.DataFrame) -> pd.Series:
    if "future_env_id" in future.columns and future["future_env_id"].notna().any():
        return future["future_env_id"].astype(str)
    return (
        future["base_env_id"].astype(str)
        + "|"
        + future["rcp"].map(normalize_rcp)
        + "|"
        + future["period"].astype(str)
    )


def validate_future_input(future: pd.DataFrame) -> pd.DataFrame:
    missing_required = {"base_env_id", "rcp", "period"} - set(future.columns)
    if missing_required:
        raise ValueError(f"Future RCP input is missing required columns: {sorted(missing_required)}")
    future = future.copy()
    future["rcp"] = future["rcp"].map(normalize_rcp)
    allowed = {f"RCP{x}" for x in RCP_LEVELS}
    bad_rcp = sorted(set(future["rcp"]) - allowed)
    if bad_rcp:
        raise ValueError(f"Unsupported RCP values: {bad_rcp}. Expected RCP2.6, RCP4.5, RCP6.0, or RCP8.5.")
    future["future_env_id"] = build_future_id(future)
    duplicate_ids = future["future_env_id"].duplicated()
    if duplicate_ids.any():
        raise ValueError(f"future_env_id must be unique; found {int(duplicate_ids.sum())} duplicates.")
    return future


def write_template(path: Path, scaling: pd.DataFrame) -> None:
    features = scaling[scaling["kernel"].isin(PROJECTED_COMPONENTS)]["feature"].tolist()
    unprefixed = [f.removeprefix("weather_api_") for f in features]
    columns = ["future_env_id", "base_env_id", "rcp", "period", *unprefixed]
    template = pd.DataFrame(columns=columns)
    write_tsv(template, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=FUTURE_OUT)
    parser.add_argument(
        "--allow-partial-components",
        action="store_true",
        help="Build K_E from whichever weighted components can be constructed.",
    )
    args = parser.parse_args()

    scaling = load_scaling()
    if not args.input.exists():
        template_path = args.out_dir / "future_rcp_weather_features_template.tsv"
        write_template(template_path, scaling)
        raise SystemExit(
            f"Missing future projection input: {args.input}. Wrote an empty schema template to {template_path}."
        )

    future = validate_future_input(pd.read_csv(args.input, sep="\t", dtype=str, low_memory=False))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(future[["future_env_id", "base_env_id", "rcp", "period"]], args.out_dir / "future_env_sample_order.tsv")

    weights = load_weights()
    component_future_kernels: dict[str, np.ndarray] = {}
    component_cross_kernels: dict[str, np.ndarray] = {}
    qc_rows = []

    for component, weight in weights.items():
        if weight <= 0:
            continue
        if component in COPY_FROM_BASE_COMPONENTS:
            z_future = copy_component_from_base(future, component)
            observed_features = z_future.shape[1]
        elif component in PROJECTED_COMPONENTS:
            z_future, observed_features = standardize_projected_component(future, scaling, component)
            if observed_features == 0 and not args.allow_partial_components:
                raise ValueError(f"No projected {component} columns were found in {args.input}.")
        else:
            continue

        z_hist = load_historical_z(component)
        shared = [c for c in z_future.columns if c in z_hist.columns]
        z_future = z_future[shared]
        z_hist = z_hist[shared]
        k_future, k_cross = component_kernels(z_future, z_hist)
        component_future_kernels[component] = k_future
        component_cross_kernels[component] = k_cross

        np.save(args.out_dir / f"K_{component}_future.npy", k_future)
        np.save(args.out_dir / f"K_{component}_future_vs_historical.npy", k_cross)
        z_future.reset_index(names="future_env_id").to_parquet(
            args.out_dir / f"future_env_features_{component}.parquet", index=False
        )
        qc_rows.append(
            {
                "kernel": component,
                "weight": weight,
                "future_env_id_total": len(future),
                "feature_count": z_future.shape[1],
                "projected_input_features_present": observed_features,
            }
        )

    missing_components = [component for component, weight in weights.items() if weight > 0 and component not in component_future_kernels]
    if missing_components and not args.allow_partial_components:
        raise ValueError(f"Could not build all weighted components: {missing_components}")

    total_weight = sum(weights[c] for c in component_future_kernels)
    k_e_future = sum(weights[c] * component_future_kernels[c] for c in component_future_kernels) / total_weight
    k_e_cross = sum(weights[c] * component_cross_kernels[c] for c in component_cross_kernels) / total_weight
    np.save(args.out_dir / "K_E_future.npy", k_e_future.astype(np.float32))
    np.save(args.out_dir / "K_E_future_vs_historical.npy", k_e_cross.astype(np.float32))

    qc = pd.DataFrame(qc_rows)
    qc.loc[len(qc)] = {
        "kernel": "K_E",
        "weight": total_weight,
        "future_env_id_total": len(future),
        "feature_count": sum(row["feature_count"] for row in qc_rows),
        "projected_input_features_present": sum(row["projected_input_features_present"] for row in qc_rows),
    }
    write_tsv(qc, args.out_dir / "future_rcp_kernel_qc.tsv")
    print(qc.to_string(index=False))
    print("K_E_future", k_e_future.shape)
    print("K_E_future_vs_historical", k_e_cross.shape)


if __name__ == "__main__":
    main()
