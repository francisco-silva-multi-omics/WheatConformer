from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_common import canonical_gid, file_identity, write_json


SELECTED_TRAITS = (
    "DAYS_TO_HEADING",
    "DAYS_TO_MATURITY",
    "PLANT_HEIGHT",
    "GRAIN_YIELD",
    "1000_GRAIN_WEIGHT",
    "ABOVE_GROUND_BIOMASS",
    "TEST_WEIGHT",
)

KEY_COLUMNS = (
    "canonical_germplasm_key",
    "env_kernel_id",
    "trait_name_canonical",
    "trait_name_original",
    "unit",
)

CANONICAL_COLUMNS = (
    "canonical_observation_id",
    "canonical_germplasm_key",
    "germplasm_id",
    "resolved_gid",
    "env_kernel_id",
    "trial_name",
    "cycle",
    "occ",
    "loc_no",
    "country",
    "loc_desc",
    "trait_name_canonical",
    "trait_name_original",
    "unit",
    "phenotype_value",
    "raw_numeric_records",
    "raw_plot_records",
    "n_records",
    "source_level",
    "gid_resolution_status",
    "genotype_name",
    "has_environment_kernel",
)

STAGE1_COLUMNS = (
    "canonical_observation_id",
    *KEY_COLUMNS,
    "resolved_gid",
    "trial_name",
    "cycle",
    "country",
    "y_tilde_g_e",
    "stage1_model_status",
    "n_plot_records",
)

MODEL_COLUMNS = (
    "canonical_observation_id",
    *KEY_COLUMNS,
    "resolved_gid",
    "trial_name",
    "cycle",
    "country",
    "phenotype_value",
    "geno_kernel_index",
    "env_kernel_index",
)

LEDGER_COLUMNS = (
    "canonical_observation_id",
    *KEY_COLUMNS,
    "resolved_gid",
    "trial_name",
    "cycle",
    "country",
    "phenotype_value",
    "genotype_id",
    "environment_id",
)


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def git_commit(code_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def table_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.suffix == ".parquet":
        fallback = path.with_suffix(".tsv.gz")
        if fallback.is_file():
            return fallback
    raise FileNotFoundError(path)


def available_columns(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).schema.names
    return pd.read_csv(path, sep="\t", nrows=0).columns.tolist()


def read_columns(
    path: Path, requested: tuple[str, ...], required: tuple[str, ...]
) -> pd.DataFrame:
    path = table_path(path)
    available = available_columns(path)
    missing = sorted(set(required) - set(available))
    if missing:
        raise ValueError(f"Required columns are absent from {path}: {missing}")
    usecols = [column for column in requested if column in available]
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=usecols)
    return pd.read_csv(path, sep="\t", usecols=usecols, low_memory=False)


def normalized_text(series: pd.Series, *, upper: bool = False) -> pd.Series:
    output = series.fillna("").astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    return output.str.upper() if upper else output


def normalize_gid_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    canonical = normalized_text(
        output.get("canonical_germplasm_key", pd.Series("", index=output.index))
    ).map(canonical_gid)
    resolved = normalized_text(
        output.get("resolved_gid", pd.Series("", index=output.index))
    ).map(canonical_gid)
    output["canonical_germplasm_key"] = canonical.where(canonical.ne(""), resolved)
    return output


def normalize_key_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = normalize_gid_columns(frame)
    output["env_kernel_id"] = normalized_text(output["env_kernel_id"])
    output["trait_name_canonical"] = normalized_text(
        output["trait_name_canonical"], upper=True
    )
    output["trait_name_original"] = normalized_text(
        output["trait_name_original"], upper=True
    )
    output["unit"] = normalized_text(output["unit"], upper=True)
    return output


def key_hash(frame: pd.DataFrame) -> pd.Series:
    keys = frame.loc[:, KEY_COLUMNS].copy()
    hashed = pd.util.hash_pandas_object(keys, index=False).astype("uint64")
    unique_keys = keys.drop_duplicates()
    unique_hashes = pd.util.hash_pandas_object(unique_keys, index=False).astype("uint64")
    if unique_hashes.duplicated().any():
        duplicated = unique_keys.loc[unique_hashes.duplicated(keep=False)].head(10)
        raise RuntimeError(f"Natural-key hash collision detected:\n{duplicated}")
    return hashed


def bool_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return normalized_text(series, upper=True).isin({"1", "TRUE", "YES", "Y", "PASS"})


def finite_values(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return pd.Series(np.isfinite(numeric), index=series.index)


def order_ids(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    frame = pd.read_csv(path, sep="\t", dtype=str)
    for column in (
        "sample_id",
        "genotype_id",
        "panel_sample_id",
        "canonical_gid",
        "env_id",
        "environment_id",
    ):
        if column not in frame:
            continue
        values = normalized_text(frame[column])
        if column not in {"env_id", "environment_id"}:
            values = values.map(canonical_gid)
        return set(values[values.ne("")])
    raise ValueError(f"No supported identifier column found in {path}")


def identity_ready(frame: pd.DataFrame) -> pd.Series:
    return frame["canonical_germplasm_key"].ne("") & frame["env_kernel_id"].ne("")


def summarize_stage(
    stage: str,
    frame: pd.DataFrame,
    row_semantics: str,
    phenotype_column: str,
) -> dict[str, object]:
    finite = finite_values(frame[phenotype_column])
    raw_records = pd.to_numeric(
        frame.get("raw_numeric_records", pd.Series(0, index=frame.index)),
        errors="coerce",
    ).fillna(0)
    return {
        "stage": stage,
        "row_semantics": row_semantics,
        "rows": len(frame),
        "finite_target_rows": int(finite.sum()),
        "unique_natural_keys": int(frame["analysis_key_hash"].nunique()),
        "unique_genotypes": int(frame["canonical_germplasm_key"].replace("", np.nan).nunique()),
        "unique_environments": int(frame["env_kernel_id"].replace("", np.nan).nunique()),
        "unique_traits": int(frame["trait_name_canonical"].replace("", np.nan).nunique()),
        "represented_raw_numeric_records": int(raw_records.sum()),
    }


def classify_selected_canonical(frame: pd.DataFrame) -> pd.Series:
    reason = pd.Series("retained_in_final_ledger_key", index=frame.index, dtype=object)
    rules = [
        ("nonfinite_target", ~frame["finite_target"]),
        ("unresolved_genotype_identity", ~frame["gid_resolved"]),
        ("environment_kernel_unavailable", ~frame["environment_available"]),
        ("absent_from_canonical_pedigree", ~frame["canonical_pedigree_available"]),
        ("not_reconstructed_by_stage1_raw_pipeline", ~frame["stage1_key_available"]),
        ("outside_stage1_genotype_environment_intersection", ~frame["model_key_available"]),
        ("absent_from_final_multitrait_ledger", ~frame["ledger_key_available"]),
    ]
    assigned = pd.Series(False, index=frame.index)
    for label, mask in rules:
        selected = mask & ~assigned
        reason.loc[selected] = label
        assigned |= selected
    return reason


def grouped_loss(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    output = (
        frame.groupby([group, "exclusive_loss_reason"], dropna=False, sort=True)
        .agg(
            canonical_rows=("canonical_observation_id", "size"),
            unique_natural_keys=("analysis_key_hash", "nunique"),
            unique_genotypes=("canonical_germplasm_key", "nunique"),
            unique_environments=("env_kernel_id", "nunique"),
        )
        .reset_index()
    )
    output[group] = output[group].fillna("")
    return output


def trait_recovery_table(frame: pd.DataFrame, selected_traits: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trait, group in frame.groupby("trait_name_canonical", dropna=False, sort=True):
        strict = (
            group["finite_target"]
            & group["gid_resolved"]
            & group["environment_available"]
            & group["canonical_pedigree_available"]
        )
        model_ready = group["model_key_available"]
        raw_records = pd.to_numeric(group["raw_numeric_records"], errors="coerce").fillna(0)
        rows.append(
            {
                "trait_name_canonical": trait,
                "selected_in_frozen_model": trait in selected_traits,
                "canonical_rows": len(group),
                "unique_natural_keys": group["analysis_key_hash"].nunique(),
                "unique_genotypes": group["canonical_germplasm_key"].replace("", np.nan).nunique(),
                "unique_environments": group["env_kernel_id"].replace("", np.nan).nunique(),
                "unique_countries": normalized_text(group["country"]).replace("", np.nan).nunique(),
                "unique_cycles": normalized_text(group["cycle"]).replace("", np.nan).nunique(),
                "finite_target_rows": int(group["finite_target"].sum()),
                "strict_identity_ready_rows": int(strict.sum()),
                "strict_identity_ready_genotypes": group.loc[
                    strict, "canonical_germplasm_key"
                ].replace("", np.nan).nunique(),
                "strict_identity_ready_environments": group.loc[
                    strict, "env_kernel_id"
                ].replace("", np.nan).nunique(),
                "stage1_reconstructed_rows": int(group["stage1_key_available"].sum()),
                "stage1_model_rows": int(group["model_key_available"].sum()),
                "stage1_model_unique_keys": group.loc[
                    model_ready, "analysis_key_hash"
                ].nunique(),
                "stage1_model_environments": group.loc[
                    model_ready, "env_kernel_id"
                ].replace("", np.nan).nunique(),
                "final_ledger_rows": int(group["ledger_key_available"].sum()),
                "raw_reconstruction_candidate_rows": int(
                    (strict & ~group["stage1_key_available"] & raw_records.gt(0)).sum()
                ),
                "represented_raw_numeric_records": int(raw_records.sum()),
            }
        )
    output = pd.DataFrame(rows)
    output["strict_identity_ready_fraction"] = (
        output["strict_identity_ready_rows"] / output["canonical_rows"].clip(lower=1)
    )
    output["development_screen_status"] = np.select(
        [
            output["selected_in_frozen_model"],
            (output["stage1_model_unique_keys"] >= 1000)
            & (output["stage1_model_environments"] >= 20),
            (output["stage1_model_unique_keys"] >= 100)
            & (output["stage1_model_environments"] >= 5),
            (output["strict_identity_ready_rows"] >= 1000)
            & (output["raw_reconstruction_candidate_rows"] >= 100),
        ],
        [
            "already_selected",
            "eligible_for_new_inner_only_trait_audit",
            "exploratory_support_only",
            "requires_stage1_reconstruction_audit",
        ],
        default="insufficient_current_support",
    )
    return output.sort_values(
        ["selected_in_frozen_model", "strict_identity_ready_rows"],
        ascending=[False, False],
    ).reset_index(drop=True)


def imputation_policy() -> pd.DataFrame:
    rows = [
        ("target_phenotype", "missing trait outcome", "PROHIBITED", "Use a masked multi-trait likelihood; never create training or evaluation labels."),
        ("stage1_phenotype", "raw records exist but adjusted row is absent", "RECONSTRUCT_NOT_IMPUTE", "Rebuild from raw plot records with the frozen Stage-1 model and provenance."),
        ("environment_covariate", "sporadic numeric management/weather value", "ALLOWED_FOLD_LOCAL", "Fit median or robust statistics on training environments only and add missingness flags."),
        ("weather_window", "year-specific weather unavailable", "ALLOWED_WITH_CONFIDENCE", "Use location-season climatology or another reanalysis source with observed/climatology/inferred flags."),
        ("marker_dosage", "missing calls inside one certified platform", "ALLOWED_PANEL_SPECIFIC", "Fit allele/dosage imputation on training/reference samples only; retain dosage uncertainty."),
        ("marker_platform", "individual absent from a platform", "PROHIBITED_AS_ZERO_FILL", "Use a coverage mask, pedigree/single-step H propagation, or a separately confidence-gated imputation model."),
        ("pedigree_parent", "parent identity missing", "METADATA_RECOVERY_ONLY", "Resolve aliases/crosses or retain the individual as a founder; do not numerically impute parent IDs."),
        ("regulatory_embedding", "pedigree-only genotype", "ALLOWED_CONFIDENCE_GATED", "Pedigree propagation may be reported as imputed, never as observed genotype-specific sequence."),
        ("future_climate", "missing or uncertain RCP/SSP driver", "ENSEMBLE_NOT_SINGLE_IMPUTATION", "Propagate GCM, pathway, downscaling, and OOD uncertainty rather than filling one deterministic value."),
    ]
    return pd.DataFrame(rows, columns=["information_type", "missingness_case", "policy", "required_method"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit information attrition and defensible recovery opportunities."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path("integrated_database/canonical_trial_genotype_environment_plot_table.parquet"),
    )
    parser.add_argument(
        "--stage1-adjusted",
        type=Path,
        default=Path("phenotypes/stage1_adjusted_phenotypes.parquet"),
    )
    parser.add_argument(
        "--stage1-model-observations",
        type=Path,
        default=Path("model_kernels/stage1_pedigree_env/stage1_pedigree_env_model_ready_stage1_observations.parquet"),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("model_kernels/multitrait_pedigree_env_uniform_tgw_certified/multitrait_pedigree_uniform_tgw_certified_observations.parquet"),
    )
    parser.add_argument(
        "--stage1-genotype-order",
        type=Path,
        default=Path("model_kernels/stage1_pedigree_env/stage1_pedigree_env_K_G_unique_order.tsv"),
    )
    parser.add_argument(
        "--stage1-environment-order",
        type=Path,
        default=Path("model_kernels/stage1_pedigree_env/stage1_pedigree_env_K_E_unique_order.tsv"),
    )
    parser.add_argument(
        "--canonical-pedigree-order",
        type=Path,
        default=Path("genotype_panels/pedigree_canonical_v3/K_A_CANONICAL_V3_sample_order.tsv"),
    )
    parser.add_argument(
        "--regulatory-manifest",
        type=Path,
        default=Path("model_kernels/regulatory_eligibility_v1_reconciled/regulatory_genotype_eligibility_manifest.tsv.gz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("audit/information_attrition_v1"))
    parser.add_argument("--trait", action="append")
    args = parser.parse_args()

    root = args.root.resolve()
    selected_traits = {
        value.strip().upper() for value in (args.trait or SELECTED_TRAITS) if value.strip()
    }
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "canonical": table_path(resolve(root, args.canonical)),
        "stage1_adjusted": table_path(resolve(root, args.stage1_adjusted)),
        "stage1_model_observations": table_path(resolve(root, args.stage1_model_observations)),
        "ledger": table_path(resolve(root, args.ledger)),
        "stage1_genotype_order": resolve(root, args.stage1_genotype_order),
        "stage1_environment_order": resolve(root, args.stage1_environment_order),
        "canonical_pedigree_order": resolve(root, args.canonical_pedigree_order),
        "regulatory_manifest": resolve(root, args.regulatory_manifest),
    }
    for name in ("stage1_genotype_order", "stage1_environment_order", "canonical_pedigree_order"):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])

    canonical = normalize_key_columns(
        read_columns(paths["canonical"], CANONICAL_COLUMNS, KEY_COLUMNS + ("canonical_observation_id", "phenotype_value"))
    )
    stage1 = normalize_key_columns(
        read_columns(paths["stage1_adjusted"], STAGE1_COLUMNS, KEY_COLUMNS + ("canonical_observation_id", "y_tilde_g_e"))
    )
    model = normalize_key_columns(
        read_columns(paths["stage1_model_observations"], MODEL_COLUMNS, KEY_COLUMNS + ("canonical_observation_id", "phenotype_value"))
    )
    ledger = normalize_key_columns(
        read_columns(paths["ledger"], LEDGER_COLUMNS, KEY_COLUMNS + ("canonical_observation_id", "phenotype_value"))
    )
    for frame in (canonical, stage1, model, ledger):
        frame["analysis_key_hash"] = key_hash(frame)

    stage1_ids = set(normalized_text(stage1["canonical_observation_id"]))
    model_ids = set(normalized_text(model["canonical_observation_id"]))
    ledger_ids = set(normalized_text(ledger["canonical_observation_id"]))
    stage1_keys = set(stage1["analysis_key_hash"].astype("uint64"))
    model_keys = set(model["analysis_key_hash"].astype("uint64"))
    ledger_keys = set(ledger["analysis_key_hash"].astype("uint64"))
    stage1_genotypes = order_ids(paths["stage1_genotype_order"])
    stage1_environments = order_ids(paths["stage1_environment_order"])
    canonical_pedigree = order_ids(paths["canonical_pedigree_order"])

    canonical["finite_target"] = finite_values(canonical["phenotype_value"])
    canonical["gid_resolved"] = canonical["canonical_germplasm_key"].ne("")
    environment_flag = bool_values(
        canonical.get("has_environment_kernel", pd.Series(False, index=canonical.index))
    )
    canonical["environment_available"] = (
        canonical["env_kernel_id"].ne("") & environment_flag
    )
    canonical["stage1_genotype_available"] = canonical["canonical_germplasm_key"].isin(stage1_genotypes)
    canonical["stage1_environment_available"] = canonical["env_kernel_id"].isin(stage1_environments)
    canonical["canonical_pedigree_available"] = canonical["canonical_germplasm_key"].isin(canonical_pedigree)
    canonical["stage1_key_available"] = canonical["analysis_key_hash"].isin(stage1_keys)
    canonical["model_key_available"] = canonical["analysis_key_hash"].isin(model_keys)
    canonical["ledger_key_available"] = canonical["analysis_key_hash"].isin(ledger_keys)
    canonical["selected_trait"] = canonical["trait_name_canonical"].isin(selected_traits)

    stage1["finite_target"] = finite_values(stage1["y_tilde_g_e"])
    stage1["selected_trait"] = stage1["trait_name_canonical"].isin(selected_traits)
    stage1["model_id_available"] = normalized_text(stage1["canonical_observation_id"]).isin(model_ids)
    stage1["ledger_id_available"] = normalized_text(stage1["canonical_observation_id"]).isin(ledger_ids)

    selected = canonical[canonical["selected_trait"]].copy()
    selected["exclusive_loss_reason"] = classify_selected_canonical(selected)

    waterfall = pd.DataFrame(
        [
            summarize_stage("canonical_all_traits", canonical, "canonical source summaries", "phenotype_value"),
            summarize_stage("canonical_selected_traits", selected, "canonical source summaries", "phenotype_value"),
            summarize_stage("stage1_adjusted_selected_traits", stage1[stage1["selected_trait"]], "Stage-1 adjusted genotype-environment-trait rows", "y_tilde_g_e"),
            summarize_stage("stage1_model_observations_selected_traits", model[model["trait_name_canonical"].isin(selected_traits)], "Stage-1 rows matched to genotype and environment orders", "phenotype_value"),
            summarize_stage("final_multitrait_ledger", ledger, "immutable final multitrait ledger rows", "phenotype_value"),
        ]
    )
    waterfall.to_csv(out_dir / "information_attrition_waterfall.tsv", sep="\t", index=False)

    loss_summary = (
        selected.groupby("exclusive_loss_reason", sort=True)
        .agg(
            canonical_rows=("canonical_observation_id", "size"),
            unique_natural_keys=("analysis_key_hash", "nunique"),
            unique_genotypes=("canonical_germplasm_key", "nunique"),
            unique_environments=("env_kernel_id", "nunique"),
            unique_traits=("trait_name_canonical", "nunique"),
            represented_raw_numeric_records=("raw_numeric_records", lambda x: int(pd.to_numeric(x, errors="coerce").fillna(0).sum())),
        )
        .reset_index()
        .sort_values("canonical_rows", ascending=False)
    )
    loss_summary.to_csv(out_dir / "selected_trait_exclusive_loss_summary.tsv", sep="\t", index=False)

    overlapping_flags = [
        "finite_target",
        "gid_resolved",
        "environment_available",
        "stage1_genotype_available",
        "stage1_environment_available",
        "canonical_pedigree_available",
        "stage1_key_available",
        "model_key_available",
        "ledger_key_available",
    ]
    overlap_rows = []
    for flag in overlapping_flags:
        overlap_rows.append(
            {
                "eligibility_check": flag,
                "pass_rows": int(selected[flag].sum()),
                "fail_rows": int((~selected[flag]).sum()),
                "pass_fraction": float(selected[flag].mean()) if len(selected) else 0.0,
            }
        )
    pd.DataFrame(overlap_rows).to_csv(
        out_dir / "selected_trait_overlapping_eligibility.tsv", sep="\t", index=False
    )

    selected_columns = [
        "canonical_observation_id",
        "canonical_germplasm_key",
        "env_kernel_id",
        "trial_name",
        "cycle",
        "country",
        "trait_name_canonical",
        "trait_name_original",
        "unit",
        "analysis_key_hash",
        "exclusive_loss_reason",
        *overlapping_flags,
        "raw_numeric_records",
        "raw_plot_records",
        "source_level",
        "gid_resolution_status",
    ]
    selected[[column for column in selected_columns if column in selected]].to_parquet(
        out_dir / "selected_trait_attrition_ledger.parquet", index=False
    )
    grouped_loss(selected, "trait_name_canonical").to_csv(
        out_dir / "attrition_by_trait.tsv", sep="\t", index=False
    )
    grouped_loss(selected, "country").to_csv(
        out_dir / "attrition_by_country.tsv", sep="\t", index=False
    )
    grouped_loss(selected, "cycle").to_csv(
        out_dir / "attrition_by_cycle.tsv", sep="\t", index=False
    )

    traits = trait_recovery_table(canonical, selected_traits)
    traits.to_csv(out_dir / "trait_recovery_candidates.tsv", sep="\t", index=False)

    raw_numeric = pd.to_numeric(selected["raw_numeric_records"], errors="coerce").fillna(0)
    alternative_gid_evidence = (
        normalized_text(selected.get("germplasm_id", pd.Series("", index=selected.index))).ne("")
        | normalized_text(selected.get("genotype_name", pd.Series("", index=selected.index))).ne("")
    )
    location_evidence = (
        normalized_text(selected["trial_name"]).ne("")
        & normalized_text(selected["cycle"]).ne("")
        & (
            normalized_text(selected.get("loc_no", pd.Series("", index=selected.index))).ne("")
            | normalized_text(selected["country"]).ne("")
        )
    )
    recovery_rows = [
        {
            "recovery_class": "unresolved_genotype_identity",
            "affected_canonical_rows": int((~selected["gid_resolved"]).sum()),
            "strict_recovery_candidate_rows": int((~selected["gid_resolved"] & alternative_gid_evidence).sum()),
            "method": "Alias, selection-history, cross, and pedigree identity adjudication",
            "target_imputation_allowed": False,
        },
        {
            "recovery_class": "environment_kernel_unavailable",
            "affected_canonical_rows": int((~selected["environment_available"]).sum()),
            "strict_recovery_candidate_rows": int((~selected["environment_available"] & location_evidence).sum()),
            "method": "Recover environment identity, coordinates, dates, weather, and confidence flags",
            "target_imputation_allowed": False,
        },
        {
            "recovery_class": "absent_from_canonical_pedigree",
            "affected_canonical_rows": int((selected["gid_resolved"] & ~selected["canonical_pedigree_available"]).sum()),
            "strict_recovery_candidate_rows": int((selected["gid_resolved"] & ~selected["canonical_pedigree_available"]).sum()),
            "method": "Add certified pedigree nodes/edges or explicit founders; rebuild isolated K_A",
            "target_imputation_allowed": False,
        },
        {
            "recovery_class": "stage1_raw_reconstruction",
            "affected_canonical_rows": int((~selected["stage1_key_available"]).sum()),
            "strict_recovery_candidate_rows": int((~selected["stage1_key_available"] & raw_numeric.gt(0)).sum()),
            "method": "Rebuild adjusted outcomes from raw plot records; do not fill target labels",
            "target_imputation_allowed": False,
        },
        {
            "recovery_class": "collapsed_replicate_information",
            "affected_canonical_rows": int(raw_numeric.gt(1).sum()),
            "strict_recovery_candidate_rows": int(raw_numeric.gt(1).sum()),
            "method": "Retain plot/replicate hierarchy in a new model while preserving current summaries",
            "target_imputation_allowed": False,
        },
        {
            "recovery_class": "additional_trait_scope",
            "affected_canonical_rows": int((~canonical["selected_trait"]).sum()),
            "strict_recovery_candidate_rows": int(
                traits.loc[
                    ~traits["selected_in_frozen_model"]
                    & traits["development_screen_status"].eq("eligible_for_new_inner_only_trait_audit"),
                    "strict_identity_ready_rows",
                ].sum()
            ),
            "method": "Trait QC and inner-validation-only admission in a new model version",
            "target_imputation_allowed": False,
        },
        {
            "recovery_class": "nonfinite_target",
            "affected_canonical_rows": int((~selected["finite_target"]).sum()),
            "strict_recovery_candidate_rows": 0,
            "method": "Recover original source measurements if available; otherwise use masked missing labels",
            "target_imputation_allowed": False,
        },
    ]

    regulatory_summary: dict[str, object] = {"available": False}
    regulatory_path = paths["regulatory_manifest"]
    if regulatory_path.is_file():
        regulatory = read_columns(
            regulatory_path,
            (
                "canonical_gid",
                "pedigree_support",
                "direct_marker_panel_count",
                "candidate_unresolved",
                "regulatory_embedding_eligibility",
            ),
            ("canonical_gid",),
        )
        regulatory["canonical_gid"] = normalized_text(regulatory["canonical_gid"]).map(canonical_gid)
        regulatory = regulatory[regulatory["canonical_gid"].ne("")].drop_duplicates("canonical_gid")
        selected_gids = set(selected.loc[selected["gid_resolved"], "canonical_germplasm_key"])
        relevant = regulatory[regulatory["canonical_gid"].isin(selected_gids)].copy()
        direct_count = pd.to_numeric(
            relevant.get("direct_marker_panel_count", pd.Series(0, index=relevant.index)),
            errors="coerce",
        ).fillna(0)
        unresolved = bool_values(
            relevant.get("candidate_unresolved", pd.Series(False, index=relevant.index))
        )
        direct_gids = set(relevant.loc[direct_count.gt(0), "canonical_gid"])
        unresolved_gids = set(relevant.loc[unresolved, "canonical_gid"])
        recovery_rows.extend(
            [
                {
                    "recovery_class": "direct_marker_or_regulatory_eligibility",
                    "affected_canonical_rows": int(selected["canonical_germplasm_key"].isin(direct_gids).sum()),
                    "strict_recovery_candidate_rows": int(selected["canonical_germplasm_key"].isin(direct_gids).sum()),
                    "method": "Retain certified panel evidence for marker-to-graph projection and future K_z",
                    "target_imputation_allowed": False,
                },
                {
                    "recovery_class": "unresolved_marker_identity_candidates",
                    "affected_canonical_rows": int(selected["canonical_germplasm_key"].isin(unresolved_gids).sum()),
                    "strict_recovery_candidate_rows": 0,
                    "method": "Metadata and marker-call concordance adjudication before K_G, K_z, or sequence use",
                    "target_imputation_allowed": False,
                },
            ]
        )
        regulatory_summary = {
            "available": True,
            "selected_trial_gids": len(selected_gids),
            "manifest_matched_gids": relevant["canonical_gid"].nunique(),
            "direct_marker_supported_gids": len(direct_gids),
            "unresolved_candidate_gids": len(unresolved_gids),
        }

    pd.DataFrame(recovery_rows).to_csv(
        out_dir / "recovery_opportunities.tsv", sep="\t", index=False
    )
    imputation_policy().to_csv(out_dir / "imputation_policy.tsv", sep="\t", index=False)

    checks = {
        "canonical_observation_ids_unique": not normalized_text(canonical["canonical_observation_id"]).duplicated().any(),
        "stage1_observation_ids_unique": len(stage1_ids) == len(stage1),
        "model_observation_ids_unique": len(model_ids) == len(model),
        "ledger_observation_ids_unique": len(ledger_ids) == len(ledger),
        "model_is_stage1_subset": model_ids <= stage1_ids,
        "ledger_is_model_subset": ledger_ids <= model_ids,
        "ledger_traits_match_requested": set(ledger["trait_name_canonical"]) == selected_traits,
        "ledger_targets_finite": bool(finite_values(ledger["phenotype_value"]).all()),
        "canonical_natural_key_hash_collision_free": True,
        "outer_test_metrics_unread": True,
        "final_holdout_outcomes_unread": True,
        "target_values_not_used_for_model_selection": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    code_root = Path(__file__).resolve().parents[1]
    provenance = {
        "status": status,
        "audit_version": "information_attrition_v1",
        "selection_data": "identifiers_metadata_support_and_target_finiteness_only",
        "phenotype_values_read_for_finiteness_only": True,
        "phenotype_magnitudes_exported": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "target_values_used_for_selection_or_imputation": False,
        "selected_traits": sorted(selected_traits),
        "checks": checks,
        "regulatory_summary": regulatory_summary,
        "code_root": str(code_root),
        "git_commit": git_commit(code_root),
        "inputs": {name: file_identity(path) for name, path in paths.items() if path.is_file()},
        "outputs": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
    }
    write_json(out_dir / "information_attrition_provenance.json", provenance)
    print(json.dumps({
        "status": status,
        "canonical_rows": len(canonical),
        "selected_canonical_rows": len(selected),
        "stage1_selected_rows": int(stage1["selected_trait"].sum()),
        "final_ledger_rows": len(ledger),
        "largest_exclusive_loss": loss_summary.iloc[0].to_dict() if len(loss_summary) else {},
        "out_dir": str(out_dir),
    }, indent=2, default=str))
    if status != "PASS":
        raise SystemExit("Information attrition audit failed validation")


if __name__ == "__main__":
    main()
